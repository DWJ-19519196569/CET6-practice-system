# pronounce.py — 口语发音评测模块
# 评分器可插拔：OpenPronounce（wav2vec2 + DTW，音素级，首选，离线）
# 架构与 engine.py 的 TTSEngine 一致：懒加载单例 + 双重检查锁。
#
# 依赖：openpronounce 包（pip install openpronounce）
#   模型（首次使用自动从 Hugging Face 下载，~1.2GB ×2）：
#     facebook/wav2vec2-large-960h         （词级转写 + 声学嵌入）
#     facebook/wav2vec2-lv-60-espeak-cv-ft （音素识别，给出具体错音）
#   参考声用本机 Kokoro（OPENPRONOUNCE_TTS=kokoro），无需联网。
#   espeak-ng 需在 PATH（本机位于 %LOCALAPPDATA%\eSpeakNG\eSpeak NG）。
import os

_sys_dir = os.path.dirname(os.path.abspath(__file__))
_espeak_dir = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'eSpeakNG', 'eSpeak NG')
if _espeak_dir and os.path.isdir(_espeak_dir):  # phonemizer 依赖 espeak-ng 二进制
    os.environ['PATH'] = _espeak_dir + os.pathsep + os.environ.get('PATH', '')
# phonemizer 通过 ctypes 加载 espeak-ng 共享库（非可执行文件），显式指定 DLL 与数据目录，
# 否则在未把 espeak-ng 装进系统 PATH 的机器上会报 "espeak not installed"。
_espeak_lib = os.path.join(_espeak_dir, 'libespeak-ng.dll')
_espeak_data = os.path.join(_espeak_dir, 'espeak-ng-data')
if os.path.isfile(_espeak_lib):
    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', _espeak_lib)
if os.path.isdir(_espeak_data):
    os.environ.setdefault('PHONEMIZER_ESPEAK_DATA_PATH', _espeak_data)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
# 参考发音用本机 Kokoro 合成（离线）；与项目 [tts].voice_main 保持一致可被调用方覆盖
os.environ.setdefault('OPENPRONOUNCE_TTS', 'kokoro')
os.environ.setdefault('OPENPRONOUNCE_TTS_VOICE', 'af_heart')

import re
import subprocess
import tempfile
import threading
from collections import Counter
from pathlib import Path

_score_lock = threading.Lock()
_scorer: 'PronScorer | None' = None


def prepare_audio(raw_bytes: bytes, mime: str = 'audio/webm') -> str:
    """把前端录音（webm/opus/ogg/wav/任意 ffmpeg 认识的格式）转成 16kHz 单声道 WAV。

    返回 WAV 文件路径（调用方负责用后删除）。用 imageio-ffmpeg 自带二进制，
    无需系统安装 ffmpeg，与 server.to_mp3 同源。
    """
    mime = (mime or '').lower()
    if 'webm' in mime:
        ext = '.webm'
    elif 'ogg' in mime or 'opus' in mime:
        ext = '.ogg'
    elif 'mp4' in mime or 'm4a' in mime:
        ext = '.m4a'
    elif 'wav' in mime:
        ext = '.wav'
    else:
        ext = '.bin'
    fd, tmp_in = tempfile.mkstemp(suffix=ext, prefix='speaking-')
    try:
        os.write(fd, raw_bytes)
    finally:
        os.close(fd)
    tmp_wav = tmp_in + '.16k.wav'
    ok = False
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run(
            [ff, '-y', '-v', 'error', '-i', tmp_in,
             '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', tmp_wav],
            capture_output=True, timeout=120)
        if r.returncode != 0 or not Path(tmp_wav).exists():
            raise RuntimeError('音频转码失败：' + r.stderr.decode('utf-8', 'replace')[:200])
        ok = True
        return tmp_wav
    finally:
        try:
            os.remove(tmp_in)
        except OSError:
            pass
        if not ok:  # 转码失败时清理残留的 tmp_wav，避免泄漏
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def _band(score: float) -> str:
    if score >= 85:
        return '优秀'
    if score >= 70:
        return '良好'
    if score >= 55:
        return '及格'
    if score >= 40:
        return '较差'
    return '需重练'


def _hint(errors: list[dict]) -> str:
    if not errors:
        return '发音标准，继续保持！'
    phones = Counter()
    words = []
    for e in errors:
        words.append(e.get('word', ''))
        for p in (e.get('phones') or []):
            exp, heard = p.get('expected'), p.get('heard')
            if exp and heard and exp != heard:
                phones[(exp, heard)] += 1
    parts = []
    top = [f'{exp} → {heard}' for (exp, heard), _ in phones.most_common(3)]
    if top:
        parts.append('重点区分：' + '、'.join(top))
    seen, wlist = set(), []
    for w in words:
        if w and w.lower() not in seen:
            seen.add(w.lower())
            wlist.append(w)
    if wlist:
        parts.append('注意词语：' + '、'.join(wlist[:6]))
    return '；'.join(parts) if parts else '部分词语发音不标准，建议多跟读范本'


def normalize_result(res: dict, reference_text: str) -> dict:
    """把 OpenPronounce 的原始输出规整为前端友好的稳定 schema。"""
    diffs = res.get('differences', {}) or {}
    raw_errors = diffs.get('errors', []) or []
    err_by_word = {}
    for e in raw_errors:
        err_by_word.setdefault((e.get('word') or '').lower(), []).append(e)

    words = []
    for w in re.findall(r"[\w']+", reference_text):
        ok = w.lower() not in err_by_word
        words.append({
            'word': w,
            'ok': ok,
            'confidence': None if ok else err_by_word[w.lower()][0].get('confidence'),
        })

    errors = []
    for e in raw_errors:
        errors.append({
            'word': e.get('word', ''),
            'expected': e.get('expected', ''),
            'actual': e.get('actual', '') or '(漏读)',
            'confidence': e.get('confidence'),
            'phones': e.get('phones', []) or [],
        })

    overall = round(float(res.get('score', 0) or 0), 1)
    return {
        'overall': overall,
        'band': _band(overall),
        'transcribe': res.get('transcribe', ''),
        'word_error_rate': diffs.get('word_error_rate'),
        'phoneme_error_rate': diffs.get('phoneme_error_rate'),
        'acoustic_distance': res.get('acoustic_distance'),
        'words': words,
        'errors': errors,
        'hint': _hint(errors),
    }


class PronScorer:
    """发音评测统一接口：输入 16k 单声道 WAV 路径 + 标准文本，输出规整评分字典。"""

    name = 'base'

    def score(self, wav_path: str, reference_text: str) -> dict:
        raise NotImplementedError


_infer_lock = threading.Lock()  # 评分串行锁：模型推理 + Kokoro 参考声是重资源，防并发导致崩溃/OOM


class OpenPronounceScorer(PronScorer):
    name = 'openpronounce'

    def score(self, wav_path: str, reference_text: str) -> dict:
        try:
            import openpronounce  # 懒加载：模型在首次 score 时才下载/加载
        except ImportError as e:
            raise RuntimeError('未安装 openpronounce，请先执行 pip install openpronounce') from e
        with _infer_lock:  # 串行化，避免并发评分互相干扰
            sound = openpronounce.load_audio(wav_path)  # librosa 直接读 WAV，无需 ffmpeg
            res = openpronounce.compare_audio_with_text(sound, reference_text)
            return normalize_result(res, reference_text)


_REGISTRY: dict[str, type[PronScorer]] = {
    'openpronounce': OpenPronounceScorer,
}


def get_scorer(name: str | None = None) -> PronScorer:
    """懒加载评测器单例（双重检查锁，防并发首次加载模型重复下载）。"""
    global _scorer
    if _scorer is not None and (name is None or _scorer.name == name):
        return _scorer
    with _score_lock:
        name = name or 'openpronounce'
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f'未知发音评测器：{name!r}（可选：{", ".join(_REGISTRY)}）')
        if _scorer is None or _scorer.name != name:
            _scorer = cls()
        return _scorer
