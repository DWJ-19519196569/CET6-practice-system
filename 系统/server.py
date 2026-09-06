# server.py — CET-6 听力系统服务层
# 启动: python server.py  (依赖本地 Qwen 8081 在线; 见 config.toml 质量开关)
import os
sys_dir = os.path.dirname(os.path.abspath(__file__))
_espeak_dir = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'eSpeakNG', 'eSpeak NG')
if _espeak_dir and os.path.isdir(_espeak_dir):  # espeak-ng 存在才加 PATH（Kokoro 本身不依赖）
    os.environ['PATH'] = _espeak_dir + os.pathsep + os.environ.get('PATH', '')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
sys.path.insert(0, sys_dir)
import base64
import io
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import engine as E
from engine import (LLMClient, TTSEngine, build_daily_prompt, parse_dialog_lines,
                    load_config, load_state, current_day, split_sentences)
from wordlist import Wordlist
import pronounce as P

cfg = load_config()
E.init_profiles(cfg)
# 口语评测的参考声与项目 TTS 主音色保持一致（af_heart 等）
os.environ['OPENPRONOUNCE_TTS_VOICE'] = cfg['tts'].get('voice_main', 'af_heart')


def _resolve_path(p: str) -> str:
    """把相对路径解析到 server.py 所在目录（sys_dir）；绝对路径原样返回。"""
    path = Path(p)
    if not path.is_absolute():
        path = Path(sys_dir) / path
    return str(path)


wordlist = Wordlist(_resolve_path(cfg['paths']['wordlist']))
state = load_state(_resolve_path(cfg['paths']['state']))
client = LLMClient(cfg)
_tts: TTSEngine | None = None
_tts_lock = threading.Lock()
_sessions: dict = {}
_session_dir_lock = threading.Lock()  # 互动 session 目录创建的并发互斥
_translate_cache: dict = {}
_cache_lock = threading.Lock()     # 内存缓存（翻译/单词TTS/词义/在线探测）读-改-写互斥
_history_lock = threading.Lock()   # 历史索引读-改-写互斥，防并发归档丢条目
SESSION_TTL_SECONDS = 30 * 60      # 互动 session 空闲 30 分钟自动清理
TRANSLATE_CACHE_MAX = 500          # 逐句翻译缓存容量上限
_online_cache: dict = {}           # /api/models 在线探测缓存（避免每次加载都等待超时）
_online_cache_ttl = 15.0

# ---------- 历史归档 ----------
def _history_dir():
    hd = cfg['paths'].get('history_dir') or os.path.join(sys_dir, '历史记录')
    d = Path(_resolve_path(hd))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _history_index() -> list[dict]:
    p = _history_dir() / 'history.json'
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            # 结构校验：必须是 dict 列表，否则走损坏备份流程（防 {} / [null] 导致后续 KeyError/TypeError）
            if not isinstance(data, list) or not all(isinstance(i, dict) for i in data):
                raise ValueError('history.json 结构非法（期望 list[dict]）')
            return data
        except Exception:
            # 索引损坏：备份而非静默清空，避免下次保存把历史全抹掉
            try:
                backup = p.with_suffix('.json.corrupt-' + time.strftime('%Y%m%d%H%M%S'))
                p.replace(backup)
                print(f'[history] history.json 损坏，已备份为 {backup.name}', flush=True)
            except Exception:
                pass
            return []
    return []


def _save_history_index(items: list[dict]):
    """原子写 history.json（先写临时文件再替换），避免并发读时读到半截 JSON。"""
    p = _history_dir() / 'history.json'
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def _archive(kind: str, payload: dict):
    """归档一条内容到历史：写 content.json + 登记到索引。返回条目 id。"""
    entry_dir = _history_dir() / kind / (time.strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:6])
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / 'content.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    item = {
        'id': entry_dir.name,
        'type': kind,
        'date': payload.get('date', date.today().isoformat()),
        'title': payload.get('title', ''),
        'path': str(entry_dir),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with _history_lock:
        items = _history_index()
        items.append(item)
        _save_history_index(items)
    return item['id']


def _history_list(kind: str) -> list[dict]:
    items = [i for i in _history_index() if i['type'] == kind]
    return sorted(items, key=lambda x: x.get('created_at', ''), reverse=True)


def _history_get(item_id: str):
    for i in _history_index():
        if i['id'] == item_id:
            cpath = Path(i['path']) / 'content.json'
            if cpath.exists():
                try:
                    content = json.loads(cpath.read_text(encoding='utf-8'))
                except Exception:
                    content = {}
                # 索引字段（id/path/created_at/type/date/title）优先，防 content 里的同名键覆盖
                return {**content, **i}
    return None


def _history_content(item_id: str) -> dict:
    """只读某条目的 content.json 内容（不含索引字段 id/path/created_at），用于合并更新时保留旧字段。"""
    for i in _history_index():
        if i['id'] == item_id:
            cpath = Path(i['path']) / 'content.json'
            if cpath.exists():
                try:
                    return json.loads(cpath.read_text(encoding='utf-8'))
                except Exception:
                    return {}
    return {}


def _delete_attached_audio(item: dict):
    """删除条目关联的音频（每日一篇/互动故事各自独立存储，历史删除时一并清理）。

    新数据中 audio_dir 指向「条目自己的唯一子目录」（每日一篇也按 时间戳_id 建子目录），
    整删安全；老数据中每日一篇的 audio_dir 可能指向共享的「每日一篇/日期」目录，
    此时只删条目自己的音频文件，避免波及同日其它条目。
    """
    import shutil
    cpath = Path(item['path']) / 'content.json'
    if not cpath.exists():
        return
    try:
        content = json.loads(cpath.read_text(encoding='utf-8'))
    except Exception:
        return
    daily_root = Path(_resolve_path(cfg['paths']['daily_dir'])).resolve()
    story_root = Path(_resolve_path(cfg['paths']['story_dir'])).resolve()
    audio_dir = content.get('audio_dir') or ''
    audio_file = content.get('audio_file') or ''
    if audio_dir:
        p = Path(_resolve_path(audio_dir)).resolve()
        # 老数据共享的「每日一篇/日期」目录不能整删
        shared_daily_date_dir = (p.parent == daily_root
                                 and re.fullmatch(r'\d{4}-\d{2}-\d{2}', p.name))
        if not (p == daily_root or p == story_root or shared_daily_date_dir):
            # 安全校验：只在 每日一篇/ 或 互动故事/ 目录之下的子目录才删
            if daily_root in p.parents or story_root in p.parents:
                shutil.rmtree(p, ignore_errors=True)
                return
    # 兜底：老条目只有 audio_file（或共享日期目录），只删条目自己的文件
    if audio_file:
        f = Path(_resolve_path(audio_file)).resolve()
        if f.exists() and (daily_root in f.parents or story_root in f.parents):
            try:
                f.unlink()
            except OSError:
                pass
            # 顺带清理同名派生文件（mp3/srt/txt），不碰目录里其它条目的文件
            stem = f.stem
            for ext in ('.mp3', '.srt', '.txt'):
                sibling = f.with_name(stem + ext)
                try:
                    if sibling.exists():
                        sibling.unlink()
                except OSError:
                    pass
            # 每日一篇还有 words.txt（词命中统计）
            if item.get('type') == 'daily':
                wf = f.with_name('words.txt')
                try:
                    if wf.exists():
                        wf.unlink()
                except OSError:
                    pass


def _history_delete(item_id: str) -> bool:
    import shutil
    with _history_lock:
        items = _history_index()
        new = []
        removed = False
        for i in items:
            if i['id'] == item_id:
                removed = True
                _delete_attached_audio(i)          # 先删关联音频目录
                shutil.rmtree(Path(i['path']), ignore_errors=True)  # 再删归档目录
            else:
                new.append(i)
        if removed:
            _save_history_index(new)
        return removed


def _history_update(item_id: str, payload: dict) -> bool:
    """更新一条已归档内容（如批改覆盖出题条目，避免历史出现两条重复）。"""
    with _history_lock:
        items = _history_index()
        for i in items:
            if i['id'] == item_id:
                cpath = Path(i['path']) / 'content.json'
                if not cpath.exists():
                    return False
                payload.setdefault('date', i.get('date', date.today().isoformat()))
                payload.setdefault('title', i.get('title', ''))
                cpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                i['title'] = payload.get('title', i.get('title', ''))
                i['date'] = payload.get('date', i.get('date'))
                _save_history_index(items)
                return True
        return False


def _archive_daily(payload: dict):
    payload.setdefault('date', date.today().isoformat())
    payload.setdefault('title', payload.get('form', '每日一篇'))
    return _archive('daily', payload)


def _archive_story(payload: dict):
    payload.setdefault('date', date.today().isoformat())
    segs = payload.get('segments') or []
    payload.setdefault('title', (segs[0][:24] if segs else '') or '互动故事')
    return _archive('story', payload)


def _trw_title(payload: dict) -> str:
    """翻译的 task 没有 title（只有 keyword/material），用 keyword 或材料开头做标题，避免显示 'translation'。"""
    task = payload.get('task', {}) or {}
    if payload.get('type') == 'translation':
        return task.get('keyword') or (task.get('material') or '')[:18] or '汉译英'
    return task.get('title') or '写作'


def _archive_trw(payload: dict):
    payload.setdefault('date', date.today().isoformat())
    payload.setdefault('title', _trw_title(payload))
    # 按 writing/translation 归档到各自独立历史
    return _archive(payload.get('type', 'translation'), payload)


def _trw_exclude(kind: str) -> list[str]:
    """去重源：读翻译/写作各自的历史（落盘），取最近条目的主题关键词，跨天/重启有效。"""
    seen = []
    for it in _history_list(kind)[:15]:  # 最近 15 条
        content = _history_get(it['id']) or {}
        task = content.get('task', {})
        kw = (task.get('keyword') or task.get('title') or '').strip()
        if kw and kw not in seen:
            seen.append(kw)
    return seen


# ---------- 启动预热 ----------

def _warmup_models():
    """后台预热：把 Kokoro TTS 与 wav2vec2 评分模型加载到 GPU/内存，避免首次使用长时间等待。"""
    try:
        t0 = time.time()
        import tempfile
        import soundfile as sf
        warm_text = 'Warm up.'
        tts = get_tts()
        with _tts_lock:
            audio = tts.synth(warm_text, voice=cfg['tts']['voice_main'])
        wav = os.path.join(tempfile.gettempdir(), 'cet6_warmup.wav')
        sf.write(wav, audio, 24000)
        try:
            scorer = P.get_scorer(cfg['speaking'].get('scorer', 'openpronounce'))
            scorer.score(wav, warm_text)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        print(f'[warmup] TTS + 发音评分模型已就绪 (耗时 {time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'[warmup] 预热失败: {e}', flush=True)


@asynccontextmanager
async def _lifespan(_app):
    threading.Thread(target=_warmup_models, daemon=True).start()
    yield


app = FastAPI(title='CET-6 听力系统', lifespan=_lifespan)

# 简单访问令牌（config [server] access_token 非空时启用）：拦截 /api/*，页面 HTML 放行
_access_token = (cfg.get('server') or {}).get('access_token', '') or ''


@app.middleware('http')
async def _auth_middleware(request, call_next):
    if _access_token and request.url.path.startswith('/api'):
        tok = (request.headers.get('authorization') or '').removeprefix('Bearer ').strip()
        if not tok:
            tok = request.cookies.get('cet6_token', '')
        if tok != _access_token:
            return JSONResponse({'detail': '需要访问令牌'}, status_code=401)
    return await call_next(request)


def get_tts() -> TTSEngine:
    global _tts
    if _tts is None:
        with _tts_lock:  # 双重检查，防并发首次请求重复加载 Kokoro 管线
            if _tts is None:
                _tts = TTSEngine(cfg)
    return _tts


# ---------- 工具 ----------

def wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, audio, 24000, format='WAV', subtype='PCM_16')
    return base64.b64encode(buf.getvalue()).decode()


def to_mp3(wav_path: str, mp3_path: str):
    """wav → mp3（imageio-ffmpeg 自带二进制，无系统依赖）。失败静默（wav 仍可用）。"""
    try:
        import imageio_ffmpeg
        import subprocess
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        # 先删旧文件再写，避免覆盖失败残留旧录音（如浏览器正在播放锁定文件时静默降级）
        try:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        except OSError:
            pass
        subprocess.run([ff, '-y', '-i', wav_path, '-codec:a', 'libmp3lame',
                        '-b:a', '128k', mp3_path], capture_output=True, timeout=120)
        return Path(mp3_path).exists()
    except Exception:
        return False


def fmt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def srt_text(timeline: list[tuple[str, float, float]]) -> str:
    """timeline: [(text, start, dur)]"""
    out = []
    for i, (text, start, dur) in enumerate(timeline, 1):
        out.append(f'{i}\n{fmt_ts(start)} --> {fmt_ts(start + dur)}\n{text.strip()}\n')
    return '\n'.join(out)


def synth_with_timeline(lines: list[tuple[str, str]]):
    """合成脚本（对话或独白），返回 (audio, timeline, word_timeline)。
    timeline 为逐句 [(label, start, dur)]；word_timeline 为逐句逐词 [{'start','dur'}]（绝对秒）。"""
    tts = get_tts()
    sil = np.zeros(int(24000 * 0.25))
    chunks, timeline, word_timeline = [], [], []
    pos = 0.0
    with _tts_lock:
        for speaker, text in lines:
            audio, words = tts.synth_line_timed(text, speaker)
            if len(audio) == 0:
                continue
            label = text if speaker == 'N' else f'{speaker}: {text}'
            timeline.append((label, pos, len(audio) / 24000))
            word_timeline.append([{'start': round(pos + w[1], 3), 'dur': round(w[2] - w[1], 3)} for w in words])
            chunks.append(audio)
            chunks.append(sil.copy())
            pos += len(audio) / 24000 + 0.25
    if not chunks:
        return np.zeros(0, dtype=np.float32), [], []
    return np.concatenate(chunks)[:-len(sil)], timeline, word_timeline


def sse(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


def llm_stream(prompt: str):
    """调 LLM 流式接口，yield 文本增量。连接失败（尚未产生任何输出）时重试一次。"""
    payload = {
        'model': client.p['model'],
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': client.temperature,
        'max_tokens': client.max_tokens,
        'stream': True,
    }
    if client.no_think and client.active == 'local':
        payload['chat_template_kwargs'] = {'enable_thinking': False}
    if client.active == 'cloud':
        payload['thinking'] = {'type': 'disabled'}
    headers = {'Authorization': f"Bearer {client.p.get('api_key', 'local')}"}
    url = client.p['base_url'].rstrip('/') + '/chat/completions'
    for attempt in range(client.max_retries + 1):
        emitted = False
        try:
            with httpx.stream('POST', url, json=payload, headers=headers, timeout=client.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith('data: '):
                        continue
                    body = line[6:]
                    if body.strip() == '[DONE]':
                        return
                    try:
                        d = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = d.get('choices') or []
                    if not choices:
                        continue  # 空 choices 防御：避免 [0] 越界
                    delta = (choices[0] or {}).get('delta', {})
                    piece = delta.get('content')
                    if piece:
                        emitted = True
                        yield piece
            return
        except httpx.TransportError:
            # 已流出部分内容就不再重试（避免前端收到重复句子）
            if emitted or attempt >= client.max_retries:
                raise
        except httpx.HTTPStatusError as e:
            # 临时 5xx（502/503/504）安全重试一次；4xx 不重试
            if emitted or attempt >= client.max_retries or not (500 <= e.response.status_code < 600):
                raise


def story_stream_factory(session: dict, prompt: str, choice: str | None = None):
    """互动段落的 SSE 生成器：LLM 流 → 切句 → TTS → 逐句推送 → 选项 → done。

    choice 非空时在「本段生成成功后」才记入 session['choices']，避免续写失败导致
    选择与段落错位（choices 多一条而 segments 没多）。
    """
    parser = E.SegmentStreamParser()
    idx = 0
    tts = get_tts()

    def gen():
        nonlocal idx
        seg_sents = []
        try:
            for delta in llm_stream(prompt):
                for sent in parser.feed(delta):
                    with _tts_lock:
                        audio, words = tts.synth_timed(sent)
                    if len(audio) == 0:
                        continue
                    seg_sents.append((sent, audio))
                    yield sse('sentence', {'i': idx, 'text': sent, 'audio': wav_b64(audio),
                                           'word_timeline': [{'start': round(w[1], 3), 'dur': round(w[2] - w[1], 3)} for w in words]})
                    idx += 1
            result = parser.finish()
            # 尾句兜底（段末无空白的半句，parser.finish 已 flush 到 sents 但没合成过）
            flushed = [s for s in parser.sents if s not in [x[0] for x in seg_sents]]
            for sent in flushed:
                with _tts_lock:
                    audio, words = tts.synth_timed(sent)
                if len(audio):
                    seg_sents.append((sent, audio))
                    yield sse('sentence', {'i': idx, 'text': sent, 'audio': wav_b64(audio),
                                           'word_timeline': [{'start': round(w[1], 3), 'dur': round(w[2] - w[1], 3)} for w in words]})
                    idx += 1
            # 会话记录
            session['segments'].append(result['segment'])
            session['segment_audio'].append([a for _, a in seg_sents])
            session['options'] = result['options']
            # 生成成功后才记录选择，保证 choices 与 segments 对齐
            if choice is not None:
                session['choices'].append(choice)
            yield sse('options', {'options': result['options']})
            yield sse('done', {'segment': result['segment'], 'n_sentences': len(seg_sents)})
        except Exception as e:
            # 生成中途出错也发一个 error 事件，前端据此解除 busy，避免卡死
            yield sse('error', {'message': str(e)})

    return gen()


def build_history(session: dict) -> str:
    ip = cfg['interactive']
    keep = ip['history_max_segments']
    parts = []
    for seg in session['segments'][-keep:]:
        parts.append(seg)
    return '\n\n'.join(parts) if parts else '(the story has just begun)'


def new_session_dir() -> Path:
    d = Path(_resolve_path(cfg['paths']['story_dir'])) / date.today().strftime('%Y-%m-%d')
    d.mkdir(parents=True, exist_ok=True)
    with _session_dir_lock:
        n = 1
        while True:
            sd = d / f'session-{n:02d}'
            try:
                sd.mkdir(exist_ok=False)  # 存在则报错重试，避免并发拿到同一目录
                return sd
            except FileExistsError:
                n += 1


def _prune_sessions():
    """清理空闲超时的互动 session（用户关页/刷新后不会显式 end，防止内存泄漏）。"""
    now = time.time()
    stale = [sid for sid, s in _sessions.items()
             if now - s.get('last_active', now) > SESSION_TTL_SECONDS]
    for sid in stale:
        _sessions.pop(sid, None)


def _get_session(sid: str):
    session = _sessions.get(sid)
    if session:
        session['last_active'] = time.time()
    return session


# ---------- 模型切换 ----------

def _check_online(base_url: str, api_key: str | None = None) -> bool:
    """探测端点在线状态，带短 TTL 缓存，避免每次加载模型栏都等待超时。"""
    now = time.time()
    with _cache_lock:
        cached = _online_cache.get(base_url)
        if cached and now - cached[0] < _online_cache_ttl:
            return cached[1]
    result = False
    try:
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else None
        r = httpx.get(base_url.rstrip('/') + '/models', headers=headers, timeout=2)
        result = r.status_code == 200
    except Exception:
        result = False
    with _cache_lock:
        _online_cache[base_url] = (now, result)
    return result


@app.get('/api/models')
def api_models():
    local = cfg['llm']['local']
    cloud = cfg['llm']['cloud']
    mk = E._read_deepseek_key()
    return {
        'active': client.active,
        'model': client.p['model'],
        'local': {'online': _check_online(local['base_url']), 'model': local['model'],
                  'base_url': local['base_url']},
        'cloud': {'online': _check_online(cloud['base_url'], mk), 'model': cloud['model'],
                  'base_url': cloud['base_url']},
        'cloud_models': list(cfg['llm']['cloud'].get('models', [])),
        'has_key': bool(mk),
    }


class ModelReq(BaseModel):
    profile: str  # 'local' | 'cloud'
    model: str | None = None  # 云端时可选指定模型（从 cloud_models 选）


@app.post('/api/model')
def api_model_switch(req: ModelReq):
    if req.model and req.profile == 'cloud':
        allowed = cfg['llm']['cloud'].get('models', [])
        if req.model not in allowed:
            raise HTTPException(400, f'未知模型：{req.model}（可选：{", ".join(allowed)}）')
    try:
        client.switch(req.profile, req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return client.status()


class ApikeyReq(BaseModel):
    key: str


@app.post('/api/model/apikey')
def api_model_apikey(req: ApikeyReq):
    """前端填写 DeepSeek 官方 API key：存文件 + 立即更新当前 client（云端生效）。"""
    key = req.key.strip()
    if not key:
        raise HTTPException(400, 'key 为空')
    E.save_deepseek_key(key)
    client.p['api_key'] = key
    _online_cache.clear()  # key 变化后强制重新探测在线状态
    return {'saved': True, 'has_key': True}


# ---------- 每日一篇 ----------

@app.post('/api/daily')
def api_daily():
    day = current_day(state)
    words = wordlist.sample(day, cfg['daily']['sample_n'])
    form_idx = day % 3
    form = E.DAILY_FORMS[form_idx]
    prompt = build_daily_prompt([w for w, _ in words], form_idx, cfg)
    t0 = time.time()
    raw = client.chat(prompt)
    gen_dt = time.time() - t0
    if not raw.strip():
        raise HTTPException(500, 'LLM 返回空文本')
    raw = raw.strip()

    # 解析与合成（按句合成，时间轴与前端展示单元一一对应，供逐句高亮）
    if form['key'] == 'conversation':
        lines = parse_dialog_lines(raw)
        if not any(s == 'A' for s, _ in lines) or not any(s == 'B' for s, _ in lines):
            raise HTTPException(500, '对话格式解析失败（缺 A:/B: 行）')
        sents = []
        for spk, line in lines:
            for s in split_sentences(line):
                sents.append((spk, s))
        lines = sents
    else:
        lines = [('N', s) for s in split_sentences(raw)]
    t0 = time.time()
    audio, timeline, word_timeline = synth_with_timeline(lines)
    tts_dt = time.time() - t0
    if len(audio) == 0:
        raise HTTPException(500, 'TTS 合成失败')

    # 归档：按「日期/时间戳_id」建唯一子目录，避免同日重复生成互相覆盖、也便于历史删除只删自己
    out_dir = (Path(_resolve_path(cfg['paths']['daily_dir']))
               / date.today().strftime('%Y-%m-%d')
               / (time.strftime('%H%M%S') + '_' + uuid.uuid4().hex[:6]))
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    audio_path = out_dir / 'story.wav'
    sf.write(audio_path, audio, 24000)
    to_mp3(str(audio_path), str(out_dir / 'story.mp3'))
    (out_dir / 'story.txt').write_text(
        f'[{form["name"]}] {date.today().isoformat()} | day {day} | 生成 {gen_dt:.0f}s + 合成 {tts_dt:.0f}s\n\n{raw}\n',
        encoding='utf-8')
    (out_dir / 'story.srt').write_text(srt_text(timeline), encoding='utf-8')
    hit = wordlist.coverage(raw)
    cov_lines = '\n'.join(f'{w}\t{m}' for w, m in hit)
    (out_dir / 'words.txt').write_text(
        f'本篇命中 {len(hit)} 词（采样 {len(words)}）\n\n{cov_lines}\n', encoding='utf-8')

    # 自动归档到历史
    _archive_daily({
        'form': form['name'], 'date': date.today().isoformat(),
        'text': raw, 'units': E.build_display_units(raw, form['key']),
        'audio_file': str(audio_path), 'audio_dir': str(out_dir),
        'duration_s': round(len(audio) / 24000, 1),
        'timeline': [{'start': round(t[1], 3), 'dur': round(t[2], 3)} for t in timeline],
        'word_timeline': word_timeline,
        'title': f"{form['name']} · {date.today().isoformat()}",
    })

    return JSONResponse({
        'form': form['name'], 'date': date.today().isoformat(), 'dir': str(out_dir),
        'audio_file': str(audio_path), 'duration_s': round(len(audio) / 24000, 1),
        'words_sampled': len(words), 'words_hit': len(hit), 'text': raw,
        'units': E.build_display_units(raw, form['key']),
        'timeline': [{'start': round(t[1], 3), 'dur': round(t[2], 3)} for t in timeline],
        'word_timeline': word_timeline,
    })

@app.get('/api/daily/audio')
def api_daily_audio(date: str, dir: str | None = None):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise HTTPException(400, 'date 必须是 YYYY-MM-DD')
    base = Path(_resolve_path(cfg['paths']['daily_dir'])) / date
    if dir:
        p = Path(_resolve_path(dir)) / 'story.wav'
        # 安全：dir 必须位于当日目录之下，防止任意路径读取
        if base.resolve() not in p.resolve().parents and p.resolve() != base.resolve():
            raise HTTPException(400, 'dir 不在当日目录内')
    else:
        # 兼容旧数据：优先最新的时间戳子目录，其次当日根目录直存的 story.wav
        candidates = list(base.glob('*/story.wav'))
        if (base / 'story.wav').exists():
            candidates.append(base / 'story.wav')
        if not candidates:
            raise HTTPException(404, 'not generated yet')
        p = max(candidates, key=lambda c: c.stat().st_mtime)
    if not p.exists():
        raise HTTPException(404, 'not generated yet')
    return FileResponse(p, media_type='audio/wav', headers={'Cache-Control': 'no-store'})


class TranslateReq(BaseModel):
    text: str


@app.post('/api/translate')
def api_translate(req: TranslateReq):
    """逐句翻译：当前激活的生成模型出中文译文，带内存缓存（重复句不重复请求）。"""
    text = req.text.strip()
    if not text or len(text) > 500:
        raise HTTPException(400, 'text 为空或过长')
    with _cache_lock:
        cached = _translate_cache.get(text)
    if cached is not None:
        return {'translation': cached, 'cached': True}
    raw = client.chat(E.build_translation_prompt(text))
    if not raw.strip():
        raise HTTPException(502, 'LLM 返回空翻译')
    result = raw.strip()
    # 简单 FIFO 容量上限，防止缓存无限增长
    with _cache_lock:
        if len(_translate_cache) >= TRANSLATE_CACHE_MAX:
            _translate_cache.pop(next(iter(_translate_cache)), None)
        _translate_cache[text] = result
    return {'translation': result, 'cached': False}


# ---------- 互动模式 ----------

class ChoiceReq(BaseModel):
    session_id: str
    choice: str  # 'A'/'B'/'C' 或自由文本（D）


def _story_vocab() -> str:
    """互动故事默认织入六级词（与每日一篇同步当天词块，无开关）。"""
    day = current_day(state)
    words = wordlist.sample(day, 25)
    return E.build_interactive_vocab_note([w for w, _ in words])


@app.post('/api/story/start')
def api_story_start():
    _prune_sessions()
    sid = uuid.uuid4().hex[:12]
    session = {'id': sid, 'segments': [], 'segment_audio': [],
               'choices': [], 'options': [], 'dir': new_session_dir(),
               'last_active': time.time()}
    _sessions[sid] = session
    ip = cfg['interactive']
    prompt = E.INTERACTIVE_START_PROMPT.format(
        min_words=ip['segment_words_min'], max_words=ip['segment_words_max'])
    # 默认织词（不破坏输出格式）：词汇备注插在 Output format 之前
    prompt = prompt.replace('Output format', _story_vocab() + '\nOutput format')
    return StreamingResponse(story_stream_factory(session, prompt),
                             media_type='text/event-stream',
                             headers={'X-Session-Id': sid})


@app.post('/api/story/choose')
def api_story_choose(req: ChoiceReq):
    session = _get_session(req.session_id)
    if not session:
        raise HTTPException(404, 'session not found')
    ip = cfg['interactive']
    prompt = E.INTERACTIVE_CONTINUE_PROMPT.format(
        history=build_history(session), choice=req.choice,
        min_words=ip['segment_words_min'], max_words=ip['segment_words_max'])
    prompt = prompt.replace('Output format', _story_vocab() + '\nOutput format')
    return StreamingResponse(story_stream_factory(session, prompt, choice=req.choice),
                             media_type='text/event-stream')


@app.post('/api/story/end')
def api_story_end(req: ChoiceReq):
    session = _sessions.pop(req.session_id, None)
    if not session:
        raise HTTPException(404, 'session not found')
    # 拼接整局音频
    sil = np.zeros(int(24000 * 0.4))
    parts = []
    for seg_audios in session['segment_audio']:
        for a in seg_audios:
            parts.append(a)
        parts.append(sil.copy())
    if parts:
        audio = np.concatenate(parts)[:-len(sil)]
    else:
        audio = np.zeros(0, dtype=np.float32)
    import soundfile as sf
    sf.write(session['dir'] / 'story.wav', audio, 24000)
    to_mp3(str(session['dir'] / 'story.wav'), str(session['dir'] / 'story.mp3'))
    # transcript：正文 + 选择记录（选择跟在对应段落之后，符合阅读顺序）
    lines = []
    for i, seg in enumerate(session['segments']):
        lines.append(seg + '\n')
        if i < len(session['choices']):
            lines.append(f'### 选择 {i + 1}: {session["choices"][i]}\n')
    (session['dir'] / 'transcript.txt').write_text('\n'.join(lines), encoding='utf-8')
    _archive_story({
        'segments': session['segments'],
        'choices': session['choices'],
        'audio_dir': str(session['dir']),
        'title': (session['segments'][0][:24] if session['segments'] else '互动故事'),
    })
    return JSONResponse({'dir': str(session['dir']),
                         'segments': len(session['segments']),
                         'duration_s': round(len(audio) / 24000, 1)})


# ---------- 翻译写作 ----------

class TrwReq(BaseModel):
    type: str  # 'writing' | 'translation'
    task: dict | None = None  # 出题结果（grade 时带回）
    answer: str = ''


@app.post('/api/trw/generate')
def api_trw_generate(req: TrwReq):
    if req.type not in ('writing', 'translation'):
        raise HTTPException(400, 'type 必须是 writing 或 translation')
    # 去重源：读翻译写作自己的历史（落盘，跨天/重启有效），注入 prompt 让 LLM 避开
    exclude = _trw_exclude(req.type)
    raw = client.chat(E.build_trw_generate_prompt(req.type, exclude))
    if not raw.strip():
        raise HTTPException(502, '出题返回空文本')
    try:
        task = E.parse_trw_generate(raw, req.type)
    except ValueError as e:
        raise HTTPException(502, f'出题解析失败：{e}')
    # 出题即归档：这样所有出过的题（含未批改）都进历史，去重不遗漏
    hid = _archive_trw({'type': req.type, 'task': task, 'status': 'generated'})
    # 带回历史条目 id，批改时更新同一条，避免历史出现「出题 + 批改」两条重复
    return JSONResponse({**task, '_history_id': hid})


@app.post('/api/trw/grade')
def api_trw_grade(req: TrwReq):
    if req.type not in ('writing', 'translation'):
        raise HTTPException(400, 'type 必须是 writing 或 translation')
    answer = req.answer.strip()
    if not answer:
        raise HTTPException(400, '答案为空')
    task = req.task or {}
    hid = task.get('_history_id')
    # 去掉内部字段，避免污染归档的 task
    task = {k: v for k, v in task.items() if not k.startswith('_')}
    raw = client.chat(E.build_trw_grade_prompt(req.type, task, answer))
    if not raw.strip():
        raise HTTPException(502, '批改返回空文本')
    try:
        result = E.parse_trw_grade(raw, req.type)
    except ValueError as e:
        raise HTTPException(502, f'批改解析失败：{e}')
    payload = {'type': req.type, 'task': task, 'answer': answer,
               'grade': result, 'status': 'graded'}
    if hid:
        existing = _history_content(hid)
        if existing.get('sentence_grades'):
            payload['sentence_grades'] = existing['sentence_grades']  # 保留单句模式的批改
    payload.setdefault('title', _trw_title(payload))
    # 有出题条目的历史 id 就原地更新（覆盖为已批改），否则（如旧客户端/测试）新归档一条
    if not (hid and _history_update(hid, payload)):
        _archive_trw(payload)
    return JSONResponse({'type': req.type, **result})


# ---------- 翻译单句模式 ----------

class SentenceRefReq(BaseModel):
    sentence: str


@app.post('/api/trw/sentence/ref')
def api_trw_sentence_ref(req: SentenceRefReq):
    """单句参考译文：中文句子 → 英文参考译文（调用当前生成模型）。"""
    sentence = req.sentence.strip()
    if not sentence or len(sentence) > 300:
        raise HTTPException(400, '句子为空或过长')
    raw = client.chat(E.build_trw_sentence_ref_prompt(sentence))
    if not raw.strip():
        raise HTTPException(502, '参考译文返回空文本')
    return JSONResponse({'reference': raw.strip()})


class SentenceGradeReq(BaseModel):
    exercise_id: str = ''        # 出题条目 id（缺省不归档）
    sentence: str
    answer: str
    index: int | None = None     # 句子序号（归档用）


@app.post('/api/trw/sentence/grade')
def api_trw_sentence_grade(req: SentenceGradeReq):
    """单句批改：打分 + 点评 + 参考译文，并增量归档到对应出题条目的 sentence_grades。"""
    sentence = req.sentence.strip()
    answer = req.answer.strip()
    if not sentence or len(sentence) > 300:
        raise HTTPException(400, '句子为空或过长')
    if not answer:
        raise HTTPException(400, '答案为空')
    raw = client.chat(E.build_trw_sentence_grade_prompt(sentence, answer))
    if not raw.strip():
        raise HTTPException(502, '批改返回空文本')
    try:
        result = E.parse_trw_sentence_grade(raw)
    except ValueError as e:
        raise HTTPException(502, f'批改解析失败：{e}')
    if req.exercise_id:
        item = _history_get(req.exercise_id)
        if item and item.get('type') == 'translation':
            key = str(req.index) if req.index is not None else str(len(item.get('sentence_grades') or {}))
            payload = {
                'type': 'translation',
                'task': item.get('task') or {},
                'answer': item.get('answer'),        # 保留整段模式的答案
                'grade': item.get('grade'),          # 保留整段模式的批改
                'sentence_grades': dict(item.get('sentence_grades') or {}),
                'status': 'assessed',
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            payload['sentence_grades'][key] = {'sentence': sentence, 'answer': answer, 'grade': result}
            payload['title'] = _trw_title(payload)
            _history_update(req.exercise_id, payload)
    return JSONResponse(result)


# ---------- 口语练习 ----------

@app.post('/api/speaking/generate')
def api_speaking_generate():
    """生成一篇朗读题：LLM 出短文 → 逐句切分 → 逐句合成范本音频，出题即归档。"""
    sp = cfg['speaking']
    day = current_day(state)
    words = wordlist.sample(day, sp['sample_n'])
    prompt = E.build_speaking_prompt([w for w, _ in words], sp['passage_words'])
    raw = client.chat(prompt)
    if not raw.strip():
        raise HTTPException(500, 'LLM 返回空文本')
    raw = raw.strip()
    sents = split_sentences(raw)
    if not sents:
        raise HTTPException(500, '短文切句失败')
    tts = get_tts()
    sent_audio = []
    sent_word_timelines = []
    with _tts_lock:
        for s in sents:
            a, words_ts = tts.synth_timed(s, voice=cfg['tts']['voice_main'])
            if len(a) == 0:
                raise HTTPException(500, 'TTS 合成失败')
            sent_audio.append(a)
            sent_word_timelines.append([{'start': round(w[1], 3), 'dur': round(w[2] - w[1], 3)} for w in words_ts])
    hid = _archive('speaking', {
        'type': 'speaking',
        'text': raw,
        'target_words': [w for w, _ in words],
        'sentences': [{'index': i, 'text': s, 'word_timeline': sent_word_timelines[i]} for i, s in enumerate(sents)],
        'status': 'generated',
        'title': raw[:24] or '口语朗读',
    })
    entry_dir = _history_dir() / 'speaking' / hid
    import soundfile as sf
    for i, a in enumerate(sent_audio):
        wav = entry_dir / f'ref_{i}.wav'
        sf.write(wav, a, 24000)
        to_mp3(str(wav), str(entry_dir / f'ref_{i}.mp3'))
    return JSONResponse({
        'id': hid,
        'text': raw,
        'words': [w for w, _ in words],
        'sentences': [
            {'index': i, 'text': s, 'audio_url': f'/api/speaking/audio?id={hid}&kind=ref&i={i}',
             'word_timeline': sent_word_timelines[i]}
            for i, s in enumerate(sents)
        ],
    })


class SpeakingAssessReq(BaseModel):
    exercise_id: str
    audio: str                      # base64 编码录音
    audio_mime: str = 'audio/webm'  # 录音容器格式（webm/opus/ogg/wav...）
    text: str = ''                  # 单句评分时的句子文本（缺省则评整篇）
    sentence_index: int | None = None  # 单句序号（用于归档与录音文件命名）


@app.post('/api/speaking/assess')
def api_speaking_assess(req: SpeakingAssessReq):
    """评测朗读：转码 → OpenPronounce 音素级评分 → 保存录音 → 归档更新。

    单句评分（sentence_index 提供）时评 req.text，结果写 sentence_scores[i]；
    否则评整篇，结果写 score。
    """
    item = _history_get(req.exercise_id)
    if not item or item.get('type') != 'speaking':
        raise HTTPException(404, '口语题不存在')
    full_text = (item.get('text') or '').strip()
    ref_text = (req.text or '').strip() or full_text
    if not ref_text:
        raise HTTPException(400, '朗读内容为空')
    try:
        raw = base64.b64decode(req.audio)
    except Exception:
        raise HTTPException(400, '录音数据无效')
    if len(raw) < 2000:  # 过短录音直接拒（约 <0.1s），避免无意义评测
        raise HTTPException(400, '录音过短，请重新朗读')
    print(f'[speaking-assess] start id={req.exercise_id} idx={req.sentence_index} raw={len(raw)}B mime={req.audio_mime}', flush=True)
    try:
        wav_path = P.prepare_audio(raw, req.audio_mime)
    except Exception as e:
        raise HTTPException(500, f'音频转码失败：{e}')
    entry_dir = Path(item['path'])
    idx = req.sentence_index
    user_name = f'user_{idx}.mp3' if idx is not None else 'user.mp3'
    saved_name = user_name
    try:
        scorer = P.get_scorer(cfg['speaking'].get('scorer', 'openpronounce'))
        result = scorer.score(wav_path, ref_text)
        to_mp3(wav_path, str(entry_dir / user_name))
        if not (entry_dir / user_name).exists():
            # ffmpeg/libmp3lame 失败时保留 WAV，回放接口会 fallback 到 .wav
            import shutil
            try:
                shutil.copyfile(wav_path, str(entry_dir / (user_name[:-4] + '.wav')))
                saved_name = user_name[:-4] + '.wav'
            except Exception:
                pass  # 兜底 WAV 也失败时不影响评分结果返回
    except Exception as e:
        print(f'[speaking-assess] ERROR {e}', flush=True)
        raise HTTPException(500, f'发音评测失败：{e}')
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    print(f'[speaking-assess] done idx={req.sentence_index} overall={result.get("overall")}', flush=True)
    payload = {
        'type': 'speaking',
        'text': full_text,
        'target_words': item.get('target_words', []),
        'sentences': item.get('sentences', []),
        'sentence_scores': dict(item.get('sentence_scores') or {}),
        'score': item.get('score'),               # 保留整篇评分（单句评分时不丢失）
        'audio_file': item.get('audio_file'),     # 保留整篇录音路径
        'status': 'assessed',
        'title': full_text[:24] or '口语朗读',
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    if idx is not None:
        payload['sentence_scores'][str(idx)] = result
    else:
        payload['score'] = result
        payload['audio_file'] = str(entry_dir / saved_name)
    if not _history_update(req.exercise_id, payload):
        _archive('speaking', payload)
    return JSONResponse({'id': req.exercise_id, 'sentence_index': idx, **result})


@app.get('/api/speaking/audio')
def api_speaking_audio(id: str, kind: str = 'user', i: int | None = None):
    """回放口语录音/范本。kind: user=用户朗读, ref=范本；i=句子序号（缺省为整篇）。"""
    item = _history_get(id)
    if not item or item.get('type') != 'speaking':
        raise HTTPException(404, 'not found')
    if kind == 'ref':
        if i is None:
            # 范本只按句生成 ref_0.mp3 / ref_1.mp3...，从不生成整篇 ref.mp3
            raise HTTPException(400, '整篇范本未生成，请指定 i 播放单句范本')
        fname = f'ref_{i}.mp3'
    else:
        fname = f'user_{i}.mp3' if i is not None else 'user.mp3'
    p = Path(item['path']) / fname
    if not p.exists():
        # to_mp3 失败时回退到已保留的 WAV（口语回放不能因 ffmpeg 失败而 404）
        p_wav = Path(item['path']) / (fname[:-4] + '.wav')
        if p_wav.exists():
            return FileResponse(p_wav, media_type='audio/wav', headers={'Cache-Control': 'no-store'})
        raise HTTPException(404, 'audio not found')
    # no-store：同句重新评分会覆盖同名录音文件，禁止浏览器缓存旧录音
    return FileResponse(p, media_type='audio/mpeg', headers={'Cache-Control': 'no-store'})


# ---------- 单词朗读（口语逐词点读） ----------

_word_tts_cache: dict = {}
WORD_TTS_CACHE_MAX = 300


@app.get('/api/tts/word')
def api_tts_word(text: str):
    """Kokoro 朗读单词/短语/短句（带内存缓存），供逐词点读与笔记本回放。"""
    text = (text or '').strip().lower()
    if not text or len(text) > 500 or not re.fullmatch(r"[a-z0-9\s.,;:!?'\-]+", text):
        raise HTTPException(400, 'text 不合法')
    with _cache_lock:
        wav = _word_tts_cache.get(text)
    if wav is None:
        tts = get_tts()
        with _tts_lock:
            audio = tts.synth(text, voice=cfg['tts']['voice_main'])
        if len(audio) == 0:
            raise HTTPException(500, 'TTS 合成失败')
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format='WAV', subtype='PCM_16')
        wav = buf.getvalue()
        with _cache_lock:  # FIFO 容量上限
            if len(_word_tts_cache) >= WORD_TTS_CACHE_MAX:
                _word_tts_cache.pop(next(iter(_word_tts_cache)), None)
            _word_tts_cache[text] = wav
    return Response(content=wav, media_type='audio/wav', headers={'Cache-Control': 'no-store'})


# ---------- 笔记本 ----------

_nb_lock = threading.RLock()  # 可重入：路由持锁后 _save_notebook 再次加锁不会死锁
_nb_path = os.path.join(sys_dir, '笔记本.json')
_NB_KEYS = ('words', 'patterns', 'writing')
word_meaning_map = {w.lower(): m for w, m in wordlist.entries}  # 词表释义速查（收录自动填释义）
_word_meaning_cache: dict = {}   # 非词表词的 AI 释义缓存
WORD_MEANING_CACHE_MAX = 1000


def _resolve_meaning(word: str) -> str:
    """词义：词表 → 内存缓存 → AI 翻译（气泡翻译与收录共享）。"""
    w = word.lower()
    m = word_meaning_map.get(w)
    if m:
        return m
    with _cache_lock:
        m = _word_meaning_cache.get(w)
    if m:
        return m
    try:
        m = (client.chat(E.build_word_meaning_prompt(word)) or '').strip()
    except Exception:
        m = ''
    if not m:
        m = '（未查到释义）'
    with _cache_lock:  # FIFO
        if len(_word_meaning_cache) >= WORD_MEANING_CACHE_MAX:
            _word_meaning_cache.pop(next(iter(_word_meaning_cache)), None)
        _word_meaning_cache[w] = m
    return m


def _load_notebook() -> dict:
    try:
        data = json.loads(Path(_nb_path).read_text(encoding='utf-8'))
    except Exception:
        data = {}
    nb = {}
    for k in _NB_KEYS:
        nb[k] = data.get(k) if isinstance(data.get(k), list) else []
    return nb


def _save_notebook(nb: dict):
    with _nb_lock:
        tmp = Path(_nb_path + '.tmp')
        tmp.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(_nb_path)


notebook = _load_notebook()


class NbWordReq(BaseModel):
    word: str
    context: str = ''
    source: str = '互动故事'


class WordTranslateReq(BaseModel):
    word: str


@app.post('/api/word/translate')
def api_word_translate(req: WordTranslateReq):
    """单词中文释义（供逐词气泡的「翻译」，默认开启）。"""
    word = req.word.strip()
    if not word or len(word) > 60 or not re.fullmatch(r"[a-zA-Z0-9'\-]+", word):
        raise HTTPException(400, 'word 不合法')
    return JSONResponse({'word': word, 'meaning': _resolve_meaning(word)})


@app.post('/api/notebook/word')
def api_notebook_word(req: NbWordReq):
    """收录生词：自动填释义（词表命中直接用，否则 AI 翻译）。"""
    word = req.word.strip().lower()
    if not word or len(word) > 60 or not re.fullmatch(r"[a-z0-9'\-]+", word):
        raise HTTPException(400, 'word 不合法')
    context = (req.context or '').strip()
    if len(context) > 2000:
        raise HTTPException(400, 'context 过长')
    meaning = _resolve_meaning(word)  # AI 释义较慢，放锁外，避免长时间占用 notebook 锁
    with _nb_lock:
        for e in notebook['words']:
            if (e.get('word') or '').lower() == word:
                return JSONResponse({'duplicate': True, 'entry': e})
        entry = {'id': uuid.uuid4().hex[:12], 'word': word, 'meaning': meaning,
                 'context': context, 'source': (req.source or '互动故事').strip(),
                 'date': date.today().isoformat()}
        notebook['words'].append(entry)
        _save_notebook(notebook)
    return JSONResponse({'duplicate': False, 'entry': entry})


class NbPatternReq(BaseModel):
    zh: str
    en: str
    want_explain: bool = False


@app.post('/api/notebook/pattern')
def api_notebook_pattern(req: NbPatternReq):
    """收录翻译句式：中文原句 + 参考译文，可选 AI 句式解读。"""
    zh = req.zh.strip()
    en = req.en.strip()
    if not zh or not en:
        raise HTTPException(400, '中英文都不能为空')
    if len(zh) > 2000 or len(en) > 2000:
        raise HTTPException(400, '句式内容过长（中英各 ≤ 2000 字符）')
    explain = ''
    if req.want_explain:
        try:
            raw = client.chat(E.build_pattern_explain_prompt(zh, en))
            explain = raw.strip()
        except Exception:
            explain = ''
    with _nb_lock:
        for e in notebook['patterns']:
            if (e.get('zh') or '') == zh:
                return JSONResponse({'duplicate': True, 'entry': e})
        entry = {'id': uuid.uuid4().hex[:12], 'zh': zh, 'en': en, 'explain': explain,
                 'source': '翻译练习', 'date': date.today().isoformat()}
        notebook['patterns'].append(entry)
        _save_notebook(notebook)
    return JSONResponse({'duplicate': False, 'entry': entry})


class NbWritingReq(BaseModel):
    text: str


@app.post('/api/notebook/writing')
def api_notebook_writing(req: NbWritingReq):
    """收录写作好段/固定表达（范文段落）。"""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, '内容为空')
    if len(text) > 5000:
        raise HTTPException(400, '内容过长（≤ 5000 字符）')
    with _nb_lock:
        for e in notebook['writing']:
            if (e.get('text') or '') == text:
                return JSONResponse({'duplicate': True, 'entry': e})
        entry = {'id': uuid.uuid4().hex[:12], 'text': text, 'source': '写作练习',
                 'date': date.today().isoformat()}
        notebook['writing'].append(entry)
        _save_notebook(notebook)
    return JSONResponse({'duplicate': False, 'entry': entry})


@app.get('/api/notebook')
def api_notebook():
    with _nb_lock:
        return JSONResponse({k: list(notebook[k]) for k in _NB_KEYS})


class NbUpdateReq(BaseModel):
    category: str
    id: str
    entry: dict


@app.post('/api/notebook/update')
def api_notebook_update(req: NbUpdateReq):
    """编辑笔记本条目（保留 id/source/date，其余字段覆盖）。"""
    if req.category not in _NB_KEYS:
        raise HTTPException(400, 'category 不合法')
    with _nb_lock:
        for i, e in enumerate(notebook[req.category]):
            if e.get('id') == req.id:
                new_entry = dict(req.entry)
                new_entry['id'] = e.get('id')
                new_entry['source'] = e.get('source') or new_entry.get('source') or ''
                new_entry['date'] = e.get('date') or new_entry.get('date') or date.today().isoformat()
                notebook[req.category][i] = new_entry
                _save_notebook(notebook)
                return JSONResponse({'updated': True, 'entry': new_entry})
    raise HTTPException(404, 'entry not found')


@app.delete('/api/notebook')
def api_notebook_delete(category: str, id: str):
    if category not in _NB_KEYS:
        raise HTTPException(400, 'category 不合法')
    with _nb_lock:
        before = len(notebook[category])
        notebook[category] = [e for e in notebook[category] if e.get('id') != id]
        if len(notebook[category]) == before:
            raise HTTPException(404, 'entry not found')
        _save_notebook(notebook)
    return {'deleted': True}


# ---------- 历史归档 API ----------

@app.get('/api/history')
def api_history(type: str):
    if type not in ('daily', 'story', 'writing', 'translation', 'speaking'):
        raise HTTPException(400, 'type 必须是 daily|story|writing|translation|speaking')
    return {'items': _history_list(type)}


@app.get('/api/history/item')
def api_history_item(id: str):
    item = _history_get(id)
    if not item:
        raise HTTPException(404, 'not found')
    return JSONResponse(item)


@app.delete('/api/history')
def api_history_delete(id: str):
    if not _history_delete(id):
        raise HTTPException(404, 'not found')
    return {'deleted': True}


# ---------- 静态 ----------

@app.get('/')
def index():
    return FileResponse(os.path.join(sys_dir, '前端.html'), headers={'Cache-Control': 'no-store'})


def _ensure_self_signed_cert(cert_dir: Path):
    """确保存在自签名证书（HTTPS 供手机麦克风等安全上下文需求）；缺失则用 cryptography 生成。"""
    cert_path = cert_dir / 'cert.pem'
    key_path = cert_dir / 'key.pem'
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        import ipaddress
        import socket
    except ImportError:
        print('[https] 未安装 cryptography，无法自动生成证书，回退为 HTTP', flush=True)
        return None, None
    cert_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'CET6-local')])
    # SAN 覆盖本机所有局域网 IP + localhost，减少手机"证书不匹配"警告
    sans = [x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
            x509.DNSName('localhost')]
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ('127.0.0.1',):
                sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except Exception:
        pass
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.TraditionalOpenSSL,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f'[https] 已生成自签名证书：{cert_path}', flush=True)
    return str(cert_path), str(key_path)


def _lan_ips() -> list[str]:
    import socket
    try:
        return [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if ip != '127.0.0.1']
    except Exception:
        return []


def _print_access_info(port: int, https: bool):
    scheme = 'https' if https else 'http'
    print('=' * 56, flush=True)
    print('CET-6 服务已启动，同一局域网内手机/平板浏览器可访问：', flush=True)
    for ip in _lan_ips():
        print(f'    {scheme}://{ip}:{port}', flush=True)
    print(f'    本机: {scheme}://127.0.0.1:{port}', flush=True)
    if https:
        print('首次用手机访问会提示证书不受信任：请选择「继续访问/信任」即可', flush=True)
    if _access_token:
        print(f'已启用访问令牌（请输入 config.toml 里设置的 access_token）', flush=True)
    print('=' * 56, flush=True)


def main():
    import uvicorn
    host = (cfg.get('server') or {}).get('host', '0.0.0.0')
    port = cfg['server']['port']
    https = bool((cfg.get('server') or {}).get('https', False))
    cert_path = key_path = None
    if https:
        cert_path, key_path = _ensure_self_signed_cert(Path(sys_dir) / 'certs')
        if not cert_path:
            https = False
    _print_access_info(port, https)
    uvicorn.run(app, host=host, port=port,
                ssl_certfile=cert_path, ssl_keyfile=key_path, log_level='info')


if __name__ == '__main__':
    main()
