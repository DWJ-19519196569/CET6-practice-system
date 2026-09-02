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
from datetime import date
from pathlib import Path

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import engine as E
from engine import (LLMClient, TTSEngine, build_daily_prompt, parse_dialog_lines,
                    load_config, load_state, current_day, split_sentences)
from wordlist import Wordlist

cfg = load_config()
E.init_profiles(cfg)


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
_translate_cache: dict = {}
_interactive_vocab = False  # 互动故事织词开关（默认关，前端 toggle）
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
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
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
                return {**i, **json.loads(cpath.read_text(encoding='utf-8'))}
    return None


def _history_delete(item_id: str) -> bool:
    import shutil
    with _history_lock:
        items = _history_index()
        new = []
        removed = False
        for i in items:
            if i['id'] == item_id:
                removed = True
                shutil.rmtree(Path(i['path']), ignore_errors=True)
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
    payload.setdefault('title', payload.get('segments', [''])[0][:24] or '互动故事')
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


app = FastAPI(title='CET-6 听力系统')


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


def synth_with_timeline(lines: list[tuple[str, str]]) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """合成脚本（对话或独白），返回 (audio, timeline)。行间 250ms 静音计入时间轴。"""
    tts = get_tts()
    sil = np.zeros(int(24000 * 0.25))
    chunks, timeline = [], []
    pos = 0.0
    with _tts_lock:
        for speaker, text in lines:
            audio = tts.synth_line(text, speaker)
            if len(audio) == 0:
                continue
            label = text if speaker == 'N' else f'{speaker}: {text}'
            timeline.append((label, pos, len(audio) / 24000))
            chunks.append(audio)
            chunks.append(sil.copy())
            pos += len(audio) / 24000 + 0.25
    if not chunks:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(chunks)[:-len(sil)], timeline


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
                    delta = d.get('choices', [{}])[0].get('delta', {})
                    piece = delta.get('content')
                    if piece:
                        emitted = True
                        yield piece
            return
        except httpx.TransportError:
            # 已流出部分内容就不再重试（避免前端收到重复句子）
            if emitted or attempt >= client.max_retries:
                raise


def story_stream_factory(session: dict, prompt: str):
    """互动段落的 SSE 生成器：LLM 流 → 切句 → TTS → 逐句推送 → 选项 → done。"""
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
                        audio = tts.synth(sent)
                    if len(audio) == 0:
                        continue
                    seg_sents.append((sent, audio))
                    yield sse('sentence', {'i': idx, 'text': sent, 'audio': wav_b64(audio)})
                    idx += 1
            result = parser.finish()
            # 尾句兜底（段末无空白的半句，parser.finish 已 flush 到 sents 但没合成过）
            flushed = [s for s in parser.sents if s not in [x[0] for x in seg_sents]]
            for sent in flushed:
                with _tts_lock:
                    audio = tts.synth(sent)
                if len(audio):
                    seg_sents.append((sent, audio))
                    yield sse('sentence', {'i': idx, 'text': sent, 'audio': wav_b64(audio)})
                    idx += 1
            # 会话记录
            session['segments'].append(result['segment'])
            session['segment_audio'].append([a for _, a in seg_sents])
            session['options'] = result['options']
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
    n = 1
    while (d / f'session-{n:02d}').exists():
        n += 1
    sd = d / f'session-{n:02d}'
    sd.mkdir()
    return sd


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

    # 解析与合成
    if form['key'] == 'conversation':
        lines = parse_dialog_lines(raw)
        if not any(s == 'A' for s, _ in lines) or not any(s == 'B' for s, _ in lines):
            raise HTTPException(500, '对话格式解析失败（缺 A:/B: 行）')
    else:
        lines = [('N', s) for s in split_sentences(raw)]
    t0 = time.time()
    audio, timeline = synth_with_timeline(lines)
    tts_dt = time.time() - t0
    if len(audio) == 0:
        raise HTTPException(500, 'TTS 合成失败')

    # 归档
    out_dir = Path(_resolve_path(cfg['paths']['daily_dir'])) / date.today().strftime('%Y-%m-%d')
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
        'audio_file': str(audio_path),
        'title': f"{form['name']} · {date.today().isoformat()}",
    })

    return JSONResponse({
        'form': form['name'], 'date': date.today().isoformat(), 'dir': str(out_dir),
        'audio_file': str(audio_path), 'duration_s': round(len(audio) / 24000, 1),
        'words_sampled': len(words), 'words_hit': len(hit), 'text': raw,
        'units': E.build_display_units(raw, form['key']),
    })

@app.get('/api/daily/audio')
def api_daily_audio(date: str):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise HTTPException(400, 'date 必须是 YYYY-MM-DD')
    p = Path(_resolve_path(cfg['paths']['daily_dir'])) / date / 'story.wav'
    if not p.exists():
        raise HTTPException(404, 'not generated yet')
    return FileResponse(p, media_type='audio/wav')


class TranslateReq(BaseModel):
    text: str


@app.post('/api/translate')
def api_translate(req: TranslateReq):
    """逐句翻译：当前激活的生成模型出中文译文，带内存缓存（重复句不重复请求）。"""
    text = req.text.strip()
    if not text or len(text) > 500:
        raise HTTPException(400, 'text 为空或过长')
    cached = _translate_cache.get(text)
    if cached is not None:
        return {'translation': cached, 'cached': True}
    raw = client.chat(E.build_translation_prompt(text))
    if not raw.strip():
        raise HTTPException(502, 'LLM 返回空翻译')
    # 简单 FIFO 容量上限，防止缓存无限增长
    if len(_translate_cache) >= TRANSLATE_CACHE_MAX:
        _translate_cache.pop(next(iter(_translate_cache)), None)
    _translate_cache[text] = raw.strip()
    return {'translation': raw.strip(), 'cached': False}


# ---------- 互动模式 ----------

class ChoiceReq(BaseModel):
    session_id: str
    choice: str  # 'A'/'B'/'C' 或自由文本（D）


def _story_vocab() -> str:
    """互动织词开启时返回词汇织入备注，关闭时返回空串。day 用今天（与每日同步）。"""
    if not _interactive_vocab:
        return ''
    day = current_day(state)
    words = wordlist.sample(day, 25)
    return E.build_interactive_vocab_note([w for w, _ in words])


class VocabReq(BaseModel):
    enabled: bool


@app.post('/api/story/vocab')
def api_story_vocab(req: VocabReq):
    global _interactive_vocab
    _interactive_vocab = req.enabled
    return {'enabled': _interactive_vocab}


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
    # 织词开关开启时不破坏输出格式：词汇备注插在 Output format 之前
    if _interactive_vocab:
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
    session['choices'].append(req.choice)
    prompt = E.INTERACTIVE_CONTINUE_PROMPT.format(
        history=build_history(session), choice=req.choice,
        min_words=ip['segment_words_min'], max_words=ip['segment_words_max'])
    if _interactive_vocab:
        prompt = prompt.replace('Output format', _story_vocab() + '\nOutput format')
    return StreamingResponse(story_stream_factory(session, prompt),
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
    if not req.answer.strip():
        raise HTTPException(400, '答案为空')
    task = req.task or {}
    hid = task.get('_history_id')
    # 去掉内部字段，避免污染归档的 task
    task = {k: v for k, v in task.items() if not k.startswith('_')}
    raw = client.chat(E.build_trw_grade_prompt(req.type, task, req.answer))
    if not raw.strip():
        raise HTTPException(502, '批改返回空文本')
    try:
        result = E.parse_trw_grade(raw, req.type)
    except ValueError as e:
        raise HTTPException(502, f'批改解析失败：{e}')
    payload = {'type': req.type, 'task': task, 'answer': req.answer,
               'grade': result, 'status': 'graded'}
    payload.setdefault('title', _trw_title(payload))
    # 有出题条目的历史 id 就原地更新（覆盖为已批改），否则（如旧客户端/测试）新归档一条
    if not (hid and _history_update(hid, payload)):
        _archive_trw(payload)
    return JSONResponse({'type': req.type, **result})


# ---------- 历史归档 API ----------

@app.get('/api/history')
def api_history(type: str):
    if type not in ('daily', 'story', 'writing', 'translation'):
        raise HTTPException(400, 'type 必须是 daily|story|writing|translation')
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
    return FileResponse(os.path.join(sys_dir, '前端.html'))


def main():
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=cfg['server']['port'], log_level='info')


if __name__ == '__main__':
    main()
