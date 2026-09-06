# engine.py — 故事引擎：prompt 模板 / LLM 客户端 / TTS 封装
import os
_espeak_dir = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'eSpeakNG', 'eSpeak NG')
if _espeak_dir and os.path.isdir(_espeak_dir):  # espeak-ng 存在才加 PATH（Kokoro 本身不依赖）
    os.environ['PATH'] = _espeak_dir + os.pathsep + os.environ.get('PATH', '')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import re
import time
from pathlib import Path

import httpx

# ---------- prompt 模板 ----------

DAILY_FORMS = {
    0: dict(
        key='conversation',
        name='长对话',
        form='a two-person dialogue between two college students',
        length_cfg='length_conversation',
        format_spec='Output the dialogue as alternating lines, each line starting with "A:" or "B:" and nothing else. No stage directions, no narration lines.',
        extra='Make it sound like a real conversation: natural interruptions, reactions, one clear topic.',
    ),
    1: dict(
        key='passage',
        name='短文',
        form='a narrative story with a clear beginning, middle and end',
        length_cfg='length_passage',
        format_spec='Output plain paragraphs only. No title, no headers, no speaker labels.',
        extra='Give the story one small but meaningful turn, not just a sequence of events.',
    ),
    2: dict(
        key='lecture',
        name='讲座',
        form='a short lecture delivered by a professor to students',
        length_cfg='length_lecture',
        format_spec='Output plain paragraphs only. Open with a brief greeting, close with a one-sentence wrap-up.',
        extra='Lecture tone: slightly formal, may use "we can see", "consider", "note that". One concrete example inside.',
    ),
}


def build_daily_prompt(words: list[str], form_idx: int, cfg: dict) -> str:
    f = DAILY_FORMS[form_idx % 3]
    length = cfg['daily'][f['length_cfg']]
    word_list = ', '.join(words)
    return f"""You are writing English listening material for a Chinese student preparing for the CET-6 exam.

Write {f['form']}, about {length} words total, at CET-6 listening difficulty.

Vocabulary requirement:
- Weave as many of these CET-6 words into the text as fits naturally: {word_list}
- Never force a word where it reads awkward; skip it instead
- You may inflect them (plural / past tense / etc.)

Style requirements:
- Sentence length mostly 12-20 words; keep spoken rhythm, readable aloud
- No slang, no internet abbreviations
- {f['extra']}

Output format: {f['format_spec']}
Output the text only. No word count, no notes, no explanation."""


def build_speaking_prompt(words: list[str], length: int) -> str:
    """口语朗读题：CET-6 难度短文，织入采样词，纯文本输出（无标题无标注）。"""
    word_list = ', '.join(words)
    return f"""You are writing English reading-aloud material for a Chinese student preparing for the CET-6 speaking test.

Write a short passage of about {length} words, suitable for reading aloud, at CET-6 difficulty.

Vocabulary requirement:
- Weave as many of these CET-6 words into the passage as fits naturally: {word_list}
- Never force a word where it reads awkward; skip it instead
- You may inflect them (plural / past tense / etc.)

Style requirements:
- Sentence length mostly 12-20 words; natural spoken rhythm, easy to read aloud
- No slang, no internet abbreviations, no hard-to-pronounce proper nouns
- One clear topic with a beginning, middle and end

Output the passage only, as plain text. No title, no word count, no notes, no explanation."""


INTERACTIVE_START_PROMPT = """You are an interactive storyteller for one reader.

Begin a NEW story with an opening segment of {min_words}-{max_words} words in English.

Rules:
- Pick any genre you find compelling for this story
- Establish an intriguing situation fast
- Introduce at most 2 characters in this opening
- End the segment at a moment of tension or decision
- Plain, vivid English at CET-6 reading level

Output format (follow EXACTLY, no extra text before or after):
<SEGMENT>
your opening segment text
</SEGMENT>
A) first option, a short English sentence (8-18 words)
B) second option, a genuinely different direction
C) third option, a genuinely different direction"""


INTERACTIVE_CONTINUE_PROMPT = """Story so far:
{history}

The reader chose: {choice}

Continue the story with the next segment ({min_words}-{max_words} words, English).

Rules:
- Strict continuity with everything above; reuse established names, places, facts
- If the reader's direction is written in Chinese, interpret its meaning and continue the story in English
- End the segment at a new moment of tension or decision
- Do not resolve the whole story unless the reader's choice clearly demands an ending; if you write an ending, still provide three options that could extend the story (e.g. epilogue directions)

Output format (follow EXACTLY, no extra text before or after):
<SEGMENT>
your next segment text
</SEGMENT>
A) first option, a short English sentence (8-18 words)
B) second option, a genuinely different direction
C) third option, a genuinely different direction"""


# ---------- LLM 客户端 ----------

class LLMClient:
    def __init__(self, cfg: dict):
        llm = cfg['llm']
        self.profiles = llm
        self.active = llm['active']
        self.temperature = llm['generation']['temperature']
        self.max_tokens = llm['generation']['max_tokens']
        # 超时与重试：过长的 600s 会让模型卡住时长时间挂起；降到 180s 并在失败时重试 1 次
        self.timeout = llm['generation'].get('timeout', 180)
        self.max_retries = llm['generation'].get('max_retries', 1)
        # 故事创作不需要 CoT；Qwen 模板开关 enable_thinking=false，实测 11s 出 300 词，
        # 默认 medium 会思考失控耗光 max_tokens（实测 8000+ 字符思考、正文为空）
        self.no_think = llm['generation'].get('no_think', True)
        self.p = self._resolve(self.active)

    @staticmethod
    def _resolve(profile: str) -> dict:
        """取 profile 配置；cloud 时从 deepseek_key.txt 注入官方 API key（前端填写）。"""
        p = dict(_LLM_PROFILES[profile])
        if profile == 'cloud':
            p['api_key'] = _read_deepseek_key()
        return p

    def switch(self, profile: str, model: str | None = None):
        if profile not in _LLM_PROFILES:
            raise ValueError(f'unknown profile: {profile}')
        self.active = profile
        self.p = self._resolve(profile)
        if model and profile == 'cloud':
            self.p['model'] = model

    def status(self) -> dict:
        return {'active': self.active, 'model': self.p['model'],
                'base_url': self.p['base_url']}

    def chat(self, prompt: str):
        payload = {
            'model': self.p['model'],
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': False,
        }
        if self.no_think and self.active == 'local':
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        if self.active == 'cloud':
            # DeepSeek 思考模型在复杂 prompt 下会思考失控耗光 max_tokens（实测 4096 tokens 全耗在推理）；
            # 官方参数 thinking.disabled 经 LiteLLM 透传，实测 494 tokens 出 2190 字符正文
            payload['thinking'] = {'type': 'disabled'}
        headers = {'Authorization': f"Bearer {self.p.get('api_key', 'local')}"}
        url = self.p['base_url'].rstrip('/') + '/chat/completions'
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return self._content(r.json())
            except httpx.TransportError:
                # 连接失败/超时重试
                if attempt >= self.max_retries:
                    raise
            except httpx.HTTPStatusError as e:
                # 临时 5xx（502/503/504）安全重试一次；4xx 不重试
                if attempt >= self.max_retries or not (500 <= e.response.status_code < 600):
                    raise

    @staticmethod
    def _content(data: dict) -> str:
        choices = data.get('choices') or []
        if not choices:
            return ''
        msg = (choices[0] or {}).get('message') or {}
        # llama.cpp / LiteLLM 可能把思考内容放 reasoning_content，正文放 content
        return msg.get('content') or ''


# ---------- 文本解析（标记协议：<SEGMENT>...</SEGMENT> + A)/B)/C) 选项行） ----------

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_SEG_RE = re.compile(r'<SEGMENT>\s*(.*?)\s*</SEGMENT>', re.DOTALL)
_OPT_RE = re.compile(r'^[ABC]\)\s*(.+)$')


def strip_think(text: str) -> str:
    return _THINK_RE.sub('', text).strip()


def parse_story(raw: str) -> dict:
    """完整解析一次互动输出（非流式调用/测试用）。"""
    text = strip_think(raw)
    m = _SEG_RE.search(text)
    if not m:
        raise ValueError(f'no <SEGMENT> found in LLM output: {text[:200]!r}')
    seg = m.group(1).strip()
    opts = []
    for line in text[m.end():].splitlines():
        om = _OPT_RE.match(line.strip())
        if om:
            opts.append(om.group(1).strip())
    if len(opts) < 3:
        while len(opts) < 3:
            opts.append('Let the story take a surprising turn.')
    return {'segment': seg, 'options': opts[:3]}


class SegmentStreamParser:
    """增量解析 LLM 流式输出：feed 返回新切出的完整句子（可立即送 TTS），finish 返回整段+选项。"""

    # 句尾 = [.!?] 序列 + 可选收尾引号/括号 + 空白（流式下不用 $ 判尾，段末句由 finish flush）
    _SENT_END_RE = re.compile(r'[.!?]+\s')

    def __init__(self):
        self.pre = ''          # <SEGMENT> 之前的缓冲
        self.body = ''         # 正文累积（</SEGMENT> 前，不含标记本身）
        self.pending = ''      # 未完结的半句
        self.sents = []        # 已切出的句子
        self.in_segment = False
        self.finished = False
        self.tail = ''         # </SEGMENT> 之后的缓冲（选项行）
        self._scope_len = 0    # 上次切句范围长度（用于增量窗口）

    def feed(self, delta: str) -> list[str]:
        """喂入增量，返回新切出的完整句子列表。"""
        if self.finished:
            self.tail += delta
            return []
        if not self.in_segment:
            self.pre += delta
            idx = self.pre.find('<SEGMENT>')
            if idx == -1:
                if len(self.pre) > 400:
                    self.pre = self.pre[-200:]
                return []
            self.in_segment = True
            rest = self.pre[idx + len('<SEGMENT>'):]
            self.pre = ''
            return self._absorb(rest)
        return self._absorb(delta)

    @staticmethod
    def _unclosed_tag_start(s: str) -> int:
        """返回 s 中最后一个未闭合 '<' 的位置（其后无 '>'），无则 -1。
        用于冻结 </SEGMENT 之类流式未闭合标签，防止其碎片被 flush 成句子。"""
        idx = s.rfind('<')
        if idx == -1:
            return -1
        if '>' in s[idx + 1:]:
            return -1
        return idx

    def _absorb(self, text: str) -> list[str]:
        self.body += text
        out = []
        # </SEGMENT> 一旦出现：它之后的内容一律不进切句范围（防标记泄漏进句子）
        end = self.body.find('</SEGMENT>')
        scope = self.body if end == -1 else self.body[:end]
        window = self.pending + scope[self._scope_len:]
        # 冻结最后一个未闭合 '<' 之后的内容：流式下 </SEGMENT 的 '<' 可能先到，
        # 若不做处理会在 finish() 被当作句子 flush（泄漏 '</SEGMENT'）
        freeze = self._unclosed_tag_start(window)
        if freeze != -1:
            cutable, pend = window[:freeze], window[freeze:]
        else:
            cutable, pend = window, ''
        while True:
            m = self._SENT_END_RE.search(cutable)
            if not m:
                break
            sent = cutable[:m.end()].strip()
            if sent:
                self.sents.append(sent)
                out.append(sent)
            cutable = cutable[m.end():]
        self.pending = cutable + pend
        self._scope_len = len(scope)
        if end != -1:
            self.tail = self.body[end + len('</SEGMENT>'):]
            self.body = scope
            # 丢弃 pending 里可能残留的 </SEGMENT 前缀碎片（正文不含 '<'）
            lt = self.pending.rfind('<')
            if lt != -1:
                self.pending = self.pending[:lt].rstrip()
            self.finished = True
        return out

    def finish(self) -> dict:
        """流结束时调用：flush 尾句，解析选项。"""
        if not self.finished:
            end = self.body.find('</SEGMENT>')
            if end != -1:
                self.tail = self.body[end + len('</SEGMENT>'):]
                self.body = self.body[:end]
            self.finished = True
        if self.pending.strip():
            self.sents.append(self.pending.strip())
            self.pending = ''
        seg = self.body.strip()
        opts = []
        for line in (self.tail or '').splitlines():
            om = _OPT_RE.match(line.strip())
            if om:
                opts.append(om.group(1).strip())
        if len(opts) < 3:
            while len(opts) < 3:
                opts.append('Let the story take a surprising turn.')
        return {'segment': seg, 'options': opts[:3]}

# ---------- JSON / 文本解析 ----------

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _balanced_json(text: str, start: int):
    """从 start 处的 '{' 起，用字符串感知的方式找配对的 '}'（容忍字符串内出现 { 或 }）。"""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _escape_newlines_in_strings(text: str) -> str:
    """把「字符串值内部」的裸换行转义为 \\n（LLM 常把长字段写成多行，破坏 JSON）。"""
    out = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            elif ch == '\\':
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == '\n' or ch == '\r':
                out.append('\\n')
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return ''.join(out)


def parse_dialog_lines(text: str) -> list[tuple[str, str]]:
    """解析 "A: xxx / B: xxx" 对话行为 (speaker, line)；非对话行归为旁白 'N'。"""
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([AB])\s*[:：]\s*(.+)$', line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
        else:
            out.append(('N', line))
    return out


_ABBREV_RE = re.compile(r'\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e|U\.S|U\.K|No)\.')


def split_sentences(text: str) -> list[str]:
    """独白文本按句切分（保持原文本完整，用于句级 srt 与前端逐句渲染）。

    保护常见缩写（Mr. / Dr. / U.S. / e.g. 等）的句点，避免在缩写处误切。
    """
    text = text.strip()
    if not text:
        return []
    # 缩写句点先换成控制字符占位，切句后再还原
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace('.', '\x00'), text)
    sents = re.split(r'(?<=[.!?])\s+', protected)
    return [s.replace('\x00', '.') for s in sents if s.strip()]


def build_display_units(text: str, form_key: str) -> list[dict]:
    """把每日一篇正文拆成前端逐句展示单元：每项 {speaker, text}。
    conversation 按 A:/B: 行保留说话人，其余题型按句切分为旁白。"""
    if form_key == 'conversation':
        units = []
        for spk, line in parse_dialog_lines(text):
            for sent in split_sentences(line):
                units.append({'speaker': spk, 'text': sent})
        return units
    return [{'speaker': 'N', 'text': s} for s in split_sentences(text)]


def build_translation_prompt(text: str) -> str:
    """逐句翻译 prompt：只输出中文译文，不解释。"""
    return f'Translate the following English into natural 简体中文. Output ONLY the Chinese translation, nothing else.\n\n{text}'


def build_interactive_vocab_note(words: list[str]) -> str:
    """互动故事织词备注：要求把采样词自然织入段落，不硬塞。默认可关（不织词）。"""
    wl = ', '.join(words)
    return (f'\n\nVocabulary weave (only if it reads naturally):\n'
            f'- Weave as many of these CET-6 words into your segment as fits naturally: {wl}\n'
            f'- Never force a word where it reads awkward; skip it instead.\n')


# ---------- JSON / 文本解析 ----------


# ---------- TTS 封装 ----------

class TTSEngine:
    """Kokoro 单例封装。CPU 推理，进程内常驻。"""

    def __init__(self, cfg: dict):
        import numpy as np
        from kokoro import KPipeline
        self.np = np
        self.cfg = cfg['tts']
        self.pipeline = KPipeline(lang_code='a')

    def synth(self, text: str, voice: str | None = None, speed: float | None = None) -> 'np.ndarray':
        voice = voice or self.cfg['voice_main']
        speed = speed or self.cfg['speed']
        chunks = [self.np.asarray(a) for _, _, a in self.pipeline(text, voice=voice, speed=speed)]
        if not chunks:
            return self.np.zeros(0, dtype=self.np.float32)
        return self.np.concatenate(chunks)

    def synth_line(self, text: str, speaker: str) -> 'np.ndarray':
        """按说话人选音色：A=女声 B=男声 N=旁白女声。句间加 250ms 静音由调用方处理。"""
        voice_map = {'A': self.cfg['voice_main'], 'B': self.cfg['voice_male'], 'N': self.cfg['voice_main']}
        return self.synth(text, voice_map.get(speaker, self.cfg['voice_main']))

    def synth_timed(self, text: str, voice: str | None = None, speed: float | None = None):
        """合成并返回 (audio, word_timings)。word_timings 为 [(word, start_ts, end_ts)]（秒），
        词级时间戳来自 kokoro 内部音素对齐（已按 speed 调整）。"""
        voice = voice or self.cfg['voice_main']
        speed = speed or self.cfg['speed']
        chunks, words = [], []
        offset = 0.0
        for res in self.pipeline(text, voice=voice, speed=speed):
            audio = self.np.asarray(res.audio) if res.audio is not None else self.np.zeros(0, dtype=self.np.float32)
            if len(audio) == 0:
                continue
            dur = len(audio) / 24000.0
            for t in (res.tokens or []):
                st, en = getattr(t, 'start_ts', None), getattr(t, 'end_ts', None)
                if t.text and st is not None and en is not None:
                    words.append((t.text, offset + float(st), offset + float(en)))
            chunks.append(audio)
            offset += dur
        if not chunks:
            return self.np.zeros(0, dtype=self.np.float32), []
        return self.np.concatenate(chunks), words

    def synth_line_timed(self, text: str, speaker: str):
        """synth_line + 词级时间戳版本。"""
        voice_map = {'A': self.cfg['voice_main'], 'B': self.cfg['voice_male'], 'N': self.cfg['voice_main']}
        return self.synth_timed(text, voice_map.get(speaker, self.cfg['voice_main']))

    def synth_script(self, lines: list[tuple[str, str]]) -> 'np.ndarray':
        """合成整段脚本（对话或独白），行间 250ms 停顿。"""
        sil = self.np.zeros(int(24000 * 0.25), dtype=self.np.float32)
        out = []
        for speaker, line in lines:
            audio = self.synth_line(line, speaker)
            if len(audio):
                out.append(audio)
                out.append(sil)
        if not out:
            return self.np.zeros(0, dtype=self.np.float32)
        return self.np.concatenate(out)[:-len(sil)]  # 去掉末尾多余静音


# ---------- 状态 ----------

def load_state(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            # 结构校验：必须是含非空 start_date 的 dict，否则走损坏备份流程
            # （防 {} / [] / {"start_date": ""} 原样返回后 current_day 崩溃）
            if not (isinstance(data, dict) and data.get('start_date')):
                raise ValueError('state.json 结构非法（期望 {"start_date": "YYYY-MM-DD"}）')
            return data
        except Exception:
            # 文件损坏/为空/结构非法时备份并重建，避免服务启动失败
            try:
                backup = p.with_suffix('.json.corrupt-' + time.strftime('%Y%m%d%H%M%S'))
                p.replace(backup)
                print(f'[state] state.json 损坏，已备份为 {backup.name}', flush=True)
            except Exception:
                pass
    state = {'start_date': time.strftime('%Y-%m-%d')}  # 首次使用当天为 day 0
    save_state(path, state)
    return state


def save_state(path: str, state: dict):
    """原子写 state.json：先写临时文件再替换，避免中断留下半截 JSON。"""
    p = Path(path)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def current_day(state: dict) -> int:
    """day = 今天 - start_date，与日历同步，无需手动推进。时钟拨早时下限为 0。"""
    import datetime
    start = datetime.date.fromisoformat(state['start_date'])
    return max((datetime.date.today() - start).days, 0)


# ---------- 配置加载 ----------

# LLM profile 注册表（由 init_profiles 填充，供运行时切换）
_LLM_PROFILES: dict = {}


def init_profiles(cfg: dict):
    """从 config 注册 local/cloud profile。"""
    global _LLM_PROFILES
    _LLM_PROFILES = {}
    for k, v in cfg['llm'].items():
        if isinstance(v, dict) and 'base_url' in v:
            _LLM_PROFILES[k] = v


def _read_deepseek_key() -> str:
    """读 DeepSeek 官方 API key（前端填写、存到 deepseek_key.txt）。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepseek_key.txt')
    try:
        k = Path(path).read_text(encoding='utf-8').strip()
        return k
    except FileNotFoundError:
        return ''


def save_deepseek_key(key: str):
    """保存 DeepSeek API key 到 deepseek_key.txt（不硬编码，便于前端填写持久化）。

    原子写：先写临时文件再替换，避免断电/异常留下空文件。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepseek_key.txt')
    tmp = path + '.tmp'
    Path(tmp).write_text(key.strip(), encoding='utf-8')
    Path(tmp).replace(path)


def load_config(path: str | None = None) -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # py<3.11
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    with open(path, 'rb') as f:
        return tomllib.load(f)


# ---------- 翻译写作模块：题库模板 / 出题 / 批改 ----------

# 写作题型骨架（稳定，LLM 据此现场编题）
WRITING_SKELETONS = {
    'op_compare': '议论文·观点对比：针对一个有争议的话题给出两种对立观点，考生须选一边立论（如：AI 该不该取代老师）',
    'phenomenon': '议论文·现象分析：描述一个当下社会/校园现象，分析原因或影响，并给出个人看法',
    'problem_solve': '议论文·问题解决：提出一个具体问题，分析危害，给出可行的解决方法',
    'letter_advice': '应用文·建议信：以给某人写信的形式，就某情况提出建议（如：给大一新生建议）',
    'picture': '图画作文：给一幅漫画（用文字描述场景），描述画面→分析寓意→发表观点',
    'chart': '图表作文：给一个数据图表（用文字/数据描述），描述数据→分析趋势→发表观点',
}

# 翻译题材（稳定）+ 关键词池（LLM 据此现编中文材料）
TRANSLATION_THEMES = {
    'culture': '中国传统文化：茶文化、节庆、丝绸、剪纸、故宫、传统美德、二十四节气等',
    'society': '社会民生发展：高铁、共享经济、养老、教育、乡村振兴、城市化、人口等',
    'tech': '科技经济：人工智能、数字经济、新能源、直播带货、智慧城市、绿色低碳等',
}


# 六级评分标准（满分 15 分，五档）
_CET6_SCALE = ('满分 15 分，分档：14 分=切题、表达准确、条理清晰；'
               '11 分=切题、表达基本准确、有条理；8 分=基本切题、语言较多错误；'
               '5 分=基本切题、表达不清；2 分=跑题、语言支离破碎。')


def _skel_list() -> str:
    return '\n'.join(f'- {key}：{desc}' for key, desc in WRITING_SKELETONS.items())


def _theme_list() -> str:
    return '\n'.join(f'- {key}：{desc}' for key, desc in TRANSLATION_THEMES.items())


def build_trw_generate_prompt(kind: str, exclude: list[str] | None = None) -> str:
    """出题 prompt：写作选一类骨架现编题，翻译选一题材现编材料。
    exclude：本次会话已出过题的主题关键词，注入 prompt 让 LLM 避开（去重）。"""
    avoid = ''
    if exclude:
        avoid = (f'\n注意避开已出过的主题（不要与这些重复）：\n' +
                 '\n'.join(f'- {kw}' for kw in exclude) + '\n请出主题与以上完全不同的新题。\n')
    if kind == 'writing':
        return (f'你是大学英语六级写作出题老师。请从以下题型骨架中选一类，现场编一道六级作文题。\n'
                f'题型骨架：\n{_skel_list()}\n'
                f'要求：\n'
                f'- 题目明确可写，要求 150-180 词\n'
                f'- 话题贴近当下大学生/社会热点\n'
                f'- 若是图画/图表作文，用文字把画面或数据描述清楚（数据合理即可，不必真实）\n' + avoid +
                f'- 只输出 JSON，不要任何额外的文字：\n'
                f'{{"type":"writing","skeleton":"<题型key>","title":"<题目>",'
                f'"keyword":"<主题关键词,1-2个词>","requirements":"<写作要求,含字数>",'
                f'"image_desc":"<图或表文字描述,无则空字符串>"}}')
    return (f'你是大学英语六级翻译出题老师。请选一个题材，现场编一段汉译英材料（150-200 字中文）。\n'
            f'题材可选：\n{_theme_list()}\n'
            f'要求：\n'
            f'- 内容地道、句型多样、贴近六级翻译难度，主题明确\n'
            f'- material 必须是单行字符串，不要包含换行符\n' + avoid +
            f'- 只输出 JSON，不要任何额外文字：\n'
            f'{{"type":"translation","theme":"<题材key>","keyword":"<主题关键词,1-2个词>","material":"<中文材料>"}}')


def build_trw_grade_prompt(kind: str, task: dict, answer: str) -> str:
    """批改 prompt：写作按六级标准打分段批注给范文；翻译给参考译文逐句点评打分。"""
    if kind == 'writing':
        title = task.get('title', '')
        req = task.get('requirements', '')
        img = task.get('image_desc', '')
        return (f'你是大学英语六级作文阅卷老师。按六级评分标准给下面作文打分。\n'
                f'评分标准：{_CET6_SCALE}\n'
                f'作文题目：{title}\n'
                f'写作要求：{req or "（无）"}\n'
                f'画面/图表描述：{img or "（无）"}\n'
                f'考生作文：\n{answer}\n'
                f'评分规则：\n'
                f'- 若考生作文没有任何实质内容（例如只有标点符号、空白、几个零散单词、或明显乱写），'
                f'score 直接给 0 分，tier 写"无实质内容"，paras 返回空数组 []（严禁虚构段落或点评）\n'
                f'- 只有存在真实作文内容时，才按档位打分并给逐段批注\n'
                f'请输出：\n'
                f'- score：分数（0-15 整数）\n'
                f'- tier：档位说明（为什么这档）\n'
                f'- paras：逐段批注数组，每段 {{para, comment}}（comment 含优点/问题/改法，须针对考生实际所写）\n'
                f'- model：同题范文（150-180 词，六级水平）\n'
                f'只输出 JSON：{{"score":0,"tier":"","paras":[],"model":""}}')
    material = task.get('material', '')
    return (f'你是大学英语六级翻译阅卷老师。下面给一段汉译英题目、考生译文，逐句点评并打分。\n'
            f'评分标准：{_CET6_SCALE}\n'
            f'原文（中文）：{material}\n'
            f'考生译文：\n{answer}\n'
            f'评分规则：\n'
            f'- 若考生译文没有任何实质内容（例如只有标点符号、空白、几个零散单词、或明显乱写），'
            f'score 直接给 0 分，tier 写"无实质内容"，reviews 返回空数组 []（严禁虚构点评）\n'
            f'- 只有存在真实译文内容时，才按档位打分并逐句点评\n'
            f'请输出：\n'
            f'- reference：标准参考译文（英文）\n'
            f'- reviews：逐句点评数组，每句 {{sentence, reference, comment}}（考生句 vs 参考句，挑词汇/语法/地道度）\n'
            f'- score：分数（0-15 整数）\n'
            f'- tier：档位说明\n'
            f'只输出 JSON：{{"reference":"","reviews":[],"score":0,"tier":""}}')

def _extract_json(raw: str) -> dict:
    """容错提取首个 JSON 对象（容忍 ```json 包裹、多余文本、字符串内裸换行/括号）。"""
    text = strip_think(raw)
    # 剥掉 ```json ... ``` 包裹
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence:
        text = fence.group(1)
    start = text.find('{')
    if start < 0:
        raise ValueError(f'no JSON found: {raw[:200]!r}')
    # 用字符串感知的括号配对定位完整对象（比贪婪 \{.*\} 更能容忍字符串里的 }）
    candidate = _balanced_json(text, start) or text[start:]
    dec = json.JSONDecoder()
    # 依次尝试：原文 → 字符串内裸换行转义（LLM 长字段写成多行的常见情况）
    for attempt in (candidate, _escape_newlines_in_strings(candidate)):
        try:
            return dec.raw_decode(attempt)[0]
        except json.JSONDecodeError:
            continue
    raise ValueError(f'JSON parse failed: {candidate[:200]!r}')


def _as_str(v) -> str:
    """把 LLM 可能返回的任意值（None/数字/对象/字符串）规整为字符串，防 .strip() 抛 AttributeError。"""
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    return str(v)


def _as_list(v) -> list:
    """把 LLM 可能返回的 None/非列表规整为列表。"""
    return v if isinstance(v, list) else []


def parse_trw_generate(raw: str, kind: str) -> dict:
    """解析出题结果（LLM 返回的 JSON 题目）。"""
    obj = _extract_json(raw)
    otype = _as_str(obj.get('type')).strip()
    if otype and otype != kind:
        raise ValueError(f'LLM 返回题型 {otype!r} 与请求 {kind!r} 不符')
    if kind == 'writing':
        return {
            'type': 'writing',
            'skeleton': _as_str(obj.get('skeleton')).strip(),
            'title': _as_str(obj.get('title')).strip(),
            'keyword': _as_str(obj.get('keyword')).strip(),
            'requirements': _as_str(obj.get('requirements')).strip(),
            'image_desc': _as_str(obj.get('image_desc')).strip(),
        }
    return {
        'type': 'translation',
        'theme': _as_str(obj.get('theme')).strip(),
        'keyword': _as_str(obj.get('keyword')).strip(),
        'material': _as_str(obj.get('material')).strip(),
    }


def _safe_score(v) -> int:
    """把 LLM 可能返回的任意值规整为 0-15 整数（防字符串/越界分数污染展示）。"""
    try:
        return max(0, min(15, int(v)))
    except (TypeError, ValueError):
        return 0


def parse_trw_grade(raw: str, kind: str) -> dict:
    """解析批改结果（写作：分数/档位/逐段批注/范文；翻译：参考译文/逐句点评/分数）。"""
    obj = _extract_json(raw)
    if kind == 'writing':
        return {
            'score': _safe_score(obj.get('score', 0)),
            'tier': _as_str(obj.get('tier')).strip(),
            'paras': _as_list(obj.get('paras')),
            'model': _as_str(obj.get('model')).strip(),
        }
    return {
        'reference': _as_str(obj.get('reference')).strip(),
        'reviews': _as_list(obj.get('reviews')),
        'score': _safe_score(obj.get('score', 0)),
        'tier': _as_str(obj.get('tier')).strip(),
    }


# ---------- 翻译单句模式 ----------

def build_trw_sentence_ref_prompt(sentence: str) -> str:
    """单句参考译文 prompt：中文句子 → 地道英文，只输出译文。"""
    return (f'你是大学英语六级翻译老师。请把这句中文译成地道、符合六级水平的英文。'
            f'只输出英文译文，不要任何解释或标注。\n\n{sentence}')


def build_trw_sentence_grade_prompt(sentence: str, answer: str) -> str:
    """单句批改 prompt：给这句翻译打 0-15 分并点评，附参考译文。"""
    return (f'你是大学英语六级翻译阅卷老师。按六级评分标准给下面这句翻译打分并点评。\n'
            f'评分标准：{_CET6_SCALE}\n'
            f'中文原句：{sentence}\n'
            f'考生译文：\n{answer}\n'
            f'评分规则：\n'
            f'- 若考生译文没有任何实质内容（例如只有标点符号、空白、几个零散单词、或明显乱写），'
            f'score 直接给 0 分，tier 写"无实质内容"，comment 写"无实质内容"（严禁虚构点评）\n'
            f'请输出：\n'
            f'- reference：标准参考译文（英文）\n'
            f'- comment：一句话点评（指出主要问题+怎么改）\n'
            f'- score：分数（0-15 整数）\n'
            f'- tier：档位说明\n'
            f'只输出 JSON：{{"reference":"","comment":"","score":0,"tier":""}}')


def parse_trw_sentence_grade(raw: str) -> dict:
    """解析单句批改结果。"""
    obj = _extract_json(raw)
    return {
        'reference': _as_str(obj.get('reference')).strip(),
        'comment': _as_str(obj.get('comment')).strip(),
        'score': _safe_score(obj.get('score', 0)),
        'tier': _as_str(obj.get('tier')).strip(),
    }


# ---------- 笔记本：AI 释义 / 句式解读 ----------

def build_word_meaning_prompt(word: str) -> str:
    """未知生词的中文释义：输出 词性缩写 + 简洁释义。"""
    return (f'请给出英语单词 "{word}" 的中文释义，格式：词性缩写 + 释义（20 字以内，多个义项用分号隔开）。'
            f'只输出释义本身，不要任何解释。')


def build_pattern_explain_prompt(zh: str, en: str) -> str:
    """翻译句式解读：点出固定句式/亮点表达，用中文简述。"""
    return (f'你是一名英语老师。请用中文简短解读下面这组中英对照句中的固定句式或亮点表达'
            f'（60 字以内，指出用了什么句型/搭配、适合什么场景）。\n'
            f'中文：{zh}\n英文：{en}')
