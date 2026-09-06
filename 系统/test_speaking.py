# test_speaking.py — 口语练习：出题 prompt / 音频预处理 / 评分归一化 / API / 历史
import base64
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as E
import pronounce as P
import server as S
from fastapi.testclient import TestClient
from unittest.mock import patch


class FakeTTS:
    def synth(self, text, voice=None):
        import numpy as np
        return np.zeros(24000, dtype=np.float32)  # 1 秒静音

    def synth_line(self, text, speaker):
        return self.synth(text)

    def synth_timed(self, text, voice=None, speed=None):
        import re
        audio = self.synth(text)
        words = [(w, float(i), float(i) + 1.0) for i, w in enumerate(re.findall(r"[\w']+", text))]
        return audio, words

    def synth_line_timed(self, text, speaker):
        return self.synth_timed(text)


class FakeScorer:
    def score(self, wav_path, reference_text):
        words = [{'word': w, 'ok': True, 'confidence': None}
                 for w in re.findall(r"[\w']+", reference_text)]
        return {
            'overall': 88.0, 'band': '优秀', 'transcribe': reference_text.upper(),
            'word_error_rate': 0.0, 'phoneme_error_rate': 0.0,
            'acoustic_distance': 6.5, 'words': words, 'errors': [],
            'hint': '发音标准，继续保持！',
        }


class PromptTests(unittest.TestCase):
    def test_build_speaking_prompt_contains_words_and_length(self):
        prompt = E.build_speaking_prompt(['battery', 'competent'], 100)
        self.assertIn('battery', prompt)
        self.assertIn('competent', prompt)
        self.assertIn('100', prompt)
        self.assertIn('reading aloud', prompt)
        self.assertIn('plain text', prompt)


class NormalizeTests(unittest.TestCase):
    def test_band(self):
        self.assertEqual(P._band(90), '优秀')
        self.assertEqual(P._band(72), '良好')
        self.assertEqual(P._band(60), '及格')
        self.assertEqual(P._band(45), '较差')
        self.assertEqual(P._band(20), '需重练')

    def test_hint_empty(self):
        self.assertIn('继续', P._hint([]))

    def test_hint_picks_phones(self):
        hint = P._hint([{'word': 'battery', 'phones': [
            {'expected': 'æ', 'heard': 'eɪ'},
            {'expected': 't', 'heard': 't'},
        ]}])
        self.assertIn('æ', hint)
        self.assertIn('battery', hint)

    def test_normalize_result_schema(self):
        res = {
            'score': 78.0, 'transcribe': 'HELLO WORLD',
            'differences': {
                'word_error_rate': 0.1, 'phoneme_error_rate': 0.2,
                'errors': [
                    {'word': 'hello', 'expected': 'həloʊ', 'actual': 'hɛloʊ',
                     'confidence': 0.9, 'phones': [{'expected': 'ə', 'heard': 'ɛ', 'confidence': 0.9}]},
                ],
            },
        }
        out = P.normalize_result(res, 'Hello world.')
        self.assertEqual(out['overall'], 78.0)
        self.assertEqual(out['band'], '良好')
        self.assertEqual(out['transcribe'], 'HELLO WORLD')
        self.assertEqual(out['words'][0]['word'], 'Hello')
        self.assertFalse(out['words'][0]['ok'])
        self.assertEqual(out['errors'][0]['word'], 'hello')


class AudioPrepTests(unittest.TestCase):
    def test_prepare_audio_converts_wav_to_16k_mono(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'in.wav'
            sf.write(src, np.zeros(8000, dtype=np.float32), 8000)  # 1s @8k 单声道
            wav = P.prepare_audio(src.read_bytes(), 'audio/wav')
            try:
                data, sr = sf.read(wav)
                self.assertEqual(sr, 16000)
                self.assertEqual(data.ndim, 1)
            finally:
                try:
                    os.remove(wav)
                except OSError:
                    pass


class SpeakingApiTests(unittest.TestCase):
    def setUp(self):
        self._old_hist = S.cfg['paths'].pop('history_dir', None)
        self._tmp = tempfile.TemporaryDirectory()
        S.cfg['paths']['history_dir'] = str(Path(self._tmp.name) / '历史记录')
        Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._old_hist is None:
            S.cfg['paths'].pop('history_dir', None)
        else:
            S.cfg['paths']['history_dir'] = self._old_hist
        self._tmp.cleanup()

    def test_generate_archives_and_returns_id(self):
        fake_text = 'A healthy routine builds a steady rhythm. It helps every student stay focused.'
        with patch.object(S.client, 'chat', return_value=fake_text), \
             patch.object(S, 'get_tts', return_value=FakeTTS()), \
             patch.object(S, 'to_mp3', return_value=True):
            r = TestClient(S.app).post('/api/speaking/generate')
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertIn('id', d)
        self.assertEqual(d['text'], fake_text)
        self.assertEqual(len(d['sentences']), 2)
        for s in d['sentences']:
            self.assertIn('index', s)
            self.assertIn('text', s)
            self.assertIn('audio_url', s)
            self.assertIn('word_timeline', s)  # 词级时间轴，供范本逐词精确高亮
            self.assertGreater(len(s['word_timeline']), 0)
        items = S._history_list('speaking')
        self.assertEqual(len(items), 1)
        content = S._history_get(items[0]['id'])
        self.assertIn('word_timeline', content['sentences'][0])

    def test_assess_missing_exercise_404(self):
        r = TestClient(S.app).post('/api/speaking/assess',
                                   json={'exercise_id': 'nope', 'audio': base64.b64encode(b'x' * 5000).decode()})
        self.assertEqual(r.status_code, 404)

    def test_assess_short_audio_400(self):
        hid = S._archive('speaking', {'type': 'speaking', 'text': 'Hello.', 'title': 'Hello.'})
        r = TestClient(S.app).post('/api/speaking/assess',
                                   json={'exercise_id': hid, 'audio': base64.b64encode(b'short').decode()})
        self.assertEqual(r.status_code, 400)

    def test_assess_success_updates_archive(self):
        hid = S._archive('speaking', {'type': 'speaking', 'text': 'Hello world.',
                                      'target_words': ['hello'], 'title': 'Hello world.'})
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / 'rec.16k.wav'
            sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
            with patch.object(S.P, 'prepare_audio', return_value=str(wav)), \
                 patch.object(S.P, 'get_scorer', return_value=FakeScorer()), \
                 patch.object(S, 'to_mp3', return_value=True):
                r = TestClient(S.app).post('/api/speaking/assess',
                                           json={'exercise_id': hid,
                                                 'audio': base64.b64encode(b'x' * 5000).decode()})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d['overall'], 88.0)
        content = S._history_get(hid)
        self.assertEqual(content['status'], 'assessed')
        self.assertEqual(content['score']['overall'], 88.0)

    def test_assess_sentence_updates_sentence_scores(self):
        hid = S._archive('speaking', {'type': 'speaking', 'text': 'Hello world. Second sentence.',
                                      'target_words': ['hello'], 'title': 'Hello world.',
                                      'sentences': [{'index': 0, 'text': 'Hello world.'},
                                                    {'index': 1, 'text': 'Second sentence.'}]})
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / 'rec.16k.wav'
            sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
            with patch.object(S.P, 'prepare_audio', return_value=str(wav)), \
                 patch.object(S.P, 'get_scorer', return_value=FakeScorer()), \
                 patch.object(S, 'to_mp3', return_value=True):
                r = TestClient(S.app).post('/api/speaking/assess',
                                           json={'exercise_id': hid,
                                                 'audio': base64.b64encode(b'x' * 5000).decode(),
                                                 'text': 'Hello world.', 'sentence_index': 0})
        self.assertEqual(r.status_code, 200, r.text)
        content = S._history_get(hid)
        self.assertIn('0', content['sentence_scores'])
        self.assertEqual(content['sentence_scores']['0']['overall'], 88.0)

    def test_history_allows_speaking(self):
        r = TestClient(S.app).get('/api/history', params={'type': 'speaking'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['items'], [])

    def test_whole_and_sentence_score_merge(self):
        """整篇评分 + 单句评分混合使用，两者都不应被覆盖丢失。"""
        hid = S._archive('speaking', {'type': 'speaking', 'text': 'Hello world. Second sentence.',
                                      'target_words': ['hello'], 'title': 'Hello world.',
                                      'sentences': [{'index': 0, 'text': 'Hello world.'},
                                                    {'index': 1, 'text': 'Second sentence.'}]})
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / 'rec.16k.wav'
            sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
            with patch.object(S.P, 'prepare_audio', return_value=str(wav)), \
                 patch.object(S.P, 'get_scorer', return_value=FakeScorer()), \
                 patch.object(S, 'to_mp3', return_value=True):
                # 整篇评分（无 sentence_index）
                TestClient(S.app).post('/api/speaking/assess',
                                       json={'exercise_id': hid,
                                             'audio': base64.b64encode(b'x' * 5000).decode()})
                # 单句评分
                TestClient(S.app).post('/api/speaking/assess',
                                       json={'exercise_id': hid,
                                             'audio': base64.b64encode(b'x' * 5000).decode(),
                                             'text': 'Hello world.', 'sentence_index': 0})
        content = S._history_get(hid)
        self.assertIn('score', content)               # 整篇评分保留
        self.assertIn('sentence_scores', content)     # 单句评分保留
        self.assertEqual(content['score']['overall'], 88.0)
        self.assertEqual(content['sentence_scores']['0']['overall'], 88.0)


class WordTtsTests(unittest.TestCase):
    def tearDown(self):
        S._word_tts_cache.clear()

    def test_tts_word_returns_audio(self):
        with patch.object(S, 'get_tts', return_value=FakeTTS()):
            r = TestClient(S.app).get('/api/tts/word', params={'text': 'hello'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers['content-type'].startswith('audio/wav'))
        self.assertGreater(len(r.content), 40)

    def test_tts_word_cache(self):
        tts = FakeTTS()
        with patch.object(S, 'get_tts', return_value=tts), \
             patch.object(tts, 'synth', wraps=tts.synth) as spied:
            TestClient(S.app).get('/api/tts/word', params={'text': 'hello'})
            TestClient(S.app).get('/api/tts/word', params={'text': 'hello'})
        self.assertEqual(spied.call_count, 1)

    def test_tts_word_invalid(self):
        r = TestClient(S.app).get('/api/tts/word', params={'text': '你好'})
        self.assertEqual(r.status_code, 400)

    def test_daily_returns_timeline_matching_units(self):
        old_daily = S.cfg['paths']['daily_dir']
        old_hist = S.cfg['paths'].pop('history_dir', None)
        tmp = tempfile.TemporaryDirectory()
        S.cfg['paths']['daily_dir'] = tmp.name
        S.cfg['paths']['history_dir'] = str(Path(tmp.name) / '历史记录')
        try:
            fake_text = 'A healthy routine builds a steady rhythm. It helps every student stay focused.'
            # day=1 → 短文（独白），避免对话题型校验 A:/B: 行
            with patch.object(S.client, 'chat', return_value=fake_text), \
                 patch.object(S, 'current_day', return_value=1), \
                 patch.object(S, 'get_tts', return_value=FakeTTS()), \
                 patch.object(S, 'to_mp3', return_value=True):
                r = TestClient(S.app).post('/api/daily')
            self.assertEqual(r.status_code, 200, r.text)
            d = r.json()
            self.assertIn('timeline', d)
            self.assertEqual(len(d['timeline']), len(d['units']))
            self.assertEqual(d['timeline'][0]['start'], 0.0)
            # 归档里也存了时间轴与时长，供历史恢复时同步播放+高亮
            items = S._history_list('daily')
            self.assertEqual(len(items), 1)
            content = S._history_get(items[0]['id'])
            self.assertIn('timeline', content)
            self.assertIn('duration_s', content)
            self.assertIn('word_timeline', content)
            self.assertEqual(len(content['timeline']), len(d['units']))
            self.assertEqual(len(content['word_timeline']), len(d['units']))
            self.assertEqual(content['date'], d['date'])
        finally:
            S.cfg['paths']['daily_dir'] = old_daily
            if old_hist is None:
                S.cfg['paths'].pop('history_dir', None)
            else:
                S.cfg['paths']['history_dir'] = old_hist
            tmp.cleanup()


class FrontendContractTests(unittest.TestCase):
    def test_frontend_contains_speaking_controls(self):
        html = (Path(__file__).parent / '前端.html').read_text(encoding='utf-8')
        self.assertIn('p-speaking', html)
        self.assertIn('sp-record', html)
        self.assertIn('MediaRecorder', html)
        self.assertIn('sp-word', html)
        self.assertIn('播放范本', html)
        self.assertIn('回放', html)
        self.assertIn('sp-replay', html)
        self.assertIn('sp-speed', html)
        self.assertIn('/api/tts/word', html)
        self.assertIn('timeline', html)
        self.assertIn('tr-mode-whole', html)
        self.assertIn('tr-mode-sentence', html)
        self.assertIn('tr-s-ref', html)
        self.assertIn('tr-s-grade', html)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self._old_path = S._nb_path
        self._old_nb = S.notebook
        self._old_mc = S._word_meaning_cache
        self._tmp = tempfile.TemporaryDirectory()
        S._nb_path = str(Path(self._tmp.name) / '笔记本.json')
        S.notebook = {'words': [], 'patterns': [], 'writing': []}
        S._word_meaning_cache = {}

    def tearDown(self):
        S._nb_path = self._old_path
        S.notebook = self._old_nb
        S._word_meaning_cache = self._old_mc
        self._tmp.cleanup()

    def test_word_add_uses_wordlist_meaning(self):
        r = TestClient(S.app).post('/api/notebook/word', json={'word': 'battery', 'context': 'The battery died.'})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertFalse(d['duplicate'])
        self.assertIn('电池', d['entry']['meaning'])
        self.assertEqual(d['entry']['context'], 'The battery died.')
        self.assertEqual(len(S.notebook['words']), 1)

    def test_word_duplicate_case_insensitive(self):
        TestClient(S.app).post('/api/notebook/word', json={'word': 'battery'})
        r = TestClient(S.app).post('/api/notebook/word', json={'word': 'Battery'})
        self.assertTrue(r.json()['duplicate'])
        self.assertEqual(len(S.notebook['words']), 1)

    def test_word_unknown_uses_ai(self):
        with patch.object(S.client, 'chat', return_value='adj. 神秘的'):
            r = TestClient(S.app).post('/api/notebook/word', json={'word': 'zzzunknownzzz'})
        self.assertEqual(r.status_code, 200)
        self.assertIn('神秘', r.json()['entry']['meaning'])

    def test_word_translate_uses_wordlist(self):
        r = TestClient(S.app).post('/api/word/translate', json={'word': 'battery'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('电池', r.json()['meaning'])

    def test_word_translate_unknown_uses_ai_and_caches(self):
        with patch.object(S.client, 'chat', return_value='v. 使困惑') as spied:
            r1 = TestClient(S.app).post('/api/word/translate', json={'word': 'zzzbafflexxx'})
            r2 = TestClient(S.app).post('/api/word/translate', json={'word': 'zzzbafflexxx'})
        self.assertEqual(r1.status_code, 200)
        self.assertIn('困惑', r1.json()['meaning'])
        self.assertIn('困惑', r2.json()['meaning'])
        self.assertEqual(spied.call_count, 1)  # 第二次命中缓存

    def test_pattern_add_with_explain(self):
        with patch.object(S.client, 'chat', return_value='用到了 boast 的固定搭配'):
            r = TestClient(S.app).post('/api/notebook/pattern',
                                       json={'zh': '茶文化历史悠久', 'en': 'Tea culture boasts a long history.',
                                             'want_explain': True})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('boast', r.json()['entry']['explain'])

    def test_writing_add_update_delete(self):
        r = TestClient(S.app).post('/api/notebook/writing', json={'text': 'First paragraph.'})
        eid = r.json()['entry']['id']
        r2 = TestClient(S.app).post('/api/notebook/update',
                                    json={'category': 'writing', 'id': eid, 'entry': {'text': 'Edited text.'}})
        self.assertEqual(r2.json()['entry']['text'], 'Edited text.')
        r3 = TestClient(S.app).delete('/api/notebook', params={'category': 'writing', 'id': eid})
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(len(S.notebook['writing']), 0)

    def test_notebook_get(self):
        TestClient(S.app).post('/api/notebook/word', json={'word': 'battery'})
        r = TestClient(S.app).get('/api/notebook')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()['words']), 1)

    def test_frontend_contains_notebook(self):
        html = (Path(__file__).parent / '前端.html').read_text(encoding='utf-8')
        self.assertIn('p-notebook', html)
        self.assertIn('nb-cat', html)
        self.assertIn('nb-review', html)
        self.assertIn('nb-export', html)
        self.assertIn('/api/notebook/word', html)
        self.assertIn('word-bubble', html)
        self.assertIn('st-word', html)


if __name__ == '__main__':
    unittest.main(verbosity=2)
