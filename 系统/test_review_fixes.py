# test_review_fixes.py — 针对代码审查发现的工程级问题的回归测试
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as E
import wordlist as W
import server as S
from fastapi.testclient import TestClient


class WordlistRobustnessTests(unittest.TestCase):
    def test_coverage_matches_inflected_forms(self):
        """coverage 应命中 LLM 使用的复数/过去式/副词等变形，而不是只精确匹配原形。"""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '单词', 'wordlist_cet6_乱序.txt')
        wl = W.Wordlist(path)
        words = [w.lower() for w, _ in wl.coverage('The batteries consistently preserved the possessions competently.')]
        self.assertIn('battery', words)
        self.assertIn('consistent', words)
        self.assertIn('preserve', words)
        self.assertIn('possession', words)
        self.assertIn('competent', words)

    def test_empty_wordlist_does_not_divide_by_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / 'empty.txt'
            empty.write_text('', encoding='utf-8')
            wl = W.Wordlist(str(empty))
            self.assertEqual(wl.total, 0)
            self.assertEqual(wl.day_chunk(0), [])
            self.assertEqual(wl.sample(0, 5), [])
            self.assertEqual(wl.coverage('hello world'), [])


class EngineParseRobustnessTests(unittest.TestCase):
    def test_parse_trw_generate_rejects_mismatched_type(self):
        with self.assertRaises(ValueError):
            E.parse_trw_generate('{"type":"translation","material":"x"}', 'writing')

    def test_parse_trw_generate_tolerates_null_fields(self):
        r = E.parse_trw_generate('{"type":"writing","title":null,"keyword":null,"requirements":null,"image_desc":null}',
                                 'writing')
        self.assertEqual(r['title'], '')
        self.assertEqual(r['keyword'], '')
        self.assertEqual(r['requirements'], '')
        self.assertEqual(r['image_desc'], '')

    def test_parse_trw_grade_tolerates_null_fields(self):
        r = E.parse_trw_grade('{"score":10,"tier":null,"paras":null,"model":null}', 'writing')
        self.assertEqual(r['tier'], '')
        self.assertEqual(r['model'], '')
        self.assertEqual(r['paras'], [])


class StatePersistenceTests(unittest.TestCase):
    def test_load_state_recovers_from_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'state.json'
            p.write_text('not-json', encoding='utf-8')
            state = E.load_state(str(p))
            self.assertEqual(state['start_date'], time.strftime('%Y-%m-%d'))
            self.assertTrue(p.exists())
            self.assertTrue(list(Path(tmp).glob('state.json.corrupt-*')))

    def test_load_state_rejects_structurally_invalid_json(self):
        """合法 JSON 但非 {start_date: ...}（如 {} / [] / 空 start_date）应重建而非原样返回。"""
        for bad in ('{}', '[]', '{"start_date": ""}'):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'state.json'
                p.write_text(bad, encoding='utf-8')
                state = E.load_state(str(p))
                self.assertEqual(state['start_date'], time.strftime('%Y-%m-%d'))
                self.assertTrue(list(Path(tmp).glob('state.json.corrupt-*')))

    def test_save_state_leaves_no_tmp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'state.json'
            E.save_state(str(p), {'start_date': '2020-01-01'})
            self.assertEqual(E.load_state(str(p))['start_date'], '2020-01-01')
            self.assertFalse(list(Path(tmp).glob('state.json.tmp')))


class HistoryIndexStructureTests(unittest.TestCase):
    def test_history_index_rejects_non_list(self):
        old = S.cfg['paths'].pop('history_dir', None)
        with tempfile.TemporaryDirectory() as tmp:
            S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
            try:
                hd = S._history_dir()
                (hd / 'history.json').write_text('{}', encoding='utf-8')
                self.assertEqual(S._history_index(), [])
                self.assertTrue(list(hd.glob('history.json.corrupt-*')))
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old


class ModelValidationTests(unittest.TestCase):
    def test_model_switch_rejects_unknown_cloud_model(self):
        r = TestClient(S.app).post('/api/model', json={'profile': 'cloud', 'model': 'not-a-real-model'})
        self.assertEqual(r.status_code, 400)


class SpeakingAudioFallbackTests(unittest.TestCase):
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

    def test_ref_whole_returns_400(self):
        hid = S._archive('speaking', {'type': 'speaking', 'text': 'Hello world.', 'title': 'Hello world.'})
        r = TestClient(S.app).get('/api/speaking/audio', params={'id': hid, 'kind': 'ref'})
        self.assertEqual(r.status_code, 400)

    def test_ref_falls_back_to_wav_when_mp3_missing(self):
        hid = S._archive('speaking', {'type': 'speaking', 'text': 'Hello world.', 'title': 'Hello world.'})
        entry = Path(S._history_dir()) / 'speaking' / hid
        (entry / 'ref_0.wav').write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
        r = TestClient(S.app).get('/api/speaking/audio', params={'id': hid, 'kind': 'ref', 'i': 0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers['content-type'].startswith('audio/wav'))


class ConfigValidationTests(unittest.TestCase):
    def test_validate_rejects_missing_section(self):
        bad = dict(S.cfg)  # 拷贝一份
        bad.pop('daily', None)
        with self.assertRaises(SystemExit):
            S._validate_config(bad)

    def test_validate_rejects_wrong_type(self):
        bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in S.cfg.items()}
        bad['daily'] = dict(bad['daily']); bad['daily']['sample_n'] = 'not-int'
        with self.assertRaises(SystemExit):
            S._validate_config(bad)

    def test_validate_rejects_missing_llm_subkey(self):
        bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in S.cfg.items()}
        bad['llm'] = dict(bad['llm']); bad['llm']['cloud'] = dict(bad['llm']['cloud'])
        bad['llm']['cloud'].pop('base_url', None)
        with self.assertRaises(SystemExit):
            S._validate_config(bad)

    def test_validate_rejects_non_str_access_token(self):
        bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in S.cfg.items()}
        bad['server'] = dict(bad['server']); bad['server']['access_token'] = 123
        with self.assertRaises(SystemExit):
            S._validate_config(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
