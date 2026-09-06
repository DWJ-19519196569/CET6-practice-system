# 新功能与回归测试：模型切换 / 翻译 / 音频 / SEGMENT 边界
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as E


class SegmentBoundaryTests(unittest.TestCase):
    def test_closing_tag_never_leaks_into_sentence(self):
        raw = ("<SEGMENT>The tapping came from outside, forty meters above the ground.</SEGMENT>\n"
               "A) Open the window.\nB) Call for help.\nC) Hide downstairs.")
        p = E.SegmentStreamParser()
        emitted = []
        for i in range(0, len(raw), 5):
            emitted.extend(p.feed(raw[i:i + 5]))
        result = p.finish()
        self.assertEqual(result['segment'],
                         'The tapping came from outside, forty meters above the ground.')
        leaked = [s for s in emitted + p.sents if 'SEGMENT' in s or s.startswith('A)')]
        self.assertEqual(leaked, [], leaked)

    def test_closing_tag_split_across_chunks_never_leaks(self):
        """真实场景：</SEGMENT 的 '<' 先于 '>' 到达（边界拆分），不得泄漏成句子。"""
        body = ("The lighthouse beam cut through the fog every twelve seconds, but tonight "
                "it illuminated something that shouldn't exist. I checked the chart twice. "
                "Nothing. My hands trembled on the brass railing as the twin beam swept past "
                "again, and this time, I saw a figure standing in its gallery.")
        raw = f'<SEGMENT>{body}</SEGMENT>\nA) aaa.\nB) bbb.\nC) ccc.'
        # 用能让 < 单独进入 pending 的 chunk 尺寸与 1 字符极限两个方向都测
        for chunk in (1, 7, 40):
            p = E.SegmentStreamParser()
            emitted = []
            for i in range(0, len(raw), chunk):
                emitted.extend(p.feed(raw[i:i + chunk]))
            result = p.finish()
            self.assertEqual(result['segment'], body, f'chunk={chunk} segment 不对')
            leaked = [s for s in emitted + p.sents if 'SEGMENT' in s]
            self.assertEqual(leaked, [], f'chunk={chunk} 泄漏: {leaked}')


class ModelSwitchTests(unittest.TestCase):
    def test_cloud_model_can_be_selected_at_runtime(self):
        cfg = E.load_config()
        E.init_profiles(cfg)
        client = E.LLMClient(cfg)
        client.switch('cloud', 'glm-5.3')
        self.assertEqual(client.active, 'cloud')
        self.assertEqual(client.p['model'], 'glm-5.3')


class DisplayUnitTests(unittest.TestCase):
    def test_dialogue_is_split_into_sentence_units_with_speakers(self):
        units = E.build_display_units('A: First sentence. Second question?\nB: Fine.', 'conversation')
        self.assertEqual(units, [
            {'speaker': 'A', 'text': 'First sentence.'},
            {'speaker': 'A', 'text': 'Second question?'},
            {'speaker': 'B', 'text': 'Fine.'},
        ])

    def test_translation_prompt_requests_chinese_only(self):
        prompt = E.build_translation_prompt('The storm was gathering.')
        self.assertIn('简体中文', prompt)
        self.assertIn('The storm was gathering.', prompt)

    def test_interactive_vocab_note_contains_words(self):
        note = E.build_interactive_vocab_note(['abandon', 'battery', 'competent'])
        self.assertIn('abandon', note)
        self.assertIn('battery', note)
        self.assertIn('competent', note)
        self.assertIn('Weave', note)

    def test_story_vocab_always_on(self):
        import server as S
        note = S._story_vocab()  # 织词恒开（无开关），直接返回非空备注
        self.assertTrue(note)
        self.assertIn('Weave', note)

    def test_trw_exclude_reads_from_history(self):
        import tempfile
        from pathlib import Path
        import server as S
        old = S.cfg['paths'].pop('history_dir', None)
        with tempfile.TemporaryDirectory() as tmp:
            S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
            Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
            try:
                S._archive_trw({'type': 'writing', 'task': {'title': 'AI', 'keyword': 'ai'}, 'status': 'generated'})
                S._archive_trw({'type': 'writing', 'task': {'title': 'Food', 'keyword': 'food'}, 'status': 'generated'})
                excl = S._trw_exclude('writing')
                self.assertIn('ai', excl)
                self.assertIn('food', excl)
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_trw_generate_archives_into_history(self):
        import tempfile
        from unittest.mock import patch
        from pathlib import Path
        from fastapi.testclient import TestClient
        import server as S
        old = S.cfg['paths'].pop('history_dir', None)
        with tempfile.TemporaryDirectory() as tmp:
            S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
            Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
            fake = '{"type":"writing","skeleton":"op_compare","title":"New","keyword":"newkw","requirements":"150 words","image_desc":""}'
            try:
                with patch.object(S.client, 'chat', return_value=fake):
                    TestClient(S.app).post('/api/trw/generate', json={'type': 'writing'})
                items = S._history_list('writing')
                self.assertEqual(len(items), 1)
                content = S._history_get(items[0]['id'])
                self.assertEqual(content['task']['keyword'], 'newkw')
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old


class AudioRouteTests(unittest.TestCase):
    def test_daily_audio_accepts_frontend_date_parameter(self):
        from fastapi.testclient import TestClient
        import server

        old_dir = server.cfg['paths']['daily_dir']
        with tempfile.TemporaryDirectory() as tmp:
            server.cfg['paths']['daily_dir'] = tmp
            folder = Path(tmp) / date.today().isoformat()
            folder.mkdir(parents=True)
            # FileResponse 只需存在；媒体解码另由真实 WAV 集成测试覆盖
            (folder / 'story.wav').write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
            try:
                response = TestClient(server.app).get(
                    '/api/daily/audio', params={'date': date.today().isoformat()})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.headers['content-type'].startswith('audio/wav'))
            finally:
                server.cfg['paths']['daily_dir'] = old_dir


class FrontendContractTests(unittest.TestCase):
    def test_frontend_contains_translation_and_cloud_model_controls(self):
        html = (Path(__file__).parent / '前端.html').read_text(encoding='utf-8')
        self.assertIn('translate-btn', html)
        self.assertIn('cloud-model-select', html)
        self.assertIn('audio-error', html)


class CloudDeepSeekTests(unittest.TestCase):
    def test_cloud_uses_official_deepseek_base(self):
        cfg = E.load_config()
        self.assertEqual(cfg['llm']['cloud']['base_url'], 'https://api.deepseek.com')

    def test_cloud_models_are_official(self):
        import server as S
        from fastapi.testclient import TestClient
        r = TestClient(S.app).get('/api/models')
        self.assertEqual(r.status_code, 200)
        models = r.json().get('cloud_models', [])
        self.assertIn('deepseek-v4-pro', models)
        self.assertIn('deepseek-v4-flash', models)

    def test_set_apikey_updates_client(self):
        import server as S
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        old_key = S.client.p.get('api_key')
        old_active = S.client.active
        try:
            S.client.switch('cloud')
            # 不真正写生产 deepseek_key.txt，只验证 client 更新
            with patch.object(S.E, 'save_deepseek_key') as mock_save:
                r = TestClient(S.app).post('/api/model/apikey', json={'key': 'sk-test-dwj'})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(S.client.p.get('api_key'), 'sk-test-dwj')
                mock_save.assert_called_once()
        finally:
            S.client.active = old_active
            S.client.p['api_key'] = old_key


if __name__ == '__main__':
    unittest.main(verbosity=2)
