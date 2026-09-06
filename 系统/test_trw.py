# test_trw.py — 翻译写作模块：题库模板 / 出题 prompt / 批改 prompt / JSON 解析
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as E


class PromptBuildTests(unittest.TestCase):
    def test_writing_generate_prompt_lists_skeletons(self):
        prompt = E.build_trw_generate_prompt('writing')
        self.assertIn('议论文', prompt)
        self.assertIn('应用文', prompt)
        self.assertIn('图画', prompt)
        self.assertIn('图表', prompt)
        self.assertIn('JSON', prompt)

    def test_translation_generate_prompt_lists_themes(self):
        prompt = E.build_trw_generate_prompt('translation')
        self.assertIn('中国传统文化', prompt)
        self.assertIn('社会民生', prompt)
        self.assertIn('科技经济', prompt)
        self.assertIn('JSON', prompt)

    def test_generate_prompt_with_exclude_avoids_repeat(self):
        prompt = E.build_trw_generate_prompt('writing', ['digital detox', 'AI education'])
        self.assertIn('digital detox', prompt)
        self.assertIn('AI education', prompt)
        self.assertIn('不要与这些重复', prompt)

    def test_translation_generate_prompt_with_exclude(self):
        prompt = E.build_trw_generate_prompt('translation', ['茶文化'])
        self.assertIn('茶文化', prompt)
        self.assertIn('不要与这些重复', prompt)


class GenerateParseTests(unittest.TestCase):
    def test_parse_writing_generate(self):
        raw = ('{"type":"writing","skeleton":"op_compare","title":"Should AI replace teachers?",'
               '"keyword":"AI education","requirements":"Write 150-180 words.","image_desc":""}')
        r = E.parse_trw_generate(raw, 'writing')
        self.assertEqual(r['type'], 'writing')
        self.assertEqual(r['skeleton'], 'op_compare')
        self.assertEqual(r['keyword'], 'AI education')
        self.assertIn('AI', r['title'])

    def test_parse_translation_generate(self):
        raw = ('{"type":"translation","theme":"culture","keyword":"茶文化",'
               '"material":"中国茶文化历史悠久，从唐代起便传遍世界。"}')
        r = E.parse_trw_generate(raw, 'translation')
        self.assertEqual(r['type'], 'translation')
        self.assertEqual(r['theme'], 'culture')
        self.assertEqual(r['keyword'], '茶文化')
        self.assertIn('茶文化', r['material'])

    def test_parse_fences_json(self):
        raw = '```json\n{"type":"writing","skeleton":"picture","title":"A picture","requirements":"150 words","image_desc":"一幅漫画：年轻人看手机，老人叹气"}\n```'
        r = E.parse_trw_generate(raw, 'writing')
        self.assertEqual(r['skeleton'], 'picture')

    def test_parse_generate_with_raw_newline_in_string(self):
        # LLM 把 requirements 写成多行（字符串内裸换行）不应崩，且换行被转义
        raw = ('{\n"type":"writing",\n"skeleton":"op_compare",\n'
               '"title":"Should Universities Ban Smartphones?",\n'
               '"keyword":"smartphones, education",\n'
               '"requirements":"Directions: For this part, you are allowed 30 minutes.\n'
               'You should write at least 150 words.",\n"image_desc":""\n}')
        r = E.parse_trw_generate(raw, 'writing')
        self.assertEqual(r['title'], 'Should Universities Ban Smartphones?')
        self.assertIn('150 words', r['requirements'])

    def test_parse_generate_with_brace_inside_string(self):
        raw = '{"type":"writing","title":"Use {brackets} in title","requirements":"150 words","image_desc":""}'
        r = E.parse_trw_generate(raw, 'writing')
        self.assertEqual(r['title'], 'Use {brackets} in title')


class GradeParseTests(unittest.TestCase):
    def test_parse_writing_grade(self):
        raw = ('{"score":12,"tier":"第二档（11-14）","paras":'
               '[{"para":"第一段","comment":"切题，语言可更丰富"},'
               '{"para":"第二段","comment":"论证充分"}],'
               '"model":"As the world hurtles into the AI era..."}')
        r = E.parse_trw_grade(raw, 'writing')
        self.assertEqual(r['score'], 12)
        self.assertEqual(len(r['paras']), 2)
        self.assertTrue(r['model'].startswith('As the world'))

    def test_parse_translation_grade(self):
        raw = ('{"reference":"The tea culture of China has a long history...",'
               '"reviews":[{"sentence":"China tea culture long history",'
               '"reference":"Chinese tea culture boasts a long history",'
               '"comment":"缺动词，建议用 boasts"}],'
               '"score":10,"tier":"第二档（11-14）以下"}')
        r = E.parse_trw_grade(raw, 'translation')
        self.assertEqual(r['score'], 10)
        self.assertEqual(len(r['reviews']), 1)
        self.assertIn('boasts', r['reviews'][0]['reference'])


class GradePromptTests(unittest.TestCase):
    def test_writing_grade_prompt_has_answer_and_standard(self):
        prompt = E.build_trw_grade_prompt('writing', {'title': 'Should AI replace teachers?'},
                                          'My essay is about ...')
        self.assertIn('My essay is about', prompt)
        self.assertIn('15', prompt)
        self.assertIn('范文', prompt)

    def test_translation_grade_prompt_has_material_and_answer(self):
        prompt = E.build_trw_grade_prompt('translation', {'material': '中国茶文化历史悠久'},
                                          'China tea culture long history')
        self.assertIn('中国茶文化历史悠久', prompt)
        self.assertIn('China tea culture', prompt)


class ApiTests(unittest.TestCase):
    def setUp(self):
        # 这些 API 测试会走真实归档路径，必须重定向 history_dir，避免污染用户真实历史
        import tempfile
        from pathlib import Path
        import server as S
        self._S = S
        self._old_hist = S.cfg['paths'].pop('history_dir', None)
        self._tmp = tempfile.TemporaryDirectory()
        S.cfg['paths']['history_dir'] = str(Path(self._tmp.name) / '历史记录')
        Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._old_hist is None:
            self._S.cfg['paths'].pop('history_dir', None)
        else:
            self._S.cfg['paths']['history_dir'] = self._old_hist
        self._tmp.cleanup()

    def test_generate_writing(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        fake = '{"type":"writing","skeleton":"op_compare","title":"Should AI replace teachers?","requirements":"150-180 words","image_desc":""}'
        with patch.object(S.client, 'chat', return_value=fake):
            r = TestClient(S.app).post('/api/trw/generate', json={'type': 'writing'})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d['skeleton'], 'op_compare')

    def test_generate_translation(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        fake = '{"type":"translation","theme":"culture","material":"中国茶文化历史悠久。"}'
        with patch.object(S.client, 'chat', return_value=fake):
            r = TestClient(S.app).post('/api/trw/generate', json={'type': 'translation'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['theme'], 'culture')

    def test_grade_writing(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        fake = ('{"score":12,"tier":"第二档","paras":[{"para":"第一段","comment":"切题，语言可丰富"}],'
                '"model":"As the world hurtles..."}')
        task = {'title': 'Should AI replace teachers?', 'requirements': '150-180 words', 'image_desc': ''}
        with patch.object(S.client, 'chat', return_value=fake):
            r = TestClient(S.app).post('/api/trw/grade',
                                       json={'type': 'writing', 'task': task, 'answer': 'My essay...'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['score'], 12)

    def test_grade_translation(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        fake = '{"reference":"Chinese tea culture has a long history","reviews":[{"sentence":"s","reference":"r","comment":"c"}],"score":10,"tier":"第二档"}'
        task = {'material': '中国茶文化历史悠久'}
        with patch.object(S.client, 'chat', return_value=fake):
            r = TestClient(S.app).post('/api/trw/grade',
                                       json={'type': 'translation', 'task': task, 'answer': 'China tea long history'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['score'], 10)


class SentenceModeTests(unittest.TestCase):
    def test_sentence_ref_prompt(self):
        prompt = E.build_trw_sentence_ref_prompt('中国茶文化历史悠久')
        self.assertIn('中国茶文化历史悠久', prompt)
        self.assertIn('英文', prompt)

    def test_sentence_grade_prompt(self):
        prompt = E.build_trw_sentence_grade_prompt('中国茶文化历史悠久', 'China tea long history')
        self.assertIn('中国茶文化历史悠久', prompt)
        self.assertIn('China tea long history', prompt)
        self.assertIn('15', prompt)
        self.assertIn('无实质内容', prompt)  # 空内容防御规则

    def test_grade_prompts_contain_empty_guard(self):
        wp = E.build_trw_grade_prompt('writing', {'title': 'T'}, '，')
        self.assertIn('无实质内容', wp)
        self.assertIn('严禁虚构', wp)
        tp = E.build_trw_grade_prompt('translation', {'material': '中国茶文化历史悠久'}, '，')
        self.assertIn('无实质内容', tp)
        self.assertIn('严禁虚构', tp)

    def test_parse_trw_sentence_grade(self):
        raw = '{"reference":"Chinese tea culture boasts a long history.","comment":"缺动词，建议用 boasts","score":11,"tier":"第二档"}'
        r = E.parse_trw_sentence_grade(raw)
        self.assertEqual(r['score'], 11)
        self.assertIn('boasts', r['reference'])
        self.assertIn('boasts', r['comment'])

    def test_sentence_ref_api(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        with patch.object(S.client, 'chat', return_value='Chinese tea culture has a long history.'):
            r = TestClient(S.app).post('/api/trw/sentence/ref', json={'sentence': '中国茶文化历史悠久'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('tea culture', r.json()['reference'])

    def test_sentence_grade_api_archives(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        old = S.cfg['paths'].pop('history_dir', None)
        with tempfile.TemporaryDirectory() as tmp:
            S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
            Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
            try:
                fake = '{"type":"translation","theme":"culture","material":"中国茶文化历史悠久。"}'
                with patch.object(S.client, 'chat', return_value=fake):
                    g = TestClient(S.app).post('/api/trw/generate', json={'type': 'translation'})
                hid = g.json()['_history_id']
                fake_grade = '{"reference":"Chinese tea culture boasts a long history.","comment":"不错","score":12,"tier":"第二档"}'
                with patch.object(S.client, 'chat', return_value=fake_grade):
                    r = TestClient(S.app).post('/api/trw/sentence/grade',
                                               json={'exercise_id': hid, 'sentence': '中国茶文化历史悠久',
                                                     'answer': 'China tea culture long history', 'index': 0})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()['score'], 12)
                content = S._history_get(hid)
                self.assertIn('0', content['sentence_grades'])
                self.assertEqual(content['sentence_grades']['0']['grade']['score'], 12)
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_whole_and_sentence_grade_merge(self):
        """整段批改 + 单句批改混合使用，两者都不应被覆盖丢失。"""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        old = S.cfg['paths'].pop('history_dir', None)
        with tempfile.TemporaryDirectory() as tmp:
            S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
            Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
            try:
                fake = '{"type":"translation","theme":"culture","material":"中国茶文化历史悠久。"}'
                with patch.object(S.client, 'chat', return_value=fake):
                    g = TestClient(S.app).post('/api/trw/generate', json={'type': 'translation'})
                hid = g.json()['_history_id']
                # 整段批改
                fake_whole = '{"reference":"Chinese tea culture has a long history.","reviews":[],"score":10,"tier":"第二档"}'
                with patch.object(S.client, 'chat', return_value=fake_whole):
                    TestClient(S.app).post('/api/trw/grade',
                                           json={'type': 'translation',
                                                 'task': {'material': '中国茶文化历史悠久', '_history_id': hid},
                                                 'answer': 'China tea culture has a long history.'})
                # 单句批改
                fake_sent = '{"reference":"Chinese tea culture boasts a long history.","comment":"不错","score":12,"tier":"第二档"}'
                with patch.object(S.client, 'chat', return_value=fake_sent):
                    TestClient(S.app).post('/api/trw/sentence/grade',
                                           json={'exercise_id': hid, 'sentence': '中国茶文化历史悠久',
                                                 'answer': 'China tea culture long history', 'index': 0})
                content = S._history_get(hid)
                self.assertIn('grade', content)               # 整段批改保留
                self.assertIn('answer', content)              # 整段答案保留
                self.assertIn('sentence_grades', content)     # 单句批改保留
                self.assertEqual(content['sentence_grades']['0']['grade']['score'], 12)
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_grade_rejects_cross_type_history_id(self):
        """整段批改携带其它类型条目的 _history_id 必须 400，且不覆盖原条目。"""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import server as S
        old = S.cfg['paths'].pop('history_dir', None)
        with tempfile.TemporaryDirectory() as tmp:
            S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
            Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
            try:
                hid = S._archive_daily({'form': 'x', 'text': 'x'})
                fake = '{"reference":"r","reviews":[],"score":10,"tier":"x"}'
                with patch.object(S.client, 'chat', return_value=fake):
                    r = TestClient(S.app).post('/api/trw/grade',
                                               json={'type': 'translation',
                                                     'task': {'material': 'x', '_history_id': hid},
                                                     'answer': 'China tea long history'})
                self.assertEqual(r.status_code, 400)
                content = S._history_get(hid)
                self.assertEqual(content['type'], 'daily')  # 原条目未被覆盖
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_split_sentences_protects_abbreviations(self):
        r = E.split_sentences('Dr. Smith went home. He was happy.')
        self.assertEqual(r, ['Dr. Smith went home.', 'He was happy.'])  # 不在 Dr. 处误切，普通句号仍切


if __name__ == '__main__':
    unittest.main(verbosity=2)
