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


if __name__ == '__main__':
    unittest.main(verbosity=2)
