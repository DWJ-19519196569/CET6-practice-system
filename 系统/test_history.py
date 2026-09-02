# test_history.py — 历史归档/恢复/删除
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server as S


def _fresh_server(tmp):
    """用一个独立的临时目录替换 server 的历史归档路径，避免污染真实归档。"""
    old = S.cfg['paths'].get('history_dir')
    S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
    Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
    return old


class HistoryIndexTests(unittest.TestCase):
    def test_archive_then_list(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                S._archive_daily({'form': '长对话', 'text': 'HeLLo world.', 'date': date.today().isoformat()})
                items = S._history_list('daily')
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]['type'], 'daily')
                self.assertIn('title', items[0])
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old


class HistoryApiTests(unittest.TestCase):
    def test_list_restore_delete(self):
        import tempfile
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                S._archive_daily({'form': '长对话', 'text': 'A short passage.', 'date': date.today().isoformat()})
                S._archive_story({'segments': ['Seg one.'], 'choices': ['A']})
                S._archive_trw({'type': 'writing', 'task': {'title': 'T'}, 'answer': 'ans', 'grade': {'score': 9}})

                cli = TestClient(S.app)

                # 列表
                r = cli.get('/api/history', params={'type': 'daily'})
                self.assertEqual(r.status_code, 200)
                items = r.json()['items']
                self.assertEqual(len(items), 1)
                hid = items[0]['id']

                # 恢复
                r2 = cli.get('/api/history/item', params={'id': hid})
                self.assertEqual(r2.status_code, 200)
                content = r2.json()
                self.assertIn('text', content)

                # 删除
                r3 = cli.delete('/api/history', params={'id': hid})
                self.assertEqual(r3.status_code, 200)
                r4 = cli.get('/api/history', params={'type': 'daily'})
                self.assertEqual(len(r4.json()['items']), 0)
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
