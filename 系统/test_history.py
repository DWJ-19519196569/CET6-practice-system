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

    def test_archive_story_empty_segments_does_not_crash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                hid = S._archive_story({'segments': []})  # 空段落不应崩（原为 IndexError）
                content = S._history_get(hid)
                self.assertEqual(content['title'], '互动故事')
                self.assertEqual(content['segments'], [])
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


class AttachedAudioTests(unittest.TestCase):
    """删除历史时，应一并删除每日一篇/互动故事各自目录里的音频（且在根目录之内才删）。"""

    def _setup(self, tmp):
        old_hist = S.cfg['paths'].pop('history_dir', None)
        old_daily = S.cfg['paths']['daily_dir']
        old_story = S.cfg['paths']['story_dir']
        S.cfg['paths']['history_dir'] = str(Path(tmp) / '历史记录')
        S.cfg['paths']['daily_dir'] = str(Path(tmp) / '每日一篇')
        S.cfg['paths']['story_dir'] = str(Path(tmp) / '互动故事')
        Path(S.cfg['paths']['history_dir']).mkdir(parents=True, exist_ok=True)
        return old_hist, old_daily, old_story

    def _teardown(self, old_hist, old_daily, old_story):
        if old_hist is None:
            S.cfg['paths'].pop('history_dir', None)
        else:
            S.cfg['paths']['history_dir'] = old_hist
        S.cfg['paths']['daily_dir'] = old_daily
        S.cfg['paths']['story_dir'] = old_story

    def test_delete_daily_removes_audio_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_hist, old_daily, old_story = self._setup(tmp)
            try:
                # 新数据：每日一篇按「日期/时间戳_id」建唯一子目录
                d = Path(tmp) / '每日一篇' / date.today().isoformat() / '120000_abc123'
                d.mkdir(parents=True)
                (d / 'story.wav').write_bytes(b'x')
                hid = S._archive_daily({'form': '长对话', 'text': 'x', 'audio_dir': str(d)})
                self.assertTrue(d.exists())
                S._history_delete(hid)
                self.assertFalse(d.exists())  # 音频目录已删
            finally:
                self._teardown(old_hist, old_daily, old_story)

    def test_delete_daily_legacy_shared_dir_keeps_siblings(self):
        """老数据：多条每日一篇共享同一「日期」目录时，删一条不得波及同目录其它条目。"""
        with tempfile.TemporaryDirectory() as tmp:
            old_hist, old_daily, old_story = self._setup(tmp)
            try:
                shared = Path(tmp) / '每日一篇' / date.today().isoformat()
                shared.mkdir(parents=True)
                (shared / 'story.wav').write_bytes(b'x')
                other = shared / 'other.wav'
                other.write_bytes(b'y')
                hid = S._archive_daily({'form': '长对话', 'text': 'x',
                                        'audio_dir': str(shared), 'audio_file': str(shared / 'story.wav')})
                S._history_delete(hid)
                self.assertTrue(shared.exists())              # 共享目录本身保留
                self.assertTrue(other.exists())               # 其它条目文件不受影响
                self.assertFalse((shared / 'story.wav').exists())  # 本条目的音频文件已删
            finally:
                self._teardown(old_hist, old_daily, old_story)

    def test_delete_story_removes_audio_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_hist, old_daily, old_story = self._setup(tmp)
            try:
                d = Path(tmp) / '互动故事' / f'{date.today().isoformat()}_session-01'
                d.mkdir(parents=True)
                (d / 'story.wav').write_bytes(b'x')
                hid = S._archive_story({'segments': ['Seg one.'], 'choices': ['A'], 'audio_dir': str(d)})
                self.assertTrue(d.exists())
                S._history_delete(hid)
                self.assertFalse(d.exists())
            finally:
                self._teardown(old_hist, old_daily, old_story)

    def test_delete_does_not_touch_outside_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_hist, old_daily, old_story = self._setup(tmp)
            try:
                outside = Path(tmp) / '别处'
                outside.mkdir()
                (outside / 'f.txt').write_bytes(b'x')
                hid = S._archive_daily({'form': '长对话', 'text': 'x', 'audio_dir': str(outside)})
                S._history_delete(hid)
                self.assertTrue(outside.exists())  # 根目录之外不删
            finally:
                self._teardown(old_hist, old_daily, old_story)

    def test_delete_refuses_path_outside_history_root(self):
        """history.json 里 path 被篡改成历史目录之外的目录时，删除不得整删该目录。"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                victim = Path(tmp) / 'victim'
                victim.mkdir()
                (victim / 'x.txt').write_text('x', encoding='utf-8')
                hid = S._archive_daily({'form': 'x', 'text': 'x'})
                items = S._history_index()
                for i in items:
                    if i['id'] == hid:
                        i['path'] = str(victim)  # 篡改 path 指向历史目录之外
                S._save_history_index(items)
                S._history_delete(hid)
                self.assertTrue(victim.exists())  # 外部目录不被删
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_index_rejects_missing_required_fields(self):
        """history.json 条目缺必需字段时视为损坏，重建为空而非 API 500。"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                hj = Path(tmp) / '历史记录' / 'history.json'
                hj.parent.mkdir(parents=True, exist_ok=True)
                hj.write_text('[{}]', encoding='utf-8')
                items = S._history_index()
                self.assertEqual(items, [])
                self.assertFalse(hj.exists())  # 原文件被备份走
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_index_rejects_path_outside_root(self):
        """history.json 条目 path 指向历史目录之外时视为损坏（防读侧越界泄露）。"""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                hj = Path(tmp) / '历史记录' / 'history.json'
                hj.parent.mkdir(parents=True, exist_ok=True)
                outside = Path(tmp) / 'outside'
                outside.mkdir()
                (outside / 'content.json').write_text('{"secret":1}', encoding='utf-8')
                entry = {'id': 'badid', 'type': 'daily', 'date': 'x', 'title': 'x',
                         'path': str(outside), 'created_at': 'x'}
                hj.write_text(json.dumps([entry], ensure_ascii=False), encoding='utf-8')
                items = S._history_index()
                self.assertEqual(items, [])  # 视为损坏，重建为空
                self.assertFalse(hj.exists())  # 原文件被备份走
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old

    def test_get_handles_non_dict_content(self):
        """content.json 是合法 JSON 但非对象（[]）时，读接口不崩、按 {} 处理。"""
        with tempfile.TemporaryDirectory() as tmp:
            old = _fresh_server(tmp)
            try:
                hid = S._archive('daily', {'text': 'x'})
                cpath = Path(S._history_get(hid)['path']) / 'content.json'
                cpath.write_text('[]', encoding='utf-8')
                item = S._history_get(hid)
                self.assertIsInstance(item, dict)
                self.assertEqual(item['type'], 'daily')
                self.assertEqual(S._history_content(hid), {})
            finally:
                if old is None:
                    S.cfg['paths'].pop('history_dir', None)
                else:
                    S.cfg['paths']['history_dir'] = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
