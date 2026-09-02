# wordlist.py — 词库模块：加载 / 词块 / 滚动窗口 / 可复现采样
import random
from pathlib import Path


class Wordlist:
    def __init__(self, path: str):
        self.entries: list[tuple[str, str]] = []
        seen = set()
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if '\t' not in line:
                continue
            w, m = line.split('\t', 1)
            w = w.strip()
            if w and w.lower() not in seen:
                seen.add(w.lower())
                self.entries.append((w, m.strip()))
        self.total = len(self.entries)

    def day_chunk(self, day: int, size: int = 300) -> list[tuple[str, str]]:
        """第 N 天（0-based）的词块，自动循环回卷。day 可为负（窗口回看昨天）。"""
        start = (day * size) % self.total
        idx = [(start + i) % self.total for i in range(min(size, self.total))]
        return [self.entries[i] for i in idx]

    def window(self, day: int, size: int = 300) -> list[tuple[str, str]]:
        """滚动窗口：昨天 + 今天的词池，去重保序。"""
        pool, seen = [], set()
        for chunk in (self.day_chunk(day - 1, size), self.day_chunk(day, size)):
            for w, m in chunk:
                if w.lower() not in seen:
                    seen.add(w.lower())
                    pool.append((w, m))
        return pool

    def sample(self, day: int, n: int = 60, size: int = 300) -> list[tuple[str, str]]:
        """从滚动窗口采样 n 个词。seed=day，同一天结果可复现。"""
        pool = self.window(day, size)
        rng = random.Random(day)
        return rng.sample(pool, min(n, len(pool)))

    def coverage(self, text: str) -> list[tuple[str, str]]:
        """检查一段文本实际覆盖了词库里哪些词（用于归档的 words.txt）。"""
        words_in_text = set()
        for token in text.replace(',', ' ').replace('.', ' ').replace(';', ' ').split():
            words_in_text.add(token.strip('"?!():\'').lower())
        return [(w, m) for w, m in self.entries if w.lower() in words_in_text]
