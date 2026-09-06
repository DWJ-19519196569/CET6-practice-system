# wordlist.py — 词库模块：加载 / 词块 / 滚动窗口 / 可复现采样
import random
from pathlib import Path


# 常见英文屈折/派生后缀（词形还原的轻量近似，无需第三方词形还原库）
_INFLECT_SUFFIXES = (
    ('ies', 'y'), ('ied', 'y'), ('ier', 'y'), ('iest', 'y'),
    ('ing', ''), ('ed', ''), ('es', ''), ('s', ''), ('ly', ''),
)


def _base_forms(token: str) -> set[str]:
    """把一个英文 token 展开为可能的基础形式（含原形与常见变形还原）。"""
    t = token.lower()
    out = {t}
    for suf, repl in _INFLECT_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            base = t[:-len(suf)] + repl
            out.add(base)
            # 双写辅音还原：stopped→stop、running→run
            if len(base) >= 2 and base[-1] == base[-2]:
                out.add(base[:-1])
            # 去 e 后补回：making→make、fired→fire
            if suf in ('ing', 'ed'):
                out.add(base + 'e')
    return out


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
        if self.total == 0:  # 空词表边界：避免 % 0 崩溃
            return []
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
        """检查一段文本实际覆盖了词库里哪些词（用于归档的 words.txt）。

        除精确匹配外，还做轻量词形还原（复数/过去式/进行时/副词等常见变形），
        命中 LLM 织入的变形词，避免 words_hit 被低估。
        """
        forms = set()
        for token in text.replace(',', ' ').replace('.', ' ').replace(';', ' ').split():
            forms.update(_base_forms(token.strip('"?!():\'')))
        return [(w, m) for w, m in self.entries if w.lower() in forms]
