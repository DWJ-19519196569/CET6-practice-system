# test_wordlist.py — 词库模块单元测试
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wordlist import Wordlist

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '单词', 'wordlist_cet6_乱序.txt')
wl = Wordlist(PATH)

assert 3900 < wl.total < 4100, f'去重后词条数异常: {wl.total}'
print(f'[1] 去重后词条总数: {wl.total}（原始 5651 行，重复词已归并）OK')

c0 = wl.day_chunk(0)
c1 = wl.day_chunk(1)
assert len(c0) == 300 and len(c1) == 300
assert c0[0][0] != c1[0][0], 'day 0 和 day 1 起点应不同'
assert all(w for w, _ in c0), '词块里有空词'
print(f'[2] day_chunk: day0 起于 "{c0[0][0]}", day1 起于 "{c1[0][0]}" OK')

c19 = wl.day_chunk(19)
start19 = (19 * 300) % wl.total
assert c19[0] == wl.entries[start19], '循环回卷起点不对'
print(f'[3] 循环回卷: day19 起于索引 {start19} ("{c19[0][0]}") OK')

w5 = wl.window(5)
assert 300 <= len(w5) <= 600, f'窗口大小异常: {len(w5)}'
ws = [x[0].lower() for x in w5]
assert len(ws) == len(set(ws)), '窗口未去重'
print(f'[4] 滚动窗口: day5 窗口 {len(w5)} 词（去重后）OK')

s_a = wl.sample(7, 60)
s_b = wl.sample(7, 60)
assert s_a == s_b, '同一天采样不可复现'
assert len(s_a) == 60
s_c = wl.sample(8, 60)
assert s_a != s_c, '不同天采样应不同'
print(f'[5] 采样可复现: day7/day7 一致, day7/day8 不同 OK')

text = "The consistent battery was competent enough to preserve the possession of the village."
cov = wl.coverage(text)
cov_words = [w for w, _ in cov]
for expect in ['consistent', 'battery', 'competent', 'preserve', 'possession']:
    assert expect in [w.lower() for w in cov_words], f'coverage 漏词: {expect}'
print(f'[6] coverage: 命中 {len(cov)} 词 {cov_words} OK')

d_neg = wl.day_chunk(-1)
print(f'[7] 负数 day（昨天回看）: day-1 起于 "{d_neg[0][0]}" OK')

print()
print('ALL WORDLIST TESTS PASSED')
