# test_stream_parser.py — SegmentStreamParser 单元测试（流式切句/选项解析/边界）
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import SegmentStreamParser, parse_story


def feed_by_chunk(text, chunk=7):
    """按 chunk 字符模拟流式喂入，收集所有切出的句子。"""
    p = SegmentStreamParser()
    got = []
    for i in range(0, len(text), chunk):
        got.extend(p.feed(text[i:i + chunk]))
    return p, got


# 1. 基本流：完整输出
raw = """<SEGMENT>
The compass glowed faintly in her palm. She heard footsteps behind the door. Three shadows waited outside the cabin. Her hand trembled as she reached for the lock.
</SEGMENT>
A) Throw the compass into the ocean to destroy it forever.
B) Turn around and confront the intruder with a rusty knife.
C) Slip through the hidden hatch beneath the floorboards."""
p, sents = feed_by_chunk(raw)
result = p.finish()
assert len(result['segment'].split()) >= 20, '正文太短'
assert len(result['options']) == 3, f'选项数: {len(result["options"])}'
assert 'ocean' in result['options'][0], '选项 A 解析错误'
assert len(sents) >= 3, f'流式切句数: {len(sents)}'
rejoined = ' '.join(sents)
assert 'glowed' in rejoined, '流式句子内容丢失'
print(f'[1] 基本流: {len(sents)} 句流式切出, 3 选项 OK')

# 2. 极小 chunk（1 字符/次）一致性
p1, s1 = feed_by_chunk(raw, 1)
r1 = p1.finish()
assert r1 == result, '1 字符流式与 7 字符结果不一致'
print(f'[2] 1 字符流: 一致 OK')

# 3. <SEGMENT> 标记跨 chunk 分裂
p3 = SegmentStreamParser()
out = []
out += p3.feed('<SEG')  # 前缀不完整
assert out == []
out += p3.feed('MENT>\nHello world. This is a test. ')
assert out == ['Hello world.', 'This is a test.'], f'标记分裂后切句: {out}'
print('[3] 标记跨块分裂: OK')

# 4. 引号场景（句号在引号内，句子粘连可接受但不得丢内容）
raw4 = '<SEGMENT>\nShe said, "I will go." Then she left the room. Nobody stopped her.\n</SEGMENT>\nA) aaa.\nB) bbb.\nC) ccc.'
p4, s4 = feed_by_chunk(raw4)
r4 = p4.finish()
body_words = len(r4['segment'].split())
assert body_words >= 10, f'引号场景丢词: {r4["segment"]!r}'
full = ' '.join(s4)
assert 'stopped' in full or 'stopped' in r4['segment'], '引号场景尾句丢失'
print(f'[4] 引号场景: 正文 {body_words} 词保留 OK')

# 5. 无尾随空白的段末句（flush 兜底）
raw5 = '<SEGMENT>\nOne. Two. The end.\n</SEGMENT>\nA) x\nB) y\nC) z'
p5, s5 = feed_by_chunk(raw5)
r5 = p5.finish()
assert 'The end.' in r5['segment'], f'段末句丢失: {r5["segment"]!r}'
print('[5] 段末 flush: OK')

# 6. 非流式完整解析
r6 = parse_story(raw)
assert len(r6['options']) == 3 and 'ocean' in r6['options'][0]
print('[6] parse_story 完整解析: OK')

# 7. 异常容错：无 SEGMENT 标记
try:
    parse_story('no markers here at all')
    print('[7] FAIL: 应当抛错')
except ValueError:
    print('[7] 无标记容错: OK')

# 8. 缩写保护：Dr. / U.S. 不应被切成独立短句
raw8 = '<SEGMENT>\nDr. Smith went home. He was happy. He lives in the U.S.\n</SEGMENT>\nA) x\nB) y\nC) z'
p8, s8 = feed_by_chunk(raw8)
r8 = p8.finish()
full8 = ' '.join(s8) + ' ' + r8['segment']
assert 'Dr.' not in [s.strip() for s in s8], f'Dr. 被误切成独立句: {s8}'
assert 'Dr. Smith went home' in full8, f'缩写句子丢失: {full8!r}'
print(f'[8] 缩写保护: {len(s8)} 句流式切出, Dr. 未误切 OK')

print()
print('ALL STREAM PARSER TESTS PASSED')
