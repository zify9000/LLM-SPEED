"""引擎纯逻辑回归测试：上下文构造（嵌套前缀/确定性/模块块级/wrap/零输入）、
复测聚合、错误语义化、超窗删减辅助、部署上限超窗贴边裁减判定、
媒体语料合成与媒体聚合口径（ADR-0020）。

运行（任一方式，无需 pytest 也可跑）：
    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/ -v
"""
import asyncio
import os
import re
import struct
import tempfile
import time
import unittest
import unittest.mock
import wave
import io

import bench
from bench import (
    ANOMALY_RETEST_MAX,
    CTX_DEV_TOLERANCE,
    CTX_EDGE_TRIM_RATIO,
    CTX_HEADROOM,
    BenchRun,
    CLIENT_CUT_FACTOR,
    OUT_HINT_FACTOR,
    SCENARIOS,
    _aggregate_media_reqs,
    _aggregate_reps,
    _align_echo_region,
    _asr_audio,
    _build_ocr_messages,
    _classify_http_error,
    _correct_filler_chars,
    _decode_burst,
    _detect_rep_anomaly,
    _fill,
    _head_sample,
    _image_size,
    _is_collapsed_reply,
    _is_ctx_overflow,
    _is_degenerate_text,
    _is_implausible_prefill_rate,
    _is_soft_ctx_overflow,
    _load_code_pool,
    _make_module_stream,
    _make_stream,
    _net_elapsed,
    _ocr_images,
    _plan_ctx_edge,
    _ratio_refold,
    _synth_document_png,
    _synth_speech_wav,
    _total_decode_rates,
    _trim_middle,
    _tts_text,
    _wav_seconds,
    _wma,
    build_messages,
    rate_prior,
)


class TestRatePrior(unittest.TestCase):
    """prefill 速率先验曲线：log2(ctx) 分段线性插值 + 末段斜率外推 + 钳制。"""

    def test_empty_and_single(self):
        self.assertIsNone(rate_prior([], 4096))
        self.assertEqual(rate_prior([(2048, 500.0)], 65536), 500.0)   # 单点平推
        self.assertEqual(rate_prior([(2048, 500.0)], 1024), 500.0)    # 低于首点取首点

    def test_interpolation(self):
        pts = [(4096, 400.0), (16384, 800.0)]   # 每翻倍 +400
        self.assertAlmostEqual(rate_prior(pts, 8192), 600.0)          # log2 中点
        self.assertAlmostEqual(rate_prior(pts, 4096), 400.0)
        self.assertAlmostEqual(rate_prior(pts, 16384), 800.0)

    def test_extrapolation_falling(self):
        pts = [(16384, 931.1), (32768, 852.8), (65536, 682.0)]
        # 末段斜率 -170.8/翻倍 → 128K ≈ 511.2（真实值 506.9）
        self.assertAlmostEqual(rate_prior(pts, 131072), 511.2, places=1)

    def test_extrapolation_clamps(self):
        # 下降趋势过猛时下钳 0.35×
        pts = [(4096, 1000.0), (8192, 100.0)]
        self.assertAlmostEqual(rate_prior(pts, 65536), 35.0)          # 0.35×100
        # 上升趋势上钳 1.3×
        pts = [(4096, 100.0), (8192, 1000.0)]
        self.assertAlmostEqual(rate_prior(pts, 65536), 1300.0)        # 1.3×1000

    def test_real_curve_replay(self):
        """用户 8/12 实测曲线（creative）：16K 峰 931.1 → 192K 415.3。"""
        pts = [(4096, 624.6), (8192, 777.6), (16384, 931.1),
               (32768, 852.8), (65536, 682.0), (131072, 506.9)]
        # 64K→128K 趋势外推 192K（=1.5×128K，log2 差 0.585）：506.9−175.1×0.585≈404.5
        self.assertAlmostEqual(rate_prior(pts, 196608), 404.5, places=1)
        # 曲内点取实测
        self.assertAlmostEqual(rate_prior(pts, 32768), 852.8)


class TestNetElapsed(unittest.TestCase):
    """净耗时换算：RTT 扣减封顶为实测值一半，基线抖动超过小 TTFT 时不爆炸。"""

    def test_normal_deduction(self):
        self.assertAlmostEqual(_net_elapsed(10.0, 0.2), 9.8)

    def test_rtt_exceeds_elapsed_capped(self):
        # rtt(0.368) > ttft(0.346) 的真实案例：原 1e-3 下界产出 79000 tok/s
        self.assertAlmostEqual(_net_elapsed(0.346, 0.368), 0.173)

    def test_none_and_zero_rtt(self):
        self.assertAlmostEqual(_net_elapsed(5.0, None), 5.0)
        self.assertAlmostEqual(_net_elapsed(5.0, 0.0), 5.0)

    def test_zero_elapsed_no_division_by_zero(self):
        self.assertGreater(_net_elapsed(0.0, 0.1), 0)


class TestDecodeBurst(unittest.TestCase):
    """突发交付甄别：投机解码周期交付（如 ACCEPT=4、tg=100 → 平均事件间隔
    40ms）不误标，亚毫秒缓冲冲刷仍标记。"""

    def test_speculative_periodic_gap_not_burst(self):
        self.assertFalse(_decode_burst(16, 0.6))    # 600ms/15 间隔 ≈ 40ms
        self.assertFalse(_decode_burst(8, 0.56))    # 560ms/7 间隔 ≈ 80ms

    def test_subms_flush_flagged(self):
        self.assertTrue(_decode_burst(32, 0.02))    # 20ms/31 间隔 ≈ 0.65ms
        self.assertFalse(_decode_burst(4, 0.001))   # chunk 太少不足为凭
        self.assertFalse(_decode_burst(32, None))   # 无 decode 时长不判


class TestFillStream(unittest.TestCase):
    """ADR-0005 确定性嵌套语料流的关键性质。"""

    POOL = ["甲" * 100, "乙" * 200, "丙" * 300]

    def setUp(self):
        self.stream, self.bounds = _make_stream(self.POOL)

    def test_deterministic(self):
        self.assertEqual((self.stream, self.bounds), _make_stream(self.POOL))

    def test_boundaries_block_starts(self):
        """边界表 = 各语料块在流中的起始偏移（含流首 0，分隔符计入累积）：
        流首为 0；每个非零边界紧随 "\n\n" 之后，且从该处开始截到流尾的内容
        恰为末尾完整块组——echo 改写区对齐的依据。"""
        self.assertEqual(self.bounds[0], 0)
        self.assertLess(self.bounds[1], self.bounds[2])
        for b in self.bounds[1:]:
            self.assertEqual(self.stream[b - 2:b], "\n\n")
        # 末块边界到流尾 = 完整块内容（无尾随分隔符计入）
        self.assertIn(len(self.stream) - self.bounds[-1],
                      {len(p) for p in self.POOL})
        # 任意块边界截断的尾部都是若干完整块（以块内容开头，非半行）
        for b in self.bounds:
            self.assertTrue(self.stream[b:].startswith(("甲", "乙", "丙")))

    def test_nested_prefix(self):
        """小档位填充必须是大档位的精确前缀（嵌套前缀精确记账的前提）。"""
        short = _fill(self.stream, 250)
        long_ = _fill(self.stream, 500)
        self.assertTrue(long_.startswith(short))

    def test_wrap_continuity(self):
        """流不够长时整流循环续接，内容仍确定且衔接无缝。"""
        total = len(self.stream)
        wrapped = _fill(self.stream, total * 2 + 123)
        self.assertEqual(len(wrapped), total * 2 + 123)
        self.assertEqual(wrapped[:total], self.stream)
        self.assertEqual(wrapped[total: total * 2], self.stream)   # 第二遍整段接续
        self.assertEqual(wrapped[total * 2:], self.stream[:123])

    def test_empty_and_nonpositive(self):
        self.assertEqual(_fill(self.stream, 0), "")
        self.assertEqual(_fill("", 10), "")


class TestModuleStream(unittest.TestCase):
    """ADR-0005 模块块级语料流：块间乱序（确定性）、块内保序（上下文关联性）。"""

    BLOCKS = [["甲" * 100, "乙" * 200], ["丙" * 300], ["丁" * 150, "戊" * 250]]

    def test_deterministic(self):
        self.assertEqual(_make_module_stream(self.BLOCKS),
                         _make_module_stream(self.BLOCKS))

    def test_blocks_contiguous_and_ordered(self):
        """块内文件必须相邻且保持给定顺序（模块关联性）；块次序允许乱序。"""
        stream, _ = _make_module_stream(self.BLOCKS)
        items = stream.split("\n\n")
        self.assertEqual(len(items), sum(len(b) for b in self.BLOCKS))
        pos = {s: i for i, s in enumerate(items)}
        for block in self.BLOCKS:
            idxs = [pos[s] for s in block]
            self.assertEqual(idxs, sorted(idxs))                     # 块内保序
            self.assertEqual(idxs, list(range(idxs[0], idxs[0] + len(idxs))))   # 块内相邻

    def test_flat_stream_properties_carry_over(self):
        """嵌套前缀前提不变：同一流截取仍互为前缀（校准精确记账不受影响）。"""
        stream, _ = _make_module_stream(self.BLOCKS)
        self.assertTrue(_fill(stream, 500).startswith(_fill(stream, 200)))

    def test_boundaries_file_granularity(self):
        """边界表粒度 = 文件（模块块内每个文件一个起始偏移），偏移落在块
        内容开头（紧随分隔符）——echo 改写区可对齐到单文件粒度。"""
        stream, bounds = _make_module_stream(self.BLOCKS)
        self.assertEqual(bounds[0], 0)
        self.assertEqual(len(bounds), sum(len(b) for b in self.BLOCKS))
        for b in bounds[1:]:
            self.assertEqual(stream[b - 2:b], "\n\n")
        # 边界单调递增且落在流内
        self.assertEqual(bounds, sorted(bounds))
        self.assertTrue(all(0 <= b < len(stream) for b in bounds))

    def test_load_code_pool_groups_by_directory(self):
        """真实语料：同目录文件聚为一块，块内文件名排序，过短文件跳过。"""
        blocks = _load_code_pool()
        self.assertTrue(blocks)
        for block in blocks:
            self.assertTrue(all(len(t) >= 2000 for t in block))

    def test_load_code_pool_fallback(self):
        """corpus 目录缺失时回退内置合成模块池（每条自成一个模块块）。"""
        with unittest.mock.patch.object(bench, "CORPUS_CODE_DIR", "/nonexistent-dir"):
            self.assertEqual(_load_code_pool(), [[s] for s in bench._CODE_POOL])


class TestEchoRegionAlign(unittest.TestCase):
    """echo 改写区起点对齐（_align_echo_region）：三层回退——块边界 →
    段/函数边界（"\\n\\n" 之后）→ 现状任意字符偏移。"""

    def test_hit_block_boundary_tail_full_blocks(self):
        """命中块边界：取满足 [0.8r, 1.6r] 的最后一个边界——改写区 = 材料
        末尾的完整块组，预填从块首开始。"""
        # 块 [0,100) / [100,350) / [350,600)，流长 600；region=300
        # → 目标起点 300；边界尾长：0→600（越上界 480）、100→500（越）、
        # 350→250 ∈ [240, 480] ✓ → 取 350（改写区 = 末尾完整块）
        filler = "A" * 100 + "\n\n" + "B" * 250 + "\n\n" + "C" * 250
        bounds = [0, 102, 354]
        self.assertEqual(_align_echo_region(filler, 300, bounds), 354)
        self.assertEqual(filler[354:354 + 160], "C" * 160, "预填从块首开始")

    def test_block_too_large_falls_to_paragraph(self):
        """块过大（无边界落在 [0.8r, 1.6r]）：落到目标起点后第一个 "\n\n"
        之后的段/函数边界——连续性完整。"""
        filler = "甲" * 500 + "\n\n" + "乙" * 280 + "\n\n" + "丙" * 100
        bounds = [0]   # 唯一边界尾长 = 全长，超 1.6r
        region = 400
        target = len(filler) - region
        idx = filler.find("\n\n", target, target + 160)
        self.assertEqual(_align_echo_region(filler, region, bounds), idx + 2)
        start = _align_echo_region(filler, region, bounds)
        self.assertGreater(start, target, "起点后移到段边界，非任意字符偏移")
        self.assertEqual(filler[start - 2:start], "\n\n", "起点紧随分隔符")

    def test_no_suitable_boundary_keeps_current_behavior(self):
        """无块边界命中且窗口内无 "\n\n"：回退现状（目标起点 = 任意字符
        偏移），行边界收口兜底。"""
        filler = "".join(f"行{i}\n" for i in range(500))   # 无 "\n\n"
        region = 300
        self.assertEqual(_align_echo_region(filler, region, [0]),
                         len(filler) - region)

    def test_filler_shorter_than_region(self):
        """len(filler) < region：region 收口为全长，起点 0（流首 0 是天然
        块边界），预填即材料开头。"""
        filler = "X" * 100
        self.assertEqual(_align_echo_region(filler, 300, [0]), 0)

    def test_empty_boundaries_falls_through(self):
        """边界表为空：直接走段边界/现状层，不炸。"""
        filler = "前" * 450 + "\n\n" + "后" * 100
        region = 300
        target = len(filler) - region
        idx = filler.find("\n\n", target, target + 120)
        self.assertEqual(_align_echo_region(filler, region, []),
                         idx + 2 if idx >= 0 else target)

    def test_boundaries_beyond_filler_ignored(self):
        """边界表超出 len(filler) 的部分不参与（filler 是流的前缀截取，
        循环续接流的后续边界不可见）。"""
        filler = "A" * 100 + "\n\n" + "B" * 250 + "\n\n" + "C" * 250
        bounds = [0, 102, 354, 700]   # 700 > len(filler)
        self.assertEqual(_align_echo_region(filler, 300, bounds), 354)

    def test_deterministic(self):
        """同输入同输出：嵌套前缀 / 多次运行的选择一致。"""
        filler = "A" * 100 + "\n\n" + "B" * 250 + "\n\n" + "C" * 250
        bounds = [0, 102, 354]
        r1 = _align_echo_region(filler, 300, bounds)
        r2 = _align_echo_region(filler, 300, bounds)
        self.assertEqual(r1, r2)
        # 前缀关系：短 filler 的选择只依赖自身长度，不随流整体变化
        self.assertEqual(_align_echo_region(filler[:500], 300, bounds),
                         _align_echo_region(filler[:500], 300, bounds[:2]))


class TestHeadSample(unittest.TestCase):
    """存档输出采样累积（_head_sample）：超 200 截断、不足全留、截断即停。"""

    def test_truncate_at_cap(self):
        buf = "x" * 200
        self.assertEqual(_head_sample(buf, "继续来的内容", 200), buf)
        self.assertEqual(_head_sample("", "字" * 250, 200), "字" * 200)

    def test_short_piece_kept_whole(self):
        self.assertEqual(_head_sample("", "你好", 200), "你好")
        self.assertEqual(_head_sample("abc", "de", 200), "abcde")

    def test_boundary_and_empty(self):
        self.assertEqual(_head_sample("a" * 199, "bcd", 200), "a" * 199 + "b")
        self.assertEqual(_head_sample("", "", 200), "", "空输出采样为空串")
        self.assertEqual(_head_sample("y" * 200, "", 200), "y" * 200)


class TestScenarioPriors(unittest.TestCase):
    """ADR-0006 输入系数分场景先验：探测校准失败路径的兜底默认值。"""

    def test_in_cpt_priors(self):
        for sc in SCENARIOS.values():
            if sc.get("kind", "llm") != "llm":
                continue   # 媒体场景（asr/ocr/tts）无文本 prompt 构造，无输入系数
            self.assertGreater(sc["in_cpt"], 0.5)
        # 实测序：C/C++ 代码的字符/token 远高于中文散文（3.9 vs 1.5）
        self.assertGreater(SCENARIOS["code"]["in_cpt"],
                           SCENARIOS["creative"]["in_cpt"])


class TestBuildMessages(unittest.TestCase):
    def test_zero_input_branch(self):
        """ctx=0：不注入参考材料，一句话命题，填充字符数为 0（ADR-0005）。"""
        msgs, est, filler_n = build_messages("creative", 0, 1.8, "n")
        self.assertEqual(filler_n, 0)
        self.assertEqual(len(msgs), 2)
        self.assertIn("800字", msgs[1]["content"])
        self.assertGreater(est, 0)

    def test_explicit_filler_chars(self):
        """显式 filler_chars 精确控制填充长度（嵌套前缀记账路径）。"""
        _, _, n1 = build_messages("code", 4096, 3.4, "n", filler_chars=5000)
        self.assertEqual(n1, 5000)

    def test_nonzero_injects_filler_and_nonce(self):
        msgs, est, filler_n = build_messages("creative", 4096, 1.8, "nonce-42")
        self.assertGreater(filler_n, 0)
        self.assertIn("nonce-42", msgs[1]["content"])
        self.assertGreater(est, 4000 * 0.9)

    def test_agent_scenario(self):
        """Agent 场景（ADR-0014）：vendored SWE-agent 轨迹语料加载（块 ≥2000
        字符），构造路径与既有场景一致（零输入档无填充、非零档注入轨迹）。"""
        pool = SCENARIOS["agent"]["filler_pool"]
        self.assertGreater(len(pool), 1)   # vendored 轨迹块（corpus/agent/）
        self.assertTrue(all(len(b) >= 2000 for b in pool))
        self.assertGreater(len(SCENARIOS["agent"]["filler_stream"]), 100000)
        msgs, est, filler_n = build_messages("agent", 4096, 3.5, "nonce-7")
        self.assertGreater(filler_n, 0)
        self.assertIn("nonce-7", msgs[1]["content"])
        _, _, filler0 = build_messages("agent", 0, 3.5, "n")
        self.assertEqual(filler0, 0)

    def test_echo_instruction_variant(self):
        """高复用改写口径（ADR-0021）：echo=True 换用 echo_instruction 并在末尾
        追加 assistant 预填（改写区开头原样预填，物理消除聊天式前言）；同
        filler_chars + 同 nonce 下 user 与正常构造仅尾部任务句不同（服务端前缀
        缓存可命中整段参考材料）；无 echo_instruction 的场景回退普通指令且无
        预填；零输入档不受影响。"""
        normal, _, fn1 = build_messages("creative", 4096, 1.5, "nonce-9",
                                        filler_chars=5000)
        echo, _, fn2 = build_messages("creative", 4096, 1.5, "nonce-9",
                                      filler_chars=5000, echo=True)
        self.assertEqual(fn1, fn2, "echo 与正常构造填充长度一致")
        self.assertEqual(normal[0], echo[0], "system 不变")
        self.assertIn("改写润色", echo[1]["content"])
        self.assertNotIn("改写润色", normal[1]["content"])
        cut = normal[1]["content"].rindex("\n\n任务：")
        self.assertEqual(normal[1]["content"][:cut], echo[1]["content"][:cut],
                         "仅尾部任务句不同，head+filler 前缀完全一致")
        # assistant 预填：改写区开头原样预填，内容必出现在参考材料末尾区段
        self.assertEqual(len(echo), 3)
        self.assertEqual(echo[2]["role"], "assistant")
        prefill = echo[2]["content"]
        self.assertTrue(0 < len(prefill) <= 170)
        self.assertIn(prefill.strip()[:60], echo[1]["content"])
        # 改写区起点对齐（ADR-0021 续）：预填 = 改写区开头原样，且改写区起点
        # 落在结构边界——块边界（紧随 "\n\n"）或流首；同参数两次构造一致
        # （确定性，嵌套前缀多次运行不受影响）
        c2, _, _ = build_messages("creative", 4096, 1.5, "nonce-9",
                                  filler_chars=5000, echo=True)
        self.assertEqual(prefill, c2[2]["content"])
        filler = _fill(SCENARIOS["creative"]["filler_stream"], 5000)
        csc = SCENARIOS["creative"]
        region = min(5000, int(csc["max_tokens_default"] * csc["out_cpt"]))
        start = _align_echo_region(filler, region, csc["filler_boundaries"])
        self.assertEqual(filler[start:start + len(prefill)], prefill,
                         "预填 = 改写区开头原样")
        self.assertTrue(start == 0 or filler[start - 2:start] == "\n\n",
                        "改写区起点对齐到块/段边界，不从半个结构单元开始")
        self.assertGreaterEqual(len(filler) - start, 0.6 * region)
        self.assertLessEqual(len(filler) - start, 1.6 * region)
        self.assertEqual(len(normal), 2, "正常构造不带 assistant 消息")
        ec, _, _ = build_messages("code", 8192, 3.9, "n2",
                                  filler_chars=8000, echo=True)
        self.assertIn("重构", ec[1]["content"])
        self.assertEqual(ec[2]["role"], "assistant")
        # agent 无 echo_instruction：echo=True 回退普通 instruction，无预填
        ag_n, _, _ = build_messages("agent", 4096, 3.5, "n3")
        ag_e, _, _ = build_messages("agent", 4096, 3.5, "n3", echo=True)
        self.assertEqual(ag_n[1]["content"], ag_e[1]["content"])
        self.assertEqual(len(ag_e), 2)
        # 零输入档：echo 参数无效果
        z1, _, _ = build_messages("creative", 0, 1.5, "n")
        z2, _, _ = build_messages("creative", 0, 1.5, "n", echo=True)
        self.assertEqual(z1, z2)

    def test_agent_turn_sizes(self):
        """Agent 轮次构成（ADR-0014/0015）：轮 1 冷启动（独立配置默认 10K，
        阶段 0）；阶段一 n1 轮短文本暖轮（N(1K) 正态钳制到区间，升序）；
        阶段二 n2 轮长文本 ladder（默认 4K 起逐轮翻倍）。默认 1+6+6：
        长轮 4K/8K/16K/32K/64K/128K。"""
        from bench import AGENT_COLD_CTX_DEFAULT, AGENT_PHASE2_BASE, _gen_agent_turns
        import random
        plan = _gen_agent_turns(6, 6, 256, 2048, random.Random(42))
        self.assertEqual(len(plan), 13)
        self.assertEqual(plan[0], (AGENT_COLD_CTX_DEFAULT, 0), "轮 1 冷启动独立成段")
        p1 = [s for s, ph in plan if ph == 1]
        p2 = [s for s, ph in plan if ph == 2]
        self.assertTrue(all(256 <= s <= 2048 for s in p1), "短轮钳制在增量区间内")
        self.assertEqual(p1, sorted(p1), "短轮按增量长度升序")
        self.assertEqual(p2, [4096, 8192, 16384, 32768, 65536, 131072],
                         "长轮为 4K 起翻倍 ladder（默认 4K~128K）")
        self.assertEqual(p2[0], AGENT_PHASE2_BASE)
        # 区间退化为单值时短轮增量恒定（测试/复现口径）
        fixed = _gen_agent_turns(2, 2, 1024, 1024, random.Random(1), p2_base=8192)
        self.assertEqual(fixed, [(10240, 0), (1024, 1), (1024, 1),
                                 (8192, 2), (16384, 2)])
        # 冷启动上下文可配：独立于两阶段
        cold = _gen_agent_turns(1, 0, 256, 2048, random.Random(2), cold_ctx=4096)
        self.assertEqual(cold[0], (4096, 0))
        self.assertEqual(len(cold), 2)
        # 单阶段：n1=0 纯长文本链 / n2=0 纯短文本链（冷启动轮恒在）
        self.assertEqual(_gen_agent_turns(0, 2, 256, 2048, random.Random(2),
                                          p2_base=8192),
                         [(10240, 0), (8192, 2), (16384, 2)])
        # 阶段二起始增量可配：ladder 从 p2_base 起翻倍
        custom = _gen_agent_turns(1, 3, 256, 2048, random.Random(3), p2_base=8192)
        self.assertEqual([s for s, _ in custom[2:]], [8192, 16384, 32768])


class TestAggregateReps(unittest.TestCase):
    """ADR-0017 复测取均值：标量均值、reqs 明细取 decode 居中的一次、all_ok 多数决。"""

    def _rep(self, decode, all_ok=True, req=0):
        return {
            "all_ok": all_ok,
            "decode_tok_s": decode,
            "ttft_s": 1.0, "ttft_net_s": 0.9,
            "prefill_tok_s": 100.0, "prefill_net_tok_s": 110.0,
            "prompt_tokens": 4000, "out_tokens": 500,
            "cache_hit_tokens": 0, "reqs": [{"req": req, "err": None}],
        }

    def test_mean_of_reps(self):
        reps = [self._rep(50, req=0), self._rep(10, req=1), self._rep(30, req=2)]
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["decode_tok_s"], 30)     # (50+10+30)/3 均值
        self.assertTrue(agg["all_ok"])
        self.assertEqual(agg["n_reps"], 3)
        # reqs 明细取 decode 居中的一次作代表样本：按 decode 升序 [10,30,50]
        # 取 len(pool)//2=1 → decode=30 那次（req=2），不能是任意一次
        self.assertEqual(agg["reqs"][0]["req"], 2)

    def test_mean_of_even_reps(self):
        """偶数次复测：均值（非中位）——[50,10] 均值 30。"""
        agg = _aggregate_reps([self._rep(50), self._rep(10)])
        self.assertEqual(agg["decode_tok_s"], 30.0)

    def test_majority_rule(self):
        ok = [self._rep(50), self._rep(10)]
        bad = self._rep(5, all_ok=False)
        self.assertTrue(_aggregate_reps(ok + [bad])["all_ok"])       # 2/3 多数
        self.assertFalse(_aggregate_reps([ok[0], bad, bad])["all_ok"])  # 1/3 少数

    def test_even_tie_is_failure(self):
        """all_ok 严格过半（>）：偶数平票判失败——repeats=2 时 1 成 1 败不算成功点。"""
        tie = _aggregate_reps([self._rep(50), self._rep(10, all_ok=False)])
        self.assertFalse(tie["all_ok"])
        self.assertTrue(_aggregate_reps(
            [self._rep(50), self._rep(10), self._rep(5, all_ok=False)])["all_ok"])

    def test_max_gap_takes_mean_across_reps(self):
        """停滞诊断 max_gap_s 取各次复测均值（与标量同口径）；无停滞不产出该键。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0]["max_gap_s"], reps[1]["max_gap_s"], reps[2]["max_gap_s"] = 1.5, 7.0, 2.0
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["max_gap_s"], 3.5)              # 各次均值，非最差 7.0
        self.assertNotIn("max_gap_s", _aggregate_reps([self._rep(1), self._rep(2)]))

    def test_stall_takes_mean_across_reps(self):
        """停滞复合记账（stall_s/stall_count）与 max_gap_s 同取各次复测均值
        （在出现停滞的复测之间）。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0].update(stall_s=1.2, stall_count=2)
        reps[1].update(stall_s=6.5, stall_count=1)
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["stall_s"], 3.85)              # (1.2+6.5)/2 均值
        self.assertEqual(agg["stall_count"], 2)             # 次数均值取整 (2+1)/2→2
        clean = _aggregate_reps([self._rep(1), self._rep(2)])
        self.assertFalse(clean.get("stall_s"))
        self.assertFalse(clean.get("decode_burst"))

    def test_burst_majority_across_reps(self):
        """突发交付多数决：过半复测突发才标记；且不被居中次样本残留污染。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0]["decode_burst"] = reps[1]["decode_burst"] = True
        self.assertTrue(_aggregate_reps(reps)["decode_burst"])      # 2/3 多数
        reps[1]["decode_burst"] = None
        self.assertFalse(_aggregate_reps(reps)["decode_burst"])     # 1/3 少数
        # 居中次本身带 True 但多数不成立时，显式覆写为否
        only_mid = [self._rep(10), self._rep(20), self._rep(30)]
        only_mid[1]["decode_burst"] = True
        self.assertFalse(_aggregate_reps(only_mid)["decode_burst"])

    def test_tok_per_chunk_mean_and_reps_summary(self):
        """平均每事件 token 数（tok_per_chunk）：聚合取各次复测均值、2 位小数，
        并进入复测摘要 reps 字段清单；旧口径复测无此字段时不产出该键（前端
        按空值回退）。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0]["tok_per_chunk"] = 4.0
        reps[1]["tok_per_chunk"] = 2.25
        reps[2]["tok_per_chunk"] = 4.0
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["tok_per_chunk"], 3.42)   # (4+2.25+4)/3 均值
        self.assertEqual(agg["reps"][1]["tok_per_chunk"], 2.25)   # 摘要逐次保真
        self.assertNotIn("tok_per_chunk",
                         _aggregate_reps([self._rep(1), self._rep(2)]))

    def test_decode_burst_detection(self):
        """突发甄别：亚毫秒平均间隔 + 足够 chunk 数 → 缓冲冲刷；真流式不误判。"""
        self.assertTrue(_decode_burst(32, 0.01))        # 32 chunk / 10ms → 0.3ms
        self.assertFalse(_decode_burst(32, 0.5))        # 16ms 间隔，真流式
        self.assertFalse(_decode_burst(3, 0.001))       # chunk 太少不足为凭
        self.assertFalse(_decode_burst(32, None))       # 无 decode 窗口
        self.assertFalse(_decode_burst(1, 0.001))       # 除零保护

    def test_all_failed_pool_falls_back_to_reps(self):
        """all_ok 全 False：聚合池回退为 reps 本身，标量均值照常算，all_ok=False。"""
        reps = [self._rep(50, all_ok=False), self._rep(10, all_ok=False),
                self._rep(30, all_ok=False)]
        agg = _aggregate_reps(reps)
        self.assertFalse(agg["all_ok"])
        self.assertEqual(agg["decode_tok_s"], 30)            # 回退池的均值
        self.assertEqual(agg["n_reps"], 3)

    def test_decode_adj_in_mean(self):
        """decode_tok_s_adj（空窗校正口径）参与聚合均值；缺该值的复测不计入。"""
        reps = [self._rep(50), self._rep(10), self._rep(30)]
        reps[0]["decode_tok_s_adj"] = 60.0
        reps[1]["decode_tok_s_adj"] = 10.0
        reps[2]["decode_tok_s_adj"] = 40.0
        agg = _aggregate_reps(reps)
        self.assertAlmostEqual(agg["decode_tok_s_adj"], 36.7, places=1)   # 均值
        reps[2]["decode_tok_s_adj"] = None                   # 该次无校正口径
        agg2 = _aggregate_reps(reps)
        self.assertEqual(agg2["decode_tok_s_adj"], 35.0)     # 仅 [60,10] 均值


class TestInterruptible(unittest.IsolatedAsyncioTestCase):
    """_interruptible：探测/RTT 基线等不经 _run_point 任务集的路径，停止也要即时生效。"""

    async def test_normal_returns_value(self):
        run = BenchRun({})

        async def work():
            await asyncio.sleep(0.01)
            return 42

        self.assertEqual(await run._interruptible(work()), 42)

    async def test_stop_cancels_promptly(self):
        run = BenchRun({})
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def hanging():
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(run._interruptible(hanging()))
        await started.wait()
        run.stop()
        t0 = time.monotonic()
        self.assertIsNone(await asyncio.wait_for(task, timeout=5))
        self.assertLess(time.monotonic() - t0, 2.0,
                        "停止后未及时返回（0.2s 轮询应远小于 2s）")
        self.assertTrue(cancelled.is_set(), "被包装的任务必须收到取消")


class TestClassifyHttpError(unittest.TestCase):
    """ADR-0008/0004 错误语义化。"""

    def test_gateway_timeout(self):
        for code in (502, 503, 504):
            msg = _classify_http_error(code, "bad gateway")
            self.assertIn("网关", msg)

    def test_context_exceeded(self):
        self.assertIn("上下文窗口", _classify_http_error(
            400, '{"error":{"message":"context length exceeds limit"}}'))
        # Kimi 用 401 报超窗（ADR-0004）
        self.assertIn("上下文窗口", _classify_http_error(
            401, "invalid_authentication_error: input context exceeds the only allowed"))

    def test_plain_error_passthrough(self):
        msg = _classify_http_error(400, "unknown parameter: xyz")
        self.assertIn("400", msg)
        self.assertIn("xyz", msg)


class TestCtxOverflowRetry(unittest.TestCase):
    """超窗回退（删减上下文重试）的判定与删减辅助。"""

    def test_overflow_detection(self):
        # OpenAI 系：code=context_length_exceeded；message 无 exceed 字样
        self.assertTrue(_is_ctx_overflow(
            400, '{"error":{"message":"This model\'s maximum context length is 8192 '
                 'tokens. However, you requested 9000 tokens",'
                 '"code":"context_length_exceeded"}}'))
        # llama.cpp 系
        self.assertTrue(_is_ctx_overflow(
            400, "the request exceeds the available context size"))
        # 普通 400 不误判
        self.assertFalse(_is_ctx_overflow(400, "unknown parameter: thinking"))
        self.assertFalse(_is_ctx_overflow(500, "internal error"))

    def test_trim_middle_keeps_head_and_tail(self):
        msgs, _, filler_n = build_messages("creative", 4096, 1.5, "nonce-9")
        content = msgs[1]["content"]
        keep = int(len(content) * 0.75)
        trimmed = _trim_middle(content, keep)
        self.assertLessEqual(len(trimmed), keep + 20)   # 删减标记仅十余字符
        self.assertTrue(trimmed.startswith(content[:100]))     # 编号行保留
        self.assertTrue(trimmed.endswith(content[-100:]))      # 任务指令保留
        self.assertIn("已删减", trimmed)
        # keep ≥ 原长时不动
        self.assertEqual(_trim_middle(content, len(content) + 1), content)

    def test_soft_overflow_detection(self):
        """fastllm 系软超窗占位回复：整体输出即 "prompt too long"（≤32 字符），
        content 与 reasoning_content 两个通道的头部文本都参与判定。"""
        self.assertTrue(_is_soft_ctx_overflow("prompt too long", 15))
        self.assertTrue(_is_soft_ctx_overflow("  Prompt Too Long\n", 18))  # 大小写/空白
        self.assertTrue(_is_soft_ctx_overflow("prompt too long", 32))      # 边界长度
        # reasoning 通道：content 头为空/非占位串，思考流头部命中同样识别
        self.assertTrue(_is_soft_ctx_overflow("", 15, "prompt too long"))
        self.assertTrue(_is_soft_ctx_overflow("你好。", 18, "  Prompt Too Long "))
        # 正常输出不误判：前缀相同但有后续内容 / 超长
        self.assertFalse(_is_soft_ctx_overflow("prompt too long, please retry", 29))
        self.assertFalse(_is_soft_ctx_overflow("prompt too long" + "x" * 20, 35))
        self.assertFalse(_is_soft_ctx_overflow("Prompt is too long to process", 29))
        self.assertFalse(_is_soft_ctx_overflow("", 0))
        # reasoning 通道非占位串不误判；超长总量直接短路
        self.assertFalse(_is_soft_ctx_overflow("你好。", 18, "首先分析问题"))
        self.assertFalse(_is_soft_ctx_overflow("", 40, "prompt too long"))

    def test_collapsed_reply_guard(self):
        """输出塌缩守卫（通用保险）：finish=stop 但输出近乎为空而 prompt 很大
        ——占位软拒绝的占位串未必命中已知文案。阈值卡在大 prompt 上，小
        prompt 的合法短输出不误伤。"""
        # 典型占位软拒绝（未知文案、单 token）：判塌缩
        self.assertTrue(_is_collapsed_reply(277867, 1, "stop", 15))
        self.assertTrue(_is_collapsed_reply(8192, 2, "stop", 32))     # 边界
        # 合法短回复不误伤：小 prompt / 输出超阈 / finish 非 stop
        self.assertFalse(_is_collapsed_reply(4096, 1, "stop", 15))    # 小 prompt
        self.assertFalse(_is_collapsed_reply(0, 1, "stop", 5))        # 0K 自由短答
        self.assertFalse(_is_collapsed_reply(20000, 3, "stop", 15))   # 输出超阈
        self.assertFalse(_is_collapsed_reply(20000, 1, "stop", 33))   # 字符超阈
        self.assertFalse(_is_collapsed_reply(20000, 1, "length", 15))  # 非自然收尾
        self.assertFalse(_is_collapsed_reply(None, 1, "stop", 15))    # 无 prompt 数据
        self.assertFalse(_is_collapsed_reply(20000, None, "stop", 15))

    def test_implausible_prefill_rate_gate(self):
        """物理速率闸：prefill 速率超该 ctx 先验曲线 10× 或绝对上限 10 万
        tok/s 即判「物理上不可能」；无先验时只按绝对上限判（不误伤真实快
        后端的首个请求）。"""
        # 绝对上限（无先验也生效）
        self.assertTrue(_is_implausible_prefill_rate(387516.0, 73728, []))
        self.assertTrue(_is_implausible_prefill_rate(100000.0, 4096, []))   # 边界
        # 先验倍数：5000 > 10×300 → 判；2500 < 3000 → 不判
        self.assertTrue(_is_implausible_prefill_rate(5000.0, 4096, [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(2500.0, 4096, [(2048, 300.0)]))
        # 无先验、低于绝对上限：不判
        self.assertFalse(_is_implausible_prefill_rate(5000.0, 4096, []))
        self.assertFalse(_is_implausible_prefill_rate(None, 4096, [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(0, 4096, [(2048, 300.0)]))


class TestCtxEdgeTrim(unittest.TestCase):
    """部署上限超窗的「贴边裁减」判定（_plan_ctx_edge，LLM 阶梯与媒体阶梯共用）。

    口径：excess = 输入构造目标 + 输出预算 + CTX_HEADROOM − max_ctx；
    excess ≤ 0 照常测量（现状不变）、0 < excess ≤ 10%·max_ctx 裁减贴边
    （ctx_eff = max_ctx − budget − CTX_HEADROOM）、超出量超阈值或贴边量非正
    维持跳过。"""

    MAX_CTX = 81920   # 10% = 8192 整除，便于构造精确边界
    EDGE = MAX_CTX - CTX_HEADROOM   # budget=0 时的贴边输入档

    def test_under_window_unchanged(self):
        """excess ≤ 0：按原档位测，构造量不动（现状行为）。"""
        mode, eff, excess = _plan_ctx_edge(self.EDGE, 0, self.MAX_CTX)
        self.assertEqual((mode, eff, excess), ("ok", self.EDGE, 0))
        mode, eff, excess = _plan_ctx_edge(self.EDGE - 4096, 512, self.MAX_CTX)
        self.assertEqual(mode, "ok")
        self.assertEqual(eff, self.EDGE - 4096)   # 原构造量原样返回
        # 0K 档零输入 + 常规输出预算：永不在窗上
        mode, eff, _ = _plan_ctx_edge(0, 512, self.MAX_CTX)
        self.assertEqual((mode, eff), ("ok", 0))

    def test_exactly_ten_percent_trims(self):
        """0 < excess ≤ 10%·max_ctx：裁减贴边，ctx_eff 落回
        max_ctx − budget − CTX_HEADROOM。"""
        in_tok = self.EDGE + 8192   # excess 恰好 = 10%·max_ctx（含 headroom 口径）
        mode, eff, excess = _plan_ctx_edge(in_tok, 0, self.MAX_CTX)
        self.assertEqual((mode, excess), ("edge", 8192))
        self.assertEqual(eff, self.EDGE)
        # 带输出预算的贴边：ctx_eff = max_ctx − budget − CTX_HEADROOM
        in_tok = self.EDGE - 4096 + 8192   # budget=4096 时 excess 同为 8192
        mode, eff, excess = _plan_ctx_edge(in_tok, 4096, self.MAX_CTX)
        self.assertEqual((mode, excess), ("edge", 8192))
        self.assertEqual(eff, self.MAX_CTX - 4096 - CTX_HEADROOM)

    def test_just_over_ten_percent_skips(self):
        """excess = 10%·max_ctx + 1：超出量超阈值，维持整档跳过。"""
        mode, eff, excess = _plan_ctx_edge(self.EDGE + 8193, 0, self.MAX_CTX)
        self.assertEqual((mode, eff, excess), ("skip", 0, 8193))

    def test_256k_reported_case(self):
        """用户实报场景还原：256K 部署测 256K 档，输出预算 8K——
        裁减为 ≈247K 贴边测量而非整档跳过。"""
        mode, eff, excess = _plan_ctx_edge(262144, 8192, 262144)
        self.assertEqual(mode, "edge")
        self.assertEqual(eff, 262144 - 8192 - CTX_HEADROOM)
        self.assertEqual(excess, 8192 + CTX_HEADROOM)

    def test_budget_gte_max_ctx_still_skips(self):
        """budget + CTX_HEADROOM ≥ max_ctx：贴边量非正（连零输入都放不下），
        即使超窗量在 10% 内也跳过。"""
        mode, eff, _ = _plan_ctx_edge(0, self.MAX_CTX, self.MAX_CTX)
        self.assertEqual(mode, "skip")
        mode, eff, _ = _plan_ctx_edge(100, self.MAX_CTX - CTX_HEADROOM + 10,
                                      self.MAX_CTX)
        self.assertEqual(mode, "skip")   # excess 在 10% 内但贴边量为负


class TestOutLimitGuards(unittest.TestCase):
    """ADR-0016 输出长度治理的参数面：三场景引导文案模板齐全、容差与断流
    系数固定（行为路径见 test_e2e_mock 的翻转/回退/断流用例）。"""

    def test_hint_templates_per_scenario(self):
        for key, sc in SCENARIOS.items():
            if sc.get("kind", "llm") != "llm":
                continue   # 媒体场景走 _one_media，无输出长度引导（ADR-0020）
            self.assertIn("{limit}", sc["out_hint"], f"{key} 缺输出长度引导模板")
        # 中文场景按字/字符引导，英文 agent 场景按 characters
        self.assertIn("字", SCENARIOS["creative"]["out_hint"])
        self.assertIn("字符", SCENARIOS["code"]["out_hint"])
        self.assertIn("characters", SCENARIOS["agent"]["out_hint"])

    def test_factors(self):
        self.assertEqual(OUT_HINT_FACTOR, 1.2)     # 引导上限 = 预设 ×1.2
        self.assertEqual(CLIENT_CUT_FACTOR, 2)     # 断流阈值 = 下发上限 ×2
        # 引导字数折算：512 tokens 创意档 × 1.5 字/token × 1.2 = 921 字
        self.assertEqual(int(512 * SCENARIOS["creative"]["out_cpt"]
                             * OUT_HINT_FACTOR), 921)


class TestEstOut(unittest.TestCase):
    """ADR-0019 输出估算口径：SSE 事件数优先，未校准时按 1 事件/token 直计，
    无事件才回退字符系数（英文输出 ~4.2 字符/token 按中文先验 1.5 估会超发
    ~2.8 倍，触发断流丢 usage 的恶性循环）。"""

    def _run(self):
        r = BenchRun.__new__(BenchRun)
        r.chunk_calib = {}
        return r

    def test_chunk_first_uncalibrated(self):
        # 未校准：100 事件直计 100（字符数 430 不参与），英文形态不虚发
        self.assertEqual(self._run()._est_out("m", "creative", 100, 430, 1.5), 100.0)

    def test_calibrated_ratio(self):
        r = self._run()
        r.chunk_calib[("m", "creative")] = 2.0   # 实测 2 事件/token
        self.assertEqual(r._est_out("m", "creative", 100, 430, 1.5), 50.0)

    def test_no_chunk_falls_back_to_chars(self):
        # 零事件（如无正文纯思考被剥离等）才回退字符系数
        self.assertEqual(self._run()._est_out("m", "creative", 0, 30, 1.5), 20.0)


class TestCalibHelpers(unittest.TestCase):
    """校准/实时读数健壮性的纯函数口径：加权滑动平均（chunk_calib 消除
    last-write-wins 覆写传染）与滑窗基准样本的校准比例折算（消除比例切换
    瞬间的差分口径错配）。"""

    def test_wma_no_old_value_seeds(self):
        self.assertEqual(_wma(None, 0.0, 2.0, 100.0), 2.0)

    def test_wma_blends_by_weight(self):
        self.assertAlmostEqual(_wma(1.0, 100.0, 3.0, 100.0), 2.0)   # 等权取中
        # 新观测权重小 → 只小幅拉动，单次观测不翻转口径
        self.assertAlmostEqual(_wma(1.0, 1000.0, 3.0, 100.0), 1300.0 / 1100.0)

    def test_wma_matches_cpt_calib_legacy_formula(self):
        # 与 cpt_calib 原式 ((old or obs) * w + obs * w_new) / (w + w_new) 等价
        for old, w, obs, w_new in [(2.0, 500.0, 2.4, 800.0), (0.2, 60000.0, 0.3, 900.0)]:
            legacy = (old * w + obs * w_new) / (w + w_new)
            self.assertAlmostEqual(_wma(old, w, obs, w_new), legacy)

    def test_ratio_refold_unchanged_when_same(self):
        self.assertEqual(_ratio_refold(50.0, 2.0, 2.0), 50.0)
        self.assertEqual(_ratio_refold(50.0, None, None), 50.0)

    def test_ratio_refold_ratio_changed(self):
        # 基准 100 事件按旧比 0.5 估 200 tok；切到新比 0.25 下应为 400 tok
        self.assertAlmostEqual(_ratio_refold(200.0, 0.5, 0.25), 400.0)
        self.assertAlmostEqual(_ratio_refold(400.0, 0.25, 0.5), 200.0)   # 反向

    def test_ratio_refold_uncalibrated_boundary(self):
        # 基准未校准（事件数直计）→ 首次播种后按新比折算，差分不爆尖峰
        self.assertAlmostEqual(_ratio_refold(100.0, None, 0.25), 400.0)
        # 基准已校准、新样本未校准（理论路径）：还原事件数直计口径
        self.assertAlmostEqual(_ratio_refold(400.0, 0.25, None), 100.0)


class TestMediaCorpus(unittest.TestCase):
    """ADR-0020 媒体语料：合成/解析往返、时长精确性、用户语料优先与确定性。"""

    def test_synth_wav_roundtrip(self):
        wav = _synth_speech_wav(5)
        self.assertAlmostEqual(_wav_seconds(wav), 5.0, places=2)
        wav30 = _synth_speech_wav(30)
        self.assertAlmostEqual(_wav_seconds(wav30), 30.0, places=2)
        # 非 wav 输入返回 None 而不是抛错
        self.assertIsNone(_wav_seconds(b"not a wav"))

    def test_asr_audio_cached_and_deterministic(self):
        bench._AUDIO_CACHE.clear()
        a1, s1 = _asr_audio(5)
        a2, s2 = _asr_audio(5)
        self.assertIs(a1, a2, "同档音频须缓存复用（阶梯/并发共享同一份负载）")
        self.assertAlmostEqual(s1, 5.0, places=2)
        self.assertAlmostEqual(s2, s1)

    def test_asr_audio_prefers_user_corpus(self):
        """corpus/asr/*.wav 存在时循环用户音频到目标时长（帧级精确）。"""
        sr = 16000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"\x01\x00" * sr * 3)   # 3s 用户音频
        with tempfile.TemporaryDirectory() as td:   # 独立临时目录，并行/残留互不干扰
            with unittest.mock.patch.object(bench, "CORPUS_ASR_DIR", td):
                with open(os.path.join(td, "clip.wav"), "wb") as f:
                    f.write(buf.getvalue())
                try:
                    bench._AUDIO_CACHE.clear()
                    audio, actual = _asr_audio(7)   # 3s 循环到 7s
                    self.assertAlmostEqual(actual, 7.0, places=2)
                    self.assertAlmostEqual(_wav_seconds(audio), 7.0, places=2)
                finally:
                    bench._AUDIO_CACHE.clear()

    def test_synth_png_roundtrip_and_size(self):
        png = _synth_document_png(1240, 1754, seed=0)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(_image_size(png), (1240, 1754))
        # 确定性：同 seed 同图
        self.assertEqual(png, _synth_document_png(1240, 1754, seed=0))

    def test_image_size_jpeg_sof(self):
        """JPEG SOF0 尺寸解析（构造最小头）。"""
        sof = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"\x00" * 14
               + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08"
               + struct.pack(">HH", 480, 640) + b"\x00" * 6)
        self.assertEqual(_image_size(sof), (640, 480))
        self.assertIsNone(_image_size(b"\x89PNG garbage"))

    def test_ocr_pool_fallback_three_resolutions(self):
        bench._IMG_CACHE = None
        pool = _ocr_images()
        self.assertGreaterEqual(len(pool), 3)
        for data, w, h, mime in pool:
            self.assertEqual(_image_size(data), (w, h))
            self.assertTrue(mime.startswith("image/"))
        bench._IMG_CACHE = None

    def test_ocr_messages_shape(self):
        msgs, img_tokens = _build_ocr_messages(4, nonce=7)
        self.assertEqual(msgs[0]["role"], "system")
        content = msgs[1]["content"]
        self.assertEqual(len(content), 5)   # 4 图 + 1 文本指令
        self.assertTrue(all(p["type"] == "image_url" for p in content[:4]))
        self.assertTrue(content[4]["text"].startswith("请识别"))
        self.assertGreater(img_tokens, 0)
        # 同 nonce 确定：不同 nonce 错图（bust 缓存）
        msgs2, _ = _build_ocr_messages(4, nonce=8)
        self.assertNotEqual(content[0]["image_url"]["url"],
                            msgs2[1]["content"][0]["image_url"]["url"])

    def test_tts_text_length_and_nonce(self):
        t1 = _tts_text(200, "a")
        self.assertGreaterEqual(len(t1), 195)   # 编号行 + 语料截取（略超可接受）
        self.assertIn("a", t1[:12])
        t2 = _tts_text(200, "b")
        self.assertNotEqual(t1[:16], t2[:16], "编号行使请求文本错开")


class TestMediaAggregate(unittest.TestCase):
    """_aggregate_media_reqs：净口径均值、吞吐只在全成功并发点、audio_est 标注。"""

    def _reqs(self, n=2, err_last=False):
        reqs = [{"req": i, "audio_s": 10.0, "total_s": 0.5, "elapsed_net_s": 0.4,
                 "rtf": 0.04, "speed_x": 25.0, "out_chars": 100,
                 "ttfa_s": 0.05, "out_bytes": 16044} for i in range(n)]
        if err_last:
            reqs[-1] = {"req": n - 1, "err": "boom"}
        return reqs

    def test_asr_point(self):
        out = _aggregate_media_reqs("asr", self._reqs(), batch_time=0.55, conc=2)
        self.assertAlmostEqual(out["rtf"], 0.04, places=4)
        self.assertAlmostEqual(out["speed_x"], 25.0, places=1)
        # 吞吐 = 总音频 20s / 批窗口 0.55s = 36.36 音频秒/秒 = 0.61 音频分钟/分钟
        self.assertAlmostEqual(out["audio_min_per_min"], 20 / 0.55 / 60, places=2)
        self.assertEqual(out["out_chars"], 100)
        self.assertNotIn("ttfa_s", out, "asr 不产出 TTFA（非流式）")

    def test_tts_point_and_audio_est(self):
        reqs = self._reqs()
        reqs[0]["audio_est"] = True
        out = _aggregate_media_reqs("tts", reqs, batch_time=0.55, conc=2)
        self.assertAlmostEqual(out["ttfa_s"], 0.05, places=3)
        self.assertTrue(out["audio_est"], "任一请求时长为估算即标注")
        self.assertNotIn("out_chars", out, "tts 不产出识别字符数")

    def test_partial_failure_no_throughput(self):
        """个别请求失败：均值只取成功请求，但吞吐置空（分母被摊薄无意义）。"""
        out = _aggregate_media_reqs("asr", self._reqs(err_last=True),
                                    batch_time=0.55, conc=2)
        self.assertAlmostEqual(out["speed_x"], 25.0, places=1)
        self.assertNotIn("audio_min_per_min", out)

    def test_all_failed_empty(self):
        self.assertEqual(_aggregate_media_reqs(
            "asr", [{"req": 0, "err": "x"}], batch_time=0.1, conc=1), {})


class TestTotalDecodeRates(unittest.TestCase):
    """_total_decode_rates：并发组总吞吐 raw/adj 双口径。adj 为空窗校正——
    窗口扣除各请求停滞累计（stall_s，首 token 后块间空窗）；首 token 前的
    prefill 等待不在 stall 内，由重叠守卫兜底置空。用例数据取自真实存档
    （Qwen3.8-27B / code / 并发 2）实测四个档位。"""

    def _req(self, first, last, out=512, stall=None):
        return {"req": 0, "first_abs": first, "last_abs": last,
                "out_tokens": out, "stall_s": stall}

    def test_stall_case_32k(self):
        """32K 案例：req1 stall 27.89s 的窗口（44.12s）混入 req0 的 prefill
        等待 → raw 23.2 失真；校正窗口 44.12−27.89=16.23 → adj ≈ 63.1。"""
        ok = [self._req(96929.32, 96945.12),
              self._req(96901.00, 96942.61, stall=27.89)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 23.2, delta=0.05)
        self.assertAlmostEqual(adj, 63.1, delta=0.05)

    def test_stall_case_4k(self):
        """4K 案例（轻度停滞）：raw ≈84.3，校正窗口 12.14−2.75=9.39 →
        adj ≈109（背景口径 ≈109.0~109.1）。"""
        ok = [self._req(96812.71, 96824.45, stall=2.75),
              self._req(96815.46, 96824.85)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 84.3, delta=0.05)
        self.assertAlmostEqual(adj, 109.05, delta=0.1)

    def test_stall_case_64k(self):
        """64K 案例（重度停滞 70.64s）：raw ≈10.3 失真，校正窗口 99.83−70.64
        =29.19 → adj ≈35.1。"""
        ok = [self._req(97005.99, 97100.85, stall=70.64),
              self._req(97076.63, 97105.82)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 10.3, delta=0.05)
        self.assertAlmostEqual(adj, 35.1, delta=0.05)

    def test_zero_overlap_serialized_128k(self):
        """128K 案例：KV 压力下 decode 区间完全串行（req1 last < req0 first，
        stall_s 全空）——重叠守卫兜底，raw 与 adj 双双置空。"""
        ok = [self._req(97303.35, 97330.00),
              self._req(97180.00, 97216.05)]
        self.assertEqual(_total_decode_rates(ok, 2), (None, None))

    def test_no_stall_adj_equals_raw(self):
        """无停滞组：校正窗口 == 毛窗口，adj == raw。"""
        ok = [self._req(100.0, 110.0), self._req(101.0, 112.0)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertEqual(adj, raw)
        self.assertAlmostEqual(raw, 1024 / 12.0, delta=0.05)

    def test_marginal_overlap_guard(self):
        """重叠不足 0.5×最短 decode：raw 与 adj 一起置空（守卫共享）。"""
        # dec 长度 [10, 1]，overlap = 0.4 < 0.5×1
        ok = [self._req(0.0, 10.0), self._req(9.6, 10.6)]
        self.assertEqual(_total_decode_rates(ok, 2), (None, None))

    def test_eff_window_nonpositive_adj_none(self):
        """停滞累计 ≥ 窗口本身：eff_window ≤ 0 → 仅 adj 置空，raw 照常产出。"""
        ok = [self._req(0.0, 10.0, stall=12.0), self._req(1.0, 11.0)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 1024 / 11.0, delta=0.05)   # 窗口 0→11
        self.assertIsNone(adj)

    def test_incomplete_guards(self):
        """守卫：ok 数不足 / 零输出 / firsts/lasts 缺失 / 窗口非正 → 双置空。"""
        r = self._req(0.0, 10.0)
        self.assertEqual(_total_decode_rates([r], 2), (None, None))   # 数不足
        self.assertEqual(_total_decode_rates(
            [self._req(0.0, 10.0, out=0), self._req(1.0, 11.0, out=0)], 2),
            (None, None))                                             # 零输出
        r1 = self._req(0.0, 10.0)
        del r1["first_abs"]
        self.assertEqual(_total_decode_rates([r1, self._req(1.0, 11.0)], 2),
                         (None, None))                                # firsts 缺失
        self.assertEqual(_total_decode_rates(
            [self._req(5.0, 5.0), self._req(5.0, 5.0)], 2), (None, None))  # 零窗口

    def test_aggregate_reps_keeps_adj(self):
        """复测聚合：decode_total_tok_s_adj 取均值并进入复测摘要字段清单；
        缺该值的复测不计入。"""
        def rep(total):
            return {"all_ok": True, "decode_tok_s": 50.0, "ttft_s": 1.0,
                    "prompt_tokens": 4000, "out_tokens": 500,
                    "decode_total_tok_s": total, "reqs": [{"req": 0}]}
        reps = [rep(100.0), rep(60.0), rep(80.0)]
        reps[0]["decode_total_tok_s_adj"] = 109.0
        reps[1]["decode_total_tok_s_adj"] = 63.1
        agg = _aggregate_reps(reps)
        self.assertAlmostEqual(agg["decode_total_tok_s"], 80.0, delta=0.05)
        self.assertAlmostEqual(agg["decode_total_tok_s_adj"], 86.05, delta=0.05)
        self.assertEqual(agg["reps"][1]["decode_total_tok_s_adj"], 63.1)
        self.assertNotIn("decode_total_tok_s_adj",
                         _aggregate_reps([rep(1.0), rep(2.0)]))


class TestCorrectFillerChars(unittest.TestCase):
    """构建偏离校正的填充字符数学：只动 filler，固定开销（指令/nonce/echo
    预填等）不参与收缩/扩张。"""

    def test_nested_kb_measured_marginal(self):
        """有嵌套记账先验：密度取本次首测的增量段实测边际——
        d = (9280−3000)/(4915−2048)，f_new = 9280 + (4096−4915)×d = 7486。"""
        f = _correct_filler_chars(9280, 4915.0, 4096, (3000.0, 2048.0),
                                  300, 3.0)
        self.assertEqual(f, 7486)

    def test_no_kb_overall_density(self):
        """无先验（首个非零档）：d = filler/(real − 开销 tokens)——
        1000/(1200−100)，f_new = 1000 + (1000−1200)×0.909 = 818。"""
        f = _correct_filler_chars(1000, 1200.0, 1000, None, 100, 1.0)
        self.assertEqual(f, 818)

    def test_underbuild_expands_filler(self):
        """实测偏小（构建不足）：等比扩张 filler——
        d = 800/700，f_new = 800 + (900−700)×1.1429 = 1029。"""
        f = _correct_filler_chars(800, 700.0, 900, None, 0, 1.0)
        self.assertEqual(f, 1029)

    def test_density_out_of_range_none(self):
        """密度越界（<0.3 字符/token，与 _calibrate_usage 增量观测钳制同界）：
        不重测。"""
        self.assertIsNone(_correct_filler_chars(5, 3000.0, 2000, None, 0, 1.0))

    def test_negative_filler_none(self):
        """开销 tokens 超过目标档（校正要收缩的量超过整个 filler）：不重测。"""
        self.assertIsNone(_correct_filler_chars(100, 600.0, 50, None, 500, 1.0))

    def test_no_change_none(self):
        """校正量不产生变化：不重测（防御，正常只在偏离 >10% 时调用）。"""
        self.assertIsNone(_correct_filler_chars(500, 500.0, 500, None, 0, 1.0))


class TestCtxDeviationRetest(unittest.IsolatedAsyncioTestCase):
    """上下文构建偏离 >CTX_DEV_TOLERANCE 时按实测密度校正重测：_one 打桩
    控制实测 prompt_tokens（按消息字符数 ÷ 固定密度建模真实分词），校准打桩
    计数。触发/不触发边界、只重测一次、重测后点数据替换、删减点位不触发。"""

    MODEL, SCEN = "m", "creative"

    def _make_run(self, results, cache="bust"):
        """results: 每次请求的覆写项；prompt_tokens 可为 callable(msgs)——
        按字符数/固定密度建模真实分词密度。返回 (run, req_calls, calib_calls,
        statuses)。"""
        run = BenchRun({"cache": cache})
        state = {"n": 0}
        req_calls, calib_calls, statuses = [], [], []
        base = {"err": None, "first_abs": 100.0, "last_abs": 101.0,
                "out_tokens": 100, "stall_s": None, "stall_count": None,
                "max_gap_s": None, "decode_burst": None,
                "decode_tok_s": 100.0, "decode_tok_s_adj": None,
                "tok_per_chunk": 1.0, "ttft_s": 0.1, "ttft_net_s": 0.09,
                "prefill_tok_s": 1000.0, "prefill_net_tok_s": 1100.0,
                "cache_hit": 0, "finish": "stop", "usage_real": True,
                "sent_chars": 10 ** 9, "text_chars": 0, "reason_chars": 0,
                "total_s": 1.0, "cache_reported": False, "cache_miss": 0}

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False,
                           est_tokens=None, img_tokens=0):
            i = min(state["n"], len(results) - 1)
            state["n"] += 1
            r = dict(base)
            r.update(results[i])
            r["req"] = req_i
            r["_msgs"] = msgs
            pt = r.get("prompt_tokens")
            r["prompt_tokens"] = (pt(msgs) if callable(pt) else pt)
            req_calls.append(r)
            return r

        async def fake_calib(key, chars, filler_n, ok):
            calib_calls.append((chars, filler_n))

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run._one = fake_one
        run._calibrate_usage = fake_calib
        run.emit = fake_emit
        return run, req_calls, calib_calls, statuses

    @staticmethod
    def _density(chars_per_token: float):
        return lambda msgs: round(
            sum(len(m["content"]) for m in msgs) / chars_per_token)

    @staticmethod
    def _nonce(msgs):
        """从构造的 user 消息头部提取 nonce 串（build_messages 把 nonce 放在
        user content 开头的「参考材料（编号 {nonce}）：」行内）。"""
        head = msgs[1]["content"]
        m = re.match(r"参考材料（编号 (\d+)）：", head)
        return m.group(1) if m else None

    async def test_deviation_triggers_and_lands_on_target(self):
        """嵌套记账下新段密度失真（实测/目标 +20%）：触发重测、校正 filler
        按实测密度收缩、重测后实测命中目标 ±5%、点数据整体替换、校准两次
        执行、ctx_retry 留痕、status 播报。"""
        key = (self.MODEL, self.SCEN)
        run, reqs, calibs, statuses = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        run.prefix_kb[key] = (2000.0, 1500.0)   # 前档记账 → 走嵌套构造路径
        run.cpt_marginal[key] = 2.0             # 滑动边际估计（对新段失真）
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 2, "偏离 >10% 应触发整批重测")
        self.assertEqual(len(calibs), 2, "两次尝试都要入校准")
        first_real = reqs[0]["prompt_tokens"]
        self.assertGreater(first_real, 4096 * 1.1, "打桩密度应产出 >10% 偏离")
        self.assertEqual(point["ctx_retry"], first_real, "留痕首次实测均值")
        self.assertEqual(point["prompt_tokens"], reqs[1]["prompt_tokens"])
        self.assertLess(point["prompt_tokens"], 4096 * 1.05,
                        f"校正重测后应命中目标 ±5%（实测 {point['prompt_tokens']}）")
        self.assertGreater(point["prompt_tokens"], 4096 * 0.95)
        # 校正只动 filler：实测偏大 → 重测构造总字符量应收缩（重测换新 nonce，
        # 仅头部编号数字与 filler 变化，指令/系统等固定开销不变，字符量差
        # 几乎全部来自 filler 校正）
        c1 = calibs[0][0]
        c2 = calibs[1][0]
        self.assertLess(c2, c1, "实测偏大 → 重测构造总字符量应收缩")
        # 点级数据整体替换：reqs 是重测批、派生字段取重测值
        self.assertEqual(point["reqs"][0]["prompt_tokens"],
                         reqs[1]["prompt_tokens"])
        # 重测换新 nonce（防前缀缓存命中共享前缀），user 消息头部编号不同
        self.assertNotEqual(self._nonce(reqs[1]["_msgs"]),
                            self._nonce(reqs[0]["_msgs"]),
                            "重测 nonce 必须与首测不同（bust 前缀缓存）")
        self.assertTrue(any("校正重测" in s for s in statuses),
                        f"应有校正重测播报: {statuses}")

    async def test_retry_msgs_prefix_busted(self):
        """回归：重测 msgs 与首测 msgs 不共享可缓存前缀——nonce 在 user
        消息开头，nonce 串不同 → 两条 user 内容从头部编号处即分叉，服务端
        前缀缓存（命中共享前缀而非整条内容）无法把首测缓存带给重测。
        （真实案例：同 nonce 重测 32K 档 cache_hit 30720/32170，TTFT 失真）"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        u1 = reqs[0]["_msgs"][1]["content"]
        u2 = reqs[1]["_msgs"][1]["content"]
        self.assertNotEqual(self._nonce(reqs[0]["_msgs"]),
                            self._nonce(reqs[1]["_msgs"]))
        # 两条 user 内容的公共前缀必须止于头部编号行内（换 nonce 即整段
        # bust），而非共享到 filler 段
        common = os.path.commonprefix([u1, u2])
        self.assertLess(len(common), u1.index("：") + 1,
                        "重测与首测的 user 内容应在头部编号处分叉")

    async def test_within_tolerance_no_retest(self):
        """偏离 ≤10%：不触发，单次请求、ctx_retry 为 None。"""
        run, reqs, calibs, statuses = self._make_run(
            [{"prompt_tokens": self._density(1.8)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(calibs), 1)
        self.assertIsNone(point["ctx_retry"])
        self.assertFalse(any("校正重测" in s for s in statuses))

    async def test_stable_cache_no_retest(self):
        """cache=stable：重测会命中前缀缓存污染读数，跳过。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)}], cache="stable")
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(calibs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_zero_ctx_no_retest(self):
        """ctx=0 档（固定开销主导、无填充）：不触发。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 0, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_no_usage_no_retest(self):
        """usage_real=False（无真实 usage 回传）：无密度样本，不触发。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": 9999, "usage_real": False}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(calibs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_retests_only_once(self):
        """重测后仍超差：不再重试（最多 1 次），接受并留 ctx_retry 痕。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 2, "只重测一次")
        self.assertEqual(len(calibs), 2)
        self.assertEqual(point["ctx_retry"], reqs[0]["prompt_tokens"])

    async def test_trimmed_point_no_retest(self):
        """删减重试成功点位（sent_chars < 构造量）：偏离源于服务端超窗保护，
        重测必然复现，不触发。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5), "sent_chars": 100}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_stop_flag_no_retest(self):
        """stop_flag 置位（首批在途期间停止）：不再发起重测批。"""

        def stop_during_call(msgs):
            run.stop_flag = True   # 模拟首批在途期间置停
            return 99999

        run, reqs, calibs, _ = self._make_run([{"prompt_tokens": stop_during_call}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1, "首批照常收数")
        self.assertEqual(len(calibs), 1, "只有首批校准，无重测批")
        self.assertIsNone(point["ctx_retry"])

    async def test_conc2_batch_retested_together(self):
        """并发 2：整批（conc 个请求）重跑，每个槽位的重测 nonce 与该槽位
        首测 nonce 不同（防前缀缓存污染）、批内槽位间也不重合，替换点数据。"""
        run, reqs, calibs, statuses = self._make_run(
            [{"prompt_tokens": self._density(1.5)}] * 2)
        point = await run._run_point(None, self.MODEL, self.SCEN, 8192, 2, 1.8)
        self.assertEqual(len(reqs), 4, "首批 2 + 重测批 2")
        self.assertEqual(len(calibs), 2)
        self.assertEqual(point["reqs"][0]["prompt_tokens"],
                         reqs[2]["prompt_tokens"])
        self.assertEqual([r["req"] for r in point["reqs"]], [0, 1],
                         "重测批按 req 序号替换原批")
        first_nonces = [self._nonce(r["_msgs"]) for r in reqs[:2]]
        retry_nonces = [self._nonce(r["_msgs"]) for r in reqs[2:]]
        self.assertNotEqual(first_nonces[0], retry_nonces[0],
                            "槽位 0 重测 nonce 应与首测不同")
        self.assertNotEqual(first_nonces[1], retry_nonces[1],
                            "槽位 1 重测 nonce 应与首测不同")
        self.assertEqual(len(set(retry_nonces)), len(retry_nonces),
                         "重测批内各槽位 nonce 互不相同")


class TestDegenerateText(unittest.TestCase):
    """_is_degenerate_text：头部采样退化重复检测（实测三样本 + 正常文本
    不误伤 + 短串保护）。"""

    # 三个真实退化样本（fastllm 部署 ≥8K 档思考流，reason_chars≈512）
    S_BRACE = "}\n" * 60                       # ① "}" 换行循环：2 种字符
    S_NUM = "        " + " ".join(f"{i}." for i in range(6, 60))  # ② 编号列表
    S_NEST = "".join(" " * i + "if (\n" for i in range(20))       # ③ 嵌套递增

    def test_real_samples_true(self):
        for s in (self.S_BRACE, self.S_NUM, self.S_NEST):
            self.assertTrue(_is_degenerate_text(s),
                            f"实测退化样本应命中: {s[:40]!r}")

    def test_normal_chinese_false(self):
        s = ("大语言模型的推理速度受显存带宽与批调度策略共同影响，长上下文档位的"
             "预填充读数尤其依赖服务端的 KV 缓存实现与逐出策略，短文档位则更多"
             "受网络往返与调度排队左右，逐请求对比才能看出真实差异。")
        self.assertGreater(len(s.strip()), 80)
        self.assertFalse(_is_degenerate_text(s))

    def test_normal_code_false(self):
        s = ('def _run_point(self, client, model, scenario, ctx, conc, cpt, rep):\n'
             '    tasks = [asyncio.create_task(self._one(client, model, m)) for m in msgs]\n'
             '    reqs, batch_time = await self._await_batch(tasks)\n')
        self.assertGreater(len(s.strip()), 80)
        self.assertFalse(_is_degenerate_text(s))

    def test_short_false(self):
        self.assertFalse(_is_degenerate_text("}\n" * 20))   # 40 字符 < 80 下限
        self.assertFalse(_is_degenerate_text("好的。"))


class TestDetectRepAnomaly(unittest.TestCase):
    """_detect_rep_anomaly：early_stop 仅 echo 模式启用（free 提前停止合法）、
    0.8×max_tokens 阈值、finish=length 不判、全 err 不判、退化重复两模式
    通用且 text/reason 通道任一命中。"""

    def _req(self, **over):
        r = {"err": None, "finish": "stop", "out_tokens": 512,
             "ttft_s": 0.1, "decode_tok_s": 50.0,
             "text_sample": "正文样本", "reason_sample": ""}
        r.update(over)
        return r

    def test_echo_early_stop(self):
        reqs = [self._req(), self._req(out_tokens=4)]   # 肇事在槽位 1
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=True),
                         ("early_stop", 1))

    def test_free_no_early_stop(self):
        reqs = [self._req(out_tokens=4)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=False))

    def test_echo_within_threshold_none(self):
        # 480/512 超过 0.8×512=409.6：预算内的正常收尾，不判
        reqs = [self._req(out_tokens=480)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_length_finish_none(self):
        reqs = [self._req(out_tokens=4, finish="length")]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_all_err_none(self):
        reqs = [self._req(out_tokens=4, err="http 500")]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_degenerate_text_sample(self):
        reqs = [self._req(text_sample="}\n" * 60)]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=False),
                         ("degenerate", 0))

    def test_degenerate_reason_sample(self):
        # 思考流退化走 reasoning 通道（fastllm 实测），text 通道正常也命中
        reqs = [self._req(reason_sample=" ".join(f"{i}." for i in range(6, 60)))]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=True),
                         ("degenerate", 0))


class TestRunRepGuarded(unittest.IsolatedAsyncioTestCase):
    """_run_rep_guarded：弃测重测编排——首次异常重测一次（换 nonce 破缓存
    由 _run_point 内部保证）、重测仍异常按实留档 anomaly、首次正常不重测、
    stop_flag 置位不检测不重测。_run_point 打桩脚本化返回，emit 收队列。"""

    ANOM_PT = {   # 实测案例形态：4K 档 echo 模式仅输出 4/512 tokens
        "reqs": [{"err": None, "finish": "stop", "out_tokens": 4,
                  "ttft_s": 0.12, "decode_tok_s": 29.1,
                  "text_sample": "好的，以下是", "reason_sample": ""}]}
    OK_PT = {
        "reqs": [{"err": None, "finish": "stop", "out_tokens": 512,
                  "ttft_s": 0.11, "decode_tok_s": 60.0,
                  "text_sample": "", "reason_sample": ""}]}

    def _make_run(self):
        run = BenchRun({})
        statuses = []

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run.emit = fake_emit
        return run, statuses

    def _script(self, run, points, post=None):
        """_run_point 打桩：按脚本顺序返回；post(i) 在第 i 次调用返回后执行
        （用于模拟测量期间置位 stop_flag）。"""
        calls = []

        async def fake_run_point(*args, **kwargs):
            i = len(calls)
            calls.append(args)
            pt = points[min(i, len(points) - 1)]
            if post:
                post(i)
            return pt

        run._run_point = fake_run_point
        return calls

    async def test_anomaly_then_ok_retest(self):
        """首次 early_stop 异常 → 弃测重测一次 → 重测正常：返回正常 pt 带
        reps_discarded（1 条，含肇事读数与样本）、无 anomaly 键、播报状态。"""
        run, statuses = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 2, "异常应弃测重测一次")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt.get("reps_discarded"), [{
            "anomaly": "early_stop", "req": 0, "out_tokens": 4,
            "finish": "stop", "ttft_s": 0.12, "decode_tok_s": 29.1,
            "text_sample": "好的，以下是", "reason_sample": ""}])
        self.assertTrue(any("仅输出 4/512 tokens" in s and "弃测重测" in s
                            for s in statuses), f"应有弃测播报: {statuses}")

    async def test_anomaly_persists_marked(self):
        """重测仍异常：接受结果、pt 带 anomaly='early_stop' 留档、reps_discarded
        仅 1 条（每个槽位最多重测 ANOMALY_RETEST_MAX 次，不无限重试）。"""
        run, _ = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.ANOM_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1 + ANOMALY_RETEST_MAX)
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), ANOMALY_RETEST_MAX)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)

    async def test_ok_no_retest_no_keys(self):
        """首次正常：不重测、不附加 anomaly/reps_discarded 键。"""
        run, statuses = self._make_run()
        calls = self._script(run, [dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("anomaly", pt)
        self.assertNotIn("reps_discarded", pt)
        self.assertEqual(statuses, [], "正常路径不应有弃测播报")

    async def test_stop_flag_no_detect_no_retest(self):
        """stop_flag 置位（测量期间停止）：不检测直接返回，不发起重测。"""
        run, statuses = self._make_run()

        def set_stop(i):
            run.stop_flag = True   # 模拟首批在途期间置停

        calls = self._script(run, [dict(self.ANOM_PT)], post=set_stop)
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1, "停止后不重测")
        self.assertNotIn("anomaly", pt)
        self.assertNotIn("reps_discarded", pt)
        self.assertEqual(statuses, [], "停止路径不应有弃测播报")

    async def test_degenerate_retest_message(self):
        """退化重复路径：播报文案不含 token 数（形态与 early_stop 区分）。"""
        run, statuses = self._make_run()
        degen = {"reqs": [{"err": None, "finish": "stop", "out_tokens": 512,
                           "ttft_s": 0.3, "decode_tok_s": 40.0,
                           "text_sample": "",
                           "reason_sample": "        " + " ".join(
                               f"{i}." for i in range(6, 60))}]}
        calls = self._script(run, [degen, dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 8192, 1, 1.8,
                                        0, 512, False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(pt["reps_discarded"][0]["anomaly"], "degenerate")
        self.assertTrue(any("退化重复" in s for s in statuses), f"{statuses}")


if __name__ == "__main__":
    unittest.main()
