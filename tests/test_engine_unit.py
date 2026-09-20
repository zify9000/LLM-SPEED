"""引擎纯逻辑回归测试：上下文构造（嵌套前缀/确定性/模块块级/wrap/零输入）、
复测聚合、错误语义化、超窗删减辅助、部署上限超窗贴边裁减判定、
agent 未回传点缓存迹象三通道判别（走平/全量预期/运行级佐证）、
媒体语料合成与媒体聚合口径（ADR-0020）、瞬时失败兜底重试（传输异常/429/5xx/
空流分类、重试后成功留 retried 痕、耗尽收口、stop_flag 中止、quiet 静默重试）、
批级并发口径（decode 峰值滑窗/批级 6 指标守卫/复测聚合）。

运行（任一方式，无需 pytest 也可跑）：
    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/ -v
"""
import asyncio
import httpx
import hashlib
import json
import os
import re
import struct
import sys
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
    DECODE_PEAK_WINDOW_S,
    FLOOR_RETEST_MAX,
    BenchRun,
    CLIENT_CUT_FACTOR,
    OUT_HINT_FACTOR,
    RATE_GATE_MIN_TOKENS,
    SCENARIOS,
    _aggregate_media_reqs,
    _aggregate_reps,
    _align_echo_region,
    _asr_audio,
    _batch_prefill_point,
    _build_ocr_messages,
    _build_translate_messages,
    _cache_hit_from_usage,
    _cache_hit_verdict,
    _cache_pollution_verdict,
    _classify_http_error,
    _conc_batch_stats,
    _correct_filler_chars,
    _decode_burst,
    _decode_peak_rate,
    _decode_sums,
    _detect_agent_anomaly,
    _detect_rep_anomaly,
    _fill,
    _flush_episodes,
    _head_sample,
    _is_collapsed_reply,
    _is_ctx_overflow,
    _is_degenerate_text,
    _is_implausible_prefill_rate,
    _is_soft_ctx_overflow,
    _is_transient_exc,
    _is_transient_http_status,
    _load_code_pool,
    _make_module_stream,
    _make_stream,
    _net_elapsed,
    _ocr_images,
    _out_hint_tail,
    _plan_ctx_edge,
    _prior_suspect,
    _ratio_refold,
    _render_in_full,
    _retry_after_s,
    _strip_in_full,
    _synth_document_png,
    _synth_speech_wav,
    _tok_per_batch,
    _total_decode_rates,
    _translate_max_tokens,
    _trim_middle,
    _tts_text,
    _wav_seconds,
    _wma,
    build_agent_messages,
    build_messages,
    rate_prior,
    TRANSIENT_RETRY_MAX,
    EMPTY_STREAM_RETRY_MAX,
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


class TestCacheHitVerdict(unittest.TestCase):
    """_cache_hit_verdict（C>0 未回传点缓存迹象判别）：三通道 OR + 运行级
    佐证——①走平（同指令零缓存档基准，小缓存档可靠）；②全量预期（实测
    TTFT 显著低于同规模全量 prefill 预期，大缓存档走平必然失效时的兜底，
    无先验不判）；③本轮已证实缓存生效（一点证实全轮继承，不再要求逐点
    证据）。三通道全灭 → None（诚实留档）。"""

    def test_walk_flat_channel(self):
        # 走平：tnet ≤ max(1.5×base, base+0.5)；base 小时 +0.5 主导
        self.assertEqual(
            _cache_hit_verdict(4096, 4600, 0.30, 0.25, None, 4096, False),
            4096)
        # base 大时 1.5× 主导
        self.assertEqual(
            _cache_hit_verdict(4096, 4600, 1.0, 0.6, None, 4096, False),
            4096)
        # 超出走平带 → None（无先验、未证实，其余通道不接手）
        self.assertIsNone(
            _cache_hit_verdict(4096, 4600, 0.9, 0.25, None, 4096, False))

    def test_full_expectation_channel(self):
        # 走平失败（0.9 > 0.25+0.5）但显著低于全量预期
        # 阈值 = 0.6×4600/3000 ≈ 0.92
        self.assertEqual(
            _cache_hit_verdict(4096, 4600, 0.9, 0.25, 3000.0, 4096, False),
            4096)
        # 超过全量预期阈值（TTFT≈全量重算）→ None
        self.assertIsNone(
            _cache_hit_verdict(4096, 4600, 0.95, 0.25, 3000.0, 4096, False))

    def test_proven_inherits_after_channels_fail(self):
        # 两通道皆败（大缓存档实测形态：读 102K 缓存 ~25s vs 全量预期 ~20s），
        # 本轮已证实 → 直接继承估算
        self.assertEqual(
            _cache_hit_verdict(102400, 103000, 25.0, 0.5, 3000.0, 102400, True),
            102400)

    def test_all_channels_fail_returns_none(self):
        # 同形态但未证实 → 三通道全灭，诚实留档 None
        self.assertIsNone(
            _cache_hit_verdict(102400, 103000, 25.0, 0.5, 3000.0, 102400, False))

    def test_prime_zero_cannot_estimate(self):
        # 预热实测 prompt 缺失（prime=0）恒 None，即便走平/佐证成立
        self.assertIsNone(
            _cache_hit_verdict(4096, 4600, 0.1, 0.2, None, 0, True))

    def test_estimate_capped_by_ptok(self):
        # 估算钳制 min(prime, ptok−1)：prime 超过测量 prompt 时取 ptok−1
        self.assertEqual(
            _cache_hit_verdict(4096, 500, 0.10, 0.25, None, 4096, False), 499)

    def test_zero_cache_rung_not_applicable(self):
        # 零缓存档不走本判别（回传/0 分流在调用点）
        self.assertIsNone(
            _cache_hit_verdict(0, 700, 0.2, None, 3000.0, 700, True))


class TestOutHintTail(unittest.TestCase):
    """_out_hint_tail（指令末尾复述输出长度约束，2026-09-16 拍板）：仅双向
    引导场景（out_hint 含 {aim}：creative/code/agent）；translate/ocr 单向
    封顶不复述；上下文贴边（ctx + 预算 + 余量 + 安全边超 max_ctx）不加。"""

    def test_two_way_scenarios_return_template(self):
        for sc in ("creative", "code", "agent"):
            tail = _out_hint_tail(SCENARIOS[sc], None, 4096, 512)
            self.assertIsNotNone(tail, f"{sc} 双向引导应复述")
            self.assertIn("{aim}", tail)
            self.assertEqual(tail, SCENARIOS[sc]["out_hint"])

    def test_one_way_scenarios_none(self):
        for sc in ("translate", "ocr"):
            self.assertIsNone(_out_hint_tail(SCENARIOS[sc], None, 4096, 512),
                              f"{sc} 单向封顶不复述")

    def test_edge_headroom_gate(self):
        sc = SCENARIOS["code"]
        # 4096 + 512 + CTX_HEADROOM(1024) + 256 = 5888 ≤ 8192 → 加
        self.assertIsNotNone(_out_hint_tail(sc, 8192, 4096, 512))
        # 5888 > 5000 → 贴边不加（省 tokens 防顶窗）
        self.assertIsNone(_out_hint_tail(sc, 5000, 4096, 512))
        # 无 max_ctx 配置恒加
        self.assertIsNotNone(_out_hint_tail(sc, None, 131072, 512))


class TestPriorSuspect(unittest.TestCase):
    """_prior_suspect（hit=None 的 C>0 点登记先验曲线的防污染判别）：速率
    超先验 1.5× → 疑似漏判的缓存命中，不登记；真无缓存后端的诚实速率
    （0.8~1.3× 先验）不误伤。"""

    def test_inflated_none_hit_point_suspect(self):
        # 实测 2026-09-16 事故形态：缓存档判别 None，34928 tok/s 虚高速率
        # vs 先验 1371 → 25× 远超 1.5× 闸，判疑似不登记
        self.assertTrue(_prior_suspect(131072, False, None, 1371.0, 34928.0))
        # 1.67× 先验（全量预期通道边缘漏判带）也判疑似
        self.assertTrue(_prior_suspect(8192, False, None, 1000.0, 1600.0))

    def test_honest_full_prefill_rate_registers(self):
        # 真无缓存后端的诚实速率落在 ±30% 带内：照常登记
        self.assertFalse(_prior_suspect(131072, False, None, 1371.0, 1300.0))
        self.assertFalse(_prior_suspect(131072, False, None, 1371.0, 2056.0))

    def test_non_suspect_shapes(self):
        # c=0 零缓存档 / 回传点 / 已有估算命中 / 无先验可对照 → 不判
        self.assertFalse(_prior_suspect(0, False, None, 1000.0, 99999.0))
        self.assertFalse(_prior_suspect(8192, True, None, 1000.0, 99999.0))
        self.assertFalse(_prior_suspect(8192, False, 8000, 1000.0, 99999.0))
        self.assertFalse(_prior_suspect(8192, False, None, None, 99999.0))


class TestCachePollutionVerdict(unittest.TestCase):
    """_cache_pollution_verdict（C>0 回传命中批指令段缓存残留污染判别）：
    实测未命中均值 < 0.6×（real_inst + 块尾 tail）且差值 >512 tokens 才判
    （小档块对齐噪声免疫）；仅 cache_reported 且 hit>0 的 ok 请求参与，
    real_inst 缺失不判。判据实测依据：128K 档 × 4K 指令档事故，未命中
    1554 vs 期望 4056+（比值 0.38）。"""

    def _req(self, prompt, hit, reported=True, err=None):
        return {"prompt_tokens": prompt, "cache_hit": hit,
                "cache_reported": reported, "err": err}

    def test_pollution_triggers(self):
        # 事故形态：prompt 134674 / hit 133120 → 未命中 1554 vs 期望 4056
        reqs = [self._req(134674, 133120)]
        self.assertEqual(
            _cache_pollution_verdict(reqs, 4056, 0), (1554, 4056))

    def test_tail_included_in_expected(self):
        # 块尾 tail 计入期望：real 2048 + tail 512 = 2560；未命中 1000
        # < 0.6×2560=1536 且差 1560 >512 → 判
        reqs = [self._req(17048, 16048)]
        self.assertEqual(
            _cache_pollution_verdict(reqs, 2048, 512), (1000, 2560))

    def test_clean_batch_not_judged(self):
        # 未命中 ≈ 期望（真命中只增量 prefill 指令段）→ 不判
        reqs = [self._req(17048, 15000)]
        self.assertIsNone(_cache_pollution_verdict(reqs, 2048, 0))

    def test_ratio_boundary_not_judged(self):
        # 恰过 0.6× 不判（严格小于）：期望 4096，0.6×=2457.6，未命中
        # 2458 贴线之上 → None（差 1638 >512 也不判）
        reqs = [self._req(17458, 15000)]
        self.assertIsNone(_cache_pollution_verdict(reqs, 4096, 0))

    def test_small_rung_immune_by_gap(self):
        # 小档免疫：256 档未命中 100（比值 0.39 <0.6）但差 156 ≤512 → 不判
        reqs = [self._req(15100, 15000)]
        self.assertIsNone(_cache_pollution_verdict(reqs, 256, 0))

    def test_unreported_gateway_not_judged(self):
        # 未回传命中的网关走估算路径（天然免疫），不纳入校验
        reqs = [self._req(134674, 0, reported=False)]
        self.assertIsNone(_cache_pollution_verdict(reqs, 4056, 0))

    def test_zero_hit_and_err_excluded(self):
        # hit=0（回传但未命中）与 err 请求不参与；无合格样本 → None
        reqs = [self._req(134674, 0),
                self._req(134674, 133120, err="boom")]
        self.assertIsNone(_cache_pollution_verdict(reqs, 4056, 0))

    def test_real_inst_missing_not_judged(self):
        reqs = [self._req(134674, 133120)]
        self.assertIsNone(_cache_pollution_verdict(reqs, None, 0))

    def test_mean_over_chains(self):
        # 跨链均值判据：一链污染一链干净 → 均值 (1554+4056)/2=2805
        # ≥0.6×4056=2433.6 → 不判（多数链未被残留污染时不动整批）
        reqs = [self._req(134674, 133120), self._req(134674, 130618)]
        self.assertIsNone(_cache_pollution_verdict(reqs, 4056, 0))


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

    def test_few_fat_chunks_branch(self):
        """少而肥分支（BURST_FAT_*）：2~4 个多 token 事件亚毫秒冲刷按每 chunk
        token 数增补甄别（e2e 的 MOCK_ACCEPT=16 冲刷用例即走此分支）。"""
        self.assertTrue(_decode_burst(4, 0.0024, 64))   # 0.8ms/间隔（n−1=3 段）、16 tok/chunk
        self.assertFalse(_decode_burst(4, 0.02, 64))    # 6.7ms/间隔：真实周期交付
        self.assertFalse(_decode_burst(2, 0.001, 64))   # 2 个不足为凭
        self.assertFalse(_decode_burst(4, 0.0024, 16))  # 4 tok/chunk 不够肥
        self.assertFalse(_decode_burst(4, 0.0024))      # 无 out_tokens 不判


class TestTokPerBatch(unittest.TestCase):
    """_tok_per_batch（均 x/批新口径）：三种交付形态——①单事件多 token
    （vLLM/MTP，fat 直接采信）；②逐 token 事件成批冲刷（llama.cpp 投机解码，
    同 verify 周期接受的 k 个 token 亚毫秒到达、周期间有明显间隙，按到达间隔
    双峰聚类按批计）；③逐 token 交付记账噪声（≈1，无成批证据不入档）。"""

    def test_vllm_fat_direct(self):
        """①单事件多 token：fat ≥ SPEC_MIN_TPC 直接采信，gaps 不参与。"""
        self.assertEqual(_tok_per_batch([0.04] * 99, 100, 350.0), 3.5)

    def test_speculative_bimodal_clustered_by_batch(self):
        """②llama.cpp 投机型：事件口径恒 =1（out=n_chunks），gaps 双峰
        （批内 0.0001×N + 批界 0.03）→ 按批数计。4 批规模 [4,3,4,3]：
        14 token / (1+3 批界) = 3.5。"""
        gaps = ([0.0001] * 3 + [0.03] + [0.0001] * 2 + [0.03]
                + [0.0001] * 3 + [0.03] + [0.0001] * 2)
        self.assertEqual(_tok_per_batch(gaps, 14, 14.0), 3.5)

    def test_single_peak_per_token_none(self):
        """③非投机逐 token 单峰（间隔全在 0.02~0.05，无亚毫秒间隙）→ None。"""
        gaps = [0.02, 0.03, 0.05] * 5
        self.assertIsNone(_tok_per_batch(gaps, 16, 16.0))

    def test_whole_buffer_flush_none(self):
        """整段缓冲冲刷（全亚毫秒、无批界）：批界中位数门槛挡住，不误判成
        一批 → None。"""
        self.assertIsNone(_tok_per_batch([0.0001] * 10, 11, 11.0))

    def test_noise_fat_102_none(self):
        """无投机服务偶发 2 token 合并：fat=1.02 属记账噪声，无成批证据
        → None。"""
        self.assertIsNone(_tok_per_batch([0.02] * 20, 50, 51.0))

    def test_empty_gaps_and_out_none(self):
        """空 gaps / 空 out_tokens：无凭据 → None。"""
        self.assertIsNone(_tok_per_batch([], 100, 100.0))
        self.assertIsNone(_tok_per_batch([0.0001, 0.03], 3, None))
        self.assertIsNone(_tok_per_batch([0.0001, 0.03], 3, 0))

    def test_bimodal_but_tpb_below_threshold_none(self):
        """双峰成立但 tpb < SPEC_MIN_TPC：批次平均不足 1.15 仍算噪声 → None。"""
        gaps = [0.0001] * 3 + [0.03] + [0.0001] * 3   # 1 批界 → 2 批
        self.assertIsNone(_tok_per_batch(gaps, 8, 2.2))   # 2.2/2 = 1.1


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

    def test_offset_window(self):
        """换料偏移（弃测重测翻落点，2026-09-17 拍板）：offset>0 时截取
        窗口循环平移——长度不变、内容不同；offset=0 与原口径逐字节一致；
        offset 超过流长时取模回绕。"""
        total = len(self.stream)
        base = _fill(self.stream, 300)
        self.assertEqual(base, _fill(self.stream, 300, offset=0),
                         "offset=0 必须保持原嵌套前缀口径")
        shifted = _fill(self.stream, 300, offset=150)
        self.assertEqual(len(shifted), 300)
        self.assertEqual(shifted, self.stream[150:450])
        self.assertNotEqual(shifted, base)
        # 回绕：offset 取模；跨流尾时循环续接
        wrapped = _fill(self.stream, 300, offset=total - 100)
        self.assertEqual(wrapped, self.stream[total - 100:] + self.stream[:200])
        self.assertEqual(_fill(self.stream, 50, offset=total + 10),
                         _fill(self.stream, 50, offset=10))


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


# 语料内容清单（相对路径 + NUL + 内容，按路径排序流式 sha256）——与
# scripts/build_agent_corpus.py 的 manifest_sha256 同算法：两条独立实现互为
# 校验，任一侧被误改都会不一致。
def _manifest_sha256(base: str, rels: list[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(rels):
        h.update(rel.encode())
        h.update(b"\0")
        with open(os.path.join(base, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class TestCorpusProvenance(unittest.TestCase):
    """语料是测速的"刻度"：改语料会让历史存档失去可比性，必须显式发生。

    本类把三份 vendored 语料的**内容清单哈希与统计区间**钉成契约（数字来源与
    核对方式见各目录 PROVENANCE.md）。语料一旦被误改/误重建/静默降级到内置回退
    池，这里立刻变红——而引擎侧的"目录缺失回退内置池"设计本身不会报错。
    故意更新语料时：跑一次下面命令，把新哈希与区间同步到 PROVENANCE 与本节。
        python -m pytest -q tests/test_engine_unit.py -k corpus -v
    """
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _dir(self, *parts):
        return os.path.join(self.ROOT, "corpus", *parts)

    def test_agent_corpus_manifest_and_block_bounds(self):
        base = self._dir("agent")
        rels = sorted(f for f in os.listdir(base)
                      if f.startswith("traj-") and f.endswith(".txt"))
        self.assertEqual(len(rels), 100, "agent 语料块数变了（PROVENANCE.md 记载 100）")
        self.assertEqual(
            _manifest_sha256(base, rels),
            "4c85ab95e43d58c782546cc7ec345a859283fdd9f540bb93a8890d1367627778",
            "agent 语料内容变了——若是有意更新，请同步 corpus/agent/PROVENANCE.md")
        chars = [len(open(os.path.join(base, r), encoding="utf-8").read()) for r in rels]
        self.assertGreaterEqual(min(chars), 2000,
                                "存在低于引擎采纳门限（_load_agent_pool ≥2000）的块")
        # 上界用实测值（DOC_MAX_BLOCK=40200）：生成器判定 40000，仓内 3 个块
        # 由早期修订脚本产出、略超——见 PROVENANCE.md 的"不同源"一节
        self.assertLessEqual(max(chars), 40200,
                             "agent 语料块超文档化上界（PROVENANCE.md 记载 40183）")

    def test_agent_corpus_satisfies_documented_contract_via_verify(self):
        """`--verify`（只读、不依赖 pyarrow）必须与测试判定一致。"""
        import subprocess
        r = subprocess.run([sys.executable,
                            os.path.join(self.ROOT, "scripts", "build_agent_corpus.py"),
                            "--verify"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("块长契约 OK", r.stdout)

    def test_code_corpus_manifest(self):
        base = self._dir("code")
        rels = [os.path.relpath(os.path.join(dp, f), base)
                for dp, _, fs in os.walk(base) for f in fs
                if f not in ("LICENSE", "README.md", "PROVENANCE.md")]
        self.assertEqual(len(rels), 15, "code 语料文件数变了（PROVENANCE.md 记载 15）")
        self.assertEqual(
            _manifest_sha256(base, rels),
            "06b408c7b5d00a2c5d8bb0160f9705d500425ca5802086a06feba162dff8d887",
            "code 语料内容变了——若是有意更新，请同步 corpus/code/PROVENANCE.md")

    def test_creative_corpus_manifest_and_size(self):
        base = self._dir("creative")
        path = os.path.join(base, "hongloumeng.txt")
        raw = open(path, encoding="utf-8").read()
        with open(path, "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(),
                "58e75b74f0c51e7863f8aa6871ec349f5c96ffdbecb5bdd5cf83ea016f553826",
                "creative 语料变了——若是有意更新，请同步 corpus/creative/PROVENANCE.md")
        # 去空白字符数 = 语料的"刻度"（设计文档记 83 万字符、循环点推到 512K 之后）
        self.assertEqual(len("".join(raw.split())), 838587,
                         "creative 语料字符尺度变了（PROVENANCE.md 记载 838,587）")

    def test_creative_corpus_has_no_pg_header(self):
        """PG 页眉已剥离（版权声明改存 LICENSE-PG.txt）——正文里不得再出现。"""
        text = open(self._dir("creative", "hongloumeng.txt"), encoding="utf-8").read()
        self.assertNotIn("Project Gutenberg", text)
        self.assertNotIn("\ufffd", text, "残留 U+FFFD 替换字符")

    def test_license_and_provenance_present(self):
        """三份 vendored 语料各有一份溯源文档；creative 另需 PG 许可声明。"""
        for d in ("agent", "code", "creative"):
            self.assertTrue(os.path.isfile(self._dir(d, "PROVENANCE.md")),
                            f"corpus/{d}/PROVENANCE.md 缺失")
        pg = open(self._dir("creative", "LICENSE-PG.txt"), encoding="utf-8").read()
        self.assertIn("Project Gutenberg", pg)
        self.assertIn("24264", pg)
        self.assertTrue(os.path.isfile(self._dir("code", "LICENSE")),
                        "corpus/code/LICENSE（上游 MIT 许可）缺失")

    def test_metric_version_is_documented(self):
        """口径版本表：每个已发布的版本都要有一句"它是什么口径"的说明——
        版本号本身没有意义，读者要能查到怎么解读（ADR-0083）。"""
        import bench as b
        self.assertGreaterEqual(b.METRIC_VERSION, 1)
        for v in range(1, b.METRIC_VERSION + 1):
            self.assertIn(v, b.METRIC_VERSION_NOTES, f"缺少 v{v} 的口径说明")
            self.assertTrue(b.METRIC_VERSION_NOTES[v].strip())
        self.assertEqual(b.LEGACY_METRIC_VERSION, 0)

    def test_save_stamps_metric_version(self):
        """落盘顶层带口径版本；点级版本随点自身保存（拼接/复测可混档）。"""
        import bench as b
        self.assertIn("METRIC_VERSION", b.__dict__)
        with tempfile.TemporaryDirectory() as td:
            run = BenchRun({}, results_dir=td)
            run.results.append({"model": "m1", "scenario": "code", "ctx_target": 0,
                                "metric_version": b.METRIC_VERSION})
            run._save()
            with open(os.path.join(td, f"{run.run_id}.json"), encoding="utf-8") as f:
                d = json.load(f)
        self.assertEqual(d["metric_version"], b.METRIC_VERSION)
        self.assertEqual(d["results"][0]["metric_version"], b.METRIC_VERSION)

    def test_engine_actually_loads_vendored_corpora(self):
        """回归"静默降级"：语料在盘时必须真被引擎加载（而非回退内置池）。"""
        self.assertTrue(bench._load_agent_pool(), "agent 语料未被加载（回退内置池？）")
        self.assertTrue(bench._load_creative_pool(), "creative 语料未被加载")
        self.assertTrue(bench._load_code_pool(), "code 语料未被加载")


class TestAnomalyTextExternalization(unittest.TestCase):
    """异常输入全文外置（ADR-0053 可精确复跑 + ADR-0082 存档瘦身）：
    超长 in_text 落 <run_id>.anomaly.txt 侧车，存档只留头尾摘要 + in_text_ref
    字节切片；短文本内联；预算耗尽截断留痕；侧车写失败绝不拖垮 _save。"""

    def _run(self, td, results):
        run = BenchRun({}, results_dir=td)
        run.results = results
        run._save()   # 同步调用即可（真实调用点在 to_thread 内）
        return run

    @staticmethod
    def _load(td, run):
        with open(os.path.join(td, f"{run.run_id}.json"), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _side(td, run):
        return os.path.join(td, f"{run.run_id}{bench.ANOMALY_TEXT_SUFFIX}")

    def test_long_in_text_externalized_to_sidecar(self):
        """100K 字 reqs.in_text + 120K 字 reps_discarded.in_text：存档远小于输入、
        截断文本带省略标记、ref 的 len/chars/sha1 正确、侧车切片字节 == 原文。"""
        a, b = "甲" * 100_000, "乙" * 120_000
        da, db = a.encode("utf-8"), b.encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            run = self._run(td, [{"model": "m1",
                                  "reqs": [{"req": 0, "in_text": a}],
                                  "reps_discarded": [{"in_text": b}]}])
            path = os.path.join(td, f"{run.run_id}.json")
            side = self._side(td, run)
            self.assertTrue(os.path.exists(side), "侧车应存在")
            self.assertLess(os.path.getsize(path), 20000,
                            "存档应远小于输入全文（截断后仅头尾摘要）")
            d = self._load(td, run)
            pt = d["results"][0]
            raw = open(side, "rb").read()
            for holder, full, data in ((pt["reqs"][0], a, da),
                                       (pt["reps_discarded"][0], b, db)):
                self.assertNotEqual(holder["in_text"], full)
                self.assertIn("……", holder["in_text"], "应有省略标记")
                ref = holder["in_text_ref"]
                self.assertEqual(ref["file"],
                                 f"{run.run_id}{bench.ANOMALY_TEXT_SUFFIX}")
                self.assertEqual((ref["len"], ref["chars"]),
                                 (len(data), len(full)))
                self.assertEqual(ref["sha1"], hashlib.sha1(data).hexdigest())
                sl = raw[ref["off"]:ref["off"] + ref["len"]]
                self.assertEqual(sl, data, "侧车切片应等于原文")
                self.assertEqual(hashlib.sha1(sl).hexdigest(), ref["sha1"])
            trunc = pt["reqs"][0]["in_text"]
            self.assertEqual(trunc[:bench.IN_TEXT_TRUNC_HEAD],
                             a[:bench.IN_TEXT_TRUNC_HEAD])
            self.assertEqual(trunc[-bench.IN_TEXT_TRUNC_TAIL:],
                             a[-bench.IN_TEXT_TRUNC_TAIL:])
            self.assertNotIn("anomaly_text_warning", d, "预算内不应告警")

    def test_short_in_text_stays_inline_no_sidecar(self):
        """短文本（含恰好等于阈值）保持内联原样，且不产生空侧车文件。"""
        short = "S" * 50
        edge = "T" * bench.IN_TEXT_INLINE_MAX_CHARS
        with tempfile.TemporaryDirectory() as td:
            run = self._run(td, [{"reqs": [{"in_text": short}],
                                  "reps_discarded": [{"in_text": edge}]}])
            d = self._load(td, run)
            pt = d["results"][0]
            self.assertEqual(pt["reqs"][0]["in_text"], short)
            self.assertEqual(pt["reps_discarded"][0]["in_text"], edge)
            self.assertNotIn("in_text_ref", pt["reqs"][0])
            self.assertNotIn("in_text_ref", pt["reps_discarded"][0])
            self.assertFalse(os.path.exists(self._side(td, run)),
                             "无超长文本不得留下空侧车")

    def test_reps_reqs_in_text_externalized(self):
        """reps[].reqs[].in_text（69% 体积的来源）也必须走外置，否则静默回归。"""
        big = "R" * 60000
        with tempfile.TemporaryDirectory() as td:
            run = self._run(td, [{"model": "m1",
                                  "reps": [{"reqs": [{"in_text": big},
                                                     {"in_text": "x"}]}]}])
            d = self._load(td, run)
            h = d["results"][0]["reps"][0]["reqs"][0]
            self.assertIn("in_text_ref", h)
            self.assertIn("……", h["in_text"])
            ref = h["in_text_ref"]
            raw = open(self._side(td, run), "rb").read()
            self.assertEqual(raw[ref["off"]:ref["off"] + ref["len"]],
                             big.encode("utf-8"))
            self.assertEqual(d["results"][0]["reps"][0]["reqs"][1]["in_text"], "x",
                             "未超阈值的兄弟字段不受影响")

    def test_budget_exhausted_marks_truncated_and_warns(self):
        """预算耗尽：条目不写侧车、不给 ref（不谎称全文留存），只截断 +
        in_text_truncated 标 + 顶层 anomaly_text_warning；无空侧车。"""
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.object(bench, "ANOMALY_TEXT_BUDGET_BYTES",
                                            1000):
                run = self._run(td, [{"reqs": [{"in_text": "A" * 5000}],
                                      "reps_discarded": [{"in_text": "B" * 5000}]}])
            d = self._load(td, run)
            pt = d["results"][0]
            for h in (pt["reqs"][0], pt["reps_discarded"][0]):
                self.assertTrue(h["in_text_truncated"])
                self.assertNotIn("in_text_ref", h)
                self.assertIn("……", h["in_text"])
            self.assertIn("anomaly_text_warning", d)
            self.assertFalse(os.path.exists(self._side(td, run)),
                             "预算内一条都没写时不得留空侧车")

    def test_budget_stops_writing_after_first_overflow(self):
        """预算只够第一条：第一条外置（有 ref），第二条起全局停写（截断+标），
        不是逐条"见缝插针"继续塞更小的条目。"""
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.object(bench, "ANOMALY_TEXT_BUDGET_BYTES",
                                            6000):
                run = self._run(td, [{"reqs": [{"in_text": "A" * 5000}],
                                      "reps_discarded": [{"in_text": "B" * 5000}]}])
            d = self._load(td, run)
            pt = d["results"][0]
            self.assertIn("in_text_ref", pt["reqs"][0])
            self.assertNotIn("in_text_ref", pt["reps_discarded"][0])
            self.assertTrue(pt["reps_discarded"][0]["in_text_truncated"])
            self.assertIn("anomaly_text_warning", d)
            self.assertEqual(len(open(self._side(td, run), "rb").read()), 5000,
                             "侧车只应含预算内的第一条")

    def test_save_survives_sidecar_write_failure(self):
        """侧车写失败（OSError）：_save 仍成功、全文保持内联、无空侧车——
        瘦身特性不得让存档本身丢失。"""
        big = "X" * 100000
        real_open = open

        def flaky(path, *a, **k):
            if str(path).endswith(bench.ANOMALY_TEXT_SUFFIX):
                raise OSError("模拟侧车不可写")
            return real_open(path, *a, **k)

        with tempfile.TemporaryDirectory() as td:
            run = BenchRun({}, results_dir=td)
            run.results = [{"reqs": [{"in_text": big}]}]
            with unittest.mock.patch("builtins.open", new=flaky):
                run._save()   # 不得抛
            d = self._load(td, run)
            self.assertEqual(d["results"][0]["reqs"][0]["in_text"], big)
            self.assertNotIn("in_text_ref", d["results"][0]["reqs"][0])
            self.assertFalse(os.path.exists(self._side(td, run)))


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


class TestRenderInFull(unittest.TestCase):
    """输入全文渲染（_render_in_full，瞬态 _in_full 的内容）：每条消息一行
    「【role】文本」、parts 列表拼接 text、连续非 text 部分以「［图片×N］」
    占位、不设长度上限（异常日志要求输入原文全部列出）。"""

    def test_plain_messages_one_line_each(self):
        msgs = [{"role": "system", "content": "系统提示"},
                {"role": "user", "content": "问题"}]
        self.assertEqual(_render_in_full(msgs),
                         "【system】系统提示\n【user】问题")

    def test_parts_list_text_and_image_placeholder(self):
        """OCR 多模态 parts：text 拼接、连续 image_url 部分合并为一个
        「［图片×N］」占位。"""
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:a"}},
            {"type": "image_url", "image_url": {"url": "data:b"}},
            {"type": "text", "text": "请识别"},
            {"type": "image_url", "image_url": {"url": "data:c"}}]}]
        self.assertEqual(_render_in_full(msgs),
                         "【user】［图片×2］请识别［图片×1］")

    def test_no_length_cap(self):
        """不设长度上限：10 万字符输入全文保留（与 in_sample 的 200 上限区分）。"""
        long_input = "长" * 100000
        msgs = [{"role": "user", "content": long_input}]
        self.assertEqual(_render_in_full(msgs), "【user】" + long_input)

    def test_strip_in_full_removes_transient_key(self):
        """_strip_in_full 定稿剥离：reqs 全部去 _in_full，肇事 req 的 in_text
        独立键保留；无 reqs 键/空点不炸。"""
        pt = {"reqs": [{"req": 0}, {"req": 1, "_in_full": "x"},
                       {"req": 2, "anomaly": "early_stop", "in_text": "全文"}]}
        _strip_in_full(pt)
        self.assertFalse(any("_in_full" in r for r in pt["reqs"]))
        self.assertEqual(pt["reqs"][2]["in_text"], "全文",
                         "in_text 独立键不受剥离影响")
        _strip_in_full({})


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

    def test_stream_offset_shifts_echo_region(self):
        """换料偏移翻 echo 改写区落点（2026-09-17 拍板）：同目标同 nonce
        同 filler_chars 下，offset>0 的构造填充长度不变、内容不同，echo
        预填（改写区开头）随之改变；缺省/0 与嵌套前缀口径逐字节一致。"""
        base, _, n1 = build_messages("code", 4096, 3.4, "nonce-5",
                                     filler_chars=20000, echo=True)
        same, _, _ = build_messages("code", 4096, 3.4, "nonce-5",
                                    filler_chars=20000, echo=True,
                                    stream_offset=0)
        shifted, _, n2 = build_messages("code", 4096, 3.4, "nonce-5",
                                        filler_chars=20000, echo=True,
                                        stream_offset=7919)
        self.assertEqual(base, same, "offset=0 与缺省口径一致")
        self.assertEqual(n1, n2, "换料只平移窗口，填充长度不变")
        self.assertNotEqual(base[1]["content"], shifted[1]["content"],
                            "平移后参考材料内容应不同")
        self.assertEqual(len(base), len(shifted), 3)
        self.assertNotEqual(base[2]["content"], shifted[2]["content"],
                            "echo 预填（改写区开头）应随落点翻牌改变")

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

    def test_agent_matrix_messages(self):
        """Agent 缓存×指令矩阵构造（ADR-0042）：预设上下文与指令拆开为两条
        user 消息（[system, user(编号行+轨迹上下文), user(指令)]）；同
        cache_chars 下预热与测量的前两条消息逐字节一致（前缀缓存命中口径，
        指令真实长度可链内差分）；指令段长度随 inst_tokens 单调增长、材料
        接续上下文之后的语料段（不与上下文同位）；模板固定开销 ~260
        chars（suffix 为引出长输出而加强，固定开销由链内差分吸收）；
        cache_tokens=0 为零缓存档，上下文
        消息仅编号行；prime=True 的指令消息为超短任务；seg_chars 显式指定
        材料字符数（偏离校正补测用）；SCENARIOS["agent"] 双阶梯默认值存在。"""
        cpt = 3.5
        sc = SCENARIOS["agent"]
        stream = sc["filler_stream"]
        # 同 cache_chars：预热与测量的前两条消息逐字节一致（前缀缓存命中口径）
        prime_m, _ep, ctx_p, seg_p = build_agent_messages(
            4096, 0, cpt, "n-1", cache_chars=5000, prime=True)
        meas_m, _e, ctx_m, seg_m = build_agent_messages(
            4096, 1024, cpt, "n-1", cache_chars=5000)
        self.assertEqual(ctx_p, ctx_m, "同 cache_chars 上下文段长度一致")
        self.assertEqual(ctx_p, 5000)
        self.assertEqual(seg_p, 0, "预热请求无指令材料")
        self.assertEqual([m["role"] for m in prime_m],
                         ["system", "user", "user"])
        self.assertEqual([m["role"] for m in meas_m],
                         ["system", "user", "user"])
        self.assertEqual(prime_m[0], meas_m[0], "system 逐字节一致")
        self.assertEqual(prime_m[1], meas_m[1],
                         "上下文消息逐字节一致（前缀缓存命中口径）")
        self.assertNotEqual(prime_m[2], meas_m[2], "指令消息分开")
        self.assertEqual(prime_m[2]["content"], "Reply with OK.",
                         "预热指令消息为超短任务")
        inst_msg = meas_m[2]["content"]
        self.assertTrue(inst_msg.startswith(sc["inst_prefix"]))
        self.assertTrue(inst_msg.endswith(sc["inst_suffix"]))
        # 指令材料接续上下文之后的语料段（流按长度取模错位，不与上下文同位）
        off = ctx_m % len(stream)
        self.assertEqual(
            inst_msg[len(sc["inst_prefix"]):len(sc["inst_prefix"]) + 40],
            stream[off:off + 40])
        # 模板固定开销（~260 chars ≈ 75 tokens@3.5）：suffix 为引出长输出而
        # 加强（thorough/detailed），64 tokens 小档被固定开销淹没属预期——
        # 指令真实长度走链内差分测量，固定开销在差分中抵消
        self.assertLessEqual(len(sc["inst_prefix"]) + len(sc["inst_suffix"]),
                             400)
        # 指令段长度随 inst_tokens 单调增长
        segs = [build_agent_messages(0, i, cpt, "n")[3]
                for i in (128, 256, 512, 1024, 2048)]
        self.assertTrue(all(a < b for a, b in zip(segs, segs[1:])),
                        f"指令段严格单调增长: {segs}")
        # seg_chars 显式指定指令材料字符数（偏离校正补测用）
        fx_m, _ef, _cz, seg_fx = build_agent_messages(
            0, 1024, cpt, "n", seg_chars=777)
        self.assertEqual(seg_fx, 777)
        self.assertEqual(fx_m[2]["content"], sc["inst_prefix"]
                         + _fill(stream, 777) + sc["inst_suffix"])
        # cache_tokens=0：零缓存档上下文消息仅编号行，无轨迹材料
        z_m, _ez, ctx_z, seg_z = build_agent_messages(0, 512, cpt, "n-2")
        self.assertEqual(ctx_z, 0)
        self.assertGreater(seg_z, 0)
        self.assertEqual(z_m[1]["content"], "参考材料（编号 n-2）：\n")
        # 双阶梯默认值（缓存 0~256K × 指令 64~8K tokens）
        self.assertEqual(sc["cache_ladder"],
                         [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
        self.assertEqual(sc["inst_ladder"],
                         [64, 128, 256, 512, 1024, 2048, 4096, 8192])

    def test_agent_seg_salt_cursor_nonoverlap(self):
        """同缓存档逐尝试错位取段（_run_agent_matrix 游标推进规则 seg_n +
        seg_n//2 + 1024 的构造侧不变量）：任意两次尝试的指令材料在语料流中
        区间不重叠、内容不互为前缀——否则取段偏移只随 ctx 定，升序测量时
        各指令档材料嵌套前缀，上一档残留被服务端算进 cache_hit（实测 128K
        档 4K 指令未命中仅 1554 的事故）。"""
        cpt = 3.5
        sc = SCENARIOS["agent"]
        stream = sc["filler_stream"]
        cache_chars = 20000
        salt = 0
        segs = []
        for inst in (64, 256, 1024, 4096):
            m, _e, ctx_n, seg_n = build_agent_messages(
                131072, inst, cpt, "n", cache_chars=cache_chars,
                seg_salt=salt)
            seg = m[2]["content"][len(sc["inst_prefix"]):
                                  len(m[2]["content"]) - len(sc["inst_suffix"])]
            self.assertEqual(len(seg), seg_n)
            off = (ctx_n + salt) % len(stream)
            segs.append((off, seg_n, seg))
            salt += seg_n + seg_n // 2 + 1024   # 与 _attempt_track 同推进规则
        for (o1, n1, _s1), (o2, _n2, _s2) in zip(segs, segs[1:]):
            self.assertGreaterEqual(o2, o1 + n1,
                                    "后一次尝试取段起点不落入前一次区间")
        for i, (_o, n1, s1) in enumerate(segs):
            for _o2, n2, s2 in segs[i + 1:]:
                if min(n1, n2) < 100:
                    continue   # 超短材料前缀比较无判别力，跳过
                self.assertFalse(s2.startswith(s1[:200]),
                                 "指令材料不互为前缀（防残留计入 cache_hit）")
                self.assertFalse(s1.startswith(s2[:200]),
                                 "指令材料不互为前缀（防残留计入 cache_hit）")


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

    def test_anomaly_rep_reqs_preferred(self):
        """留档样本优先取异常轮：pool 中带 anomaly 的复测（重测仍异常按实
        留档）的 reqs 入档（肇事 req 带 anomaly + in_text），而非 decode
        居中次——留档样本必须来自异常那次；无异常维持居中口径；reps 摘要
        逐次带 anomaly（无异常为 None，前端按空值处理）。"""
        reps = [self._rep(50, req=0), self._rep(10, req=1), self._rep(30, req=2)]
        reps[1]["anomaly"] = "early_stop"
        reps[1]["reqs"] = [{"req": 1, "err": None, "anomaly": "early_stop",
                            "in_text": "【user】肇事输入全文"}]
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["anomaly"], "early_stop")
        self.assertEqual(agg["reqs"], [{"req": 1, "err": None,
                                        "anomaly": "early_stop",
                                        "in_text": "【user】肇事输入全文"}],
                         "留档 reqs 必须来自异常轮，非 decode 居中次")
        self.assertEqual([r["anomaly"] for r in agg["reps"]],
                         [None, "early_stop", None],
                         "reps 摘要逐次带 anomaly（无异常为 None）")
        # 对照：无异常复测维持 decode 居中口径、摘要 anomaly 全 None
        clean = _aggregate_reps([self._rep(50, req=0), self._rep(10, req=1),
                                 self._rep(30, req=2)])
        self.assertEqual(clean["reqs"][0]["req"], 2)
        self.assertTrue(all(r["anomaly"] is None for r in clean["reps"]))

    def test_total_s_aggregated(self):
        """agent 矩阵端到端总时长（ADR-0042）：聚合行 total_s 取均值 2 位小数，
        复测子行逐次留档 total_s。"""
        r1, r2 = self._rep(50), self._rep(10)
        r1["total_s"], r2["total_s"] = 1.234, 2.345
        agg = _aggregate_reps([r1, r2])
        self.assertEqual(agg["total_s"], 1.79)   # 均值 1.7895 → 2 位小数
        self.assertEqual([r["total_s"] for r in agg["reps"]], [1.234, 2.345])

    def test_majority_rule(self):
        ok = [self._rep(50), self._rep(10)]
        bad = self._rep(5, all_ok=False)
        self.assertTrue(_aggregate_reps(ok + [bad])["all_ok"])       # 2/3 多数
        self.assertFalse(_aggregate_reps([ok[0], bad, bad])["all_ok"])  # 1/3 少数

    def test_even_tie_is_failure(self):
        """all_ok 严格过半（>）：偶数平票判失败——repeats=2 时 1 成 1 败不算成功点。"""
        tie = _aggregate_reps([self._rep(50), self._rep(10, all_ok=False)])
        self.assertFalse(tie["all_ok"])

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

    def test_flush_mean_across_reps(self):
        """停滞回吐剔除量（flush_tok/flush_count/flush_s）与 stall_s 同取各次
        复测均值，并进入复测摘要 reps 字段清单；无回吐不产出该键（前端空值
        回退）。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0].update(flush_tok=100, flush_count=2, flush_s=1.5)
        reps[1].update(flush_tok=300, flush_count=2, flush_s=2.5)
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["flush_tok"], 200)             # (100+300)/2 均值
        self.assertEqual(agg["flush_count"], 2)
        self.assertEqual(agg["flush_s"], 2.0)               # (1.5+2.5)/2 均值
        self.assertEqual(agg["reps"][0]["flush_tok"], 100)  # 摘要逐次保真
        self.assertEqual(agg["reps"][0]["flush_s"], 1.5)
        self.assertIsNone(agg["reps"][2]["flush_tok"])      # 无回吐复测 = None
        clean = _aggregate_reps([self._rep(1), self._rep(2)])
        self.assertNotIn("flush_tok", clean)
        self.assertNotIn("flush_count", clean)
        self.assertNotIn("flush_s", clean)

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

    def test_all_failed_pool_falls_back_to_reps(self):
        """all_ok 全 False：聚合池回退为 reps 本身，标量均值照常算，all_ok=False。"""
        reps = [self._rep(50, all_ok=False), self._rep(10, all_ok=False),
                self._rep(30, all_ok=False)]
        agg = _aggregate_reps(reps)
        self.assertFalse(agg["all_ok"])
        self.assertEqual(agg["decode_tok_s"], 30)            # 回退池的均值
        self.assertEqual(agg["n_reps"], 3)

    def test_err_persisted_in_reps_summary_and_point(self):
        """失败原因落档：失败轮的 err（取自该轮 reqs 首个失败请求）进 reps
        摘要逐轮字段；聚合点 err = 失败轮错误文本去重拼接；全部成功则不设
        err 键（含代表样本带入残留键的显式清除）；reps 摘要 err 恒存在
        （无失败为 None，前端按空值回退）。"""
        ok1, ok2 = self._rep(50, req=0), self._rep(10, req=1)
        bad1 = self._rep(5, all_ok=False, req=2)
        bad1["reqs"] = [{"req": 0, "err": "HTTP 500 CUDA out of memory"}]
        bad2 = self._rep(3, all_ok=False, req=3)
        bad2["reqs"] = [{"req": 0, "err": "HTTP 500 CUDA out of memory"}]
        agg = _aggregate_reps([ok1, bad1, bad2])
        self.assertFalse(agg["all_ok"])
        # 同因失败去重：两轮同文错误只留一条
        self.assertEqual(agg["err"], "HTTP 500 CUDA out of memory")
        self.assertEqual([r["err"] for r in agg["reps"]],
                         [None, "HTTP 500 CUDA out of memory",
                          "HTTP 500 CUDA out of memory"])
        # 异因拼接（；分隔）
        bad2["reqs"] = [{"req": 0, "err": "连接被重置"}]
        agg2 = _aggregate_reps([ok1, bad1, bad2])
        self.assertEqual(agg2["err"], "HTTP 500 CUDA out of memory；连接被重置")
        # 全部失败（pool 回退 reps）：err 同样落档
        allbad = _aggregate_reps([bad1, bad2])
        self.assertEqual(allbad["err"], "HTTP 500 CUDA out of memory；连接被重置")
        # 全部成功：点不带 err 键、摘要 err 全 None
        clean = _aggregate_reps([ok1, ok2])
        self.assertNotIn("err", clean)
        self.assertTrue(all(r["err"] is None for r in clean["reps"]))

    def test_err_key_stripped_when_representative_carries_it(self):
        """代表样本（dict(rep_point) 拷贝）若带点级 err 键（单轮点装配位会
        设置），聚合后全成功须显式清除——否则失败轮的旧 err 污染干净聚合点。"""
        r1, r2 = self._rep(50), self._rep(10)
        r1["err"] = "历史残留"   # 模拟单轮点装配位写入的 err
        agg = _aggregate_reps([r1, r2])
        self.assertTrue(agg["all_ok"])
        self.assertNotIn("err", agg)

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
        """物理速率闸：prefill 速率超该未命中量的先验曲线 10× 或绝对上限
        10 万 tok/s 即判「物理上不可能」；未命中量 <8K 与无先验时都只按
        绝对上限判（小 prompt 区诚实速率随 TTFT 固定开销摊薄近线性爬升，
        10× 先验带无判别力——实测 agent 零缓存档 inst=1024/4096 的
        569~976 tok/s 被误杀，同运行 306~722 tok/s 的兄弟点反而过闸）。"""
        # 绝对上限（无先验也生效，x 大小无关）
        self.assertTrue(_is_implausible_prefill_rate(387516.0, 73728, []))
        self.assertTrue(_is_implausible_prefill_rate(100000.0, 16384, []))   # 边界
        self.assertTrue(_is_implausible_prefill_rate(100001.0, 100, []),
                        "小 prompt 也受绝对上限约束")
        # 先验倍数（x ≥8K 才走先验带）：5000 > 10×300 → 判；2500 < 3000 → 不判
        self.assertTrue(_is_implausible_prefill_rate(5000.0, 16384, [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(2500.0, 16384, [(2048, 300.0)]))
        # 下限边界：x=8192 恰好启用先验带，8191 只按绝对上限判
        self.assertTrue(_is_implausible_prefill_rate(5000.0, RATE_GATE_MIN_TOKENS,
                                                     [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(5000.0, 8191, [(2048, 300.0)]))
        # 小 prompt 区先验带误杀场景还原（实测 0K 档 inst=1024：速率 722、
        # 先验曲线底端 ~60）：不判
        self.assertFalse(_is_implausible_prefill_rate(722.0, 1024, [(0, 60.0)]))
        self.assertFalse(_is_implausible_prefill_rate(99000.0, 4096, [(2048, 300.0)]),
                         "低于绝对上限即过闸（软拒绝假成功本就发生在长 prompt）")
        # 无先验、低于绝对上限：不判
        self.assertFalse(_is_implausible_prefill_rate(5000.0, 16384, []))
        self.assertFalse(_is_implausible_prefill_rate(None, 16384, [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(0, 16384, [(2048, 300.0)]))


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
            # _one 统一 format(limit=…, aim=…)：translate/ocr 单向封顶无 {aim}，
            # 多余 kwargs 被 str.format 忽略——同一次调用
            # 全场景可跑（构造断言：不抛 KeyError 即通过）
            self.assertIsInstance(sc["out_hint"].format(limit=1234, aim=1000),
                                  str, key)
        # 中文场景按字/字符引导，英文 agent 场景按 characters
        self.assertIn("字", SCENARIOS["creative"]["out_hint"])
        self.assertIn("字符", SCENARIOS["code"]["out_hint"])
        self.assertIn("characters", SCENARIOS["agent"]["out_hint"])

    def test_creative_code_hint_bidirectional(self):
        """creative/code out_hint 双向引导（ADR-0016，与 agent 同推广）：aim
        目标 + 封顶子句齐备——单向封顶无下限引导，实测模型往远低于预算的
        长度收敛，短输出 decode 窗口不稳定。"""
        for key in ("creative", "code"):
            sc = SCENARIOS[key]
            self.assertIn("{aim}", sc["out_hint"])
            self.assertIn("{limit}", sc["out_hint"])
            hint = sc["out_hint"].format(limit=921, aim=768)
            # aim 目标句在前、封顶子句在后（引导为主、封顶兜底）
            self.assertLess(hint.index("约 768"), hint.index("不得超出"))
            # 封顶子句必须保留（防线 2 本职：服务端不截断时防写超预算跑飞）
            self.assertIn("不得超出", sc["out_hint"])

    def test_agent_hint_bidirectional(self):
        """agent out_hint 双向引导（ADR-0016 + 输出长度治理）：aim 目标 ≈
        limit ÷ OUT_HINT_FACTOR（limit 已含 1.2 余量）——防模型以「材料
        截断」为由如实短答（实测 308/310/401 tokens 就 finish=stop），单向
        封顶无下限引导罩不住。"""
        sc = SCENARIOS["agent"]
        self.assertIn("{aim}", sc["out_hint"])
        self.assertIn("{limit}", sc["out_hint"])
        # aim/limit 同源折算：tokens × out_cpt，limit 再 ×1.2 余量
        tokens = 512
        aim = int(tokens * sc["out_cpt"])
        limit = int(tokens * sc["out_cpt"] * OUT_HINT_FACTOR)
        self.assertEqual(aim, int(limit / OUT_HINT_FACTOR))
        hint = sc["out_hint"].format(limit=limit, aim=aim)
        self.assertIn(f"about {aim} characters", hint)
        self.assertIn(f"exceed {limit} characters", hint)
        # 引导方向：目标句在前、硬上限句在后（thorough 引导为主、封顶兜底）
        self.assertLess(hint.index("aiming"), hint.index("never exceed"))

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
        self.assertEqual(png[12:16], b"IHDR")
        self.assertEqual(struct.unpack(">II", png[16:24]), (1240, 1754))
        # 确定性：同 seed 同图
        self.assertEqual(png, _synth_document_png(1240, 1754, seed=0))

    def test_ocr_pool_fixed_three_resolutions(self):
        """OCR 图池固定内置合成（ADR-0037）：3 种分辨率文档页、确定性生成——
        跨环境统一负载标准，不再有用户文件通道。"""
        bench._IMG_CACHE = None
        pool = _ocr_images()
        self.assertEqual([(w, h) for _, w, h, _ in pool],
                         [(1240, 1754), (1654, 2339), (2480, 3508)])
        for data, w, h, mime in pool:
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (w, h))
            self.assertEqual(mime, "image/png")
        bench._IMG_CACHE = None
        self.assertEqual([d for d, *_ in _ocr_images()],
                         [d for d, *_ in pool])   # 重建逐字节一致（确定性）
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


class TestTranslateScenario(unittest.TestCase):
    """翻译场景（中→英，原文字长阶梯替代 ctx_list、矩阵可自定义）：构造形状、
    nonce 错开、输出预算随档伸缩且受全局 max_tokens 上界约束（ADR-0034）。"""

    def test_messages_shape_and_length(self):
        sc = SCENARIOS["translate"]
        msgs = _build_translate_messages(200, "12345")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        user = msgs[1]["content"]
        self.assertIn("12345", user[:20])   # 编号行在原文开头（bust 前缀缓存）
        self.assertTrue(user.endswith(sc["instruction"]))
        # 原文截取命中目标字符数（编号行 + 分隔 + 指令为固定开销）
        self.assertGreaterEqual(len(user), 200 + len(sc["instruction"]))
        self.assertLess(len(user), 200 + 64 + len(sc["instruction"]))

    def test_nonce_staggers_requests(self):
        a = _build_translate_messages(200, "a")[1]["content"]
        b = _build_translate_messages(200, "b")[1]["content"]
        self.assertNotEqual(a[:24], b[:24], "编号行使并发/复测请求文本错开")

    def test_max_tokens_scales_with_rung(self):
        budgets = [_translate_max_tokens(r, 1.5)
                   for r in SCENARIOS["translate"]["ladder"]]
        self.assertEqual(budgets, sorted(budgets))   # 随档单调递增
        # 6400 字原文译文 ~4-8K tokens：预算必须显著超过定长 1024，否则必截断
        self.assertGreater(budgets[-1], 4096)
        # 同档下更高的校准 cpt（每 token 字符更多）→ 原文 tokens 估值更低 → 预算更低
        self.assertLess(_translate_max_tokens(800, 3.0),
                        _translate_max_tokens(800, 1.5))

    def test_max_tokens_capped_by_global(self):
        """全局 max_tokens 对翻译场景同样生效（上界语义）：随档预算超出时按
        cap 截断，低档位预算不足 cap 时不受影响。"""
        uncapped = _translate_max_tokens(6400, 1.5)
        self.assertGreater(uncapped, 1024)
        self.assertEqual(_translate_max_tokens(6400, 1.5, cap=1024), 1024)
        low = _translate_max_tokens(50, 1.5)   # 50 字档预算 ~99，低于 cap
        self.assertEqual(_translate_max_tokens(50, 1.5, cap=1024), low)
        self.assertEqual(_translate_max_tokens(50, 1.5, cap=0), low)   # 无 cap

    def test_ladder_metadata(self):
        sc = SCENARIOS["translate"]
        self.assertEqual(sc.get("kind", "llm"), "llm")   # 与创意/代码同请求路径
        self.assertEqual(sc["ladder"], [50, 100, 200, 400, 800, 1600, 3200, 6400])
        self.assertEqual(sc["unit"], "字原文")
        self.assertIn("{limit}", sc["out_hint"])   # 防线 2 输出长度引导模板
        # 无 echo_instruction：echo 模板续写对翻译无意义，自动回退普通构造
        self.assertNotIn("echo_instruction", sc)


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

    def _req(self, first, last, out=512, stall=None, flush=None, flush_s=None):
        d = {"req": 0, "first_abs": first, "last_abs": last,
             "out_tokens": out, "stall_s": stall}
        if flush is not None:
            d["flush_tok"] = flush
        if flush_s is not None:
            d["flush_s"] = flush_s
        return d

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

    def test_flush_tok_subtracted_from_adj_numerator(self):
        """停滞回吐剔除（flush_tok）同步从 adj 分子扣除：停滞时间全额扣了而
        回吐 tokens 留在分子会双重虚高（真实事故：34 tok/s 被算成 216~333）。"""
        ok = [self._req(0.0, 10.0), self._req(1.0, 11.0, flush=300)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 1024 / 11.0, delta=0.05)
        self.assertAlmostEqual(adj, (1024 - 300) / 11.0, delta=0.05)

    def test_flush_tok_nonpositive_numerator_adj_none(self):
        """回吐剔除后分子 ≤0：adj 置空（raw 口径不动）。"""
        ok = [self._req(0.0, 10.0, flush=600), self._req(1.0, 11.0, flush=600)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 1024 / 11.0, delta=0.05)
        self.assertIsNone(adj)

    def test_flush_s_subtracted_from_window(self):
        """回吐交付时长（flush_s）撤出有效窗口：窗口 11s − stall 0 − flush_s
        2.0 = 9s；分子 1024−300 = 724 → adj = 80.4。"""
        ok = [self._req(0.0, 10.0, flush=300, flush_s=1.0),
              self._req(1.0, 11.0, flush_s=1.0)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 1024 / 11.0, delta=0.05)
        self.assertAlmostEqual(adj, (1024 - 300) / 9.0, delta=0.05)

    def test_adj_numerator_below_16_none(self):
        """剔除后分子 <16 tokens：adj 置空（健康段不足不硬出数，与 req 级
        同口径；实测事故：8 tokens/0.15s 曾读出 51.2 的垃圾总吞吐）。"""
        ok = [self._req(0.0, 10.0, flush=509), self._req(1.0, 11.0, flush=509)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 1024 / 11.0, delta=0.05)
        self.assertIsNone(adj)   # 分子 1024−1018 = 6 < 16

    def test_adj_window_below_0_2s_none(self):
        """剔除后有效窗口 <0.2s：adj 置空。"""
        ok = [self._req(0.0, 10.0, flush=100, flush_s=5.0),
              self._req(1.0, 11.0, flush_s=5.95)]
        raw, adj = _total_decode_rates(ok, 2)
        self.assertAlmostEqual(raw, 1024 / 11.0, delta=0.05)
        self.assertIsNone(adj)   # eff_window = 11−10.95 = 0.05 < 0.2

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


class TestFlushEpisodes(unittest.TestCase):
    """停滞回吐（stall-flush）识别纯函数：per-event token 速率判据（交付速率
    ≥ FLUSH_RATE_RATIO×r_pre），episode = ≥0.5s 停滞 + 紧邻 burst-like 事件段；
    门槛 Σ burst est_tok ≥ FLUSH_MIN_TOK，剔除封顶 r_pre×stall_s。
    ev_toks = 每事件 est tokens（Σ = out_tokens 真值归一后）。"""

    @staticmethod
    def _ones(gaps):
        """每事件 1 token 的辅助（事件数 = gaps 数 + 1）。"""
        return [1.0] * (len(gaps) + 1)

    def test_no_stall_empty(self):
        self.assertEqual(_flush_episodes([0.03] * 100, [1.0] * 101), [])

    def test_stall_then_flush_detected(self):
        """形态①（亚毫秒 1-token 串）：停滞 + 回吐串检出，下标/事件数/tokens
        正确（r_pre = 9/0.27 ≈ 33.3，cap 450 不封顶）。"""
        gaps = [0.03] * 9 + [13.5] + [0.0001] * 429
        eps = _flush_episodes(gaps, [1.0] * 440)
        self.assertEqual(len(eps), 1)
        e = eps[0]
        self.assertEqual(e["stall_idx"], 9)
        self.assertEqual(e["start_idx"], 10)
        self.assertEqual(e["end_idx"], 440)
        self.assertEqual(e["burst_events"], 430)
        self.assertAlmostEqual(e["burst_tok"], 430.0)

    def test_small_flush_not_detected(self):
        """停滞 + 小回吐（Σ <8 tokens）：不认定——正常投机交付与记账噪声
        都不动数。"""
        gaps = [0.03] * 9 + [0.6] + [0.0001] * 4
        self.assertEqual(_flush_episodes(gaps, [1.0] * 15), [])

    def test_real_mtp_rhythm_not_detected(self):
        """真 MTP 节奏：均匀 ~30ms 批界中夹 2~4 个亚毫秒串、前置无 ≥0.5s
        静默——绝不误伤（关键区分特征：回吐段前面必有大停滞）。"""
        gaps = []
        for _ in range(20):
            gaps.append(0.03)
            gaps.extend([0.0001] * 3)
        self.assertEqual(_flush_episodes(gaps, [1.0] * (len(gaps) + 1)), [])

    def test_multiple_episodes(self):
        """多次停滞回吐：各自独立检出，已归属事件不重复计（第二个停滞的
        r_pre 排除第一段回吐事件）。"""
        gaps = ([0.03] * 5 + [1.2] + [0.0001] * 9
                + [0.03] * 5 + [2.0] + [0.0001] * 9)
        eps = _flush_episodes(gaps, [1.0] * 31)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["stall_idx"], 5)
        self.assertEqual(eps[0]["start_idx"], 6)
        self.assertEqual(eps[0]["end_idx"], 16)
        self.assertEqual(eps[0]["burst_events"], 10)
        self.assertEqual(eps[1]["stall_idx"], 20)
        self.assertEqual(eps[1]["burst_events"], 10)

    def test_missing_inputs_conservative(self):
        """事件序列缺省/长度不配：保守返回无 episode。"""
        gaps = [0.03, 13.5] + [0.0001] * 20
        self.assertEqual(_flush_episodes(gaps, []), [])
        self.assertEqual(_flush_episodes([], [1.0] * 5), [])
        self.assertEqual(_flush_episodes(gaps, [1.0] * (len(gaps) + 2)), [])
        self.assertEqual(_flush_episodes(gaps, [0.0] * (len(gaps) + 1)), [])

    def test_merged_3tok_8ms_detected(self):
        """形态②（~8ms 3-token 合并串）：速率判据检出，per-event 实测累计
        直接给出真实回吐量（round-2 的均摊+积压 hack 退役）。"""
        gaps = [0.029] * 79 + [6.0] + [0.008] * 58
        ev_toks = [1.0] * 80 + [3.0] * 59   # 健康 1 tok/事件，回吐 3 tok/事件
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        e = eps[0]
        self.assertEqual(e["stall_idx"], 79)
        self.assertEqual(e["start_idx"], 80)
        self.assertEqual(e["end_idx"], 139)
        self.assertEqual(e["burst_events"], 59)
        self.assertAlmostEqual(e["burst_tok"], 177.0, delta=1)   # 真实回吐量
        self.assertAlmostEqual(e["burst_s"], 0.464, places=2)

    def test_big_merge_26tok_22ms_detected(self):
        """形态③（~22ms 26-token 大合并，间隔判据原理性漏检的形态）：
        交付 ~1180 tok/s vs 健康 ~34——速率判据必抓；Σ = 9×26 = 232 被
        r_pre×stall 封顶（健康 34.5 tok/s × 6.61s ≈ 227.9，真实案例的
        232 也同样轻微越界被截）。"""
        gaps = [0.029] * 23 + [6.61] + [0.022] * 8
        ev_toks = [1.0] * 24 + [26.0] * 9
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        e = eps[0]
        self.assertEqual(e["stall_idx"], 23)
        self.assertEqual(e["burst_events"], 9)
        self.assertAlmostEqual(e["burst_tok"], (23 / (23 * 0.029)) * 6.61,
                               delta=0.1)

    def test_mtp_single_batch_after_hiccup_not_detected(self):
        """MTP 单批 k≈3 打嗝后到达：即使亚毫秒到达 Σ≈3 <8 不认定（生成
        速率与批节奏均正常，非积压）。"""
        us = 0.0001
        gaps = ([0.03, us, us] * 4 + [0.03, us]     # 事件 0..14：5 批 k=3
                + [0.6]                              # 打嗝停滞（事件 15 前）
                + [us, us])                          # 单批 3 事件亚毫秒到达
        ev_toks = [1.0] * 18
        self.assertEqual(_flush_episodes(gaps, ev_toks), [])

    def test_mtp_multi_batch_burst_detected(self):
        """MTP ≥3 批连冲（≥9 tokens）：本就是积压，认定正确。"""
        us = 0.0001
        gaps = [0.03, us, us] * 4 + [0.03, us] + [0.6] + [us] * 8
        ev_toks = [1.0] * 24
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["burst_events"], 9)
        self.assertAlmostEqual(eps[0]["burst_tok"], 9.0)

    def test_zero_content_events_do_not_break_run(self):
        """est≈0 的零内容/keepalive 事件跳过不中断回吐串（否则串被截断、
        剔除量减半）。"""
        gaps = [0.03] * 9 + [6.0] + [0.008] * 3
        ev_toks = [1.0] * 10 + [10.0, 0.0, 10.0, 10.0]   # 14 事件
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        e = eps[0]
        self.assertEqual(e["start_idx"], 10)
        self.assertEqual(e["end_idx"], 14)
        self.assertAlmostEqual(e["burst_tok"], 30.0, delta=0.5)

    def test_flush_capped_at_rpre_times_stall(self):
        """防过剔上限：回吐量明显超过 r_pre×stall_s → 封顶（健康 ~33 tok/s
        × 1.0s 停滞 ≈ 33.3，回吐 100 tokens 只剔 33.3）。"""
        gaps = [0.03] * 9 + [1.0] + [0.0001] * 99
        eps = _flush_episodes(gaps, [1.0] * 110)   # r_pre = 9/0.27 ≈ 33.3
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["burst_events"], 100)
        self.assertAlmostEqual(eps[0]["burst_tok"], (9 / (9 * 0.03)) * 1.0,
                               delta=0.5)

    def test_flush_below_cap_threshold_abandoned(self):
        """封顶后 < FLUSH_MIN_TOK：放弃认定——停滞期最多生成 r×S，封顶值
        太小（<8 tokens）不值得剔除。"""
        gaps = [0.1] * 19 + [0.5] + [0.0001] * 49
        # Σ burst = 50；r_pre = 19/1.9 = 10，cap = 0.5×10 = 5 < 8 → 不认定
        self.assertEqual(_flush_episodes(gaps, [1.0] * 70), [])

    def test_r_pre_missing_falls_back_to_subms_absolute(self):
        """r_pre 缺失（停滞在流首、无停滞前正常事件）：仅按亚毫秒绝对规则
        放行，保守不误伤——8ms 合并串不检出、亚毫秒串照常检出。"""
        # 8ms 串 > 2ms：不检出（r_pre 缺失时速率判据不可用）
        gaps = [0.6] + [0.008] * 20
        self.assertEqual(_flush_episodes(gaps, [1.0] * 22), [])
        # 亚毫秒串 < 2ms：照常检出（无 r_pre 不封顶，Σ = 21）
        gaps = [0.6] + [0.0001] * 20
        eps = _flush_episodes(gaps, [1.0] * 22)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["burst_events"], 21)
        self.assertAlmostEqual(eps[0]["burst_tok"], 21.0)

    def test_rate_criterion_works_with_few_normal_events(self):
        """停滞前正常事件 <30（甚至 5 个）：r_pre 照常折算，速率判据可用
        （round-2 的样本数下限退役）。8ms×1tok 回吐（速率 125 ≥ 3×20）检出，
        封顶 0.6×20=12。"""
        gaps = [0.05] * 5 + [0.6] + [0.008] * 20
        eps = _flush_episodes(gaps, [1.0] * 27)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["burst_events"], 21)
        self.assertAlmostEqual(eps[0]["burst_tok"], 12.0, delta=0.5)

    def test_tail_delivery_continues_at_1_2x(self):
        """拖尾形态（20260914-115607 inst=1024 案例）：回吐先以 ~5× 猛冲、
        后以 ~2× 降速拖尾——延续条件放宽到 >1.2× 后整段剔除（round-3 的 3×
        延续在拖尾处断链、~55 tokens 漏剔 → adj 69.8 而真实 ~32）。"""
        gaps = ([0.028] * 22 + [6.8]                 # 健康 23 事件 + 停滞
                + [0.15] * 6                         # 猛冲段：25 tok/0.15s ≈ 5×
                + [0.15] * 5)                        # 拖尾段：11 tok/0.15s ≈ 2.2×
        ev_toks = ([1.0] * 23 + [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 28.0]
                   + [11.0] * 5)
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        e = eps[0]
        self.assertEqual(e["burst_events"], 12)      # 猛冲 7 + 拖尾 5 全段
        self.assertAlmostEqual(e["burst_tok"], 233.0, delta=1)   # Σ 未封顶
        self.assertAlmostEqual(e["burst_s"], 11 * 0.15, delta=0.01)

    def test_no_backlog_stall_not_detected(self):
        """无积压真停滞：服务端 hang 后恢复正常交付（速率 =r_pre ≤1.2×）→
        段只有停滞段首事件、Σ <8 不认定。"""
        gaps = [0.03] * 9 + [6.0] + [0.03] * 10
        self.assertEqual(_flush_episodes(gaps, [1.0] * 20), [])

    def test_tail_sustained_capped_at_rpre_times_stall(self):
        """cap 护栏：拖尾交付速率持续 1.5× 不降（合规延续）但总剔除量超过
        r_pre×S → 封顶。头 3×25 tok @0.15s（5×）+ 尾 40×2 tok @0.04s
        （1.5× 持续）；Σ = 155 > cap = 33.3×1.0。"""
        gaps = [0.03] * 9 + [1.0] + [0.15] * 2 + [0.04] * 40
        ev_toks = [1.0] * 10 + [25.0] * 3 + [2.0] * 40
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["burst_events"], 43)
        self.assertAlmostEqual(eps[0]["burst_tok"], (9 / (9 * 0.03)) * 1.0,
                               delta=0.5)


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
    """_detect_rep_anomaly：early_stop 阈值 = max(比例, 有效地板)——echo
    0.8×max_tokens、free 不按比例只按地板（自主收尾合法，但输出几个字
    就停完全不可信，2026-09-16 拍板）、finish=length 不判、全 err 不判、
    退化重复两模式通用且 text/reason 通道任一命中。"""

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

    def test_free_below_floor_early_stop(self):
        # free 模式不按比例罚，但 4 < 有效地板 100：完全不可信同判
        reqs = [self._req(out_tokens=4)]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=False),
                         ("early_stop", 0))

    def test_free_above_floor_no_early_stop(self):
        # free 模式 150 ≥ 地板 100（虽 <0.8×512）：自主收尾合法，不判
        reqs = [self._req(out_tokens=150)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=False))

    def test_echo_above_floor_below_ratio_still_flagged(self):
        # echo 模式 150 ≥ 地板但 <0.8×512=409.6：续写口径仍判
        reqs = [self._req(out_tokens=150)]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=True),
                         ("early_stop", 0))

    def test_echo_within_threshold_none(self):
        # 480/512 超过 0.8×512=409.6：预算内的正常收尾，不判
        reqs = [self._req(out_tokens=480)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_floor_capped_by_budget(self):
        # 小预算地板随预算钳制：max_tokens=32 时地板=32，满预算输出不判
        reqs = [self._req(out_tokens=32)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 32, echo_mode=False))
        reqs = [self._req(out_tokens=31)]
        self.assertEqual(_detect_rep_anomaly(reqs, 32, echo_mode=False),
                         ("early_stop", 0))

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
    """_run_rep_guarded：弃测重测编排——重测预算按异常形态分两档（低于
    有效地板的早停最多 FLOOR_RETEST_MAX 次、其余异常最多
    ANOMALY_RETEST_MAX 次，重测换 nonce 破缓存 + stream_salt 换语料
    落点由 _run_point 内部落实）、预算耗尽仍异常高于地板按实留档
    anomaly、仍低于地板按测量失败留档（读数置空 + err）、首次正常不重测、
    stop_flag 置位不检测不重测。_run_point 打桩脚本化返回，emit 收队列。"""

    ANOM_PT = {   # 实测案例形态：4K 档 echo 模式仅输出 4/512 tokens
        "reqs": [{"err": None, "finish": "stop", "out_tokens": 4,
                  "ttft_s": 0.12, "decode_tok_s": 29.1,
                  "text_sample": "好的，以下是", "reason_sample": "",
                  "in_sample": "指令材料头部",
                  "_in_full": "【user】" + "指令材料" * 150}]}
    OK_PT = {
        "reqs": [{"err": None, "finish": "stop", "out_tokens": 512,
                  "ttft_s": 0.11, "decode_tok_s": 60.0,
                  "text_sample": "", "reason_sample": "",
                  "_in_full": "【user】正常输入全文"}]}

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
        （用于模拟测量期间置位 stop_flag）。reqs 逐个浅拷贝——真实 _run_point
        每批返回全新 req 字典，共享 fixture 的 reqs 列表会让守卫对肇事 req 的
        anomaly/in_text 标记跨用例/跨调用串扰。"""
        calls = []

        async def fake_run_point(*args, **kwargs):
            i = len(calls)
            calls.append((args, kwargs))
            src = points[min(i, len(points) - 1)]
            pt = {**src, "reqs": [dict(r) for r in src["reqs"]]}
            if post:
                post(i)
            return pt

        run._run_point = fake_run_point
        return calls

    async def test_anomaly_then_ok_retest(self):
        """首次 early_stop 异常 → 弃测重测一次 → 重测正常：返回正常 pt 带
        reps_discarded（1 条，含肇事读数与样本、rep 轮次号、in_text 输入全文
        抄本）、无 anomaly 键、播报状态；返回 pt 的 reqs 已剥离 _in_full 瞬态
        字段（正常批存档不得带输入全文）。"""
        run, statuses = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 2, "异常应弃测重测一次")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt.get("reps_discarded"), [{
            "anomaly": "early_stop", "req": 0, "rep": 1,
            "out_tokens": 4,
            "finish": "stop", "ttft_s": 0.12, "decode_tok_s": 29.1,
            "text_sample": "好的，以下是", "reason_sample": "",
            "in_sample": "指令材料头部",
            "in_text": "【user】" + "指令材料" * 150}])
        self.assertGreater(len(pt["reps_discarded"][0]["in_text"]), 200,
                           "in_text 为输入全文抄本，不设 200 字符采样上限")
        self.assertNotIn("_in_full", pt["reqs"][0],
                         "返回 pt 的 reqs 应已剥离 _in_full 瞬态字段")
        self.assertTrue(any("仅输出 4/512 tokens" in s and "弃测重测" in s
                            for s in statuses), f"应有弃测播报: {statuses}")

    async def test_anomaly_persists_marked(self):
        """重测仍异常但高于有效地板：接受结果、pt 带 anomaly='early_stop'
        按实留档、reps_discarded 仅 1 条（不无限重试）；肇事 req 打
        anomaly + in_text（输入全文抄本），全部 req 剥离 _in_full。"""
        run, _ = self._make_run()
        above = {"reqs": [dict(self.ANOM_PT["reqs"][0], out_tokens=200)]}
        calls = self._script(run, [dict(self.ANOM_PT), above])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1 + ANOMALY_RETEST_MAX)
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), ANOMALY_RETEST_MAX)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop",
                         "重测仍异常的肇事 req 应带 anomaly 标记")
        self.assertIsNone(culprit.get("err"),
                          "高于地板的持续异常不按测量失败留档")
        self.assertNotIn("err", pt)
        self.assertEqual(culprit["in_text"], "【user】" + "指令材料" * 150,
                         "肇事 req 应带输入全文 in_text")
        self.assertNotIn("_in_full", culprit, "_in_full 剥离后全文留在 in_text")

    async def test_anomaly_persists_below_floor_failed(self):
        """重测预算（FLOOR_RETEST_MAX=2，2026-09-17 收紧）耗尽仍低于有效
        输出地板（4 < 100）：读数完全不可信——按测量失败留档：pt err 落档、
        all_ok=False、点读数置空不进聚合；肇事 req 带 err + anomaly +
        in_text；reps_discarded 逐条留痕每次弃测（ADR-0032 的按实留档
        宽容止于地板之上）。"""
        run, statuses = self._make_run()
        anom_full = {**self.ANOM_PT, "out_tokens": 4, "decode_tok_s": 29.1,
                     "prefill_tok_s": 300.0, "all_ok": True}
        calls = self._script(run, [dict(anom_full), dict(anom_full)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1 + FLOOR_RETEST_MAX,
                         "低于地板的早停最多重测 FLOOR_RETEST_MAX 次")
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertFalse(pt["all_ok"])
        self.assertIn("早停", pt["err"])
        self.assertIsNone(pt["out_tokens"], "垃圾读数应置空不进聚合")
        self.assertIsNone(pt["decode_tok_s"])
        self.assertIsNone(pt["prefill_tok_s"])
        self.assertEqual(len(pt["reps_discarded"]), FLOOR_RETEST_MAX)
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop")
        self.assertIn("早停", culprit["err"])
        self.assertEqual(culprit["in_text"], "【user】" + "指令材料" * 150)
        self.assertTrue(any("测量失败留档" in s for s in statuses),
                        f"应有失败留档播报: {statuses}")

    async def test_below_floor_recovers_on_later_retest(self):
        """低于地板的早停在第 2 次重测恢复：持续重测直到正常——返回正常
        pt（无 anomaly 键）、reps_discarded 逐条留痕 2 次弃测、读数来自
        恢复批；高于地板的异常无此宽限（仍 1 次，见上例）。"""
        run, _ = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.ANOM_PT),
                                   dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 3, "首测 + 2 次弃测重测后恢复")
        self.assertNotIn("anomaly", pt)
        self.assertNotIn("err", pt)
        self.assertEqual(len(pt["reps_discarded"]), 2)
        self.assertTrue(all(d["anomaly"] == "early_stop"
                            for d in pt["reps_discarded"]))
        self.assertEqual(pt["reqs"][0]["out_tokens"], 512,
                         "最终读数来自恢复批")

    async def test_retest_passes_stream_salt(self):
        """弃测重测换料（2026-09-17 拍板）：首测不带 stream_salt（=0，嵌套
        前缀确定性不变），每次重测带递增 stream_salt 驱动 _run_point 平移
        语料窗口翻 echo 落点——确定型早停与落点绑定，同料重测只会复现
        同一答案。"""
        run, _ = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.ANOM_PT),
                                   dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("stream_salt", calls[0][1],
                         "首测不传盐（等价 salt=0，保持嵌套前缀口径）")
        self.assertEqual(calls[1][1].get("stream_salt"), 1)
        self.assertEqual(calls[2][1].get("stream_salt"), 2)
        self.assertNotIn("anomaly", pt)

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


class TestAgentMatrixAnomalyGuard(unittest.IsolatedAsyncioTestCase):
    """_run_agent_matrix 的输出异常弃测重测（ADR-0032 口径接入 agent 矩阵）：
    测量批早停（free 阈值 0.5×）/退化命中 → 整批弃测、换新 nonce+错位指令
    材料重测一次（reps_discarded 留痕、点干净）、重测仍异常按实留档
    anomaly；与指令长度偏离校正的次序为先校正后异常判定（各 1 次、总尝试
    ≤3 次）。_one 打桩脚本化返回（quiet 预热/基线批固定返回），emit 收队列。"""

    MODEL = "m"
    INST = 512

    def _make_run(self):
        run = BenchRun({"max_tokens": 512, "agent_cache_ladder": [0],
                        "agent_inst_ladder": [self.INST]})
        statuses = []

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run.emit = fake_emit
        return run, statuses

    def _script_one(self, run, base_prompt, meas_script):
        """_one 打桩：quiet 预热/基线批固定返回 base_prompt 实测 tokens；
        测量批（quiet=False）按 meas_script 逐次返回（元素为对基准 req 字典
        的键覆写；耗尽后重复最后一项）。calls 记录 (quiet, 末条消息) 供断言
        调用序与 nonce tag。"""
        calls = []

        def base_req(req_i, mt):
            return {"req": req_i, "err": None, "usage_real": True,
                    "finish": "stop", "out_tokens": mt,
                    "prompt_tokens": 0, "ttft_s": 0.05, "decode_tok_s": 60.0,
                    "text_sample": "OK", "reason_sample": "",
                    "in_sample": "指令材料头部",
                    "_in_full": "【user】指令输入全文",
                    "max_gap_s": None, "cache_reported": False}

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False, **kw):
            calls.append((quiet, msgs))   # msgs[1] 带编号 nonce、msgs[2] 为指令消息
            if quiet:
                return {"req": req_i, "err": None,
                        "prompt_tokens": base_prompt, "usage_real": True}
            i = min(len([c for c in calls if not c[0]]) - 1,
                    len(meas_script) - 1)
            r = base_req(req_i, max_tokens)
            r["prompt_tokens"] = base_prompt + self.INST   # 差分恰中档位目标
            r.update(meas_script[i])
            return r

        run._one = fake_one
        return calls

    async def _run_matrix(self, run):
        await run._run_agent_matrix(None, self.MODEL, "agent", 1, 3.5,
                                    None, 1)
        self.assertTrue(run.results, "repeats=1 应即时落盘出点")
        return run.results[0]

    async def test_early_stop_retest_clean(self):
        """首测早停（out 4/512 < 0.5×）→ 弃测整批重测一次 → 点干净（无
        anomaly 键）、reps_discarded 恰 1 条（0032 形状含肇事读数与样本）、
        读数来自重测正常批、播报弃测状态。"""
        run, statuses = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "The task context is incomplete.",
             "decode_tok_s": 30.0},
            {"out_tokens": 512, "decode_tok_s": 60.0},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 4, "零缓存档：基线+测量 ×（首测+重测）")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt["reps_discarded"], [{
            "anomaly": "early_stop", "req": 0, "rep": 1,
            "out_tokens": 4,
            "finish": "stop", "ttft_s": 0.05, "decode_tok_s": 30.0,
            "text_sample": "The task context is incomplete.",
            "reason_sample": "", "in_sample": "指令材料头部",
            "in_text": "【user】指令输入全文"}])
        self.assertNotIn("_in_full", pt["reqs"][0],
                         "点位 reqs 应剥离 _in_full 瞬态字段")
        self.assertEqual(pt["out_tokens"], 512, "最终读数来自重测正常批")
        self.assertEqual(pt["finish"], "stop")
        self.assertTrue(pt["all_ok"])
        self.assertTrue(any("提前停止：仅输出 4/512 tokens" in s
                            and "弃测重测" in s for s in statuses), statuses)

    async def test_persistent_early_stop_marked(self):
        """重测仍早停但高于有效地板（200 ≥ min(100,512)，仍 <0.5×512 判据）：
        anomaly='early_stop' 按实留档、reps_discarded 仅 1 条（最多重测 1 次，
        不无限重试洗掉）、出点不中断；留档样本取自异常轮——肇事 req 带
        anomaly + in_text（输入全文抄本），全部 req 剥离 _in_full。"""
        run, _ = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "short answer"},
            {"out_tokens": 200, "text_sample": "still short but above floor"},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 4, "首测 + 异常重测各一轮（基线+测量）")
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), 1)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop",
                         "重测仍异常的肇事 req 应带 anomaly 标记")
        self.assertIsNone(culprit.get("err"),
                          "高于地板的持续异常不按测量失败留档")
        self.assertEqual(culprit["in_text"], "【user】指令输入全文",
                         "肇事 req 应带输入全文 in_text")
        self.assertFalse(any("_in_full" in r for r in pt["reqs"]),
                         "点位 reqs 应全部剥离 _in_full 瞬态字段")
        self.assertTrue(pt["all_ok"])

    async def test_persistent_early_stop_below_floor_failed(self):
        """重测预算（FLOOR_RETEST_MAX=2，2026-09-17 收紧）耗尽仍低于有效
        输出地板（7 < min(100,512)=100）：读数完全不可信——按测量失败留档：
        pt err 落档、all_ok=False、读数置空不进聚合；anomaly 标记与
        reps_discarded（逐条留痕每次弃测）保留供复核。"""
        run, _ = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "short answer"},
            {"out_tokens": 7, "text_sample": "short answer again"},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 2 * (1 + FLOOR_RETEST_MAX),
                         "首测 + FLOOR_RETEST_MAX 次弃测重测（基线+测量）")
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), FLOOR_RETEST_MAX)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)
        self.assertFalse(pt["all_ok"], "重测仍低于地板：点按测量失败留档")
        self.assertIn("早停", pt["err"])
        self.assertIsNone(pt["out_tokens"], "垃圾读数应置空不进聚合")
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop")
        self.assertIn("早停", culprit["err"])
        self.assertEqual(culprit["in_text"], "【user】指令输入全文")
        self.assertFalse(any("_in_full" in r for r in pt["reqs"]))

    async def test_degenerate_retest(self):
        """退化重复（文本通道样本 ≥80 字符且字符集 ≤12）弃测重测：留痕
        anomaly='degenerate'、重测正常即点干净。"""
        run, statuses = self._make_run()
        degen = " " + " ".join(f"{i}." for i in range(6, 60))
        calls = self._script_one(run, 300, [
            {"out_tokens": 512, "text_sample": degen},
            {"out_tokens": 512, "text_sample": "OK"},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 4)
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt["reps_discarded"][0]["anomaly"], "degenerate")
        self.assertEqual(pt["out_tokens"], 512)
        self.assertTrue(any("退化重复" in s and "弃测重测" in s
                            for s in statuses), statuses)

    async def test_input_correction_then_anomaly(self):
        """次序与有界性：指令长度偏离先触发输入校正（nonce tag 'r'）、校正批
        又早停才触发输出异常重测（tag 逐次 'a0'/'a1'/…，2026-09-16 地板
        多次重试换盐）——共 3 次测量尝试；重测正常即点干净且
        inst_real_tokens 收敛。"""
        run, statuses = self._make_run()
        calls = self._script_one(run, 300, [
            {"prompt_tokens": 600, "out_tokens": 512},    # 差分 300，偏离 -41%
            {"out_tokens": 4, "text_sample": "incomplete"},   # 校正批早停
            {"prompt_tokens": 812, "out_tokens": 512},        # 异常重测正常
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 6, "基线+测量 ×3（首测/校正/异常重测）")
        tags = [re.search(rf"-z{self.INST}(\w*)）：", msgs[1]["content"]).group(1)
                or ""
                for quiet, msgs in calls if not quiet]
        self.assertEqual(tags, ["", "r", "a0"],
                         f"先输入校正（r）后异常重测（a0）: {tags}")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(len(pt["reps_discarded"]), 1)
        self.assertIsNotNone(pt["inst_real_tokens"])
        self.assertLess(abs(pt["inst_real_tokens"] - self.INST),
                        0.1 * self.INST + 8, pt["inst_real_tokens"])

    async def test_retest_shifts_instruction_material(self):
        """异常重测换料：指令材料在语料流中错位取段（seg_salt），长度不变、
        内容不同——防整条 prompt 二次命中缓存。"""
        run, _ = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "x"},
            {"out_tokens": 512, "text_sample": "OK"},
        ])
        await self._run_matrix(run)
        meas = [msgs[2]["content"] for quiet, msgs in calls if not quiet]
        seg1 = meas[0][len(SCENARIOS["agent"]["inst_prefix"]):]
        seg2 = meas[1][len(SCENARIOS["agent"]["inst_prefix"]):]
        self.assertEqual(len(seg1), len(seg2), "指令材料长度不变")
        self.assertNotEqual(seg1, seg2, "指令材料内容错位（换料防二次命中缓存）")


class TestAgentMatrixCachePollution(unittest.IsolatedAsyncioTestCase):
    """_run_agent_matrix 的指令段缓存残留污染兜底重测（C>0 回传命中批）：
    实测未命中 tokens 远低于指令档期望（_cache_pollution_verdict）→ 整批
    弃测、游标换新段重测一次（reps_discarded 标 cache_pollution 带实测
    uncached/expected、两次尝试指令材料错位）；重测仍偏差过大
    point["anomaly"] 按实留档不再重试；干净批不触发。_one 打桩脚本化
    cache_hit，emit 收队列。"""

    MODEL = "m"
    PRIME = 15000   # 预热实测 prompt（缓存档上下文 tokens）
    INST = 2048

    def _make_run(self):
        run = BenchRun({"max_tokens": 512, "agent_cache_ladder": [4096],
                        "agent_inst_ladder": [self.INST]})
        statuses = []

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run.emit = fake_emit
        return run, statuses

    def _script_one(self, run, hit_script):
        """_one 打桩：quiet 预热批固定返回 PRIME 实测 prompt；测量批
        （quiet=False）prompt = PRIME + INST（差分恰中档位目标、不触发
        输入校正），cache_hit 按 hit_script 逐次返回（耗尽重复最后一项）。
        calls 记录 (quiet, msgs) 供断言调用序与指令材料错位。"""
        calls = []

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False, **kw):
            calls.append((quiet, msgs))
            if quiet:
                return {"req": req_i, "err": None,
                        "prompt_tokens": self.PRIME, "usage_real": True}
            i = min(len([c for c in calls if not c[0]]) - 1,
                    len(hit_script) - 1)
            return {"req": req_i, "err": None, "usage_real": True,
                    "finish": "stop", "out_tokens": max_tokens,
                    "prompt_tokens": self.PRIME + self.INST,
                    "ttft_s": 0.05, "decode_tok_s": 60.0,
                    "cache_reported": True, "cache_hit": hit_script[i],
                    "text_sample": "OK", "reason_sample": "",
                    "in_sample": "指令材料头部",
                    "_in_full": "【user】指令输入全文",
                    "max_gap_s": None}

        run._one = fake_one
        return calls

    async def _run_matrix(self, run):
        await run._run_agent_matrix(None, self.MODEL, "agent", 1, 3.5,
                                    None, 1)
        self.assertTrue(run.results, "repeats=1 应即时落盘出点")
        return run.results[0]

    async def test_pollution_retest_clean(self):
        """首测批未命中 500（期望 2048，比值 0.24）→ 判污染弃测、游标换段
        重测一次 → 重测干净：点无 anomaly、reps_discarded 恰 1 条带实测
        uncached/expected、两次尝试指令材料错位不重叠、播报换料重测。"""
        run, statuses = self._make_run()
        calls = self._script_one(
            run, [self.PRIME + self.INST - 500, self.PRIME])
        pt = await self._run_matrix(run)
        meas = [msgs[2]["content"] for quiet, msgs in calls if not quiet]
        self.assertEqual(len(meas), 2, "首测 + 污染重测各一轮测量批")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt["reps_discarded"], [{
            "anomaly": "cache_pollution", "rep": 1,
            "uncached_tokens": 500, "expected_tokens": self.INST}])
        self.assertNotEqual(meas[0], meas[1],
                            "污染重测指令材料应随游标错位换段")
        pre, suf = (SCENARIOS["agent"]["inst_prefix"],
                    SCENARIOS["agent"]["inst_suffix"])
        seg1 = meas[0][len(pre):len(meas[0]) - len(suf)]
        seg2 = meas[1][len(pre):len(meas[1]) - len(suf)]
        self.assertFalse(seg2.startswith(seg1[:200]),
                         "两次尝试指令材料不互为前缀")
        self.assertEqual(pt["inst_real_tokens"], self.INST)
        self.assertTrue(any("实测未命中 500 tokens 与指令档期望 2048 偏差过大"
                            in s and "换料重测" in s for s in statuses),
                        statuses)

    async def test_pollution_persistent_marked(self):
        """重测仍偏差过大：point["anomaly"]="cache_pollution" 按实留档、
        reps_discarded 仅 1 条（最多重测 1 次，不无限重试洗掉）、出点不
        中断。"""
        run, _ = self._make_run()
        calls = self._script_one(
            run, [self.PRIME + self.INST - 500, self.PRIME + self.INST - 500])
        pt = await self._run_matrix(run)
        self.assertEqual(len([c for c in calls if not c[0]]), 2,
                         "污染重测仅一次")
        self.assertEqual(pt["anomaly"], "cache_pollution")
        self.assertEqual(len(pt["reps_discarded"]), 1)
        self.assertEqual(pt["reps_discarded"][0]["anomaly"], "cache_pollution")
        self.assertTrue(pt["all_ok"])

    async def test_clean_batch_no_retest(self):
        """干净批（未命中 ≈ 指令档期望）不触发：仅一轮测量批、无
        reps_discarded 键。"""
        run, _ = self._make_run()
        calls = self._script_one(run, [self.PRIME])
        pt = await self._run_matrix(run)
        self.assertEqual(len([c for c in calls if not c[0]]), 1)
        self.assertNotIn("reps_discarded", pt)
        self.assertNotIn("anomaly", pt)


class TestPrefillGateCallSite(unittest.IsolatedAsyncioTestCase):
    """物理速率闸调用点（_one 收口）：x 取未命中 prompt tokens（缓存回传时
    = prompt − cache_hit，否则全量）、未命中量 <8K 只按绝对上限判——走真实
    _one 请求链路（假 client 定时流控制 TTFT→rate），先验曲线预置紧带。"""

    MODEL = "m"
    PRIOR = [(2048, 300.0)]   # 10× 带 = 3000 tok/s

    class _Resp:
        def __init__(self, lines):
            self.status_code = 200
            self.headers = {}
            self._lines = lines   # async 迭代器（定时流控制 TTFT）

        async def aiter_lines(self):
            async for ln in self._lines:
                yield ln

        async def aread(self):
            return b""

    class _Ctx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, resp):
            self.resp = resp

        def stream(self, *a, **k):
            return TestPrefillGateCallSite._Ctx(self.resp)

    @staticmethod
    def _sse(prompt_tokens, completion_tokens, delay_s, cache_hit=0):
        """定时 SSE：首块延迟 delay_s 秒（TTFT≈delay → rate≈prompt/delay）。
        usage 回传 cache_hit（缓存命中口径用例用）。"""
        async def lines():
            await asyncio.sleep(delay_s)
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "测"}}]})
            yield "data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt_tokens,
                           "completion_tokens": completion_tokens,
                           "prompt_cache_hit_tokens": cache_hit}})
            yield "data: [DONE]"
        return lines()

    async def _one_ok(self, usage, delay):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u" * 40}]
        run = BenchRun({})
        run.prefill_curve[(self.MODEL, "agent")] = list(self.PRIOR)
        events = []

        async def fake_emit(ev):
            events.append(ev)

        run.emit = fake_emit
        r = await run._one(self._Client(self._Resp(self._sse(
            usage["prompt_tokens"], usage["completion_tokens"], delay,
            usage.get("prompt_cache_hit_tokens", 0)))),
            self.MODEL, msgs, "agent", 0, 0, 1, 1.8, 512)
        return r, events

    async def test_small_prompt_prior_band_not_applied(self):
        """未命中量 1024 <8K：速率 51200 > 先验带 3000 也过闸（实测 0K 档
        inst=1024 误杀场景）——点干净、无「物理上不可能」播报。"""
        usage = {"prompt_tokens": 1024, "completion_tokens": 3}
        r, events = await self._one_ok(usage, 0.02)   # rate ≈ 51200 tok/s
        self.assertIsNone(r["err"])
        self.assertFalse(any("物理上不可能" in (e.get("msg") or "")
                             for e in events))

    async def test_in_sample_archived(self):
        """输入侧头部采样入档（ADR-0053）：req 级 in_sample = 末条消息文本
        头部 ≤200 字符（agent 矩阵末条即指令、echo 末条为预填）——异常日志
        复盘输入侧；超窗删减重试后反映实际发送内容。agent 为双向引导场景，
        短消息的 200 字符采样窗内可见指令末尾复述约束（ADR-0076）"""
        usage = {"prompt_tokens": 1024, "completion_tokens": 3}
        r, _ = await self._one_ok(usage, 0.02)
        self.assertTrue(
            r.get("in_sample", "").startswith("u" * 40 + "\n\nLength requirement"),
            f"in_sample = 末条消息头部（user 'u'*40 + 指令末尾复述约束）: "
            f"{r.get('in_sample')!r}")

    async def test_large_prompt_prior_band_applies(self):
        """未命中量 16384 ≥8K：速率 8192 > 先验带 3000 → 判「物理上不可能」
        按错误收口（绝对上限 10 万之下，证明走的是先验带分支）。ADR-0052：
        确定性失败除 error tick 外随发 status toast:true 悬浮岛提醒。"""
        usage = {"prompt_tokens": 16384, "completion_tokens": 3}
        r, events = await self._one_ok(usage, 2.0)   # rate ≈ 8192 tok/s
        self.assertIn("物理上不可能", r["err"] or "")
        self.assertIn("先验曲线", r["err"])
        toasts = [e for e in events
                  if e.get("type") == "status" and e.get("toast")]
        self.assertTrue(any("物理上不可能" in (e.get("msg") or "")
                            for e in toasts),
                        "闸门收口应随发 status toast:true 失败提醒（ADR-0052）")

    async def test_cache_hit_x_is_uncached_amount(self):
        """缓存回传 hit>0：x = prompt_tokens − cache_hit（与 gate_rate 未命中
        口径同源）——未命中量 <8K 时先验带不启用，全量 16384 超带不误杀。"""
        usage = {"prompt_tokens": 16384, "completion_tokens": 3,
                 "prompt_cache_hit_tokens": 12000}   # 未命中 4384 <8192
        r, events = await self._one_ok(usage, 0.2)   # rate ≈ 81920（低于绝对上限）
        self.assertIsNone(r["err"])
        self.assertFalse(any("物理上不可能" in (e.get("msg") or "")
                             for e in events))


class TestTransientRetry(unittest.IsolatedAsyncioTestCase):
    """瞬时失败兜底重试：可重试/不可重试分类（传输异常/429/5xx/空流 vs 其他
    4xx 与数据有效性判定）、重试后成功（req 干净且带 retried 留痕）、重试耗尽
    （返回最后一次 err、既有 error tick 不变）、stop_flag 置位立即中止、quiet
    探测/预热静默重试。走真实 _one/_one_media 请求链路（最小假 client 脚本化
    失败注入），退避时长打桩为零避免拖慢测试。"""

    MODEL = "m"
    USAGE = {"prompt_tokens": 1, "completion_tokens": 3}   # 极小 prompt 不触塌缩守卫/速率闸

    # -- 最小假件 -----------------------------------------------------------
    class _Resp:
        """_one/_one_media 所需的最小 httpx.Response 形状。"""
        def __init__(self, status=200, lines=(), headers=None, body=""):
            self.status_code = status
            self._lines = list(lines)
            self.headers = headers or {}
            self._body = body

        @property
        def text(self):
            return self._body

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def aread(self):
            return self._body.encode("utf-8")

        def json(self):
            return json.loads(self._body)

    class _StreamCtx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _ChatClient:
        """chat 流式假客户端：按脚本逐次返回（异常直接抛 = 传输抖动；_Resp
        正常走流；脚本耗尽后重复最后一项）。"""
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        def stream(self, *a, **k):
            i = min(self.calls, len(self.script) - 1)
            self.calls += 1
            item = self.script[i]
            if isinstance(item, Exception):
                raise item
            return TestTransientRetry._StreamCtx(item)

    class _AsrClient:
        """asr 非流式假客户端（脚本化同 _ChatClient）。"""
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        async def post(self, path, **k):
            i = min(self.calls, len(self.script) - 1)
            self.calls += 1
            item = self.script[i]
            if isinstance(item, Exception):
                raise item
            return item

    # -- 脚手架 -------------------------------------------------------------
    MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    @classmethod
    def _sse(cls, chunks=3, usage=None):
        """最小 SSE 行流：chunks 个正文 token + 可选 usage 帧 + [DONE]。"""
        lines = ["data: " + json.dumps(
            {"choices": [{"delta": {"content": "测"}}]}) for _ in range(chunks)]
        if usage:
            lines.append("data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": usage}))
        lines.append("data: [DONE]")
        return lines

    def _make_run(self):
        run = BenchRun({})
        events = []

        async def fake_emit(ev):
            events.append(ev)

        run.emit = fake_emit
        return run, events

    def _statuses(self, events):
        return [e["msg"] for e in events if e.get("type") == "status"]

    def _error_ticks(self, events):
        return [(e["msg"], e.get("req")) for e in events
                if e.get("type") == "tick" and e.get("phase") == "error"]

    async def _call_one(self, run, client, **over):
        with unittest.mock.patch.object(bench, "TRANSIENT_RETRY_BASE_S", 0.01):
            return await run._one(client, self.MODEL, self.MSGS, "creative",
                                  0, 0, 1, 1.8, 128, **over)

    # -- 纯函数分类 ---------------------------------------------------------
    def test_transient_status_classification(self):
        self.assertTrue(_is_transient_http_status(429))
        for s in (500, 502, 503, 504, 599):
            self.assertTrue(_is_transient_http_status(s), s)
        for s in (400, 401, 403, 404, 422):
            self.assertFalse(_is_transient_http_status(s), s)

    def test_transient_exc_classification(self):
        self.assertTrue(_is_transient_exc(httpx.ConnectError("连接被重置")))
        self.assertTrue(_is_transient_exc(httpx.ReadTimeout("read timeout")))
        self.assertTrue(_is_transient_exc(httpx.RemoteProtocolError("裸断连")))
        # 端点探测耗尽/超窗收口等非传输异常不重试
        self.assertFalse(_is_transient_exc(RuntimeError("无可用端点")))
        self.assertFalse(_is_transient_exc(ValueError("x")))

    def test_retry_after_parsing(self):
        self.assertIsNone(_retry_after_s({}))
        self.assertIsNone(_retry_after_s({"retry-after": "abc"}))
        self.assertIsNone(_retry_after_s({"retry-after": "-3"}))
        self.assertEqual(_retry_after_s({"retry-after": "2"}), 2.0)
        self.assertEqual(_retry_after_s({"retry-after": " 0.5 "}), 0.5)
        self.assertEqual(_retry_after_s({"retry-after": "120"}),
                         bench.RETRY_AFTER_CAP_S, "超大值封顶")

    # -- 请求链路 -----------------------------------------------------------
    async def test_transport_error_retried_then_success(self):
        """传输异常（连接重置）兜底重试一次后成功：req 干净、带 retried=1、
        正常读数字段照常产出、status 播报一次兜底重试、无 error tick。"""
        run, events = self._make_run()
        client = self._ChatClient([
            httpx.ConnectError("连接被重置"),
            self._Resp(lines=self._sse(usage=self.USAGE)),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 2, "传输异常应整体重试一次")
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1, "重试后成功应留 retried 痕")
        self.assertEqual(result["out_tokens"], 3, "usage 口径照常")
        self.assertIsNotNone(result["first_abs"])
        statuses = self._statuses(events)
        self.assertEqual(len([s for s in statuses if "兜底重试 1/2" in s]), 1,
                         f"应恰好一条兜底重试播报: {statuses}")
        self.assertIn(self.MODEL, statuses[0])   # 播报含模型/上下文/请求序号
        self.assertIn("r0", statuses[0])
        self.assertEqual(self._error_ticks(events), [], "成功路径不得有 error tick")

    async def test_http_500_retried_with_retry_after(self):
        """HTTP 503 带 Retry-After 头：遵守该头退避（封顶）后重试成功。"""
        run, events = self._make_run()
        client = self._ChatClient([
            self._Resp(status=503, headers={"retry-after": "0.02"},
                       body="upstream unavailable"),
            self._Resp(lines=self._sse(usage=self.USAGE)),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 2)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1)
        statuses = self._statuses(events)
        self.assertTrue(any("0.02s 后兜底重试 1/2" in s for s in statuses),
                        f"Retry-After 头应被遵守: {statuses}")

    async def test_http_4xx_not_retried(self):
        """其他 4xx（鉴权/参数等确定性错误）：不重试，直接按既有错误口径收口。"""
        run, events = self._make_run()
        client = self._ChatClient([
            self._Resp(status=401, body="invalid api key"),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1, "4xx 不得重试")
        self.assertEqual(result["err"], "HTTP 401: invalid api key")
        self.assertNotIn("retried", result)
        self.assertEqual(self._error_ticks(events),
                         [("HTTP 401: invalid api key", 0)],
                         "既有 error tick 口径不变")

    async def test_retry_exhausted_returns_last_err(self):
        """重试耗尽（共 3 次尝试）：返回最后一次 err、retried 键缺席、
        播报 2 次兜底重试、最终既有 error tick 恰一条。"""
        run, events = self._make_run()
        client = self._ChatClient([httpx.ReadTimeout("read timed out")])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1 + TRANSIENT_RETRY_MAX, "共 3 次尝试")
        self.assertEqual(result["err"], "read timed out")
        self.assertNotIn("retried", result)
        statuses = self._statuses(events)
        self.assertTrue(any("兜底重试 1/2" in s for s in statuses), statuses)
        self.assertTrue(any("兜底重试 2/2" in s for s in statuses), statuses)
        self.assertEqual(self._error_ticks(events),
                         [("read timed out", 0)])

    async def test_empty_stream_retried(self):
        """空流（200 但无任何输出 token）：走专属重试预算
        （EMPTY_STREAM_RETRY_MAX，2026-09-17 拍板——Lvllm「空 EOS」间歇
        故障每次重试约 50% 自愈），两次空流后第三次成功。"""
        run, events = self._make_run()
        client = self._ChatClient([
            self._Resp(lines=["data: [DONE]"]),
            self._Resp(lines=["data: [DONE]"]),
            self._Resp(lines=self._sse(usage=self.USAGE)),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 3)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 2)
        statuses = self._statuses(events)
        self.assertTrue(any(f"兜底重试 2/{EMPTY_STREAM_RETRY_MAX}" in s
                            for s in statuses), statuses)

    async def test_empty_stream_exhausted(self):
        """空流耗尽专属预算（1+EMPTY_STREAM_RETRY_MAX 次尝试）：返回既有的
        「未收到任何输出 token」错误口径；专属预算独立于通用瞬时重试
        （TRANSIENT_RETRY_MAX）。"""
        run, events = self._make_run()
        client = self._ChatClient([self._Resp(lines=["data: [DONE]"])])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1 + EMPTY_STREAM_RETRY_MAX)
        self.assertIn("未收到任何输出 token", result["err"])
        self.assertNotIn("retried", result)

    async def test_stop_flag_aborts_retry(self):
        """stop_flag 已置位：不再重试，立即按既有错误口径收口（无兜底播报）。"""
        run, events = self._make_run()

        class _StopClient:
            def __init__(self, run):
                self.run = run
                self.calls = 0

            def stream(self, *a, **k):
                self.calls += 1
                self.run.stop_flag = True   # 模拟请求在途期间用户点停止
                raise httpx.ConnectError("连接被重置")

        client = _StopClient(run)
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1, "停止后不得再重试")
        self.assertEqual(result["err"], "连接被重置")
        self.assertNotIn("retried", result)
        self.assertEqual(self._statuses(events), [], "停止路径不播报兜底重试")

    async def test_quiet_probe_retries_silently(self):
        """quiet=True（探测/预热请求）：同样享受兜底重试但全程静默——不发
        status 播报也不发 tick（agent 矩阵预热依赖重试，且不刷日志）。
        usage 量取不触发系数校准播报的形态（obs/cobs 均越界不采纳），隔离
        重试静默性主题与既有校准播报行为。"""
        run, events = self._make_run()
        client = self._ChatClient([
            httpx.ConnectError("连接被重置"),
            self._Resp(lines=self._sse(
                usage={"prompt_tokens": 1, "completion_tokens": 100})),
        ])
        result = await self._call_one(run, client, quiet=True)
        self.assertEqual(client.calls, 2)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1)
        self.assertEqual([e for e in events if e.get("type") in ("status", "tick")],
                         [], f"quiet 重试必须静默: {events}")

    async def test_clean_success_no_retried_key(self):
        """一次成功（无重试）：结果不带 retried 键，成功路径逐字节不变。"""
        run, events = self._make_run()
        client = self._ChatClient([self._Resp(lines=self._sse(usage=self.USAGE))])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(result["err"])
        self.assertNotIn("retried", result)
        self.assertEqual(result["out_tokens"], 3)

    async def test_media_asr_retried_then_success(self):
        """_one_media（asr）传输异常兜底重试：重试后成功带 retried 留痕。"""
        run, events = self._make_run()
        client = self._AsrClient([
            httpx.ConnectError("连接被重置"),
            self._Resp(body='{"text": "模拟识别结果。"}'),
        ])
        with unittest.mock.patch.object(bench, "TRANSIENT_RETRY_BASE_S", 0.01):
            result = await run._one_media(client, self.MODEL, "asr", "asr",
                                          5, 0, 1, 0)
        self.assertEqual(client.calls, 2)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1)
        self.assertGreater(result["out_chars"], 0)
        statuses = self._statuses(events)
        self.assertTrue(any("兜底重试 1/2" in s for s in statuses), statuses)

    async def test_media_asr_500_not_retried_beyond_budget(self):
        """_one_media HTTP 500：重试耗尽后按既有 _fail 口径收口（error tick
        + err 返回），不无限重试。"""
        run, events = self._make_run()
        client = self._AsrClient([self._Resp(status=500, body="boom")])
        with unittest.mock.patch.object(bench, "TRANSIENT_RETRY_BASE_S", 0.01):
            result = await run._one_media(client, self.MODEL, "asr", "asr",
                                          5, 0, 1, 0)
        self.assertEqual(client.calls, 1 + TRANSIENT_RETRY_MAX)
        self.assertEqual(result["err"], "HTTP 500: boom")
        self.assertNotIn("retried", result)
        self.assertEqual(self._error_ticks(events),
                         [("HTTP 500: boom", 0)])


class TestPrefillContinueParams(unittest.IsolatedAsyncioTestCase):
    """echo 真·预填续写参数（ADR-0079）：末条为 assistant 预填时 payload 带
    add_generation_prompt=false + continue_final_message=true（模板不闭合
    轮次、从预填末尾直接续写——闭合形态是 32K+ 确定性早停的根因）；末条
    非 assistant 不下发；网关 400 点名参数时成对省略、按模型锁定、同端点
    重试（回退闭合轮次形态），锁定后不再下发。"""

    MODEL = "m"
    ECHO_MSGS = [{"role": "system", "content": "s"},
                 {"role": "user", "content": "u" * 40},
                 {"role": "assistant", "content": "预填正文"}]

    class _Resp:
        def __init__(self, status=200, lines=(), body=""):
            self.status_code = status
            self._lines = list(lines)
            self.headers = {}
            self._body = body

        @property
        def text(self):
            return self._body

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def aread(self):
            return self._body.encode("utf-8")

    class _Ctx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _CapClient:
        """脚本化假客户端：逐次返回脚本项并记录每次请求的 payload。"""
        def __init__(self, script):
            self.script = list(script)
            self.payloads = []

        def stream(self, *a, **k):
            i = min(len(self.payloads), len(self.script) - 1)
            self.payloads.append(k["json"])
            return TestPrefillContinueParams._Ctx(self.script[i])

    @staticmethod
    def _ok_resp():
        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "测"}}]}),
                 "data: " + json.dumps(
                     {"choices": [{"delta": {}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 3}}),
                 "data: [DONE]"]
        return TestPrefillContinueParams._Resp(200, lines)

    def _make_run(self):
        run = BenchRun({})
        events = []

        async def fake_emit(ev):
            events.append(ev)

        run.emit = fake_emit
        return run, events

    async def _call(self, run, client, msgs):
        return await run._one(client, self.MODEL, msgs, "creative",
                              0, 0, 1, 1.8, 128)

    async def test_echo_prefill_sends_continue_params(self):
        """末条 assistant 预填 → payload 带续写参数对。"""
        run, _ = self._make_run()
        client = self._CapClient([self._ok_resp()])
        r = await self._call(run, client, self.ECHO_MSGS)
        self.assertIsNone(r["err"])
        p = client.payloads[0]
        self.assertIs(p.get("add_generation_prompt"), False)
        self.assertIs(p.get("continue_final_message"), True)

    async def test_non_assistant_last_omits_params(self):
        """末条为 user（常规/agent/媒体形态）→ 不下发续写参数。"""
        run, _ = self._make_run()
        client = self._CapClient([self._ok_resp()])
        r = await self._call(run, client, [{"role": "system", "content": "s"},
                                           {"role": "user", "content": "u"}])
        self.assertIsNone(r["err"])
        self.assertNotIn("add_generation_prompt", client.payloads[0])
        self.assertNotIn("continue_final_message", client.payloads[0])

    async def test_400_locks_and_retries_without_params(self):
        """网关 400 点名 add_generation_prompt → 成对省略同端点重试成功、
        模型入 prefill_locked、播报；锁定后的后续请求不再下发。"""
        run, events = self._make_run()
        bad = self._Resp(400, body='{"error":{"message":"unknown parameter: '
                                   'add_generation_prompt"}}')
        client = self._CapClient([bad, self._ok_resp(), self._ok_resp()])
        r = await self._call(run, client, self.ECHO_MSGS)
        self.assertIsNone(r["err"])
        self.assertEqual(len(client.payloads), 2, "400 后应同端点重试")
        self.assertNotIn("add_generation_prompt", client.payloads[1])
        self.assertNotIn("continue_final_message", client.payloads[1])
        self.assertIn(self.MODEL, run.prefill_locked)
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("add_generation_prompt" in s for s in statuses),
                        f"应有省略播报: {statuses}")
        # 锁定后后续请求不再下发
        r2 = await self._call(run, client, self.ECHO_MSGS)
        self.assertIsNone(r2["err"])
        self.assertNotIn("add_generation_prompt", client.payloads[2])


class TestStallFlushOne(unittest.IsolatedAsyncioTestCase):
    """停滞回吐剔除（_one 收口级）：合成流复现真实事故形状（20260913-234719
    fastllm 27B：平稳段 + 13.5s 单次停滞 + 430 tokens 亚毫秒回吐），走真实
    _one 请求链路（假 client 定时流），断言剔除后净口径与原始留痕并存。"""

    MODEL = "m"

    class _Resp:
        def __init__(self, lines):
            self.status_code = 200
            self.headers = {}
            self._lines = lines

        async def aiter_lines(self):
            async for ln in self._lines:
                yield ln

        async def aread(self):
            return b""

    class _Ctx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, resp):
            self.resp = resp

        def stream(self, *a, **k):
            return TestStallFlushOne._Ctx(self.resp)

    class _SeqClient:
        """按脚本逐次返回的流式假客户端（_run_point 并发批用）。"""
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        def stream(self, *a, **k):
            i = min(self.calls, len(self.script) - 1)
            self.calls += 1
            return TestStallFlushOne._Ctx(self.script[i])

    @staticmethod
    def _plan_lines(plan, usage, clock=None):
        """定时 SSE：plan = [(间隔秒, 事件数)] 或 [(间隔秒, 事件数, 每事件
        字符数)]——先等间隔，再连发 n 个内容事件（组内到达间隔≈0，构成
        亚毫秒回吐串；间隔出现在组首事件前，即组与组之间；ch 缺省 1）；末尾
        补 usage 帧 + [DONE]。

        clock 给定时用**假时钟推进**代替真实等待：判据门限（
        BURST_AVG_GAP_MS=1ms、STALL_THRESHOLD_S=0.5s）是墙钟量，而真实
        asyncio 调度在负载下会把"亚毫秒串"拉过 1ms，于是同一个用例时过时不过
        （实测 flush_tok 411/414/422 漂移，极端情况 episode 判不出直接
        KeyError）。假时钟把时间轴完全交给测试，判定与机器负载解耦。"""
        async def lines():
            for item in plan:
                gap, n = item[0], item[1]
                ch = item[2] if len(item) > 2 else 1
                if clock is None:
                    if gap:
                        await asyncio.sleep(gap)
                else:
                    clock.now += gap
                for i in range(n):
                    # 组内亚毫秒（<BURST_AVG_GAP_MS）；gap=0 的组首个事件也要
                    # 有正增量——同刻到达会被 _one 记为 0 间隔，纯属浪费判据
                    if clock is not None and (i or not gap):
                        clock.now += 0.0002
                    yield "data: " + json.dumps(
                        {"choices": [{"delta": {"content": "测" * ch}}]})
            yield "data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": usage})
            yield "data: [DONE]"
        return lines()

    class _Clock:
        """可推进的假时钟（只有 bench.py 会看到它，见 _ClockShim）。"""
        def __init__(self, now=1000.0):
            self.now = now

    class _ClockShim:
        """把 bench.py 视角下的 time.perf_counter 换成假时钟，其余属性透传。

        直接 patch 全局 time.perf_counter 会牵连 pytest 计时等无关模块；
        bench.py 是 `import time` + `time.perf_counter()` 的用法，替换模块属性
        即可精确命中被测代码。"""
        def __init__(self, clock):
            self._clock = clock

        def __getattr__(self, name):
            return getattr(time, name)

        def perf_counter(self):
            return self._clock.now

    @staticmethod
    def _make_run():
        run = BenchRun({})

        async def fake_emit(ev):
            pass

        run.emit = fake_emit
        return run

    MSGS = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u" * 40}]

    async def _run_one(self, plan, usage, max_tokens=1024, conc=1):
        """用假时钟跑一次 _one（时间轴确定，与机器负载无关）。"""
        clock = self._Clock()
        run = self._make_run()
        resp = self._Resp(self._plan_lines(plan, usage, clock=clock))
        with unittest.mock.patch.object(bench, "time", self._ClockShim(clock)):
            return await run._one(self._Client(resp), self.MODEL, self.MSGS,
                                  "creative", 0, 0, conc, 1.8, max_tokens)

    async def test_real_case_shape_flush_removed(self):
        """复现真实案例形状：47 tokens × 30ms + 13.5s 停滞 + 430 亚毫秒回吐
        （每事件 1 token）→ 剔除后 adj ≈ 平稳段速率（绝非 200+）、tpc 为
        None、flush_tok ≈ 430；原始 stall_s 留痕与毛口径 decode_tok_s 不变。"""
        usage = {"prompt_tokens": 100, "completion_tokens": 477}
        plan = [(0.03, 1)] * 47 + [(13.5, 430)]
        r = await self._run_one(plan, usage)
        self.assertIsNone(r["err"])
        self.assertEqual(r["flush_count"], 1)
        self.assertAlmostEqual(r["flush_tok"], 430, delta=1)
        self.assertAlmostEqual(r["stall_s"], 13.5, delta=0.05,
                               msg="原始停滞留痕不变")
        self.assertEqual(r["stall_count"], 1)
        adj = r["decode_tok_s_adj"]
        self.assertIsNotNone(adj)
        self.assertGreaterEqual(adj, 25, "健康段被误剔到无读数")
        self.assertLessEqual(adj, 45, f"adj={adj} 仍被回吐注水（应 ≈ 平稳段 ~34）")
        self.assertIsNone(r["tok_per_chunk"], "回吐串不得被当成投机批次交付")
        self.assertLess(r["decode_tok_s"], 60, "毛口径仍含停滞（原始窗口语义不变）")

    async def test_flush_tok_clamped_to_out_minus_one(self):
        """剔除总量钳制 ≤ out_tokens−1（至少留 1 token 给净口径）；out_clean
        不足 32 时 adj 置空。usage=10 tokens、40 事件（回吐折算 9.25 → 钳 9）。"""
        usage = {"prompt_tokens": 1, "completion_tokens": 10}
        plan = [(0, 1), (0.6, 37), (0.03, 2)]
        r = await self._run_one(plan, usage, max_tokens=128)
        self.assertIsNone(r["err"])
        self.assertEqual(r["flush_count"], 1)
        self.assertEqual(r["flush_tok"], 9, "回吐折算 9.25 应钳制到 out−1")
        self.assertIsNone(r["decode_tok_s_adj"], "out_clean=1 <32 净口径置空")
        self.assertIsNone(r["tok_per_chunk"])

    async def test_no_flush_no_keys(self):
        """无回吐：flush 两键不出现（与「有值才出现」的留痕风格一致）。"""
        usage = {"prompt_tokens": 1, "completion_tokens": 3}
        plan = [(0.03, 1), (0.03, 1), (0.03, 1)]
        r = await self._run_one(plan, usage, max_tokens=128)
        self.assertIsNone(r["err"])
        self.assertNotIn("flush_tok", r)
        self.assertNotIn("flush_count", r)
        self.assertIsNone(r["tok_per_chunk"])   # 逐 token 记账噪声不入档

    async def test_merged_cadence_case_shape(self):
        """复现 20260914-002125 形态（合成流）：健康 ~80 事件（29ms、1
        token/事件）+ 6.0s 停滞 + ~59 事件（8ms、3 token/事件合并交付）。
        速率判据检出 episode 后：adj ≈ 健康段速率、tpc=None、flush_tok =
        实测累计（per-event 记账，无折算偏差），原始 stall_s 留痕不变。"""
        usage = {"prompt_tokens": 100, "completion_tokens": 257}   # 80×1 + 59×3
        plan = ([(0.029, 1, 1)] * 80 + [(6.0, 1, 1)]
                + [(0.008, 1, 3)] * 59)
        r = await self._run_one(plan, usage)
        self.assertIsNone(r["err"])
        self.assertEqual(r["flush_count"], 1)
        self.assertAlmostEqual(r["flush_tok"], 177, delta=2,   # 59×3 实测累计
                               msg="per-event 记账应还原真实回吐量")
        self.assertAlmostEqual(r["stall_s"], 6.0, delta=0.05,
                               msg="原始停滞留痕不变")
        self.assertEqual(r["stall_count"], 1)
        adj = r["decode_tok_s_adj"]
        self.assertIsNotNone(adj)
        self.assertGreaterEqual(adj, 25, "健康段被误剔到无读数")
        self.assertLessEqual(adj, 45, f"adj={adj} 仍被回吐注水（应 ≈ 健康段 ~34）")
        self.assertIsNone(r["tok_per_chunk"], "8ms 合并回吐串不得当成投机批次")
        self.assertLess(r["decode_tok_s"], 60, "毛口径仍含停滞（原始窗口语义不变）")

    async def test_big_merge_case_shape(self):
        """复现 20260914-093936 形态（合成流，事件间隔判据原理性漏检）：
        ~24 个 1-token 事件 @~29ms + 6.61s 停滞 + 9 个 26-token fat 事件
        @~22ms（交付 ~1180 tok/s vs 健康 ~40 tok/s）→ 速率判据检出：flush_tok
        ≈ 234（实测累计）、adj ≈ 健康段速率（绝非 133.9）、tpc=None、
        stall_s = 6.61 留痕。健康节奏取 ~24ms 使 r_pre×S 上限（~275）不干涉
        剔除量（29ms 形态下 cap≈225 会截到 225，由纯函数用例覆盖）。"""
        usage = {"prompt_tokens": 100, "completion_tokens": 258}   # 24×1 + 9×26
        plan = ([(0.024, 1, 1)] * 24 + [(6.61, 1, 1)]
                + [(0.022, 1, 26)] * 9)
        r = await self._run_one(plan, usage)
        self.assertIsNone(r["err"])
        self.assertEqual(r["flush_count"], 1)
        self.assertAlmostEqual(r["flush_tok"], 234, delta=2,
                               msg="per-event 记账应还原真实回吐量")
        self.assertAlmostEqual(r["stall_s"], 6.61, delta=0.05,
                               msg="原始停滞留痕不变")
        self.assertEqual(r["stall_count"], 1)
        adj = r["decode_tok_s_adj"]
        self.assertIsNotNone(adj)
        self.assertGreaterEqual(adj, 28, "健康段被误剔到无读数")
        self.assertLessEqual(adj, 50, f"adj={adj} 仍被回吐注水（应 ≈ 健康段 ~40）")
        self.assertIsNone(r["tok_per_chunk"], "22ms 26-token 合并回吐不得当成投机批次")
        self.assertLess(r["decode_tok_s"], 60, "毛口径仍含停滞（原始窗口语义不变）")

    async def test_tail_case_shape(self):
        """复现 20260914-115607 inst=1024 拖尾形态（合成流）：健康 ~23 tok
        @~25ms + 6.8s 停滞 + 回吐（前 178 tok 以 ~5× 猛冲、后 55 tok 以 ~2×
        拖尾）→ 延续放宽到 >1.2× 后整段剔除：flush_tok = 233（实测累计）、
        adj ≈ 健康段速率（round-3 在拖尾断链漏剔尾巴 ~55 tokens → 69.8）、
        flush_s ≈ 回吐交付时长、stall_s 留痕。健康节奏取 ~25ms 使 r_pre×S
        上限（~272）不干涉剔除量。"""
        usage = {"prompt_tokens": 100, "completion_tokens": 256}   # 23+178+55
        plan = ([(0.024, 1, 1)] * 23 + [(6.8, 1, 1)]
                + [(0.15, 1, 25)] * 6 + [(0.15, 1, 28)]
                + [(0.15, 1, 11)] * 5)
        r = await self._run_one(plan, usage)
        self.assertIsNone(r["err"])
        self.assertEqual(r["flush_count"], 1)
        self.assertAlmostEqual(r["flush_tok"], 233, delta=2,
                               msg="拖尾应整段剔除（233 = 178+55 实测累计）")
        self.assertAlmostEqual(r["flush_s"], 12 * 0.15, delta=0.01,
                               msg="flush_s = 剔除段交付总时长（停滞后首个交付"
                                   "事件 → 最后一个回吐事件，12 个间隔）")
        self.assertAlmostEqual(r["stall_s"], 6.8, delta=0.05,
                               msg="原始停滞留痕不变")
        self.assertEqual(r["stall_count"], 1)
        adj = r["decode_tok_s_adj"]
        self.assertIsNotNone(adj)
        self.assertGreaterEqual(adj, 28, "健康段被误剔到无读数")
        self.assertLessEqual(adj, 50, f"adj={adj} 拖尾漏剔/仍被注水（应 ≈ 健康段 ~40）")
        self.assertLess(r["decode_tok_s"], 60, "毛口径仍含停滞（原始窗口语义不变）")

    # 回吐流：10×30ms 健康 + 0.6s 停滞 + 15×零间隔亚毫秒回吐；无回吐对照流。
    # 计划由调用方交给 _run_point_plans（假时钟推进，判定与负载解耦）
    FLUSH_PLAN = ([(0.03, 1)] * 10 + [(0.6, 1)] + [(0, 1)] * 15,
                  {"prompt_tokens": 1, "completion_tokens": 26})
    PLAIN_PLAN = ([(0.01, 3)], {"prompt_tokens": 1, "completion_tokens": 3})

    async def _run_point_plans(self, plans, conc):
        """plans = [(plan, usage), ...] 按序作为各请求的流；假时钟推进时间轴。"""
        clock = self._Clock()
        run = self._make_run()
        script = [self._Resp(self._plan_lines(p, u, clock=clock))
                  for p, u in plans]
        with unittest.mock.patch.object(bench, "time", self._ClockShim(clock)):
            return await run._run_point(self._SeqClient(script), self.MODEL,
                                        "creative", 0, conc, 1.8)

    async def test_point_flush_conc1_only(self):
        """点级透传（_run_point 批级，ADR-0074 新口径）：停滞/回吐诊断仅
        conc=1 产出——conc=1 点透传该请求的 flush 三键（无回吐点保持
        None）；conc>1 批内请求共享服务端资源，逐请求停滞/回吐只是资源挤占
        的观测噪声、无判读意义，点级恒 None（_flush_episodes 仅 conc=1 执行，
        旧「worst-req 取剔除量最大的请求」口径随 conc>1 诊断退役）。回吐流
        构造为物理自洽形态（回吐 tokens ≤ r_pre×stall_s：0.6s 停滞 × 健康
        ~33 tok/s ≈ 20 上限，16 个回吐 tokens 不被封顶）。"""
        pt = await self._run_point_plans([self.FLUSH_PLAN], 1)
        self.assertTrue(pt["all_ok"])
        self.assertEqual(pt["flush_tok"], 16)   # e10(停滞段首事件) + 15 回吐事件
        self.assertEqual(pt["flush_count"], 1)
        # flush_s = 回吐段交付总时长：亚毫秒串 ≈0（与毛口径「亚毫秒串时间
        # ≈0」一致）；假时钟下 15 事件 ×0.2ms = 3ms，上界仍留余量
        self.assertLessEqual(pt["flush_s"], 0.1,
                             msg="flush_s 与 flush_tok 同源同请求（亚毫秒段）")
        self.assertAlmostEqual(pt["stall_s"], 0.6, delta=0.05,
                               msg="原始停滞留痕透传")
        # 对照①：conc=1 全无回吐，点级 flush 三键为 None
        pt2 = await self._run_point_plans([self.PLAIN_PLAN], 1)
        self.assertTrue(pt2["all_ok"])
        self.assertIsNone(pt2["flush_tok"])
        self.assertIsNone(pt2["flush_count"])
        self.assertIsNone(pt2["flush_s"])
        # 对照②：conc>1 批（含回吐请求）点级 flush 三键保持 None（新口径）
        pt3 = await self._run_point_plans([self.PLAIN_PLAN, self.FLUSH_PLAN], 2)
        self.assertTrue(pt3["all_ok"])
        self.assertIsNone(pt3["flush_tok"])
        self.assertIsNone(pt3["flush_count"])
        self.assertIsNone(pt3["flush_s"])


class TestFlushEpisodesPure(unittest.TestCase):
    """_flush_episodes 纯函数口径（间隔与每事件 token 全部显式给定）：把
    "哪些形态算回吐"钉死在算法层，不依赖任何时钟。集成侧 TestStallFlushOne
    负责"接线"（_one 是否按该口径剔除并留痕），两侧分工避免用墙钟测算法。"""

    def test_real_case_shape(self):
        """真实事故①形态：47×30ms 平稳（r_pre≈33 tok/s）+ 13.5s 停滞 +
        430 个亚毫秒 1-token 回吐；r_pre×stall≈450 上限不干涉（430<450）。"""
        gaps = [0.03] * 46 + [13.5] + [0.0002] * 429
        ev_toks = [1.0] * 477
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        e = eps[0]
        self.assertEqual(e["stall_idx"], 46)
        self.assertEqual(e["start_idx"], 47)     # 停滞段首事件（其后即回吐串）
        self.assertEqual(e["end_idx"], 477)      # 全部事件入段
        self.assertEqual(e["burst_events"], 430)
        self.assertAlmostEqual(e["burst_tok"], 430, delta=0.01)
        self.assertAlmostEqual(e["burst_s"], 429 * 0.0002, delta=1e-6,
                               msg="交付时长 = 串内间隙之和")

    def test_over_removal_capped_by_healthy_rate(self):
        """剔除封顶 r_pre×stall_s：停滞期间服务端最多生成这么多 token，
        剔超了就会误杀正常生成。10×30ms 平稳（r_pre≈33）+ 1.0s 停滞 +
        100 个回吐事件 → burst_tok 被压到 ≈33.3，而非 100。"""
        gaps = [0.03] * 9 + [1.0] + [0.0002] * 99
        ev_toks = [1.0] * 110
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        self.assertAlmostEqual(eps[0]["burst_tok"], 100 / 3.0, delta=0.2)
        self.assertLess(eps[0]["burst_tok"], 100)

    def test_mtp_batch_cadence_not_flagged(self):
        """MTP 批节奏 = 健康速率：3 token/事件 @5ms（r_pre≈600 tok/s）在
        停滞后照常交付——速率未达 3×，不得误判为回吐。"""
        gaps = [0.005] * 9 + [1.0] + [0.005] * 9
        ev_toks = [3.0] * 20
        self.assertEqual(_flush_episodes(gaps, ev_toks), [])

    def test_zero_token_events_do_not_break_segment(self):
        """零内容/keepalive 事件（est_tok=0）跳过不中断回吐段。"""
        # 10 平稳 + 停滞 + 10 回吐 = 20 事件 / 19 间隙；第 3 个回吐事件为
        # keepalive（est_tok=0）——它不得把回吐段拦腰截断（Σ 仍 9 ≥ 门槛）
        gaps = [0.03] * 9 + [1.0] + [0.0002] * 9
        ev_toks = [1.0] * 10 + [1.0, 1.0, 0.0] + [1.0] * 7
        eps = _flush_episodes(gaps, ev_toks)
        self.assertEqual(len(eps), 1)
        self.assertGreaterEqual(eps[0]["burst_tok"], 8)

    def test_small_or_malformed_inputs(self):
        """认定门槛 Σburst < FLUSH_MIN_TOK(8) 不成立；长度错配/全零 token
        直接返回空（消费方不做形状假设）。"""
        self.assertEqual(_flush_episodes([], [1.0]), [])
        self.assertEqual(_flush_episodes([0.03], [1.0, 1.0]), [])          # 长度错配
        self.assertEqual(_flush_episodes([0.03, 1.0], [0.0, 0.0, 0.0]), [])
        # 停滞 + 仅 3 个回吐 token：低于门槛，不认定
        self.assertEqual(_flush_episodes([0.03, 1.0, 0.0002], [1.0, 1.0, 3.0]), [])

    def test_short_stall_below_threshold_ignored(self):
        """停滞门限 0.5s：0.4s 空窗只是正常抖动，不触发回吐剔除。"""
        gaps = [0.03] * 9 + [0.4] + [0.0002] * 20
        ev_toks = [1.0] * 31
        self.assertEqual(_flush_episodes(gaps, ev_toks), [])


class TestDecodePeakRate(unittest.TestCase):
    """_decode_peak_rate（并发 decode 峰值，DECODE_PEAK_WINDOW_S 右对齐滑窗）：
    全批 token 交付事件合并时间轴后 max(窗内 tok 合计 ÷ window)；突发/回吐
    峰值如实计入（其不可信已由 decode_burst/flush 字段分别标注，本口径不
    重复甄别）；无有效事件返回 None。"""

    def test_empty_none(self):
        self.assertIsNone(_decode_peak_rate([]))
        self.assertIsNone(_decode_peak_rate([(0.0, [], [])]))
        self.assertIsNone(_decode_peak_rate([(0.0, [], []), (1.0, [], [])]))

    def test_single_uniform(self):
        """单请求匀速：10 事件、间隔 0.2s、每事件 2 tok。1s 窗（右对齐、
        端点含入）最多罩住 6 个事件 = 12 tok → 峰值 12.0（手算核对）。"""
        tl = (0.0, [0.2] * 9, [2.0] * 10)
        self.assertEqual(_decode_peak_rate([tl]), 12.0)

    def test_two_requests_overlap_doubles(self):
        """两请求时间轴完全重叠：任一窗内 tokens 翻倍 → 峰值翻倍。"""
        tl = (0.0, [0.2] * 9, [2.0] * 10)
        self.assertEqual(_decode_peak_rate([tl, tl]), 24.0)

    def test_window_boundary_half_rate(self):
        """1s 窗边界：tokens 均匀分布在 2s 上（11 事件 @0.2s、各 1 tok，
        均速 5.5 tok/s），窗内最多罩住 [0,1] 闭区间上的 6 个事件 → 峰值 6.0
        ——≈ 一半速率，边界含端点略高于均速。"""
        tl = (0.0, [0.2] * 10, [1.0] * 11)
        self.assertEqual(_decode_peak_rate([tl]), 6.0)

    def test_subms_burst_counted_honestly(self):
        """亚毫秒突发：50 tok 挤在 0.01s → 窗宽恒为 window（不按实际跨度
        缩放），峰值 = 50/1.0 = 50.0 如实计入、不爆炸；单 fat 事件同值。"""
        gaps = [0.01 / 49] * 49
        self.assertEqual(_decode_peak_rate([(0.0, gaps, [1.0] * 50)]), 50.0)
        self.assertEqual(_decode_peak_rate([(0.0, [], [50.0])]), 50.0)

    def test_custom_window_parameter(self):
        """window 参数生效（默认 DECODE_PEAK_WINDOW_S=1.0）：两事件相距
        10s，1s 窗各罩一个 → 100.0；20s 窗全罩 → 200÷20 = 10.0。"""
        self.assertEqual(DECODE_PEAK_WINDOW_S, 1.0)
        tl = (0.0, [10.0], [100.0, 100.0])
        self.assertEqual(_decode_peak_rate([tl]), 100.0)
        self.assertEqual(_decode_peak_rate([tl], window=20.0), 10.0)


class TestConcBatchStats(unittest.TestCase):
    """_conc_batch_stats（批级并发口径 6 指标）：批起点 = min(start_abs)；
    prefill/total span 从批起点量起、decode_span = total − prefill；
    守卫 len(ok)==conc 与计时字段齐备（与 _total_decode_rates 同口径），
    span 分母 ≤0 只置空对应速率键；峰值只看 _tl 有无，不依赖 span。"""

    @staticmethod
    def _none_stats():
        return {"prefill_span_s": None, "total_span_s": None,
                "decode_span_s": None, "prefill_conc_tok_s": None,
                "decode_conc_tok_s": None, "decode_peak_tok_s": None}

    @staticmethod
    def _req(req, start, first, last, ptok, otok, tl=None):
        r = {"req": req, "start_abs": start, "first_abs": first,
             "last_abs": last, "prompt_tokens": ptok, "out_tokens": otok}
        if tl is not None:
            r["_tl"] = tl
        return r

    def test_incomplete_guards_all_none(self):
        """len(ok)!=conc / start_abs 缺失 / first_abs 或 last_abs 为 None：
        6 键全 None。"""
        one = self._req(0, 0.0, 1.0, 3.0, 100, 50)
        self.assertEqual(_conc_batch_stats([one], 2), self._none_stats())
        no_start = self._req(0, 0.0, 1.0, 3.0, 100, 50)
        del no_start["start_abs"]
        pair = [self._req(1, 0.5, 2.0, 4.0, 100, 50)]
        self.assertEqual(_conc_batch_stats([no_start] + pair, 2),
                         self._none_stats())
        self.assertEqual(
            _conc_batch_stats([self._req(0, 0.0, None, 3.0, 100, 50)] + pair, 2),
            self._none_stats())
        self.assertEqual(
            _conc_batch_stats([self._req(0, 0.0, 1.0, None, 100, 50)] + pair, 2),
            self._none_stats())

    def test_normal_conc2_batch(self):
        """正常 2 并发批（手算逐一核对）：批起点 = min(0.0, 0.5) = 0.0；
        prefill_span = max(1.0, 2.0) − 0 = 2.0；total_span = max(3.0, 4.0)
        = 4.0；decode_span = 2.0；prefill_conc = (1000+500)÷2 = 750.0；
        decode_conc = (200+300)÷2 = 250.0；峰值：合并时间轴
        (1,100)(2,100)(2,150)(3,150)，窗 [2,3] 含 400 tok → 400.0。"""
        r0 = self._req(0, 0.0, 1.0, 3.0, 1000, 200, (1.0, [1.0], [100.0, 100.0]))
        r1 = self._req(1, 0.5, 2.0, 4.0, 500, 300, (2.0, [1.0], [150.0, 150.0]))
        stats = _conc_batch_stats([r0, r1], 2)
        self.assertEqual(stats["prefill_span_s"], 2.0)
        self.assertEqual(stats["total_span_s"], 4.0)
        self.assertEqual(stats["decode_span_s"], 2.0)
        self.assertEqual(stats["prefill_conc_tok_s"], 750.0)
        self.assertEqual(stats["decode_conc_tok_s"], 250.0)
        self.assertEqual(stats["decode_peak_tok_s"], 400.0)

    def test_decode_span_nonpositive(self):
        """decode_span ≤ 0（所有请求 last ≤ max(first)）：decode_conc_tok_s
        置 None 而 prefill/span 正常（decode_span_s 如实记 0.0）；峰值不依赖
        span，照常产出（200.0）。"""
        r0 = self._req(0, 0.0, 2.0, 2.0, 1000, 100, (2.0, [], [100.0]))
        r1 = self._req(1, 0.5, 3.0, 3.0, 500, 100, (3.0, [], [100.0]))
        stats = _conc_batch_stats([r0, r1], 2)
        self.assertEqual(stats["prefill_span_s"], 3.0)        # max(first)−0
        self.assertEqual(stats["prefill_conc_tok_s"], 500.0)  # 1500÷3
        self.assertEqual(stats["total_span_s"], 3.0)          # max(last)−0
        self.assertEqual(stats["decode_span_s"], 0.0)
        self.assertIsNone(stats["decode_conc_tok_s"])
        self.assertEqual(stats["decode_peak_tok_s"], 200.0)

    def test_no_tl_peak_none_rest_normal(self):
        """无 _tl 键：峰值 None，其余 5 键照常产出（峰值只看 _tl 有无）。"""
        r0 = self._req(0, 0.0, 1.0, 3.0, 1000, 200)
        r1 = self._req(1, 0.5, 2.0, 4.0, 500, 300)
        stats = _conc_batch_stats([r0, r1], 2)
        self.assertIsNone(stats["decode_peak_tok_s"])
        self.assertEqual(stats["prefill_span_s"], 2.0)
        self.assertEqual(stats["total_span_s"], 4.0)
        self.assertEqual(stats["decode_span_s"], 2.0)
        self.assertEqual(stats["prefill_conc_tok_s"], 750.0)
        self.assertEqual(stats["decode_conc_tok_s"], 250.0)

    def test_prefill_span_nonpositive_only_prefill_cleared(self):
        """分母级守卫①：prefill_span ≤ 0（max(first) == min(start)）只置空
        prefill 两键，total/decode 与峰值照常产出（span 起点 = 2.0）。"""
        r0 = self._req(0, 2.0, 2.0, 6.0, 1000, 100, (2.0, [2.0], [50.0, 50.0]))
        r1 = self._req(1, 2.0, 2.0, 5.0, 500, 100, (2.0, [1.0], [50.0, 50.0]))
        stats = _conc_batch_stats([r0, r1], 2)
        self.assertIsNone(stats["prefill_span_s"])
        self.assertIsNone(stats["prefill_conc_tok_s"])
        self.assertEqual(stats["total_span_s"], 4.0)          # max(last)−2
        self.assertEqual(stats["decode_span_s"], 4.0)         # 4−0
        self.assertEqual(stats["decode_conc_tok_s"], 50.0)    # 200÷4
        self.assertEqual(stats["decode_peak_tok_s"], 150.0)   # 手算见注释
        # 峰值手算：合并 (2,50)(2,50)(3,50)(4,50)，窗 [2,3] 含 150 tok

    def test_total_span_nonpositive_only_three_cleared(self):
        """分母级守卫②：total_span ≤ 0 只置空 total/decode 三键；prefill
        两键同样因 span ≤0 置空；peak 只看 _tl，照常产出（20.0）。"""
        r0 = self._req(0, 3.0, 3.0, 3.0, 1000, 100, (3.0, [], [10.0]))
        r1 = self._req(1, 3.0, 3.0, 3.0, 500, 100, (3.0, [], [10.0]))
        stats = _conc_batch_stats([r0, r1], 2)
        self.assertEqual(
            stats, {**self._none_stats(), "decode_peak_tok_s": 20.0})


class TestBatchPrefillPoint(unittest.TestCase):
    """_batch_prefill_point（批级 prefill 点口径，_run_point 与 agent 矩阵
    共用）：TTFT = 批开始→全部首 token（stats["prefill_span_s"]）、速度 =
    Σtokens ÷ span、净口径经 _net_elapsed 扣 RTT；span 缺失（守卫未过/分母
    ≤0，如部分失败批）返回 {} 由调用方回退逐请求均值；rtt None 时净口径
    两键 None。"""

    def test_span_missing_returns_empty(self):
        """span 缺失 / stats 空 / span=0（分母级守卫）：返回 {}，调用方回退
        逐请求均值口径。"""
        self.assertEqual(_batch_prefill_point(4096, {}, 0.2), {})
        self.assertEqual(_batch_prefill_point(
            4096, {"prefill_span_s": None, "total_span_s": None}, 0.2), {})
        self.assertEqual(_batch_prefill_point(
            4096, {"prefill_span_s": 0.0}, 0.2), {})

    def test_rtt_none_net_keys_none(self):
        """rtt None（基线缺失）：毛口径两键照常产出，净口径两键 None。"""
        self.assertEqual(_batch_prefill_point(8192, {"prefill_span_s": 2.0}, None),
                         {"ttft_s": 2.0, "prefill_tok_s": 4096.0,
                          "ttft_net_s": None, "prefill_net_tok_s": None})

    def test_normal_values_hand_computed(self):
        """正常值手算：span=2.0、Σ=8192、rtt=0.2 → ttft_s = span = 2.0、
        prefill = 8192÷2 = 4096.0、ttft_net = 2.0−0.2 = 1.8、
        prefill_net = 8192÷1.8 = 4551.1。"""
        bp = _batch_prefill_point(8192, {"prefill_span_s": 2.0}, 0.2)
        self.assertEqual(bp["ttft_s"], 2.0)
        self.assertEqual(bp["prefill_tok_s"], 4096.0)
        self.assertEqual(bp["ttft_net_s"], 1.8)
        self.assertAlmostEqual(bp["prefill_net_tok_s"], 4551.1, delta=0.05)

    def test_conc1_equals_single_request(self):
        """conc=1 与单请求口径严格相等：批 span 与单请求 TTFT 同源（起点/
        终点取同一 perf_counter 值）——start=10.0/first=12.0 的批 stats →
        ttft 2.0、prefill 4096÷2 = 2048.0（与逐请求均值口径同值）；rtt=0
        时净口径 = 毛口径。"""
        req = {"req": 0, "start_abs": 10.0, "first_abs": 12.0,
               "last_abs": 14.0, "prompt_tokens": 4096, "out_tokens": 100}
        stats = _conc_batch_stats([req], 1)
        self.assertEqual(stats["prefill_span_s"], 2.0)
        bp = _batch_prefill_point(4096, stats, 0.0)
        self.assertEqual(bp["ttft_s"], 2.0)
        self.assertEqual(bp["prefill_tok_s"], 2048.0)
        self.assertEqual(bp["ttft_net_s"], 2.0)
        self.assertEqual(bp["prefill_net_tok_s"], 2048.0)


class TestDecodeSums(unittest.TestCase):
    """_decode_sums（点级 decode Σ÷Σ 口径，ADR-0074 追加）：decode 总耗时 =
    Σ各请求 decode 窗口（last_abs − first_abs）、均速 = Σout_tokens ÷ Σ窗口；
    conc=1 退化为单请求口径严格相等；pairs 空 → (None, None)。"""

    def test_empty_ok_returns_none_pair(self):
        """空批 / 无齐备时间轴的请求（缺 first_abs/last_abs/out_tokens 或
        last≤first）：两键均 None。"""
        self.assertEqual(_decode_sums([]), (None, None))
        self.assertEqual(_decode_sums([{"req": 0, "err": None}]), (None, None))
        self.assertEqual(_decode_sums(
            [{"req": 0, "first_abs": 10.0, "last_abs": 12.0,
              "out_tokens": 600}]), (2.0, 300.0))   # 对照：齐备请求正常产出
        self.assertEqual(_decode_sums(
            [{"req": 0, "first_abs": 10.0, "out_tokens": 600},
             {"req": 1, "last_abs": 12.0, "out_tokens": 600}]), (None, None))
        self.assertEqual(_decode_sums(
            [{"req": 0, "first_abs": 12.0, "last_abs": 12.0,
              "out_tokens": 600}]), (None, None))   # last≤first 守卫

    def test_single_request_equals_per_request(self):
        """单请求（conc=1 等价性）：Σ÷Σ 退化为逐请求口径——窗口 2.0s、
        out 600 → decode_time_s=2.0、decode_tok_s=300.0，与逐请求
        decode_tok_s = round(600÷2, 1) 严格相等。"""
        req = {"req": 0, "first_abs": 10.0, "last_abs": 12.0,
               "out_tokens": 600}
        dt, dc = _decode_sums([req])
        self.assertEqual(dt, 2.0)
        self.assertEqual(dc, round(600 / (req["last_abs"] - req["first_abs"]), 1))
        self.assertEqual(dc, 300.0)

    def test_two_unequal_windows_sum_ratio_not_mean(self):
        """两请求窗口不等：Σ÷Σ ≠ 逐请求速率算术均值——窗口 1.0s/100 tok
        与 3.0s/600 tok → Σ÷Σ = (100+600)÷4 = 175.0，而算术均值 =
        (100÷1 + 600÷3)÷2 = 150.0；总耗时 = Σ窗口 = 4.0。"""
        r1 = {"req": 0, "first_abs": 10.0, "last_abs": 11.0,
              "out_tokens": 100}
        r2 = {"req": 1, "first_abs": 10.0, "last_abs": 13.0,
              "out_tokens": 600}
        dt, dc = _decode_sums([r1, r2])
        self.assertEqual(dt, 4.0)
        self.assertEqual(dc, 175.0)
        mean = (r1["out_tokens"] / (r1["last_abs"] - r1["first_abs"])
                + r2["out_tokens"] / (r2["last_abs"] - r2["first_abs"])) / 2
        self.assertNotEqual(dc, round(mean, 1))   # 口径差异：≠ 均值 150.0

    def test_rounding(self):
        """精度：decode_time_s round 2、decode_tok_s round 1——窗口和
        0.33+0.33=0.66、Σout=100 → 均速 151.5。"""
        rs = [{"req": 0, "first_abs": 10.0, "last_abs": 10.33,
               "out_tokens": 50},
              {"req": 1, "first_abs": 10.0, "last_abs": 10.33,
               "out_tokens": 50}]
        dt, dc = _decode_sums(rs)
        self.assertEqual(dt, 0.66)
        self.assertEqual(dc, 151.5)


class TestConcBatchStatsAggregate(unittest.TestCase):
    """批级并发口径的复测聚合冒烟：6 新键（_ROUND 白名单）与其他标量同取
    各次复测均值，reps 子行逐次保真带新键；旧口径复测缺新键时聚合点不产出
    该键、子行按 None 留档（前端空值回退）。"""

    @staticmethod
    def _rep(decode, batch):
        rep = {"all_ok": True, "decode_tok_s": decode, "ttft_s": 1.0,
               "prefill_tok_s": 100.0, "prompt_tokens": 4000,
               "out_tokens": 500, "reqs": [{"req": 0, "err": None}]}
        rep.update(batch)
        return rep

    def test_six_new_keys_mean_and_reps_subrows(self):
        """2 个 rep 的迷你 point（6 新键不同值）→ 聚合取均值；reps 子行
        逐次带新键原值。"""
        batch1 = {"prefill_span_s": 2.0, "total_span_s": 4.0,
                  "decode_span_s": 2.0, "prefill_conc_tok_s": 750.0,
                  "decode_conc_tok_s": 250.0, "decode_peak_tok_s": 800.0}
        batch2 = {"prefill_span_s": 3.0, "total_span_s": 6.0,
                  "decode_span_s": 3.0, "prefill_conc_tok_s": 850.0,
                  "decode_conc_tok_s": 350.0, "decode_peak_tok_s": 1000.0}
        agg = _aggregate_reps([self._rep(50, batch1), self._rep(30, batch2)])
        self.assertEqual(agg["decode_tok_s"], 40.0)   # 对照标量同取均值
        self.assertEqual(agg["prefill_span_s"], 2.5)
        self.assertEqual(agg["total_span_s"], 5.0)
        self.assertEqual(agg["decode_span_s"], 2.5)
        self.assertEqual(agg["prefill_conc_tok_s"], 800.0)
        self.assertEqual(agg["decode_conc_tok_s"], 300.0)
        self.assertEqual(agg["decode_peak_tok_s"], 900.0)
        self.assertEqual(agg["reps"][0]["decode_peak_tok_s"], 800.0)
        self.assertEqual(agg["reps"][1]["prefill_span_s"], 3.0)
        self.assertEqual(agg["reps"][1]["decode_conc_tok_s"], 350.0)

    def test_legacy_reps_without_new_keys(self):
        """旧口径复测（无 6 新键）：聚合点不产出新键、子行以 None 留档。"""
        legacy = {"all_ok": True, "decode_tok_s": 50.0, "ttft_s": 1.0,
                  "prefill_tok_s": 100.0, "prompt_tokens": 4000,
                  "out_tokens": 500, "reqs": [{"req": 0, "err": None}]}
        agg = _aggregate_reps([dict(legacy), dict(legacy)])
        self.assertNotIn("prefill_span_s", agg)
        self.assertNotIn("decode_peak_tok_s", agg)
        self.assertIsNone(agg["reps"][0]["decode_peak_tok_s"])
        self.assertIsNone(agg["reps"][1]["decode_conc_tok_s"])


if __name__ == "__main__":
    unittest.main()
