"""口径计算与聚合统计组：速率先验曲线与缓存命中/污染判别、净耗时与交付
形态口径（突发/每批 token）、校准与实时读数辅助（加权滑动平均/比例折算）、
背景输出估算、复测聚合（标量均值/代表样本/多数决）、并发总吞吐 raw/adj
双口径、停滞回吐 episode 识别与 _one 收口剔除、decode 峰值滑窗、批级并发
6 指标与批 prefill 点、decode Σ÷Σ、填充字符校正、媒体聚合口径（ADR-0020）。

自 tests/test_engine_unit.py 拆分。

运行（先 cd 到仓库根目录）：
    .venv/bin/python -m pytest tests/unit/test_engine_metrics.py -q
    .venv/bin/python -m unittest tests.unit.test_engine_metrics -v
"""
import asyncio
import json
import time
import unittest
import unittest.mock

import bench
from bench import (
    DECODE_PEAK_WINDOW_S,
    BenchRun,
    SCENARIOS,
    _aggregate_media_reqs,
    _aggregate_reps,
    _batch_prefill_point,
    _cache_hit_verdict,
    _cache_pollution_verdict,
    _conc_batch_stats,
    _correct_filler_chars,
    _decode_burst,
    _decode_peak_rate,
    _decode_sums,
    _flush_episodes,
    _net_elapsed,
    _out_hint_tail,
    _prior_suspect,
    _ratio_refold,
    _tok_per_batch,
    _total_decode_rates,
    _wma,
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

    def test_near_duplicate_reps_do_not_break_extrapolation(self):
        """repeats≥2 的同档近重复点不得毁掉外推（2026-10-17 真机事故）。

        曲线由 _note_prefill_rate 逐复测登记，同一档位会留下三个「几乎同 x」
        的点（131018/131075/131411，x 差 0.26%）。旧实现拿它们当末两点算斜率：
        分母（Δ log2）只有 0.0037，斜率被放大成 −2897/倍频程，外推到 256K 得负值
        再钳到 0.35× 地板 → 等待期显示 1310 tok/s（实测 3702）。
        近重复合并成一组后，外推沿 64K→128K 的真实斜率。"""
        curve = [(2048, 1200.0),
                 (4112, 1715.0),
                 (8212, 1919.8), (8191, 1999.0), (8194, 1994.8),
                 (16307, 2078.0), (16340, 2185.2), (16374, 2183.9),
                 (32765, 3160.3), (32769, 3163.8), (32781, 3168.0),
                 (63657, 3486.6), (65493, 3480.4), (65596, 3465.6),
                 (131411, 3744.2), (131018, 3754.4), (131075, 3754.9)]
        got = rate_prior(curve, 262144)
        # 真值 3660：先验应落在真值的 [0.8, 1.3] 带内，而不是 0.35× 地板
        self.assertGreater(got, 0.8 * 3660, f"外推被钳到地板（{got:.0f}）")
        self.assertLess(got, 1.3 * 3660, f"外推越上界（{got:.0f}）")
        # 各档等待期同样不得靠近重复点算斜率（旧实现 16K 档算出 672，真值 2148）
        self.assertGreater(rate_prior(curve[:8], 32768), 0.7 * 3163)

    def test_near_duplicate_merge_averages(self):
        """近重复点合并：组内速率取均值（同档三轮的噪声互打对）。"""
        from bench import _rate_groups
        g = _rate_groups([(8191, 2000.0), (8194, 1990.0), (8212, 1980.0),
                          (16384, 2100.0)])
        self.assertEqual(len(g), 2)
        self.assertAlmostEqual(g[0][1], 1990.0)
        # 差超过合并阈值（log2 0.05 ≈ 3.5%）就不归同一组
        self.assertEqual(len(_rate_groups([(4096, 1.0), (4400, 2.0)])), 2)


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

    def test_short_stall_below_threshold_ignored(self):
        """停滞门限 0.5s：0.4s 空窗只是正常抖动，不触发回吐剔除。"""
        gaps = [0.03] * 9 + [0.4] + [0.0002] * 20
        ev_toks = [1.0] * 31
        self.assertEqual(_flush_episodes(gaps, ev_toks), [])


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


if __name__ == "__main__":
    unittest.main()
