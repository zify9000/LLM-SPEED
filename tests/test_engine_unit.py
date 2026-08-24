"""引擎纯逻辑回归测试：上下文构造（嵌套前缀/确定性/模块块级/wrap/零输入）、
复测聚合、错误语义化、超窗删减辅助。

运行（任一方式，无需 pytest 也可跑）：
    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/ -v
"""
import asyncio
import time
import unittest
import unittest.mock

import bench
from bench import (
    BenchRun,
    SCENARIOS,
    _aggregate_reps,
    _classify_http_error,
    _decode_burst,
    _fill,
    _is_ctx_overflow,
    _load_code_pool,
    _make_module_stream,
    _make_stream,
    _median,
    _net_elapsed,
    _trim_middle,
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


class TestFillStream(unittest.TestCase):
    """ADR-0005 确定性嵌套语料流的关键性质。"""

    def setUp(self):
        self.stream = _make_stream(["甲" * 100, "乙" * 200, "丙" * 300])

    def test_deterministic(self):
        self.assertEqual(self.stream, _make_stream(["甲" * 100, "乙" * 200, "丙" * 300]))

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
        items = _make_module_stream(self.BLOCKS).split("\n\n")
        self.assertEqual(len(items), sum(len(b) for b in self.BLOCKS))
        pos = {s: i for i, s in enumerate(items)}
        for block in self.BLOCKS:
            idxs = [pos[s] for s in block]
            self.assertEqual(idxs, sorted(idxs))                     # 块内保序
            self.assertEqual(idxs, list(range(idxs[0], idxs[0] + len(idxs))))   # 块内相邻

    def test_flat_stream_properties_carry_over(self):
        """嵌套前缀前提不变：同一流截取仍互为前缀（校准精确记账不受影响）。"""
        stream = _make_module_stream(self.BLOCKS)
        self.assertTrue(_fill(stream, 500).startswith(_fill(stream, 200)))

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


class TestScenarioPriors(unittest.TestCase):
    """ADR-0006 输入系数分场景先验：探测校准失败路径的兜底默认值。"""

    def test_in_cpt_priors(self):
        for sc in SCENARIOS.values():
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
    """ADR-0006 复测取中位：标量中位、decode 中位的一次、all_ok 多数决。"""

    def _rep(self, decode, all_ok=True):
        return {
            "all_ok": all_ok,
            "decode_tok_s": decode,
            "ttft_s": 1.0, "ttft_net_s": 0.9,
            "prefill_tok_s": 100.0, "prefill_net_tok_s": 110.0,
            "prompt_tokens": 4000, "out_tokens": 500,
            "cache_hit_tokens": 0, "reqs": [{"req": 0, "err": None}],
        }

    def test_median_of_odd(self):
        reps = [self._rep(50), self._rep(10), self._rep(30)]
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["decode_tok_s"], 30)
        self.assertTrue(agg["all_ok"])
        self.assertEqual(agg["n_reps"], 3)
        # reqs 明细取 decode 中位的那一次
        self.assertEqual(agg["reqs"][0]["req"], 0)

    def test_majority_rule(self):
        ok = [self._rep(50), self._rep(10)]
        bad = self._rep(5, all_ok=False)
        self.assertTrue(_aggregate_reps(ok + [bad])["all_ok"])       # 2/3 多数
        self.assertFalse(_aggregate_reps([ok[0], bad, bad])["all_ok"])  # 1/3 少数

    def test_median_even(self):
        self.assertEqual(_median([1, 4]), 2.5)
        self.assertEqual(_median([4, 1, 9, 2]), 3.0)

    def test_max_gap_takes_worst_across_reps(self):
        """停滞诊断 max_gap_s 取各次复测最差（max，不是中位）；无停滞不产出该键。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0]["max_gap_s"], reps[1]["max_gap_s"], reps[2]["max_gap_s"] = 1.5, 7.0, 2.0
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["max_gap_s"], 7.0)              # 各次最差，非中位 2.0
        self.assertNotIn("max_gap_s", _aggregate_reps([self._rep(1), self._rep(2)]))

    def test_stall_takes_worst_across_reps(self):
        """停滞复合记账（stall_s/stall_count）与 max_gap_s 同取各次复测最差。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0].update(stall_s=1.2, stall_count=2)
        reps[1].update(stall_s=6.5, stall_count=1)
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["stall_s"], 6.5)
        self.assertEqual(agg["stall_count"], 2)   # 次数取各次最大，不随时长那条
        clean = _aggregate_reps([self._rep(1), self._rep(2)])
        self.assertFalse(clean.get("stall_s"))
        self.assertFalse(clean.get("decode_burst"))

    def test_burst_majority_across_reps(self):
        """突发交付多数决：过半复测突发才标记；且不被中位次残留污染。"""
        reps = [self._rep(50), self._rep(40), self._rep(30)]
        reps[0]["decode_burst"] = reps[1]["decode_burst"] = True
        self.assertTrue(_aggregate_reps(reps)["decode_burst"])      # 2/3 多数
        reps[1]["decode_burst"] = None
        self.assertFalse(_aggregate_reps(reps)["decode_burst"])     # 1/3 少数
        # 中位次本身带 True 但多数不成立时，显式覆写为否
        only_mid = [self._rep(10), self._rep(20), self._rep(30)]
        only_mid[1]["decode_burst"] = True
        self.assertFalse(_aggregate_reps(only_mid)["decode_burst"])

    def test_decode_burst_detection(self):
        """突发甄别：亚毫秒平均间隔 + 足够 chunk 数 → 缓冲冲刷；真流式不误判。"""
        self.assertTrue(_decode_burst(32, 0.01))        # 32 chunk / 10ms → 0.3ms
        self.assertFalse(_decode_burst(32, 0.5))        # 16ms 间隔，真流式
        self.assertFalse(_decode_burst(3, 0.001))       # chunk 太少不足为凭
        self.assertFalse(_decode_burst(32, None))       # 无 decode 窗口
        self.assertFalse(_decode_burst(1, 0.001))       # 除零保护

    def test_all_failed_pool_falls_back_to_reps(self):
        """all_ok 全 False：聚合池回退为 reps 本身，标量中位照常算，all_ok=False。"""
        reps = [self._rep(50, all_ok=False), self._rep(10, all_ok=False),
                self._rep(30, all_ok=False)]
        agg = _aggregate_reps(reps)
        self.assertFalse(agg["all_ok"])
        self.assertEqual(agg["decode_tok_s"], 30)            # 回退池的中位
        self.assertEqual(agg["n_reps"], 3)

    def test_decode_adj_in_median(self):
        """decode_tok_s_adj（空窗校正口径）参与聚合中位；缺该值的复测不计入。"""
        reps = [self._rep(50), self._rep(10), self._rep(30)]
        reps[0]["decode_tok_s_adj"] = 60.0
        reps[1]["decode_tok_s_adj"] = 10.0
        reps[2]["decode_tok_s_adj"] = 40.0
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["decode_tok_s_adj"], 40.0)      # [60,10,40] 中位
        reps[2]["decode_tok_s_adj"] = None                   # 该次无校正口径
        agg2 = _aggregate_reps(reps)
        self.assertEqual(agg2["decode_tok_s_adj"], 35.0)     # 仅 [60,10] 中位


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


if __name__ == "__main__":
    unittest.main()
