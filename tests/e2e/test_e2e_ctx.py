"""e2e 组 · 上下文窗治理与物理速率闸（mock 网关真跑 + 引擎最小矩阵）。

覆盖：
- K 上下文窗治理：超窗整档 point_skipped（ADR-0005）、超窗贴边裁减实测
  （ctx_edge 留痕原档位）、未识别秒拒的删减回退（CTX_EDGE_FB_RATIO 军备）
  与阶梯耗尽追加提示、非贴边档不军备且空流未耗尽专属预算不得抢先改 shape、
  空流专属重试耗尽后强制删减改 shape 自救（ADR-0085）、超窗删减中段重试、
  fastllm 系软超窗判错（content / reasoning 双通道）、输出塌缩守卫（自定义
  占位串绕过已知文案匹配时仍按 empty_reply_suspected 收口，不出假速度）。
- L 物理速率闸（先验带分支）集成例：MOCK_PP 拉到 8000 的假成功必须判 err，
  且实测速率不得污染 prefill 先验曲线 / 嵌套前缀记账 / 增量系数校准。

自 tests/test_e2e_mock.py 拆分（原文件已删）。
精简记录（原 K 簇 10 例 → 8 例、L 簇 2 例 → 1 例；另收回拆分漏网的
test_collapsed_placeholder_reply_suspected 一例、断言原样保留 → K 簇 9 例）：
- test_edge_fallback_not_armed_below_ratio 与
  test_empty_stream_retry_keeps_ctx_before_fallback 断言同轴（「专属重试预算
  未耗尽前不得改 shape」的两面：非贴边档不军备 / 瞬时空流先重试），合并为
  test_edge_fallback_not_armed_before_retry_exhausted 一条、分两段执行。
- test_soft_ctx_overflow_rejected 与 test_soft_ctx_overflow_via_reasoning_channel
  仅占位串通道不同（content / reasoning），断言同构，合并为一条按通道参数化。
- L 簇只留 test_implausible_prefill_rate_not_registered（大 prompt 集成例）：
  另一例 test_implausible_prefill_rate_small_prompt_not_gated 验证的「小 prompt
  未命中量 <8K 只按绝对上限判、不误杀诚实点」分支已由 tests/unit 的
  TestPrefillGateCallSite 逐条覆盖（纯调用点断言，无需再起网关），故删除。

运行（任一方式，都要求**从仓库根目录**执行，无需 pytest 也可跑）：
    .venv/bin/python -m pytest tests/e2e/test_e2e_ctx.py -q
    .venv/bin/python -m unittest tests.e2e.test_e2e_ctx -v
    .venv/bin/python -m unittest discover -s tests/e2e -p "test_e2e_ctx.py"
"""
import asyncio

# 顺序是硬约束：_harness 顶部先 import tests._shared（落 MOCK_* 播种），之后
# 才 import mock_server——mock_server 在 import 时读环境变量，反序会采到默认值
# （如 PP=1200 而非 3000），断言口径全错。
from tests.e2e._harness import MockGatewayTest

import bench
import mock_server
from bench import CTX_HEADROOM, BenchRun


class TestCtxWindowE2E(MockGatewayTest):
    """上下文窗治理（K）与物理速率闸集成例（L）。"""

    async def test_ctx_over_window_skipped(self):
        """超窗跳过：model_max_ctx=8K 时 65536 档只发 point_skipped，
        不产生对应 point（0K 档照常，ADR-0005）。"""
        _, events = await self._run_and_collect(
            ctx_list=[0, 65536], model_max_ctx={"mock-llm-7b": 8192})
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(skipped), 1, f"65536 档应恰好一条 point_skipped: {skipped}")
        self.assertEqual(skipped[0]["ctx"], 65536)
        self.assertEqual(skipped[0]["count"], 1)
        self.assertIn("超出部署上限", skipped[0]["msg"],
                      "point_skipped 须带 msg 供前端常驻展示（瞬态状态行会被冲掉）")
        ctxs = [e["point"]["ctx_target"] for e in events if e.get("type") == "point"]
        self.assertNotIn(65536, ctxs, "超窗档不得产生 point")
        self.assertIn(0, ctxs, "0K 档应照常出点")

    async def test_ctx_over_window_edge_trim(self):
        """超窗贴边裁减：档位连同输出预算超出部署上限、但超出量 ≤10%·max_ctx
        时不跳过——上下文裁减到能放下（ctx_eff = max_ctx − max_tokens −
        CTX_HEADROOM）实测一次；point.ctx_target 记实测档位、ctx_edge 留痕
        原档位（前端按 ctx_target 展示实际档位是有意的诚实呈现）。"""
        # ctx=7168：excess = 7168+512+CTX_HEADROOM−8192 = 512 ≤ 10%·8192 → 贴边
        _, events = await self._run_and_collect(
            ctx_list=[0, 7168], model_max_ctx={"mock-llm-7b": 8192},
            max_tokens=512)
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2, f"两档都应出点: {pts}")
        trimmed = next(p for p in pts if p["ctx_target"] != 0)
        self.assertEqual(trimmed["ctx_target"],
                         8192 - 512 - CTX_HEADROOM,
                         "ctx_target 应记裁减后的贴边档位")
        self.assertEqual(trimmed["ctx_edge"], 7168,
                         "ctx_edge 应留痕贴边前的原档位")
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(skipped, [], "阈值内的超窗档不得再跳过")
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("贴边测量" in s for s in statuses),
                        f"应有贴边裁减播报: {statuses}")

    async def test_edge_fallback_trim_on_unrecognized_rejection(self):
        """贴边档未识别秒拒的回退裁减（CTX_EDGE_FB_RATIO 军备）：cfg 部署
        max_ctx=3800、ctx=2700（2700+16+1024=3740 ≥ 3800×0.85 军备、
        预检 excess=-60 不跳档不贴边）、mock 对 prompt+预算 >2600 回不含
        超窗关键词的 500（MOCK_HARD_FAIL_CTX）——瞬时兜底重试耗尽后按
        CTX_RETRY_KEEP 阶梯删减回退：点最终成功、prompt 明显缩小、req 留
        edge_fallback 痕、有状态播报。"""
        old = mock_server.HARD_FAIL_CTX
        mock_server.HARD_FAIL_CTX = 2600
        try:
            _, events = await self._run_and_collect(
                ctx_list=[2700], max_tokens=16,
                model_max_ctx={"mock-llm-7b": 3800})
        finally:
            mock_server.HARD_FAIL_CTX = old
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertTrue(pts[0]["all_ok"],
                        f"贴边回退裁减后点应成功: {pts[0].get('reqs')}")
        # 首档回退（0.75×）裁后 prompt ≈2200 落入窗口：显著小于构造目标 2700
        self.assertLess(pts[0]["prompt_tokens"], 2550)
        self.assertGreater(pts[0]["prompt_tokens"], 1900)
        req = pts[0]["reqs"][0]
        self.assertEqual(req.get("edge_fallback"), 1,
                         f"首档回退即应救回并留痕: {req}")
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("回退重试" in s and "疑似" in s for s in statuses),
                        f"应有贴边回退状态播报: {statuses}")

    async def test_edge_fallback_exhausted_appends_hint(self):
        """回退阶梯耗尽：HARD_FAIL_CTX=700 让 0.75/0.5/0.3 三档裁减（裁后
        prompt 分别 ≈1750/1250/720 tokens）仍超限——点判失败，err 文本追加
        「疑似上下文/显存超限」提示；点级 err 落档（失败原因随存档可回溯，
        不再只有瞬态 toast）。"""
        old = mock_server.HARD_FAIL_CTX
        mock_server.HARD_FAIL_CTX = 700
        try:
            _, events = await self._run_and_collect(
                ctx_list=[2700], max_tokens=16,
                model_max_ctx={"mock-llm-7b": 3800})
        finally:
            mock_server.HARD_FAIL_CTX = old
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"], "阶梯耗尽后点必须判失败")
        req = pts[0]["reqs"][0]
        self.assertIn("大概率上下文/显存超限", req.get("err") or "")
        self.assertIn("大概率上下文/显存超限", pts[0].get("err") or "",
                      "点级 err 须落档（repeats=1 不经聚合层）")

    async def test_edge_fallback_not_armed_before_retry_exhausted(self):
        """「专属重试预算未耗尽前不得改 shape」的两面（原
        test_edge_fallback_not_armed_below_ratio +
        test_empty_stream_retry_keeps_ctx_before_fallback 合并为一条、分两段）。

        ①非贴边档不军备：未配 model_max_ctx 时持续 500（FLAKY 钩子）只走
        瞬时兜底重试，不做删减回退——err 无超限提示、无 edge_fallback 痕。
        ②空流未耗尽专属预算：瞬时空流（服务端抖动，同文本重试确实有效）必须
        保留原语义——先走同文本重试，不得因一次空流就删减语料，否则口径被
        污染（ADR-0085 边界，①的反面）。"""
        with self.subTest(case="persistent_500_non_edge"):
            mock_server.FLAKY_FAIL = 100   # 持续 500，瞬时重试额度必然耗尽
            try:
                _, events = await self._run_and_collect(
                    skip_probe=True, ctx_list=[4096], max_tokens=16)
            finally:
                mock_server.FLAKY_FAIL = 0
            pts = [e["point"] for e in events if e.get("type") == "point"]
            self.assertEqual(len(pts), 1)
            self.assertFalse(pts[0]["all_ok"])
            req = pts[0]["reqs"][0]
            self.assertNotIn("显存超限", req.get("err") or "")
            self.assertNotIn("edge_fallback", req)

        with self.subTest(case="empty_stream_retry_first"):
            mock_server.EMPTY_STREAM_FAIL = 1   # 仅首发空流，重试即成功
            try:
                _, events = await self._run_and_collect(
                    skip_probe=True, ctx_list=[4096], max_tokens=16)
            finally:
                mock_server.EMPTY_STREAM_FAIL = 0
            pts = [e["point"] for e in events if e.get("type") == "point"]
            self.assertEqual(len(pts), 1)
            self.assertTrue(pts[0]["all_ok"])
            req = pts[0]["reqs"][0]
            self.assertNotIn("edge_fallback", req,
                             "同文本重试已自愈时不得改 shape")
            self.assertEqual(req.get("retried"), 1, "应留空流重试痕")

    async def test_empty_stream_exhausted_forces_edge_fallback(self):
        """ADR-0085：**非贴边**档的同文本空流重试耗尽后，必须改 shape 自救。

        复现 2026-09-22 实机故障：Lvllm 服务端按 shape 重建 Triton kernel 时
        CUDA OOM，对外表现为 200 + 零 token（不体现在 HTTP 状态上）。此时
        同文本重试是同一 shape、必然重走同一条编译路径，而唯一能改 shape 的
        贴边回退因档位不贴边（4K 档 vs max_ctx 262144，离门槛差 29 倍）而
        未军备——原始实现因此反复撞墙直到判失败。

        构造：空流钩子设为 1+EMPTY_STREAM_RETRY_MAX 次（恰好耗尽同文本重试
        预算），耗尽后服务端恢复——断言引擎经删减回退（改 shape）把该点救回。
        服务端若持续坏下去（钩子设更大）则任何重试策略都救不回，不在本用例
        范围（那种情况断言的是错误口径正确，见 test_empty_stream_* 单元用例）。"""
        mock_server.EMPTY_STREAM_FAIL = 1 + bench.EMPTY_STREAM_RETRY_MAX
        try:
            run, events = await self._run_and_collect(
                skip_probe=True, ctx_list=[4096], max_tokens=16)
        finally:
            mock_server.EMPTY_STREAM_FAIL = 0
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        # 核心断言：同文本重试耗尽后，改 shape 回退把该点救回
        self.assertTrue(pts[0]["all_ok"],
                        f"空流耗尽后应经改 shape 回退救回，实际："
                        f"{pts[0].get('err')}")
        req = pts[0]["reqs"][0]
        self.assertIn("edge_fallback", req,
                      "改 shape 回退必须留痕（存档据此区分救回路径）")
        # 非贴边档不得谎报超限（原提示会把排查方向带偏到 max_ctx 上）
        self.assertNotIn("大概率上下文/显存超限", req.get("err") or "")

    async def test_ctx_overflow_trim_retry(self):
        """超窗回退：prompt 临近模型窗口上限、首请求被判 context exceeded 时，
        引擎删减中段填充语料重试而非直接判失败——点最终成功且实际
        prompt_tokens 小于构造目标（MOCK_MAX_CTX=3800：4K 档首请求
        ~4096+16 tokens 超限，删减 75% 后 ~3126+16 落入窗口）。"""
        old = mock_server.MAX_CTX
        mock_server.MAX_CTX = 3800
        try:
            run, events = await self._run_and_collect(
                ctx_list=[4096], max_tokens=16)
        finally:
            mock_server.MAX_CTX = old
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertTrue(pts[0]["all_ok"],
                        f"删减重试后点应成功: {pts[0].get('reqs')}")
        # 实际 prompt_tokens ≈ 0.75×4096（删减后），显著小于构造目标 4096
        self.assertLess(pts[0]["prompt_tokens"], 3300)
        self.assertGreater(pts[0]["prompt_tokens"], 2800)
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("删减" in s and "重试" in s for s in statuses),
                        f"应有删减重试状态播报: {statuses}")

    async def test_soft_ctx_overflow_rejected(self):
        """fastllm 系软超窗（MOCK_SOFT_MAX_CTX）：HTTP 200 + 正文被替换为
        "prompt too long"（finish=stop、带 usage）——不再记成「拒绝耗时 ÷
        prompt 长度」的数十万 tok/s 假 prefill，该点按错误收口。

        占位串通道参数化（原 test_soft_ctx_overflow_rejected +
        test_soft_ctx_overflow_via_reasoning_channel 合并为一条）：content 为
        默认形态，reasoning 复现思考模型实测形态（content 为空、占位串走
        reasoning_content，实测事故：该通道下 38 万 tok/s 假 prefill 入档）
        ——头部文本采集覆盖两个通道，识别与判错口径一致。"""
        old_s, old_f = mock_server.SOFT_MAX_CTX, mock_server.SOFT_FIELD
        try:
            for field in ("content", "reasoning"):
                with self.subTest(field=field):
                    mock_server.SOFT_MAX_CTX = 3800   # 4K 档首请求 ~4096+16 tokens 超限
                    mock_server.SOFT_FIELD = field
                    _, events = await self._run_and_collect(ctx_list=[4096], max_tokens=16)
                    pts = [e["point"] for e in events if e.get("type") == "point"]
                    self.assertEqual(len(pts), 1)
                    self.assertFalse(pts[0]["all_ok"], "软超窗点必须判失败，不得出假速度")
                    req = pts[0]["reqs"][0]
                    self.assertIsNotNone(req.get("err"))
                    self.assertIn("prompt too long", req["err"])
                    self.assertIn("上下文", req["err"])
                    self.assertIsNone(req.get("prefill_tok_s"), "软超窗不得记录 prefill 读数")
                    ticks = [e["msg"] for e in events if e.get("type") == "tick"
                             and e.get("phase") == "error"]
                    self.assertTrue(any("占位回复" in t for t in ticks),
                                    f"应有软超窗错误帧: {ticks}")
        finally:
            mock_server.SOFT_MAX_CTX, mock_server.SOFT_FIELD = old_s, old_f

    async def test_collapsed_placeholder_reply_suspected(self):
        """输出塌缩守卫（通用保险）：占位串非已知文案（MOCK_SOFT_TEXT 自定义，
        绕过 "prompt too long" 精确匹配）但 finish=stop、单 token、prompt 很大
        ——判 empty_reply_suspected 按错误收口，不出假速度。"""
        old_s, old_t = mock_server.SOFT_MAX_CTX, mock_server.SOFT_TEXT
        mock_server.SOFT_MAX_CTX, mock_server.SOFT_TEXT = 8000, "context overflow"
        try:
            # 16K 档：实测 prompt 远大于守卫门槛 8192（校准漂移不贴边）
            _, events = await self._run_and_collect(ctx_list=[16384], max_tokens=16)
        finally:
            mock_server.SOFT_MAX_CTX, mock_server.SOFT_TEXT = old_s, old_t
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"], "塌缩占位回复必须判失败")
        req = pts[0]["reqs"][0]
        self.assertIn("empty_reply_suspected", req.get("err") or "")
        self.assertIsNone(req.get("prefill_tok_s"), "塌缩点不得记录 prefill 读数")

    async def test_implausible_prefill_rate_not_registered(self):
        """物理速率闸（先验带分支）：MOCK_PP 拉到 8000（> 先验 300 的 10×）
        模拟占位软拒绝类假成功——16K 档未命中量 ≥8K 走先验带，点判 err、
        all_ok=False，且实测速率不得登记进 prefill_curve、不得更新 prefix_kb
        嵌套前缀记账与增量系数校准（污染后续档位的构造与估值先验）。"""
        old_pp = mock_server.PP
        mock_server.PP = 8000.0
        try:
            run = BenchRun({
                "gateway_url": self.base_url,
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [16384], "concurrencies": [1],
                "max_tokens": 16, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            key = ("mock-llm-7b", "creative")
            run.cpt_calib[key] = 1.8                    # 跳过探测
            run.prefix_kb[key] = (3000.0, 2048.0)       # 预置记账，断言不被污染
            run.prefill_curve[key] = [(2048, 300.0)]    # 预置先验：闸门 10× = 3000
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=30)
                    if ev is None:
                        break
                    events.append(ev)
            finally:
                await task
        finally:
            mock_server.PP = old_pp
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"], "超先验 10× 的假速率必须判失败")
        req = pts[0]["reqs"][0]
        self.assertIn("物理上不可能", req.get("err") or "")
        self.assertEqual(run.prefill_curve[key], [(2048, 300.0)],
                         "先验曲线不得被假速率污染")
        self.assertEqual(run.prefix_kb[key], (3000.0, 2048.0),
                         "嵌套前缀记账不得被假成功污染")
        self.assertIsNone(run.cpt_marginal.get(key),
                          "增量系数校准不得被假成功污染")
        self.assertIn("done", [e["type"] for e in events])
