"""e2e 流水线组（进程内 mock 网关 + 引擎最小矩阵全链路）。

覆盖：cfg 事件首发且脱敏、prefill 批级口径恒等式（TTFT=批 span、速度=Σ÷span）
与 mock 设定一致（±20%）、双并发总吞吐、mock 运行不落盘、echo 回复模式预填
续写、多并发构建偏离早判（首个带 usage 的请求完成即判定、在途余量取消）、
异常输出弃测重测全链路（MOCK_EARLY_STOP 一次性钩子：首测 early-stop 弃测、
重测正常出点、reps_discarded 留痕带 rep 轮次号与 in_text 输入全文；正常 req
存档无 _in_full/in_text 瞬态字段；reps 摘要逐次带 anomaly）、瞬时失败兜底重试
全链路（MOCK_FLAKY_FAIL 一次性钩子：首发点请求 500 → 兜底重试成功、reqs 留
retried 痕、SSE 播报兜底重试状态）、多地址择优（server.pick_gateway_url：
优选地址排前即被选中、顺序互换不翻转、单地址直通、全不可达回退首个）、
thinking 协商（无思考网关零参数下发 / 默认思考模型被检出 / 网关拒参 400 回退
一次 / 进程级能力记忆零试探 / auto 跟随模型默认形态）、无前缀 chat 端点回退
锁定、repeats=3 均值聚合、输入系数校准命中目标、temperature 拒参锁定。

自 tests/test_e2e_mock.py 拆分（原文件已删）。

运行（任一方式，都要求**从仓库根目录**执行，无需 pytest 也可跑）：
    .venv/bin/python -m pytest tests/e2e/test_e2e_pipeline.py -q
    .venv/bin/python -m unittest tests.e2e.test_e2e_pipeline -v
    .venv/bin/python -m unittest discover -s tests/e2e -p "test_e2e_pipeline.py"
"""
import asyncio
import os
import shutil
import tempfile

# 顺序是硬约束：先 import tests._shared（_harness 顶部已触发 MOCK_* 播种），
# 再 import mock_server（后者在 import 时读环境变量）。
from tests.e2e._harness import MockGatewayTest      # noqa: E402

import bench                                        # noqa: E402
import mock_server                                  # noqa: E402
from bench import BenchRun                          # noqa: E402


class TestE2EPipeline(MockGatewayTest):

    async def test_full_pipeline(self):
        tmp = tempfile.mkdtemp(prefix="llmspeed-e2e-")
        self.addAsyncCleanup(shutil.rmtree, tmp, ignore_errors=True)
        results_dir = os.path.join(tmp, "results")   # 不存在的子路径，断言 mock 运行不落盘
        run = BenchRun(self.std_cfg(ctx_list=[0, 4096], concurrencies=[1, 2],
                                    cache="bust"), results_dir=results_dir)
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())

        events = await self._drain(q, task)
        points = [e["point"] for e in events if e.get("type") == "point"]

        types = [e["type"] for e in events]
        self.assertEqual(types[0], "cfg", "cfg 事件必须首发（ADR-0009）")
        self.assertNotIn("api_key", events[0]["cfg"], "cfg 事件必须脱敏")
        self.assertIn("done", types)
        self.assertEqual(len(points), 4, "2 档上下文 × 2 并发 = 4 个点")

        by_key = {(p["ctx_target"], p["concurrency"]): p for p in points}
        self.assertTrue(all(p["all_ok"] for p in points), "全点应成功")

        # prefill 批级口径（ADR-0074）：TTFT = 批开始→全部首 token、速度 =
        # Σprompt_tokens ÷ prefill_span_s、净口径扣 RTT——恒等式用点自身字段
        # 交叉验证（Σ 取自 reqs 明细，比魔法数字稳）
        for p in points:
            tok_sum = sum(r["prompt_tokens"] for r in p["reqs"])
            span = p.get("prefill_span_s")
            self.assertIsNotNone(span, "全成功批必须产出批级 prefill 口径")
            self.assertEqual(p["ttft_s"], span,
                             "批级 TTFT 应等于 prefill_span_s（批开始→全部首 token）")
            self.assertAlmostEqual(p["prefill_tok_s"], round(tok_sum / span, 1),
                                   delta=0.11,
                                   msg="批级 prefill 应 = Σprompt_tokens ÷ span")
            self.assertLess(p["ttft_net_s"], p["ttft_s"], "净口径应扣除 RTT 基线")
            self.assertGreaterEqual(p["prefill_net_tok_s"], p["prefill_tok_s"],
                                    "净口径 span 更小，净 prefill 不得低于毛口径")
            self.assertAlmostEqual(p["decode_tok_s"], 300, delta=75,
                                   msg=f"ctx={p['ctx_target']} decode 偏离 mock 设定")
            # 无停滞侧：空窗校正口径不得偏离毛值（DECODE_GAP_CAP 只削病理拖尾）；
            # conc>1 停滞/回吐诊断退役（ADR-0074），flush/校正口径不产出
            adj = p.get("decode_tok_s_adj")
            if p["concurrency"] > 1:
                self.assertIsNone(adj)
                self.assertIsNone(p.get("flush_tok"))
            elif adj is not None:
                self.assertAlmostEqual(adj, p["decode_tok_s"], delta=1.0,
                                       msg=f"ctx={p['ctx_target']} 无停滞点校正值偏离毛值")

        # conc=1 点幅值（批级口径与单请求口径严格相等，原断言不变）：mock 端到端
        # 无网络噪声、RTT 已扣——ctx=0 档 TTFT 仅 ~30ms 固定开销占比大，只做
        # 合理性区间断言；4K 档 TTFT ~1.4s，固定开销可忽略，±15% 即足够敏感
        for p in points:
            if p["concurrency"] != 1:
                continue
            if p["ctx_target"] == 0:
                self.assertTrue(1000 <= p["prefill_net_tok_s"] <= 6000,
                                msg=f"ctx=0 净 prefill {p['prefill_net_tok_s']} 越出合理区间")
            else:
                self.assertAlmostEqual(p["prefill_net_tok_s"], 3000, delta=450,
                                       msg=f"ctx={p['ctx_target']} 净 prefill 偏离 mock 设定")

        # conc>1 批级幅度：mock 并行 prefill 时批级口径把并发 prefill 记在批头
        # ——4K 档 ≈ 2× 单请求均值（conc=2 实测 ~5940 vs conc=1 ~2977）
        p1 = by_key[(4096, 1)]["prefill_net_tok_s"]
        p2 = by_key[(4096, 2)]["prefill_net_tok_s"]
        self.assertGreater(p2, 1.5 * p1,
                           f"批级 prefill {p2} 应 ≈ 2× 单请求 {p1}")

        # 双并发总吞吐 ≈ 2 × 单请求 decode
        s1 = by_key[(4096, 1)]["decode_tok_s"]
        t2 = by_key[(4096, 2)]["decode_total_tok_s"]
        self.assertIsNotNone(t2, "decode 重叠充分时必须产出总吞吐")
        self.assertTrue(1.5 * s1 <= t2 <= 2.6 * s1, f"总吞吐 {t2} 应 ≈ 2×{s1}")

        # 正常 req 存档不带输入全文：瞬态 _in_full 已在点定稿前剥离，
        # 异常取证键 in_text 也只在异常相关记录出现
        for p in points:
            for r in p["reqs"]:
                self.assertNotIn("_in_full", r, "正常 req 不得带 _in_full")
                self.assertNotIn("in_text", r, "正常 req 不得带 in_text")

        # 回复模式（ADR-0021）默认自由回答：消息序列 [system, user]，无预填
        self.assertEqual(mock_server.LAST_CHAT_ROLES, ["system", "user"])

        # 口径版本（ADR-0083）：走真实测速路径的每个点都必须带点级版本戳，
        # 否则拼接/复测把不同口径的点混进同一存档时无法判别
        self.assertTrue(points)
        for p in points:
            self.assertEqual(p.get("metric_version"), bench.METRIC_VERSION,
                             "测点缺 metric_version——口径版本机制失效")

        # mock 运行不落历史记录（ADR-0010）
        self.assertTrue(run.mock_seen)
        self.assertFalse(os.path.isdir(results_dir), "mock 运行不得创建 results 目录")
        self.assertFalse(os.listdir(tmp), "mock 运行不得向 results_dir 写入任何文件")

    async def test_reply_mode_echo(self):
        """回复模式=模板续写（ADR-0021）：创意/代码场景请求末尾带 assistant
        预填消息（mock 记录角色序列），测速照常出点；零输入档不受影响。"""
        tmp = tempfile.mkdtemp(prefix="llmspeed-e2e-")
        self.addAsyncCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run = BenchRun(self.std_cfg(ctx_list=[0, 4096], cache="bust",
                                    reply_mode="echo"),
                       results_dir=os.path.join(tmp, "results"))
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())
        events = await self._drain(q, task)
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 2, "0K + 4K 共 2 个点")
        self.assertTrue(all(p["all_ok"] for p in points))
        self.assertTrue(all(p["decode_tok_s"] for p in points))
        # 最后一次 chat 请求是 4K 点（echo 模式）：末尾带 assistant 预填
        self.assertEqual(mock_server.LAST_CHAT_ROLES,
                         ["system", "user", "assistant"])

    async def test_anomaly_early_stop_retest(self):
        """异常输出弃测重测全链路（ADR-0032，MOCK_EARLY_STOP 一次性钩子）：
        echo 模式下首测 finish=stop 仅 4/64 tokens（真实案例形态：4K 档
        4/512）→ 弃测整批重跑，重测批恢复满预算正常输出。不打桩
        _run_point/_detect_rep_anomaly——走真实 HTTP 流式链路验证守卫编排。
        max_tokens=64 压在 80 字符退化判据之下：正常批 "测"×64 不会命中
        degenerate，重测成功即干净正常（点位 anomaly 键缺席），断言只表达
        early_stop 机制本身。档位取 4K：输入全文超 200 采样上限，验证
        reps_discarded.in_text 保留全长输入原文。"""
        tmp = tempfile.mkdtemp(prefix="llmspeed-e2e-")
        self.addAsyncCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run = BenchRun(self.std_cfg(ctx_list=[4096], cache="bust",
                                    reply_mode="echo"),
                       results_dir=os.path.join(tmp, "results"))
        run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测：钩子留给出点请求
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        old_early = mock_server.EARLY_STOP
        mock_server.EARLY_STOP = 4   # 首发点请求照常 prefill 后只吐 4 token 即 stop
        task = asyncio.create_task(run.run())
        try:
            events = await self._drain(q, task)
        finally:
            mock_server.EARLY_STOP = old_early
        points = [e["point"] for e in events if e.get("type") == "point"]
        statuses = [e.get("msg") or "" for e in events if e.get("type") == "status"]
        toasts = [e.get("msg") or "" for e in events
                  if e.get("type") == "status" and e.get("toast")]
        self.assertEqual(len(points), 1, "4K 单点")
        p = points[0]
        self.assertEqual(p.get("reps_discarded"), [{
            "anomaly": "early_stop", "req": 0, "rep": 1,
            "out_tokens": 4,
            "finish": "stop", "ttft_s": p["reps_discarded"][0]["ttft_s"],
            "decode_tok_s": p["reps_discarded"][0]["decode_tok_s"],
            "text_sample": "测" * 4, "reason_sample": "",
            "in_sample": p["reps_discarded"][0]["in_sample"],
            "in_text": p["reps_discarded"][0]["in_text"]}],
            "弃测留痕恰 1 条：脚本 early-stop 批的关键读数与头部样本")
        self.assertTrue(p["reps_discarded"][0]["in_sample"],
                        "弃测留痕带输入侧样本（末条消息头部，ADR-0053）")
        # 弃测留痕带输入全文（in_text）：实际发送消息逐条「【role】文本」渲染，
        # 长度超 200 采样上限（echo 4K 输入含 system 引导 + user 材料 + assistant 预填）
        it = p["reps_discarded"][0]["in_text"]
        self.assertGreater(len(it), 200, "in_text 为输入全文，不设采样上限")
        self.assertTrue(it.startswith("【system】"), f"全文应逐条渲染角色: {it[:80]!r}")
        self.assertIn("【user】", it)
        self.assertIn("【assistant】", it)
        self.assertEqual(p["reps_discarded"][0]["rep"], 1,
                         "弃测轮次号（1-based）应留痕")
        self.assertNotIn("anomaly", p,
                         "重测成功即正常留档（repeats=1 无聚合）——点位不带 anomaly")
        self.assertEqual(p["out_tokens"], 64,
                         "最终点位读数来自重测的正常批（usage 64 tokens）")
        self.assertEqual(p["finish"], "stop")
        self.assertTrue(p["all_ok"])
        # 弃测重测必须弹悬浮岛提醒（2026-09-22 拍板补齐）：此前该 status 未带
        # toast，用户在页面上只看到实时表多出一行 ✗，不知道「正在重测」——
        # 重测是改变结果的动作，必须可见。断言走真实事件流，防回归
        self.assertTrue(any("检测到异常输出" in m and "弃测重测" in m
                            for m in toasts),
                        f"弃测重测应随发 status toast:true；实际 toast={toasts}")
        self.assertTrue(p.get("decode_tok_s"), "重测批 decode 读数正常产出")
        self.assertTrue(any("仅输出 4/64 tokens" in s and "弃测重测" in s
                            for s in statuses),
                        f"SSE 应有弃测重测播报: {[s for s in statuses if '弃测' in s]}")

    async def test_ctx_dev_early_judgement_cancels_batch(self):
        """多并发构建偏离早判（2026-09-16 拍板）：预置错位输入系数（3.6 vs
        mock 1.8）→ 实测 prompt ≈ 2×目标、偏离坐实；conc=3 时首个带 usage
        的请求完成即判定，在途余量取消（SSE 播报含「在途余量已取消」），
        校正重测后点位落在目标档位（不再等全批跑完再全批重测）。"""
        tmp = tempfile.mkdtemp(prefix="llmspeed-e2e-")
        self.addAsyncCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run = BenchRun(self.std_cfg(scenarios=["code"], ctx_list=[4096],
                                    concurrencies=[3], max_tokens=32,
                                    cache="bust"),
                       results_dir=os.path.join(tmp, "results"))
        # 错位系数跳过探测并放大构建偏差（实测 prompt ≈ 2×目标）
        run.cpt_calib[("mock-llm-7b", "code")] = 3.6
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())
        events = await self._drain(q, task)
        points = [e["point"] for e in events if e.get("type") == "point"]
        statuses = [e.get("msg") or "" for e in events if e.get("type") == "status"]
        self.assertEqual(len(points), 1, "4K 单点")
        p = points[0]
        self.assertTrue(p["all_ok"])
        self.assertIsNotNone(p.get("ctx_retry"), "偏离首测实测 prompt 应留痕")
        self.assertAlmostEqual(p["prompt_tokens"], 4096, delta=500,
                               msg="校正重测后应落在目标档位")
        self.assertTrue(any("在途余量已取消" in s for s in statuses),
                        f"并发批偏离应提前判定并取消在途余量: "
                        f"{[s for s in statuses if '偏离' in s]}")

    async def test_flaky_500_recovered_by_retry(self):
        """瞬时失败兜底重试全链路（MOCK_FLAKY_FAIL 一次性钩子）：预置输入系数
        跳过探测后，首发点请求网关 500 → 兜底重试成功——点 all_ok=True、
        reqs 留 retried=1 痕、SSE 播报兜底重试状态。不打桩，走真实 HTTP 链路。"""
        old = mock_server.FLAKY_FAIL
        mock_server.FLAKY_FAIL = 1
        try:
            # 跳过探测：首发 chat 请求即 0K 点请求，恰好吃到 500 钩子
            _, events = await self._run_and_collect(skip_probe=True)
        finally:
            mock_server.FLAKY_FAIL = old
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        p = pts[0]
        self.assertTrue(p["all_ok"], f"兜底重试后点应成功: {p['reqs']}")
        req = p["reqs"][0]
        self.assertIsNone(req["err"])
        self.assertEqual(req.get("retried"), 1, "reqs 应留兜底重试痕")
        self.assertEqual(mock_server.FLAKY_FAIL, 0, "一次性钩子用后归零")
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("兜底重试 1/2" in s and "HTTP 500" in s
                            for s in statuses),
                        f"SSE 应有兜底重试播报: {statuses}")

    async def test_pick_gateway_url_prefers_reachable(self):
        """多地址择优：按配置顺序取第一个可达地址（内网等优选地址排前即被
        选中，绕开隧道中继）；全不可达回退首个；单地址直通。"""
        import server
        mock_url = self.base_url
        dead_url = "http://127.0.0.1:1"     # 端口 1 必然拒绝
        p = {"name": "e2e-pick", "gateway_url": dead_url,
             "gateway_urls": [dead_url, mock_url]}
        url, lat = await server.pick_gateway_url(p)
        self.assertEqual(url, mock_url, "死地址 + 活地址应选活地址")
        self.assertIsNotNone(lat)
        # 顺序互换仍选活地址
        p["gateway_urls"] = [mock_url, dead_url]
        url2, _ = await server.pick_gateway_url(p)
        self.assertEqual(url2, mock_url)
        # 单地址直通不探测
        p["gateway_urls"] = [dead_url]
        url3, lat3 = await server.pick_gateway_url(p)
        self.assertEqual((url3, lat3), (dead_url, None))
        # 全不可达回退首个
        p["gateway_urls"] = [dead_url, "http://127.0.0.1:2"]
        url4, lat4 = await server.pick_gateway_url(p)
        self.assertEqual((url4, lat4), (dead_url, None))

    async def test_thinking_param_never_sent_when_gateway_has_no_thinking(self):
        """disabled 模式 + 无思考网关：thinking 参数从不下发——
        不再产生 LiteLLM 侧的 400 报错（本 Issue 修复点）。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON, mock_server.REJECT_THINKING = 0, "1"
        try:
            statuses = await self._run_and_collect_status()
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej
        self.assertFalse(any("不接受 thinking" in s for s in statuses),
                         f"无思考网关不应收到 thinking 参数: {statuses}")
        self.assertFalse(any("默认输出思考流" in s for s in statuses))

    async def test_thinking_default_on_detected_across_gateway_rejections(self):
        """disabled 模式 + 默认思考模型（探测检出 reasoning）：后续请求下发
        thinking=disabled。

        参数化 REJECT_THINKING 两取值（同旋钮）：网关接受时零 400、网关拒参时
        400 回退后锁定省略——检出提示恒在，「不接受 thinking」提示次数 0 / 1。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON = 3
        try:
            for rej, expect_hits in (("", 0), ("1", 1)):
                with self.subTest(reject_thinking=rej):
                    mock_server.REJECT_THINKING = rej
                    statuses = await self._run_and_collect_status()
                    self.assertTrue(any("默认输出思考流" in s for s in statuses),
                                    f"应检出默认思考流: {statuses}")
                    hits = [s for s in statuses if "不接受 thinking" in s]
                    self.assertEqual(len(hits), expect_hits,
                                     f"400 回退提示次数不符（期望 {expect_hits}）: {hits}")
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej

    async def test_thinking_locked_and_auto_modes(self):
        """进程级能力记忆与 auto 模式的思考协商（两子场景，均 + 网关拒参）。

        - cfg thinking_unsupported=True（模拟第二轮起）：默认思考模型也不再
          试探 disabled 参数——零 400，且提示思考流计入测速；
        - thinking=auto（默认，ADR-0004 重定位）：从不下发 thinking 参数
          （即使网关拒参也零 400），状态消息明示按模型默认形态测速。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON, mock_server.REJECT_THINKING = 3, "1"
        cases = (
            # (子场景, cfg 覆盖, 必含状态帧, 必不含状态帧)
            ("已锁定", {"thinking_unsupported": True}, "思考流将计入测速", None),
            ("auto", {"thinking": "auto"}, "按模型默认形态测速",
             "将下发 thinking=disabled"),
        )
        try:
            for name, cfg_extra, must, must_not in cases:
                with self.subTest(case=name):
                    statuses = await self._run_and_collect_status(**cfg_extra)
                    self.assertFalse(any("不接受 thinking" in s for s in statuses),
                                     f"不应再试探 thinking 参数: {statuses}")
                    self.assertTrue(any(must in s for s in statuses),
                                    f"应明示「{must}」: {statuses}")
                    if must_not is not None:
                        self.assertFalse(any(must_not in s for s in statuses))
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej

    async def test_no_v1_prefix_fallback(self):
        """无前缀 chat 端点（MOCK_NO_V1，DeepSeek 风格）：/v1/chat 404 →
        回退无前缀并锁定，提示恰好一次，点成功。"""
        old = mock_server.NO_V1
        mock_server.NO_V1 = "1"
        try:
            run, events = await self._run_and_collect()
        finally:
            mock_server.NO_V1 = old
        self.assertIn("done", [e.get("type") for e in events])
        hits = [e["msg"] for e in events
                if e.get("type") == "status" and "不带 /v1 前缀" in e["msg"]]
        self.assertEqual(len(hits), 1, f"无前缀回退提示应恰好一次: {hits}")
        self.assertEqual(run.api_prefix, "", "api_prefix 应锁定为无前缀")
        self.assertTrue(run.prefix_locked)
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"])

    async def test_repeats_mean_aggregation(self):
        """repeats=3 全链路：n_reps==3、reps 摘要 3 条（含每次全字段口径）、
        点级 decode_tok_s 为三次复测的均值（ADR-0017 聚合口径）。"""
        _, events = await self._run_and_collect(max_tokens=32, repeats=3)
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertEqual(p["n_reps"], 3)
        self.assertEqual(len(p["reps"]), 3)
        self.assertTrue(all(r["all_ok"] for r in p["reps"]))
        self.assertTrue(all(r["anomaly"] is None for r in p["reps"]),
                        "reps 摘要逐次带 anomaly 键，无异常为 None")
        mean = round(sum(r["decode_tok_s"] for r in p["reps"]) / len(p["reps"]), 1)
        self.assertEqual(p["decode_tok_s"], mean,
                         "点级 decode_tok_s 应取三次复测均值")

    async def test_cpt_calibration_hits_target(self):
        """MOCK_CPT=3.9（与默认估算 1.8 偏差超一倍）：探测校准输入系数后
        第二点 prompt_tokens 命中目标 ±10%，status 出现校准提示。"""
        old = mock_server.CPT
        mock_server.CPT = 3.9
        try:
            _, events = await self._run_and_collect(ctx_list=[4096, 8192])
        finally:
            mock_server.CPT = old
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("校准" in s for s in statuses),
                        f"应出现输入系数校准提示: {statuses}")
        points = {e["point"]["ctx_target"]: e["point"]
                  for e in events if e.get("type") == "point"}
        self.assertEqual(len(points), 2)
        self.assertTrue(all(p["all_ok"] for p in points.values()))
        hit = points[8192]["prompt_tokens"]
        self.assertTrue(8192 * 0.9 <= hit <= 8192 * 1.1,
                        f"校准后 8K 档 prompt_tokens={hit} 应命中目标 ±10%")

    async def test_temperature_rejected_locks_model(self):
        """MOCK_REJECT_TEMPERATURE=1：400 点名 temperature → 省略重试并
        按模型锁定，提示恰好一次，后续请求零 400（ADR-0004）。"""
        old = mock_server.REJECT_TEMPERATURE
        mock_server.REJECT_TEMPERATURE = "1"
        try:
            run, events = await self._run_and_collect()
        finally:
            mock_server.REJECT_TEMPERATURE = old
        hits = [e["msg"] for e in events
                if e.get("type") == "status" and "不接受 temperature" in e["msg"]]
        self.assertEqual(len(hits), 1, f"温度拒参提示应恰好一次: {hits}")
        self.assertIn("mock-llm-7b", run.temperature_locked)
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertTrue(all(p["all_ok"] for p in points))
