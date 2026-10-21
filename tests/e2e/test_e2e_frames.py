"""e2e 帧/读数判据组（进程内 mock 网关 + 引擎最小矩阵全链路）。

覆盖：SSE 注释心跳下 prefill 估值帧不饿死（含 0.95s 节流边界）、响应头延迟型
网关估值帧覆盖等响应头阶段、decode 中段停滞被 max_gap_s 捕获（复合记账与空窗
校正口径）、起步突发滑窗判据（亚毫秒冲刷标记；chunk 数不足时由「每 chunk token
数」增补判据标记少而肥冲刷）、真实投机解码周期交付不误伤（周期交付不误标突发/
速率收敛、高接受率 cobs 走新校准钳制播种）、平均每交付批 token 数
（tok_per_chunk：ACCEPT=4 时 ≈4、ACCEPT=1 基线无成批证据为 None）、停滞空窗随
滑窗滑出基准恢复读数。

自 tests/test_e2e_mock.py 拆分（原文件已删）。

运行（任一方式，都要求**从仓库根目录**执行，无需 pytest 也可跑）：
    .venv/bin/python -m pytest tests/e2e/test_e2e_frames.py -q
    .venv/bin/python -m unittest tests.e2e.test_e2e_frames -v
    .venv/bin/python -m unittest discover -s tests/e2e -p "test_e2e_frames.py"
"""
import asyncio

# 顺序是硬约束：先 import tests._shared（_harness 顶部已触发 MOCK_* 播种），
# 再 import mock_server（后者在 import 时读环境变量）。
from tests.e2e._harness import MockGatewayTest      # noqa: E402

import bench                                        # noqa: E402
import mock_server                                  # noqa: E402
from bench import BenchRun                          # noqa: E402


class TestE2EFrames(MockGatewayTest):

    async def test_prefill_estimates_not_starved(self):
        """prefill 估值帧不得饿死/冻结（两类网关姿态各一子例）：

        - SSE 注释心跳型（0.05s 间隔发 keepalive 注释）：1s 读行超时分支永远
          轮不到，估值必须按节流持续发出（前端「最新 Prefill 速度」KPI 实时
          刷新依赖这些帧）；
        - 响应头延迟型（LiteLLM→vLLM 链路实测：prefill 期间连响应头都不发）：
          估值帧必须覆盖等响应头阶段——若只在读行循环里发，整个 prefill 一帧
          都没有，前端 KPI 冻结到首 token。"""
        cases = (
            # (子场景, mock 旋钮, 旋钮值, 缺帧报错语, 是否复核 0.95s 节流边界)
            ("心跳注释", "KA", 0.05, "心跳注释下 prefill 估值帧被饿死", True),
            ("等响应头", "DELAY_HDR", True, "等响应头阶段估值帧缺失", False),
        )
        for name, knob, value, miss_msg, check_throttle in cases:
            with self.subTest(case=name):
                old_knob, old_pp = getattr(mock_server, knob), mock_server.PP
                setattr(mock_server, knob, value)
                mock_server.PP = 600.0   # 4K prefill ≈ 6.8s
                try:
                    run = BenchRun(self.std_cfg(ctx_list=[4096], max_tokens=8))
                    run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测，直奔主题
                    # 探测正常时会播种曲线。两个子例故意用不同覆盖度的曲线，顺带
                    # 验 weak 标记的两个方向：只有低档锚点（比本档低整一个倍频程）
                    # → 带 weak；有 4000 的近档锚点 → 不带（2026-10-17 真机单点复测
                    # 256K 全程 1100+ 就是前一种处境，见 ADR-0134）
                    run.prefill_curve[("mock-llm-7b", "creative")] = (
                        [(2048, 600.0)] if check_throttle
                        else [(2048, 600.0), (4000, 600.0)])
                    q: asyncio.Queue = asyncio.Queue()
                    run.subs.add(q)
                    task = asyncio.create_task(run.run())
                    events = await self._drain(q, task, timeout=25)
                finally:
                    setattr(mock_server, knob, old_knob)
                    mock_server.PP = old_pp

                ticks = [e for e in events
                         if e.get("type") == "tick" and e.get("phase") == "prefill"]
                est = [t for t in ticks if t.get("ttft") is None and t.get("speed")]
                # 带 ttft 的帧分两档：首 token 帧分子仍是估算（带 est），
                # 流末帧分子才是 usage 真值（不带 est，前端据此上「实测」牌）
                first_tok = [t for t in ticks if t.get("ttft") is not None and t.get("est")]
                measured = [t for t in ticks
                            if t.get("ttft") is not None and not t.get("est")]
                self.assertGreaterEqual(len(est), 3, f"{miss_msg}（仅 {len(est)} 帧）")
                self.assertEqual(len(first_tok), 1, "首 token 估值帧应恰好一帧")
                self.assertEqual(len(measured), 1, "流末权威 prefill 帧应恰好一帧")
                # 先验速率封顶：估值在先验实测速率带内（PP=600），不再从
                # est_prompt/1s（≈4096，数量级虚高）双曲起步
                for t in est:
                    self.assertTrue(300 <= t["speed"] <= 900,
                                    f"估值 {t['speed']} 越出先验速率带 [300,900]")
                # 先验覆盖度上到线上（前端据此说明「按低档速率保守显示，实测通常更快」）
                for t in est:
                    self.assertEqual(bool(t.get("weak")), check_throttle,
                                     f"weak 标记错（先验覆盖度判据失效）: {t}")
                if check_throttle:
                    # est_tick 0.95s 节流边界回归：相邻估值帧不得隔拍成 ~2s 一帧
                    est_ts = [t["ts"] for t in est]
                    for a, b in zip(est_ts, est_ts[1:]):
                        self.assertLess(b - a, 1.8,
                                        f"估值帧间隔 {b - a:.2f}s 超出 0.95s 节流边界")

    async def test_prefill_frames_split_estimate_from_measured(self):
        """首 token 帧只能算「估」，usage 到手的流末帧才是「实测」（真机事故回归）。

        2026-10-17 真机 256K 档：实发 19.7 万 token，而首 token 帧的分子按
        「字符 ÷ 全局平均系数」估成 22~26 万 → KPI 显示 4200~5000（真值 3722），
        且该帧被前端当实测冻结，直到整点（3 轮 ≈3 分钟）跑完才回写。
        本例故意把 cpt_calib 播成偏小的 0.6（mock 真值 1.8）使估算分子成倍
        虚高，断言：①首 token 帧带 est 且明显高于真值（=它确实是估的）；
        ②流末补一帧不带 est 的权威 prefill，读数与存档 usage 口径闭合。
        cache=stable 是为旁路构建偏离校正重测（仅 bust 生效），否则首测会被
        整批替换、两帧分属两次尝试。"""
        run = BenchRun(self.std_cfg(ctx_list=[4096], max_tokens=8,
                                    cache="stable", repeats=1))
        run.cpt_calib[("mock-llm-7b", "creative")] = 0.6   # 故意偏小三倍
        run.prefill_curve[("mock-llm-7b", "creative")] = [(2048, 600.0)]
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())
        events = await self._drain(q, task, timeout=25)
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        point = pts[0]
        self.assertTrue(point["all_ok"], f"点应成功: {point.get('reqs')}")
        frames = [e for e in events if e.get("type") == "tick"
                  and e.get("phase") == "prefill" and e.get("ttft") is not None]
        first = [t for t in frames if t.get("est")]
        final = [t for t in frames if not t.get("est")]
        self.assertEqual(len(first), 1, f"首 token 估值帧应恰好一帧: {frames}")
        self.assertEqual(len(final), 1, f"流末权威 prefill 帧应恰好一帧: {frames}")
        truth = point["prefill_tok_s"]
        # 估算分子虚高 ≈ 真系数/坏系数 = 1.8/0.6 = 3×（TTFT 两侧同值）
        self.assertGreater(first[0]["speed"], 1.5 * truth,
                           f"构造偏 3× 时首 token 帧应显著虚高："
                           f"{first[0]['speed']} vs 真值 {truth}")
        # 权威帧用 usage 真值：与点级毛/净两口径之一闭合（RTT 基线在
        # 进程内 mock 上只有毫秒级，两者几平相等）
        self.assertTrue(any(abs(final[0]["speed"] - c) / c <= 0.05
                            for c in (point["prefill_tok_s"],
                                      point["prefill_net_tok_s"]) if c),
                        f"权威帧 {final[0]['speed']} 与点级 usage 口径不闭合："
                        f"{point['prefill_tok_s']} / {point['prefill_net_tok_s']}")

    async def test_max_gap_detects_decode_stall(self):
        """decode 中段停滞被 max_gap_s 捕获：传输/调度空窗会拖尾 decode 窗口、
        读数偏慢不可信，点级需要显式标记（实测隧道拥塞曾放大窗口 14 倍）。"""
        old_stall = mock_server.STALL
        mock_server.STALL = 4.0
        try:
            run = BenchRun(self.std_cfg(max_tokens=32))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = await self._drain(q, task)
        finally:
            mock_server.STALL = old_stall
        points = [e["point"] for e in events if e.get("type") == "point"]

        self.assertEqual(len(points), 1)
        gap = points[0].get("max_gap_s")
        self.assertIsNotNone(gap, "停滞点必须带 max_gap_s")
        self.assertGreaterEqual(gap, 3.5, f"4s 停滞应被捕获（实测 {gap}）")
        self.assertLessEqual(gap, 6.0, f"空窗估值不应远超注入值（实测 {gap}）")
        # 停滞复合记账：单次 4s 停滞满额计入累计、次数为 1
        self.assertGreaterEqual(points[0].get("stall_s") or 0, 3.5,
                                "4s 停滞应满额计入 stall_s")
        self.assertEqual(points[0].get("stall_count"), 1, "单次停滞计数应为 1")
        self.assertFalse(points[0].get("decode_burst"),
                         "正常速率流式不应被误判为突发交付")
        # 空窗校正口径：校正窗口把 4s 停滞封顶到 3s，校正值应显著高于毛值
        raw, adj = points[0].get("decode_tok_s"), points[0].get("decode_tok_s_adj")
        self.assertIsNotNone(adj, "停滞点必须带 decode_tok_s_adj")
        self.assertGreater(adj, raw * 1.2, f"校正值 {adj} 应显著高于毛值 {raw}")

    async def test_decode_burst_flagged(self):
        """缓冲冲刷式交付（全部 chunk 亚毫秒间隔到达）：decode 窗口测的是网关
        dispatch 而非真实流式，速率虚高——点级必须带 decode_burst 标记。

        两形态子例：

        - ACCEPT=1：逐 token 事件全量冲刷（MOCK_TG 拉到 5µs/token，事件循环
          开销下平均间隔仍 ≪1ms）；
        - ACCEPT=16：少而肥的 chunk（网关缓冲合并 + 投机解码并存），64 token
          装进 4 个 16 token 事件、亚毫秒冲刷——chunk 数过不了
          BURST_MIN_CHUNKS=5，必须由「每 chunk token 数」增补判据标记突发。

        冲刷线在此临时放宽到 50ms（与组内同类用例同口径）：真 HTTP 链路上一次
        进程调度抖动就会把"均值 <1ms"整体抬起（本用例曾在多任务并行的机器上偶发
        失败），拿生产阈值断言到达间隔属环境抖动高发区。1ms 边界与两条判据分支由
        TestDecodeBurst 纯函数单测覆盖，本用例只验「冲刷形态 → 标记」的全链路接通。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        old_gap = bench.BURST_AVG_GAP_MS
        bench.BURST_AVG_GAP_MS = 50.0
        cases = (
            # (子例, MOCK_ACCEPT（None=不覆盖）, max_tokens, 标记缺位报错语,
            #  是否复核 usage 读数)
            ("逐 token 全量冲刷", None, 32,
             "亚毫秒间隔交付必须标记 decode_burst", False),
            ("少而肥 chunk", 16, 64,
             "4 个 16 token 事件的亚毫秒冲刷必须标记 decode_burst", True),
        )
        try:
            for name, accept, max_tokens, burst_msg, check_usage in cases:
                with self.subTest(case=name):
                    mock_server.ACCEPT = old_accept if accept is None else accept
                    mock_server.TG = 200000.0   # 5µs/token：事件循环开销下平均间隔仍 ≪1ms
                    run = BenchRun(self.std_cfg(max_tokens=max_tokens))
                    run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
                    q: asyncio.Queue = asyncio.Queue()
                    run.subs.add(q)
                    task = asyncio.create_task(run.run())
                    events = await self._drain(q, task)
                    points = [e["point"] for e in events if e.get("type") == "point"]
                    self.assertEqual(len(points), 1)
                    if check_usage:
                        self.assertTrue(points[0]["all_ok"])
                        self.assertEqual(points[0]["out_tokens"], 64,
                                         "out_tokens 应采信 usage 的 completion_tokens")
                    self.assertTrue(points[0].get("decode_burst"), burst_msg)
        finally:
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg
            bench.BURST_AVG_GAP_MS = old_gap

    async def test_accept_periodic_delivery_not_burst(self):
        """真实投机解码周期交付（MOCK_ACCEPT>1）不误标突发、速率仍 ≈ tg——
        接受率两子例：

        - ACCEPT=4（每事件 4 token、间隔 k/tg=40ms）：40ms 平均间隔远高于
          BURST_AVG_GAP_MS=1ms 冲刷线，不得误标 decode_burst；40ms 也远低于
          0.5s 停滞闸——亚阈值的周期性停顿不得记入 stall_s/max_gap_s；
        - ACCEPT=8：cobs = 事件数/完成数 = 0.125，低于旧校准钳制下限 0.2（会静默
          丢弃观测）、落在新下限 0.05 之上——校准必须播种并播报「SSE 事件计数口径」
          状态帧；最终 out_tokens 仍采信 usage 不受校准影响，80ms 平均事件间隔
          不得误标突发。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG

        async def one_point(accept):
            mock_server.ACCEPT, mock_server.TG = accept, 100.0
            run = BenchRun(self.std_cfg(max_tokens=64))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            return await self._drain(q, task)

        cases = (
            # (MOCK_ACCEPT, 突发标记反证语, decode 速率报错前缀, 校准播种状态帧)
            (4, "40ms 事件间隔的周期交付不得误标突发",
             "有效 decode 速率应 ≈ MOCK_TG", None),
            (8, "80ms 事件间隔不得误标突发",
             "decode 应 ≈ MOCK_TG", "SSE 事件计数口径"),
        )
        try:
            for accept, burst_msg, rate_msg, seed_needle in cases:
                with self.subTest(accept=accept):
                    events = await one_point(accept)
                    if seed_needle is not None:
                        statuses = [e["msg"] for e in events
                                    if e.get("type") == "status"]
                        self.assertTrue(any(seed_needle in s for s in statuses),
                                        f"cobs=0.125 应被新钳制收下并播种校准: {statuses}")
                    points = [e["point"] for e in events if e.get("type") == "point"]
                    self.assertEqual(len(points), 1)
                    self.assertTrue(points[0]["all_ok"])
                    self.assertEqual(points[0]["out_tokens"], 64,
                                     "out_tokens 应采信 usage 的 completion_tokens")
                    self.assertFalse(points[0].get("decode_burst"), burst_msg)
                    self.assertAlmostEqual(points[0]["decode_tok_s"], 100, delta=30,
                                           msg=f"{rate_msg}（实测 "
                                               f"{points[0]['decode_tok_s']}）")
                    if accept == 4:
                        self.assertIsNone(points[0].get("stall_s"),
                                          "40ms << 0.5s 停滞闸，不得记入 stall_s")
                        self.assertIsNone(points[0].get("stall_count"))
                        mg = points[0].get("max_gap_s")
                        self.assertTrue(mg is None or mg < 0.5,
                                        f"亚阈值空窗不得触停滞诊断（max_gap_s={mg}）")
        finally:
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg

    async def test_tok_per_chunk_accept4_and_baseline(self):
        """平均每交付批 token 数（tok_per_chunk）端到端：MOCK_ACCEPT=4 投机周期
        交付，max_tokens=64 → usage 64 token / 16 事件 = 4；ACCEPT=1 基线
        每 token 一事件（无成批证据，≈1 噪声口径）→ None。"""
        old_accept = mock_server.ACCEPT

        async def one_point():
            run = BenchRun(self.std_cfg(max_tokens=64))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = await self._drain(q, task)
            points = [e["point"] for e in events if e.get("type") == "point"]
            return points[0]

        try:
            mock_server.ACCEPT = 4
            p4 = await one_point()
            mock_server.ACCEPT = 1
            p1 = await one_point()
        finally:
            mock_server.ACCEPT = old_accept

        self.assertTrue(p4["all_ok"])
        self.assertEqual(p4["out_tokens"], 64,
                         "out_tokens 应采信 usage 的 completion_tokens")
        self.assertAlmostEqual(
            p4["tok_per_chunk"], 4.0, delta=0.5,
            msg=f"每事件 4 token → tok_per_chunk ≈ 4（实测 "
                f"{p4.get('tok_per_chunk')}）")
        # 新口径：≈1 是逐 token 交付的记账噪声，不入档（SPEC_MIN_TPC 1.15 以下
        # 且无间隔双峰证据 → None），不再落 1.0
        self.assertIsNone(
            p1["tok_per_chunk"],
            msg=f"每 token 一事件（无成批交付证据）→ tok_per_chunk 应为 None"
                f"（实测 {p1.get('tok_per_chunk')}）")

    async def test_stall_recovers_after_window_slide(self):
        """滑窗差分基准陈旧回归：4s 停滞的空窗必须随 3s 窗口滑出——恢复后第
        2~3 个 tick 起实时读数回到 tg ±50%，而不是带着停滞前的旧基准塌陷
        整段停滞期+3s。旧实现基准样本永不被弹出，恢复后读数长期 ≈ 真值×
        3/(3+停滞时长)。"""
        old_stall, old_tg = mock_server.STALL, mock_server.TG
        # 同 test_burst_decode_window_speed：max_tokens=240 → 240 字符单字符
        # 重复正文命中退化重复判据会触发弃测重测，两批 tick 混流污染滑窗断言；
        # 本测试主题是停滞空窗随滑窗滑出，屏蔽检测
        old_det = bench._detect_rep_anomaly
        mock_server.STALL, mock_server.TG = 4.0, 100.0
        bench._detect_rep_anomaly = lambda *a, **k: None
        # 停滞锚在剩余 token 中点：前后各 120 token @100 tok/s ≈ 1.2s，
        # 停滞后 ≥1s 的恢复窗口出 2~3 个 tick
        try:
            run = BenchRun(self.std_cfg(max_tokens=240))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = await self._drain(q, task)
        finally:
            mock_server.STALL, mock_server.TG = old_stall, old_tg
            bench._detect_rep_anomaly = old_det
        ticks = [e for e in events
                 if (e.get("type") == "tick" and e.get("phase") == "decode"
                     and e.get("ttft") is not None)]

        # 末帧是流末权威读数（整窗均值口径，停滞未被校正、必然偏低），不在
        # 滑窗回归断言范围内
        stream_ticks = ticks[:-1]
        ts = [t["ts"] for t in stream_ticks]
        stall_idx = next((i for i in range(len(ts) - 1) if ts[i + 1] - ts[i] > 2.0),
                         None)
        self.assertIsNotNone(stall_idx, f"应能从 tick 序列定位 4s 停滞: {ts}")
        post = stream_ticks[stall_idx + 1:]
        self.assertGreaterEqual(len(post), 2,
                                f"停滞恢复窗口应至少出 2 个 tick: {ts}")
        # 第 1 个恢复 tick 起步兜底（整程均值）允许偏低；第 2 个起窗口内只含
        # 恢复后样本，读数必须回到 tg ±50%
        for t in post[1:]:
            self.assertTrue(50 <= t["speed"] <= 150,
                            f"恢复 tick 读数 {t['speed']} 应在 100±50% 内"
                            f"（全序列 {[x['speed'] for x in stream_ticks]}）")

    async def test_delivery_grain_marks_unmeasured_decode_ticks_as_est(self):
        """ADR-0145：本轮还没实测过「几个 token 算一个包」时，decode tick 的分子是估算
        ——必须带 est/est_batch（前端据此挂「≈」、不上「实测 Decode」牌），跑完落定的
        权威帧不得带 est。这两件以前在存档里没有任何证据，用户看到的就是"实测值偶尔
        偏 20%"。

        mock 运行不读不写交付粒度台账（假网关的交付形态是编的），所以走的是 cold
        （1 事件/包 直计）那一路：ACCEPT=4 时它把 token 数估成实际的 1/4 →
        out_est_gap ≈ -0.75、req 带 calib_src="cold"；同轮第二个请求已经实测过
        → calib_src="run"、中途帧不再带 est。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        mock_server.ACCEPT, mock_server.TG = 4, 40.0   # 64 token ≈ 1.6s：出多个中段 tick

        async def collect(**cfg_extra):
            run = BenchRun(self.std_cfg(max_tokens=64, **cfg_extra))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            try:
                return await self._drain(q, task)
            finally:
                await task

        try:
            cold_events = await collect()
            run_events = await collect(repeats=2)
        finally:
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg

        def ticks(events):
            return [e for e in events if e.get("type") == "tick"
                    and e.get("phase") == "decode" and e.get("speed") is not None]

        t1 = ticks(cold_events)
        self.assertGreaterEqual(len(t1), 2, f"应有多个 decode tick: {t1}")
        # 末帧是流末权威读数（usage 真值）：不入「中途帧」断言范围
        mid = t1[:-1]
        self.assertTrue(mid, "应有中途 tick（非流末权威帧）")
        self.assertTrue(all(t.get("est") for t in mid),
                        f"本轮未实测过事件/token 比的中途帧必须标 est: {mid}")
        self.assertTrue(all(t.get("est_batch") == 1.0 for t in mid),
                        "未校准时的交付粒度就是 1 个 token/包")
        self.assertFalse(t1[-1].get("est"), "流末权威帧（usage 真值）不该标 est")

        t2 = ticks(run_events)
        rep1 = [t for t in t2 if "|rep1|" in t["tag"]][:-1]
        self.assertTrue(rep1, f"复测第二轮应有中途帧: {[t['tag'] for t in t2]}")
        self.assertTrue(all(not t.get("est") for t in rep1),
                        "本轮已实测过事件/token 比后，中途帧不该再是估值帧")

        r1 = [r for p in (e["point"] for e in cold_events if e.get("type") == "point")
              for r in (p.get("reqs") or []) if not r.get("err")]
        self.assertTrue(r1, "应有成功请求明细")
        self.assertEqual(r1[0].get("calib_src"), "cold",
                         f"交付粒度来源要看得出从哪来: {r1[0].get('calib_src')}")
        # ACCEPT=4：每包 4 个 token，按 1 事件/包直计就是真值的 1/4 → -0.75
        self.assertAlmostEqual(r1[0].get("out_est_gap"), -0.75, delta=0.15,
                               msg="未校准直计对投机网关应低估 k 倍（ADR-0024）")
        reps2 = [rp for p in (e["point"] for e in run_events if e.get("type") == "point")
                 for rp in (p.get("reps") or [])]
        # 聚合行只取代表样本的 reqs（哪一轮被取不定），所以「这一轮从种子起步、
        # 后来改用本轮实测」这条证据必须活在「复测摘要」里
        srcs2 = {rp.get("calib_src") for rp in reps2 if rp.get("calib_src")}
        self.assertEqual(srcs2, {"cold", "run"},
                         f"复测摘要要留住了来源标记: {srcs2}")
        gaps = [rp.get("out_est_gap") for rp in reps2
                if rp.get("out_est_gap") is not None]
        self.assertTrue(gaps, "复测摘要要带上估算对真值的差距")
        self.assertLess(min(abs(g) for g in gaps), 0.15,
                        f"本轮实测过之后应几乎无偏差: {gaps}")


    async def test_probe_calibration_does_not_turn_src_run(self):
        """探测请求的校准不得把交付粒度来源提前翻成 "run"（ADR-0145 边界）。

        探测只有 ~13-64 个包，压不过 256 权重的历史种子；若由它转正，本轮第一个
        正式请求的中途读数仍八成是种子却被当实测（不挂 ≈），断流判定也失去
        保守口径。本例**不预置 cpt_calib**（探测会真跑）：正式点 rep0 的中途帧
        必须仍带 est（来源 cold），rep1 才转正。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        mock_server.ACCEPT, mock_server.TG = 4, 40.0   # 64 token ≈ 1.6s：出多个中段 tick
        try:
            run = BenchRun(self.std_cfg(ctx_list=[4096], max_tokens=64, repeats=2))
            # 不预置 cpt_calib —— 探测请求真跑（本例前提）
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            try:
                events = await self._drain(q, task, timeout=25)
            finally:
                await task
        finally:
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg

        ticks = [e for e in events if e.get("type") == "tick"
                 and e.get("phase") == "decode" and e.get("speed") is not None
                 and "|4096|" in e.get("tag", "")]
        rep0 = [t for t in ticks if "|rep0|" in t["tag"]][:-1]   # 末帧是权威帧
        rep1 = [t for t in ticks if "|rep1|" in t["tag"]][:-1]
        self.assertTrue(rep0, f"正式点 rep0 应有中途帧: {[t['tag'] for t in ticks]}")
        self.assertTrue(all(t.get("est") for t in rep0),
                        f"探测转正后 rep0 中途帧丢了 est 标: {rep0}")
        self.assertTrue(all(t.get("est_src") == "cold" for t in rep0),
                        f"mock 无台账，来源应为 cold: {rep0}")
        self.assertTrue(all(abs((t.get("est_batch") or 0) - 4.0) < 0.6 for t in rep0),
                        f"est_batch 应是探测实测的 4 token/包: {rep0}")
        self.assertTrue(rep1, f"复测第二轮应有中途帧: {[t['tag'] for t in ticks]}")
        self.assertTrue(all(not t.get("est") for t in rep1),
                        f"正式请求校准后（rep1）才转正: {rep1}")
        reps = [rp for p in (e["point"] for e in events if e.get("type") == "point")
                for rp in (p.get("reps") or [])]
        srcs = {rp.get("calib_src") for rp in reps if rp.get("calib_src")}
        self.assertEqual(srcs, {"cold", "run"},
                         f"rep0 应记 cold、rep1 才记 run: {srcs}")

    async def test_final_frames_marked_est_when_usage_missing(self):
        """usage 未回传时，流末帧的分子是估算兜底值（est_prompt/est_chunks）：
        prefill/decode 两帧都必须带 est——不带会被前端当「实测」冻结 KPI，
        usage 缺失（或 client_cut）的网关会一路骗到点收口。"""
        old = mock_server.NO_USAGE
        mock_server.NO_USAGE = "1"
        try:
            run = BenchRun(self.std_cfg(ctx_list=[4096], max_tokens=32))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            try:
                events = await self._drain(q, task, timeout=25)
            finally:
                await task
        finally:
            mock_server.NO_USAGE = old

        reqs = [r for p in (e["point"] for e in events if e.get("type") == "point")
                for r in (p.get("reqs") or []) if not r.get("err")]
        self.assertTrue(reqs, "应有成功请求明细（usage 缺失走估算兜底）")
        self.assertFalse(reqs[0].get("usage_real"), "前提：本例全程无 usage")
        ticks = [e for e in events if e.get("type") == "tick"]
        pf_final = [t for t in ticks if t.get("phase") == "prefill"
                    and t.get("ttft") is not None]
        self.assertTrue(pf_final, "应有带 ttft 的 prefill 帧")
        self.assertTrue(all(t.get("est") for t in pf_final),
                        f"usage 未回时所有 prefill 帧（含流末帧）都必须带 est: {pf_final}")
        dec = [t for t in ticks if t.get("phase") == "decode"
               and t.get("speed") is not None]
        self.assertTrue(dec, "应有 decode 帧")
        self.assertTrue(all(t.get("est") for t in dec),
                        f"usage 未回时所有 decode 帧（含流末帧）都必须带 est: {dec}")
        self.assertTrue(dec[-1].get("est_final"),
                        f"流末估值帧要带 est_final（前端文案区分「不会有真值帧覆盖」）: {dec[-1]}")
