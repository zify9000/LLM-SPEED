"""端到端回归：mock 网关（进程内 uvicorn）+ 引擎最小矩阵全链路。

覆盖：cfg 事件首发且脱敏、RTT 净口径与 mock 设定一致（±20%）、双并发总吞吐、
mock 运行不落盘、SSE 注释心跳下 prefill 估值帧不饿死（含 0.95s 节流边界）、
响应头延迟型网关估值帧覆盖等响应头阶段、探测期停止即时生效、无前缀 chat 端点
回退锁定、repeats=3 均值聚合、输入系数校准命中目标、temperature 拒参锁定、
超窗 point_skipped、/models 慢响应下基线期停止、起步突发滑窗上下界、
无正文/裸断连请求级错误、串行 prefill 总吞吐置空、无 usage 估算兜底、
超窗删减上下文重试、agent 连续任务链逐轮出点（前缀缓存命中/无缓存退化）、
输出上限治理（忽略 max_tokens 时断流并改用 max_completion_tokens、新名被 400
拒收时回退防振荡、system 提示长度引导注入）。
运行：python3 -m unittest tests.test_e2e_mock -v
"""
import asyncio
import os
import re
import socket
import threading
import time
import unittest

# mock_server 在 import 时读取环境变量：先设速度，再 import
os.environ.setdefault("MOCK_PP", "3000")
os.environ.setdefault("MOCK_TG", "300")
os.environ.setdefault("MOCK_CPT", "1.8")

import mock_server          # noqa: E402
import uvicorn              # noqa: E402
from bench import BenchRun, SCENARIOS  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_raw_die_server(port: int, n_chunks: int = 5) -> socket.socket:
    """原始 socket HTTP 服务（模拟裸断连网关，绕过 ASGI 的干净收口）：
    GET 返回模型列表（RTT 基线与 /v1 前缀探测用）；POST chat 流出 n_chunks 个
    SSE 块后，chunked 流不发终止块直接 FIN——httpx 判「不完整 chunked 读」。
    新版 uvicorn/starlette 会把 StreamingResponse 生成器异常干净收口（客户端
    视作正常结束），mock 进程内已无法模拟断连，故用原始 socket。"""

    def _handle(conn: socket.socket):
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk
            head, _, rest = data.partition(b"\r\n\r\n")
            if not head.startswith(b"POST"):
                body = b'{"data":[{"id":"raw-llm"}]}'
                conn.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                             b"content-length: %d\r\nx-mock-server: 1\r\n\r\n%s"
                             % (len(body), body))
                return
            m = re.search(rb"content-length:\s*(\d+)", head, re.I)   # 读全请求体再回
            need = int(m.group(1)) if m else 0
            while len(rest) < need:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                rest += chunk
            conn.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                         b"transfer-encoding: chunked\r\nx-mock-server: 1\r\n\r\n")
            frame = 'data: {"choices":[{"delta":{"content":"测"}}]}\n\n'.encode()
            for _ in range(n_chunks):
                conn.sendall(b"%x\r\n%s\r\n" % (len(frame), frame))
                time.sleep(0.01)
            # 裸断连：不发 chunked 终止块/[DONE]/usage，with 收口直接关连接

    def _loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    threading.Thread(target=_loop, daemon=True).start()
    return srv


class TestMockEndToEnd(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.server = uvicorn.Server(uvicorn.Config(
            mock_server.app, host="127.0.0.1", port=cls.port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 10
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.05)
        if not cls.server.started:
            raise RuntimeError("mock uvicorn 启动失败")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)

    async def test_full_pipeline(self):
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "api_key": "sk-should-not-leak",
            "models": ["mock-llm-7b"],
            "scenarios": ["creative"],
            "ctx_list": [0, 4096],
            "concurrencies": [1, 2],
            "max_tokens": 64, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", "cache": "bust",
        }, results_dir="/tmp/llmspeed-e2e")
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())

        events, points = [], []
        try:
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=30)
                if ev is None:
                    break
                events.append(ev)
                if ev.get("type") == "point":
                    points.append(ev["point"])
        finally:
            await task

        types = [e["type"] for e in events]
        self.assertEqual(types[0], "cfg", "cfg 事件必须首发（ADR-0009）")
        self.assertNotIn("api_key", events[0]["cfg"], "cfg 事件必须脱敏")
        self.assertIn("done", types)
        self.assertEqual(len(points), 4, "2 档上下文 × 2 并发 = 4 个点")

        by_key = {(p["ctx_target"], p["concurrency"]): p for p in points}
        self.assertTrue(all(p["all_ok"] for p in points), "全点应成功")

        # prefill 净口径 ≈ MOCK_PP（mock 端到端无网络噪声，RTT 已扣除）。
        # ctx=0 档 TTFT 仅 ~17ms，uvicorn/解析的固定开销占比大，只做合理性区间
        # 断言；4K 档 TTFT ~1.4s，固定开销可忽略，±15% 即足够敏感
        for p in points:
            if p["ctx_target"] == 0:
                self.assertTrue(1000 <= p["prefill_net_tok_s"] <= 6000,
                                msg=f"ctx=0 净 prefill {p['prefill_net_tok_s']} 越出合理区间")
            else:
                self.assertAlmostEqual(p["prefill_net_tok_s"], 3000, delta=450,
                                       msg=f"ctx={p['ctx_target']} 净 prefill 偏离 mock 设定")
            self.assertAlmostEqual(p["decode_tok_s"], 300, delta=75,
                                   msg=f"ctx={p['ctx_target']} decode 偏离 mock 设定")
            # 无停滞侧：空窗校正口径不得偏离毛值（DECODE_GAP_CAP 只削病理拖尾）
            adj = p.get("decode_tok_s_adj")
            if adj is not None:
                self.assertAlmostEqual(adj, p["decode_tok_s"], delta=1.0,
                                       msg=f"ctx={p['ctx_target']} 无停滞点校正值偏离毛值")

        # 双并发总吞吐 ≈ 2 × 单请求 decode
        s1 = by_key[(4096, 1)]["decode_tok_s"]
        t2 = by_key[(4096, 2)]["decode_total_tok_s"]
        self.assertIsNotNone(t2, "decode 重叠充分时必须产出总吞吐")
        self.assertTrue(1.5 * s1 <= t2 <= 2.6 * s1, f"总吞吐 {t2} 应 ≈ 2×{s1}")

        # mock 运行不落历史记录（ADR-0010）
        self.assertTrue(run.mock_seen)
        self.assertFalse(os.path.isdir("/tmp/llmspeed-e2e"))

    async def test_prefill_estimates_survive_keepalive(self):
        """SSE 注释心跳不饿死 prefill 估值帧：网关以 0.05s 间隔发 keepalive
        注释时 1s 读行超时分支永远轮不到，估值必须按节流持续发出
        （前端「最新 Prefill 速度」KPI 实时刷新依赖这些帧）。"""
        old_ka, old_pp = mock_server.KA, mock_server.PP
        mock_server.KA, mock_server.PP = 0.05, 600.0   # 高频心跳；4K prefill ≈ 6.8s
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1],
                "max_tokens": 8, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测，直奔主题
            run.prefill_curve[("mock-llm-7b", "creative")] = [(2048, 600.0)]   # 探测正常时会播种曲线
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            ticks = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=25)
                    if ev is None:
                        break
                    if ev.get("type") == "tick" and ev.get("phase") == "prefill":
                        ticks.append(ev)
            finally:
                await task
        finally:
            mock_server.KA, mock_server.PP = old_ka, old_pp

        est = [t for t in ticks if t.get("ttft") is None and t.get("speed")]
        measured = [t for t in ticks if t.get("ttft") is not None]
        self.assertGreaterEqual(len(est), 3,
                                f"心跳注释下 prefill 估值帧被饿死（仅 {len(est)} 帧）")
        self.assertEqual(len(measured), 1, "首 token 实测帧应恰好一帧")
        # 先验速率封顶：估值在先验实测速率带内（PP=600），不再从 est_prompt/1s
        # （≈4096，数量级虚高）双曲起步
        for t in est:
            self.assertTrue(300 <= t["speed"] <= 900,
                            f"估值 {t['speed']} 越出先验速率带 [300,900]")
        # est_tick 0.95s 节流边界回归：相邻估值帧不得隔拍成 ~2s 一帧
        est_ts = [t["ts"] for t in est]
        for a, b in zip(est_ts, est_ts[1:]):
            self.assertLess(b - a, 1.8, f"估值帧间隔 {b - a:.2f}s 超出 0.95s 节流边界")

    async def test_prefill_estimates_cover_header_wait(self):
        """响应头延迟型网关（LiteLLM→vLLM 链路实测：prefill 期间连响应头都
        不发）：估值帧必须覆盖等响应头阶段——若只在读行循环里发，整个
        prefill 一帧都没有，前端 KPI 冻结到首 token。"""
        old_hdr, old_pp = mock_server.DELAY_HDR, mock_server.PP
        mock_server.DELAY_HDR, mock_server.PP = True, 600.0   # 4K prefill ≈ 6.8s 全在等响应头
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1],
                "max_tokens": 8, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测，直奔主题
            run.prefill_curve[("mock-llm-7b", "creative")] = [(2048, 600.0)]   # 探测正常时会播种曲线
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            ticks = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=25)
                    if ev is None:
                        break
                    if ev.get("type") == "tick" and ev.get("phase") == "prefill":
                        ticks.append(ev)
            finally:
                await task
        finally:
            mock_server.DELAY_HDR, mock_server.PP = old_hdr, old_pp

        est = [t for t in ticks if t.get("ttft") is None and t.get("speed")]
        measured = [t for t in ticks if t.get("ttft") is not None]
        self.assertGreaterEqual(len(est), 3,
                                f"等响应头阶段估值帧缺失（仅 {len(est)} 帧）")
        self.assertEqual(len(measured), 1, "首 token 实测帧应恰好一帧")
        # 先验速率封顶：估值在先验实测速率带内（PP=600），数量级虚高不再出现
        for t in est:
            self.assertTrue(300 <= t["speed"] <= 900,
                            f"估值 {t['speed']} 越出先验速率带 [300,900]")

    async def test_max_gap_detects_decode_stall(self):
        """decode 中段停滞被 max_gap_s 捕获：传输/调度空窗会拖尾 decode 窗口、
        读数偏慢不可信，点级需要显式标记（实测隧道拥塞曾放大窗口 14 倍）。"""
        old_stall = mock_server.STALL
        mock_server.STALL = 4.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 32, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            points = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=30)
                    if ev is None:
                        break
                    if ev.get("type") == "point":
                        points.append(ev["point"])
            finally:
                await task
        finally:
            mock_server.STALL = old_stall

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
        dispatch 而非真实流式，速率虚高——点级必须带 decode_burst 标记。"""
        old_tg = mock_server.TG
        mock_server.TG = 200000.0   # 5µs/token：事件循环开销下平均间隔仍 ≪1ms
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 32, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            points = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=30)
                    if ev is None:
                        break
                    if ev.get("type") == "point":
                        points.append(ev["point"])
            finally:
                await task
        finally:
            mock_server.TG = old_tg

        self.assertEqual(len(points), 1)
        self.assertTrue(points[0].get("decode_burst"),
                        "亚毫秒间隔交付必须标记 decode_burst")

    async def test_agent_scenario_smoke(self):
        """Agent 场景端到端（ADR-0014/0015）：SWE-agent 轨迹语料构造 prompt，
        连续任务链逐轮出点（4 轮 = 1 冷启动 + 2 短 + 1 长，长轮增量 8K）。"""
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "models": ["mock-llm-7b"], "scenarios": ["agent"],
            "ctx_list": [0, 4096], "concurrencies": [1],   # ctx_list 对 agent 不适用
            "agent_turns": 3, "agent_turn_delta": 2048,
            "agent_cold_ctx": 1024, "agent_phase2_base": 8192,
            "max_tokens": 32, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled",
        })
        run.cpt_calib[("mock-llm-7b", "agent")] = 3.5   # 跳过探测
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())
        points = []
        try:
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=30)
                if ev is None:
                    break
                if ev.get("type") == "point":
                    points.append(ev["point"])
        finally:
            await task
        self.assertEqual(len(points), 4, "4 轮链出 4 个点（与 ctx_list 无关）")
        self.assertTrue(all(p["all_ok"] for p in points))
        self.assertEqual([p["turn"] for p in points], [0, 1, 2, 3])
        # 轮次构成：冷启动 1024 + 短轮增量 2048 × 2 + 长轮 ladder 首档 8192
        self.assertEqual([p["ctx_target"] for p in points],
                         [1024, 3072, 5120, 13312])
        self.assertEqual([p["turn_phase"] for p in points], [0, 1, 1, 2])
        last = points[-1]
        self.assertGreater(last["prompt_tokens"], 6000, "末轮应注入足量轨迹语料")

    async def test_pick_gateway_url_prefers_reachable(self):
        """多地址择优：按配置顺序取第一个可达地址（内网等优选地址排前即被
        选中，绕开隧道中继）；全不可达回退首个；单地址直通。"""
        import server
        mock_url = f"http://127.0.0.1:{self.port}"
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

    async def _run_and_collect(self, **cfg_extra):
        """跑一次最小矩阵（probe 必走），返回 (run, 全部事件列表)。"""
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "models": ["mock-llm-7b"], "scenarios": ["creative"],
            "ctx_list": [0], "concurrencies": [1],
            "max_tokens": 16, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", **cfg_extra,
        })
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
        return run, events

    async def _run_and_collect_status(self, thinking="disabled", **cfg_extra):
        """跑一次最小矩阵（probe 必走，检测逻辑全靠它），返回状态消息列表。"""
        _, events = await self._run_and_collect(thinking=thinking, **cfg_extra)
        types = [e.get("type") for e in events]
        self.assertIn("done", types)
        return [e["msg"] for e in events if e.get("type") == "status"]

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

    async def test_thinking_default_on_detected_and_disabled(self):
        """disabled 模式 + 默认思考模型（探测检出 reasoning）：
        后续请求下发 thinking=disabled（网关接受时无 400）。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON, mock_server.REJECT_THINKING = 3, ""
        try:
            statuses = await self._run_and_collect_status()
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej
        self.assertTrue(any("默认输出思考流" in s for s in statuses),
                        f"应检出默认思考流: {statuses}")
        self.assertFalse(any("不接受 thinking" in s for s in statuses))

    async def test_thinking_rejected_falls_back_once(self):
        """disabled 模式 + 默认思考模型 + 网关拒参：下发 disabled 被 400 →
        锁定后省略，报错提示恰好出现一次。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON, mock_server.REJECT_THINKING = 3, "1"
        try:
            statuses = await self._run_and_collect_status()
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej
        self.assertTrue(any("默认输出思考流" in s for s in statuses))
        hits = [s for s in statuses if "不接受 thinking" in s]
        self.assertEqual(len(hits), 1, f"400 回退提示应恰好一次: {hits}")

    async def test_thinking_seeded_lock_skips_param(self):
        """进程级能力记忆下发（cfg thinking_unsupported=True，模拟第二轮起）：
        默认思考模型也不再试探 disabled 参数——零 400，且提示思考流计入测速。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON, mock_server.REJECT_THINKING = 3, "1"
        try:
            statuses = await self._run_and_collect_status(thinking_unsupported=True)
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej
        self.assertFalse(any("不接受 thinking" in s for s in statuses),
                         f"已锁定的网关不应再收到 thinking 参数: {statuses}")
        self.assertTrue(any("思考流将计入测速" in s for s in statuses),
                        f"应明示思考流计入测速: {statuses}")

    async def test_thinking_auto_follows_model_default(self):
        """auto 模式（默认，ADR-0004 重定位）+ 默认思考模型：从不下发 thinking
        参数（即使网关拒参也零 400），状态消息明示按模型默认形态测速。"""
        old_r, old_rej = mock_server.REASON, mock_server.REJECT_THINKING
        mock_server.REASON, mock_server.REJECT_THINKING = 3, "1"
        try:
            statuses = await self._run_and_collect_status(thinking="auto")
        finally:
            mock_server.REASON, mock_server.REJECT_THINKING = old_r, old_rej
        self.assertTrue(any("按模型默认形态测速" in s for s in statuses),
                        f"auto 模式应明示按默认形态测速: {statuses}")
        self.assertFalse(any("不接受 thinking" in s for s in statuses),
                         f"auto 模式不应下发 thinking 参数: {statuses}")
        self.assertFalse(any("将下发 thinking=disabled" in s for s in statuses))

    async def test_stop_during_probe(self):
        """探测期停止即时生效：probe 不走 _run_point 任务集，曾经无法中断
        （点停止要等整个探测跑完，慢网关上可达分钟级）。"""
        old_pp = mock_server.PP
        mock_server.PP = 200.0   # 探测 2048 ctx → prefill ~10s，足够停在探测中段
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1],
                "max_tokens": 8, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            task = asyncio.create_task(run.run())
            await asyncio.sleep(1.0)   # RTT 基线（毫秒级）后已进入探测 prefill
            t = time.monotonic()
            run.stop()
            await asyncio.wait_for(task, timeout=15)
            self.assertLess(time.monotonic() - t, 5,
                            "探测期停止未即时生效（曾需等整个探测完成）")
        finally:
            mock_server.PP = old_pp
        self.assertIn("stopped", [e["type"] for e in run.history])
        self.assertNotIn("done", [e["type"] for e in run.history])

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

    async def test_agent_chain_turns_with_cache(self):
        """agent 连续任务链（MOCK_CACHE=1 前缀缓存）：不以下文档位为变量，
        4 轮（1 冷启动 + 短轮固定增量 1024 × 2 + 长轮 ladder 8192）逐轮出点。
        暖轮命中上一轮全文（hit>0）；短文本暖轮 TTFT 不随上下文增长
        （≈ 冷轮），长文本轮 TTFT 随自身增量放大；增量 prefill 口径
        全程 ≈ mock PP 设定。"""
        old_cache = mock_server.CACHE
        mock_server.CACHE = "1"
        mock_server._SEEN.clear()
        try:
            run, events = await self._run_and_collect(
                scenarios=["agent"], ctx_list=[65536],   # ctx_list 对 agent 不适用
                concurrencies=[1], agent_turns=3, agent_turn_delta=[1024, 1024],
                agent_cold_ctx=1024, agent_phase2_base=8192,
                max_tokens=32)
        finally:
            mock_server.CACHE = old_cache
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "4 轮链应出 4 个点（与 ctx_list 无关）")
        self.assertEqual([p["turn"] for p in pts], [0, 1, 2, 3])
        self.assertEqual([p["ctx_target"] for p in pts], [1024, 2048, 3072, 11264])
        self.assertEqual([p["turn_delta"] for p in pts], [1024, 1024, 1024, 8192])
        self.assertEqual([p["turn_phase"] for p in pts], [0, 1, 1, 2])
        self.assertTrue(all(p["all_ok"] for p in pts))
        cold, warm = pts[0], pts[1:]
        # 冷轮仅命中与探测请求共享的系统提示级前缀（~百 tokens 内），
        # 不得命中上一轮正文（首轮之前没有正文）
        self.assertLess(cold["cache_hit_tokens"], 150, "冷轮只应命中系统提示级共享前缀")
        for p in warm:
            self.assertTrue(p["cache_reported"], "mock 回传命中字段，非估算口径")
            self.assertGreater(p["cache_hit_tokens"], 700,
                               f"轮{p['turn']} 应命中前轮大部分上下文")
            # 增量 prefill 口径：未命中 tokens ÷ TTFT ≈ MOCK_PP=3000
            self.assertAlmostEqual(p["prefill_tok_s"], 3000, delta=900,
                                   msg=f"轮{p['turn']} 增量 prefill 偏离 mock 设定")
        # 短文本暖轮 TTFT 不随上下文增长：轮1 prompt 2K 与轮0 冷启动 1K 耗时同档
        self.assertLess(pts[1]["ttft_s"], cold["ttft_s"] * 2 + 0.3)
        # 长文本轮 TTFT 随自身增量放大：轮3 未命中 8K+ vs 轮1 未命中 1K+
        self.assertGreater(pts[3]["ttft_s"], pts[1]["ttft_s"] * 2)
        # 每轮实测速率登记进先验曲线：除探测播种点外应有链轮次的实测点
        curve = run.prefill_curve.get(("mock-llm-7b", "agent"), [])
        self.assertGreaterEqual(len(curve), 5, f"探测 + 4 轮链应登记 ≥5 个先验点: {curve}")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_chain_without_cache_degrades_honestly(self):
        """无前缀缓存能力（MOCK_CACHE 关）：暖轮 TTFT 随全量 prompt 增长——
        缓存迹象判别为「无缓存」：命中列 None（前端显示未回传，不展示假设
        命中数），prefill 退回全量口径实测真值（≈ mock PP），不按假设命中折算。"""
        mock_server._SEEN.clear()
        _, events = await self._run_and_collect(
            scenarios=["agent"], concurrencies=[1],
            agent_turns=3, agent_turn_delta=[1024, 1024],
            agent_cold_ctx=1024, agent_phase2_base=8192,
            max_tokens=32)
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4)
        cold, warm3 = pts[0], pts[3]
        self.assertFalse(warm3["cache_reported"], "无缓存字段回传时必须标记非真值")
        self.assertIsNone(warm3["cache_hit_tokens"],
                          "无缓存迹象时命中为 None（前端显示未回传）")
        # 轮3 全量 11264 tokens 重算：TTFT ≈ 11× 冷轮（轮0=1024）
        self.assertGreater(warm3["ttft_s"], cold["ttft_s"] * 4.0)
        # 全量口径：prompt_tokens ÷ TTFT ≈ MOCK_PP=3000（真实可测速率）
        self.assertAlmostEqual(warm3["prefill_tok_s"], 3000, delta=450)

    async def test_agent_chain_cache_assumed_when_ttft_flat(self):
        """缓存生效但网关剥离命中字段（MOCK_CACHE_NOREPORT）：短文本暖轮 TTFT
        相对冷轮走平 → 判别「有缓存迹象」，命中按链内差分估算（非 None）、
        cache_reported=False（前端加 ≈），prefill 为增量口径 ≈ mock PP。
        长文本轮增量占比过半、缓存收益在 TTFT 上不可分辨 → 宁缺毋假显示未回传。"""
        old = mock_server.CACHE_NOREPORT
        mock_server.CACHE_NOREPORT = "1"
        mock_server._SEEN.clear()
        try:
            _, events = await self._run_and_collect(
                scenarios=["agent"], concurrencies=[1],
                agent_turns=3, agent_turn_delta=[1024, 1024],
                agent_cold_ctx=1024, agent_phase2_base=8192,
                max_tokens=32)
        finally:
            mock_server.CACHE_NOREPORT = old
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4)
        cold, w2, w4 = pts[0], pts[1], pts[3]
        # 轮1（短文本，增量 1K ≪ 总 2K）：TTFT 走平 → 差分估算
        self.assertFalse(w2["cache_reported"])
        self.assertIsNotNone(w2["cache_hit_tokens"], "有缓存迹象应按差分估算命中")
        self.assertGreater(w2["cache_hit_tokens"], 700, "轮1 估算命中应≈前轮全文")
        self.assertAlmostEqual(w2["prefill_tok_s"], 3000, delta=900,
                               msg="轮1 增量 prefill 偏离 mock 设定")
        self.assertLess(w2["ttft_s"], cold["ttft_s"] * 2 + 0.3)
        # 轮3（长文本，增量 8K ≈ 总量 73%）：缓存收益不可分辨 → 未回传
        self.assertIsNone(w4["cache_hit_tokens"],
                          "长文本轮缓存收益不可分辨，不应展示假设命中")
        self.assertFalse(w4["cache_reported"])

    async def test_agent_chain_ctx_limit_early_end(self):
        """agent 链轮目标超部署上限：链提前结束，point_skipped 带 msg 说明原因
        （前端常驻展示——此前只有瞬态 status 行，跳过原因会被后续状态吞没）。"""
        mock_server._SEEN.clear()
        _, events = await self._run_and_collect(
            scenarios=["agent"], concurrencies=[1],
            agent_turns=3, agent_turn_delta=[1024, 1024],
            agent_cold_ctx=1024, agent_phase2_base=8192,
            model_max_ctx={"mock-llm-7b": 8192}, max_tokens=32)
        pts = [e["point"] for e in events if e.get("type") == "point"]
        # 轮 3 目标 11264 > 上限 8192−32−1024=7136：链于轮 2 后结束
        self.assertEqual([p["turn"] for p in pts], [0, 1, 2])
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(skipped), 1, f"末轮应恰好一条 point_skipped: {skipped}")
        self.assertEqual(skipped[0]["count"], 1)
        self.assertIn("链提前结束", skipped[0]["msg"],
                      "point_skipped 须带 msg 供前端常驻展示")
        self.assertIn("done", [e["type"] for e in events])

    async def test_stop_during_models_delay_baseline(self):
        """MOCK_MODELS_DELAY=6（/models 慢响应）下 RTT 基线期停止：
        _interruptible 0.2s 轮询取消在途 GET，不等整个慢响应。"""
        old = mock_server.MODELS_DELAY
        mock_server.MODELS_DELAY = 6.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 8, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            task = asyncio.create_task(run.run())
            await asyncio.sleep(0.8)   # 首个 GET /models 在途（响应需 6s）
            t = time.monotonic()
            run.stop()
            await asyncio.wait_for(task, timeout=15)
            self.assertLess(time.monotonic() - t, 4,
                            "基线期停止未即时生效（等在途 /models 慢响应）")
        finally:
            mock_server.MODELS_DELAY = old
        self.assertIn("stopped", [e["type"] for e in run.history])
        self.assertNotIn("done", [e["type"] for e in run.history])

    async def test_burst_decode_window_speed(self):
        """MOCK_BURST=15 起步突发 + 低速 TG=20：滑窗差分把突发留在基准
        样本——起步 tick 不虚高（≤TG×1.5），后续收敛 TG±25%（ADR-0007）。"""
        old_burst, old_tg = mock_server.BURST, mock_server.TG
        mock_server.BURST, mock_server.TG = 15, 20.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 128, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            # 跳过探测直奔滑窗主题；预置 chunk_calib=1.0（事件即 token）——
            # 探测自身也带突发，其事件/token 比会把 tick 估算口径带偏
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8
            run.chunk_calib[("mock-llm-7b", "creative")] = 1.0
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            ticks = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=30)
                    if ev is None:
                        break
                    if ev.get("type") == "tick" and ev.get("phase") == "decode":
                        ticks.append(ev)
            finally:
                await task
        finally:
            mock_server.BURST, mock_server.TG = old_burst, old_tg
        self.assertGreaterEqual(len(ticks), 5, f"低速 decode 应有多个 tick: {len(ticks)}")
        self.assertLessEqual(ticks[0]["speed"], 20 * 1.5,
                             f"起步 tick {ticks[0]['speed']} 虚高（突发摊入读数）")
        for t in ticks[1:]:
            self.assertTrue(20 * 0.75 <= t["speed"] <= 20 * 1.25,
                            f"decode tick {t['speed']} 未收敛 TG±25%")

    async def test_no_content_point_error(self):
        """MOCK_NO_CONTENT=1（只思考无正文/零输出）：请求 err 含
        「未收到任何输出 token」、点级 all_ok=False。"""
        old = mock_server.NO_CONTENT
        mock_server.NO_CONTENT = "1"
        try:
            _, events = await self._run_and_collect()
        finally:
            mock_server.NO_CONTENT = old
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertFalse(p["all_ok"], "无正文点必须降级为失败")
        self.assertIn("未收到任何输出 token", p["reqs"][0]["err"])

    async def test_die_at_request_error(self):
        """裸断连（原始 socket 服务：流出 5 个 SSE 块后 chunked 流不发终止块
        直接 FIN）→ 请求级 err、点级 all_ok=False（模拟服务端/链路中途崩断）。
        MOCK_DIE_AT 的生成器抛错在新版 uvicorn/starlette 下会被干净收口
        （客户端视作正常结束），已无法模拟真断连，改用原始 socket 模拟。"""
        port = _free_port()
        srv = _start_raw_die_server(port)
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{port}",
                "models": ["raw-llm"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 32, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
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
            srv.close()
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertFalse(p["all_ok"], "裸断连点必须降级为失败")
        err = p["reqs"][0]["err"]
        self.assertTrue(err, "裸断连应产生请求级 err")
        self.assertNotIn("未收到任何输出 token", err,
                         "已流出 5 个 token，不应走零输出诊断")

    async def test_serialize_prefill_no_decode_overlap(self):
        """MOCK_SERIALIZE=1：并发 prefill 全局串行 → conc=2 的 decode 区间
        不重叠，decode_total_tok_s 置空（ADR-0006 门槛），点本身成功。"""
        old_s, old_pp = mock_server.SERIALIZE, mock_server.PP
        mock_server.SERIALIZE, mock_server.PP = "1", 1000.0   # 4K prefill ≈ 4.1s/请求
        try:
            _, events = await self._run_and_collect(
                ctx_list=[4096], concurrencies=[2], max_tokens=16)
        finally:
            mock_server.SERIALIZE, mock_server.PP = old_s, old_pp
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertTrue(p["all_ok"])
        self.assertAlmostEqual(p["decode_tok_s"], 300, delta=75)
        self.assertIsNone(p["decode_total_tok_s"],
                          "串行 prefill 下 decode 不重叠，总吞吐必须置空（ADR-0006）")
        reqs = p["reqs"]   # 引擎已按 req 序号排序
        self.assertGreater(reqs[1]["ttft_s"] - reqs[0]["ttft_s"], 3.0,
                           "第二个请求的 TTFT 应等出一个完整 prefill")

    async def test_no_usage_estimation_fallback(self):
        """MOCK_NO_USAGE=1：流末无 usage 帧 → usage_real=False，终值按
        字符/输出系数估算兜底，chunk_calib 不播种（校准依赖真实 usage）。"""
        old = mock_server.NO_USAGE
        mock_server.NO_USAGE = "1"
        try:
            run, events = await self._run_and_collect()
        finally:
            mock_server.NO_USAGE = old
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertTrue(p["all_ok"], "无 usage 不是错误，估算兜底应照常出点")
        r = p["reqs"][0]
        self.assertFalse(r["usage_real"])
        # 兜底口径：16 字符正文 ÷ 创意场景 out_cpt 1.5 ≈ 11（≠ 真实 16）
        self.assertEqual(r["out_tokens"], round(16 / 1.5))
        self.assertEqual(run.chunk_calib, {}, "无 usage 时 chunk_calib 不得播种")

    async def test_max_tokens_ignored_flip_and_cut(self):
        """MOCK_IGNORE_MAX_TOKENS=1（opencode zen 行为：忽略旧名、认新名）：
        探测请求（cap 64）输出跑飞 4× → 客户端断流兜底并判定改用
        max_completion_tokens；后续正式点被服务端按新名精确截断（ADR-0016）。"""
        old = mock_server.IGNORE_MT
        mock_server.IGNORE_MT = "1"
        try:
            run, events = await self._run_and_collect(max_tokens=16)
        finally:
            mock_server.IGNORE_MT = old
        self.assertEqual(run.mt_param.get("mock-llm-7b"), "max_completion_tokens",
                         "探测断流后应判定改用新参数名")
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("改用 max_completion_tokens" in s for s in statuses),
                        f"应有参数名翻转提示: {statuses}")
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"], f"翻转后点应成功: {points[0].get('reqs')}")
        self.assertEqual(points[0]["reqs"][0]["out_tokens"], 16,
                         "新名生效后服务端应精确截断在 max_tokens")

    async def test_mct_rejected_falls_back_no_oscillation(self):
        """IGNORE_MT + REJECT_MCT（两种参数名都不生效的极端网关）：探测断流
        翻转新名 → 正式请求 400 → 回退旧名并锁定（mt_mct_rejected）→ 之后只
        断流兜底不再翻转（防振荡），点仍成功、finish=client_cut（ADR-0016）。"""
        old_i, old_r = mock_server.IGNORE_MT, mock_server.REJECT_MCT
        mock_server.IGNORE_MT, mock_server.REJECT_MCT = "1", "1"
        try:
            run, events = await self._run_and_collect(max_tokens=16)
        finally:
            mock_server.IGNORE_MT, mock_server.REJECT_MCT = old_i, old_r
        self.assertEqual(run.mt_param.get("mock-llm-7b"), "max_tokens",
                         "新名被 400 拒收后应回退旧名")
        self.assertIn("mock-llm-7b", run.mt_mct_rejected)
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("已回退 max_tokens" in s for s in statuses),
                        f"应有新名 400 回退提示: {statuses}")
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertTrue(points[0]["all_ok"], f"断流兜底下点应成功: {points[0].get('reqs')}")
        self.assertEqual(points[0]["reqs"][0]["finish"], "client_cut",
                         "两种参数名都失效时应以客户端断流收口")
        # 断流把输出收在 2× 上限附近（估算口径），而非跑飞 4×（64）
        self.assertLessEqual(points[0]["reqs"][0]["out_tokens"], 16 * 2 + 8)

    async def test_out_hint_injected_into_system_prompt(self):
        """ADR-0016 防线 2：system 提示注入输出字数上限（max_tokens × out_cpt
        先验 × 1.2）——零输入档 sent_chars 应恰好多出引导文案长度。"""
        _, events = await self._run_and_collect(max_tokens=16)
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        sc = SCENARIOS["creative"]
        hint = "\n\n" + sc["out_hint"].format(limit=int(16 * sc["out_cpt"] * 1.2))
        expect = len(sc["system"]) + len(hint) + len(sc["zero_instruction"])
        self.assertEqual(points[0]["reqs"][0]["sent_chars"], expect,
                         "system 提示应恰好追加输出长度引导文案")


if __name__ == "__main__":
    unittest.main()
