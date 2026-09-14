"""端到端回归：mock 网关（进程内 uvicorn）+ 引擎最小矩阵全链路。

覆盖：cfg 事件首发且脱敏、RTT 净口径与 mock 设定一致（±20%）、双并发总吞吐、
mock 运行不落盘、SSE 注释心跳下 prefill 估值帧不饿死（含 0.95s 节流边界）、
响应头延迟型网关估值帧覆盖等响应头阶段、探测期停止即时生效、无前缀 chat 端点
回退锁定、repeats=3 均值聚合、输入系数校准命中目标、temperature 拒参锁定、
超窗 point_skipped、超窗贴边裁减（ctx_edge 留痕）、/models 慢响应下基线期停止、起步突发滑窗上下界、
无正文/裸断连请求级错误、串行 prefill 总吞吐置空、无 usage 估算兜底（SSE 事件数直计，英文形态输出不虚发）、
超窗删减上下文重试、软超窗（fastllm 系 200+"prompt too long"占位回复）判错（content/reasoning 双通道）、
输出塌缩守卫（未知占位文案判 empty_reply_suspected）、物理速率闸（假速率判错且不污染先验曲线/记账校准）、
agent 缓存×指令矩阵逐组合出点（前缀缓存命中回传/无缓存退化/非回传网关
TTFT 走平判别估算/缓存命中但读取慢时走平失效改由全量预期·佐证通道判别）、预算跳档（C+I 超部署预算整档 point_skipped）、超窗矩阵提前收口，
存档输出采样（text_sample/reason_sample 头部 ≤200 字符截获：超长截断、不足全留）、
repeats=1 透明度提醒、MOCK_TG_PATTERN 逐请求交替 decode 速度、
瞬时失败兜底重试全链路（MOCK_FLAKY_FAIL 一次性钩子：首发点请求 500 →
兜底重试成功、reqs 留 retried 痕、SSE 播报兜底重试状态）、
输出上限治理（忽略 max_tokens 时断流并改用 max_completion_tokens、新名被 400
拒收时回退防振荡、system 提示长度引导注入）、投机解码周期交付（MOCK_ACCEPT
不误标突发/速率收敛、高接受率 cobs 走新校准钳制播种、停滞空窗随滑窗滑出基准
恢复读数）、平均每交付批 token 数（tok_per_chunk：ACCEPT=4 时 ≈4、ACCEPT=1 基线
无成批证据为 None）、少而肥 chunk 冲刷（2~4 个多 token 事件亚毫秒到达判突发、
真实投机周期交付不误伤）、异常输出弃测重测全链路（MOCK_EARLY_STOP
一次性钩子：首测 early-stop 弃测、重测正常出点、reps_discarded 留痕带
rep 轮次号与 in_text 输入全文；正常 req 存档无 _in_full/in_text 瞬态字段；
reps 摘要逐次带 anomaly）；
agent 矩阵 MOCK_EARLY_STOP_LONG 逐发钩子：首测异常重测正常/持续异常
anomaly 留档（肇事 req 带 anomaly+in_text）、物理速率闸下限（未命中量 <8K 只按绝对上限判，小 prompt
诚实点不误杀）。
运行：python3 -m unittest tests.test_e2e_mock -v
"""
import asyncio
import os
import re
import shutil
import socket
import tempfile
import threading
import time
import unittest

# mock_server 在 import 时读取环境变量：进程内强制覆盖，开发者 shell 里残留的
# MOCK_* 导出不得污染断言口径
os.environ["MOCK_PP"] = "3000"
os.environ["MOCK_TG"] = "300"
os.environ["MOCK_CPT"] = "1.8"

import mock_server          # noqa: E402
import bench                # noqa: E402
import uvicorn              # noqa: E402
from bench import CTX_HEADROOM, BenchRun, SCENARIOS  # noqa: E402


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
        tmp = tempfile.mkdtemp(prefix="llmspeed-e2e-")
        self.addAsyncCleanup(shutil.rmtree, tmp, ignore_errors=True)
        results_dir = os.path.join(tmp, "results")   # 不存在的子路径，断言 mock 运行不落盘
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "api_key": "sk-should-not-leak",
            "models": ["mock-llm-7b"],
            "scenarios": ["creative"],
            "ctx_list": [0, 4096],
            "concurrencies": [1, 2],
            "max_tokens": 64, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", "cache": "bust",
        }, results_dir=results_dir)
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

        # 正常 req 存档不带输入全文：瞬态 _in_full 已在点定稿前剥离，
        # 异常取证键 in_text 也只在异常相关记录出现
        for p in points:
            for r in p["reqs"]:
                self.assertNotIn("_in_full", r, "正常 req 不得带 _in_full")
                self.assertNotIn("in_text", r, "正常 req 不得带 in_text")

        # 回复模式（ADR-0021）默认自由回答：消息序列 [system, user]，无预填
        self.assertEqual(mock_server.LAST_CHAT_ROLES, ["system", "user"])

        # mock 运行不落历史记录（ADR-0010）
        self.assertTrue(run.mock_seen)
        self.assertFalse(os.path.isdir(results_dir), "mock 运行不得创建 results 目录")
        self.assertFalse(os.listdir(tmp), "mock 运行不得向 results_dir 写入任何文件")

    async def test_reply_mode_echo(self):
        """回复模式=模板续写（ADR-0021）：创意/代码场景请求末尾带 assistant
        预填消息（mock 记录角色序列），测速照常出点；零输入档不受影响。"""
        tmp = tempfile.mkdtemp(prefix="llmspeed-e2e-")
        self.addAsyncCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "api_key": "sk-should-not-leak",
            "models": ["mock-llm-7b"],
            "scenarios": ["creative"],
            "ctx_list": [0, 4096],
            "concurrencies": [1],
            "max_tokens": 64, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", "cache": "bust",
            "reply_mode": "echo",
        }, results_dir=os.path.join(tmp, "results"))
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
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "api_key": "sk-should-not-leak",
            "models": ["mock-llm-7b"],
            "scenarios": ["creative"],
            "ctx_list": [4096],
            "concurrencies": [1],
            "max_tokens": 64, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", "cache": "bust",
            "reply_mode": "echo",
        }, results_dir=os.path.join(tmp, "results"))
        run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测：钩子留给出点请求
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        old_early = mock_server.EARLY_STOP
        mock_server.EARLY_STOP = 4   # 首发点请求照常 prefill 后只吐 4 token 即 stop
        task = asyncio.create_task(run.run())
        points, statuses = [], []
        try:
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=30)
                if ev is None:
                    break
                if ev.get("type") == "point":
                    points.append(ev["point"])
                elif ev.get("type") == "status":
                    statuses.append(ev.get("msg") or "")
        finally:
            await task
            mock_server.EARLY_STOP = old_early
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
        self.assertTrue(p.get("decode_tok_s"), "重测批 decode 读数正常产出")
        self.assertTrue(any("仅输出 4/64 tokens" in s and "弃测重测" in s
                            for s in statuses),
                        f"SSE 应有弃测重测播报: {[s for s in statuses if '弃测' in s]}")

    async def test_flaky_500_recovered_by_retry(self):
        """瞬时失败兜底重试全链路（MOCK_FLAKY_FAIL 一次性钩子）：预置输入系数
        跳过探测后，首发点请求网关 500 → 兜底重试成功——点 all_ok=True、
        reqs 留 retried=1 痕、SSE 播报兜底重试状态。不打桩，走真实 HTTP 链路。"""
        old = mock_server.FLAKY_FAIL
        mock_server.FLAKY_FAIL = 1
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 16, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8
            # 跳过探测：首发 chat 请求即 0K 点请求，恰好吃到 500 钩子
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

    async def test_decode_burst_few_fat_chunks(self):
        """少而肥的 chunk（MOCK_ACCEPT=16，网关缓冲合并 + 投机解码并存）：
        64 token 装进 4 个 16 token 事件、亚毫秒冲刷——chunk 数过不了
        BURST_MIN_CHUNKS=5，必须由「每 chunk token 数」增补判据标记突发。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        mock_server.ACCEPT, mock_server.TG = 16, 200000.0   # 事件间隔 16/200000 = 0.08ms
        # 冲刷线临时放宽到 50ms：真 HTTP 链路下客户端逐事件处理开销本身 ~1ms，
        # 拿生产阈值 1ms 断言到达间隔是环境抖动高发区（曾在低负载下确定性失败）；
        # 1ms 边界与少而肥分支的判定逻辑由 TestDecodeBurst 纯函数单测覆盖，
        # 本用例只验「冲刷形态 → 标记」的全链路接通
        old_gap = bench.BURST_AVG_GAP_MS
        bench.BURST_AVG_GAP_MS = 50.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 64, "repeats": 1, "timeout_s": 30,
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
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg
            bench.BURST_AVG_GAP_MS = old_gap

        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"])
        self.assertEqual(points[0]["out_tokens"], 64,
                         "out_tokens 应采信 usage 的 completion_tokens")
        self.assertTrue(points[0].get("decode_burst"),
                        "4 个 16 token 事件的亚毫秒冲刷必须标记 decode_burst")

    async def test_accept16_slow_delivery_not_burst(self):
        """真实投机形态（MOCK_ACCEPT=16 + 低 tg）：4 个 16 token 事件、间隔
        160ms 的周期交付——满足「少而肥」的 chunk 数与 token 数条件，但平均
        间隔远高于 1ms 冲刷线，不得误标 decode_burst。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        mock_server.ACCEPT, mock_server.TG = 16, 100.0   # 事件间隔 16/100 = 160ms
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 64, "repeats": 1, "timeout_s": 30,
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
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg

        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"])
        self.assertEqual(points[0]["out_tokens"], 64)
        self.assertFalse(points[0].get("decode_burst"),
                         "160ms 事件间隔的真实投机交付不得误标突发")
        # 有效 decode 速率仍 ≈ tg：64 token ÷ (64−16)/100 s = 133，±40% 容差
        # 兜 4 个事件下的调度抖动
        self.assertAlmostEqual(points[0]["decode_tok_s"], 133, delta=53,
                               msg=f"decode 应 ≈ MOCK_TG 投机口径（实测 "
                                   f"{points[0]['decode_tok_s']}）")

    async def test_accept_periodic_delivery_not_burst(self):
        """MOCK_ACCEPT=4（投机解码周期交付：每事件 4 token、间隔 k/tg=40ms）：
        有效 decode 速率仍 ≈ tg，不得误标 decode_burst（40ms 平均间隔远高于
        BURST_AVG_GAP_MS=1ms 冲刷线），40ms 也远低于 0.5s 停滞闸——亚阈值的
        周期性停顿不得记入 stall_s/max_gap_s。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        mock_server.ACCEPT, mock_server.TG = 4, 100.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 64, "repeats": 1, "timeout_s": 30,
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
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg

        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"])
        self.assertEqual(points[0]["out_tokens"], 64,
                         "out_tokens 应采信 usage 的 completion_tokens")
        self.assertFalse(points[0].get("decode_burst"),
                         "40ms 事件间隔的周期交付不得误标突发")
        # 容差 ±30%：64 token ÷ (64−4)/100 s ≈ 107，叠加调度抖动留足余量
        self.assertAlmostEqual(points[0]["decode_tok_s"], 100, delta=30,
                               msg=f"有效 decode 速率应 ≈ MOCK_TG（实测 "
                                   f"{points[0]['decode_tok_s']}）")
        self.assertIsNone(points[0].get("stall_s"),
                          "40ms << 0.5s 停滞闸，不得记入 stall_s")
        self.assertIsNone(points[0].get("stall_count"))
        mg = points[0].get("max_gap_s")
        self.assertTrue(mg is None or mg < 0.5,
                        f"亚阈值空窗不得触停滞诊断（max_gap_s={mg}）")

    async def test_tok_per_chunk_accept4_and_baseline(self):
        """平均每交付批 token 数（tok_per_chunk）端到端：MOCK_ACCEPT=4 投机周期
        交付，max_tokens=64 → usage 64 token / 16 事件 = 4；ACCEPT=1 基线
        每 token 一事件（无成批证据，≈1 噪声口径）→ None。"""
        old_accept = mock_server.ACCEPT

        async def one_point():
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 64, "repeats": 1, "timeout_s": 30,
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

    async def test_accept8_high_ratio_calibration(self):
        """MOCK_ACCEPT=8：cobs = 事件数/完成数 = 0.125，低于旧校准钳制下限
        0.2（会静默丢弃观测）、落在新下限 0.05 之上——校准必须播种并播报
        「SSE 事件计数口径」状态帧；最终 out_tokens 仍采信 usage 不受校准影响，
        80ms 平均事件间隔不得误标突发。"""
        old_accept, old_tg = mock_server.ACCEPT, mock_server.TG
        mock_server.ACCEPT, mock_server.TG = 8, 100.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 64, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
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
        finally:
            mock_server.ACCEPT, mock_server.TG = old_accept, old_tg

        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("SSE 事件计数口径" in s for s in statuses),
                        f"cobs=0.125 应被新钳制收下并播种校准: {statuses}")
        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"])
        self.assertEqual(points[0]["out_tokens"], 64,
                         "最终口径走 usage，校准不影响 out_tokens")
        # 64 ÷ (64−8)/100 ≈ 114 tok/s，±30% 容差兜调度抖动
        self.assertAlmostEqual(points[0]["decode_tok_s"], 100, delta=30,
                               msg=f"decode 应 ≈ MOCK_TG（实测 "
                                   f"{points[0]['decode_tok_s']}）")
        self.assertFalse(points[0].get("decode_burst"),
                         "80ms 事件间隔不得误标突发")

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
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                # 停滞锚在剩余 token 中点：前后各 120 token @100 tok/s ≈ 1.2s，
                # 停滞后 ≥1s 的恢复窗口出 2~3 个 tick
                "max_tokens": 240, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            ticks = []
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=30)
                    if ev is None:
                        break
                    if (ev.get("type") == "tick" and ev.get("phase") == "decode"
                            and ev.get("ttft") is not None):
                        ticks.append(ev)
            finally:
                await task
        finally:
            mock_server.STALL, mock_server.TG = old_stall, old_tg
            bench._detect_rep_anomaly = old_det

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

    async def _run_agent_matrix_collect(self, cpt=1.8, preset=None, **cfg_extra):
        """跑一次 agent 缓存×指令矩阵（预置输入系数跳过探测），返回
        (run, 全部事件列表)。cpt=None 不预置（走探测路径）；preset 为启动前
        对 run 对象的额外播种钩子（如毒药化 prefill 先验曲线）。"""
        run = BenchRun({
            "gateway_url": f"http://127.0.0.1:{self.port}",
            "models": ["mock-llm-7b"], "scenarios": ["agent"],
            "ctx_list": [], "concurrencies": [1],
            "max_tokens": 32, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", **cfg_extra,
        })
        if cpt is not None:
            # 预置输入系数对齐 mock 分词（MOCK_CPT=1.8），跳过探测；
            # 传错位值可放大指令长度偏差（偏离校正用例）
            run.cpt_calib[("mock-llm-7b", "agent")] = cpt
        if preset:
            preset(run)
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

    async def test_agent_matrix_smoke(self):
        """Agent 矩阵端到端（ADR-0042）：缓存×指令阶梯逐组合出点，点位带
        ctx_target=缓存档 / inst_tokens=指令档 / inst_real_tokens（链内差分），
        无旧链的 turn 系字段。"""
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[0, 4096], agent_inst_ladder=[128, 256])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 缓存档 × 2 指令档 = 4 点（预热不计点）")
        self.assertEqual([(p["ctx_target"], p["inst_tokens"]) for p in pts],
                         [(0, 128), (0, 256), (4096, 128), (4096, 256)],
                         "缓存档升序外层 × 指令档升序内层")
        for p in pts:
            self.assertTrue(p["all_ok"])
            self.assertNotIn("turn", p)
            self.assertNotIn("turn_phase", p)
            self.assertNotIn("turn_delta", p)
            self.assertIsNotNone(p["total_s"], "端到端总时长应产出")
            self.assertIn("inst_real_tokens", p, "指令真实长度字段应在位")
        self.assertLess(pts[0]["prompt_tokens"], 500, "零缓存档为纯指令小 prompt")
        self.assertGreater(pts[2]["prompt_tokens"], 3500, "缓存档注入足量轨迹语料")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_prime_ticks_visible(self):
        """预热/基线 quiet 请求发实时 tick（带 prime 标记、不入 KPI 语义）+
        批终 done 帧：大缓存档预热 prefill 长达数十秒，全程无帧会让前端实时表
        与「进行中请求」空转（实时组件"未加载"错觉的来源）"""
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[4096], agent_inst_ladder=[128])
        prime_ticks = [e for e in events
                       if e.get("type") == "tick" and e.get("prime")]
        self.assertTrue(prime_ticks, "预热请求应发带 prime 标记的 tick")
        phases = {e.get("phase") for e in prime_ticks}
        self.assertIn("prefill", phases, "预热请求发出即有 prefill 帧")
        self.assertIn("done", phases, "预热批结束应有 done 终帧（前端撤行用）")
        # 预热帧带场景定位字段（前端建行/分组用），且不污染测量点 tick
        for e in prime_ticks:
            self.assertEqual(e.get("scenario"), "agent")
            self.assertEqual(e.get("ctx"), 4096)
        measure_ticks = [e for e in events
                         if e.get("type") == "tick" and not e.get("prime")
                         and e.get("phase") in ("prefill", "decode")]
        self.assertTrue(measure_ticks, "测量请求 tick 不带 prime 标记")

    async def test_agent_matrix_gate_follows_cache_estimate(self):
        """速率闸跟随缓存命中判别口径（ADR-0052）：网关未回传命中但缓存档
        经预热实测断言大概率命中（est_tokens = 指令档 + 块尾）时，闸门按预估
        增量口径评估而非全量——全量口径在缓存加速的短 TTFT 上会误杀诚实点
        （DeepSeek 33K 缓存档全量 8580 tok/s 撞 10× 先验线实例）。毒药先验
        100 tok/s（×10 线 = 1000，mock 全量 ~1200 必杀）下缓存档点仍应通过"""
        def poison(run):
            run.prefill_curve[("mock-llm-7b", "agent")] = [(8192.0, 100.0)]
        _, events = await self._run_agent_matrix_collect(
            preset=poison, agent_cache_ladder=[8192], agent_inst_ladder=[128])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertTrue(pts)
        for p in pts:
            self.assertTrue(p["all_ok"],
                            f"缓存档点不应被速率闸误杀: {p.get('reqs')}")
        self.assertFalse(any("物理上不可能" in (r.get("err") or "")
                             for p in pts for r in p.get("reqs", [])),
                         "缓存档（预估增量口径）不得触发速率闸")

    async def test_failure_toast_on_rate_gate(self):
        """确定性失败弹悬浮岛（ADR-0052）：速率闸判假（非缓存档走全量口径、
        未命中量 ≥8K 先验臂生效）除 error tick 外随发 status toast:true——
        此前只有实时表一个 ✗，界面无任何提醒"""
        def poison(run):
            run.prefill_curve[("mock-llm-7b", "agent")] = [(8192.0, 100.0)]
        _, events = await self._run_agent_matrix_collect(
            preset=poison, agent_cache_ladder=[0], agent_inst_ladder=[8192])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"], "毒药先验下全量口径应判假成功")
        self.assertTrue(any("物理上不可能" in (r.get("err") or "")
                            for r in pts[0]["reqs"]))
        toasts = [e for e in events
                  if e.get("type") == "status" and e.get("toast")]
        self.assertTrue(any("失败" in e.get("msg", "") for e in toasts),
                        "请求最终失败应弹悬浮岛提醒（status toast:true）")

    async def test_agent_matrix_inst_real_tokens_correction(self):
        """指令真实长度差分与偏离校正补测（ADR-0042）：预置输入系数与 mock
        分词错位（3.5 vs 1.8）放大指令长度偏差 → 触发「实测指令长度偏离目标，
        按实测密度校正补测一次」，补测后 inst_real_tokens 收敛到档位目标
        （±10%+8 内），补测替换点位（不出新点）。"""
        _, events = await self._run_agent_matrix_collect(
            cpt=3.5, agent_cache_ladder=[0], agent_inst_ladder=[1024])
        # 预置输入系数与 mock 分词错位（3.5 vs 1.8）制造构造密度偏差：
        # 1024 档按 3.5 构造 3.5K 字符材料、mock 按 1.8 折 token → 实测 ~2 倍
        # 偏离，必触发校正补测
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "校正补测替换点位，不新增点")
        p = pts[0]
        self.assertTrue(p["all_ok"])
        self.assertIsNotNone(p["inst_real_tokens"])
        self.assertLess(abs(p["inst_real_tokens"] - 1024), 0.1 * 1024 + 8,
                        f"补测后指令真实长度应收敛到 1024±10%: {p['inst_real_tokens']}")
        self.assertTrue(any("校正补测" in e.get("msg", "")
                            for e in events if e.get("type") == "status"),
                        "应发校正补测 status 说明")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_early_stop_retest(self):
        """agent 矩阵输出异常弃测重测（ADR-0032 口径接入 agent 矩阵，
        MOCK_EARLY_STOP_LONG 一次性钩子）：首测批 finish=stop 仅 4/32 tokens
        （< 0.5×max_tokens，实测「材料截断」短答形态）→ 整批弃测、换新
        nonce+错位指令材料重测——重测正常即点干净（anomaly 键缺席）、
        reps_discarded 恰 1 条留痕、最终读数来自重测批、SSE 有弃测播报；
        预热/基线超短任务不命中钩子（>64 字符判据），不干扰预热。"""
        old = mock_server.EARLY_STOP_LONG
        mock_server.EARLY_STOP_LONG = 1
        try:
            _, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0], agent_inst_ladder=[512])
        finally:
            mock_server.EARLY_STOP_LONG = old
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "弃测重测替换点位，不新增点")
        p = pts[0]
        self.assertEqual(p["reps_discarded"], [{
            "anomaly": "early_stop", "req": 0, "rep": 1,
            "out_tokens": 4,
            "finish": "stop", "ttft_s": p["reps_discarded"][0]["ttft_s"],
            "decode_tok_s": p["reps_discarded"][0]["decode_tok_s"],
            "text_sample": "测" * 4, "reason_sample": "",
            "in_sample": p["reps_discarded"][0]["in_sample"],
            "in_text": p["reps_discarded"][0]["in_text"]}])
        self.assertTrue(p["reps_discarded"][0]["in_sample"],
                        "弃测留痕带输入侧样本（末条消息头部，ADR-0053）")
        # 矩阵弃测留痕同样带输入全文（预设上下文 + 指令两条 user 消息）
        it = p["reps_discarded"][0]["in_text"]
        self.assertGreater(len(it), 200, "in_text 为输入全文，不设采样上限")
        self.assertTrue(it.startswith("【system】"), f"全文应逐条渲染角色: {it[:80]!r}")
        self.assertEqual(it.count("【user】"), 2,
                         "矩阵构造为 system + 两条 user 消息")
        self.assertNotIn("anomaly", p, "重测成功即正常留档")
        self.assertEqual(p["out_tokens"], 32, "最终读数来自重测正常批")
        self.assertTrue(p["all_ok"])
        statuses = [e.get("msg", "") for e in events if e.get("type") == "status"]
        self.assertTrue(any("提前停止：仅输出 4/32 tokens" in s
                            and "弃测重测" in s for s in statuses),
                        f"SSE 应有弃测重测播报: {[s for s in statuses if '弃测' in s]}")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_persistent_early_stop_marked(self):
        """agent 矩阵持续早停（MOCK_EARLY_STOP_LONG 大值）：弃测重测一次后
        仍异常 → point['anomaly']='early_stop' 按实留档、reps_discarded 仅
        1 条（最多重测 ANOMALY_RETEST_MAX 次，不无限重试洗掉）、出点不中断。"""
        old = mock_server.EARLY_STOP_LONG
        mock_server.EARLY_STOP_LONG = 99
        try:
            _, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0], agent_inst_ladder=[512])
        finally:
            mock_server.EARLY_STOP_LONG = old
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        p = pts[0]
        self.assertEqual(p["anomaly"], "early_stop")
        self.assertEqual(len(p["reps_discarded"]), 1)
        self.assertEqual(p["reps_discarded"][0]["out_tokens"], 4)
        self.assertEqual(p["out_tokens"], 4, "两轮均早停：读数按实留档")
        # 留档样本取自异常轮：肇事 req 带 anomaly + in_text（输入全文抄本），
        # 全部 req 剥离 _in_full 瞬态字段（存档/SSE 不得带全文）
        culprit = p["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop")
        self.assertGreater(len(culprit["in_text"]), 200)
        self.assertTrue(culprit["in_text"].startswith("【system】"))
        self.assertFalse(any("_in_full" in r for r in p["reqs"]))
        self.assertTrue(p["all_ok"])
        self.assertIn("done", [e["type"] for e in events])

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

    async def test_edge_fallback_not_armed_below_ratio(self):
        """非贴边档不军备：未配 model_max_ctx 时持续 500（FLAKY 钩子）只走
        瞬时兜底重试，不做删减回退——err 无超限提示、无 edge_fallback 痕。"""
        mock_server.FLAKY_FAIL = 100   # 持续 500，瞬时重试额度必然耗尽
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1],
                "max_tokens": 16, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测
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
            mock_server.FLAKY_FAIL = 0
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"])
        req = pts[0]["reqs"][0]
        self.assertNotIn("显存超限", req.get("err") or "")
        self.assertNotIn("edge_fallback", req)

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
        prompt 长度」的数十万 tok/s 假 prefill，该点按错误收口。"""
        old = mock_server.SOFT_MAX_CTX
        mock_server.SOFT_MAX_CTX = 3800   # 4K 档首请求 ~4096+16 tokens 超限
        try:
            _, events = await self._run_and_collect(ctx_list=[4096], max_tokens=16)
        finally:
            mock_server.SOFT_MAX_CTX = old
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

    async def test_soft_ctx_overflow_via_reasoning_channel(self):
        """软超窗占位串走 reasoning_content 通道（MOCK_SOFT_MAX_CTX_FIELD=
        reasoning，思考模型 fastllm 实测形态）：content 为空，识别照样命中
        ——头部文本采集覆盖思考通道（实测事故：占位串走 reasoning 通道时
        38 万 tok/s 假 prefill 入档）。"""
        old_s, old_f = mock_server.SOFT_MAX_CTX, mock_server.SOFT_FIELD
        mock_server.SOFT_MAX_CTX, mock_server.SOFT_FIELD = 3800, "reasoning"
        try:
            _, events = await self._run_and_collect(ctx_list=[4096], max_tokens=16)
        finally:
            mock_server.SOFT_MAX_CTX, mock_server.SOFT_FIELD = old_s, old_f
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"], "reasoning 通道占位串同样必须判失败")
        req = pts[0]["reqs"][0]
        self.assertIn("prompt too long", req.get("err") or "")
        self.assertIsNone(req.get("prefill_tok_s"), "不得记录假 prefill 读数")

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
                "gateway_url": f"http://127.0.0.1:{self.port}",
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

    async def test_implausible_prefill_rate_small_prompt_not_gated(self):
        """物理速率闸下限（RATE_GATE_MIN_TOKENS）：同样 MOCK_PP=8000（超先验
        300 的 10× 带），但未命中量 4096 <8K——小 prompt 区先验带无判别力
        （诚实速率随 TTFT 固定开销摊薄近线性爬升），只按绝对上限判，点不误杀
        （实测 agent 零缓存档 inst=1024/4096 的 569~976 tok/s 误杀场景）。"""
        old_pp = mock_server.PP
        mock_server.PP = 8000.0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1],
                "max_tokens": 16, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            key = ("mock-llm-7b", "creative")
            run.cpt_calib[key] = 1.8                    # 跳过探测
            run.prefill_curve[key] = [(2048, 300.0)]    # 旧闸门在此会误杀本点
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
        self.assertTrue(pts[0]["all_ok"], "小 prompt 区诚实点不得被先验带误杀")
        self.assertIsNone(pts[0]["reqs"][0].get("err"))
        self.assertIn("done", [e["type"] for e in events])

    async def test_asr_rtf_against_mock(self):
        """ASR 全链路（MOCK_ASR_PP=50，即 50 倍实时）：时长阶梯×并发出点，
        净 RTF ≈ 1/50，倍速/吞吐口径闭合；ctx_list 不适用（场景自带阶梯）。"""
        old_ladder = SCENARIOS["asr"]["ladder"]
        old_pp = mock_server.ASR_PP
        SCENARIOS["asr"]["ladder"], mock_server.ASR_PP = [5, 30], 50.0
        try:
            run, events = await self._run_and_collect(
                scenarios=["asr"], ctx_list=[], concurrencies=[1, 2],
                model_kind={"mock-llm-7b": "asr"})
        finally:
            SCENARIOS["asr"]["ladder"], mock_server.ASR_PP = old_ladder, old_pp
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 档时长 × 2 并发 = 4 个点")
        self.assertTrue(all(p["all_ok"] for p in pts), f"全点应成功: {pts}")
        for p in pts:
            self.assertEqual(p["kind"], "asr")
            self.assertTrue(0.008 <= p["rtf"] <= 0.04,
                            f"RTF {p['rtf']} 应 ≈ 1/50（净口径）")
            self.assertAlmostEqual(p["speed_x"], 50, delta=20)
            self.assertAlmostEqual(p["audio_s"], p["ctx_target"], delta=0.1)
            self.assertIsNotNone(p.get("audio_min_per_min"), "全成功并发点须有吞吐")
        # 并发 2 的吞吐应显著高于并发 1（mock 无串行化，近似线性）
        t1 = next(p["audio_min_per_min"] for p in pts
                  if p["ctx_target"] == 30 and p["concurrency"] == 1)
        t2 = next(p["audio_min_per_min"] for p in pts
                  if p["ctx_target"] == 30 and p["concurrency"] == 2)
        self.assertGreater(t2, 1.5 * t1, f"并发吞吐应近线性: {t1} → {t2}")
        self.assertGreater(next(p["out_chars"] for p in pts
                                if p["ctx_target"] == 30), 30)

    async def test_media_ladder_override_effective(self):
        """媒体场景自定义阶梯生效：cfg 的 <场景>_ladder 覆盖场景默认阶梯——
        出点档位集合 = 覆盖值而非默认值（asr 默认 [5,15,60,300,1800]、
        ocr 默认 [1,2,4,8,16]）。"""
        _, events = await self._run_and_collect(
            scenarios=["asr"], ctx_list=[], concurrencies=[1],
            asr_ladder=[5, 15], model_kind={"mock-llm-7b": "asr"})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2, "覆盖阶梯 2 档 × 1 并发 = 2 个点")
        self.assertEqual(sorted({p["ctx_target"] for p in pts}), [5, 15],
                         "出点档位应为覆盖阶梯 [5,15]，不得混入默认档位")
        self.assertTrue(all(p["all_ok"] and p["kind"] == "asr" for p in pts))
        # ocr 同口径：覆盖阶梯生效
        _, events = await self._run_and_collect(
            scenarios=["ocr"], ctx_list=[], concurrencies=[1],
            max_tokens=64, ocr_ladder=[2],
            model_kind={"mock-llm-7b": "ocr"})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "覆盖阶梯 [2] 只出 1 个点")
        self.assertEqual(pts[0]["ctx_target"], 2)
        self.assertEqual(pts[0]["n_img"], 2)
        self.assertTrue(pts[0]["all_ok"])

    async def test_asr_ctx_overflow_skipped(self):
        """ASR 超窗保护：时长×50 tokens/秒 超 max_ctx 档位直接 point_skipped
        （不发出必然失败的请求）。"""
        old_ladder = SCENARIOS["asr"]["ladder"]
        SCENARIOS["asr"]["ladder"] = [30, 120]   # 1500 / 6000 tokens
        try:
            _, events = await self._run_and_collect(
                scenarios=["asr"], ctx_list=[], concurrencies=[1, 2],
                model_kind={"mock-llm-7b": "asr"},
                model_max_ctx={"mock-llm-7b": 4096})
        finally:
            SCENARIOS["asr"]["ladder"] = old_ladder
        pts = [e["point"] for e in events if e.get("type") == "point"]
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(pts), 2, "30s 档 × 2 并发应正常出点")
        self.assertEqual(len(skipped), 1, "120s 档应整档跳过")
        self.assertEqual(skipped[0]["ctx"], 120)
        self.assertEqual(skipped[0]["count"], 2, "跳过计数须含全部并发，前端进度才对")
        self.assertIn("超出部署上限", skipped[0]["msg"])

    async def test_tts_against_mock(self):
        """TTS 全链路（MOCK_TTS_RATE=500 字/秒）：文本阶梯出点，音频时长按
        返回 wav 解析（非语速估算），TTFA 为首 chunk 到达、≈0。"""
        old_ladder = SCENARIOS["tts"]["ladder"]
        old_rate, old_xr = mock_server.TTS_RATE, mock_server.TTS_XR
        SCENARIOS["tts"]["ladder"] = [50, 200]
        mock_server.TTS_RATE, mock_server.TTS_XR = 500.0, 1000.0
        try:
            _, events = await self._run_and_collect(
                scenarios=["tts"], ctx_list=[], concurrencies=[1],
                model_kind={"mock-llm-7b": "tts"})
        finally:
            SCENARIOS["tts"]["ladder"] = old_ladder
            mock_server.TTS_RATE, mock_server.TTS_XR = old_rate, old_xr
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2)
        for p in pts:
            self.assertEqual(p["kind"], "tts")
            self.assertNotIn("audio_est", p, "mock 回合法 wav，时长必须为解析值")
            expect = p["ctx_target"] / 500.0
            self.assertAlmostEqual(p["audio_s"], expect, delta=max(0.03, expect * 0.1))
            self.assertLess(p["ttfa_s"], 0.5, f"首 chunk 即刻到达: {p['ttfa_s']}")
            self.assertTrue(0.5 <= p["speed_x"] <= 5000, f"倍速口径异常: {p['speed_x']}")
            self.assertGreater(p["out_bytes"], 1000)

    async def test_ocr_images_with_mock(self):
        """OCR 全链路：VLM 读图请求（多模态 content parts）mock 正常受理，
        流式 TTFT/decode 读数 + ms/img 与并发吞吐口径闭合。"""
        old_ladder = SCENARIOS["ocr"]["ladder"]
        SCENARIOS["ocr"]["ladder"] = [1, 4]
        try:
            _, events = await self._run_and_collect(
                scenarios=["ocr"], ctx_list=[], concurrencies=[1, 2],
                max_tokens=64,
                model_kind={"mock-llm-7b": "ocr"})
        finally:
            SCENARIOS["ocr"]["ladder"] = old_ladder
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 档张数 × 2 并发")
        self.assertTrue(all(p["all_ok"] for p in pts), f"读图请求不得判失败: {pts}")
        for p in pts:
            self.assertEqual(p["kind"], "ocr")
            self.assertEqual(p["n_img"], p["ctx_target"])
            self.assertGreater(p["ms_per_img"], 0)
            self.assertIsNotNone(p.get("img_per_s"), "全成功并发点须有吞吐")
            self.assertIsNotNone(p.get("ttft_s"), "VLM 流式路径应有 TTFT")
            self.assertIsNotNone(p.get("decode_tok_s"))
            req = p["reqs"][0]
            self.assertIsNone(req.get("err"))
            self.assertGreater(req["out_tokens"], 0)

    async def test_media_kind_mismatch_skipped(self):
        """模型 kind 与场景 kind 不匹配（直调引擎绕过服务端校验时）：引擎第二道
        闸跳过该场景组合并播报，不出点、不报错死亡。"""
        _, events = await self._run_and_collect(
            scenarios=["asr"], ctx_list=[], concurrencies=[1],
            model_kind={"mock-llm-7b": "llm"})   # llm 模型 × asr 场景
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(pts, [])
        types = [e["type"] for e in events]
        self.assertIn("done", types)
        skips = [e["msg"] for e in events if e.get("type") == "status"
                 and "不匹配" in e.get("msg", "")]
        self.assertTrue(skips, "应有 kind 不匹配跳过播报")

    async def test_media_model_kinds_multi_capability(self):
        """多能力模型（model_kinds=["llm","ocr"]，ADR-0020 多选扩展）：ocr 场景
        正常出点不被能力闸误拦；能力集不含的 kind 照旧跳过播报。"""
        old_ladder = SCENARIOS["ocr"]["ladder"]
        SCENARIOS["ocr"]["ladder"] = [1]
        try:
            _, events = await self._run_and_collect(
                scenarios=["ocr"], ctx_list=[], concurrencies=[1],
                max_tokens=64,
                model_kinds={"mock-llm-7b": ["llm", "ocr"]})
        finally:
            SCENARIOS["ocr"]["ladder"] = old_ladder
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "多能力模型应正常跑 ocr 场景")
        _, events = await self._run_and_collect(
            scenarios=["asr"], ctx_list=[], concurrencies=[1],
            model_kinds={"mock-llm-7b": ["llm", "ocr"]})   # 能力集不含 asr
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(pts, [])
        skips = [e["msg"] for e in events if e.get("type") == "status"
                 and "不匹配" in e.get("msg", "")]
        self.assertTrue(skips, "能力集不含 asr 时应有跳过播报")

    async def test_translate_against_mock(self):
        """翻译场景全链路（ADR-0034）：原文字长阶梯出点（ctx_list 不适用，
        阶梯由 cfg.translate_ladder 自定义下发），走 chat 文本流路径
        （kind=llm），prompt 随档位放大、输出预算随档伸缩。"""
        _, events = await self._run_and_collect(
            scenarios=["translate"], ctx_list=[], concurrencies=[1, 2],
            translate_ladder=[50, 200])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 档原文字长 × 2 并发 = 4 个点")
        self.assertTrue(all(p["all_ok"] for p in pts), f"全点应成功: {pts}")
        for p in pts:
            self.assertEqual(p["kind"], "llm")   # 与创意/代码同一请求路径
            self.assertIn(p["ctx_target"], [50, 200])   # 档位 = 原文字数（非 tokens）
            self.assertIsNotNone(p.get("ttft_s"))
            self.assertIsNotNone(p.get("decode_tok_s"))
            self.assertGreater(p["out_tokens"], 0)
        p50 = next(p for p in pts
                   if p["ctx_target"] == 50 and p["concurrency"] == 1)
        p200 = next(p for p in pts
                    if p["ctx_target"] == 200 and p["concurrency"] == 1)
        self.assertGreater(p200["prompt_tokens"], p50["prompt_tokens"],
                           "原文 4 倍，prompt 应显著放大")

    async def test_translate_ctx_overflow_skipped(self):
        """翻译场景超窗保护：原文 tokens 估值 + 当档输出预算超 max_ctx 的档位
        整档 point_skipped（不发出必然失败的请求）。"""
        _, events = await self._run_and_collect(
            scenarios=["translate"], ctx_list=[], concurrencies=[1, 2],
            translate_ladder=[50, 3200],
            model_max_ctx={"mock-llm-7b": 2048})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(pts), 2, "50 字档 × 2 并发应正常出点")
        self.assertEqual(len(skipped), 1, "3200 字档应整档跳过")
        self.assertEqual(skipped[0]["ctx"], 3200)
        self.assertEqual(skipped[0]["count"], 2, "跳过计数须含全部并发，前端进度才对")
        self.assertIn("超出部署上限", skipped[0]["msg"])

    async def test_agent_matrix_cache_hit_reported(self):
        """agent 矩阵（MOCK_CACHE=1 前缀缓存）：预热请求把上下文写入缓存且
        不计入测点（总点数=组合数）；测量请求命中回传（cache_hit_tokens>0、
        cache_reported 真），prefill 为增量口径（≈ mock PP，远小于全量口径）；
        同缓存档逐尝试错位取段（游标），指令材料不再嵌套前缀——各指令档
        命中同为上下文主体（不再随档增大而命中更多）；指令真实长度
        （测量 prompt − 预热/基线 prompt 链内差分）≈ 档位目标（CPT 对齐时不
        触发偏离校正）。"""
        old_cache = mock_server.CACHE
        mock_server.CACHE = "1"
        mock_server._SEEN.clear()
        try:
            run, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 4096], agent_inst_ladder=[512, 2048])
        finally:
            mock_server.CACHE = old_cache
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2×2 组合应出 4 点（预热不计入测点）")
        by = {(p["ctx_target"], p["inst_tokens"]): p for p in pts}
        self.assertTrue(all(p["all_ok"] for p in pts))
        # 零缓存档：mock 回传命中字段但只命中系统提示级+编号行前缀
        # （基线批与测量同 nonce，命中含编号行与 system 双向 out_hint，
        # ~168 tokens）
        for it in (512, 2048):
            p = by[(0, it)]
            self.assertTrue(p["cache_reported"])
            self.assertLess(p["cache_hit_tokens"], 240,
                            "零缓存档只应命中系统提示级共享前缀")
            # 指令真实长度（链内差分）：CPT 对齐时不触发偏离校正
            self.assertIsNotNone(p["inst_real_tokens"],
                                 "测量/基线批 usage 齐备时应产出指令真实长度")
            self.assertLess(abs(p["inst_real_tokens"] - it), 0.1 * it + 8,
                            f"指令真实长度应贴近档位 {it}")
        # 缓存档：命中 ≈ 已缓存上下文主体（回传真值，非估算口径）
        h512, h2048 = by[(4096, 512)], by[(4096, 2048)]
        for p in (h512, h2048):
            self.assertTrue(p["cache_reported"], "mock 回传命中字段，非估算口径")
            self.assertGreater(p["cache_hit_tokens"], 3500,
                               "应命中预热的上下文主体")
            self.assertIsNotNone(p["inst_real_tokens"])
            # 增量 prefill 口径：未命中 tokens ÷ TTFT ≈ MOCK_PP=3000
            self.assertAlmostEqual(p["prefill_tok_s"], 3000, delta=900,
                                   msg="增量 prefill 偏离 mock 设定")
            # 口径一致性（additive 双口径）：全量口径 ≫ 增量口径——上下文主体
            # 命中，只增量 prefill 指令段（2048 档未命中占比 ~1/3，比值贴 3
            # 以下，阈值取 2）
            self.assertGreater(p.get("prefill_full_tok_s"),
                               p["prefill_tok_s"] * 2)
            for r in p["reqs"]:
                self.assertIsNotNone(r.get("prefill_uncached_tok_s"),
                                     "缓存命中请求应带 req 级未命中口径字段")
            # 端到端总时长 + conc=1 单链总吞吐同单请求 decode 口径
            self.assertIsNotNone(p["total_s"])
            self.assertIsNotNone(p["decode_total_tok_s"])
            self.assertAlmostEqual(p["decode_total_tok_s"], p["decode_tok_s"],
                                   delta=1.0)
        # 指令材料逐尝试错位取段（游标，不再嵌套前缀）：2048 档段不含 512
        # 档段，两档命中同为预热的上下文主体——差异仅指令模板共享前缀
        # （inst_prefix 百余字符）与边界 token，远小于一个指令档
        self.assertAlmostEqual(h2048["cache_hit_tokens"],
                               h512["cache_hit_tokens"], delta=256)
        # TTFT 随自身增量放大：2048 档未命中 ≈ 2K tokens vs 512 档 ≈ 0.5K
        self.assertGreater(h2048["ttft_s"], h512["ttft_s"] * 2)
        # 预热完成 status 带实测 prompt 量（前端展示用）
        self.assertTrue(any("实测 prompt" in e.get("msg", "")
                            for e in events if e.get("type") == "status"))
        # 每点实测速率登记进先验曲线：（0,512) 与 (4096,512) 未命中量同为
        # ~0.5K，同 ctx 坐标互相覆盖 → 4 点落 3 个曲线点
        curve = run.prefill_curve.get(("mock-llm-7b", "agent"), [])
        self.assertGreaterEqual(len(curve), 3,
                                f"4 点应登记 ≥3 个先验点（同 x 覆盖）: {curve}")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_without_cache_degrades_honestly(self):
        """无前缀缓存能力（MOCK_CACHE 关）：缓存档 TTFT 随全量 prompt 增长——
        缓存迹象判别为「无缓存」：命中列 None（前端显示未回传，不展示假设
        命中数），prefill 退回全量口径实测真值（≈ mock PP），不按假设命中
        折算。"""
        mock_server._SEEN.clear()
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[0, 4096], agent_inst_ladder=[512])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2)
        by = {p["ctx_target"]: p for p in pts}
        base, cached = by[0], by[4096]
        self.assertFalse(cached["cache_reported"], "无缓存字段回传时必须标记非真值")
        self.assertIsNone(cached["cache_hit_tokens"],
                          "无缓存迹象时命中为 None（前端显示未回传）")
        # 缓存档全量 ~4.6K tokens 重算：TTFT ≈ 7× 零缓存档（~0.7K tokens）
        self.assertGreater(cached["ttft_s"], base["ttft_s"] * 3)
        # 全量口径：prompt_tokens ÷ TTFT ≈ MOCK_PP=3000（真实可测速率）
        self.assertAlmostEqual(cached["prefill_tok_s"], 3000, delta=450)
        # 命中 None 时增量口径 = 全量口径（无假设命中折算）
        self.assertAlmostEqual(cached.get("prefill_full_tok_s"),
                               cached["prefill_tok_s"], delta=0.05)

    async def test_agent_matrix_cache_assumed_when_ttft_flat(self):
        """缓存生效但网关剥离命中字段（MOCK_CACHE_NOREPORT）：缓存档测量 TTFT
        相对同指令零缓存档走平 → 判别「有缓存迹象」，命中按预热实测 prompt
        估算（cache_reported=False，前端加 ≈），prefill 为增量口径 ≈ mock PP。"""
        old = mock_server.CACHE_NOREPORT
        mock_server.CACHE_NOREPORT = "1"
        mock_server._SEEN.clear()
        try:
            _, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 4096], agent_inst_ladder=[512])
        finally:
            mock_server.CACHE_NOREPORT = old
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2)
        by = {p["ctx_target"]: p for p in pts}
        base, cached = by[0], by[4096]
        self.assertFalse(cached["cache_reported"])
        self.assertIsNotNone(cached["cache_hit_tokens"],
                             "有缓存迹象应按预热实测 prompt 估算命中")
        self.assertGreater(cached["cache_hit_tokens"], 3000,
                           "估算命中应为预热实测 prompt 量级")
        self.assertAlmostEqual(cached["prefill_tok_s"], 3000, delta=900,
                               msg="增量 prefill 偏离 mock 设定")
        self.assertLess(cached["ttft_s"], base["ttft_s"] * 2 + 0.3,
                        "缓存档 TTFT 应相对零缓存档走平")

    async def test_agent_matrix_cache_hit_slow_read_estimated(self):
        """缓存命中但读取慢（MOCK_CACHE_READ_TOK_S=N，N≫PP，模拟实测
        DeepSeek 后端 KV 大部在 CPU 的形态）：大缓存档 TTFT 随命中规模线性
        增长、相对零缓存档不走平（走平判别失效），但显著低于同规模全量
        prefill 预期 → 全量预期/佐证通道判别估算命中（cache_reported=False），
        prefill 为增量口径（被缓存读取摊薄，远小于全量口径）；估算命中点
        不登记进 prefill 先验曲线（假全量速率会污染估值锚点）。"""
        mock_server.CACHE_READ_S = 8000.0
        mock_server._SEEN.clear()
        try:
            run, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 8192], agent_inst_ladder=[512, 2048])
        finally:
            mock_server.CACHE_READ_S = 0.0
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4)
        by = {(p["ctx_target"], p["inst_tokens"]): p for p in pts}
        self.assertTrue(all(p["all_ok"] for p in pts))
        for it in (512, 2048):
            base, cached = by[(0, it)], by[(8192, it)]
            # 走平判别失效：缓存读取耗时 ≫ 零缓存档全量 prefill，TTFT 不走平
            self.assertGreater(cached["ttft_s"], base["ttft_s"] + 0.55,
                               f"inst {it} 缓存档 TTFT 应显著高于零缓存档")
            # 仍判出估算命中（全量预期/佐证通道）：非回传、非真值口径
            self.assertFalse(cached["cache_reported"])
            self.assertIsNotNone(cached["cache_hit_tokens"],
                                 "TTFT 显著低于同规模全量预期应估算命中")
            self.assertGreater(cached["cache_hit_tokens"], 7000,
                               "估算命中应为预热实测 prompt 量级")
            # prefill 增量口径：TTFT 含缓存读取耗时，速率被摊薄
            self.assertLess(cached["prefill_tok_s"], 1500,
                            "增量 prefill 应远小于全量口径")
            self.assertGreater(cached.get("prefill_full_tok_s", 0),
                               cached["prefill_tok_s"] * 3)
        # 先验曲线污染防护：估算命中点不登记，曲线只含零缓存档全量正证据
        curve = run.prefill_curve.get(("mock-llm-7b", "agent"), [])
        self.assertEqual(len(curve), 2, f"只应登记 2 个零缓存档先验点: {curve}")
        self.assertTrue(all(x < 4096 for x, _ in curve),
                        f"先验点 x 应为零缓存档 prompt 规模: {curve}")

    async def test_agent_matrix_ctx_limit_skips_rungs(self):
        """预算跳档（ADR-0042）：C+I 连同输出预算超部署上限的组合整档跳过
        （point_skipped 按缓存档聚合 count），其余组合照常出点——矩阵不再
        「提前终止」而是「整档跳过」。"""
        mock_server._SEEN.clear()
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[0, 4096, 16384], agent_inst_ladder=[512],
            model_max_ctx={"mock-llm-7b": 8192})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        # 上限 8192−32−1024=7136：缓存 16384 档全部组合超预算
        self.assertEqual([(p["ctx_target"], p["inst_tokens"]) for p in pts],
                         [(0, 512), (4096, 512)])
        self.assertTrue(all(p["all_ok"] for p in pts))
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(skipped), 1, f"16384 档应恰好一条 point_skipped: {skipped}")
        self.assertEqual(skipped[0]["ctx"], 16384)
        self.assertEqual(skipped[0]["count"], 1)
        self.assertIn("超出部署上限", skipped[0]["msg"],
                      "point_skipped 须带 msg 供前端常驻展示")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_terminates_on_overflow(self):
        """某组合软超窗（MOCK_SOFT_MAX_CTX 卡在中段组合）矩阵提前收口：失败
        组合照常出点（all_ok=False），其后组合 point_skipped（count=剩余组合
        数，实测事故：超窗后仍继续构造更大组合打爆上限）。"""
        old = mock_server.SOFT_MAX_CTX
        mock_server.SOFT_MAX_CTX = 4500   # 缓存 4096 的预热/128 指令档可过；2048 档（~6K）超限
        try:
            _, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 4096, 8192],
                agent_inst_ladder=[128, 2048])
        finally:
            mock_server.SOFT_MAX_CTX = old
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        combos = [(p["ctx_target"], p["inst_tokens"]) for p in pts]
        self.assertEqual(combos, [(0, 128), (0, 2048), (4096, 128), (4096, 2048)],
                         "矩阵必须在超窗组合处收口，不得跑完")
        self.assertTrue(all(p["all_ok"] for p in pts[:3]), "前三个组合应正常成功")
        failed = pts[3]
        self.assertFalse(failed["all_ok"], "超窗组合必须判失败")
        self.assertIn("prompt too long", failed["reqs"][0].get("err") or "")
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(skipped), 1, f"剩余组合应恰好一条 point_skipped: {skipped}")
        self.assertEqual(skipped[0]["count"], 2, "剩余组合 (8192,128)+(8192,2048)")
        self.assertEqual(skipped[0]["ctx"], 8192)
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
        # mock 的合成信标正文是单字符重复（"测"×N），max_tokens=128 → 128 字符
        # ≥80 且 distinct=1 ≤12，命中退化重复判据触发弃测重测——两次测量批的
        # decode tick 混进同一订阅流污染滑窗断言。本测试主题是滑窗差分 tick
        # 机制、与异常输出守卫无关，故屏蔽检测（守卫本身的行为覆盖在
        # test_engine_unit 的 _run_rep_guarded 用例）
        old_det = bench._detect_rep_anomaly
        mock_server.BURST, mock_server.TG = 15, 20.0
        bench._detect_rep_anomaly = lambda *a, **k: None
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
            bench._detect_rep_anomaly = old_det
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
        SSE 事件数直计兜底（1 事件/token，mock 每事件 1 字符即 16 tokens，
        与真实值一致），chunk_calib 不播种（校准依赖真实 usage）。"""
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
        # 兜底口径：16 个内容事件直计 = 16（= 真实 token 数，旧字符口径为 11）
        self.assertEqual(r["out_tokens"], 16)
        self.assertEqual(r["finish"], "stop")
        self.assertEqual(run.chunk_calib, {}, "无 usage 时 chunk_calib 不得播种")

    async def test_no_usage_multichar_chunks_not_overshot(self):
        """英文形态输出（MOCK_CHUNK_CHARS=4：1 事件/token、每事件 4 字符）
        + 无 usage：旧口径按中文 1.5 字符/token 估算 64/1.5≈43，超断流阈值
        32 触发 client_cut；新口径按事件数直计 16，自然收尾 finish=length——
        fastllm 英文输出 decode 虚高 2.8 倍的回归（ADR-0019）。"""
        old_u, old_ch = mock_server.NO_USAGE, mock_server.CH
        mock_server.NO_USAGE, mock_server.CH = "1", 4
        try:
            run, events = await self._run_and_collect()
        finally:
            mock_server.NO_USAGE, mock_server.CH = old_u, old_ch
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertTrue(p["all_ok"], f"英文形态输出不得误判断流: {p['reqs']}")
        r = p["reqs"][0]
        self.assertFalse(r["usage_real"])
        self.assertEqual(r["finish"], "stop", "自然收尾（mock  finish_reason 恒为 stop），不得 client_cut")
        self.assertEqual(r["out_tokens"], 16, "事件数直计 = 真实 token 数")
        self.assertEqual(r["text_chars"], 64)
        # decode 读数按真实 token 数 / 实测窗口，接近 mock TG（±20%）
        self.assertGreater(r["decode_tok_s"], 300 * 0.8)

    async def test_archive_output_samples(self):
        """存档输出采样（text_sample/reason_sample，各 ≤200 字符头部）：
        正文超 200 截断（64 tokens × 4 字符 = 256 → 200）、思考流不足全留
        （mock 思考流每 token 恒 1 字符 × 2 tokens）——退化输出形态
        （~1 字符/token 塌缩）与 echo 改写口径的存档取证字段，仅供存档
        定位，前端不消费。"""
        old_ch, old_re = mock_server.CH, mock_server.REASON
        mock_server.CH, mock_server.REASON = 4, 2
        try:
            run, events = await self._run_and_collect(max_tokens=64)
        finally:
            mock_server.CH, mock_server.REASON = old_ch, old_re
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        r = points[0]["reqs"][0]
        self.assertIsNone(r["err"], f"采样测试请求应成功: {r}")
        self.assertEqual(len(r["text_sample"]), 200, "正文超 200 截断")
        self.assertEqual(r["text_chars"], 256)
        self.assertEqual(r["text_sample"], "测" * 200, "截断即停于 200 字符")
        self.assertEqual(r["reason_sample"], "思" * 2, "思考流不足全留")
        self.assertEqual(r["reason_chars"], 2)

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
        """ADR-0016 防线 2：system 提示注入输出长度双向引导（aim 目标 +
        max_tokens × out_cpt 先验 × 1.2 封顶）——零输入档 sent_chars 应恰好多出
        引导文案长度。"""
        _, events = await self._run_and_collect(max_tokens=16)
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        sc = SCENARIOS["creative"]
        hint = "\n\n" + sc["out_hint"].format(limit=int(16 * sc["out_cpt"] * 1.2),
                                              aim=int(16 * sc["out_cpt"]))
        expect = len(sc["system"]) + len(hint) + len(sc["zero_instruction"])
        self.assertEqual(points[0]["reqs"][0]["sent_chars"], expect,
                         "system 提示应恰好追加输出长度引导文案")

    async def test_repeats_one_transparency_notice(self):
        """repeats=1 透明度：运行开始播报一次「单次测量无统计效力」提醒；
        repeats≥3 不播报（不淹没复测流程的状态行）。"""
        statuses1 = await self._run_and_collect_status()
        self.assertTrue(any("无统计效力" in s for s in statuses1),
                        f"repeats=1 应有透明度提醒: {statuses1}")
        _, events3 = await self._run_and_collect(repeats=3)
        statuses3 = [e["msg"] for e in events3 if e.get("type") == "status"]
        self.assertFalse(any("无统计效力" in s for s in statuses3),
                         f"repeats=3 不应出现该提醒: {statuses3}")

    async def test_tg_pattern_alternates_decode_speed(self):
        """MOCK_TG_PATTERN（如 "60,120"）：逐请求交替 decode 速度，两档都被
        引擎采到（双峰复现的 mock 基建）。"""
        old_pp, old_pat = mock_server.PP, mock_server.TG_PATTERN
        mock_server.PP = 3000.0
        mock_server.TG_PATTERN = [60.0, 120.0]
        mock_server._REQ_SEQ = 0
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{self.port}",
                "models": ["mock-llm-7b"], "scenarios": ["creative"],
                "ctx_list": [0, 2048], "concurrencies": [1],
                "max_tokens": 32, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测：chat 请求恰两个
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
            mock_server.PP, mock_server.TG_PATTERN = old_pp, old_pat
        self.assertEqual(len(points), 2)
        d0, d1 = points[0]["decode_tok_s"], points[1]["decode_tok_s"]
        self.assertTrue(45 <= d0 <= 85, f"第一请求应落在 60 tok/s 档: {d0}")
        self.assertTrue(85 <= d1 <= 165, f"第二请求应落在 120 tok/s 档: {d1}")


if __name__ == "__main__":
    unittest.main()
