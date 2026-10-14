"""mcp_api 组：MCP 服务（mcp_server.py）回归——纯格式化 / 工具层端到端 / stdio 握手。

自 tests/test_mcp_server.py 拆分，组内就这一个文件，三层分工：

1. 纯格式化函数：喂合成存档断言数据卡片/对比表渲染（含贴边裁减点归一到
   原档位的 cmpGearOf 同口径对齐、agent 缓存×指令矩阵、媒体表）。
2. 工具层端到端：mock 网关 + 真实 server（uvicorn 线程）+ 直接调用工具
   函数——配置/模型列表/场景/起测/状态/结果/历史/对比全链路。
3. stdio 协议握手：子进程拉起 mcp_server.py，按 JSON-RPC 帧 initialize →
   notifications/initialized → tools/list → tools/call，验证真实线协议。

两个入口都在仓库根目录跑（mock 子进程路径与 `tests._shared` 导入都以此为基准）：

    .venv/bin/python -m pytest tests/mcp_api/test_mcp_server.py -q
    .venv/bin/python -m unittest tests.mcp_api.test_mcp_server -v
    .venv/bin/python -m unittest discover -s tests/mcp_api -p test_mcp_server.py
"""
import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

# 顺序是硬约束：mock_server 在 import 时读取 MOCK_* 环境变量，而播种落在
# tests._shared 的模块级——先 import 它（触发播种），再 import mock_server。
# 反过来播种就晚了：pytest 同进程下「先 import 者生效」，口径分裂成默认值
# 而非与 test_e2e_mock 一致的 3000/300/1.8。
import tests._shared                    # noqa: E402,F401  —— 触发 MOCK_* 播种
from tests._shared import _free_port    # noqa: E402

import mock_server                      # noqa: E402
import server                           # noqa: E402
import uvicorn                          # noqa: E402
import mcp_server                       # noqa: E402


class _StripMockHeader:
    """剥掉 mock 网关的 x-mock-server 标记头（纯 ASGI 包装，不动 mock_server）。

    引擎见到该头会按设计「mock 运行不落历史记录」（bench.py mock_seen →
    _save 跳过）——防的是假数据污染真实存档。本套 e2e 的 RESULTS_DIR 指向
    临时目录，恰恰要验证 起测→落盘→取卡片 全链路，故剥头恢复落盘。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_no_marker(message):
            if message["type"] == "http.response.start":
                message["headers"] = [(k, v) for k, v in message["headers"]
                                      if k.lower() != b"x-mock-server"]
            await send(message)

        return await self.app(scope, receive, send_no_marker)


# ---------------------------------------------------------------------------
# 合成存档
# ---------------------------------------------------------------------------

_META = {
    "code": {"label": "代码生成", "kind": "llm"},
    "creative": {"label": "创意写作", "kind": "llm"},
    "agent": {"label": "Agent 调用", "kind": "llm"},
    "asr": {"label": "语音转写", "kind": "asr", "unit": "s"},
    "ocr": {"label": "图像识别", "kind": "ocr", "unit": "张"},
}


def _llm_pt(model="mA", scen="code", ctx=0, conc=1, edge=None, **kw):
    p = {"model": model, "scenario": scen, "ctx_target": ctx, "concurrency": conc,
         "prompt_tokens": 167, "ttft_s": 0.52, "prefill_tok_s": 517.9,
         "decode_tok_s": 29.4, "total_span_s": 9.22, "cache_hit_tokens": 109,
         "all_ok": True}
    if edge is not None:
        p["ctx_edge"] = edge
    p.update(kw)
    return p


def _agent_pt(model="mA", cache=0, inst=64, conc=1, **kw):
    p = {"model": model, "scenario": "agent", "ctx_target": cache,
         "inst_tokens": inst, "concurrency": conc,
         "ttft_s": 0.95, "prefill_tok_s": 163.0, "decode_tok_s": 67.2,
         "total_s": 1.36, "cache_hit_tokens": 0, "cache_reported": False,
         "all_ok": True}
    p.update(kw)
    return p


def _asr_pt(model="mA", sec=30, conc=1, **kw):
    p = {"model": model, "scenario": "asr", "kind": "asr", "ctx_target": sec,
         "concurrency": conc, "audio_s": float(sec), "elapsed_s": 6.1,
         "rtf": 0.2, "speed_x": 5.0, "audio_min_per_min": 9.8, "all_ok": True}
    p.update(kw)
    return p


def _archive(points, models=("mA",), scenarios=("code",), local=True,
             run_id="20260101-000000-ab12", info=True):
    cfg = {"provider": "p1", "provider_label": "P1", "provider_local": local,
           "gateway_url": "http://gw.example", "models": list(models),
           "scenarios": list(scenarios), "max_tokens": 1024, "repeats": 3,
           "reply_mode": "free"}
    if info:
        cfg["model_info"] = {m: {"deploy_label": "dep", "hardware": "RTX 4090",
                                 "framework": "llama.cpp", "params": "-c 262144",
                                 "quant": "Q8_0", "kv_quant": "Q8_0",
                                 "max_ctx": 262144} for m in models}
    return {"run_id": run_id, "started_at": "2026-01-01 00:00:00",
            "metric_version": 1, "cfg": cfg, "results": list(points)}


# ---------------------------------------------------------------------------
# 1. 纯格式化
# ---------------------------------------------------------------------------

class TestFormatting(unittest.TestCase):
    def test_gear_of_edge_normalization(self):
        self.assertEqual(mcp_server._gear_of({"ctx_edge": 262144,
                                              "ctx_target": 260608}), 262144)
        self.assertEqual(mcp_server._gear_of({"ctx_target": 4096}), 4096)
        self.assertEqual(mcp_server._gear_of({"ctx_target": 0}), 0)
        self.assertIsNone(mcp_server._gear_of({}))

    def test_gear_label_by_scenario(self):
        gl = mcp_server._gear_label
        self.assertEqual(gl("code", 0, None), "0K")
        self.assertEqual(gl("code", 4096, None), "4K")
        self.assertEqual(gl("code", 260608, None), "254.5K")  # 裁后原值诚实显示
        self.assertEqual(gl("agent", 131072, None), "128K")
        self.assertEqual(gl("asr", 30, "s"), "30s")
        self.assertEqual(gl("ocr", 4, "张"), "4 张")
        self.assertEqual(gl("translate", 4000, "字"), "4000 字")
        self.assertEqual(gl("tts", 800, "字"), "800 字")

    def test_status_and_hit(self):
        self.assertEqual(mcp_server._status_of({"all_ok": True}), "✓")
        self.assertEqual(mcp_server._status_of({"all_ok": False,
                                                "err": "boom"}), "✗ boom")
        self.assertEqual(mcp_server._status_of({"all_ok": True,
                                                "anomaly": "early_stop"}),
                         "⚠ early_stop")
        # 缓存命中三态：回传真值 / 估算≈ / 无读数
        self.assertEqual(mcp_server._hit_of({"cache_hit_tokens": 109}), "109")
        self.assertEqual(mcp_server._hit_of({"cache_hit_tokens": 0,
                                             "cache_reported": False}), "≈0")
        self.assertEqual(mcp_server._hit_of({"cache_hit_tokens": None}), "—")

    def test_card_llm_edge_point_at_original_gear(self):
        """贴边裁减点在卡片里显示原逻辑档位（✂裁减事实由 err/anomaly 之外
        的口径保留在存档），不得出现 254.5K 这种裁后值档位行。"""
        arc = _archive([
            _llm_pt(ctx=0), _llm_pt(ctx=4096),
            _llm_pt(ctx=260608, edge=262144, prompt_tokens=218463),
        ])
        text = mcp_server.render_card(arc, _META)
        self.assertIn("# ⚡ 测速数据卡片 · P1", text)
        self.assertIn("**mA** · dep · RTX 4090 · llama.cpp", text)
        self.assertIn("max_tokens=1024 · repeats=3", text)
        self.assertIn("256K", text)
        self.assertNotIn("254.5K", text)
        # 本地部署隐去网关地址（与前端卡片同口径）
        self.assertNotIn("gw.example", text)

    def test_card_cloud_shows_gateway(self):
        arc = _archive([_llm_pt()], local=False)
        self.assertIn("gw.example", mcp_server.render_card(arc, _META))

    def test_card_multi_model_sections_and_failed_point(self):
        arc = _archive([_llm_pt(model="mA"), _llm_pt(model="mB"),
                        _llm_pt(model="mB", ctx=4096, all_ok=False, err="空流")],
                       models=("mA", "mB"))
        text = mcp_server.render_card(arc, _META)
        self.assertIn("**mA**", text)
        self.assertIn("**mB**", text)
        self.assertIn("✗ 空流", text)

    def test_card_agent_matrix(self):
        arc = _archive([
            _agent_pt(cache=0, inst=64), _agent_pt(cache=0, inst=128),
            _agent_pt(cache=4096, inst=64),
        ], scenarios=("agent",))
        text = mcp_server.render_card(arc, _META)
        self.assertIn("Agent 调用（agent）", text)
        self.assertIn("缓存 \\ 指令", text)
        self.assertIn("0.95 / 67.2", text)
        # 缺口位（4K×128 无点）显式 —，不编数
        self.assertIn("—", text)

    def test_card_media_asr(self):
        arc = _archive([_asr_pt(sec=30), _asr_pt(sec=60)],
                       scenarios=("asr",))
        text = mcp_server.render_card(arc, _META)
        self.assertIn("语音转写（asr）", text)
        self.assertIn("RTF", text)
        self.assertIn("30s", text)
        self.assertIn("0.2", text)

    def test_card_empty(self):
        self.assertIn("无测速点",
                      mcp_server.render_card(_archive([]), _META))

    def test_compare_aligns_edge_point_across_runs(self):
        """run A 的贴边裁减点与 run B 的正常 256K 点同档对齐——这正是前端
        「128K 后线断」事故的口径：不归一就被拆成两行、各自出 null。"""
        a = _archive([_llm_pt(ctx=0), _llm_pt(ctx=260608, edge=262144)],
                     run_id="20260101-000000-aa01")
        b = _archive([_llm_pt(ctx=0), _llm_pt(ctx=262144,
                                              prefill_tok_s=403.0)],
                     run_id="20260101-010000-bb02")
        text = mcp_server.render_compare([a, b], "code", _META)
        self.assertIn("对比测速 · 代码生成", text)
        self.assertEqual(text.count("256K"), 1)     # 一行，不是两行
        self.assertNotIn("254.5K", text)
        self.assertIn("403.0", text)
        self.assertIn("aa01", text)
        self.assertIn("bb02", text)

    def test_compare_missing_and_failed_cells(self):
        a = _archive([_llm_pt(ctx=0), _llm_pt(ctx=4096)],
                     run_id="20260101-000000-aa01")
        b = _archive([_llm_pt(ctx=0), _llm_pt(ctx=4096, all_ok=False)],
                     run_id="20260101-010000-bb02")
        c = _archive([_llm_pt(ctx=0)], run_id="20260101-020000-cc03")
        text = mcp_server.render_compare([a, b, c], "code", _META)
        self.assertIn("✗", text)      # 失败单元格
        self.assertIn("—", text)      # 缺失单元格

    def test_compare_agent_rows_by_cache_inst(self):
        a = _archive([_agent_pt(cache=0, inst=64),
                      _agent_pt(cache=4096, inst=64)],
                     scenarios=("agent",), run_id="20260101-000000-aa01")
        b = _archive([_agent_pt(cache=0, inst=64, ttft_s=1.1)],
                     scenarios=("agent",), run_id="20260101-010000-bb02")
        text = mcp_server.render_compare([a, b], "agent", _META)
        self.assertIn("指令", text)
        self.assertIn("4K", text)
        self.assertIn("1.1", text)

    def test_compare_media_rtf_columns(self):
        a = _archive([_asr_pt(sec=30)], scenarios=("asr",),
                     run_id="20260101-000000-aa01")
        b = _archive([_asr_pt(sec=30, rtf=0.5)], scenarios=("asr",),
                     run_id="20260101-010000-bb02")
        text = mcp_server.render_compare([a, b], "asr", _META)
        self.assertIn("RTF", text)
        self.assertIn("倍速", text)
        self.assertIn("0.5", text)


# ---------------------------------------------------------------------------
# 2. 工具层端到端（mock 网关 + 真实 server）
# ---------------------------------------------------------------------------

class _ServerPair:
    """进程内服务器对：mock 网关 + 真实测速台服务（非 TestCase，不参与收集）。

    TestToolsEndToEnd 与 TestStdioHandshake 各挂一次、各起各的一对（unittest 对
    每个测试类分别调 setUpClass）。原先 stdio 类**继承**工具层类，把 8 个工具层
    用例连带跑第二遍；架设逻辑抽到这里后两个类只装自己的用例。
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="llmspeed-mcp-")
        # mock 网关
        cls.mock_port = _free_port()
        cls.mock = uvicorn.Server(uvicorn.Config(
            _StripMockHeader(mock_server.app), host="127.0.0.1", port=cls.mock_port,
            log_level="warning", loop="asyncio"))   # loop=asyncio：见 test_e2e_mock 段错误教训
        threading.Thread(target=cls.mock.run, daemon=True).start()
        # 真实测速台服务：配置指向临时 config.json（seed 一个挂 mock 网关的 provider）
        cls._cfg_path, cls._results = server.CONFIG_PATH, server.RESULTS_DIR
        server.CONFIG_PATH = os.path.join(cls.tmp, "config.json")
        server.RESULTS_DIR = os.path.join(cls.tmp, "results")
        with open(server.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"providers": [{
                "name": "mockp", "label": "Mock Provider",
                "gateway_url": f"http://127.0.0.1:{cls.mock_port}",
                "local": True,
                "deployments": [{
                    "label": "测试部署", "models": ["mock-llm-7b"],
                    "quants": ["Q8_0"], "kv_quants": ["Q8_0"],
                    "hardware": "RTX", "framework": "mock",
                    "params": "-c 8192", "max_ctx": 8192, "active": True,
                }],
            }]}, f)
        cls.srv_port = _free_port()
        cls.srv = uvicorn.Server(uvicorn.Config(
            server.app, host="127.0.0.1", port=cls.srv_port,
            log_level="warning", loop="asyncio"))
        threading.Thread(target=cls.srv.run, daemon=True).start()
        deadline = time.time() + 10
        while not (cls.mock.started and cls.srv.started) and time.time() < deadline:
            time.sleep(0.05)
        if not (cls.mock.started and cls.srv.started):
            raise RuntimeError("测试服务器启动失败")
        cls._base = mcp_server.BASE
        mcp_server.BASE = f"http://127.0.0.1:{cls.srv_port}"

    @classmethod
    def tearDownClass(cls):
        mcp_server.BASE = cls._base
        server.CONFIG_PATH, server.RESULTS_DIR = cls._cfg_path, cls._results
        cls.srv.should_exit = True
        cls.mock.should_exit = True
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestToolsEndToEnd(_ServerPair, unittest.IsolatedAsyncioTestCase):
    """起真服务器打真 HTTP：工具函数逐个过。server.CONFIG_PATH/RESULTS_DIR
    指向临时目录，不碰工作区真实配置与存档。"""

    async def test_list_providers(self):
        text = await mcp_server.list_providers()
        self.assertIn("mockp", text)
        self.assertIn("Mock Provider", text)
        self.assertIn("测试部署", text)
        self.assertIn("mock-llm-7b", text)
        self.assertIn("max_ctx 8192", text)

    async def test_list_models(self):
        text = await mcp_server.list_models("mockp")
        self.assertIn("mock-llm-7b", text)
        self.assertIn("mock-llm-14b", text)

    async def test_list_scenarios(self):
        text = await mcp_server.list_scenarios()
        self.assertIn("creative", text)
        self.assertIn("创意写作", text)
        self.assertIn("agent", text)

    async def test_error_surface_unreachable(self):
        old = mcp_server.BASE
        mcp_server.BASE = f"http://127.0.0.1:{_free_port()}"   # 无服务端口
        try:
            with self.assertRaisesRegex(RuntimeError, "无法连接"):
                await mcp_server.list_providers()
        finally:
            mcp_server.BASE = old

    async def test_api_error_mapping(self):
        """_api 错误面：未知 provider 的模型列表 / 未知 run 的 404 / 活跃或不
        存在 run 的结果请求，一律翻成 RuntimeError 上交（404、「服务端拒绝」
        文案透传）。原三条同测 _api 错误映射，合并为 subTest 参数化。"""
        cases = [
            # (标签, 期望消息片段或 None=原断言只查异常类型, 工具函数, 参数)
            ("list_models 未知 provider", None,
             mcp_server.list_models, "no-such-provider"),
            ("bench_stop 未知 run 404", "404",
             mcp_server.bench_stop, "20260101-000000-zz99"),
            ("bench_result 活跃或不存在的 run", "服务端拒绝",
             mcp_server.bench_result, "20260101-000000-zz99"),
        ]
        for label, pattern, fn, arg in cases:
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, pattern or ""):
                    await fn(arg)

    async def _run_tiny_bench(self) -> str:
        """最小矩阵起测并等到存档落盘，返回 run_id（mock 网关秒级完成）。"""
        text = await mcp_server.bench_start(
            provider="mockp", models=["mock-llm-7b"], scenarios=["creative"],
            ctx_list=[0], concurrencies=[1], max_tokens=64, repeats=1,
            timeout_s=30, note="mcp-e2e")
        self.assertIn("已启动测速", text)
        run_id = text.split("`")[1]
        deadline = time.time() + 60
        while time.time() < deadline:
            status = await mcp_server.bench_status(run_id)
            if "已完成" in status:
                return run_id
            await asyncio.sleep(0.5)
        self.fail(f"测速未在 60s 内完成，最后状态：{status}")

    async def test_full_flow(self):
        run_id = await self._run_tiny_bench()
        # 数据卡片
        card = await mcp_server.bench_result(run_id)
        self.assertIn("测速数据卡片 · Mock Provider", card)
        self.assertIn("mock-llm-7b", card)
        self.assertIn("创意写作（creative）", card)
        self.assertIn("0K", card)
        self.assertIn("mcp-e2e", card)
        # 卡片不含凭据/内部键
        self.assertNotIn("api_key", card)
        # 历史列表
        hist = await mcp_server.bench_history()
        self.assertIn(run_id, hist)
        self.assertIn("创意写作", hist)
        # 第二轮用于对比
        run_id2 = await self._run_tiny_bench()
        cmp_text = await mcp_server.bench_compare([run_id, run_id2],
                                                  scenario="creative")
        self.assertIn("对比测速 · 创意写作", cmp_text)
        self.assertIn(run_id[-4:], cmp_text)
        self.assertIn(run_id2[-4:], cmp_text)
        self.assertIn("0K", cmp_text)


# ---------------------------------------------------------------------------
# 3. stdio 协议握手（真实线协议，子进程）
# ---------------------------------------------------------------------------

class TestStdioHandshake(_ServerPair, unittest.IsolatedAsyncioTestCase):
    """复用 _ServerPair 的服务器对：以子进程拉起 mcp_server.py 走 stdio JSON-RPC。"""

    def _rpc(self, proc, out_q, msg, expect_id, timeout=20):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = out_q.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    self.fail(f"MCP 子进程提前退出（rc={proc.returncode}）")
                continue
            resp = json.loads(line)
            if resp.get("id") == expect_id:
                return resp
        self.fail(f"等待 id={expect_id} 的响应超时")

    def test_stdio_handshake_and_tools(self):
        env = dict(os.environ)
        env["LLM_SPEED_BASE"] = f"http://127.0.0.1:{self.srv_port}"
        proc = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "..",
                                          "mcp_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env)
        out_q: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [out_q.put(l) for l in proc.stdout],
                         daemon=True).start()
        try:
            # initialize
            r = self._rpc(proc, out_q, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "mcp-test", "version": "0.1"}}}, 1)
            self.assertEqual(r["result"]["serverInfo"]["name"], "llm-speed")
            proc.stdin.write(json.dumps({
                "jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            proc.stdin.flush()
            # tools/list
            r = self._rpc(proc, out_q, {"jsonrpc": "2.0", "id": 2,
                                        "method": "tools/list", "params": {}}, 2)
            names = {t["name"] for t in r["result"]["tools"]}
            for t in ("list_providers", "list_models", "list_scenarios",
                      "bench_start", "bench_status", "bench_result",
                      "bench_stop", "bench_history", "bench_compare"):
                self.assertIn(t, names)
            # tools/call：list_providers 打到真实服务
            r = self._rpc(proc, out_q, {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "list_providers", "arguments": {}}}, 3)
            content = r["result"]["content"]
            self.assertFalse(r["result"].get("isError"))
            self.assertEqual(content[0]["type"], "text")
            self.assertIn("mockp", content[0]["text"])
        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
