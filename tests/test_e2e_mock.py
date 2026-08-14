"""端到端回归：mock 网关（进程内 uvicorn）+ 引擎最小矩阵全链路。

覆盖：cfg 事件首发且脱敏、RTT 净口径与 mock 设定一致（±20%）、双并发总吞吐、
mock 运行不落盘。运行：python3 -m unittest tests.test_e2e_mock -v
"""
import asyncio
import os
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
from bench import BenchRun  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
        self.assertEqual(types[0], "cfg", "cfg 事件必须首发（ADR-0022）")
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

        # 双并发总吞吐 ≈ 2 × 单请求 decode
        s1 = by_key[(4096, 1)]["decode_tok_s"]
        t2 = by_key[(4096, 2)]["decode_total_tok_s"]
        self.assertIsNotNone(t2, "decode 重叠充分时必须产出总吞吐")
        self.assertTrue(1.5 * s1 <= t2 <= 2.6 * s1, f"总吞吐 {t2} 应 ≈ 2×{s1}")

        # mock 运行不落历史记录（ADR-0010）
        self.assertTrue(run.mock_seen)
        self.assertFalse(os.path.isdir("/tmp/llmspeed-e2e"))


if __name__ == "__main__":
    unittest.main()
