"""SSE 回放与 seq 契约回归（ADR-0009）：GET /api/bench/events/{run_id}。

用 TestClient 读流（进程内 ASGI，不起真实端口），向 server.RUNS 注入鸭式假
run（仅带端点读取的 history/subs/finished_at）验证：
① 收到的事件流以 history 回放为前缀；② seq 单调且 cfg 事件 seq=0；
③ 回放后新事件实时到达；④ 终止帧后连接正常关闭；⑤ 空闲心跳 ': ping' 注释帧。

注意：starlette 0.38 的 TestClient 会把整个流缓冲到连接关闭后才交付
（stream().__enter__ 阻塞至生成器结束），故「回放后到达的新事件/终止帧」
由生产者线程在订阅注册后经 call_soon_threadsafe 注入，读取侧再断言完整流。

运行：python3 -m unittest tests.test_sse_replay -v
"""
import asyncio
import json
import threading
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import server


class _Subs(set):
    """订阅集合：捕获连接注册的队列与其所在事件循环，
    供生产者线程用 call_soon_threadsafe 安全注入事件。"""

    def __init__(self):
        super().__init__()
        self.ready = threading.Event()
        self.loop = None
        self.q = None

    def add(self, q):
        super().add(q)
        self.q = q
        self.loop = asyncio.get_running_loop()
        self.ready.set()


class _FakeRun:
    """鸭式 run：只带 SSE 端点读取的属性（history/subs/finished_at）。"""

    def __init__(self, history, finished_at=None):
        self.history = history
        self.finished_at = finished_at
        self.subs = _Subs()


def _history():
    """预置回放历史：cfg 首发且 seq=0，后续事件 seq 单调递增。"""
    return [{"type": "cfg", "seq": 0, "cfg": {"models": ["m1"]}},
            {"type": "status", "seq": 1, "msg": "测试 m1"},
            {"type": "point", "seq": 2, "point": {"all_ok": True}}]


def _data(lines):
    """从 SSE 行序列中挑出 data 行并解析为 dict 列表。"""
    return [json.loads(l[len("data: "):]) for l in lines if l.startswith("data: ")]


class TestSseReplay(unittest.TestCase):
    def setUp(self):
        self._runs = server.RUNS
        server.RUNS = {}
        self.client = TestClient(server.app)

    def tearDown(self):
        server.RUNS = self._runs

    def _put_run(self, run, run_id="r1"):
        server.RUNS[run_id] = run
        return f"/api/bench/events/{run_id}"

    def _produce(self, run, events, delay=0.0):
        """生产者线程：订阅注册（可选延时）后按序向订阅队列注入事件。
        返回 (线程, 错误列表)；主线程读完流后须 join 并检查错误。"""
        err = []

        def worker():
            try:
                if not run.subs.ready.wait(timeout=5):
                    raise TimeoutError("订阅未及时注册")
                if delay:
                    time.sleep(delay)
                for ev in events:
                    run.subs.loop.call_soon_threadsafe(run.subs.q.put_nowait, ev)
            except Exception as e:  # noqa: BLE001
                err.append(e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t, err

    def _join_producer(self, t, err):
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "生产者线程未结束")
        self.assertFalse(err, err)

    def test_replay_prefix_seq_live_event_and_terminal(self):
        history = _history()
        run = _FakeRun(history)
        url = self._put_run(run)
        live = {"type": "status", "seq": 3, "msg": "新事件"}
        t, err = self._produce(run, [live, None])   # 回放后到达的新事件 + 终止帧
        with self.client.stream("GET", url) as resp:
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/event-stream", resp.headers["content-type"])
            lines = list(resp.iter_lines())   # ④ None 终止帧后连接正常关闭（流读完）
        self._join_producer(t, err)
        got = _data(lines)
        # ① 事件流以 history 回放为前缀；③ 回放后注入的新事件实时到达
        self.assertEqual(got[:len(history)], history)
        self.assertEqual(got[len(history):], [live])
        # ② seq 单调且 cfg 类事件 seq=0
        self.assertEqual([e["seq"] for e in got], [0, 1, 2, 3])
        self.assertEqual(got[0]["type"], "cfg")

    def test_finished_run_replays_then_closes(self):
        """已结束的 run（finished_at 非空）：回放完 history 即关闭连接，不空转心跳。"""
        history = _history()
        run = _FakeRun(history, finished_at=1234567890.0)
        url = self._put_run(run, "r2")
        with self.client.stream("GET", url) as resp:
            self.assertEqual(resp.status_code, 200)
            lines = list(resp.iter_lines())   # ④ 回放末尾即终止，连接正常关闭
        self.assertEqual(_data(lines), history)
        self.assertFalse(any(l == ": ping" for l in lines), "已结束运行不应有心跳空转")

    def test_heartbeat_comment_frame(self):
        """空闲连接每 15s 发 ': ping' 注释帧（探活/防代理缓冲）；超时缩短后断言收到。"""
        history = _history()[:1]
        run = _FakeRun(history)
        url = self._put_run(run, "r3")
        orig_wait_for = asyncio.wait_for

        async def fast_wait_for(aw, timeout):
            return await orig_wait_for(aw, 0.2 if timeout == 15 else timeout)

        with mock.patch.object(asyncio, "wait_for", fast_wait_for):
            t, err = self._produce(run, [None], delay=0.6)   # 让心跳跑几拍再终止
            with self.client.stream("GET", url) as resp:
                self.assertEqual(resp.status_code, 200)
                lines = list(resp.iter_lines())
        self._join_producer(t, err)
        self.assertEqual(_data(lines), history)   # 回放仍是事件流前缀
        pings = [l for l in lines if l == ": ping"]
        self.assertGreaterEqual(len(pings), 1, "未收到心跳注释帧")


if __name__ == "__main__":
    unittest.main()
