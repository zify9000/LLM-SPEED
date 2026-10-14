"""e2e 组的共享 harness：进程内 mock 网关 + 引擎最小矩阵的跑批基建。

从 tests/test_e2e_mock.py 复制提取（原文件已删），供 tests/e2e/ 下的用例
复用。要点：

- MOCK_* 播种与 `import mock_server` 的**顺序约束**（本模块顶部，务必先
  import tests._shared 再 import mock_server；测试文件自身也一样——反序时
  mock_server 会采到默认值而非播种值，如 PP=1200 而非 3000，断言口径全错）。
- MockGatewayTest 基类：进程内 uvicorn 线程拉起 mock_server.app，网关根地址
  暴露为 cls.base_url；调用方从仓库根目录运行（unittest / pytest 双入口）。
- _RawServer / _start_raw_die_server：原始 socket 假网关，用来模拟「裸断连」。
- std_cfg：标准 cfg 工厂（gateway_url 自动填 mock 网关）；_drain：订阅队列
  排水到 None 哨兵的 helper。
"""
import asyncio
import re
import socket
import threading
import time
import unittest

# 顺序是硬约束（原 test_e2e_mock.py 顶部同一约定）：mock_server 在 import 时读取
# MOCK_* 环境变量，而播种落在 tests._shared 的模块级——先 import 它（触发播种），
# 再 import mock_server。反过来播种就晚了：开发者 shell 里残留的 MOCK_* 导出
# 会污染断言口径（本该被播种覆盖回与各方一致的值）。
import tests._shared                    # noqa: E402,F401  —— 触发 MOCK_* 播种
from tests._shared import _free_port    # noqa: E402

import mock_server                      # noqa: E402
import uvicorn                          # noqa: E402
from bench import BenchRun              # noqa: E402


class _RawServer:
    """原始 socket 服务句柄：持有监听 socket 与停摆事件。

    socket.socket 是 C 类型、不允许挂自定义属性，故用薄包装把 stop 事件
    与监听 fd 一起交给调用方（`with` 退出即关停，见 close()）。
    """
    def __init__(self, sock: socket.socket, stop: threading.Event):
        self.sock = sock
        self.stop = stop

    def close(self) -> None:
        """先置 stop 唤醒 accept() 轮询线程，再关监听 fd。

        顺序要紧——只 close() 的话线程要等下一次 accept() 抛 OSError 才返回，
        期间仍持有已关闭 fd，白占一个线程到进程退出。"""
        self.stop.set()
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _start_raw_die_server(port: int, n_chunks: int = 5) -> _RawServer:
    """原始 socket HTTP 服务（模拟裸断连网关，绕过 ASGI 的干净收口）：
    GET 返回模型列表（RTT 基线与 /v1 前缀探测用）；POST chat 流出 n_chunks 个
    SSE 块后，chunked 流不发终止块直接 FIN——httpx 判「不完整 chunked 读」。
    新版 uvicorn/starlette 会把 StreamingResponse 生成器异常干净收口（客户端
    视作正常结束），mock 进程内已无法模拟断连，故用原始 socket。

    线程收口：accept() 带超时轮询 + stop 事件——旧实现 `while True: accept()`
    永久阻塞，调用方 close() 后该线程仍停在已关闭 fd 的 accept() 上活到进程
    退出。**这不是全量测试段错误的主因**（主因见 MockGatewayTest.setUpClass
    的 uvloop 说明），但同样是不该留的收口缺陷：线程要干净退出，不靠进程
    结束兜底。现在关闭时置 stop 并唤醒，线程在 0.2s 内退出。"""
    stop = threading.Event()

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
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue      # 定期醒来查 stop（不能死等，否则关不掉）
            except OSError:
                return        # 监听 socket 已被关闭
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(8)
    sock.settimeout(0.2)       # accept() 轮询周期——关停延迟上限
    threading.Thread(target=_loop, daemon=True).start()
    return _RawServer(sock, stop)


# ---------------------------------------------------------------------------
# 跑批 helper
# ---------------------------------------------------------------------------

def std_cfg(gateway_url, **overrides) -> dict:
    """标准 cfg 工厂：原 test_e2e_mock.py 内联重复最多的那份（~45 例）基准形态。

    ctx_list / concurrencies / max_tokens / scenarios 等按用例需要传 overrides
    覆盖；reply_mode、cache="bust"、agent 矩阵档等也走 overrides。
    """
    cfg = {
        "gateway_url": gateway_url,
        "api_key": "sk-should-not-leak",
        "models": ["mock-llm-7b"],
        "scenarios": ["creative"],
        "ctx_list": [0],
        "concurrencies": [1],
        "max_tokens": 64,
        "repeats": 1,
        "timeout_s": 30,
        "thinking": "disabled",
    }
    cfg.update(overrides)
    return cfg


async def _drain(q: asyncio.Queue, task=None, timeout: float = 30) -> list:
    """把订阅队列排水到 None 哨兵（引擎每个订阅队列以 None 结束运行），返回全部事件。

    task 给定时 finally 里 await 它——排水超时或抛错也要把 run 任务收干净
    （原 test_e2e_mock.py 各用例的 try/finally 模式收进这一处）。
    """
    events = []
    try:
        while True:
            ev = await asyncio.wait_for(q.get(), timeout=timeout)
            if ev is None:
                break
            events.append(ev)
    finally:
        if task is not None:
            await task
    return events


class MockGatewayTest(unittest.IsolatedAsyncioTestCase):
    """进程内 mock 网关（uvicorn 线程）跑批基类。

    类属性：cls.port 与 cls.base_url（网关根地址，= 原 test_e2e_mock 里
    f"http://127.0.0.1:{{cls.port}}" 的等价物，helper 与用例都从这里取）。
    """

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        # loop="asyncio" 是**必须**的：uvicorn 默认 loop="auto"，在 server.run()
        # 里调 uvloop.install() 把**全局** EventLoopPolicy 换成 uvloop 的。此后
        # 本进程内所有 IsolatedAsyncioTestCase 新建的 loop 都成了 uvloop，而
        # test_engine_unit 的假时钟用例（mock.patch bench.time + await）在这种
        # 跨线程组合下会让 CPython 在格式化 traceback 时栈行走踩空 → 段错误
        # （实测：原 test_e2e_mock 的 74 例任意一例先跑 + TestStallFlushOne 后跑 = 必崩；
        #  loop="asyncio" 全绿；uvloop 0.22.1 / CPython 3.13.13）。
        # mock 网关只是被本地 httpx 客户端打，用哪个 loop 实现无观测差异。
        cls.server = uvicorn.Server(uvicorn.Config(
            mock_server.app, host="127.0.0.1", port=cls.port, log_level="warning",
            loop="asyncio"))
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

    def std_cfg(self, **overrides) -> dict:
        """标准 cfg 工厂（gateway_url 自动填本类 mock 网关）。"""
        return std_cfg(self.base_url, **overrides)

    async def _drain(self, q: asyncio.Queue, task=None, timeout: float = 30) -> list:
        """实例直通的队列排水 helper（见模块级 _drain）。"""
        return await _drain(q, task, timeout)

    async def _run_and_collect(self, skip_probe=False, **cfg_extra):
        """跑一次最小矩阵（probe 必走），返回 (run, 全部事件列表)。

        skip_probe=True 时预置 cpt 校准跳过探测请求——探测本身也打一次 chat，
        会消耗 mock 的一次性钩子（FLAKY_FAIL/EMPTY_STREAM_FAIL 等），断言
        「第 N 发请求」的用例必须跳过它。"""
        run = BenchRun({
            "gateway_url": self.base_url,
            "models": ["mock-llm-7b"], "scenarios": ["creative"],
            "ctx_list": [0], "concurrencies": [1],
            "max_tokens": 16, "repeats": 1, "timeout_s": 30,
            "thinking": "disabled", **cfg_extra,
        })
        if skip_probe:
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8
        q: asyncio.Queue = asyncio.Queue()
        run.subs.add(q)
        task = asyncio.create_task(run.run())
        events = await _drain(q, task)
        return run, events

    async def _run_and_collect_status(self, thinking="disabled", **cfg_extra):
        """跑一次最小矩阵（probe 必走，检测逻辑全靠它），返回状态消息列表。"""
        _, events = await self._run_and_collect(thinking=thinking, **cfg_extra)
        types = [e.get("type") for e in events]
        self.assertIn("done", types)
        return [e["msg"] for e in events if e.get("type") == "status"]

    async def _run_agent_matrix_collect(self, cpt=1.8, preset=None, **cfg_extra):
        """跑一次 agent 缓存×指令矩阵（预置输入系数跳过探测），返回
        (run, 全部事件列表)。cpt=None 不预置（走探测路径）；preset 为启动前
        对 run 对象的额外播种钩子（如毒药化 prefill 先验曲线）。"""
        run = BenchRun({
            "gateway_url": self.base_url,
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
        events = await _drain(q, task)
        return run, events
