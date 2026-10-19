"""tests/ 共享设施：MOCK 环境播种、空闲端口、存档 helper、server 目录重定向基类。

用法：供 tests/api 与 tests/e2e 的用例**显式 import**
（`from tests._shared import ...`）。刻意不用 pytest 专有设施（无 conftest.py、
无 fixture）——共享设施就是普通模块 + 显式 import，unittest / pytest 双入口
行为一致：

    python -m unittest discover -s tests        # 分组入口 discover -s tests/<组> 同
    python -m pytest tests -q

两种入口都要求**从仓库根目录运行**：仓库根在 sys.path 上，`tests` 包才可导入；
分组跑的起始目录（unit/api/e2e/mcp_api）也不会遮蔽 bench/server/mock_server
这些真实模块。

MOCK_* 播种必须早于 `import mock_server`（后者在 import 时读环境变量），故本
模块**不**导入 mock_server：调用方先 `import tests._shared`，再 import mock_server
（示例见 tests/e2e/_harness.py）。
"""
import hashlib
import json
import os
import socket
import tempfile
import unittest

# mock_server 在 import 时读取环境变量：必须先于一切可能链式 import 到 mock_server
# 的项目模块生效，并**强制覆盖**——开发者 shell 里残留的 MOCK_* 导出不得污染断言
# 口径；各处同值（与 test_e2e_mock / test_mcp_server 顶部两段逐字重复的播种一致），
# 防 pytest 同进程下「先 import 者生效」造成的口径分裂。
os.environ["MOCK_PP"] = "3000"
os.environ["MOCK_TG"] = "300"
os.environ["MOCK_CPT"] = "1.8"

from fastapi.testclient import TestClient      # noqa: E402

import bench                                   # noqa: E402
import server                                  # noqa: E402


# ---------------------------------------------------------------------------
# 端口
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """向内核要一个空闲端口（bind(0) 取号后立即释放），给进程内 uvicorn 用。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# 合成存档（test_server_guard / test_history_edit / test_anomaly_text 三处
# 近亲复制的合并；_point 逐字节相同）
# ---------------------------------------------------------------------------

def _point(**kw):
    """最小 llm 测点骨架；**kw 覆盖任意键（ctx_target / reqs / all_ok / ...）。"""
    p = {"model": "m1", "scenario": "code", "kind": "llm", "ctx_target": 4096,
         "concurrency": 1, "reqs": [{"req": 0, "err": None}], "all_ok": True,
         "ttft_s": 1.0}
    p.update(kw)
    return p


def write_archive(rid, points=None, metric_version=None, **cfg_extra):
    """写一份存档到 server.RESULTS_DIR/<rid>.json，返回写出的 dict。

    points=None 时用 [_point()] 单点（test_server_guard 的形态）。
    metric_version=None 表示**不写该字段**（= 遗留口径存档，与本机制引入前的
    存量形态一致）；显式传值才写顶层版本。
    cfg_extra 覆盖 cfg 任意键（provider / models / scenarios / ctx_list / ...）。
    """
    cfg = {"provider": "p1", "models": ["m1"], "scenarios": ["code"],
           "ctx_list": [4096], "concurrencies": [1], "max_tokens": 512,
           "repeats": 1}
    cfg.update(cfg_extra)
    d = {"run_id": rid, "started_at": "2026-09-16 22:00:00",
         "cfg": cfg, "results": points if points is not None else [_point()]}
    if metric_version is not None:
        d["metric_version"] = metric_version
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"), "w",
              encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def read_archive(rid):
    """读回 server.RESULTS_DIR/<rid>.json。"""
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _side(rid):
    """异常输入全文侧车文件路径（<run_id>.anomaly.txt）。"""
    return os.path.join(server.RESULTS_DIR, f"{rid}{bench.ANOMALY_TEXT_SUFFIX}")


def _put_sidecar(rid, data: bytes):
    with open(_side(rid), "wb") as f:
        f.write(data)


def _ref_for(rid, data: bytes, text: str):
    """侧车引用记录（文件名字节切片 + sha1），与引擎落档同构。"""
    return {"file": f"{rid}{bench.ANOMALY_TEXT_SUFFIX}", "off": 0,
            "len": len(data), "chars": len(text),
            "sha1": hashlib.sha1(data).hexdigest()}


# ---------------------------------------------------------------------------
# 环境变量保存 / 还原
# ---------------------------------------------------------------------------

_UNSET = object()


def snapshot_env(*names) -> dict:
    """记下指定环境变量的原值（含「原本未设置」哨兵），供 restore_env 还原。"""
    return {n: os.environ.get(n, _UNSET) for n in names}


def restore_env(snapshot: dict) -> None:
    """把 snapshot_env 的结果还原回去：原本未设置的删除，其余回写原值。

    比散落各处的 `os.environ.pop(k, None)` 严谨——后者会把 shell 里本来就有的
    值一并抹掉（现有 fixture 的泄漏缺陷，见 test_config_api / test_server_guard）。
    """
    for name, old in snapshot.items():
        if old is _UNSET:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class ServerDirCase(unittest.TestCase):
    """把 server 的工作目录 / 配置路径 / 存档目录重定向进临时目录的基类。

    - REDIRECT_CONFIG=True（默认）：BASE / CONFIG_PATH / RESULTS_DIR 全指向 tmp；
      REDIRECT_CONFIG=False：只重定向 RESULTS_DIR（= test_history_env.py 的现状
      ——只碰存档、不碰配置）。RESULTS_DIR 一律取 <tmp>/results 子目录（原文件
      里有直接用 tmp 根的写法，等价，只是路径形态统一到子目录）。
    - 保存并把 server.BASE / CONFIG_PATH / RESULTS_DIR 重定向进临时目录，
      清空 server._HIST_CACHE（历史列表摘要缓存，跨用例不清会串档）；tearDown
      恢复原值并再次清缓存。tmp 由 setUp 里的 addCleanup 收尾。
    - self.tmp 是 TemporaryDirectory（self.tmp.name 即临时目录路径），
      self.client 是挂 server.app 的 TestClient（四个原 fixture 同款）。
    - server.GLOBAL_API_KEY 一并保存并置空（原 _GuardBase 与 TestConfigApi 各写
      一份的上收），tearDown 还原。
    - 需要指定环境变量（API_KEY_*/GLOBAL_API_KEY 等）的用例在 setUp 里调
      self.save_env("API_KEY_P1", ...)：保存原值并自动在用例结束后还原。
    """
    REDIRECT_CONFIG = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._srv_paths = (server.BASE, server.CONFIG_PATH, server.RESULTS_DIR)
        if self.REDIRECT_CONFIG:
            server.BASE = self.tmp.name
            server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        os.makedirs(server.RESULTS_DIR, exist_ok=True)
        self._srv_key = server.GLOBAL_API_KEY
        server.GLOBAL_API_KEY = ""
        server._HIST_CACHE.clear()
        # 运行注册表也一并隔离：用例常往里塞**假 run**（"有测速在跑"这类断言），
        # 残留会让后面任意一个起测速的用例撞上那个假对象（真机踩过：跨文件残留的
        # SimpleNamespace 让 mcp 全链路用例拿到 500，排查时根本想不到源头在别的文件）
        self._srv_runs = dict(server.RUNS)
        server.RUNS.clear()
        self.client = TestClient(server.app)

    def tearDown(self):
        server.BASE, server.CONFIG_PATH, server.RESULTS_DIR = self._srv_paths
        server.GLOBAL_API_KEY = self._srv_key
        server._HIST_CACHE.clear()
        server.RUNS.clear()
        server.RUNS.update(self._srv_runs)

    def save_env(self, *names):
        """保存环境变量原值，用例结束时自动还原（含还原「原本未设置」）。"""
        self.addCleanup(restore_env, snapshot_env(*names))
