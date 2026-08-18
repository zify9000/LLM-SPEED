"""配置写入接口回归（ADR-0001）：PUT /api/config 校验与原子写、PUT /api/config/key 凭据单向上行。

测试内把 server.BASE/CONFIG_PATH 指向临时目录，不触碰真实 config.json/.env。
"""
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

import server


def _prov(**kw):
    p = {"name": "p1", "label": "P1", "gateway_url": "http://127.0.0.1:4000",
         "local": False, "deployments": []}
    p.update(kw)
    return p


class TestConfigApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._base, self._cfg = server.BASE, server.CONFIG_PATH
        self._global_key = server.GLOBAL_API_KEY
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.GLOBAL_API_KEY = ""
        self.client = TestClient(server.app)

    def tearDown(self):
        server.BASE, server.CONFIG_PATH = self._base, self._cfg
        server.GLOBAL_API_KEY = self._global_key
        os.environ.pop("API_KEY_P1", None)
        self.tmp.cleanup()

    def test_roundtrip(self):
        dep = {"label": "部署A", "models": ["m1", "m2"], "quant": "Q8_0",
               "hardware": "RTX", "framework": "llama.cpp", "params": "-c 4096",
               "max_ctx": 4096}
        r = self.client.put("/api/config", json={"providers": [_prov(local=True, deployments=[dep])]})
        self.assertEqual(r.status_code, 200)
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["max_ctx"], 4096)
        got = self.client.get("/api/config").json()
        self.assertEqual(got["providers"][0]["name"], "p1")
        self.assertEqual(got["providers"][0]["local"], True)
        self.assertEqual(got["providers"][0]["deployments"][0]["quant"], "Q8_0")
        self.assertNotIn("api_key", json.dumps(on_disk))

    def test_invalid_rejected(self):
        bad = [
            {"providers": []},                                          # 空列表
            {"providers": [_prov(name="名字")]},                         # 非法名称
            {"providers": [_prov(gateway_url="localhost:4000")]},       # 缺协议
            {"providers": [_prov(), _prov()]},                          # 名称重复
            {"providers": ["not-a-dict"]},
        ]
        for body in bad:
            self.assertEqual(self.client.put("/api/config", json=body).status_code, 400, body)

    def test_max_ctx_sanitize(self):
        r = self.client.put("/api/config", json={"providers": [
            _prov(deployments=[{"label": "x", "models": ["m"], "max_ctx": 3.7}]),
            _prov(name="p2", deployments=[{"label": "y", "models": ["m"], "max_ctx": 0}]),
        ]})
        self.assertEqual(r.status_code, 200)
        deps = json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"]
        self.assertEqual(deps[0]["deployments"][0]["max_ctx"], 3)       # 3.7 → int
        self.assertNotIn("max_ctx", deps[1]["deployments"][0])          # 0 视为未配置

    def test_gateway_urls_normalize(self):
        """多地址：gateway_urls 规范化（去重、首选同步 gateway_url）；旧单地址字段兼容。"""
        r = self.client.put("/api/config", json={"providers": [
            _prov(gateway_urls=["http://a:1", "http://b:2", "http://a:1"]),
            _prov(name="p2", gateway_url="http://legacy:3"),
        ]})
        self.assertEqual(r.status_code, 200)
        provs = json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"]
        self.assertEqual(provs[0]["gateway_urls"], ["http://a:1", "http://b:2"])   # 去重
        self.assertEqual(provs[0]["gateway_url"], "http://a:1")                    # 首选=首个
        self.assertEqual(provs[1]["gateway_urls"], ["http://legacy:3"])            # 旧字段迁移
        got = self.client.get("/api/config").json()["providers"]
        self.assertEqual(got[0]["gateway_urls"], ["http://a:1", "http://b:2"])

    def test_gateway_urls_invalid(self):
        bad = [
            {"providers": [_prov(gateway_urls="not-a-list")]},               # 非数组
            {"providers": [_prov(gateway_urls=[])]},                         # 空数组
            {"providers": [_prov(gateway_urls=[] , gateway_url="")]},        # 两字段皆空
            {"providers": [_prov(gateway_urls=["localhost:1"])]},            # 缺协议
            {"providers": [_prov(gateway_urls=[f"http://h{i}:1" for i in range(9)])]},  # 超 8 个
        ]
        for body in bad:
            self.assertEqual(self.client.put("/api/config", json=body).status_code, 400, body)

    def test_key_write_and_clear(self):
        self.client.put("/api/config", json={"providers": [_prov()]})
        r = self.client.put("/api/config/key", json={"provider": "p1", "key": "sk-abc#1"})
        self.assertEqual(r.status_code, 200)
        env = open(os.path.join(self.tmp.name, ".env"), encoding="utf-8").read()
        self.assertIn('API_KEY_P1="sk-abc#1"', env)                     # 引号包裹，# 不破坏解析
        self.assertTrue(self.client.get("/api/config").json()["providers"][0]["has_key"])
        self.assertEqual(os.environ["API_KEY_P1"], "sk-abc#1")          # 即时生效

        r = self.client.put("/api/config/key", json={"provider": "p1", "clear": True})
        self.assertEqual(r.status_code, 200)
        env = open(os.path.join(self.tmp.name, ".env"), encoding="utf-8").read()
        self.assertNotIn("API_KEY_P1", env)
        self.assertFalse(self.client.get("/api/config").json()["providers"][0]["has_key"])

    def test_key_unknown_provider(self):
        self.assertEqual(self.client.put("/api/config/key",
                         json={"provider": "ghost", "key": "x"}).status_code, 400)


class TestValidateBenchCfg(unittest.TestCase):
    """_validate_bench_cfg 测速矩阵钳制（ADR-0001）：类型/范围/条数不合法一律 400。"""

    def _cfg(self, **kw):
        cfg = {"models": ["m1"], "scenarios": ["creative"],
               "ctx_list": [0, 4096], "concurrencies": [1, 2]}
        cfg.update(kw)
        return cfg

    def test_invalid_clamped_400(self):
        bad = [
            self._cfg(ctx_list=[4 * 1048576 + 1]),          # ctx > 4M
            self._cfg(ctx_list=[-1]),                       # 负档位
            self._cfg(ctx_list=[0] * 17),                   # 超 16 档
            self._cfg(scenarios=["ghost"]),                 # 场景非白名单
            self._cfg(models=[f"m{i}" for i in range(33)]),  # models 超 32
            self._cfg(models=[""]),                         # 空模型名
            self._cfg(concurrencies=[0]),                   # 并发低于下界
            self._cfg(concurrencies=[65]),                  # 并发高于上界
            self._cfg(concurrencies=[1.5]),                 # 并发非整数
        ]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)

    def test_valid_passes(self):
        server._validate_bench_cfg(self._cfg())                     # 常规放行
        server._validate_bench_cfg(self._cfg(ctx_list=[4 * 1048576],
                                             concurrencies=[64]))   # 边界值放行


class _FakeBenchRun:
    """鸭式 BenchRun：只记录装配结果，不启动任何真实测速/网络。"""
    captured: list = []

    def __init__(self, cfg, results_dir=None):
        self.cfg = cfg
        self.results_dir = results_dir
        self.run_id = f"fake-{len(_FakeBenchRun.captured)}"
        self.finished_at = None
        _FakeBenchRun.captured.append(self)


async def _fake_drive(run, provider_name):
    """替代 server._drive：真实实现会 await run.run() 发起测速。"""


class TestBenchStartAssembly(unittest.TestCase):
    """POST /api/bench/start 的 provider 装配：ADR-0003 云端强制单发、
    deployments 命中注入 model_max_ctx/model_info、_CAP 能力记忆经 cfg 下发。
    BenchRun 换为假类只捕获 cfg，不起真实测速与端口。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = {k: getattr(server, k) for k in
                       ("BASE", "CONFIG_PATH", "RESULTS_DIR", "RUNS",
                        "BenchRun", "_drive", "GLOBAL_API_KEY")}
        self._cap = dict(server._CAP)
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        server.RUNS = {}
        server.BenchRun = _FakeBenchRun
        server._drive = _fake_drive
        server.GLOBAL_API_KEY = ""
        server._CAP.clear()
        _FakeBenchRun.captured = []
        self.client = TestClient(server.app)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(server, k, v)
        server._CAP.clear()
        server._CAP.update(self._cap)
        self.tmp.cleanup()

    def _start(self, **kw):
        body = {"provider": "p1", "models": ["m1"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1, 2]}
        body.update(kw)
        return self.client.post("/api/bench/start", json=body)

    def test_cloud_provider_forces_single_concurrency(self):
        self.client.put("/api/config", json={"providers": [_prov(local=False)]})
        r = self._start(concurrencies=[1, 2, 4])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        # 云端不做并发/吞吐测试：强制单发，不信任前端传参（ADR-0009）
        self.assertEqual(cfg["concurrencies"], [1])
        self.assertFalse(cfg["provider_local"])
        self.assertEqual(cfg["provider_label"], "P1")
        self.assertEqual(cfg["gateway_url"], "http://127.0.0.1:4000")   # 服务端注入
        self.assertEqual(cfg["api_key"], "")
        self.assertNotIn("thinking_unsupported", cfg)   # 无能力记忆时不下发

    def test_deployments_inject_model_ctx_and_info(self):
        dep = {"label": "部署A", "models": ["m1"], "quant": "Q8_0",
               "hardware": "RTX 4090", "framework": "llama.cpp",
               "params": "-c 8192", "max_ctx": 8192}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        r = self._start(models=["m1", "m2"])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_max_ctx"], {"m1": 8192})   # m2 未命中部署映射
        self.assertEqual(cfg["model_info"]["m1"]["hardware"], "RTX 4090")
        self.assertEqual(cfg["model_info"]["m1"]["deploy_label"], "部署A")
        self.assertEqual(cfg["model_info"]["m1"]["quant"], "Q8_0")
        self.assertNotIn("m2", cfg["model_info"])
        self.assertEqual(cfg["concurrencies"], [1, 2])   # 本地保留前端并发档位

    def test_cap_memory_seeded_into_cfg(self):
        self.client.put("/api/config", json={"providers": [_prov()]})
        server._CAP["p1"] = {"thinking_unsupported": True, "temperature_locked": ["m1"]}
        r = self._start()
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertTrue(cfg["thinking_unsupported"])
        self.assertEqual(cfg["temperature_locked"], ["m1"])


class TestAtomicWrite(unittest.TestCase):
    """_atomic_write：原子替换不留残渣、失败保旧文件、凭据文件权限收紧。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, name="a.json"):
        return os.path.join(self.tmp.name, name)

    def test_no_tmp_residue(self):
        p = self._path()
        server._atomic_write(p, '{"a": 1}')
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"a": 1}')
        self.assertEqual(os.listdir(self.tmp.name), ["a.json"])   # 无 .tmp 残留

    def test_replace_failure_keeps_old_and_cleans_tmp(self):
        p = self._path()
        server._atomic_write(p, "old")
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                server._atomic_write(p, "new")
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "old")                     # 旧文件完好
        self.assertEqual(os.listdir(self.tmp.name), ["a.json"])   # tmp 已清理

    def test_private_new_file_0600(self):
        p = self._path(".env")
        server._atomic_write(p, "K=V\n", private=True)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)   # 新建收紧
        os.chmod(p, 0o644)
        server._atomic_write(p, "K=W\n", private=True)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o644)   # 已存在沿用原权限


class TestHistoryApi(unittest.TestCase):
    """历史记录端点：路径穿越防护 detail/delete 对称、单删/清空只动 RESULTS_DIR 内 json。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        server.RESULTS_DIR = self.tmp.name
        self.client = TestClient(server.app)
        for rid in ("r-one", "r-two"):
            with open(os.path.join(self.tmp.name, f"{rid}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"run_id": rid, "results": []}, f)
        with open(os.path.join(self.tmp.name, "keep.txt"), "w") as f:
            f.write("x")   # 非 json 文件不受清空影响

    def tearDown(self):
        server.RESULTS_DIR = self._results
        self.tmp.cleanup()

    def test_traversal_rejected_symmetrically(self):
        for method in (self.client.get, self.client.delete):
            r = method("/api/bench/history/a..b")
            self.assertEqual(r.status_code, 400, r.text)   # 含 .. 对称拒绝
            r = method("/api/bench/history/ghost")
            self.assertEqual(r.status_code, 404, r.text)   # 合法但不存在

    def test_detail_returns_archive(self):
        r = self.client.get("/api/bench/history/r-one")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["run_id"], "r-one")

    def test_delete_one_keeps_others(self):
        r = self.client.delete("/api/bench/history/r-one")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "r-one.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "r-two.json")))

    def test_clear_removes_all_json(self):
        r = self.client.delete("/api/bench/history")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(os.listdir(self.tmp.name), ["keep.txt"])   # json 全清、其他保留


if __name__ == "__main__":
    unittest.main()
