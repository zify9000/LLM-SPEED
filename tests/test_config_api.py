"""配置写入接口回归（ADR-0026）：PUT /api/config 校验与原子写、PUT /api/config/key 凭据单向上行。

测试内把 server.BASE/CONFIG_PATH 指向临时目录，不触碰真实 config.json/.env。
"""
import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
