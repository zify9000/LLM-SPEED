"""历史存档环境编辑端点回归：POST /api/bench/history/{run_id}/env。

只更新 cfg.model_info[model] 的六个环境字段（kinds 等保留、旧 quants 数组
键移除），原子写回；穿越/不存在/损坏 与 note 端点同口径。

测试内把 server.RESULTS_DIR 指向临时目录，不触碰真实 results/。
"""
import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import server


class TestHistoryEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        server.RESULTS_DIR = self.tmp.name
        self.client = TestClient(server.app)
        self.rid = "r-env"
        self.path = os.path.join(self.tmp.name, f"{self.rid}.json")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.rid,
                       "cfg": {"provider": "p1", "models": ["m1", "other"],
                               "model_info": {"m1": {"deploy_label": "旧部署",
                                                     "quant": "Q8_0",
                                                     "kinds": ["llm", "ocr"],
                                                     "hardware": "旧机"}},
                               "note": "备注不动"}},
                      f)

    def tearDown(self):
        server.RESULTS_DIR = self._results
        self.tmp.cleanup()

    def _env(self, model="m1", **info):
        return self.client.post(f"/api/bench/history/{self.rid}/env",
                                json={"model": model, "info": info})

    def test_env_update_persists_and_keeps_other_fields(self):
        """成功更新：基础字段落盘可重读、旧 quants 数组键移除、kinds/note 等
        未提交字段保留、无 .tmp 原子写残渣。"""
        r = self._env(deploy_label="新部署", quant="Q4_K_M", kv_quant="q8_0",
                      hardware="RTX 4090", framework="llama.cpp", params="-c 8192")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["ok"], True)
        self.assertEqual(r.json()["model_info"], {
            "deploy_label": "新部署", "quant": "Q4_K_M", "kv_quant": "q8_0",
            "hardware": "RTX 4090", "framework": "llama.cpp", "params": "-c 8192",
            "kinds": ["llm", "ocr"]})
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        entry = d["cfg"]["model_info"]["m1"]
        self.assertEqual(entry["quant"], "Q4_K_M")
        self.assertEqual(entry["kv_quant"], "q8_0")
        self.assertNotIn("quants", entry)              # 旧数组键不再写出
        self.assertEqual(entry["kinds"], ["llm", "ocr"])   # 未提交字段保留
        self.assertEqual(d["cfg"]["note"], "备注不动")
        self.assertNotIn("other", d["cfg"]["model_info"])   # 其他模型条目不受影响
        self.assertFalse([f for f in os.listdir(self.tmp.name)
                          if f.endswith(".tmp")])       # 原子写无残渣

    def test_env_quant_fields_stripped_and_optional(self):
        """quant/kv_quant 规整：strip、缺省放空字符串（无量化信息）。"""
        r = self._env(quant="  Q8_0  ", kv_quant=None, deploy_label="x",
                      hardware="", framework="", params="")
        self.assertEqual(r.status_code, 200, r.text)
        mi = r.json()["model_info"]
        self.assertEqual(mi["quant"], "Q8_0")
        self.assertEqual(mi["kv_quant"], "")

    def test_env_string_fields_clamped(self):
        """长度上限镜像 _validate_providers：label/hardware/framework/params
        超长截断而非拒绝（64/200/100/300）。"""
        r = self._env(deploy_label="L" * 100, hardware="H" * 250,
                      framework="F" * 150, params="P" * 400)
        self.assertEqual(r.status_code, 200, r.text)
        mi = r.json()["model_info"]
        self.assertEqual(len(mi["deploy_label"]), 64)
        self.assertEqual(len(mi["hardware"]), 200)
        self.assertEqual(len(mi["framework"]), 100)
        self.assertEqual(len(mi["params"]), 300)

    def test_env_max_ctx_written_cleared_preserved(self):
        """max_ctx 三态：正整数落盘（浮点取整）、null 显式清空、省略键保留
        原条目值（旧客户端不回写时不误抹）。"""
        r = self._env(max_ctx=262144)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["model_info"]["max_ctx"], 262144)
        r = self._env(quant="Q4_K_M", max_ctx=196608.7)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["model_info"]["max_ctx"], 196608)   # 浮点取整
        r = self._env(max_ctx=None)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["model_info"]["max_ctx"], None)     # 显式清空
        r = self._env(quant="Q8_0")       # 省略 max_ctx 键 → 保留 null 原值
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["model_info"]["max_ctx"], None)
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["cfg"]["model_info"]["m1"]["max_ctx"], None)

    def test_env_max_ctx_invalid_400(self):
        """max_ctx 非法值（bool/0/负/非数值）拒绝且存档不动，镜像部署校验
        口径。"""
        for bad in (True, False, 0, -5, "262144", [262144]):
            r = self._env(max_ctx=bad)
            self.assertEqual(r.status_code, 400, bad)
            self.assertIn("max_ctx", r.json()["detail"])
        # nan/inf 无法经 TestClient(json=) 上送（httpx 编码器拒绝），改以原始
        # 文本体直发，验证服务端 math.isfinite 校验拦下
        for bad in ("NaN", "Infinity", "-Infinity"):
            r = self.client.post(f"/api/bench/history/{self.rid}/env",
                                 content=f'{{"model":"m1","info":{{"max_ctx":{bad}}}}}',
                                 headers={"Content-Type": "application/json"})
            self.assertEqual(r.status_code, 400, bad)
            self.assertIn("max_ctx", r.json()["detail"])
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["cfg"]["model_info"]["m1"].get("deploy_label"), "旧部署")
        self.assertNotIn("max_ctx", d["cfg"]["model_info"]["m1"])

    def test_env_traversal_and_missing_rejected(self):
        """穿越/不存在/损坏 与 note 端点对称：400/404。"""
        body = {"model": "m1", "info": {"quant": "Q8_0"}}
        for rid in ("a..b", "a\\b"):
            r = self.client.post(f"/api/bench/history/{rid}/env", json=body)
            self.assertEqual(r.status_code, 400, rid)
        r = self.client.post("/api/bench/history/ghost/env", json=body)
        self.assertEqual(r.status_code, 404, r.text)
        with open(os.path.join(self.tmp.name, "broken.json"), "w",
                  encoding="utf-8") as f:
            f.write("{not json")
        r = self.client.post("/api/bench/history/broken/env", json=body)
        self.assertEqual(r.status_code, 404, r.text)
        self.assertIn("损坏", r.json()["detail"])

    def test_env_model_validated(self):
        """model 须为非空字符串且存在于存档 cfg.models，否则 400 且存档不动。"""
        for model in ("ghost-m", "", None, 42):
            r = self.client.post(f"/api/bench/history/{self.rid}/env",
                                 json={"model": model, "info": {"quant": "Q8_0"}})
            self.assertEqual(r.status_code, 400, model)
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["cfg"]["model_info"]["m1"]["quant"], "Q8_0")   # 未被改写

    def test_env_info_invalid_400(self):
        """info 本体与字段类型校验：非对象/字段非字符串/quant·kv_quant 非法
        一律 400。"""
        bad_info = ["not-a-dict", 42,
                    {"quant": 42},                     # 非字符串
                    {"quant": ["Q8_0"]},               # 数组（旧数组格式不再收）
                    {"kv_quant": 42},                  # 非字符串
                    {"quant": "Q" * 65},               # 超 64 拒绝
                    {"kv_quant": "Q" * 65},            # 超 64 拒绝
                    {"quant": "", "hardware": 42},     # 字段非字符串
                    {"quant": "", "params": ["-c 4096"]}]
        for info in bad_info:
            r = self.client.post(f"/api/bench/history/{self.rid}/env",
                                 json={"model": "m1", "info": info})
            self.assertEqual(r.status_code, 400, info)
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["cfg"]["model_info"]["m1"]["deploy_label"], "旧部署")


if __name__ == "__main__":
    unittest.main()
