"""api 组：历史存档端点契约（路径穿越防护 detail/delete 对称、单删/清空只动
RESULTS_DIR 内 json、改名重写全部模型名引用位并留痕）。

自 tests/test_config_api.py 拆分：TestHistoryApi 原样搬入，setUp/tearDown 补
server._HIST_CACHE.clear()（历史列表摘要缓存按文件名键控、mtime/size 命中即复用，
跨用例不清会串档）。

    python -m pytest tests/api/test_history_api.py -q
    python -m unittest discover -s tests/api -p test_history_api.py
    python -m unittest tests.api.test_history_api -v      # 单模块入口
（两种入口都从仓库根目录运行。）
"""
import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import server


class TestHistoryApi(unittest.TestCase):
    """历史记录端点：路径穿越防护 detail/delete 对称、单删/清空只动 RESULTS_DIR 内 json。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        server.RESULTS_DIR = self.tmp.name
        server._HIST_CACHE.clear()
        self.client = TestClient(server.app)
        for rid in ("r-one", "r-two"):
            with open(os.path.join(self.tmp.name, f"{rid}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"run_id": rid, "results": []}, f)
        with open(os.path.join(self.tmp.name, "keep.txt"), "w") as f:
            f.write("x")   # 非 json 文件不受清空影响

    def tearDown(self):
        server.RESULTS_DIR = self._results
        server._HIST_CACHE.clear()
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

    def _write_rename_archive(self, rid="r-ren"):
        """含全部模型名引用位的存档：cfg.models/model_info/model_kind(s)/
        model_max_ctx/temperature_locked + 逐点 model 字段。"""
        d = {"run_id": rid, "started_at": "2026-09-11 09:00:00",
             "cfg": {"provider": "p1", "models": ["old-m", "other"],
                     "model_info": {"old-m": {"hardware": "RTX"}},
                     "model_kind": {"old-m": "llm", "other": "llm"},
                     "model_kinds": {"old-m": ["llm"], "other": ["llm"]},
                     "model_max_ctx": {"old-m": 8192},
                     "temperature_locked": ["old-m"]},
             "results": [{"model": "old-m", "scenario": "creative"},
                         {"model": "other", "scenario": "creative"},
                         {"model": "old-m", "scenario": "code"}]}
        with open(os.path.join(self.tmp.name, f"{rid}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f)
        return rid

    def test_rename_rewrites_all_references(self):
        """改名适配（本地部署改名后存档对不上号）：所有引用位改写为新名，
        顶层追加 model_renames 审计留痕；重复改名留痕追加而非覆盖。"""
        rid = self._write_rename_archive()
        r = self.client.post(f"/api/bench/history/{rid}/rename",
                             json={"old": "old-m", "new": "new-m"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["points"], 2)
        with open(os.path.join(self.tmp.name, f"{rid}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        cfg = d["cfg"]
        self.assertEqual(cfg["models"], ["new-m", "other"])
        self.assertEqual(list(cfg["model_info"]), ["new-m"])
        self.assertEqual(cfg["model_kind"]["new-m"], "llm")
        self.assertEqual(cfg["model_kinds"]["new-m"], ["llm"])
        self.assertEqual(cfg["model_max_ctx"], {"new-m": 8192})
        self.assertEqual(cfg["temperature_locked"], ["new-m"])
        self.assertEqual([p["model"] for p in d["results"]],
                         ["new-m", "other", "new-m"])
        self.assertEqual(d["model_renames"][0]["from"], "old-m")
        self.assertEqual(d["model_renames"][0]["to"], "new-m")
        self.assertTrue(d["model_renames"][0]["at"])
        r = self.client.post(f"/api/bench/history/{rid}/rename",
                             json={"old": "new-m", "new": "newer-m"})
        self.assertEqual(r.status_code, 200, r.text)
        with open(os.path.join(self.tmp.name, f"{rid}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual([x["to"] for x in d["model_renames"]],
                         ["new-m", "newer-m"])

    def test_rename_rejected(self):
        """改名入参防护：穿越/不存在 与 detail 对称；old 不在存档、new 撞名、
        新旧同名、空值一律 400 且存档不动。"""
        rid = self._write_rename_archive()
        r = self.client.post("/api/bench/history/a..b/rename",
                             json={"old": "a", "new": "b"})
        self.assertEqual(r.status_code, 400, r.text)
        r = self.client.post("/api/bench/history/ghost/rename",
                             json={"old": "a", "new": "b"})
        self.assertEqual(r.status_code, 404, r.text)
        for body in ({"old": "ghost-m", "new": "x"},
                     {"old": "old-m", "new": "other"},
                     {"old": "old-m", "new": "old-m"},
                     {"old": "", "new": "x"}, {"old": "old-m"}):
            r = self.client.post(f"/api/bench/history/{rid}/rename", json=body)
            self.assertEqual(r.status_code, 400, body)
        with open(os.path.join(self.tmp.name, f"{rid}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["cfg"]["models"], ["old-m", "other"])
        self.assertNotIn("model_renames", d)


if __name__ == "__main__":
    unittest.main()
