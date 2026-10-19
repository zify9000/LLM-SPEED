"""api 组：速率先验基线（跨运行台账）的查看与清理端点（ADR-0140）。

为什么单独立一个文件：先验直接决定等待期「预估」读数的平台值，错了会一路骗到
屏幕上（2026-10-19 真机：脏锚点把 256K 估成 2523，实测 490），而它此前没有任
何界面——想剔一条脏锚点只能手改 JSON。这两个端点是那件事的补救。

    python -m pytest tests/api/test_prefill_ledger_api.py -q
"""
import json
import os
import tempfile
import time
import types
import unittest

from fastapi.testclient import TestClient

import bench
import server

_NOW = 1_800_000_000.0
_KEY = "Qwen3.8-27B|fastllm-nvfp4|nvfp4|fp8|ctx262144"


class TestPrefillLedgerApi(unittest.TestCase):
    """GET /api/prefill-ledger 只读视图；POST /api/prefill-ledger/prune 删锚点。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        self._providers = server.load_providers
        server.RESULTS_DIR = self.tmp.name
        self.path = os.path.join(self.tmp.name, bench.PREFILL_LEDGER_FILE)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.RESULTS_DIR = self._results
        server.load_providers = self._providers
        server.RUNS.clear()

    def _seed(self, table=None):
        bench._ledger_save(self.path, table or {
            f"{_KEY}|http://10.0.0.3:13080": {
                "code": [[4103.0, 1239.0, _NOW - 86400],
                         [65000.0, 900.0, _NOW - 3600],
                         [65100.0, 880.0, _NOW - 1800]]}})

    def test_get_empty_when_no_ledger(self):
        r = self.client.get("/api/prefill-ledger")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["empty"], "没台账要报空表，不是 404")

    def test_get_shows_normalized_table_and_current_key(self):
        """旧键（带网关尾缀）在视图里已归一；`current` 要能把这张表标成「会被用上」。"""
        self._seed()
        server.load_providers = lambda: [{
            "name": "local", "label": "本地网关", "gateway_urls": ["http://x:1"],
            "deployments": [{"label": "fastllm-nvfp4", "models": ["Qwen3.8-27B"],
                             "quants": ["nvfp4"], "kv_quants": ["fp8"], "active": True}],
            "model_profiles": [{"model": "Qwen3.8-27B", "max_ctx": 262144}]}]
        d = self.client.get("/api/prefill-ledger").json()
        self.assertEqual([t["key"] for t in d["tables"]], [_KEY], "地址尾缀已剥掉")
        self.assertIn(_KEY, d["current"], "与当前登记一致 → 界面能标「这次会用上」")
        anchors = d["tables"][0]["scenarios"][0]["anchors"]
        self.assertEqual(len(anchors), 2, "同档两条并成一档")
        self.assertEqual(anchors[1]["n"], 2)
        self.assertTrue(d["file"].endswith(bench.PREFILL_LEDGER_FILE))

    def test_prune_removes_bucket_and_keeps_backup_and_audit(self):
        self._seed()
        r = self.client.post("/api/prefill-ledger/prune",
                             json={"key": _KEY, "scenario": "code", "ctx": 65000})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["removed"], d["remaining"]), (2, 1), "同档两条一起删")
        left = json.load(open(self.path, encoding="utf-8"))
        self.assertEqual([p[0] for p in left[_KEY]["code"]], [4103.0])
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, d["backup"])),
                        "删前的原件要能整张拿回去")
        audit = open(os.path.join(self.tmp.name, bench.PREFILL_LEDGER_AUDIT),
                     encoding="utf-8").read().strip().splitlines()
        self.assertEqual(len(audit), 1)
        self.assertEqual(json.loads(audit[0])["key"], _KEY)

    def test_prune_whole_table_then_errors(self):
        self._seed()
        d = self.client.post("/api/prefill-ledger/prune", json={"key": _KEY}).json()
        self.assertEqual(d["removed"], 3)
        self.assertTrue(self.client.get("/api/prefill-ledger").json()["empty"])
        # 表已空 → 不猜、不"没删到也算成功"
        self.assertEqual(self.client.post("/api/prefill-ledger/prune",
                                          json={"key": _KEY}).status_code, 404)
        self.assertEqual(self.client.post("/api/prefill-ledger/prune",
                                          json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/prefill-ledger/prune",
                                          json={"key": _KEY, "ctx": 0}).status_code, 400)

    def test_prune_blocked_while_running(self):
        """有测速在跑时改台账会被那次运行收口覆盖，所以一律 409（不是悄悄成功）。"""
        self._seed()
        server.RUNS["busy"] = types.SimpleNamespace(finished_at=None)
        r = self.client.post("/api/prefill-ledger/prune", json={"key": _KEY})
        self.assertEqual(r.status_code, 409)
        self.assertIn("测速", r.json()["detail"])
        self.assertTrue(self.client.get("/api/prefill-ledger").json()["busy"])
        raw = json.load(open(self.path, encoding="utf-8"))
        self.assertEqual(len(raw[_KEY + "|http://10.0.0.3:13080"]["code"]), 3,
                         "被拦下时盘上一个字节都不该动（键还是盘上原样，没顺手归一）")


class TestPrefillLedgerView(unittest.TestCase):
    """数据层视图口径：与引擎装载同源（过期条目不显示，界面不排第二套真相）。"""

    def test_view_drops_stale_entries(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, bench.PREFILL_LEDGER_FILE)
            old = _NOW - (bench.PREFILL_LEDGER_MAX_AGE_S + 86400)
            bench._ledger_save(p, {"m|d": {"code": [[4103.0, 1200.0, old],
                                                    [8103.0, 1100.0, _NOW]]}})
            view = bench.prefill_ledger_view(p, now=_NOW)
            self.assertEqual(view["max_age_days"], 90)
            anchors = view["tables"][0]["scenarios"][0]["anchors"]
            self.assertEqual([a["ctx"] for a in anchors], [8103],
                             "过期锚点引擎本来就不用，界面也不许显示")
            self.assertGreaterEqual(anchors[0]["age_d"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
