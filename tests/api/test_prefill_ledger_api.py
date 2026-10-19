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
from tests._shared import ServerDirCase

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


class TestPrefillRehome(ServerDirCase):
    """改存档的环境信息 / 模型名 → 台账里那次测速写的读数跟着换身份（ADR-0141）。

    这是用户报的缺口：「历史记录切换或更新部署环境信息后，速率先验基线并不会同步更
    新」。不搬，先验就永久漂在旧键上：按新登记跑测查不到这张表（几轮白攒），而那张
    「这次不会用上」的旧表还会被当成垃圾删掉。"""
    REDIRECT_CONFIG = False

    def setUp(self):
        super().setUp()
        self.rid = "r-rehome"
        self.path = os.path.join(server.RESULTS_DIR, bench.PREFILL_LEDGER_FILE)
        self.arch = os.path.join(server.RESULTS_DIR, f"{self.rid}.json")
        self.cfg = {"provider": "p1", "models": ["m1"],
                    "model_info": {"m1": {"deploy_label": "fastllm-nvfp4",
                                          "quant": "nvfp4", "kv_quant": "fp8",
                                          "hardware": "2080Ti×2",
                                          "framework": "fastllm v1.8.2"}},
                    "model_max_ctx": {"m1": 262144}}
        with open(self.arch, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.rid, "cfg": self.cfg, "results": []}, f)
        self.old_key = bench._prefill_fingerprint("m1", self.cfg)
        bench._ledger_save(self.path, {self.old_key: {"code": [
            [4103.0, 1200.0, _NOW, self.rid],
            [8103.0, 1100.0, _NOW - 10, self.rid],
            [4103.0, 999.0, _NOW - 20, ""]]}})      # 第三条是旧格式（来源未知）

    def _env(self, **info):
        # 端点对六个环境字段是「整组覆盖」（不带的键写空），前端两个入口也都整组提交，
        # 所以测试同样整组提交：只传一两个字段等于把其余字段抹空，身份位会跟着变
        full = {"deploy_label": "fastllm-nvfp4", "quant": "nvfp4", "kv_quant": "fp8",
                "hardware": "2080Ti\u00d72", "framework": "fastllm v1.8.2", "params": ""}
        full.update(info)
        return self.client.post(f"/api/bench/history/{self.rid}/env",
                               json={"model": "m1", "info": full})

    def _raw(self):
        return json.load(open(self.path, encoding="utf-8"))

    def test_env_edit_moves_only_that_runs_rows(self):
        r = self._env(deploy_label="vllm2080ti-fp8", framework="vLLM-2080Ti v0.2.2")
        self.assertEqual(r.status_code, 200, r.text)
        p = r.json()["prior"]
        self.assertEqual(p["moved"], 2, "只搬那次测速写的两条")
        self.assertEqual(p["unknown"], 1, "来源未知的旧读数如实报数，不猜它属于谁")
        new_key = bench._prefill_fingerprint("m1", {**self.cfg, "model_info": {
            "m1": {**self.cfg["model_info"]["m1"], "deploy_label": "vllm2080ti-fp8",
                   "framework": "vLLM-2080Ti v0.2.2"}}})
        raw = self._raw()
        self.assertEqual(sorted(r[0] for r in raw[new_key]["code"]), [4103.0, 8103.0])
        self.assertEqual({r[3] for r in raw[new_key]["code"]}, {self.rid},
                     "新表里只应带着来源等于本次测速的读数")
        self.assertEqual([r[0] for r in raw[self.old_key]["code"]], [4103.0],
                     "来源未知的那条留在旧表（搬不动就不能猜）")
        audit = [json.loads(x) for x in open(
            os.path.join(server.RESULTS_DIR, bench.PREFILL_LEDGER_AUDIT),
            encoding="utf-8").read().strip().splitlines()]
        self.assertEqual(audit[-1]["op"], "rehome")
        self.assertEqual(audit[-1]["run"], self.rid)
        self.assertTrue(os.path.exists(os.path.join(server.RESULTS_DIR, p["backup"])),
                        "搬家前也要备份原件（算错了能把整张表拿回去）")

    def test_env_edit_without_identity_change_touches_nothing(self):
        """只改核心参数这类不进身份位的字段 → 不搬也不写盘（没变身份就没必要动）。"""
        before = open(self.path, "rb").read()
        p = self._env(params="dflash2草稿4、批块2048").json()["prior"]
        self.assertIsNone(p)
        self.assertEqual(open(self.path, "rb").read(), before)

    def test_rename_moves_table_to_new_model_name(self):
        r = self.client.post(f"/api/bench/history/{self.rid}/rename",
                            json={"old": "m1", "new": "m1-vllm"})
        self.assertEqual(r.status_code, 200, r.text)
        p = r.json()["prior"]
        self.assertEqual(p["moved"], 2)
        raw = self._raw()
        new_key = next(k for k in raw if k.startswith("m1-vllm|"))
        self.assertEqual(len([x for x in raw[new_key]["code"] if x[3] == self.rid]), 2)
        self.assertEqual([x[3] for x in raw[self.old_key]["code"]], [""],
                         "搬不走的只有来源未知那条：它得留在旧表，界面按 legacy 报出来")

    def test_rehome_flags_risk_while_a_run_is_active(self):
        """台账是运行开头装载、收口整张写回：在跑时搬的家可能被盖掉，得当场说明，
        不能让用户以为方就成。"""
        server.RUNS["busy"] = types.SimpleNamespace(finished_at=None)
        p = self._env(framework="vLLM-2080Ti v0.2.2").json()["prior"]
        self.assertEqual(p["moved"], 2)
        self.assertIn("收口", p.get("risk", ""))

    def test_view_exposes_provenance(self):
        """面板要能看出这张表来自几轮、有多少旧数据没标来源（否则下次改环境又捐了）。"""
        d = self.client.get("/api/prefill-ledger").json()
        t = [x for x in d["tables"] if x["key"] == self.old_key][0]
        self.assertEqual((t["runs"], t["legacy"]), (1, 1))
        self._env(framework="vLLM-2080Ti v0.2.2")
        d = self.client.get("/api/prefill-ledger").json()
        old = [x for x in d["tables"] if x["key"] == self.old_key][0]
        self.assertEqual((old["runs"], old["legacy"]), (0, 1),
                     "搬完剩下的全是旧数据")


class TestPrefillLedgerProvenance(unittest.TestCase):
    """数据层：读数带来源（ADR-0141）。旧文件只有三位，装载不能因此拒收。"""

    def test_note_records_source_and_load_keeps_it(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, bench.PREFILL_LEDGER_FILE)
            t = {}
            self.assertTrue(bench._prefill_ledger_note(t, "m|d", "code", 4103.0, 1200.0,
                                                       now=_NOW, src="run-9"))
            bench._ledger_save(p, t)
            rows = bench._prefill_ledger_load(p, now=_NOW)["m|d"]["code"]
            self.assertEqual(rows[0][3], "run-9", "来源得能活过一轮读写")

    def test_load_accepts_legacy_three_field_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, bench.PREFILL_LEDGER_FILE)
            bench._ledger_save(p, {"m|d": {"code": [[4103.0, 1200.0, _NOW]]}})
            rows = bench._prefill_ledger_load(p, now=_NOW)["m|d"]["code"]
            self.assertEqual(rows[0], [4103.0, 1200.0, _NOW, ""],
                             "旧格式补空串 = 来源未知，不能整条丢（丢了下轮就没先验）")

    def test_rehome_needs_ledger_and_run(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, bench.PREFILL_LEDGER_FILE)
            self.assertEqual(bench.prefill_ledger_rehome(p, "r1", "a", "b")["moved"], 0,
                             "没台账也不报错（台账是优化项，不能拖死改名）")
            bench._ledger_save(p, {"a": {"code": [[4103.0, 1200.0, _NOW, "r1"]]}})
            self.assertEqual(bench.prefill_ledger_rehome(p, "", "a", "b")["moved"], 0)
            self.assertEqual(bench.prefill_ledger_rehome(p, "r2", "a", "b")["moved"], 0,
                             "不是它写的就不搬")
            self.assertEqual(list(json.load(open(p, encoding="utf-8"))), ["a"],
                             "一点没搬动就不写盘")
            r = bench.prefill_ledger_rehome(p, "r1", "a", "b")
            self.assertEqual((r["moved"], r["unknown"], r["dropped"]), (1, 0, 0))
            self.assertEqual(list(json.load(open(p, encoding="utf-8"))), ["b"],
                             "旧表被搬空就摘掉，不在界面上留一张幽灵表")


if __name__ == "__main__":
    unittest.main(verbosity=2)
