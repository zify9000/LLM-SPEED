"""api 组：decode 交付粒度基线（跨运行台账）的查看、清理与搬家（ADR-0145 草稿）。

为什么单独立一个文件：跑动中那格 Decode 的 token 数是「包数 ÷ 每包几个 token」，
那个比此前只活在单次 run 的内存里——本轮学到的比用在下一档就偏一个固定倍数（用户
2026-10-20 目检：中途稳定 90+、落定 70+）。接了跨运行台账就必须有查看与清理入口，
否则脏种子只能手改 JSON（同 ADR-0140 给 prefill 先验补的那两个端点）。

    python -m pytest tests/api/test_delivery_ledger_api.py -q
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
_KEY = "gw:local|Qwen3.8-27B|fastllm-nvfp4|nvfp4|fp8|ctx262144"   # 与 prefill 台账同指纹


def _row(ctx=4103.0, x=4.9, src="r1", concurrent=None):
    return [ctx, x, _NOW, src] + ([1] if concurrent else [])


class TestDeliveryLedgerApi(unittest.TestCase):
    """GET /api/delivery-ledger 只读视图；POST /api/delivery-ledger/prune 删锚点。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        self._providers = server.load_providers
        server.RESULTS_DIR = self.tmp.name
        self.path = os.path.join(self.tmp.name, bench.DELIVERY_LEDGER_FILE)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.RESULTS_DIR = self._results
        server.load_providers = self._providers
        server.RUNS.clear()

    def _seed(self, table=None):
        bench._ledger_save(self.path, table or {
            f"{_KEY}|http://10.0.0.3:13080": {
                "code|c1": [_row(4103.0, 4.9), _row(4110.0, 4.7, src="r2"),
                            _row(65000.0, 2.0, src="r2", concurrent=True)]}})

    def test_get_empty_when_no_ledger(self):
        r = self.client.get("/api/delivery-ledger")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["empty"], "没台账要报空表，不是 404")

    def test_get_shows_bucket_median_in_tokens_per_chunk(self):
        """界面口径是「token / 包」（引擎内部那个数是"一个 token 算几个包"的倒数，
        直接照搬会让人看不懂），同档取中位数并给出条数与并跑标记。"""
        self._seed()
        server.load_providers = lambda: [{
            "name": "local", "label": "本地网关", "gateway_urls": ["http://x:1"],
            "deployments": [{"label": "fastllm-nvfp4", "models": ["Qwen3.8-27B"],
                             "quants": ["nvfp4"], "kv_quants": ["fp8"], "active": True}],
            "model_profiles": [{"model": "Qwen3.8-27B", "max_ctx": 262144,
                                "ctx_presets": [{"label": "4K", "tokens": 4096}]}]}]
        d = self.client.get("/api/delivery-ledger").json()
        self.assertEqual([t["key"] for t in d["tables"]], [_KEY], "地址尾缀已剥掉")
        self.assertIn(_KEY, d["current"], "与当前登记一致 → 界面能标「这次会用上」")
        self.assertEqual([s["scenario"] for s in d["tables"][0]["scenarios"]], ["code"])
        self.assertEqual([s["conc"] for s in d["tables"][0]["scenarios"]], [1])
        anchors = d["tables"][0]["scenarios"][0]["anchors"]
        self.assertEqual(len(anchors), 2)
        self.assertAlmostEqual(anchors[0]["x"], 4.8, places=1,
                               msg="同档两条取中位数（4.9 / 4.7）")
        self.assertEqual(anchors[0]["n"], 2)
        self.assertTrue(anchors[1]["concurrent"], "并跑期间写的锚点要看得见")
        self.assertTrue(d["file"].endswith(bench.DELIVERY_LEDGER_FILE))

    def test_get_lists_stale_and_legacy_keys(self):
        """引擎装载侧会整表不采纳的（旧口径无网关段、整表过期），视图必须如实列出：
        不列就是「我存了东西，界面却说没有」。"""
        self._seed({
            _KEY: {"code|c1": [_row()]},
            "A|fastllm": {"code|c1": [_row()]},
            f"gw:local|Old|{time.time() - 10 * 86400:.0f}": {"code|c1": [_row()]},
        })
        d = self.client.get("/api/delivery-ledger").json()
        self.assertEqual(d["stale_keys"], ["A|fastllm"],
                         "旧口径遗留表只进 stale_keys，不进 tables")
        self.assertNotIn("A|fastllm", [t["key"] for t in d["tables"]])

    def test_prune_removes_bucket_and_keeps_backup_and_audit(self):
        self._seed()
        r = self.client.post("/api/delivery-ledger/prune",
                             json={"key": _KEY, "scenario": "code",
                                   "conc": 1, "ctx": 65000})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["removed"], d["remaining"]), (1, 2))
        left = json.load(open(self.path, encoding="utf-8"))
        # 写盘时键已归一（地址尾缀剔掉），与 prefill 台账同口径
        self.assertEqual([p[0] for p in left[_KEY]["code|c1"]], [4103.0, 4110.0],
                         "只删那一档")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, d["backup"])),
                        "删前的原件要能整张拿回去")
        audit = open(os.path.join(self.tmp.name, bench.DELIVERY_LEDGER_AUDIT),
                     encoding="utf-8").read().strip().splitlines()
        self.assertEqual(json.loads(audit[0])["kind"], "delivery")

    def test_prune_whole_table_then_errors(self):
        self._seed()
        d = self.client.post("/api/delivery-ledger/prune", json={"key": _KEY}).json()
        self.assertEqual(d["removed"], 3)
        self.assertTrue(self.client.get("/api/delivery-ledger").json()["empty"])
        self.assertEqual(self.client.post("/api/delivery-ledger/prune",
                                          json={"key": _KEY}).status_code, 404)
        self.assertEqual(self.client.post("/api/delivery-ledger/prune",
                                          json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/delivery-ledger/prune",
                                          json={"key": _KEY, "ctx": 0}).status_code, 400)
        self.assertEqual(self.client.post("/api/delivery-ledger/prune",
                                          json={"key": "nope", "scenario": "code",
                                                "conc": 1, "ctx": 4096}).status_code, 404,
                         "猜键名删（「code」还是「code|c1」）要报错，不能静默没删到")

    def test_prune_blocked_while_running(self):
        """有测速在跑时改台账会被那次运行收口覆盖，所以一律 409（不是悄悄成功）。"""
        self._seed()
        server.RUNS["busy"] = types.SimpleNamespace(finished_at=None)
        r = self.client.post("/api/delivery-ledger/prune", json={"key": _KEY})
        self.assertEqual(r.status_code, 409)
        self.assertIn("测速", r.json()["detail"])
        self.assertTrue(self.client.get("/api/delivery-ledger").json()["busy"])
        raw = json.load(open(self.path, encoding="utf-8"))
        self.assertEqual(len(raw[_KEY + "|http://10.0.0.3:13080"]["code|c1"]), 3,
                         "被拦下时盘上一个字节都不该动")


class TestDeliveryLedgerMoves(ServerDirCase):
    """两张台账同指纹归户 → 改环境/改模型名/改网关名必须一起搬（ADR-0141 / 0143）。

    只搬一张会出现「这一轮 prefill 先验是新身份、decode 种子是旧身份」的半对新状态，
    比两张都旧更难查——那正是 2026-10-17 prefill 换户事故再演一遍的形状。"""

    def _prov(self, name, **kw):
        p = {"name": name, "label": "本地网关", "gateway_urls": ["http://x:1", "http://y:1"],
             "local": True, "deployments": [{"label": "d1", "models": ["m1"],
                                            "quants": ["fp8"], "active": True}]}
        p.update(kw)
        return p

    def _paths(self):
        return (os.path.join(server.RESULTS_DIR, bench.PREFILL_LEDGER_FILE),
                os.path.join(server.RESULTS_DIR, bench.DELIVERY_LEDGER_FILE))

    def test_rehome_moves_both_ledgers(self):
        pp, dp = self._paths()
        bench._ledger_save(pp, {"old": {"code": [[4103.0, 1200.0, _NOW, "r1"]]}})
        bench._ledger_save(dp, {"old": {"code|c1": [_row(src="r1")],
                                        "creative|c1": [_row(65000.0, 2.0, src="r1")]}})
        out = server._prefill_rehome("r1", "old", "new")
        self.assertEqual((out["prior_moved"], out["delivery_moved"]), (1, 2),
                         f"两张台账要一起搬: {out}")
        self.assertEqual(list(json.load(open(pp, encoding="utf-8"))), ["new"])
        self.assertEqual(list(json.load(open(dp, encoding="utf-8"))), ["new"])

    def test_rehome_reports_rows_the_ledger_cannot_attribute(self):
        """旧数据没有来源位 → 搬不动要如实报，不猜归属（搬错比不搬更坏）。"""
        dp = self._paths()[1]
        bench._ledger_save(dp, {"old": {"code|c1": [_row(src="")]}})
        out = server._prefill_rehome("r1", "old", "new")
        self.assertEqual(out["delivery_moved"], 0)
        self.assertEqual(out["unknown"], 1)
        self.assertEqual(list(json.load(open(dp, encoding="utf-8"))), ["old"])

    def test_put_config_rekeys_both_ledgers(self):
        pp, dp = self._paths()
        self.client.put("/api/config", json={"providers": [self._prov("p1")]}).json()
        bench._ledger_save(pp, {"gw:p1|m1|d1": {"code": [[4103.0, 1200.0, _NOW, "r1"]]}})
        bench._ledger_save(dp, {"gw:p1|m1|d1": {"code|c1": [_row(src="r1")]}})
        r = self.client.put("/api/config", json={"providers": [self._prov("p1-new")]}).json()
        note = r.get("prior_ledger", "")
        self.assertIn("prefill 先验", note, str(r)[:160])
        self.assertIn("decode 交付粒度", note,
                      "只换一户的回复会把「种子还挂在旧网关名下」藏起来")
        self.assertEqual(list(json.load(open(pp, encoding="utf-8"))), ["gw:p1-new|m1|d1"])
        self.assertEqual(list(json.load(open(dp, encoding="utf-8"))), ["gw:p1-new|m1|d1"])

    def test_rekey_is_noop_when_delivery_ledger_missing(self):
        """交付粒度台账还没生成过（新机器）时，改名不得报错、不得拖垮保存配置。"""
        pp = self._paths()[0]
        self.client.put("/api/config", json={"providers": [self._prov("p1")]}).json()
        bench._ledger_save(pp, {"gw:p1|m1|d1": {"code": [[4103.0, 1200.0, _NOW, "r1"]]}})
        r = self.client.put("/api/config", json={"providers": [self._prov("p1-new")]}).json()
        self.assertIn("prefill 先验", r.get("prior_ledger", ""), str(r)[:160])
        self.assertFalse(os.path.exists(self._paths()[1]),
                         "没台账就不要凭空建一个空文件")


if __name__ == "__main__":
    unittest.main(verbosity=2)
