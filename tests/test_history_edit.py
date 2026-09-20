"""历史存档编辑回归：单点删除（POST /api/bench/history/{run_id}/delete-point）
与存档拼接（POST /api/bench/history/{run_id}/splice）。

删除点：身份五元组 + turn 消歧命中移除、deletions 审计留痕、删空留壳、
身份不唯一 409、校验失败 400/404。
拼接：来源全量测点合并进目标（同身份六元组覆盖、新身份追加）、拼接点打
spliced_from/spliced_at 溯源标、splices 审计（新增数 + 覆盖清单含前后
all_ok）、口径参数差异随 warnings 返回不阻断、源=目标 400、缺失/损坏
404、非 dict 点跳过。

测试内把 server.BASE/CONFIG_PATH/RESULTS_DIR 指向临时目录，不触碰真实
配置与存档（fixture 模式同 test_history_retest.py）。
"""
import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import server


def _point(**kw):
    p = {"model": "m1", "scenario": "code", "kind": "llm", "ctx_target": 4096,
         "concurrency": 1, "reqs": [{"req": 0, "err": None}], "all_ok": True,
         "ttft_s": 1.0}
    p.update(kw)
    return p


def _write(rid, points, metric_version=None, **cfg_extra):
    """写一份存档。metric_version=None 表示**不写该字段**（= 遗留口径存档，
    与本机制引入前的存量形态一致）；显式传值才写顶层版本。"""
    cfg = {"provider": "p1", "models": ["m1"], "scenarios": ["code"],
           "ctx_list": [4096], "concurrencies": [1], "max_tokens": 512,
           "repeats": 1}
    cfg.update(cfg_extra)
    d = {"run_id": rid, "started_at": "2026-09-16 22:00:00",
         "cfg": cfg, "results": points}
    if metric_version is not None:
        d["metric_version"] = metric_version
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"), "w",
              encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def _read(rid):
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"),
              encoding="utf-8") as f:
        return json.load(f)


IDENT = {"model": "m1", "scenario": "code", "ctx_target": 4096,
         "inst_tokens": None, "concurrency": 1, "turn": None}


class TestHistoryEditApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._base, self._cfgp, self._res = (server.BASE, server.CONFIG_PATH,
                                             server.RESULTS_DIR)
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        os.makedirs(server.RESULTS_DIR, exist_ok=True)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.BASE, server.CONFIG_PATH, server.RESULTS_DIR = (
            self._base, self._cfgp, self._res)
        self.tmp.cleanup()

    # ---------- 单点删除 ----------

    def test_delete_point_removes_and_audits(self):
        """命中即删：目标点移除、其余点不动、deletions 审计留痕（身份 +
        prev_all_ok + at）。"""
        _write("a1", [_point(), _point(ctx_target=8192, all_ok=False)])
        r = self.client.post("/api/bench/history/a1/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["remaining"], 1)
        d = _read("a1")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["ctx_target"], 8192, "应保留另一点")
        dele = d["deletions"]
        self.assertEqual(len(dele), 1)
        self.assertEqual(dele[0]["ctx_target"], 4096)
        self.assertTrue(dele[0]["prev_all_ok"])
        self.assertIn("at", dele[0])

    def test_delete_point_turn_disambiguates(self):
        """旧版 agent 链测点（同五元组、turn 不同）：按 turn 精确删除。"""
        _write("a1", [_point(scenario="agent", inst_tokens=512, turn=1),
                      _point(scenario="agent", inst_tokens=512, turn=2)])
        body = {**IDENT, "scenario": "agent", "inst_tokens": 512, "turn": 2}
        r = self.client.post("/api/bench/history/a1/delete-point", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        d = _read("a1")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["turn"], 1, "应只删 turn=2")
        self.assertEqual(d["deletions"][0]["turn"], 2)

    def test_delete_point_not_found_404(self):
        _write("a1", [_point()])
        r = self.client.post("/api/bench/history/a1/delete-point",
                             json={**IDENT, "ctx_target": 16384})
        self.assertEqual(r.status_code, 404)

    def test_delete_point_invalid_ident_400(self):
        _write("a1", [_point()])
        for bad in ({**IDENT, "model": ""},
                    {**IDENT, "scenario": "nope"},
                    {**IDENT, "concurrency": 0},
                    {**IDENT, "inst_tokens": -3},
                    {**IDENT, "turn": -1}):
            r = self.client.post("/api/bench/history/a1/delete-point", json=bad)
            self.assertEqual(r.status_code, 400, bad)

    def test_delete_point_duplicate_identity_409(self):
        """身份六元组命中多个测点（存档异常）：不猜，409 拒绝且不删不留痕。"""
        _write("a1", [_point(), _point()])
        r = self.client.post("/api/bench/history/a1/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 409)
        d = _read("a1")
        self.assertEqual(len(d["results"]), 2, "409 时不应删除任何点")
        self.assertNotIn("deletions", d, "409 时不应留审计痕")

    def test_corrupted_archive_404(self):
        """存档 JSON 损坏：删除点与拼接统一 404（不 500）。"""
        with open(os.path.join(server.RESULTS_DIR, "bad.json"), "w",
                  encoding="utf-8") as f:
            f.write("{not json")
        r = self.client.post("/api/bench/history/bad/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 404)
        self.assertIn("损坏", r.json()["detail"])
        _write("dst", [_point()])
        r2 = self.client.post("/api/bench/history/dst/splice",
                              json={"source_run_id": "bad"})
        self.assertEqual(r2.status_code, 404)

    def test_delete_point_archive_missing_404(self):
        r = self.client.post("/api/bench/history/nope/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 404)

    def test_delete_point_empty_results_allowed(self):
        """删空 results 允许：存档留壳（整档删除走 DELETE 端点）。"""
        _write("a1", [_point()])
        r = self.client.post("/api/bench/history/a1/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(_read("a1")["results"], [])

    # ---------- 存档拼接 ----------

    def test_splice_replace_and_append(self):
        """同身份点以来源覆盖（含 all_ok 翻转）、新身份点追加；拼接点打
        spliced_from/spliced_at；splices 审计含覆盖清单前后 all_ok。"""
        _write("dst", [_point(all_ok=False), _point(ctx_target=8192)],
               note="主存档")
        _write("src", [_point(all_ok=True, ttft_s=2.0),
                       _point(ctx_target=32768, concurrency=5)])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual((j["added"], j["replaced"]), (1, 1))
        d = _read("dst")
        self.assertEqual(len(d["results"]), 3, "覆盖不新增、新身份追加")
        over = next(p for p in d["results"] if p["ctx_target"] == 4096)
        self.assertTrue(over["all_ok"], "同身份点应被来源覆盖")
        self.assertEqual(over["ttft_s"], 2.0)
        self.assertEqual(over["spliced_from"], "src")
        self.assertIn("spliced_at", over)
        keep = next(p for p in d["results"] if p["ctx_target"] == 8192)
        self.assertNotIn("spliced_from", keep, "未涉及的点不打溯源标")
        self.assertEqual(d["cfg"].get("note"), "主存档", "目标 cfg 不动")
        sp = d["splices"]
        self.assertEqual(len(sp), 1)
        self.assertEqual(sp[0]["source_run_id"], "src")
        self.assertEqual(sp[0]["added"], 1)
        self.assertEqual(sp[0]["replaced"][0]["prev_all_ok"], False)
        self.assertEqual(sp[0]["replaced"][0]["new_all_ok"], True)

    def test_splice_warnings_on_cfg_mismatch(self):
        """口径参数（max_tokens/reply_mode/thinking）不一致：不阻断，
        warnings 随响应返回；一致时 warnings 为空。"""
        _write("dst", [_point()], max_tokens=512)
        _write("src", [_point(ctx_target=8192)], max_tokens=256,
               reply_mode="echo")
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        w = r.json()["warnings"]
        self.assertTrue(any("输出预算" in x for x in w), w)
        self.assertTrue(any("回复模式" in x for x in w), w)
        self.assertFalse(any("思考模式" in x for x in w), w)

    def test_splice_warns_on_metric_version_mismatch(self):
        """口径版本不一致必须告警（ADR-0083）：本机制引入前的存档没有
        metric_version，与当前口径的读数混拼后不可直接比——顶层差异与
        "来源内部版本不统一"都要提示，且均不阻断。"""
        _write("dst", [_point(metric_version=1)], metric_version=1)
        _write("src", [_point(ctx_target=8192)])           # 遗留存档（无版本字段）
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        w = r.json()["warnings"]
        self.assertTrue(any("口径版本不一致" in x for x in w), w)
        self.assertTrue(any("遗留" in x for x in w), w)
        # 拼接进来的点保留自己的版本（不冒充目标档的版本）
        d = _read("dst")
        spl = [p for p in d["results"] if p.get("ctx_target") == 8192]
        self.assertEqual(len(spl), 1)
        self.assertNotIn("metric_version", spl[0],
                         "拼接点不得被改写成目标存档的口径版本")

    def test_splice_warns_when_source_versions_mixed(self):
        """来源存档内测点版本不统一时单独提示（顶层字段只能代表运行版本，
        点级可能因历史复测/拼接而不一致）。"""
        _write("dst", [_point(metric_version=1)], metric_version=1)
        _write("src", [_point(ctx_target=8192, metric_version=1),
                       _point(ctx_target=16384)], metric_version=1)
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        w = r.json()["warnings"]
        self.assertTrue(any("测点口径版本不统一" in x for x in w), w)

    def test_splice_no_warning_when_versions_match(self):
        _write("dst", [_point(metric_version=1)], metric_version=1)
        _write("src", [_point(ctx_target=8192, metric_version=1)], metric_version=1)
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(any("口径版本" in x for x in r.json()["warnings"]))

    def test_splice_same_archive_400(self):
        _write("dst", [_point()])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "dst"})
        self.assertEqual(r.status_code, 400)

    def test_splice_missing_source_404(self):
        _write("dst", [_point()])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "nope"})
        self.assertEqual(r.status_code, 404)

    def test_splice_skips_non_dict_points(self):
        """来源 results 混入非 dict 项：跳过不炸，其余正常合并。"""
        _write("dst", [_point()])
        _write("src", [_point(ctx_target=8192), None, "junk"])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["added"], 1)
        self.assertEqual(len(_read("dst")["results"]), 2)

    def test_splice_turn_points_match_by_turn(self):
        """agent 链测点拼接：同五元组按 turn 匹配覆盖，不串轮次。"""
        _write("dst", [_point(scenario="agent", inst_tokens=512, turn=1,
                              all_ok=False)])
        _write("src", [_point(scenario="agent", inst_tokens=512, turn=1),
                       _point(scenario="agent", inst_tokens=512, turn=2)])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual((j["added"], j["replaced"]), (1, 1))
        d = _read("dst")
        self.assertEqual(len(d["results"]), 2)
        self.assertTrue(next(p for p in d["results"]
                             if p["turn"] == 1)["all_ok"])


if __name__ == "__main__":
    unittest.main()
