"""api 组 · 历史存档编辑回归：单点删除（POST /api/bench/history/{run_id}/
delete-point）与存档拼接（POST /api/bench/history/{run_id}/splice）。

删除点：身份五元组 + turn 消歧命中移除、deletions 审计留痕、删空留壳、
身份不唯一 409、校验失败 400/404。
拼接：来源全量测点合并进目标（同身份六元组覆盖、新身份追加）、拼接点打
spliced_from/spliced_at 溯源标、splices 审计（新增数 + 覆盖清单含前后
all_ok）、口径参数差异随 warnings 返回不阻断、源=目标 400、缺失/损坏
404、非 dict 点跳过。

自 tests/test_history_edit.py 拆分：目录重定向改用共享基类
tests._shared.ServerDirCase（BASE/CONFIG_PATH/RESULTS_DIR 三全局一并重定向
进临时目录），合成存档 helper 改用共享的 _point / write_archive / read_archive。

运行（仓库根目录）：
    python -m pytest tests/api/test_history_edit.py -q
    python -m unittest discover -s tests/api -p test_history_edit.py
"""
import os
import unittest

from tests._shared import ServerDirCase, _point, read_archive, write_archive

import server


IDENT = {"model": "m1", "scenario": "code", "ctx_target": 4096,
         "inst_tokens": None, "concurrency": 1, "turn": None}


class TestHistoryEditApi(ServerDirCase):
    # ---------- 单点删除 ----------

    def test_delete_point_removes_and_audits(self):
        """命中即删：目标点移除、其余点不动、deletions 审计留痕（身份 +
        prev_all_ok + at）。"""
        write_archive("a1", [_point(), _point(ctx_target=8192, all_ok=False)])
        r = self.client.post("/api/bench/history/a1/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["remaining"], 1)
        d = read_archive("a1")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["ctx_target"], 8192, "应保留另一点")
        dele = d["deletions"]
        self.assertEqual(len(dele), 1)
        self.assertEqual(dele[0]["ctx_target"], 4096)
        self.assertTrue(dele[0]["prev_all_ok"])
        self.assertIn("at", dele[0])

    def test_delete_point_turn_disambiguates(self):
        """旧版 agent 链测点（同五元组、turn 不同）：按 turn 精确删除。"""
        write_archive("a1", [_point(scenario="agent", inst_tokens=512, turn=1),
                             _point(scenario="agent", inst_tokens=512, turn=2)])
        body = {**IDENT, "scenario": "agent", "inst_tokens": 512, "turn": 2}
        r = self.client.post("/api/bench/history/a1/delete-point", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        d = read_archive("a1")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["turn"], 1, "应只删 turn=2")
        self.assertEqual(d["deletions"][0]["turn"], 2)

    def test_delete_point_not_found_404(self):
        write_archive("a1", [_point()])
        r = self.client.post("/api/bench/history/a1/delete-point",
                             json={**IDENT, "ctx_target": 16384})
        self.assertEqual(r.status_code, 404)

    def test_delete_point_invalid_ident_400(self):
        write_archive("a1", [_point()])
        for bad in ({**IDENT, "model": ""},
                    {**IDENT, "scenario": "nope"},
                    {**IDENT, "concurrency": 0},
                    {**IDENT, "inst_tokens": -3},
                    {**IDENT, "turn": -1}):
            r = self.client.post("/api/bench/history/a1/delete-point", json=bad)
            self.assertEqual(r.status_code, 400, bad)

    def test_delete_point_duplicate_identity_409(self):
        """身份六元组命中多个测点（存档异常）：不猜，409 拒绝且不删不留痕。"""
        write_archive("a1", [_point(), _point()])
        r = self.client.post("/api/bench/history/a1/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 409)
        d = read_archive("a1")
        self.assertEqual(len(d["results"]), 2, "409 时不应删除任何点")
        self.assertNotIn("deletions", d, "409 时不应留审计痕")

    def test_corrupted_archive_404(self):
        """存档不存在或存档 JSON 损坏：删除点与拼接统一 404（不 500），同走
        _load_archive 的 404 分支。"""
        r = self.client.post("/api/bench/history/nope/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 404)
        with open(os.path.join(server.RESULTS_DIR, "bad.json"), "w",
                  encoding="utf-8") as f:
            f.write("{not json")
        r = self.client.post("/api/bench/history/bad/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 404)
        self.assertIn("损坏", r.json()["detail"])
        write_archive("dst", [_point()])
        r2 = self.client.post("/api/bench/history/dst/splice",
                              json={"source_run_id": "bad"})
        self.assertEqual(r2.status_code, 404)

    def test_delete_point_empty_results_allowed(self):
        """删空 results 允许：存档留壳（整档删除走 DELETE 端点）。"""
        write_archive("a1", [_point()])
        r = self.client.post("/api/bench/history/a1/delete-point", json=IDENT)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(read_archive("a1")["results"], [])

    # ---------- 存档拼接 ----------

    def test_splice_replace_and_append(self):
        """同身份点以来源覆盖（含 all_ok 翻转）、新身份点追加；拼接点打
        spliced_from/spliced_at；splices 审计含覆盖清单前后 all_ok。"""
        write_archive("dst", [_point(all_ok=False), _point(ctx_target=8192)],
                      note="主存档")
        write_archive("src", [_point(all_ok=True, ttft_s=2.0),
                              _point(ctx_target=32768, concurrency=5)])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual((j["added"], j["replaced"]), (1, 1))
        d = read_archive("dst")
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
        write_archive("dst", [_point()], max_tokens=512)
        write_archive("src", [_point(ctx_target=8192)], max_tokens=256,
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
        "来源内部版本不统一"都要提示，且均不阻断；版本一致时（反向臂）不得
        出现任何口径版本告警。"""
        write_archive("dst", [_point(metric_version=1)], metric_version=1)
        write_archive("src", [_point(ctx_target=8192)])    # 遗留存档（无版本字段）
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        w = r.json()["warnings"]
        self.assertTrue(any("口径版本不一致" in x for x in w), w)
        self.assertTrue(any("遗留" in x for x in w), w)
        # 拼接进来的点保留自己的版本（不冒充目标档的版本）
        d = read_archive("dst")
        spl = [p for p in d["results"] if p.get("ctx_target") == 8192]
        self.assertEqual(len(spl), 1)
        self.assertNotIn("metric_version", spl[0],
                         "拼接点不得被改写成目标存档的口径版本")
        # 反向臂：双方版本一致 → 无任何「口径版本」告警（不误报）
        write_archive("dst2", [_point(metric_version=1)], metric_version=1)
        write_archive("src2", [_point(ctx_target=8192, metric_version=1)],
                      metric_version=1)
        r = self.client.post("/api/bench/history/dst2/splice",
                             json={"source_run_id": "src2"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(any("口径版本" in x for x in r.json()["warnings"]))

    def test_splice_warns_when_source_versions_mixed(self):
        """来源存档内测点版本不统一时单独提示（顶层字段只能代表运行版本，
        点级可能因历史复测/拼接而不一致）。"""
        write_archive("dst", [_point(metric_version=1)], metric_version=1)
        write_archive("src", [_point(ctx_target=8192, metric_version=1),
                              _point(ctx_target=16384)], metric_version=1)
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        w = r.json()["warnings"]
        self.assertTrue(any("测点口径版本不统一" in x for x in w), w)

    def test_splice_same_archive_400(self):
        write_archive("dst", [_point()])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "dst"})
        self.assertEqual(r.status_code, 400)

    def test_splice_missing_source_404(self):
        write_archive("dst", [_point()])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "nope"})
        self.assertEqual(r.status_code, 404)

    def test_splice_skips_non_dict_points(self):
        """来源 results 混入非 dict 项：跳过不炸，其余正常合并。"""
        write_archive("dst", [_point()])
        write_archive("src", [_point(ctx_target=8192), None, "junk"])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["added"], 1)
        self.assertEqual(len(read_archive("dst")["results"]), 2)

    def test_splice_turn_points_match_by_turn(self):
        """agent 链测点拼接：同五元组按 turn 匹配覆盖，不串轮次。"""
        write_archive("dst", [_point(scenario="agent", inst_tokens=512, turn=1,
                                     all_ok=False)])
        write_archive("src", [_point(scenario="agent", inst_tokens=512, turn=1),
                              _point(scenario="agent", inst_tokens=512, turn=2)])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual((j["added"], j["replaced"]), (1, 1))
        d = read_archive("dst")
        self.assertEqual(len(d["results"]), 2)
        self.assertTrue(next(p for p in d["results"]
                             if p["turn"] == 1)["all_ok"])


if __name__ == "__main__":
    unittest.main()
