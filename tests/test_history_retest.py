"""历史存档单点复测回归：POST /api/bench/history/{run_id}/retest 的校验与
cfg 窄化（文本/翻译/媒体/agent 矩阵四类点、ctx_edge 贴边档按裁前原档位
下发），以及 _merge_retest 守望合并（产出点替换原存档 + retests 审计留痕
+ 复测独立存档删除；未产出/原存档失踪时不合并、复测存档保留）。

测试内把 server.BASE/CONFIG_PATH/RESULTS_DIR 指向临时目录，bench_start
打桩不真跑测速，不触碰真实配置与存档。
"""
import json
import os
import tempfile
import time
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import server


def _archive_point(**kw):
    p = {"model": "m1", "scenario": "code", "kind": "llm", "ctx_target": 4096,
         "concurrency": 1, "reqs": [{"req": 0, "err": None}], "all_ok": True,
         "ttft_s": 1.0}
    p.update(kw)
    return p


def _write_archive(path, points, **cfg_extra):
    cfg = {"provider": "p1", "models": ["m1"], "scenarios": ["code"],
           "ctx_list": [4096], "concurrencies": [1], "max_tokens": 512,
           "repeats": 3, "note": "原始任务"}
    cfg.update(cfg_extra)
    d = {"run_id": os.path.basename(path)[:-5],
         "started_at": "2026-09-14 21:01:20", "cfg": cfg, "results": points}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


class TestRetestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._base, self._cfgp, self._res = (server.BASE, server.CONFIG_PATH,
                                             server.RESULTS_DIR)
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        os.makedirs(server.RESULTS_DIR, exist_ok=True)
        with open(server.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"providers": [{
                "name": "p1", "label": "P1",
                "gateway_url": "http://127.0.0.1:4000", "local": True,
                "deployments": []}]}, f)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.BASE, server.CONFIG_PATH, server.RESULTS_DIR = (
            self._base, self._cfgp, self._res)
        self.tmp.cleanup()

    def _archive(self, points, **cfg_extra) -> str:
        rid = "20260914-210120-72a8"
        _write_archive(os.path.join(server.RESULTS_DIR, f"{rid}.json"),
                       points, **cfg_extra)
        return rid

    def _post(self, rid, body):
        with mock.patch("server.bench_start",
                        new=mock.AsyncMock(
                            return_value={"run_id": "20990101-000000-ffff"})
                        ) as bs:
            r = self.client.post(f"/api/bench/history/{rid}/retest", json=body)
        return r, bs

    def test_narrows_cfg_for_text_point(self):
        """文本点复测：cfg 沿用原运行参数但窄化到该点；note 不继承。"""
        rid = self._archive([_archive_point()])
        r, bs = self._post(rid, {"model": "m1", "scenario": "code",
                                 "ctx_target": 4096, "concurrency": 1,
                                 "inst_tokens": None})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["archive"], rid)
        cfg = bs.call_args.args[0]
        self.assertEqual(cfg["models"], ["m1"])
        self.assertEqual(cfg["scenarios"], ["code"])
        self.assertEqual(cfg["concurrencies"], [1])
        self.assertEqual(cfg["ctx_list"], [4096])
        self.assertEqual(cfg["repeats"], 3)          # 原参数沿用
        self.assertEqual(cfg["max_tokens"], 512)
        self.assertNotIn("note", cfg)                # 任务备注不继承

    def test_edge_point_retests_original_rung(self):
        """贴边裁减点（ctx_target=260608、ctx_edge=262144）按裁前原档位下发，
        引擎重新走贴边逻辑。"""
        rid = self._archive([_archive_point(ctx_target=260608,
                                            ctx_edge=262144)])
        r, bs = self._post(rid, {"model": "m1", "scenario": "code",
                                 "ctx_target": 260608, "concurrency": 1,
                                 "inst_tokens": None})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(bs.call_args.args[0]["ctx_list"], [262144])

    def test_agent_matrix_point(self):
        """agent 缓存×指令矩阵点：窄化为单缓存档 × 单指令档，ctx_list 清空。"""
        rid = self._archive([_archive_point(
            scenario="agent", ctx_target=8192, inst_tokens=1024)],
            scenarios=["agent"], agent_cache_ladder=[0, 8192],
            agent_inst_ladder=[512, 1024], ctx_list=[])
        r, bs = self._post(rid, {"model": "m1", "scenario": "agent",
                                 "ctx_target": 8192, "concurrency": 1,
                                 "inst_tokens": 1024})
        self.assertEqual(r.status_code, 200, r.text)
        cfg = bs.call_args.args[0]
        self.assertEqual(cfg["agent_cache_ladder"], [8192])
        self.assertEqual(cfg["agent_inst_ladder"], [1024])
        self.assertEqual(cfg["ctx_list"], [])

    def test_ladder_scenario_point(self):
        """固定阶梯场景（翻译/媒体）：档位值回对应 <场景>_ladder 字段。"""
        rid = self._archive([_archive_point(scenario="translate",
                                            ctx_target=2400, kind="llm")],
                            scenarios=["translate"], translate_ladder=[2400])
        r, bs = self._post(rid, {"model": "m1", "scenario": "translate",
                                 "ctx_target": 2400, "concurrency": 1,
                                 "inst_tokens": None})
        self.assertEqual(r.status_code, 200, r.text)
        cfg = bs.call_args.args[0]
        self.assertEqual(cfg["translate_ladder"], [2400])
        self.assertEqual(cfg["ctx_list"], [])

    def test_unknown_point_404(self):
        rid = self._archive([_archive_point()])
        r, _ = self._post(rid, {"model": "m1", "scenario": "code",
                                "ctx_target": 8192, "concurrency": 1,
                                "inst_tokens": None})
        self.assertEqual(r.status_code, 404)

    def test_legacy_turn_chain_rejected(self):
        """旧版 agent 连续任务链测点（turn 字段）不支持单点复测。"""
        rid = self._archive([_archive_point(scenario="agent", ctx_target=12000,
                                            turn=3)])
        r, _ = self._post(rid, {"model": "m1", "scenario": "agent",
                                "ctx_target": 12000, "concurrency": 1,
                                "inst_tokens": None})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("链", r.json()["detail"])

    def test_bad_body_rejected(self):
        rid = self._archive([_archive_point()])
        for body in ({"model": "", "scenario": "code", "ctx_target": 4096,
                      "concurrency": 1},
                     {"model": "m1", "scenario": "nope", "ctx_target": 4096,
                      "concurrency": 1},
                     {"model": "m1", "scenario": "code", "ctx_target": -1,
                      "concurrency": 1},
                     {"model": "m1", "scenario": "code", "ctx_target": 4096,
                      "concurrency": 0}):
            r, _ = self._post(rid, body)
            self.assertEqual(r.status_code, 400, f"{body} 应 400: {r.text}")


class TestMergeRetest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._res = server.RESULTS_DIR
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        os.makedirs(server.RESULTS_DIR, exist_ok=True)

    def tearDown(self):
        server.RESULTS_DIR = self._res
        server.RUNS.pop("rt-run", None)
        self.tmp.cleanup()

    def _fake_run(self, results):
        run = types.SimpleNamespace(finished_at=time.time(), results=results,
                                    mock_seen=False)
        server.RUNS["rt-run"] = run
        return run

    async def test_merge_replaces_point_and_deletes_retest_archive(self):
        """合并：新点替换原存档同 identity 点（其余点不动）、点带
        retested_at、存档顶层追加 retests 审计、复测运行独立存档删除。"""
        old = _archive_point(all_ok=False, err="HTTP 500")
        other = _archive_point(ctx_target=8192)
        _write_archive(os.path.join(server.RESULTS_DIR, "arch1.json"),
                       [old, other])
        new = _archive_point(all_ok=True, ttft_s=0.8)
        self._fake_run([new])
        # 复测运行自身的存档文件（合并成功后应删除）
        _write_archive(os.path.join(server.RESULTS_DIR, "rt-run.json"), [new])
        ident = ("m1", "code", 4096, None, 1)
        await server._merge_retest("arch1", "rt-run", ident)
        with open(os.path.join(server.RESULTS_DIR, "arch1.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(len(d["results"]), 2)
        self.assertTrue(d["results"][0]["all_ok"], "新点应替换原位")
        self.assertEqual(d["results"][0]["ttft_s"], 0.8)
        self.assertIn("retested_at", d["results"][0])
        self.assertEqual(d["results"][1]["ctx_target"], 8192, "其余点不动")
        aud = d["retests"]
        self.assertEqual(len(aud), 1)
        self.assertEqual(aud[0]["retest_run_id"], "rt-run")
        self.assertFalse(aud[0]["prev_all_ok"])
        self.assertTrue(aud[0]["new_all_ok"])
        self.assertFalse(os.path.exists(
            os.path.join(server.RESULTS_DIR, "rt-run.json")),
            "合并成功后复测独立存档应删除")

    async def test_merge_skipped_when_no_matching_point(self):
        """未产出可合并点（停止/跳档/identity 漂移）：原存档不动，复测运行
        的独立存档保留为记录。"""
        _write_archive(os.path.join(server.RESULTS_DIR, "arch1.json"),
                       [_archive_point()])
        stray = _archive_point(ctx_target=8192)   # identity 不匹配
        self._fake_run([stray])
        _write_archive(os.path.join(server.RESULTS_DIR, "rt-run.json"), [stray])
        ident = ("m1", "code", 4096, None, 1)
        await server._merge_retest("arch1", "rt-run", ident)
        with open(os.path.join(server.RESULTS_DIR, "arch1.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertNotIn("retests", d)
        self.assertNotIn("retested_at", json.dumps(d["results"]))
        self.assertTrue(os.path.exists(
            os.path.join(server.RESULTS_DIR, "rt-run.json")),
            "未合并时复测独立存档应保留")

    async def test_merge_skipped_for_mock_run(self):
        """mock 网关运行不落档也不合并（mock 数据不得混入真实存档）。"""
        _write_archive(os.path.join(server.RESULTS_DIR, "arch1.json"),
                       [_archive_point(all_ok=False)])
        run = self._fake_run([_archive_point()])
        run.mock_seen = True
        await server._merge_retest("arch1", "rt-run",
                                   ("m1", "code", 4096, None, 1))
        with open(os.path.join(server.RESULTS_DIR, "arch1.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertNotIn("retests", d)
        self.assertFalse(d["results"][0]["all_ok"], "mock 数据不得合并入档")


if __name__ == "__main__":
    unittest.main()
