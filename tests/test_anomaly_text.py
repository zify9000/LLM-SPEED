"""异常输入全文侧车回归：下载端点、生命周期（单档删除 / 清空）、
splice 引用迁移，以及 splice / _merge_retest 共用的 _relocate_anomaly_refs。

背景（ADR-0053 + ADR-0082）：超长 in_text 落 <run_id>.anomaly.txt 侧车，
存档只留头尾摘要 + in_text_ref 字节切片；本文件钉住"存档可见 ⇒ 切片可取"
这条不变量在各生命周期操作后仍然成立。

与 test_history_edit.py 同口径：把 server.BASE/CONFIG_PATH/RESULTS_DIR 指向
临时目录，不碰真实存档、不走网络。
"""
import hashlib
import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import bench
import server


def _point(**kw):
    p = {"model": "m1", "scenario": "code", "kind": "llm", "ctx_target": 4096,
         "concurrency": 1, "reqs": [{"req": 0, "err": None}], "all_ok": True,
         "ttft_s": 1.0}
    p.update(kw)
    return p


def _write(rid, points, **cfg_extra):
    cfg = {"provider": "p1", "models": ["m1"], "scenarios": ["code"],
           "ctx_list": [4096], "concurrencies": [1], "max_tokens": 512,
           "repeats": 1}
    cfg.update(cfg_extra)
    d = {"run_id": rid, "started_at": "2026-09-16 22:00:00",
         "cfg": cfg, "results": points}
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"), "w",
              encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def _read(rid):
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _side(rid):
    return os.path.join(server.RESULTS_DIR, f"{rid}{bench.ANOMALY_TEXT_SUFFIX}")


def _put_sidecar(rid, data: bytes):
    with open(_side(rid), "wb") as f:
        f.write(data)


def _ref_for(rid, data: bytes, text: str):
    return {"file": f"{rid}{bench.ANOMALY_TEXT_SUFFIX}", "off": 0,
            "len": len(data), "chars": len(text),
            "sha1": hashlib.sha1(data).hexdigest()}


class TestAnomalyTextApi(unittest.TestCase):
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

    # ---------- 下载端点 ----------

    def test_endpoint_returns_exact_text(self):
        """有侧车：200 + text/plain; charset=utf-8 + 原文逐字节一致。"""
        full = "异常输入全文" * 100
        data = full.encode("utf-8")
        _write("r1", [_point(reqs=[{"in_text": "……截断……",
                                    "in_text_ref": _ref_for("r1", data, full)}])])
        _put_sidecar("r1", data)
        r = self.client.get("/api/bench/history/r1/anomaly-text")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.content, data)
        self.assertEqual(r.text, full)
        self.assertIn("text/plain", r.headers["content-type"])
        self.assertIn("charset=utf-8", r.headers["content-type"])
        self.assertIn("inline", r.headers["content-disposition"])

    def test_endpoint_404_when_no_sidecar(self):
        _write("r1", [_point()])
        r = self.client.get("/api/bench/history/r1/anomaly-text")
        self.assertEqual(r.status_code, 404)
        self.assertIn("异常文本存档", r.json()["detail"])

    def test_endpoint_400_invalid_run_id(self):
        for bad in ("a..b", "x.json", "a" * 65):
            r = self.client.get(f"/api/bench/history/{bad}/anomaly-text")
            self.assertEqual(r.status_code, 400, bad)

    # ---------- 生命周期：单档删除 / 清空 ----------

    def test_delete_one_removes_sidecar(self):
        _write("r1", [_point()])
        _put_sidecar("r1", b"abc")
        r = self.client.delete("/api/bench/history/r1")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["anomaly_text_removed"])
        self.assertFalse(os.path.exists(_side("r1")))
        self.assertFalse(os.path.exists(
            os.path.join(server.RESULTS_DIR, "r1.json")))

    def test_delete_one_without_sidecar_reports_false(self):
        _write("r1", [_point()])
        r = self.client.delete("/api/bench/history/r1")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["anomaly_text_removed"])

    def test_clear_removes_sidecars_and_counts_separately(self):
        """清空：JSON 与侧车都删；removed 语义不变（只计 JSON），侧车数另报
        anomaly_text_removed；目录名以 .anomaly.txt 结尾仍跳过不删。"""
        _write("r1", [_point()])
        _write("r2", [_point()])
        _put_sidecar("r1", b"aaa")
        _put_sidecar("r2", b"bb")
        os.makedirs(os.path.join(server.RESULTS_DIR, "dir.anomaly.txt"))
        r = self.client.delete("/api/bench/history")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["removed"], 2, "removed 仍只计 JSON 存档")
        self.assertEqual(j["anomaly_text_removed"], 2)
        self.assertFalse(os.path.exists(_side("r1")))
        self.assertFalse(os.path.exists(_side("r2")))
        self.assertTrue(os.path.isdir(
            os.path.join(server.RESULTS_DIR, "dir.anomaly.txt")))

    # ---------- 拼接：切片迁移 ----------

    def test_splice_copies_slice_and_rewrites_ref(self):
        """splice：来源切片复制进目标侧车、ref 改指目标文件（新 off、
        len/chars/sha1 不变）；目标档自包含，来源档删了也不断链。"""
        full = "来源异常全文" * 500
        data = full.encode("utf-8")
        src_ref = _ref_for("src", data, full)
        _write("src", [_point(ctx_target=8192,
                              reqs=[{"in_text": "……截断……",
                                     "in_text_ref": src_ref}])])
        _put_sidecar("src", data)
        _write("dst", [_point()])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(any("异常输入全文" in w for w in r.json()["warnings"]))
        self.assertEqual(open(_side("dst"), "rb").read(), data)
        spl = next(p for p in _read("dst")["results"]
                   if p["ctx_target"] == 8192)
        ref = spl["reqs"][0]["in_text_ref"]
        self.assertEqual(ref["file"], f"dst{bench.ANOMALY_TEXT_SUFFIX}")
        self.assertEqual(ref["off"], 0)
        self.assertEqual((ref["len"], ref["chars"]),
                         (src_ref["len"], src_ref["chars"]))
        self.assertEqual(ref["sha1"], src_ref["sha1"])
        raw = open(_side("dst"), "rb").read()
        self.assertEqual(raw[ref["off"]:ref["off"] + ref["len"]], data)
        self.assertEqual(hashlib.sha1(raw[ref["off"]:ref["off"] + ref["len"]])
                         .hexdigest(), ref["sha1"])

    def test_splice_warns_and_keeps_ref_when_source_sidecar_missing(self):
        """来源侧车不可读：不改 ref、随 warnings 报出"全文缺失"、不建空侧车，
        但拼接本身照常成功（截断文本仍在存档里）。"""
        full = "丢失的全文" * 300
        data = full.encode("utf-8")
        src_ref = _ref_for("src", data, full)
        _write("src", [_point(ctx_target=8192,
                              reqs=[{"in_text": "……截断……",
                                     "in_text_ref": src_ref}])])
        _write("dst", [_point()])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        w = r.json()["warnings"]
        self.assertTrue(any("异常输入全文缺失" in x and "src" in x for x in w), w)
        spl = next(p for p in _read("dst")["results"]
                   if p["ctx_target"] == 8192)
        self.assertEqual(spl["reqs"][0]["in_text_ref"], src_ref,
                         "来源不可读时不得改写 ref")
        self.assertFalse(os.path.exists(_side("dst")),
                         "没有可复制切片时不得创建空侧车")

    def test_splice_without_refs_creates_no_sidecar(self):
        _write("dst", [_point()])
        _write("src", [_point(ctx_target=8192)])
        r = self.client.post("/api/bench/history/dst/splice",
                             json={"source_run_id": "src"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(_side("dst")))

    # ---------- 共用助手（等价覆盖 _merge_retest 的引用迁移） ----------

    def test_relocate_helper_copies_and_rewrites(self):
        """_merge_retest 与 splice 共用 _relocate_anomaly_refs：直接验证切片
        复制 + ref 改写（复测合并整体跑要起后台测速运行，这里打共用助手）。"""
        full = "复测全文" * 400
        data = full.encode("utf-8")
        ref = _ref_for("newrun", data, full)
        _put_sidecar("newrun", data)
        point = {"reqs": [{"in_text": "……", "in_text_ref": ref}]}
        warns = server._relocate_anomaly_refs(point, "newrun", "orig")
        self.assertEqual(warns, [])
        self.assertEqual(open(_side("orig"), "rb").read(), data)
        new_ref = point["reqs"][0]["in_text_ref"]
        self.assertEqual(new_ref["file"], f"orig{bench.ANOMALY_TEXT_SUFFIX}")
        self.assertEqual(new_ref["off"], 0)
        self.assertEqual((new_ref["len"], new_ref["chars"]),
                         (ref["len"], ref["chars"]))
        self.assertEqual(new_ref["sha1"], ref["sha1"])

    def test_relocate_helper_warns_on_sha1_mismatch(self):
        """来源切片能读但 sha1 与 ref 不符：照样复制并按 spec 保留原 sha1，
        但要告警请人工复核（不能静默把坏切片当好的用）。"""
        data = b"actual bytes"
        bad_ref = {"file": "newrun.anomaly.txt", "off": 0, "len": len(data),
                   "chars": len(data), "sha1": "0" * 40}
        _put_sidecar("newrun", data)
        point = {"reqs": [{"in_text_ref": bad_ref}]}
        warns = server._relocate_anomaly_refs(point, "newrun", "orig")
        self.assertTrue(any("sha1" in w for w in warns), warns)
        self.assertEqual(open(_side("orig"), "rb").read(), data)
        self.assertEqual(point["reqs"][0]["in_text_ref"]["file"],
                         f"orig{bench.ANOMALY_TEXT_SUFFIX}")


if __name__ == "__main__":
    unittest.main()
