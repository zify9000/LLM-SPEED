"""服务端守卫回归（ADR-0081 / ADR-0082）。

覆盖：
- 写请求的 Host / 同源 / 请求体限额守卫（DNS rebinding 与跨源改写 provider
  网关地址 → API Key 外泄的防线）
- 测速配置白名单重建（未知键丢弃、模型名长度上限）
- 存档读写的线程池化与历史列表摘要缓存
- 凭据文件写入权限（0o600）与 .env 解析容错（export / 引号 / 行内注释）
- 历史改名：模型名非法字符拒绝、审计数组同步改名、目标名快照冲突拒绝
- run_id 白名单、损坏存档语义、多地址择优不带凭据探测

测试内把 server.BASE/CONFIG_PATH/RESULTS_DIR 指向临时目录，不触碰真实
配置与存档（fixture 模式同 test_history_edit.py）。
"""
import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import server


def _point(**kw):
    p = {"model": "m1", "scenario": "code", "kind": "llm", "ctx_target": 4096,
         "concurrency": 1, "reqs": [{"req": 0, "err": None}], "all_ok": True,
         "ttft_s": 1.0}
    p.update(kw)
    return p


def _write(rid, points=None, **cfg_extra):
    cfg = {"provider": "p1", "models": ["m1"], "scenarios": ["code"],
           "ctx_list": [4096], "concurrencies": [1], "max_tokens": 512,
           "repeats": 1}
    cfg.update(cfg_extra)
    d = {"run_id": rid, "started_at": "2026-09-17 10:00:00",
         "cfg": cfg, "results": points if points is not None else [_point()]}
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"), "w",
              encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def _read(rid):
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"),
              encoding="utf-8") as f:
        return json.load(f)


class _GuardBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._base, self._cfgp, self._res = (server.BASE, server.CONFIG_PATH,
                                             server.RESULTS_DIR)
        self._global_key = server.GLOBAL_API_KEY
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        os.makedirs(server.RESULTS_DIR, exist_ok=True)
        server.GLOBAL_API_KEY = ""
        server._HIST_CACHE.clear()
        self.client = TestClient(server.app)

    def tearDown(self):
        server.BASE, server.CONFIG_PATH, server.RESULTS_DIR = (
            self._base, self._cfgp, self._res)
        server.GLOBAL_API_KEY = self._global_key
        server._HIST_CACHE.clear()
        self.tmp.cleanup()


class TestWriteGuards(_GuardBase):
    """写接口的 Host / 同源 / 体积守卫。"""

    _CFG = {"providers": []}

    def test_trusted_host_rejects_rebinding(self):
        """Host 不在白名单 → 400（DNS rebinding 的 Host 过不去）。"""
        r = self.client.put("/api/config", json=self._CFG,
                            headers={"host": "evil.example"})
        self.assertEqual(r.status_code, 400)

    def test_trusted_host_allows_loopback_and_testclient(self):
        for host in ("127.0.0.1", "localhost", "testserver"):
            r = self.client.put("/api/config", json=self._CFG, headers={"host": host})
            self.assertEqual(r.status_code, 200, host)

    def test_cross_origin_write_rejected(self):
        r = self.client.put("/api/config", json=self._CFG,
                            headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_same_origin_write_allowed(self):
        r = self.client.put("/api/config", json=self._CFG,
                            headers={"origin": "http://testserver",
                                     "host": "testserver"})
        self.assertEqual(r.status_code, 200)

    def test_cross_site_fetch_metadata_rejected(self):
        r = self.client.put("/api/config", json=self._CFG,
                            headers={"sec-fetch-site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        r2 = self.client.put("/api/config", json=self._CFG,
                             headers={"sec-fetch-site": "same-origin"})
        self.assertEqual(r2.status_code, 200)

    def test_read_requests_not_guarded(self):
        """只读请求不受同源/Host 之外的额外限制（Host 仍受校验）。"""
        r = self.client.get("/api/config", headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 200)

    def test_oversized_body_rejected(self):
        r = self.client.put("/api/config",
                            json={"providers": [], "pad": "x" * (server._MAX_BODY_BYTES + 16)})
        self.assertEqual(r.status_code, 413)


class TestBenchCfgWhitelist(_GuardBase):
    def test_unknown_keys_dropped(self):
        clean = server._clean_bench_cfg({
            "models": ["m"], "scenarios": ["code"], "concurrencies": [1],
            "junk": "A" * 1000, "gateway_url": "http://attacker.example",
            "api_key": "sk-evil", "model_info": {"m": {}}})
        self.assertNotIn("junk", clean)
        # 服务端注入类字段不接受客户端指定（否则可把 key 指向任意网关）
        self.assertNotIn("gateway_url", clean)
        self.assertNotIn("api_key", clean)
        self.assertNotIn("model_info", clean)
        self.assertEqual(clean["models"], ["m"])

    def test_model_name_length_capped(self):
        with self.assertRaises(Exception) as ctx:
            server._clean_bench_cfg({"models": ["m" * (server._MODEL_NAME_MAX + 1)]})
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_free_params_validated(self):
        for bad in ({"cache": "weird"}, {"temperature": 5}, {"cpt": 0},
                    {"temperature": True}):
            with self.assertRaises(Exception) as ctx:
                server._validate_bench_cfg(
                    {"models": ["m"], "scenarios": ["code"], "ctx_list": [0],
                     "concurrencies": [1], **bad})
            self.assertEqual(getattr(ctx.exception, "status_code", None), 400, bad)


class TestArchiveIo(_GuardBase):
    def test_history_summary_cache_and_refresh(self):
        _write("r1")
        first = self.client.get("/api/bench/history").json()["history"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["run_id"], "r1")
        self.assertEqual(first[0]["n_points"], 1)
        # 命中缓存（同 mtime/size）后内容仍正确
        self.assertEqual(self.client.get("/api/bench/history").json(), {"history": first})
        # 存档被改写 → mtime/size 变化 → 摘要刷新
        d = _read("r1")
        d["results"].append(_point(ctx_target=8192))
        with open(os.path.join(server.RESULTS_DIR, "r1.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        refreshed = self.client.get("/api/bench/history").json()["history"]
        self.assertEqual(refreshed[0]["n_points"], 2)

    def test_corrupt_archive_excluded_from_list_but_detail_404(self):
        _write("ok1")
        with open(os.path.join(server.RESULTS_DIR, "broken.json"), "w",
                  encoding="utf-8") as f:
            f.write("{not json")
        ids = [h["run_id"] for h in self.client.get("/api/bench/history").json()["history"]]
        self.assertEqual(ids, ["ok1"])            # 坏档不进列表，也不阻塞其余条目
        self.assertEqual(self.client.get("/api/bench/history/broken").status_code, 404)
        self.assertEqual(self.client.get("/api/bench/history/nope").status_code, 404)

    def test_clear_skips_directories(self):
        _write("ok1")
        os.makedirs(os.path.join(server.RESULTS_DIR, "dir.json"))   # 目录名以 .json 结尾
        r = self.client.delete("/api/bench/history")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["removed"], 1)
        self.assertTrue(os.path.isdir(os.path.join(server.RESULTS_DIR, "dir.json")))

    def test_run_id_whitelist(self):
        for bad in ("a/b", "..", "a" * 65, "a b", "x.json"):
            with self.assertRaises(Exception) as ctx:
                server._valid_run_id(bad)
            self.assertEqual(getattr(ctx.exception, "status_code", None), 400, bad)
        server._valid_run_id("20260917-101010-abcd")   # 合法形态不抛

    def test_atomic_write_private_uses_0600(self):
        p = os.path.join(self.tmp.name, ".env")
        with open(p, "w", encoding="utf-8") as f:
            f.write("API_KEY=old\n")
        os.chmod(p, 0o664)
        server._atomic_write(p, "API_KEY=new\n", private=True)
        self.assertEqual(os.stat(p).st_mode & 0o777, 0o600)
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "API_KEY=new\n")
        # 非 private 文件不加限制（config.json 走默认 mkstemp 0600 亦可）
        q = os.path.join(self.tmp.name, "config.json")
        server._atomic_write(q, "{}\n")
        self.assertTrue(os.path.exists(q))

    def test_atomic_write_leaves_no_tmp_residue_on_failure(self):
        p = os.path.join(self.tmp.name, "cfg.json")
        with mock.patch("os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                server._atomic_write(p, "{}\n")
        leftovers = [f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class TestDotenvParsing(_GuardBase):
    def test_quotes_inline_comment_export(self):
        env_path = os.path.join(self.tmp.name, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write('\n'.join([
                "# 注释行",
                "export API_KEY_GUARD_A=sk-aaa  # 行内注释",
                'API_KEY_GUARD_B="sk-bbb#not-comment"',
                "API_KEY_GUARD_C='sk-ccc'",
                "NOT_A_PAIR",
                "",
            ]) + "\n")
        for k in ("API_KEY_GUARD_A", "API_KEY_GUARD_B", "API_KEY_GUARD_C"):
            os.environ.pop(k, None)
        try:
            server._load_dotenv()
            self.assertEqual(os.environ["API_KEY_GUARD_A"], "sk-aaa")
            self.assertEqual(os.environ["API_KEY_GUARD_B"], "sk-bbb#not-comment")
            self.assertEqual(os.environ["API_KEY_GUARD_C"], "sk-ccc")
        finally:
            for k in ("API_KEY_GUARD_A", "API_KEY_GUARD_B", "API_KEY_GUARD_C"):
                os.environ.pop(k, None)


class TestRenameGuard(_GuardBase):
    def _rename(self, body, rid="r1"):
        return self.client.post(f"/api/bench/history/{rid}/rename", json=body)

    def test_rejects_injection_chars(self):
        _write("r1")
        for bad in ("x' onmouseover='alert(1)", '<img src=x>', 'a"b', "a\\b",
                    "a&b", "a\nb"):
            r = self._rename({"old": "m1", "new": bad})
            self.assertEqual(r.status_code, 400, bad)
        self.assertEqual(_read("r1")["cfg"]["models"], ["m1"])   # 未改动

    def test_accepts_normal_names_and_renames_audit(self):
        _write("r1", models=["m1"],
               **{})
        d = _read("r1")
        d["retests"] = [{"model": "m1", "scenario": "code", "at": "x"}]
        d["deletions"] = [{"model": "m1", "scenario": "code", "at": "x"}]
        d["splices"] = [{"source_run_id": "s1", "at": "x",
                         "replaced": [{"model": "m1", "scenario": "code"}]}]
        with open(os.path.join(server.RESULTS_DIR, "r1.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        r = self._rename({"old": "m1", "new": "m1-改名 v2"})
        self.assertEqual(r.status_code, 200, r.text)
        got = _read("r1")
        self.assertEqual(got["cfg"]["models"], ["m1-改名 v2"])
        self.assertEqual(got["results"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["retests"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["deletions"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["splices"][0]["replaced"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["model_renames"][-1]["audit_entries"], 3)

    def test_rejects_snapshot_key_collision(self):
        """新名已在 model_info 等快照键中存在时拒绝，避免静默覆盖部署信息。"""
        _write("r1", models=["m1", "m2"])
        d = _read("r1")
        d["cfg"]["model_info"] = {"m2": {"hardware": "已存在的快照"}}
        with open(os.path.join(server.RESULTS_DIR, "r1.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        r = self._rename({"old": "m1", "new": "m2"})
        self.assertEqual(r.status_code, 400)


class TestCredentialKeyApi(_GuardBase):
    def test_key_with_newline_or_too_long_rejected(self):
        server._atomic_write(server.CONFIG_PATH, json.dumps({"providers": [
            {"name": "p1", "gateway_urls": ["http://127.0.0.1:4000"],
             "deployments": []}]}), )
        for bad in ("sk-a\n", "sk-a\r\nb", "x" * 513):
            r = self.client.put("/api/config/key", json={"provider": "p1", "key": bad})
            self.assertEqual(r.status_code, 400)
        r = self.client.put("/api/config/key", json={"provider": "p1", "key": "sk-ok"})
        self.assertEqual(r.status_code, 200)
        env_path = os.path.join(self.tmp.name, ".env")
        self.assertEqual(os.stat(env_path).st_mode & 0o777, 0o600)
        with open(env_path, encoding="utf-8") as f:
            self.assertIn("sk-ok", f.read())


class TestGatewayPick(_GuardBase):
    def test_probe_does_not_carry_key(self):
        """多地址择优探测不带 Authorization：候选里可能混着明文内网地址，
        旧实现把同一把 key 并发发往全部候选（含最终不选中的）。"""
        seen = []

        async def fake_probe(url, headers, timeout=3.0):
            seen.append((url, dict(headers)))
            return 0.01 if url.endswith(":2") else None

        p = {"name": "p1", "gateway_urls": ["http://a:1", "http://b:2"],
             "gateway_url": "http://a:1"}
        os.environ["API_KEY_P1"] = "sk-secret"
        try:
            with mock.patch.object(server, "_probe_url", fake_probe):
                url, lat = asyncio.run(server.pick_gateway_url(p))
        finally:
            os.environ.pop("API_KEY_P1", None)
        self.assertEqual(url, "http://b:2")
        self.assertEqual(len(seen), 2)
        for url, headers in seen:
            self.assertNotIn("Authorization", headers, url)


if __name__ == "__main__":
    unittest.main()
