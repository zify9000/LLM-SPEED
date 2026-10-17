"""api 组：服务端守卫与本地配置/存档 IO 的接口回归（自 tests/test_server_guard.py 拆分）。

覆盖：
- 写请求的 Host / 同源 / 请求体限额守卫（DNS rebinding 与跨源改写 provider
  网关地址 → API Key 外泄的防线）
- 测速配置白名单重建（未知键丢弃、模型名长度上限）
- 存档历史列表摘要缓存、损坏存档语义、run_id 白名单
- 存档写入原子替换（不留 .tmp 残渣、失败保旧文件）、凭据文件写入权限（0o600）
  与 .env 解析容错（export / 引号 / 行内注释）
- 历史改名：模型名非法字符拒绝、审计数组同步改名、目标名快照冲突拒绝
- 多地址择优不带凭据探测、前端依赖本地化契约

运行（须从仓库根目录）：
    python -m pytest tests/api/test_server_guard.py -q
    python -m unittest discover -s tests/api -p test_server_guard.py
    python -m unittest tests.api.test_server_guard -v

测试内把 server.BASE/CONFIG_PATH/RESULTS_DIR 指向临时目录（tests._shared 的
ServerDirCase），不触碰真实配置与存档。
"""
import asyncio
import hashlib
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

# 顺序约束：先经 tests._shared（模块级播种 MOCK_*、并 import server）再落 server。
from tests._shared import ServerDirCase, _point, read_archive, write_archive

import server


class TestWriteGuards(ServerDirCase):
    """写接口的 Host / 同源 / 体积守卫。"""

    _CFG = {"providers": []}

    def test_trusted_host_rejected_and_loopback_allowed(self):
        """Host 不在白名单 → 400（DNS rebinding 的 Host 过不去）；回环地址与
        TestClient 的默认 Host 放行。"""
        r = self.client.put("/api/config", json=self._CFG,
                            headers={"host": "evil.example"})
        self.assertEqual(r.status_code, 400)
        for host in ("127.0.0.1", "localhost", "testserver"):
            with self.subTest(host=host):
                r = self.client.put("/api/config", json=self._CFG,
                                    headers={"host": host})
                self.assertEqual(r.status_code, 200, host)

    def test_origin_and_sec_fetch_guards(self):
        """写请求跨源 / 跨站拒绝、同源放行；只读请求不受同源限制。"""
        for headers in ({"origin": "http://evil.example"},
                        {"sec-fetch-site": "cross-site"}):
            with self.subTest(headers=headers):
                r = self.client.put("/api/config", json=self._CFG,
                                    headers=headers)
                self.assertEqual(r.status_code, 403)
        for headers in ({"origin": "http://testserver", "host": "testserver"},
                        {"sec-fetch-site": "same-origin"}):
            with self.subTest(headers=headers):
                r = self.client.put("/api/config", json=self._CFG,
                                    headers=headers)
                self.assertEqual(r.status_code, 200, headers)
        # 只读请求不受同源/Host 之外的额外限制（Host 仍受校验）
        r = self.client.get("/api/config", headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 200)

    def test_oversized_body_rejected(self):
        r = self.client.put("/api/config",
                            json={"providers": [], "pad": "x" * (server._MAX_BODY_BYTES + 16)})
        self.assertEqual(r.status_code, 413)


class TestBenchCfgWhitelist(ServerDirCase):
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


class TestArchiveIo(ServerDirCase):
    def test_history_summary_cache_and_refresh(self):
        write_archive("r1")
        first = self.client.get("/api/bench/history").json()["history"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["run_id"], "r1")
        self.assertEqual(first[0]["n_points"], 1)
        # 命中缓存（同 mtime/size）后内容仍正确；ADR-0126 的披露字段一并钉住
        again = self.client.get("/api/bench/history").json()
        self.assertEqual(again["history"], first)
        self.assertEqual((again["total"], again["matched"], again["truncated"]),
                         (1, 1, False))
        # 存档被改写 → mtime/size 变化 → 摘要刷新
        d = read_archive("r1")
        d["results"].append(_point(ctx_target=8192))
        with open(os.path.join(server.RESULTS_DIR, "r1.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        refreshed = self.client.get("/api/bench/history").json()["history"]
        self.assertEqual(refreshed[0]["n_points"], 2)

    def test_corrupt_archive_excluded_from_list_but_detail_404(self):
        write_archive("ok1")
        with open(os.path.join(server.RESULTS_DIR, "broken.json"), "w",
                  encoding="utf-8") as f:
            f.write("{not json")
        ids = [h["run_id"] for h in self.client.get("/api/bench/history").json()["history"]]
        self.assertEqual(ids, ["ok1"])            # 坏档不进列表，也不阻塞其余条目
        self.assertEqual(self.client.get("/api/bench/history/broken").status_code, 404)
        self.assertEqual(self.client.get("/api/bench/history/nope").status_code, 404)

    def test_clear_skips_directories(self):
        write_archive("ok1")
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


class TestAtomicWrite(unittest.TestCase):
    """_atomic_write：原子替换不留残渣、失败保旧文件、凭据文件权限收紧。

    用**裸**临时目录（不套 ServerDirCase）：断言里的 os.listdir 要求目录下只有
    目标文件，而 ServerDirCase 的临时目录下自带 results/ 子目录。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, name="a.json"):
        return os.path.join(self.tmp.name, name)

    def test_replace_failure_keeps_old_and_cleans_tmp(self):
        p = self._path()
        server._atomic_write(p, '{"a": 1}')
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"a": 1}')
        self.assertEqual(os.listdir(self.tmp.name), ["a.json"])   # 无 .tmp 残留
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                server._atomic_write(p, "new")
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"a": 1}')                # 旧文件完好
        self.assertEqual(os.listdir(self.tmp.name), ["a.json"])   # tmp 已清理
        leftovers = [f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_private_file_0600(self):
        """凭据文件写入一律收紧到 0o600，含"旧文件权限偏宽"的纠正。

        旧实现对已存在的 .env 沿用原权限，于是 `cp .env.example .env` 带来的
        0664（同机其他用户可读）会一直传下去，API Key 长期可被读走——这是
        缺陷不是特性，故断言随实现一并修正。"""
        p = self._path(".env")
        server._atomic_write(p, "API_KEY=new\n", private=True)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)   # 新建收紧
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "API_KEY=new\n")
        for wide in (0o664, 0o644):
            with self.subTest(mode=oct(wide)):
                os.chmod(p, wide)
                server._atomic_write(p, "K=W\n", private=True)
                self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)   # 已存在也收紧
                with open(p, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "K=W\n")
        # 非 private 文件不加限制（config.json 走默认 mkstemp 0600 亦可）
        q = self._path("config.json")
        server._atomic_write(q, "{}\n")
        self.assertTrue(os.path.exists(q))


class TestDotenvParsing(ServerDirCase):
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


class TestRenameGuard(ServerDirCase):
    def _rename(self, body, rid="r1"):
        return self.client.post(f"/api/bench/history/{rid}/rename", json=body)

    def test_rejects_injection_chars(self):
        write_archive("r1")
        for bad in ("x' onmouseover='alert(1)", '<img src=x>', 'a"b', "a\\b",
                    "a&b", "a\nb"):
            r = self._rename({"old": "m1", "new": bad})
            self.assertEqual(r.status_code, 400, bad)
        self.assertEqual(read_archive("r1")["cfg"]["models"], ["m1"])   # 未改动

    def test_accepts_normal_names_and_renames_audit(self):
        write_archive("r1", models=["m1"],
                      **{})
        d = read_archive("r1")
        d["retests"] = [{"model": "m1", "scenario": "code", "at": "x"}]
        d["deletions"] = [{"model": "m1", "scenario": "code", "at": "x"}]
        d["splices"] = [{"source_run_id": "s1", "at": "x",
                         "replaced": [{"model": "m1", "scenario": "code"}]}]
        with open(os.path.join(server.RESULTS_DIR, "r1.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        r = self._rename({"old": "m1", "new": "m1-改名 v2"})
        self.assertEqual(r.status_code, 200, r.text)
        got = read_archive("r1")
        self.assertEqual(got["cfg"]["models"], ["m1-改名 v2"])
        self.assertEqual(got["results"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["retests"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["deletions"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["splices"][0]["replaced"][0]["model"], "m1-改名 v2")
        self.assertEqual(got["model_renames"][-1]["audit_entries"], 3)

    def test_rejects_snapshot_key_collision(self):
        """新名已在 model_info 等快照键中存在时拒绝，避免静默覆盖部署信息。"""
        write_archive("r1", models=["m1", "m2"])
        d = read_archive("r1")
        d["cfg"]["model_info"] = {"m2": {"hardware": "已存在的快照"}}
        with open(os.path.join(server.RESULTS_DIR, "r1.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        r = self._rename({"old": "m1", "new": "m2"})
        self.assertEqual(r.status_code, 400)


class TestVendoredAssets(unittest.TestCase):
    """前端依赖本地化契约：页面不得外链、vendored 文件必须可服务且未被替换。

    依赖走 CDN 时，第三方脚本运行在**能调用本机全部 API** 的页面里（改配置、
    删历史、触发对云端付费网关的测速）；本地化把这条供应链面直接消掉，但
    "vendor 文件被误删/被换版本/静态挂载失效"都要能被回归抓住（见
    static/vendor/README.md）。文件哈希在此硬编码——换库时同步改这里与 README。

    本类**不继承 ServerDirCase**：那些用例把 server.BASE 指向临时目录，而这里要
    校验真实仓库里的 vendored 资源（StaticFiles 在 import 时已捕获真实目录）。"""
    ROOT = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    VENDOR = {
        "echarts.min.js":
            "42f8329d989b6f6539dd2b15bbdf0d82025762ac112fbb60dc57b27d7bcf3946",
        "html2canvas.min.js":
            "e87e550794322e574a1fda0c1549a3c70dae5a93d9113417a429016838eab8cb",
    }

    def setUp(self):
        self.client = TestClient(server.app)

    def test_vendor_files_match_documented_hashes(self):
        for name, want in self.VENDOR.items():
            path = os.path.join(self.ROOT, "static", "vendor", name)
            self.assertTrue(os.path.isfile(path), f"{name} 缺失（static/vendor/）")
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(got, want,
                             f"{name} 内容变了——换版本请同步 static/vendor/README.md、"
                             f"本测试与 static/index.html 的引用")

    def test_vendor_assets_served(self):
        for name, _ in self.VENDOR.items():
            r = self.client.get(f"/static/vendor/{name}")
            self.assertEqual(r.status_code, 200, name)
            self.assertIn("javascript", r.headers.get("content-type", ""), name)
            self.assertGreater(len(r.content), 50_000, f"{name} 疑似截断")

    def test_index_loads_only_local_resources(self):
        with open(os.path.join(self.ROOT, "static", "index.html"),
                  encoding="utf-8") as f:
            html = f.read()
        self.assertIn("/static/vendor/echarts.min.js", html)
        self.assertIn("/static/vendor/html2canvas.min.js", html)
        # 真正会发起请求的标签不得指外部；<a href>（用户点击跳转）允许
        for tag in ("script", "link", "img", "iframe", "source", "video", "audio"):
            self.assertNotRegex(
                html, rf'<{tag}[^>]*\s(src|href)="https?://',
                f"<{tag}> 引用了外部资源——应本地 vendored")
        self.assertNotIn("fonts.googleapis.com", html)


class TestCredentialKeyApi(ServerDirCase):
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


class TestGatewayPick(ServerDirCase):
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
