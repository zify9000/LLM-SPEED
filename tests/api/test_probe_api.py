"""api 组：`POST /api/probe`（Provider 抽屉里「测这条网关地址通不通」）的接口回归。

覆盖：
- 返回形状只有两个标量 `{ok, ms}`（可达性开关，不是响应内容读取通道）
- 探测**不带任何凭据**（ADR-0081 对 `pick_gateway_url` 的同一要求，这里口径共用
  同一个 `_probe_url`，所以边界也必须一样）
- 延迟下限 1ms（0.0002s 不报 0，UI 上不会出现"0ms"这种"瞬时"错觉）
- 非 http/https（含漏写协议的裸 `host:port`）一律 400，且**一次网络都不发**
- 跨源写请求被同源中间件挡在门外（Origin 不匹配 / Sec-Fetch-Site 非 same-origin）

与 `test_server_guard.py` 的分工：那里测"择优选路"这条**内部**链路（
`pick_gateway_url` 按序取第一个可达 + 不带凭据），这里测抽屉里**单地址**这条新链路。

运行（须从仓库根目录）：
    python -m pytest tests/api/test_probe_api.py -q
    python -m unittest discover -s tests/api -p test_probe_api.py
    python -m unittest tests.api.test_probe_api -v
"""
import asyncio
import unittest
from unittest import mock

from tests._shared import ServerDirCase

import server


class TestProbeEndpoint(ServerDirCase):
    def _probe(self, payload, **headers):
        return self.client.post("/api/probe", json=payload, headers=headers)

    def test_shape_and_no_credential(self):
        """可达 → {ok:true, ms:int}，且探测请求头是空的（不带 Authorization）。"""
        seen = []

        async def fake_probe(url, headers, timeout=3.0):
            seen.append((url, dict(headers)))
            return 0.012

        with mock.patch.object(server, "_probe_url", fake_probe):
            r = self._probe({"url": "http://127.0.0.1:4000/"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"ok": True, "ms": 12})
        self.assertEqual(len(seen), 1)
        # 地址原样透传（不加工、不补协议），headers 为空 = 不带凭据
        self.assertEqual(seen[0][0], "http://127.0.0.1:4000/")
        self.assertEqual(seen[0][1], {})

    def test_unreachable_is_ok_false(self):
        """不可达不是错误：HTTP 200 + ok=false，前端才有统一文案可写。"""
        async def fake_probe(url, headers, timeout=3.0):
            return None

        with mock.patch.object(server, "_probe_url", fake_probe):
            r = self._probe({"url": "http://127.0.0.1:59999"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"ok": False, "ms": None})

    def test_ms_floor_is_one(self):
        """亚毫秒延迟取整到 1ms，不出现 "0ms"。"""
        async def fake_probe(url, headers, timeout=3.0):
            return 0.0002

        with mock.patch.object(server, "_probe_url", fake_probe):
            self.assertEqual(self._probe({"url": "http://a:1"}).json(), {"ok": True, "ms": 1})

    def test_rejects_non_http_scheme_without_touching_network(self):
        """协议白名单挡在探测之前：漏写协议的裸地址、非 http(s)、空值都 400，
        且 `_probe_url` 一次都不被调用（否则就成了任意 scheme 的探测通道）。"""
        calls = []

        async def fake_probe(url, headers, timeout=3.0):
            calls.append(url)
            return 0.001

        for bad in ("ftp://x", "file:///etc/passwd", "127.0.0.1:4000", "",
                    "http://", "  "):
            with mock.patch.object(server, "_probe_url", fake_probe):
                r = self._probe({"url": bad})
            self.assertEqual(r.status_code, 400, f"{bad!r} 应被拒：{r.text}")
        with mock.patch.object(server, "_probe_url", fake_probe):
            self.assertEqual(self._probe({}).status_code, 400)   # 缺字段
        self.assertEqual(calls, [])

    def test_cross_origin_blocked(self):
        """POST 属写方法，同源守卫必须覆盖它：否则本机接口就成了跨源页面手里的
        内网可达性探针（DNS rebinding / 跨源表单两条路都已在 ADR-0081 堵死）。"""
        r = self._probe({"url": "http://127.0.0.1:4000"},
                        Origin="http://evil.example")
        self.assertEqual(r.status_code, 403, r.text)
        r = self._probe({"url": "http://127.0.0.1:4000"},
                        **{"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403, r.text)


class TestProbeHelper(unittest.TestCase):
    """`_probe_url` 的可达判据（端点与择优共用，改它会同时改两条链路的语义）。"""

    def _latency(self, status_of):
        """用 MockTransport 顶掉真网络，返回 (延迟 or None, 实际请求过的路径)。"""
        import httpx

        paths = []

        async def handler(request):
            paths.append(request.url.path)
            return httpx.Response(status_of(request.url.path))

        class Client(httpx.AsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(handler)
                super().__init__(*a, **kw)

        with mock.patch.object(httpx, "AsyncClient", Client):
            lat = asyncio.run(server._probe_url("http://gw:4000", {}))
        return lat, paths

    def test_401_counts_as_reachable(self):
        """无鉴权拿到 401 算可达（可达性与凭据无关，ADR-0081），且拿到非 404
        就短路——不再试第二个前缀。"""
        lat, paths = self._latency(lambda path: 401)
        self.assertIsNotNone(lat)
        self.assertEqual(paths, ["/v1/models"])

    def test_both_404_is_unreachable(self):
        """两种前缀都 404 → 判不可达（多半填成根路径或非 OpenAI 兼容接口）。"""
        lat, paths = self._latency(lambda path: 404)
        self.assertIsNone(lat)
        self.assertEqual(paths, ["/v1/models", "/models"])

    def test_5xx_is_unreachable(self):
        lat, _ = self._latency(lambda path: 503)
        self.assertIsNone(lat)


if __name__ == "__main__":
    unittest.main()
