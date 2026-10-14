"""api 组：`provider.model_profiles`（模型级「上限 + 类型」声明，ADR-0117 / 0118）。

覆盖两条链路：

- **写入侧**（`PUT /api/config` → `_validate_providers`）：规范化（模型名 trim、kinds 按
  白名单归序、缺省 `["llm"]` 不落盘、`max_ctx` 非正/非有限视为未配置）与显式拒绝
  （缺模型名、同模型重复声明、kinds 越界、条目非对象、名字超长、超过 64 条）；
- **解析侧**（`match_deployments`）：`max_ctx` 与 `kinds` **只认** `model_profiles`，
  不向 `deployments` 兜底（ADR-0118：两处都能算同一字段迟早排出两种结果）；部署环境
  只贡献报告来源（deploy_label / quant / kv_quant / hardware / framework / params）。
- **旧配置迁移**（`_migrate_profiles`，读取时在内存里做、不写盘）：把写在部署环境里的
  `max_ctx`/`kinds` 物化成模型声明，所以老配置的钳制与类型口径**逐项不变**；部署里的
  原值保留，之后两份数据允许分叉。这里重点兜两个事故：① 迁移漏了 → 上限当场静默失效
  （比报错更糟）；② 迁移造出重名条目 → 下次保存被 400 拒掉。

与 `tests/unit/test_config_api*.py` 的分工：那里管 provider 主体（名称/地址/部署卡）的
校验，这里只管新增的这一个数组与它的合并口径。

运行（须从仓库根目录）：
    python -m pytest tests/api/test_model_profiles.py -q
    python -m unittest tests.api.test_model_profiles -v
"""
import json
import pathlib
import unittest

from tests._shared import ServerDirCase

import server


def _prov(**kw):
    p = {"name": "p1", "gateway_url": "http://127.0.0.1:4000"}
    p.update(kw)
    return p


class TestModelProfilesApi(ServerDirCase):
    def _put(self, provider):
        return self.client.put("/api/config", json={"providers": [provider]})

    def _get_profiles(self):
        return self.client.get("/api/config").json()["providers"][0].get("model_profiles")

    def test_roundtrip_and_normalization(self):
        r = self._put(_prov(model_profiles=[
            {"model": "  m1 ", "kinds": ["ocr", "llm"], "max_ctx": 131072},   # 归序 → [llm, ocr]
            {"model": "m2", "kinds": ["llm"]},                                # 缺省值不落盘
            {"model": "m3", "kinds": [], "max_ctx": 0},                       # 空/非正 → 丢字段
        ]))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._get_profiles(), [
            {"model": "m1", "kinds": ["llm", "ocr"], "max_ctx": 131072},
            {"model": "m2"},
            {"model": "m3"},
        ])

    def test_non_positive_and_bool_max_ctx_dropped(self):
        for bad in (-8, 0, True):
            with self.subTest(max_ctx=bad):
                self.assertEqual(self._put(_prov(model_profiles=[
                    {"model": "m", "max_ctx": bad}])).status_code, 200)
                self.assertEqual(self._get_profiles(), [{"model": "m"}])

    def test_non_finite_max_ctx_dropped(self):
        """NaN / Infinity：标准 JSON 没有这两个字面量，但 `json.loads` 会接受——
        所以必须走 `math.isfinite` 守卫（与 deployments 同口径）。只能手拼原始
        请求体发送（TestClient 的 json= 编码阶段就会拒绝非有限值）。"""
        for lit in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(max_ctx=lit):
                body = ('{"providers":[{"name":"p1","gateway_url":"http://127.0.0.1:4000",'
                        '"model_profiles":[{"model":"m","max_ctx":' + lit
                        + '}' + ']' + '}' + ']' + '}')
                self.assertEqual(json.loads(body.replace(lit, "1")),   # 形状自检
                                 {"providers": [{"name": "p1",
                                                 "gateway_url": "http://127.0.0.1:4000",
                                                 "model_profiles": [{"model": "m",
                                                                      "max_ctx": 1}]}]})
                r = self.client.put("/api/config", content=body.encode(),
                                    headers={"Content-Type": "application/json"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(self._get_profiles(), [{"model": "m"}])

    def test_rejects(self):
        cases = [
            ([{"kinds": ["llm"]}], "缺少模型名"),
            ([{"model": "m"}, {"model": "m"}], "重复声明"),
            ([{"model": "m", "kinds": ["vlm"]}], "kinds 元素须取自"),
            ([{"model": "m", "kinds": "llm"}], "kinds 须为数组"),
            (["m"], "必须是对象"),
            ([{"model": "x" * 129}], "≤128 字符"),
            ([{"model": f"m{i}"} for i in range(65)], "最多 64 条"),
        ]
        for profiles, frag in cases:
            with self.subTest(frag=frag):
                r = self._put(_prov(model_profiles=profiles))
                self.assertEqual(r.status_code, 400, r.text)
                self.assertIn(frag, r.json()["detail"])

    def test_legacy_deployments_surface_as_profiles_on_read(self):
        """旧配置（上限只写在部署环境里）读取时物化成声明：抽屉里看得见、改得动。

        迁移**只在内存里**——GET 不得改写 config.json（边界：不静默改用户配置），
        落盘发生在用户点「保存配置」之后。这条同时兜住"迁移漏了 → 上限静默失效"。
        """
        self.assertEqual(self._put(_prov(deployments=[
            {"label": "D", "models": ["m"], "max_ctx": 262144,
             "kinds": ["llm", "ocr"]}])).status_code, 200)
        raw = json.loads(pathlib.Path(server.CONFIG_PATH).read_text(encoding="utf-8"))
        dep = raw["providers"][0]["deployments"][0]
        self.assertEqual(dep["max_ctx"], 262144)
        self.assertEqual(raw["providers"][0].get("model_profiles"), [])   # 读没写盘
        # 下发给前端的是迁移后的视图（上限/类型的唯一计算来源）
        self.assertEqual(self._get_profiles(),
                         [{"model": "m", "kinds": ["llm", "ocr"], "max_ctx": 262144}])
        # 前端把视图存回来 → 这次才真正落盘，且部署里的历史值原样不动
        self.assertEqual(self._put(_prov(
            deployments=[dep], model_profiles=self._get_profiles())).status_code, 200)
        raw = json.loads(pathlib.Path(server.CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertEqual(raw["providers"][0]["model_profiles"],
                         [{"model": "m", "kinds": ["llm", "ocr"], "max_ctx": 262144}])
        self.assertEqual(raw["providers"][0]["deployments"][0]["max_ctx"], 262144)

    def test_absent_array_stays_absent_as_empty(self):
        """旧配置没有该字段：下发为空数组，不得凭空造条目。"""
        self.assertEqual(self._put(_prov()).status_code, 200)
        self.assertEqual(self._get_profiles(), [])


class TestProfileResolution(unittest.TestCase):
    """读取侧合并口径（`match_deployments` 是唯一合并点）。"""

    def test_profile_only(self):
        p = {"model_profiles": [{"model": "k3", "kinds": ["llm", "ocr"], "max_ctx": 262144}]}
        info = server.match_deployments(p, ["k3"])
        self.assertEqual(info["k3"]["kinds"], ["llm", "ocr"])
        self.assertEqual(info["k3"]["max_ctx"], 262144)
        # 云端没有部署卡：这些字段留空串而不是缺键，前端按空值省略该段
        self.assertEqual(info["k3"]["deploy_label"], "")
        self.assertEqual(info["k3"]["hardware"], "")
        self.assertNotIn("other", info)

    def test_profile_wins_over_deployment(self):
        p = {"deployments": [{"label": "旧部署", "models": ["m"], "max_ctx": 8192,
                              "kinds": ["llm"], "hardware": "A100"}],
             "model_profiles": [{"model": "m", "kinds": ["asr"], "max_ctx": 32768}]}
        info = server.match_deployments(p, ["m"])["m"]
        self.assertEqual(info["kinds"], ["asr"])
        self.assertEqual(info["max_ctx"], 32768)
        self.assertEqual(info["hardware"], "A100")     # 部署侧的硬件信息不被抹掉
        self.assertEqual(info["deploy_label"], "旧部署")

    def test_no_fallback_to_deployment(self):
        """关键回归（ADR-0118）：解析层**不得**再从部署环境读上限/类型。

        旧行为是"profile 没写 max_ctx 就回退部署里的值"，现在必须断掉：部署里的
        131072 是历史值，生效与否只取决于它有没有被迁进 model_profiles。
        """
        p = {"deployments": [{"models": ["m"], "max_ctx": 131072, "kinds": ["ocr"]}],
             "model_profiles": [{"model": "m", "kinds": ["asr"]}]}
        info = server.match_deployments(p, ["m"])["m"]
        self.assertEqual(info["kinds"], ["asr"])
        self.assertIsNone(info["max_ctx"])

    def test_defaults_and_no_hit(self):
        p = {"model_profiles": [{"model": "m"}]}
        self.assertEqual(server.match_deployments(p, ["m"])["m"]["kinds"], ["llm"])
        self.assertIsNone(server.match_deployments(p, ["m"])["m"]["max_ctx"])
        self.assertEqual(server.match_deployments(p, ["zzz"]), {})

    def test_helpers(self):
        p = {"model_profiles": [{"model": "m", "max_ctx": 100}]}
        self.assertEqual(server._resolve_model_profile(p, "m")["max_ctx"], 100)
        self.assertIsNone(server._resolve_model_profile(p, "other"))
        self.assertIsNone(server._resolve_model_profile({}, "m"))
        self.assertEqual(server._profile_kinds({}), ["llm"])
        self.assertEqual(server._profile_kinds({"kinds": []}), ["llm"])
        self.assertEqual(server._profile_kinds({"kinds": ["tts"]}), ["tts"])

    def test_deployment_contributes_metadata_only(self):
        """解析层里部署环境只剩报告来源：量化/硬件照旧，上限与类型不再由它决定。"""
        p = {"deployments": [{"label": "D", "models": ["m"], "max_ctx": 4096,
                              "kinds": ["ocr"], "quants": ["Q8_0"], "kv_quants": ["Q4"],
                              "active": True}]}
        info = server.match_deployments(p, ["m"])["m"]
        self.assertEqual(info["quant"], "Q8_0")
        self.assertEqual(info["kv_quant"], "Q4")
        self.assertEqual(info["deploy_label"], "D")
        self.assertIsNone(info["max_ctx"])
        self.assertEqual(info["kinds"], ["llm"])


class TestLegacyMigration(unittest.TestCase):
    """`_migrate_profiles`：旧部署环境里的上限/类型 → 模型声明（ADR-0118）。"""

    def test_moves_ctx_and_kinds(self):
        p = {"deployments": [{"label": "D", "models": ["a", "b"], "max_ctx": 262144,
                              "kinds": ["llm", "ocr"], "hardware": "2080Ti"}]}
        self.assertEqual(server._migrate_profiles(p), [
            {"model": "a", "kinds": ["llm", "ocr"], "max_ctx": 262144},
            {"model": "b", "kinds": ["llm", "ocr"], "max_ctx": 262144},
        ])
        # 原值留在部署里（允许两份数据分叉），且入参不被改写
        self.assertEqual(p["deployments"][0]["max_ctx"], 262144)
        self.assertEqual(p["deployments"][0]["kinds"], ["llm", "ocr"])

    def test_existing_profile_wins_and_no_duplicate(self):
        """已有声明的模型不得再搬，否则重名条目会让下次保存吃 400。"""
        p = {"deployments": [{"models": ["m"], "max_ctx": 8192, "kinds": ["ocr"]}],
             "model_profiles": [{"model": "m", "max_ctx": 32768}]}
        self.assertEqual(server._migrate_profiles(p), [{"model": "m", "max_ctx": 32768}])

    def test_only_the_active_deployment_is_moved(self):
        """一模型挂多套部署：与 _resolve_deployment 同规则，只搬生效那套的说法。"""
        p = {"deployments": [{"label": "旧", "models": ["m"], "max_ctx": 8192},
                             {"label": "新", "models": ["m"], "max_ctx": 131072,
                              "active": True}]}
        self.assertEqual(server._migrate_profiles(p), [{"model": "m", "max_ctx": 131072}])

    def test_legacy_single_kind_field_supported(self):
        p = {"deployments": [{"models": ["m"], "kind": "ocr"}]}
        self.assertEqual(server._migrate_profiles(p), [{"model": "m", "kinds": ["ocr"]}])

    def test_nothing_to_say_makes_no_entry(self):
        """没上限也没类型的部署不造空条目（空条目会污染列表且无意义）。"""
        self.assertEqual(server._migrate_profiles(
            {"deployments": [{"models": ["m"], "hardware": "A100", "max_ctx": 0}]}), [])
        self.assertEqual(server._migrate_profiles({"deployments": []}), [])
        self.assertEqual(server._migrate_profiles({}), [])

    def test_garbage_is_skipped(self):
        p = {"deployments": [
            "not-a-dict",
            {"models": [""], "max_ctx": 4096},
            {"models": [None], "max_ctx": 4096},
            {"models": ["m"], "max_ctx": "abc"},          # 坏上限 → 不造条目
            {"models": ["n"], "max_ctx": "8192"},         # 数字字符串 → 收下
            {"models": ["k"], "kinds": ["vlm"], "max_ctx": 100},   # 越界 kind 丢弃
        ]}
        self.assertEqual(server._migrate_profiles(p),
                         [{"model": "n", "max_ctx": 8192},
                          {"model": "k", "max_ctx": 100}])

    def test_non_dict_deployment_entry_does_not_crash(self):
        """手改 config.json 混进字符串条目：解析与迁移都只能跳过，不能 500。"""
        p = {"deployments": ["oops", {"models": ["m"], "max_ctx": 4096}]}
        self.assertEqual(server._migrate_profiles(p), [{"model": "m", "max_ctx": 4096}])
        self.assertEqual(server.match_deployments(p, ["m"])["m"]["deploy_label"], "")

    def test_stops_at_server_side_cap(self):
        """迁移不得造出校验会拒（400）的配置：到 64 条即停。"""
        p = {"deployments": [{"models": [f"m{i}", ], "max_ctx": 1024}
                             for i in range(server._MAX_MODEL_PROFILES + 20)]}
        self.assertEqual(len(server._migrate_profiles(p)), server._MAX_MODEL_PROFILES)


if __name__ == "__main__":
    unittest.main()
