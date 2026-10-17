"""api 组：PUT /api/config 写入契约（校验、归一、凭据单向上行）+ GET /api/config
场景元数据契约 + _validate_bench_cfg 测速矩阵钳制（ADR-0001）。

自 tests/test_config_api.py 拆分：TestConfigApi / TestValidateBenchCfg 搬入，
并把放错类的 GET /api/config 契约用例（test_config_scenarios_carry_kind_and_ladder）
从 TestBenchStartAssembly 迁回。环境隔离改用 tests._shared.ServerDirCase：
API_KEY_P1 由 save_env 保存/还原（原 tearDown 无条件 pop，会把 shell 里本来
就有的值一并抹掉），并补 server._HIST_CACHE.clear()。

    python -m pytest tests/api/test_config.py -q
    python -m unittest discover -s tests/api -p test_config.py
    python -m unittest tests.api.test_config -v      # 单模块入口
（两种入口都从仓库根目录运行。）
"""
import json
import os
import unittest

from fastapi import HTTPException

from bench import SCENARIOS, TIMEOUT_S_MAX
import server
from tests._shared import ServerDirCase
from tests.api.test_bench_start import _prov


class TestConfigApi(ServerDirCase):
    def setUp(self):
        super().setUp()
        self.save_env("API_KEY_P1")   # 端点会写 os.environ：原值保存、用例后还原

    def test_roundtrip(self):
        dep = {"label": "部署A", "models": ["m1", "m2"], "quants": ["Q8_0"],
               "kv_quants": ["Q8_0"],
               "hardware": "RTX", "framework": "llama.cpp", "params": "-c 4096",
               "max_ctx": 4096}
        r = self.client.put("/api/config", json={"providers": [_prov(local=True, deployments=[dep])]})
        self.assertEqual(r.status_code, 200)
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        # 上限写在部署里是旧形态（ADR-0122 已废弃）：保存不再把它写回，而是折算进声明
        self.assertNotIn("max_ctx", on_disk["providers"][0]["deployments"][0])
        self.assertEqual(on_disk["providers"][0]["model_profiles"],
                         [{"model": "m1", "max_ctx": 4096},
                          {"model": "m2", "max_ctx": 4096}])
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["quants"], ["Q8_0"])
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["kv_quants"], ["Q8_0"])
        self.assertNotIn("quant", on_disk["providers"][0]["deployments"][0])
        got = self.client.get("/api/config").json()
        self.assertEqual(got["providers"][0]["name"], "p1")
        self.assertEqual(got["providers"][0]["local"], True)
        self.assertEqual(got["providers"][0]["deployments"][0]["quants"], ["Q8_0"])
        self.assertNotIn("api_key", json.dumps(on_disk))

    def test_deployment_groups_roundtrip(self):
        """部署环境分组：deployment_groups（组长顺序表）按序落盘；每条部署的
        group 归属白名单化；空/空白 group 不落盘（未分组保持零字段）。分组纯
        组织用，不影响解析字段。"""
        deps = [
            {"label": "在线", "models": ["m1"], "group": "生产"},
            {"label": "离线", "models": ["m2"], "group": "测试"},
            {"label": "无组", "models": ["m3"], "group": ""},
        ]
        r = self.client.put("/api/config", json={"providers": [
            _prov(local=True, deployments=deps,
                  deployment_groups=["生产", "测试"])]})
        self.assertEqual(r.status_code, 200, r.text)
        on_disk = json.load(open(server.CONFIG_PATH,
                                 encoding="utf-8"))["providers"][0]
        self.assertEqual(on_disk["deployment_groups"], ["生产", "测试"])
        self.assertEqual(on_disk["deployments"][0]["group"], "生产")
        self.assertEqual(on_disk["deployments"][1]["group"], "测试")
        self.assertNotIn("group", on_disk["deployments"][2].keys())  # 空分组不落盘
        # 组名去空白去重、超长拒绝；未登记组名不强制回填（前端保存时自愈）
        self.assertEqual(self.client.put("/api/config", json={"providers": [
            _prov(local=True, deployment_groups=[" 生产 ", "生产", ""])]}).status_code, 200)
        got = self.client.get("/api/config").json()["providers"][0]
        self.assertEqual(got["deployment_groups"], ["生产"])
        bad = self.client.put("/api/config", json={"providers": [
            _prov(local=True, deployment_groups=["x" * 41])]}).status_code
        self.assertEqual(bad, 400)

    def test_quants_normalize(self):
        """量化多选（quants 数组落盘）：旧单值 quant 字符串（含逗号分隔多值）
        PUT 后归一为 quants、不再写 quant（与 kinds/kind 先例同口径）；多值
        原样保留；非法输入（非数组/单项超长/超 8 项）一律 400。kv_quants
        校验口径与 quants 完全一致。"""
        deps = [
            {"label": "旧字段", "models": ["m1"], "quant": "Q8_0, Q4_K_M"},
            {"label": "多值", "models": ["m2"], "quants": ["Q8_0", "Q4_K_M"]},
        ]
        r = self.client.put("/api/config",
                            json={"providers": [_prov(local=True, deployments=deps)]})
        self.assertEqual(r.status_code, 200, r.text)
        on_disk = json.load(open(server.CONFIG_PATH,
                                 encoding="utf-8"))["providers"][0]["deployments"]
        self.assertEqual(on_disk[0]["quants"], ["Q8_0", "Q4_K_M"])   # 旧字段逗号拆分
        self.assertNotIn("quant", on_disk[0])
        self.assertEqual(on_disk[1]["quants"], ["Q8_0", "Q4_K_M"])   # 多值原样保留
        # 非法输入
        bad = [
            {"quants": "Q8_0"},                              # 非数组
            {"quants": ["x" * 65]},                          # 单项超 64 字符
            {"quants": [f"Q{i}" for i in range(9)]},         # 超 8 项
        ]
        for q in bad:
            dep = {"label": "x", "models": ["m3"], **q}
            r = self.client.put("/api/config",
                                json={"providers": [_prov(local=True, deployments=[dep])]})
            self.assertEqual(r.status_code, 400, q)
        # kv_quants：校验口径与 quants 完全一致（同一循环镜像，取代表样本）
        bad_kv = [
            {"kv_quants": "Q8_0"},                           # 非数组
            {"kv_quants": [f"Q{i}" for i in range(9)]},      # 超 8 项
        ]
        for q in bad_kv:
            dep = {"label": "x", "models": ["m3"], **q}
            r = self.client.put("/api/config",
                                json={"providers": [_prov(local=True, deployments=[dep])]})
            self.assertEqual(r.status_code, 400, q)

    def test_invalid_rejected(self):
        bad = [
            {"providers": [_prov(name="名字")]},                         # 非法名称
            {"providers": [_prov(gateway_url="localhost:4000")]},       # 缺协议
            {"providers": [_prov(), _prov()]},                          # 名称重复
            {"providers": ["not-a-dict"]},
        ]
        for body in bad:
            self.assertEqual(self.client.put("/api/config", json=body).status_code, 400, body)

    def test_empty_providers_allowed(self):
        """空 providers 是文档化的空态（前端空态引导新增）：必须能保存。

        旧实现拒绝空数组，用户删掉最后一个 provider 后保存 400、只能手改
        config.json 才能回到空态。"""
        r = self.client.put("/api/config", json={"providers": []})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["providers"], 0)
        self.assertEqual(self.client.get("/api/config").json()["providers"], [])
        self.assertEqual(json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"], [])

    def test_max_ctx_sanitize(self):
        """上限旧写法的净化口径（ADR-0122）：3.7 取整、0 当未配置。字段本身已废弃，
        因此不再落回 `deployments`，而是折算进 `model_profiles`。"""
        r = self.client.put("/api/config", json={"providers": [
            _prov(deployments=[{"label": "x", "models": ["m"], "max_ctx": 3.7}]),
            _prov(name="p2", deployments=[{"label": "y", "models": ["m"], "max_ctx": 0}]),
        ]})
        self.assertEqual(r.status_code, 200)
        deps = json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"]
        self.assertEqual(deps[0]["model_profiles"], [{"model": "m", "max_ctx": 3}])
        self.assertEqual(deps[1]["model_profiles"], [])               # 0 视为未配置
        for p in deps:
            self.assertNotIn("max_ctx", p["deployments"][0])          # 旧字段不再落盘

    def test_gateway_urls_normalize_and_reject(self):
        """多地址：gateway_urls 规范化（去重、首选同步 gateway_url）；旧单地址
        字段兼容；非法输入（非数组/空数组/两字段皆空/缺协议/超 8 个）一律 400。"""
        with self.subTest(case="规范化与旧字段迁移"):
            r = self.client.put("/api/config", json={"providers": [
                _prov(gateway_urls=["http://a:1", "http://b:2", "http://a:1"]),
                _prov(name="p2", gateway_url="http://legacy:3"),
            ]})
            self.assertEqual(r.status_code, 200)
            provs = json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"]
            self.assertEqual(provs[0]["gateway_urls"], ["http://a:1", "http://b:2"])   # 去重
            self.assertEqual(provs[0]["gateway_url"], "http://a:1")                    # 首选=首个
            self.assertEqual(provs[1]["gateway_urls"], ["http://legacy:3"])            # 旧字段迁移
            got = self.client.get("/api/config").json()["providers"]
            self.assertEqual(got[0]["gateway_urls"], ["http://a:1", "http://b:2"])
        with self.subTest(case="非法输入拒绝"):
            bad = [
                {"providers": [_prov(gateway_urls="not-a-list")]},               # 非数组
                {"providers": [_prov(gateway_urls=[])]},                         # 空数组
                {"providers": [_prov(gateway_urls=[] , gateway_url="")]},        # 两字段皆空
                {"providers": [_prov(gateway_urls=["localhost:1"])]},            # 缺协议
                {"providers": [_prov(gateway_urls=[f"http://h{i}:1" for i in range(9)])]},  # 超 8 个
            ]
            for body in bad:
                self.assertEqual(self.client.put("/api/config", json=body).status_code, 400, body)

    def test_active_deployment_roundtrip_and_conflict(self):
        """1 对多映射 + 激活：active 置位落盘、未置位不留字段；同模型多套部署、
        仅一套激活放行；同一模型同时激活多套部署 → 400（生效口径二义）。"""
        with self.subTest(case="激活置位落盘"):
            deps = [
                {"label": "环境A", "models": ["m1"], "active": True, "max_ctx": 8192},
                {"label": "环境B", "models": ["m1"], "max_ctx": 4096},   # 同模型备用，未激活
                {"label": "环境C", "models": ["m2"], "active": False},   # 显式 false = 未激活
            ]
            r = self.client.put("/api/config",
                                json={"providers": [_prov(local=True, deployments=deps)]})
            self.assertEqual(r.status_code, 200, r.text)
            on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"][0]["deployments"]
            self.assertEqual(on_disk[0].get("active"), True)
            self.assertNotIn("active", on_disk[1])
            self.assertNotIn("active", on_disk[2])
        with self.subTest(case="同模型重复激活拒绝"):
            deps = [
                {"label": "环境A", "models": ["m1"], "active": True},
                {"label": "环境B", "models": ["m1", "m2"], "active": True},
            ]
            r = self.client.put("/api/config",
                                json={"providers": [_prov(local=True, deployments=deps)]})
            self.assertEqual(r.status_code, 400)
            self.assertIn("m1", r.json()["detail"])
            # 不同模型各激活一套：不冲突，放行
            deps[1]["models"] = ["m2"]
            r = self.client.put("/api/config",
                                json={"providers": [_prov(local=True, deployments=deps)]})
            self.assertEqual(r.status_code, 200, r.text)

    def test_key_write_and_clear(self):
        self.client.put("/api/config", json={"providers": [_prov()]})
        r = self.client.put("/api/config/key", json={"provider": "p1", "key": "sk-abc#1"})
        self.assertEqual(r.status_code, 200)
        env = open(os.path.join(self.tmp.name, ".env"), encoding="utf-8").read()
        self.assertIn('API_KEY_P1="sk-abc#1"', env)                     # 引号包裹，# 不破坏解析
        self.assertTrue(self.client.get("/api/config").json()["providers"][0]["has_key"])
        self.assertEqual(os.environ["API_KEY_P1"], "sk-abc#1")          # 即时生效

        r = self.client.put("/api/config/key", json={"provider": "p1", "clear": True})
        self.assertEqual(r.status_code, 200)
        env = open(os.path.join(self.tmp.name, ".env"), encoding="utf-8").read()
        self.assertNotIn("API_KEY_P1", env)
        self.assertFalse(self.client.get("/api/config").json()["providers"][0]["has_key"])

    def test_key_unknown_provider(self):
        self.assertEqual(self.client.put("/api/config/key",
                         json={"provider": "ghost", "key": "x"}).status_code, 400)

    def test_config_scenarios_carry_kind_and_ladder(self):
        """/api/config 场景列表带 kind 与阶梯：前端按 kind 过滤/分组与展示档位。

        （自 TestBenchStartAssembly 迁入：本是 GET /api/config 契约，与 bench/start
        装配无关。）"""
        r = self.client.get("/api/config")
        self.assertEqual(r.status_code, 200)
        scens = {s["key"]: s for s in r.json()["scenarios"]}
        self.assertEqual(scens["creative"]["kind"], "llm")
        self.assertEqual(scens["asr"]["kind"], "asr")
        self.assertEqual(scens["asr"]["ladder"], SCENARIOS["asr"]["ladder"])
        self.assertEqual(scens["ocr"]["kind"], "ocr")
        self.assertIsNone(scens["creative"]["ladder"])
        # 翻译场景：llm kind（同 chat 文本流路径）但带固定原文字长阶梯
        self.assertEqual(scens["translate"]["kind"], "llm")
        self.assertEqual(scens["translate"]["ladder"], SCENARIOS["translate"]["ladder"])
        self.assertEqual(scens["translate"]["unit"], "字原文")


class TestValidateBenchCfg(unittest.TestCase):
    """_validate_bench_cfg 测速矩阵钳制（ADR-0001）：类型/范围/条数不合法一律 400。"""

    def _cfg(self, **kw):
        cfg = {"models": ["m1"], "scenarios": ["creative"],
               "ctx_list": [0, 4096], "concurrencies": [1, 2]}
        cfg.update(kw)
        return cfg

    def test_invalid_clamped_400(self):
        bad = [
            self._cfg(ctx_list=[4 * 1048576 + 1]),          # ctx > 4M
            self._cfg(ctx_list=[-1]),                       # 负档位
            self._cfg(ctx_list=[0] * 17),                   # 超 16 档
            self._cfg(scenarios=["ghost"]),                 # 场景非白名单
            self._cfg(models=[f"m{i}" for i in range(33)]),  # models 超 32
            self._cfg(models=[""]),                         # 空模型名
            self._cfg(concurrencies=[0]),                   # 并发低于下界
            self._cfg(concurrencies=[65]),                  # 并发高于上界
            self._cfg(concurrencies=[1.5]),                 # 并发非整数
        ]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)

    def test_valid_passes(self):
        # 缺省放行：各校验的 None 跳过分支统一在此验证（其余用例不再重复）
        server._validate_bench_cfg(self._cfg())
        server._validate_bench_cfg(self._cfg(ctx_list=[4 * 1048576],
                                             concurrencies=[64]))   # 边界值放行

    def test_reply_mode_validated(self):
        """回复模式（ADR-0021）：仅 free/echo 放行，其余值 400（含非字符串）。"""
        for bad in ("rewrite", ["echo"]):
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(self._cfg(reply_mode=bad))
            self.assertEqual(cm.exception.status_code, 400, bad)
        server._validate_bench_cfg(self._cfg(reply_mode="free"))
        server._validate_bench_cfg(self._cfg(reply_mode="echo"))

    def test_translate_ladder_validated(self):
        """翻译场景自定义阶梯（ADR-0034）：translate_ladder ≤16 档、逐值
        1~65536 整数（字原文）；越界/类型错 400（缺省放行见 test_valid_passes，
        引擎侧缺省取场景默认阶梯）。"""
        bad = [self._cfg(translate_ladder=[0]),          # 低于下界 1
               self._cfg(translate_ladder=[65537]),      # 超上限
               self._cfg(translate_ladder=[1.5]),        # 非整数
               self._cfg(translate_ladder=[50] * 17)]    # 超 16 档
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(translate_ladder=[50, 6400]))

    def test_agent_ladders_validated(self):
        """Agent 缓存×指令矩阵双阶梯（ADR-0042）：agent_cache_ladder ≤16 档、
        逐值 0~4M tokens（已缓存上下文）；agent_inst_ladder ≤16 档、逐值
        16~65536 tokens（单步指令长度）。越界/类型错 400（缺省放行见
        test_valid_passes，引擎侧缺省取场景默认阶梯）；旧连续任务链键（agent_turns /
        agent_turn_delta / agent_turns_p1/p2 / agent_cold_ctx /
        agent_phase2_base）不再校验，传入被忽略；纯 Agent 场景 ctx_list 可空。"""
        M = 4 * 1048576
        bad_cache = [self._cfg(agent_cache_ladder=[-1]),        # 低于下界 0
                     self._cfg(agent_cache_ladder=[M + 1]),     # 超上限
                     self._cfg(agent_cache_ladder=[True]),      # bool 非整数
                     self._cfg(agent_cache_ladder=[0] * 17),    # 超 16 档
                     self._cfg(agent_cache_ladder="0,4096")]    # 非列表
        for cfg in bad_cache:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        bad_inst = [self._cfg(agent_inst_ladder=[8]),           # 低于下界 16
                    self._cfg(agent_inst_ladder=[65537]),       # 超上限
                    self._cfg(agent_inst_ladder=[512] * 17)]    # 超 16 档
        for cfg in bad_inst:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        # 边界值放行：cache 档含 0、上到 4M；inst 档 16~65536
        server._validate_bench_cfg(self._cfg(agent_cache_ladder=[0, M],
                                             agent_inst_ladder=[16, 65536]))
        # 旧连续任务链键不再校验：传入被忽略、不 400
        server._validate_bench_cfg(self._cfg(
            agent_turns=8, agent_turn_delta=[256, 2048], agent_turns_p1=4,
            agent_turns_p2=4, agent_cold_ctx=10240, agent_phase2_base=8192))
        # 纯 Agent 场景不以上下文档位为变量：ctx_list 可为空
        server._validate_bench_cfg(self._cfg(scenarios=["agent"], ctx_list=[]))

    def test_media_ladders_validated(self):
        """媒体场景自定义阶梯：asr_ladder/ocr_ladder/tts_ladder 各 ≤16 档、
        逐值 int 且落在对应范围（asr 1~3600 秒 / ocr 1~64 张 / tts 1~65536 字）；
        越界/非 int/非 list 一律 400（三个字段共用一条校验循环，类型/档数子句
        各取一个代表）；空数组合法（引擎回退场景默认阶梯）。"""
        bad = [self._cfg(asr_ladder=[0]),              # 低于下界 1
               self._cfg(asr_ladder=[3601]),           # 超上限
               self._cfg(asr_ladder=[5.5]),            # 非整数
               self._cfg(asr_ladder=[5] * 17),         # 超 16 档
               self._cfg(asr_ladder="5,15"),           # 非列表
               self._cfg(ocr_ladder=[0]),              # ocr 范围下界
               self._cfg(ocr_ladder=[65]),             # ocr 范围上界
               self._cfg(tts_ladder=[0]),              # tts 范围下界
               self._cfg(tts_ladder=[65537])]          # tts 范围上界
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        # 合法放行：边界值 / 恰 16 档 / 空数组（回退默认）
        server._validate_bench_cfg(self._cfg(asr_ladder=[1, 3600],
                                             ocr_ladder=[1, 64],
                                             tts_ladder=[1, 65536]))
        server._validate_bench_cfg(self._cfg(asr_ladder=[5] * 16))
        server._validate_bench_cfg(self._cfg(asr_ladder=[], ocr_ladder=[],
                                             tts_ladder=[]))

    def test_repeats_timeout_max_tokens_thinking_validated(self):
        """repeats/timeout_s/max_tokens/thinking 白名单：bench_start 直接透传
        引擎，非法值一律 400——repeats 1~5（对齐引擎钳制）、timeout_s 正数
        ≤TIMEOUT_S_MAX、max_tokens 正整数 ≤32768、thinking 取引擎三值枚举。

        timeout_s 上界取自 bench.TIMEOUT_S_MAX（不写死数字）：该字段是「连续无
        输出多久判为断开」的 idle 超时，超长上下文 prefill 会长时间零输出，
        上界必须够宽（2026-09-24 256K 档事故：按总时长直觉填 1200 会误掐）。"""
        bad = [
            self._cfg(repeats=0), self._cfg(repeats=6),
            self._cfg(repeats=True),                       # bool 冒充整数
            self._cfg(timeout_s=0), self._cfg(timeout_s=TIMEOUT_S_MAX + 1),
            self._cfg(timeout_s="600"),
            self._cfg(max_tokens=0), self._cfg(max_tokens=32769),
            self._cfg(max_tokens="512"),
            self._cfg(thinking="off"),
        ]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(repeats=5,
                                             timeout_s=TIMEOUT_S_MAX,
                                             max_tokens=32768))
        for th in ("auto", "enabled", "disabled"):
            server._validate_bench_cfg(self._cfg(thinking=th))
        with self.subTest(case="timeout_s 上界容得下超长档位的 prefill 静默"):
            # 旧硬顶 3600 让 512K/1M 档（prefill 静默可超 1 小时）无从配置；
            # 同时拒绝把 3600 当成新上界。
            self.assertGreater(server.TIMEOUT_S_MAX, 3600)
            server._validate_bench_cfg(self._cfg(timeout_s=3601))   # 旧顶曾拒
            server._validate_bench_cfg(self._cfg(timeout_s=7200))


if __name__ == "__main__":
    unittest.main()
