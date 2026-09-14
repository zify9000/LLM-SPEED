"""配置写入接口回归（ADR-0001）：PUT /api/config 校验与原子写、PUT /api/config/key 凭据单向上行。

测试内把 server.BASE/CONFIG_PATH 指向临时目录，不触碰真实 config.json/.env。
"""
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from bench import SCENARIOS
import server


def _prov(**kw):
    p = {"name": "p1", "label": "P1", "gateway_url": "http://127.0.0.1:4000",
         "local": False, "deployments": []}
    p.update(kw)
    return p


class TestConfigApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._base, self._cfg = server.BASE, server.CONFIG_PATH
        self._global_key = server.GLOBAL_API_KEY
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.GLOBAL_API_KEY = ""
        self.client = TestClient(server.app)

    def tearDown(self):
        server.BASE, server.CONFIG_PATH = self._base, self._cfg
        server.GLOBAL_API_KEY = self._global_key
        os.environ.pop("API_KEY_P1", None)
        self.tmp.cleanup()

    def test_roundtrip(self):
        dep = {"label": "部署A", "models": ["m1", "m2"], "quants": ["Q8_0"],
               "kv_quants": ["Q8_0"],
               "hardware": "RTX", "framework": "llama.cpp", "params": "-c 4096",
               "max_ctx": 4096}
        r = self.client.put("/api/config", json={"providers": [_prov(local=True, deployments=[dep])]})
        self.assertEqual(r.status_code, 200)
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["max_ctx"], 4096)
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
        # kv_quants：校验口径与 quants 完全一致
        bad_kv = [
            {"kv_quants": "Q8_0"},                           # 非数组
            {"kv_quants": ["x" * 65]},                       # 单项超 64 字符
            {"kv_quants": [f"Q{i}" for i in range(9)]},      # 超 8 项
        ]
        for q in bad_kv:
            dep = {"label": "x", "models": ["m3"], **q}
            r = self.client.put("/api/config",
                                json={"providers": [_prov(local=True, deployments=[dep])]})
            self.assertEqual(r.status_code, 400, q)

    def test_invalid_rejected(self):
        bad = [
            {"providers": []},                                          # 空列表
            {"providers": [_prov(name="名字")]},                         # 非法名称
            {"providers": [_prov(gateway_url="localhost:4000")]},       # 缺协议
            {"providers": [_prov(), _prov()]},                          # 名称重复
            {"providers": ["not-a-dict"]},
        ]
        for body in bad:
            self.assertEqual(self.client.put("/api/config", json=body).status_code, 400, body)

    def test_max_ctx_sanitize(self):
        r = self.client.put("/api/config", json={"providers": [
            _prov(deployments=[{"label": "x", "models": ["m"], "max_ctx": 3.7}]),
            _prov(name="p2", deployments=[{"label": "y", "models": ["m"], "max_ctx": 0}]),
        ]})
        self.assertEqual(r.status_code, 200)
        deps = json.load(open(server.CONFIG_PATH, encoding="utf-8"))["providers"]
        self.assertEqual(deps[0]["deployments"][0]["max_ctx"], 3)       # 3.7 → int
        self.assertNotIn("max_ctx", deps[1]["deployments"][0])          # 0 视为未配置

    def test_gateway_urls_normalize(self):
        """多地址：gateway_urls 规范化（去重、首选同步 gateway_url）；旧单地址字段兼容。"""
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

    def test_gateway_urls_invalid(self):
        bad = [
            {"providers": [_prov(gateway_urls="not-a-list")]},               # 非数组
            {"providers": [_prov(gateway_urls=[])]},                         # 空数组
            {"providers": [_prov(gateway_urls=[] , gateway_url="")]},        # 两字段皆空
            {"providers": [_prov(gateway_urls=["localhost:1"])]},            # 缺协议
            {"providers": [_prov(gateway_urls=[f"http://h{i}:1" for i in range(9)])]},  # 超 8 个
        ]
        for body in bad:
            self.assertEqual(self.client.put("/api/config", json=body).status_code, 400, body)

    def test_active_deployment_roundtrip(self):
        """1 对多映射 + 激活：active 置位落盘、未置位不留字段；同模型
        多套部署、仅一套激活放行。"""
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

    def test_active_deployment_conflict_rejected(self):
        """同一模型同时激活多套部署 → 400（生效口径二义）。"""
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
        server._validate_bench_cfg(self._cfg())                     # 常规放行
        server._validate_bench_cfg(self._cfg(ctx_list=[4 * 1048576],
                                             concurrencies=[64]))   # 边界值放行

    def test_reply_mode_validated(self):
        """回复模式（ADR-0021）：仅 free/echo 放行，其余值 400。"""
        for bad in ("rewrite", "", 1, True, ["echo"]):
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(self._cfg(reply_mode=bad))
            self.assertEqual(cm.exception.status_code, 400, bad)
        server._validate_bench_cfg(self._cfg(reply_mode="free"))
        server._validate_bench_cfg(self._cfg(reply_mode="echo"))
        server._validate_bench_cfg(self._cfg())   # 缺省放行（引擎按 free）

    def test_translate_ladder_validated(self):
        """翻译场景自定义阶梯（ADR-0034）：translate_ladder ≤16 档、逐值
        1~65536 整数（字原文）；越界/类型错 400，缺省放行由引擎取场景默认。"""
        bad = [self._cfg(translate_ladder=[0]),          # 0 字非法
               self._cfg(translate_ladder=[-50]),        # 负档
               self._cfg(translate_ladder=[65537]),      # 超上限
               self._cfg(translate_ladder=[1.5]),        # 非整数
               self._cfg(translate_ladder=[True]),       # bool 非整数
               self._cfg(translate_ladder=[50] * 17),    # 超 16 档
               self._cfg(translate_ladder="6400")]       # 非列表
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(translate_ladder=[50, 6400]))
        server._validate_bench_cfg(self._cfg())   # 缺省放行

    def test_agent_ladders_validated(self):
        """Agent 缓存×指令矩阵双阶梯（ADR-0042）：agent_cache_ladder ≤16 档、
        逐值 0~4M tokens（已缓存上下文）；agent_inst_ladder ≤16 档、逐值
        16~65536 tokens（单步指令长度）。越界/类型错 400；缺省放行由引擎取
        场景默认阶梯；旧连续任务链键（agent_turns / agent_turn_delta /
        agent_turns_p1/p2 / agent_cold_ctx / agent_phase2_base）不再校验，
        传入被忽略；纯 Agent 场景 ctx_list 可空。"""
        M = 4 * 1048576
        bad_cache = [self._cfg(agent_cache_ladder=[-1]),        # 负档
                     self._cfg(agent_cache_ladder=[M + 1]),     # 超上限
                     self._cfg(agent_cache_ladder=[True]),      # bool 非整数
                     self._cfg(agent_cache_ladder=["4096"]),    # 字符串项
                     self._cfg(agent_cache_ladder=[1.5]),       # 非整数
                     self._cfg(agent_cache_ladder=[0] * 17),    # 超 16 档
                     self._cfg(agent_cache_ladder="0,4096")]    # 非列表
        for cfg in bad_cache:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        bad_inst = [self._cfg(agent_inst_ladder=[8]),           # 低于下界 16
                    self._cfg(agent_inst_ladder=[0]),
                    self._cfg(agent_inst_ladder=[65537]),       # 超上限
                    self._cfg(agent_inst_ladder=[True]),
                    self._cfg(agent_inst_ladder=[1.5]),
                    self._cfg(agent_inst_ladder=[512] * 17),    # 超 16 档
                    self._cfg(agent_inst_ladder=512)]           # 非列表
        for cfg in bad_inst:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        # 边界值放行：cache 档含 0、上到 4M；inst 档 16~65536
        server._validate_bench_cfg(self._cfg(agent_cache_ladder=[0, M],
                                             agent_inst_ladder=[16, 65536]))
        server._validate_bench_cfg(self._cfg())   # 缺省放行（引擎取场景默认）
        # 旧连续任务链键不再校验：传入被忽略、不 400
        server._validate_bench_cfg(self._cfg(
            agent_turns=8, agent_turn_delta=[256, 2048], agent_turns_p1=4,
            agent_turns_p2=4, agent_cold_ctx=10240, agent_phase2_base=8192))
        # 纯 Agent 场景不以上下文档位为变量：ctx_list 可为空
        server._validate_bench_cfg(self._cfg(scenarios=["agent"], ctx_list=[]))

    def test_media_ladders_validated(self):
        """媒体场景自定义阶梯：asr_ladder/ocr_ladder/tts_ladder 各 ≤16 档、
        逐值 int 且落在对应范围（asr 1~3600 秒 / ocr 1~64 张 / tts 1~65536 字）；
        越界（0 与上限+1）/非 list/float/bool 值一律 400；空数组合法
        （引擎回退场景默认阶梯）；缺省放行。"""
        bad = [self._cfg(asr_ladder=[0]),              # 低于下界 1
               self._cfg(asr_ladder=[3601]),           # 超上限
               self._cfg(asr_ladder=[5.5]),            # 非整数
               self._cfg(asr_ladder=[True]),           # bool 非整数
               self._cfg(asr_ladder="5,15"),           # 非列表
               self._cfg(ocr_ladder=[0]),
               self._cfg(ocr_ladder=[65]),
               self._cfg(ocr_ladder=[1.5]),
               self._cfg(ocr_ladder=[False]),
               self._cfg(tts_ladder=[0]),
               self._cfg(tts_ladder=[65537]),
               self._cfg(tts_ladder=[50.0]),
               self._cfg(tts_ladder=["50"]),
               self._cfg(asr_ladder=[5] * 17),         # 超 16 档
               self._cfg(ocr_ladder=[1] * 17),
               self._cfg(tts_ladder=[50] * 17)]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        # 合法放行：边界值 / 恰 16 档 / 空数组（回退默认）/ 缺省
        server._validate_bench_cfg(self._cfg(asr_ladder=[1, 3600],
                                             ocr_ladder=[1, 64],
                                             tts_ladder=[1, 65536]))
        server._validate_bench_cfg(self._cfg(asr_ladder=[5] * 16,
                                             ocr_ladder=[1] * 16,
                                             tts_ladder=[50] * 16))
        server._validate_bench_cfg(self._cfg(asr_ladder=[], ocr_ladder=[],
                                             tts_ladder=[]))
        server._validate_bench_cfg(self._cfg())   # 缺省放行（引擎取场景默认）

    def test_repeats_timeout_max_tokens_thinking_validated(self):
        """repeats/timeout_s/max_tokens/thinking 白名单：bench_start 直接透传
        引擎，非法值一律 400——repeats 1~5（对齐引擎钳制）、timeout_s 正数
        ≤3600s、max_tokens 正整数 ≤32768、thinking 取引擎三值枚举。"""
        bad = [
            self._cfg(repeats=0), self._cfg(repeats=6),
            self._cfg(repeats=1.5), self._cfg(repeats=True),
            self._cfg(timeout_s=0), self._cfg(timeout_s=-1200),
            self._cfg(timeout_s=3601), self._cfg(timeout_s="600"),
            self._cfg(max_tokens=0), self._cfg(max_tokens=32769),
            self._cfg(max_tokens=1.5), self._cfg(max_tokens="512"),
            self._cfg(thinking="off"), self._cfg(thinking=1),
            self._cfg(thinking=["auto"]),
        ]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(repeats=5, timeout_s=3600,
                                             max_tokens=32768))
        for th in ("auto", "enabled", "disabled"):
            server._validate_bench_cfg(self._cfg(thinking=th))
        server._validate_bench_cfg(self._cfg())   # 缺省放行（引擎取默认值）


class _FakeBenchRun:
    """鸭式 BenchRun：只记录装配结果与启动前告警事件，不启动任何真实测速/网络。"""
    captured: list = []
    emitted: list = []

    def __init__(self, cfg, results_dir=None):
        self.cfg = cfg
        self.results_dir = results_dir
        self.run_id = f"fake-{len(_FakeBenchRun.captured)}"
        self.finished_at = None
        _FakeBenchRun.captured.append(self)

    async def emit(self, ev):
        _FakeBenchRun.emitted.append(ev)


async def _fake_drive(run, provider_name):
    """替代 server._drive：真实实现会 await run.run() 发起测速。"""


class TestBenchStartAssembly(unittest.TestCase):
    """POST /api/bench/start 的 provider 装配：ADR-0003 云端强制单发、
    deployments 命中注入 model_max_ctx/model_info、_CAP 能力记忆经 cfg 下发。
    BenchRun 换为假类只捕获 cfg，不起真实测速与端口。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = {k: getattr(server, k) for k in
                       ("BASE", "CONFIG_PATH", "RESULTS_DIR", "RUNS",
                        "BenchRun", "_drive", "GLOBAL_API_KEY")}
        self._cap = dict(server._CAP)
        server.BASE = self.tmp.name
        server.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        server.RESULTS_DIR = os.path.join(self.tmp.name, "results")
        server.RUNS = {}
        server.BenchRun = _FakeBenchRun
        server._drive = _fake_drive
        server.GLOBAL_API_KEY = ""
        server._CAP.clear()
        _FakeBenchRun.captured = []
        _FakeBenchRun.emitted = []
        self.client = TestClient(server.app)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(server, k, v)
        server._CAP.clear()
        server._CAP.update(self._cap)
        self.tmp.cleanup()

    def _start(self, **kw):
        body = {"provider": "p1", "models": ["m1"], "scenarios": ["creative"],
                "ctx_list": [4096], "concurrencies": [1, 2]}
        body.update(kw)
        return self.client.post("/api/bench/start", json=body)

    def test_cloud_provider_forces_single_concurrency(self):
        self.client.put("/api/config", json={"providers": [_prov(local=False)]})
        r = self._start(concurrencies=[1, 2, 4])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        # 云端不做并发/吞吐测试：强制单发，不信任前端传参（ADR-0009）
        self.assertEqual(cfg["concurrencies"], [1])
        self.assertFalse(cfg["provider_local"])
        self.assertEqual(cfg["provider_label"], "P1")
        self.assertEqual(cfg["gateway_url"], "http://127.0.0.1:4000")   # 服务端注入
        self.assertEqual(cfg["api_key"], "")
        self.assertNotIn("thinking_unsupported", cfg)   # 无能力记忆时不下发

    def test_deployments_inject_model_ctx_and_info(self):
        dep = {"label": "部署A", "models": ["m1"], "quants": ["Q8_0", "Q4_K_M"],
               "kv_quants": ["Q4_K_M"],
               "hardware": "RTX 4090", "framework": "llama.cpp",
               "params": "-c 8192", "max_ctx": 8192}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        r = self._start(models=["m1", "m2"])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_max_ctx"], {"m1": 8192})   # m2 未命中部署映射
        self.assertEqual(cfg["model_info"]["m1"]["hardware"], "RTX 4090")
        self.assertEqual(cfg["model_info"]["m1"]["deploy_label"], "部署A")
        self.assertEqual(cfg["model_info"]["m1"]["quant"], "Q8_0")
        self.assertEqual(cfg["model_info"]["m1"]["kv_quant"], "Q4_K_M")   # kv_quants 首项
        self.assertNotIn("quants", cfg["model_info"]["m1"])
        self.assertNotIn("kv_quants", cfg["model_info"]["m1"])
        self.assertNotIn("m2", cfg["model_info"])
        self.assertEqual(cfg["concurrencies"], [1, 2])   # 本地保留前端并发档位

    def test_match_deployments_legacy_quant_compat(self):
        """旧存盘配置仅有 quant 单字段：match_deployments 兼容读取为单值
        quant（不再发 quants 数组键）。"""
        info = server.match_deployments(
            {"deployments": [{"label": "旧", "models": ["m1"], "quant": "Q8_0"}]},
            ["m1"])
        self.assertEqual(info["m1"]["quant"], "Q8_0")
        self.assertNotIn("quants", info["m1"])

    def test_deployment_miss_emits_status_warning(self):
        """未命中显式告警：provider 配了 deployments 但被测模型没命中任何映射
        时，启动前发 status 告警事件（复用现有协议）；命中路径与未配 deployments
        的 provider 不告警。"""
        dep = {"label": "部署A", "models": ["m1"], "max_ctx": 8192}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        r = self._start(models=["m2"])   # m2 未命中映射
        self.assertEqual(r.status_code, 200, r.text)
        warns = [ev for ev in _FakeBenchRun.emitted
                 if ev.get("type") == "status"]
        self.assertEqual(len(warns), 1)
        self.assertIn("m2", warns[0]["msg"])
        self.assertIn("未映射部署环境", warns[0]["msg"])
        self.assertIn("max_ctx 钳制不可用", warns[0]["msg"])
        # 命中路径：不再告警
        _FakeBenchRun.emitted.clear()
        r = self._start(models=["m1"])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse([ev for ev in _FakeBenchRun.emitted
                          if ev.get("type") == "status"])
        # provider 未配 deployments：无从谈起映射，不告警
        self.client.put("/api/config", json={"providers": [_prov(local=True)]})
        _FakeBenchRun.emitted.clear()
        r = self._start(models=["m2"])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse([ev for ev in _FakeBenchRun.emitted
                          if ev.get("type") == "status"])

    def test_agent_matrix_over_budget_warns_not_blocks(self):
        """agent 矩阵最大组合校验：max(缓存档)+max(指令档) 超部署预算（max_ctx
        − max_tokens − 余量）时发 status 告警但不阻断（引擎对超预算组合整档
        跳过）；预算充足或无 max_ctx 来源时不告警。"""
        dep = {"label": "部署A", "models": ["m1"], "max_ctx": 4096}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        # 默认矩阵（缓存最大 256K + 指令最大 8K）远超 4K 部署预算
        r = self._start(models=["m1"], scenarios=["agent"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)   # 不阻断
        warns = [ev["msg"] for ev in _FakeBenchRun.emitted
                 if ev.get("type") == "status" and "agent" in ev.get("msg", "")]
        self.assertEqual(len(warns), 1)
        self.assertIn("m1", warns[0])
        self.assertIn("agent 矩阵最大组合", warns[0])
        self.assertIn("超出上限的组合将被跳过", warns[0])
        # 预算充足（max_ctx 抬高）：不告警
        dep = {"label": "部署A", "models": ["m1"], "max_ctx": 4194304}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        _FakeBenchRun.emitted.clear()
        r = self._start(models=["m1"], scenarios=["agent"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse([ev for ev in _FakeBenchRun.emitted
                          if ev.get("type") == "status"])
        # 未配 max_ctx 的部署：无预算可言，不告警（引擎本来就没有钳制来源）
        dep = {"label": "部署A", "models": ["m1"]}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        _FakeBenchRun.emitted.clear()
        r = self._start(models=["m1"], scenarios=["agent"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse([ev for ev in _FakeBenchRun.emitted
                          if ev.get("type") == "status"])

    def test_active_deployment_wins_over_inactive(self):
        """同一模型映射多套部署：激活的那套生效（model_info 与 max_ctx
        同口径）；无 active 标记回退首个命中。"""
        deps = [
            {"label": "备用", "models": ["m1"], "hardware": "旧机", "max_ctx": 4096},
            {"label": "主力", "models": ["m1"], "hardware": "新机",
             "max_ctx": 16384, "active": True},
        ]
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=deps)]})
        r = self._start(models=["m1"])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_info"]["m1"]["deploy_label"], "主力")
        self.assertEqual(cfg["model_max_ctx"], {"m1": 16384})
        # 回退口径：取消激活（环境B 不再 active）→ 首个命中（备用）生效
        deps[1] = {"label": "主力", "models": ["m1"], "hardware": "新机",
                   "max_ctx": 16384}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=deps)]})
        r = self._start(models=["m1"])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_info"]["m1"]["deploy_label"], "备用")
        self.assertEqual(cfg["model_max_ctx"], {"m1": 4096})

    def test_cap_memory_seeded_into_cfg(self):
        self.client.put("/api/config", json={"providers": [_prov()]})
        server._CAP["p1"] = {"thinking_unsupported": True, "temperature_locked": ["m1"]}
        r = self._start()
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertTrue(cfg["thinking_unsupported"])
        self.assertEqual(cfg["temperature_locked"], ["m1"])

    def test_deployments_inject_model_kinds(self):
        """ADR-0020/多选扩展：部署的 kinds 随 model_info/model_kinds 下发；未映射
        部署的模型按 [llm]。媒体测速场景按 kind 分发。旧单值 kind 字段兼容读取。"""
        dep = {"label": "whisper", "models": ["m1"], "hardware": "RTX 4090",
               "framework": "faster-whisper", "kind": "asr"}   # 旧单值字段：兼容入口
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        # 旧单值 kind 在写盘时已归一为 kinds 数组
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["kinds"], ["asr"])
        self.assertNotIn("kind", on_disk["providers"][0]["deployments"][0])
        # 映射到 asr 部署的模型：能力集随 model_info/model_kinds 下发
        r = self._start(models=["m1"], scenarios=["asr"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_info"]["m1"]["kinds"], ["asr"])
        self.assertEqual(cfg["model_kinds"], {"m1": ["asr"]})
        # 未映射部署的模型按 [llm]（与 LLM 场景匹配）
        r = self._start(models=["m2"], scenarios=["creative"])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_kinds"], {"m2": ["llm"]})

    def test_deployments_kinds_multi_select(self):
        """kinds 多选（文本模型支持视觉输入 = llm+ocr）：校验白名单、去重、
        缺省 [llm] 不落盘；空数组/非法元素/非数组一律 400。"""
        dep = {"label": "vlm", "models": ["m1"], "kinds": ["ocr", "llm", "ocr"]}
        r = self.client.put("/api/config",
                            json={"providers": [_prov(local=True, deployments=[dep])]})
        self.assertEqual(r.status_code, 200, r.text)
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["kinds"],
                         ["ocr", "llm"])   # 去重、保序
        # 多能力模型：两种 kind 的场景各自可启动（一次运行仍限一种）
        r = self._start(models=["m1"], scenarios=["ocr"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(_FakeBenchRun.captured[-1].cfg["model_kinds"],
                         {"m1": ["ocr", "llm"]})
        r = self._start(models=["m1"], scenarios=["creative"])
        self.assertEqual(r.status_code, 200, r.text)
        # 显式 [llm] 与缺省同口径：不落盘
        dep = {"label": "plain", "models": ["m2"], "kinds": ["llm"]}
        r = self.client.put("/api/config",
                            json={"providers": [_prov(local=True, deployments=[dep])]})
        self.assertEqual(r.status_code, 200, r.text)
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        self.assertNotIn("kinds", on_disk["providers"][0]["deployments"][0])
        # 非法输入
        for bad in ({"kinds": []}, {"kinds": ["vlm"]}, {"kinds": "ocr"}):
            dep = {"label": "x", "models": ["m3"], **bad}
            r = self.client.put(
                "/api/config",
                json={"providers": [_prov(local=True, deployments=[dep])]})
            self.assertEqual(r.status_code, 400, bad)

    def test_mixed_scenario_kinds_400(self):
        """一次运行一种 kind 的边界不因多能力模型打破：混选文本+图像识别场景
        直接 400（此前单 kind 模型天然不可能混选，多选后需显式拦截）。"""
        dep = {"label": "vlm", "models": ["m1"], "kinds": ["llm", "ocr"]}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        r = self._start(models=["m1"], scenarios=["creative", "ocr"],
                        ctx_list=[4096])
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("一种类型", r.text)

    def test_kind_mismatch_400(self):
        """模型 kind 与场景 kind 不匹配：400 明确提示（前端模型选择器按
        kind 过滤场景，这里是直调 API 的第二道闸）。"""
        self.client.put("/api/config", json={"providers": [_prov(local=True)]})
        r = self._start(scenarios=["asr"], ctx_list=[])
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("不匹配", r.text)

    def test_media_scenarios_need_no_ctx_list(self):
        """媒体场景（asr/ocr/tts）按场景自带阶梯出点：ctx_list 可缺省；
        混合 LLM+媒体场景时 LLM 仍要求 ctx_list。"""
        dep = {"label": "whisper", "models": ["m1"], "kind": "asr"}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        r = self._start(scenarios=["asr"], models=["m1"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        r = self._start(scenarios=["asr", "creative"], models=["m1"], ctx_list=[])
        self.assertEqual(r.status_code, 400, r.text)   # creative 需要 ctx_list

    def test_translate_needs_no_ctx_list(self):
        """翻译场景（translate，llm kind）按固定原文字长阶梯出点：ctx_list 可
        缺省；混合创意场景时仍要求 ctx_list（与媒体场景同口径）。"""
        self.client.put("/api/config", json={"providers": [_prov(local=True)]})
        r = self._start(scenarios=["translate"], models=["m1"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        r = self._start(scenarios=["translate", "creative"], models=["m1"],
                        ctx_list=[])
        self.assertEqual(r.status_code, 400, r.text)   # creative 需要 ctx_list

    def test_config_scenarios_carry_kind_and_ladder(self):
        """/api/config 场景列表带 kind 与阶梯：前端按 kind 过滤/分组与展示档位。"""
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


class TestAtomicWrite(unittest.TestCase):
    """_atomic_write：原子替换不留残渣、失败保旧文件、凭据文件权限收紧。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, name="a.json"):
        return os.path.join(self.tmp.name, name)

    def test_no_tmp_residue(self):
        p = self._path()
        server._atomic_write(p, '{"a": 1}')
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"a": 1}')
        self.assertEqual(os.listdir(self.tmp.name), ["a.json"])   # 无 .tmp 残留

    def test_replace_failure_keeps_old_and_cleans_tmp(self):
        p = self._path()
        server._atomic_write(p, "old")
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                server._atomic_write(p, "new")
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "old")                     # 旧文件完好
        self.assertEqual(os.listdir(self.tmp.name), ["a.json"])   # tmp 已清理

    def test_private_new_file_0600(self):
        p = self._path(".env")
        server._atomic_write(p, "K=V\n", private=True)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)   # 新建收紧
        os.chmod(p, 0o644)
        server._atomic_write(p, "K=W\n", private=True)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o644)   # 已存在沿用原权限


class TestHistoryApi(unittest.TestCase):
    """历史记录端点：路径穿越防护 detail/delete 对称、单删/清空只动 RESULTS_DIR 内 json。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        server.RESULTS_DIR = self.tmp.name
        self.client = TestClient(server.app)
        for rid in ("r-one", "r-two"):
            with open(os.path.join(self.tmp.name, f"{rid}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"run_id": rid, "results": []}, f)
        with open(os.path.join(self.tmp.name, "keep.txt"), "w") as f:
            f.write("x")   # 非 json 文件不受清空影响

    def tearDown(self):
        server.RESULTS_DIR = self._results
        self.tmp.cleanup()

    def test_traversal_rejected_symmetrically(self):
        for method in (self.client.get, self.client.delete):
            r = method("/api/bench/history/a..b")
            self.assertEqual(r.status_code, 400, r.text)   # 含 .. 对称拒绝
            r = method("/api/bench/history/ghost")
            self.assertEqual(r.status_code, 404, r.text)   # 合法但不存在

    def test_detail_returns_archive(self):
        r = self.client.get("/api/bench/history/r-one")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["run_id"], "r-one")

    def test_delete_one_keeps_others(self):
        r = self.client.delete("/api/bench/history/r-one")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "r-one.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "r-two.json")))

    def test_clear_removes_all_json(self):
        r = self.client.delete("/api/bench/history")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(os.listdir(self.tmp.name), ["keep.txt"])   # json 全清、其他保留

    def _write_rename_archive(self, rid="r-ren"):
        """含全部模型名引用位的存档：cfg.models/model_info/model_kind(s)/
        model_max_ctx/temperature_locked + 逐点 model 字段。"""
        d = {"run_id": rid, "started_at": "2026-09-11 09:00:00",
             "cfg": {"provider": "p1", "models": ["old-m", "other"],
                     "model_info": {"old-m": {"hardware": "RTX"}},
                     "model_kind": {"old-m": "llm", "other": "llm"},
                     "model_kinds": {"old-m": ["llm"], "other": ["llm"]},
                     "model_max_ctx": {"old-m": 8192},
                     "temperature_locked": ["old-m"]},
             "results": [{"model": "old-m", "scenario": "creative"},
                         {"model": "other", "scenario": "creative"},
                         {"model": "old-m", "scenario": "code"}]}
        with open(os.path.join(self.tmp.name, f"{rid}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(d, f)
        return rid

    def test_rename_rewrites_all_references(self):
        """改名适配（本地部署改名后存档对不上号）：所有引用位改写为新名，
        顶层追加 model_renames 审计留痕；重复改名留痕追加而非覆盖。"""
        rid = self._write_rename_archive()
        r = self.client.post(f"/api/bench/history/{rid}/rename",
                             json={"old": "old-m", "new": "new-m"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["points"], 2)
        with open(os.path.join(self.tmp.name, f"{rid}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        cfg = d["cfg"]
        self.assertEqual(cfg["models"], ["new-m", "other"])
        self.assertEqual(list(cfg["model_info"]), ["new-m"])
        self.assertEqual(cfg["model_kind"]["new-m"], "llm")
        self.assertEqual(cfg["model_kinds"]["new-m"], ["llm"])
        self.assertEqual(cfg["model_max_ctx"], {"new-m": 8192})
        self.assertEqual(cfg["temperature_locked"], ["new-m"])
        self.assertEqual([p["model"] for p in d["results"]],
                         ["new-m", "other", "new-m"])
        self.assertEqual(d["model_renames"][0]["from"], "old-m")
        self.assertEqual(d["model_renames"][0]["to"], "new-m")
        self.assertTrue(d["model_renames"][0]["at"])
        r = self.client.post(f"/api/bench/history/{rid}/rename",
                             json={"old": "new-m", "new": "newer-m"})
        self.assertEqual(r.status_code, 200, r.text)
        with open(os.path.join(self.tmp.name, f"{rid}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual([x["to"] for x in d["model_renames"]],
                         ["new-m", "newer-m"])

    def test_rename_rejected(self):
        """改名入参防护：穿越/不存在 与 detail 对称；old 不在存档、new 撞名、
        新旧同名、空值一律 400 且存档不动。"""
        rid = self._write_rename_archive()
        r = self.client.post("/api/bench/history/a..b/rename",
                             json={"old": "a", "new": "b"})
        self.assertEqual(r.status_code, 400, r.text)
        r = self.client.post("/api/bench/history/ghost/rename",
                             json={"old": "a", "new": "b"})
        self.assertEqual(r.status_code, 404, r.text)
        for body in ({"old": "ghost-m", "new": "x"},
                     {"old": "old-m", "new": "other"},
                     {"old": "old-m", "new": "old-m"},
                     {"old": "", "new": "x"}, {"old": "old-m"}):
            r = self.client.post(f"/api/bench/history/{rid}/rename", json=body)
            self.assertEqual(r.status_code, 400, body)
        with open(os.path.join(self.tmp.name, f"{rid}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["cfg"]["models"], ["old-m", "other"])
        self.assertNotIn("model_renames", d)


if __name__ == "__main__":
    unittest.main()
