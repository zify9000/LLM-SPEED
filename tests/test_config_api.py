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
        dep = {"label": "部署A", "models": ["m1", "m2"], "quant": "Q8_0",
               "hardware": "RTX", "framework": "llama.cpp", "params": "-c 4096",
               "max_ctx": 4096}
        r = self.client.put("/api/config", json={"providers": [_prov(local=True, deployments=[dep])]})
        self.assertEqual(r.status_code, 200)
        on_disk = json.load(open(server.CONFIG_PATH, encoding="utf-8"))
        self.assertEqual(on_disk["providers"][0]["deployments"][0]["max_ctx"], 4096)
        got = self.client.get("/api/config").json()
        self.assertEqual(got["providers"][0]["name"], "p1")
        self.assertEqual(got["providers"][0]["local"], True)
        self.assertEqual(got["providers"][0]["deployments"][0]["quant"], "Q8_0")
        self.assertNotIn("api_key", json.dumps(on_disk))

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

    def test_agent_chain_params_validated(self):
        """Agent 连续任务链参数钳制：agent_turns 1~32；agent_turn_delta 为
        阶段一短轮的钳制区间——256~32768 的整数或 [min, max]（须 min ≤ max）。
        越界或类型错 400；缺省放行由引擎取默认。纯 Agent 场景 ctx_list 可空。"""
        bad = [self._cfg(agent_turns=0), self._cfg(agent_turns=33),
               self._cfg(agent_turns=1.5), self._cfg(agent_turns=True),
               self._cfg(agent_turn_delta=100), self._cfg(agent_turn_delta=40000),
               self._cfg(agent_turn_delta=[100, 2048]),
               self._cfg(agent_turn_delta=[2048, 256]),     # min > max
               self._cfg(agent_turn_delta=[256, 2048, 4096]),
               self._cfg(agent_turn_delta="2048")]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(agent_turns=8, agent_turn_delta=2048))
        server._validate_bench_cfg(self._cfg(agent_turn_delta=[256, 2048]))
        # 纯 Agent 场景不以上下文档位为变量：ctx_list 可为空
        server._validate_bench_cfg(self._cfg(scenarios=["agent"], ctx_list=[]))

    def test_agent_phase_turns_validated(self):
        """两阶段轮数配置：agent_turns_p1/p2 各 0~32、合计 1~32；给了一个就
        必须两个都给；旧配置 agent_turns 单值仍放行（引擎对半切兼容）。"""
        bad = [self._cfg(agent_turns_p1=-1, agent_turns_p2=4),
               self._cfg(agent_turns_p1=33, agent_turns_p2=4),
               self._cfg(agent_turns_p1=1.5, agent_turns_p2=4),
               self._cfg(agent_turns_p1=0, agent_turns_p2=0),     # 合计为 0
               self._cfg(agent_turns_p1=16, agent_turns_p2=17),   # 合计 33
               self._cfg(agent_turns_p1=4),                       # 只给一个
               self._cfg(agent_turns_p2=4)]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(agent_turns_p1=4, agent_turns_p2=4))
        server._validate_bench_cfg(self._cfg(agent_turns_p1=0, agent_turns_p2=4))
        # 阶段二 ladder 起始增量：1024~65536 的整数
        for bad_base in (512, 131072, 1.5, "4096", True):
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(self._cfg(agent_phase2_base=bad_base))
            self.assertEqual(cm.exception.status_code, 400, bad_base)
        server._validate_bench_cfg(self._cfg(agent_phase2_base=8192))

    def test_agent_cold_ctx_validated(self):
        """冷启动轮上下文独立配置（ADR-0015）：agent_cold_ctx 为
        1024~262144 的整数；越界/类型错 400，缺省放行（引擎取默认 10K）。"""
        bad = [self._cfg(agent_cold_ctx=100), self._cfg(agent_cold_ctx=300000),
               self._cfg(agent_cold_ctx=1.5), self._cfg(agent_cold_ctx=True),
               self._cfg(agent_cold_ctx="10240")]
        for cfg in bad:
            with self.assertRaises(HTTPException) as cm:
                server._validate_bench_cfg(cfg)
            self.assertEqual(cm.exception.status_code, 400, cfg)
        server._validate_bench_cfg(self._cfg(agent_cold_ctx=10240))
        server._validate_bench_cfg(self._cfg())   # 缺省放行

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
        dep = {"label": "部署A", "models": ["m1"], "quant": "Q8_0",
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
        self.assertNotIn("m2", cfg["model_info"])
        self.assertEqual(cfg["concurrencies"], [1, 2])   # 本地保留前端并发档位

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

    def test_agent_chain_over_budget_warns_not_blocks(self):
        """agent 链总目标校验：最大轮目标超部署预算（max_ctx − max_tokens −
        余量）时发 status 告警但不阻断（引擎逐轮跳过）；预算充足不告警。"""
        dep = {"label": "部署A", "models": ["m1"], "max_ctx": 4096}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        # 默认链（1 冷 10K + 阶段一 6 轮短 + 阶段二 ladder 4K 起翻倍）远超预算
        r = self._start(models=["m1"], scenarios=["agent"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)   # 不阻断
        warns = [ev["msg"] for ev in _FakeBenchRun.emitted
                 if ev.get("type") == "status" and "agent" in ev.get("msg", "")]
        self.assertEqual(len(warns), 1)
        self.assertIn("m1", warns[0])
        self.assertIn("轮次将被跳过", warns[0])
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

    def test_deployments_inject_model_kind(self):
        """ADR-0020：部署的 kind 随 model_info/model_kind 下发；未映射部署的
        模型按 llm。媒体测速场景按 kind 分发。"""
        dep = {"label": "whisper", "models": ["m1"], "hardware": "RTX 4090",
               "framework": "faster-whisper", "kind": "asr"}
        self.client.put("/api/config",
                        json={"providers": [_prov(local=True, deployments=[dep])]})
        # 映射到 asr 部署的模型：kind 随 model_info/model_kind 下发
        r = self._start(models=["m1"], scenarios=["asr"], ctx_list=[])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_info"]["m1"]["kind"], "asr")
        self.assertEqual(cfg["model_kind"], {"m1": "asr"})
        # 未映射部署的模型按 llm（与 LLM 场景匹配）
        r = self._start(models=["m2"], scenarios=["creative"])
        self.assertEqual(r.status_code, 200, r.text)
        cfg = _FakeBenchRun.captured[-1].cfg
        self.assertEqual(cfg["model_kind"], {"m2": "llm"})

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


if __name__ == "__main__":
    unittest.main()
