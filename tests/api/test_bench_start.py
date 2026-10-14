"""api 组：POST /api/bench/start 的 provider 装配与启动前校验
（ADR-0003 云端强制单发 / ADR-0009 / ADR-0020）。

自 tests/test_config_api.py 拆分：TestBenchStartAssembly 与它的桩件
（_FakeBenchRun / _fake_drive / _prov）原样搬入，其中 kinds 两例（旧单值兼容 +
多选归一）与 media/translate 两例（ctx_list 可缺省）各并为一条 subTest 用例。
_prov 是 config / bench_start 两个文件的共用桩件，只此一份，test_config.py 从
本模块 import。setUp 补 server._HIST_CACHE.clear()，并把 API_KEY_P1 保存后
清空（环境残留会经 server.provider_key 混进 cfg["api_key"]，污染装配断言）。

    python -m pytest tests/api/test_bench_start.py -q
    python -m unittest discover -s tests/api -p test_bench_start.py
    python -m unittest tests.api.test_bench_start -v      # 单模块入口
（两种入口都从仓库根目录运行。）
"""
import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import server
from tests._shared import restore_env, snapshot_env


def _prov(**kw):
    p = {"name": "p1", "label": "P1", "gateway_url": "http://127.0.0.1:4000",
         "local": False, "deployments": []}
    p.update(kw)
    return p


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
        server._HIST_CACHE.clear()   # 历史列表摘要缓存跨用例不清会串档
        # 环境残留（shell 导出 / 真实 .env 被 server 载入）会经 provider_key 混进
        # cfg["api_key"]（源文件里靠 TestConfigApi.tearDown 的裸 pop 偶然掩盖）：
        # 原值保存后清空，用例结束还原，断言口径只认用例自己配的 key。
        self.addCleanup(restore_env, snapshot_env("API_KEY_P1"))
        os.environ.pop("API_KEY_P1", None)
        _FakeBenchRun.captured = []
        _FakeBenchRun.emitted = []
        self.client = TestClient(server.app)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(server, k, v)
        server._CAP.clear()
        server._CAP.update(self._cap)
        server._HIST_CACHE.clear()
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

    def test_deployments_kinds_single_and_multi_select(self):
        """ADR-0020/多选扩展：部署的 kinds 随 model_info/model_kinds 下发；未映射
        部署的模型按 [llm]。媒体测速场景按 kind 分发。旧单值 kind 字段兼容读取。

        kinds 多选（文本模型支持视觉输入 = llm+ocr）：校验白名单、去重、
        缺省 [llm] 不落盘；空数组/非法元素/非数组一律 400。"""
        with self.subTest(case="旧单值 kind 兼容与 kind 分发"):
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
        with self.subTest(case="kinds 多选：写盘归一与非法输入拒绝"):
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

    def test_media_and_translate_need_no_ctx_list(self):
        """媒体场景（asr/ocr/tts）按场景自带阶梯出点：ctx_list 可缺省；
        混合 LLM+媒体场景时 LLM 仍要求 ctx_list。

        翻译场景（translate，llm kind）按固定原文字长阶梯出点：ctx_list 可
        缺省；混合创意场景时仍要求 ctx_list（与媒体场景同口径）。"""
        with self.subTest(case="媒体场景（asr）"):
            dep = {"label": "whisper", "models": ["m1"], "kind": "asr"}
            self.client.put("/api/config",
                            json={"providers": [_prov(local=True, deployments=[dep])]})
            r = self._start(scenarios=["asr"], models=["m1"], ctx_list=[])
            self.assertEqual(r.status_code, 200, r.text)
            r = self._start(scenarios=["asr", "creative"], models=["m1"], ctx_list=[])
            self.assertEqual(r.status_code, 400, r.text)   # creative 需要 ctx_list
        with self.subTest(case="翻译场景（translate）"):
            self.client.put("/api/config", json={"providers": [_prov(local=True)]})
            r = self._start(scenarios=["translate"], models=["m1"], ctx_list=[])
            self.assertEqual(r.status_code, 200, r.text)
            r = self._start(scenarios=["translate", "creative"], models=["m1"],
                            ctx_list=[])
            self.assertEqual(r.status_code, 400, r.text)   # creative 需要 ctx_list


if __name__ == "__main__":
    unittest.main()
