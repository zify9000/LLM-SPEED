"""e2e 组·媒体场景（ASR / TTS / OCR / 翻译）全链路回归。

覆盖索引（M 簇，自 tests/test_e2e_mock.py 拆分）：
- test_asr_rtf_against_mock：ASR 全链路（时长阶梯×并发出点、净 RTF ≈
  1/MOCK_ASR_PP、倍速/吞吐口径闭合；ctx_list 不适用，场景自带阶梯）；
- test_media_ladder_override_effective：cfg 的 <场景>_ladder 覆盖场景默认阶梯
  （asr / ocr 两形态）；
- test_media_ctx_overflow_skipped：超窗保护（ASR / 翻译两例参数化）——超
  max_ctx 的档位整档 point_skipped（不发出必然失败的请求）；
- test_tts_against_mock：TTS 全链路（文本阶梯出点、音频时长按返回 wav 解析
  而非语速估算、TTFA 为首 chunk 到达）；
- test_ocr_images_with_mock：OCR 全链路（VLM 多模态 parts 读图、流式
  TTFT/decode 读数 + ms/img 与并发吞吐口径闭合）；
- test_media_kind_mismatch_skipped：模型 kind 与场景 kind 不匹配时引擎第二道
  闸跳过并播报（不出点、不报错死亡）；
- test_media_model_kinds_multi_capability：多能力模型（model_kinds 多选）正常
  出点，不被能力闸误拦；
- test_translate_against_mock：翻译场景全链路（原文字长阶梯、走 chat 文本流
  路径 kind=llm、prompt 随档位放大）。

运行（仓库根目录，两种入口）：
    python -m pytest tests/e2e/test_e2e_media.py -q
    python -m unittest discover -s tests/e2e -p test_e2e_media.py
"""
import unittest

from tests.e2e._harness import MockGatewayTest   # noqa: E402  —— 触发 MOCK_* 播种（须先于 mock_server）

import bench                                     # noqa: E402
import mock_server                               # noqa: E402
from bench import SCENARIOS                      # noqa: E402


class TestMediaScenarios(MockGatewayTest):
    """媒体场景（ASR/TTS/OCR/翻译）全链路用例组。"""

    async def test_asr_rtf_against_mock(self):
        """ASR 全链路（MOCK_ASR_PP=50，即 50 倍实时）：时长阶梯×并发出点，
        净 RTF ≈ 1/50，倍速/吞吐口径闭合；ctx_list 不适用（场景自带阶梯）。"""
        old_ladder = SCENARIOS["asr"]["ladder"]
        old_pp = mock_server.ASR_PP
        SCENARIOS["asr"]["ladder"], mock_server.ASR_PP = [5, 30], 50.0
        try:
            run, events = await self._run_and_collect(
                scenarios=["asr"], ctx_list=[], concurrencies=[1, 2],
                model_kind={"mock-llm-7b": "asr"})
        finally:
            SCENARIOS["asr"]["ladder"], mock_server.ASR_PP = old_ladder, old_pp
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 档时长 × 2 并发 = 4 个点")
        self.assertTrue(all(p["all_ok"] for p in pts), f"全点应成功: {pts}")
        for p in pts:
            self.assertEqual(p["kind"], "asr")
            self.assertTrue(0.008 <= p["rtf"] <= 0.04,
                            f"RTF {p['rtf']} 应 ≈ 1/50（净口径）")
            self.assertAlmostEqual(p["speed_x"], 50, delta=20)
            self.assertAlmostEqual(p["audio_s"], p["ctx_target"], delta=0.1)
            self.assertIsNotNone(p.get("audio_min_per_min"), "全成功并发点须有吞吐")
        # 并发 2 的吞吐应显著高于并发 1（mock 无串行化，近似线性）
        t1 = next(p["audio_min_per_min"] for p in pts
                  if p["ctx_target"] == 30 and p["concurrency"] == 1)
        t2 = next(p["audio_min_per_min"] for p in pts
                  if p["ctx_target"] == 30 and p["concurrency"] == 2)
        self.assertGreater(t2, 1.5 * t1, f"并发吞吐应近线性: {t1} → {t2}")
        self.assertGreater(next(p["out_chars"] for p in pts
                                if p["ctx_target"] == 30), 30)

    async def test_media_ladder_override_effective(self):
        """媒体场景自定义阶梯生效：cfg 的 <场景>_ladder 覆盖场景默认阶梯——
        出点档位集合 = 覆盖值而非默认值（asr 默认 [5,15,60,300,1800]、
        ocr 默认 [1,2,4,8,16]）。"""
        _, events = await self._run_and_collect(
            scenarios=["asr"], ctx_list=[], concurrencies=[1],
            asr_ladder=[5, 15], model_kind={"mock-llm-7b": "asr"})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2, "覆盖阶梯 2 档 × 1 并发 = 2 个点")
        self.assertEqual(sorted({p["ctx_target"] for p in pts}), [5, 15],
                         "出点档位应为覆盖阶梯 [5,15]，不得混入默认档位")
        self.assertTrue(all(p["all_ok"] and p["kind"] == "asr" for p in pts))
        # ocr 同口径：覆盖阶梯生效
        _, events = await self._run_and_collect(
            scenarios=["ocr"], ctx_list=[], concurrencies=[1],
            max_tokens=64, ocr_ladder=[2],
            model_kind={"mock-llm-7b": "ocr"})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "覆盖阶梯 [2] 只出 1 个点")
        self.assertEqual(pts[0]["ctx_target"], 2)
        self.assertEqual(pts[0]["n_img"], 2)
        self.assertTrue(pts[0]["all_ok"])

    async def test_media_ctx_overflow_skipped(self):
        """媒体场景超窗保护（ASR / 翻译两例参数化，原两例的断言逐字保留）：
        档位规模（时长×50 tokens/秒；原文 tokens 估值 + 当档输出预算）超
        max_ctx 的档位整档 point_skipped，不发出必然失败的请求。"""
        cases = [
            {   # ASR：时长×50 tokens/秒，120s 档 = 6000 tokens > 4096
                "name": "asr",
                "ladder": ("asr", [30, 120]),
                "cfg": {"scenarios": ["asr"], "ctx_list": [], "concurrencies": [1, 2],
                        "model_kind": {"mock-llm-7b": "asr"},
                        "model_max_ctx": {"mock-llm-7b": 4096}},
                "ok_msg": "30s 档 × 2 并发应正常出点",
                "skip_msg": "120s 档应整档跳过",
                "skip_ctx": 120,
            },
            {   # 翻译：原文 tokens 估值 + 当档输出预算超 max_ctx
                "name": "translate",
                "ladder": None,
                "cfg": {"scenarios": ["translate"], "ctx_list": [],
                        "concurrencies": [1, 2], "translate_ladder": [50, 3200],
                        "model_max_ctx": {"mock-llm-7b": 2048}},
                "ok_msg": "50 字档 × 2 并发应正常出点",
                "skip_msg": "3200 字档应整档跳过",
                "skip_ctx": 3200,
            },
        ]
        for case in cases:
            with self.subTest(scenario=case["name"]):
                old_ladder = None
                if case["ladder"]:
                    scene, ladder = case["ladder"]
                    old_ladder = SCENARIOS[scene]["ladder"]
                    SCENARIOS[scene]["ladder"] = ladder
                try:
                    _, events = await self._run_and_collect(**case["cfg"])
                finally:
                    if old_ladder is not None:
                        SCENARIOS[case["ladder"][0]]["ladder"] = old_ladder
                pts = [e["point"] for e in events if e.get("type") == "point"]
                skipped = [e for e in events if e.get("type") == "point_skipped"]
                self.assertEqual(len(pts), 2, case["ok_msg"])
                self.assertEqual(len(skipped), 1, case["skip_msg"])
                self.assertEqual(skipped[0]["ctx"], case["skip_ctx"])
                self.assertEqual(skipped[0]["count"], 2,
                                 "跳过计数须含全部并发，前端进度才对")
                self.assertIn("超出部署上限", skipped[0]["msg"])

    async def test_tts_against_mock(self):
        """TTS 全链路（MOCK_TTS_RATE=500 字/秒）：文本阶梯出点，音频时长按
        返回 wav 解析（非语速估算），TTFA 为首 chunk 到达、≈0。"""
        old_ladder = SCENARIOS["tts"]["ladder"]
        old_rate, old_xr = mock_server.TTS_RATE, mock_server.TTS_XR
        SCENARIOS["tts"]["ladder"] = [50, 200]
        mock_server.TTS_RATE, mock_server.TTS_XR = 500.0, 1000.0
        try:
            _, events = await self._run_and_collect(
                scenarios=["tts"], ctx_list=[], concurrencies=[1],
                model_kind={"mock-llm-7b": "tts"})
        finally:
            SCENARIOS["tts"]["ladder"] = old_ladder
            mock_server.TTS_RATE, mock_server.TTS_XR = old_rate, old_xr
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2)
        for p in pts:
            self.assertEqual(p["kind"], "tts")
            # 口径版本（ADR-0083）：媒体点走 _run_media_point 的独立构造点，
            # 必须同样带版本戳（三处构造点各有一条断言，防漏挂）
            self.assertEqual(p.get("metric_version"), bench.METRIC_VERSION)
            self.assertNotIn("audio_est", p, "mock 回合法 wav，时长必须为解析值")
            expect = p["ctx_target"] / 500.0
            self.assertAlmostEqual(p["audio_s"], expect, delta=max(0.03, expect * 0.1))
            self.assertLess(p["ttfa_s"], 0.5, f"首 chunk 即刻到达: {p['ttfa_s']}")
            self.assertTrue(0.5 <= p["speed_x"] <= 5000, f"倍速口径异常: {p['speed_x']}")
            self.assertGreater(p["out_bytes"], 1000)

    async def test_ocr_images_with_mock(self):
        """OCR 全链路：VLM 读图请求（多模态 content parts）mock 正常受理，
        流式 TTFT/decode 读数 + ms/img 与并发吞吐口径闭合。"""
        old_ladder = SCENARIOS["ocr"]["ladder"]
        SCENARIOS["ocr"]["ladder"] = [1, 4]
        try:
            _, events = await self._run_and_collect(
                scenarios=["ocr"], ctx_list=[], concurrencies=[1, 2],
                max_tokens=64,
                model_kind={"mock-llm-7b": "ocr"})
        finally:
            SCENARIOS["ocr"]["ladder"] = old_ladder
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 档张数 × 2 并发")
        self.assertTrue(all(p["all_ok"] for p in pts), f"读图请求不得判失败: {pts}")
        for p in pts:
            self.assertEqual(p["kind"], "ocr")
            self.assertEqual(p["n_img"], p["ctx_target"])
            self.assertGreater(p["ms_per_img"], 0)
            self.assertIsNotNone(p.get("img_per_s"), "全成功并发点须有吞吐")
            self.assertIsNotNone(p.get("ttft_s"), "VLM 流式路径应有 TTFT")
            self.assertIsNotNone(p.get("decode_tok_s"))
            req = p["reqs"][0]
            self.assertIsNone(req.get("err"))
            self.assertGreater(req["out_tokens"], 0)

    async def test_media_kind_mismatch_skipped(self):
        """模型 kind 与场景 kind 不匹配（直调引擎绕过服务端校验时）：引擎第二道
        闸跳过该场景组合并播报，不出点、不报错死亡。"""
        _, events = await self._run_and_collect(
            scenarios=["asr"], ctx_list=[], concurrencies=[1],
            model_kind={"mock-llm-7b": "llm"})   # llm 模型 × asr 场景
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(pts, [])
        types = [e["type"] for e in events]
        self.assertIn("done", types)
        skips = [e["msg"] for e in events if e.get("type") == "status"
                 and "不匹配" in e.get("msg", "")]
        self.assertTrue(skips, "应有 kind 不匹配跳过播报")

    async def test_media_model_kinds_multi_capability(self):
        """多能力模型（model_kinds=["llm","ocr"]，ADR-0020 多选扩展）：ocr 场景
        正常出点不被能力闸误拦（kind 不匹配的跳过播报由
        test_media_kind_mismatch_skipped 锁定，不在此重复验证）。"""
        old_ladder = SCENARIOS["ocr"]["ladder"]
        SCENARIOS["ocr"]["ladder"] = [1]
        try:
            _, events = await self._run_and_collect(
                scenarios=["ocr"], ctx_list=[], concurrencies=[1],
                max_tokens=64,
                model_kinds={"mock-llm-7b": ["llm", "ocr"]})
        finally:
            SCENARIOS["ocr"]["ladder"] = old_ladder
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "多能力模型应正常跑 ocr 场景")
        self.assertTrue(pts[0]["all_ok"])

    async def test_translate_against_mock(self):
        """翻译场景全链路（ADR-0034）：原文字长阶梯出点（ctx_list 不适用，
        阶梯由 cfg.translate_ladder 自定义下发），走 chat 文本流路径
        （kind=llm），prompt 随档位放大、输出预算随档伸缩。"""
        _, events = await self._run_and_collect(
            scenarios=["translate"], ctx_list=[], concurrencies=[1, 2],
            translate_ladder=[50, 200])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 档原文字长 × 2 并发 = 4 个点")
        self.assertTrue(all(p["all_ok"] for p in pts), f"全点应成功: {pts}")
        for p in pts:
            self.assertEqual(p["kind"], "llm")   # 与创意/代码同一请求路径
            self.assertIn(p["ctx_target"], [50, 200])   # 档位 = 原文字数（非 tokens）
            self.assertIsNotNone(p.get("ttft_s"))
            self.assertIsNotNone(p.get("decode_tok_s"))
            self.assertGreater(p["out_tokens"], 0)
        p50 = next(p for p in pts
                   if p["ctx_target"] == 50 and p["concurrency"] == 1)
        p200 = next(p for p in pts
                    if p["ctx_target"] == 200 and p["concurrency"] == 1)
        self.assertGreater(p200["prompt_tokens"], p50["prompt_tokens"],
                           "原文 4 倍，prompt 应显著放大")


if __name__ == "__main__":
    unittest.main()
