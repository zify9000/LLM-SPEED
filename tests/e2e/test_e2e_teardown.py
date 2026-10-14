"""e2e 组·停止时效 + 收口与采样全链路回归。

覆盖索引（I 簇停止时效 + O 簇收口与采样，自 tests/test_e2e_mock.py 拆分）：
- test_stop_takes_effect_promptly：停止即时生效（探测期 / RTT 基线期两例
  参数化）——探测期停止不等整个探测跑完、慢 /models 在途 GET 被轮询取消；
- test_stop_keeps_interrupted_point_as_err：停止时被中断的那一档留下 err 点
  （stopped_mid_point + 半截读数不留档 + 终态 stopped）；
- test_burst_decode_window_speed：起步突发滑窗差分（起步 tick 不虚高、
  后续收敛 TG±25%）；
- test_no_content_point_error：零输出真空流 → 请求 err、点级失败；
- test_die_at_request_error：裸断连（原始 socket 假网关）→ 请求级 err；
- test_serialize_prefill_no_decode_overlap：串行 prefill 下 decode 不重叠、
  总吞吐置空；
- test_no_usage_estimation_fallback：无 usage 帧按 SSE 事件数直计兜底
  （单字符事件 / 英文多字符事件两例参数化，不虚发、不误判断流）；
- test_archive_output_samples：存档输出采样（text_sample/reason_sample 头部
  ≤200 字符截获：超长截断、不足全留）；
- test_max_tokens_ignored_flip_and_cut：忽略 max_tokens 的网关 → 改用
  max_completion_tokens 并精确截断；
- test_mct_rejected_falls_back_no_oscillation：新名被 400 拒收 → 回退旧名
  锁定、只断流兜底不再翻转；
- test_out_hint_injected_into_system_prompt：system 注入 + 指令末尾复述输出
  长度引导（结构判据：引导片段恰好两份、字符增量大于单份）；
- test_repeats_one_transparency_notice：repeats=1 的「无统计效力」透明度提醒；
- test_tg_pattern_alternates_decode_speed：MOCK_TG_PATTERN 逐请求交替 decode
  速度，两档都被采到。

运行（仓库根目录，两种入口）：
    python -m pytest tests/e2e/test_e2e_teardown.py -q
    python -m unittest discover -s tests/e2e -p test_e2e_teardown.py
"""
import asyncio
import time
import unittest

# 顺序是硬约束：先 import 本 harness（它先 import tests._shared 完成 MOCK_* 播种，
# 再 import mock_server），否则 mock_server 采到默认值、断言口径全错
from tests.e2e._harness import (                 # noqa: E402
    MockGatewayTest, _free_port, _start_raw_die_server,
)

import bench                                     # noqa: E402
import mock_server                               # noqa: E402
from bench import BenchRun, SCENARIOS            # noqa: E402


class TestStopTimeliness(MockGatewayTest):
    """停止时效：停止在探测 / 基线阶段即时生效，且中断点留档收口。"""

    async def test_stop_takes_effect_promptly(self):
        """停止即时生效（两阶段参数化，原 test_stop_during_probe 与
        test_stop_during_models_delay_baseline 两例合并）：

        - probe：探测期停止——probe 不走 _run_point 任务集，曾经无法中断
          （点停止要等整个探测跑完，慢网关上可达分钟级）；
        - models_delay：MOCK_MODELS_DELAY=6（/models 慢响应）下 RTT 基线期
          停止——_interruptible 0.2s 轮询取消在途 GET，不等整个慢响应。"""
        cases = [
            {
                "name": "probe",
                "patch": {"PP": 200.0},   # 探测 2048 ctx → prefill ~10s，足够停在探测中段
                "ctx_list": [4096],
                "wait_s": 1.0,            # RTT 基线（毫秒级）后已进入探测 prefill
                "limit_s": 5,
                "msg": "探测期停止未即时生效（曾需等整个探测完成）",
            },
            {
                "name": "models_delay",
                "patch": {"MODELS_DELAY": 6.0},
                "ctx_list": [0],
                "wait_s": 0.8,            # 首个 GET /models 在途（响应需 6s）
                "limit_s": 4,
                "msg": "基线期停止未即时生效（等在途 /models 慢响应）",
            },
        ]
        for case in cases:
            with self.subTest(phase=case["name"]):
                olds = {k: getattr(mock_server, k) for k in case["patch"]}
                for k, v in case["patch"].items():
                    setattr(mock_server, k, v)
                try:
                    run = BenchRun(self.std_cfg(ctx_list=case["ctx_list"],
                                                max_tokens=8))
                    task = asyncio.create_task(run.run())
                    await asyncio.sleep(case["wait_s"])
                    t = time.monotonic()
                    run.stop()
                    await asyncio.wait_for(task, timeout=15)
                    self.assertLess(time.monotonic() - t, case["limit_s"],
                                    case["msg"])
                finally:
                    for k, v in olds.items():
                        setattr(mock_server, k, v)
                self.assertIn("stopped", [e["type"] for e in run.history])
                self.assertNotIn("done", [e["type"] for e in run.history])

    async def test_stop_keeps_interrupted_point_as_err(self):
        """停止时被中断的那一档必须留 err 点，不能整档无痕消失。

        复现 2026-09-24 256K 档事故的可观测性缺口：旧行为下 run.results 只含
        已完成点，那一档既不 point_skipped 也无 err 点，事后无从判断它是被跳过、
        失败还是被停止——只能去翻推理引擎日志反推。"""
        old_pp = mock_server.PP
        mock_server.PP = 260.0   # 4096 ctx → prefill ~16s，够停在点中段
        try:
            run = BenchRun(self.std_cfg(ctx_list=[4096], max_tokens=8))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳探测，直接进测量批
            task = asyncio.create_task(run.run())
            await asyncio.sleep(1.2)   # 已进入 4096 档的 prefill 等待期
            run.stop()
            await asyncio.wait_for(task, timeout=20)
        finally:
            mock_server.PP = old_pp
        pts = [e["point"] for e in run.history if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, f"应留下 1 个中断点: {pts}")
        p = pts[0]
        self.assertEqual(p["ctx_target"], 4096)
        self.assertFalse(p["all_ok"])
        self.assertTrue(p.get("stopped_mid_point"))
        self.assertIn("手动停止", p.get("err") or "")
        # 半截读数不得留档
        self.assertIsNone(p["ttft_s"])
        self.assertIsNone(p["prefill_tok_s"])
        self.assertEqual(p["reqs"], [])
        # 终态仍是 stopped（收口口径不变）
        types = [e["type"] for e in run.history]
        self.assertIn("stopped", types)
        self.assertNotIn("done", types)


class TestTeardownAndSampling(MockGatewayTest):
    """收口与采样：异常收口口径（空输出/裸断连/无 usage/串行 prefill）、
    存档采样、输出上限治理与运行期播报。"""

    async def test_burst_decode_window_speed(self):
        """MOCK_BURST=15 起步突发 + 低速 TG=20：滑窗差分把突发留在基准
        样本——起步 tick 不虚高（≤TG×1.5），后续收敛 TG±25%（ADR-0007）。"""
        old_burst, old_tg = mock_server.BURST, mock_server.TG
        # mock 的合成信标正文是单字符重复（"测"×N），max_tokens=128 → 128 字符
        # ≥80 且 distinct=1 ≤12，命中退化重复判据触发弃测重测——两次测量批的
        # decode tick 混进同一订阅流污染滑窗断言。本测试主题是滑窗差分 tick
        # 机制、与异常输出守卫无关，故屏蔽检测（守卫本身的行为覆盖在
        # test_engine_unit 的 _run_rep_guarded 用例）
        old_det = bench._detect_rep_anomaly
        mock_server.BURST, mock_server.TG = 15, 20.0
        bench._detect_rep_anomaly = lambda *a, **k: None
        try:
            run = BenchRun(self.std_cfg(max_tokens=128))
            # 跳过探测直奔滑窗主题；预置 chunk_calib=1.0（事件即 token）——
            # 探测自身也带突发，其事件/token 比会把 tick 估算口径带偏
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8
            run.chunk_calib[("mock-llm-7b", "creative")] = 1.0
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = await self._drain(q, task)
        finally:
            mock_server.BURST, mock_server.TG = old_burst, old_tg
            bench._detect_rep_anomaly = old_det
        ticks = [e for e in events
                 if e.get("type") == "tick" and e.get("phase") == "decode"]
        self.assertGreaterEqual(len(ticks), 5, f"低速 decode 应有多个 tick: {len(ticks)}")
        self.assertLessEqual(ticks[0]["speed"], 20 * 1.5,
                             f"起步 tick {ticks[0]['speed']} 虚高（突发摊入读数）")
        for t in ticks[1:]:
            self.assertTrue(20 * 0.75 <= t["speed"] <= 20 * 1.25,
                            f"decode tick {t['speed']} 未收敛 TG±25%")

    async def test_no_content_point_error(self):
        """MOCK_NO_CONTENT=1（零输出：REASON 默认 0，故正文与思考内容均为空，
        即真空流）→ 请求 err 含「未收到任何输出」、点级 all_ok=False。

        注：本用例配置下是**彻底的空**（非「只思考无正文」——那种形态
        reasoning_content 非空、会置 first 而不走空流分支；要复现需同时设
        MOCK_REASON>0）。文案已按此改准（2026-09-22）。"""
        old = mock_server.NO_CONTENT
        mock_server.NO_CONTENT = "1"
        try:
            _, events = await self._run_and_collect()
        finally:
            mock_server.NO_CONTENT = old
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertFalse(p["all_ok"], "无正文点必须降级为失败")
        self.assertIn("未收到任何输出", p["reqs"][0]["err"])

    async def test_die_at_request_error(self):
        """裸断连（原始 socket 服务：流出 5 个 SSE 块后 chunked 流不发终止块
        直接 FIN）→ 请求级 err、点级 all_ok=False（模拟服务端/链路中途崩断）。
        MOCK_DIE_AT 的生成器抛错在新版 uvicorn/starlette 下会被干净收口
        （客户端视作正常结束），已无法模拟真断连，改用原始 socket 模拟。"""
        port = _free_port()
        srv = _start_raw_die_server(port)
        try:
            run = BenchRun({
                "gateway_url": f"http://127.0.0.1:{port}",
                "models": ["raw-llm"], "scenarios": ["creative"],
                "ctx_list": [0], "concurrencies": [1],
                "max_tokens": 32, "repeats": 1, "timeout_s": 30,
                "thinking": "disabled",
            })
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = await self._drain(q, task)
        finally:
            srv.close()
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertFalse(p["all_ok"], "裸断连点必须降级为失败")
        err = p["reqs"][0]["err"]
        self.assertTrue(err, "裸断连应产生请求级 err")
        self.assertNotIn("未收到任何输出", err,
                         "已流出 5 个 token，不应走零输出诊断")

    async def test_serialize_prefill_no_decode_overlap(self):
        """MOCK_SERIALIZE=1：并发 prefill 全局串行 → conc=2 的 decode 区间
        不重叠，decode_total_tok_s 置空（ADR-0006 门槛），点本身成功。"""
        old_s, old_pp = mock_server.SERIALIZE, mock_server.PP
        mock_server.SERIALIZE, mock_server.PP = "1", 1000.0   # 4K prefill ≈ 4.1s/请求
        try:
            _, events = await self._run_and_collect(
                ctx_list=[4096], concurrencies=[2], max_tokens=16)
        finally:
            mock_server.SERIALIZE, mock_server.PP = old_s, old_pp
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertTrue(p["all_ok"])
        self.assertAlmostEqual(p["decode_tok_s"], 300, delta=75)
        self.assertIsNone(p["decode_total_tok_s"],
                          "串行 prefill 下 decode 不重叠，总吞吐必须置空（ADR-0006）")
        reqs = p["reqs"]   # 引擎已按 req 序号排序
        self.assertGreater(reqs[1]["ttft_s"] - reqs[0]["ttft_s"], 3.0,
                           "第二个请求的 TTFT 应等出一个完整 prefill")

    async def test_no_usage_estimation_fallback(self):
        """无 usage 帧估算兜底（单字符事件 / 英文多字符事件两例参数化，原
        test_no_usage_estimation_fallback 与
        test_no_usage_multichar_chunks_not_overshot 两例合并）：MOCK_NO_USAGE=1
        时 usage_real=False，终值按 SSE 事件数直计兜底（1 事件/token，mock 每
        事件 1 token），chunk_calib 不播种（校准依赖真实 usage）；英文形态
        （MOCK_CHUNK_CHARS=4：1 事件/token、每事件 4 字符）下旧口径按中文
        1.5 字符/token 估算 64/1.5≈43、超断流阈值 32 触发 client_cut，新口径
        自然收尾 finish=stop（fastllm 英文输出 decode 虚高 2.8 倍的回归，
        ADR-0019）。"""
        old_u = mock_server.NO_USAGE
        mock_server.NO_USAGE = "1"
        try:
            for ch in (None, 4):
                with self.subTest(chunk_chars=ch):
                    old_ch = mock_server.CH
                    if ch is not None:
                        mock_server.CH = ch
                    try:
                        run, events = await self._run_and_collect()
                    finally:
                        mock_server.CH = old_ch
                    points = [e["point"] for e in events if e.get("type") == "point"]
                    self.assertEqual(len(points), 1)
                    p = points[0]
                    r = p["reqs"][0]
                    self.assertFalse(r["usage_real"])
                    # 兜底口径：16 个内容事件直计 = 16（= 真实 token 数，旧字符口径为 11）
                    self.assertEqual(r["out_tokens"], 16, "事件数直计 = 真实 token 数")
                    if ch is None:
                        self.assertTrue(p["all_ok"], "无 usage 不是错误，估算兜底应照常出点")
                        self.assertEqual(r["finish"], "stop")
                        self.assertEqual(run.chunk_calib, {},
                                         "无 usage 时 chunk_calib 不得播种")
                    else:
                        self.assertTrue(p["all_ok"],
                                        f"英文形态输出不得误判断流: {p['reqs']}")
                        self.assertEqual(r["finish"], "stop",
                                         "自然收尾（mock  finish_reason 恒为 stop），不得 client_cut")
                        self.assertEqual(r["text_chars"], 64)
                        # decode 读数按真实 token 数 / 实测窗口，接近 mock TG（±20%）
                        self.assertGreater(r["decode_tok_s"], 300 * 0.8)
        finally:
            mock_server.NO_USAGE = old_u

    async def test_archive_output_samples(self):
        """存档输出采样（text_sample/reason_sample，各 ≤200 字符头部）：
        正文超 200 截断（64 tokens × 4 字符 = 256 → 200）、思考流不足全留
        （mock 思考流每 token 恒 1 字符 × 2 tokens）——退化输出形态
        （~1 字符/token 塌缩）与 echo 改写口径的存档取证字段，仅供存档
        定位，前端不消费。"""
        old_ch, old_re = mock_server.CH, mock_server.REASON
        mock_server.CH, mock_server.REASON = 4, 2
        try:
            run, events = await self._run_and_collect(max_tokens=64)
        finally:
            mock_server.CH, mock_server.REASON = old_ch, old_re
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        r = points[0]["reqs"][0]
        self.assertIsNone(r["err"], f"采样测试请求应成功: {r}")
        self.assertEqual(len(r["text_sample"]), 200, "正文超 200 截断")
        self.assertEqual(r["text_chars"], 256)
        self.assertEqual(r["text_sample"], "测" * 200, "截断即停于 200 字符")
        self.assertEqual(r["reason_sample"], "思" * 2, "思考流不足全留")
        self.assertEqual(r["reason_chars"], 2)

    async def test_max_tokens_ignored_flip_and_cut(self):
        """MOCK_IGNORE_MAX_TOKENS=1（opencode zen 行为：忽略旧名、认新名）：
        探测请求（cap 64）输出跑飞 4× → 客户端断流兜底并判定改用
        max_completion_tokens；后续正式点被服务端按新名精确截断（ADR-0016）。"""
        old = mock_server.IGNORE_MT
        mock_server.IGNORE_MT = "1"
        try:
            run, events = await self._run_and_collect(max_tokens=16)
        finally:
            mock_server.IGNORE_MT = old
        self.assertEqual(run.mt_param.get("mock-llm-7b"), "max_completion_tokens",
                         "探测断流后应判定改用新参数名")
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("改用 max_completion_tokens" in s for s in statuses),
                        f"应有参数名翻转提示: {statuses}")
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["all_ok"], f"翻转后点应成功: {points[0].get('reqs')}")
        self.assertEqual(points[0]["reqs"][0]["out_tokens"], 16,
                         "新名生效后服务端应精确截断在 max_tokens")

    async def test_mct_rejected_falls_back_no_oscillation(self):
        """IGNORE_MT + REJECT_MCT（两种参数名都不生效的极端网关）：探测断流
        翻转新名 → 正式请求 400 → 回退旧名并锁定（mt_mct_rejected）→ 之后只
        断流兜底不再翻转（防振荡），点仍成功、finish=client_cut（ADR-0016）。"""
        old_i, old_r = mock_server.IGNORE_MT, mock_server.REJECT_MCT
        mock_server.IGNORE_MT, mock_server.REJECT_MCT = "1", "1"
        try:
            run, events = await self._run_and_collect(max_tokens=16)
        finally:
            mock_server.IGNORE_MT, mock_server.REJECT_MCT = old_i, old_r
        self.assertEqual(run.mt_param.get("mock-llm-7b"), "max_tokens",
                         "新名被 400 拒收后应回退旧名")
        self.assertIn("mock-llm-7b", run.mt_mct_rejected)
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("已回退 max_tokens" in s for s in statuses),
                        f"应有新名 400 回退提示: {statuses}")
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertTrue(points[0]["all_ok"], f"断流兜底下点应成功: {points[0].get('reqs')}")
        self.assertEqual(points[0]["reqs"][0]["finish"], "client_cut",
                         "两种参数名都失效时应以客户端断流收口")
        # 断流把输出收在 2× 上限附近（估算口径），而非跑飞 4×（64）
        self.assertLessEqual(points[0]["reqs"][0]["out_tokens"], 16 * 2 + 8)

    async def test_out_hint_injected_into_system_prompt(self):
        """ADR-0016 防线 2：system 提示注入输出长度双向引导（aim 目标 +
        max_tokens × out_cpt 先验 × 1.2 封顶）；ADR-0076 同文案在指令末尾
        复述一份——结构判据：实际发出的输入全文里引导片段恰好出现两次、
        且相对「system + 零输入指令」底稿的字符增量大于单份引导长度（原先
        的精确字符算术一改模板就误报，已降级为结构判据）。"""
        old_cache = mock_server.CACHE
        mock_server.CACHE = "1"    # 让 mock 记录输入全文（_SEEN），供结构判据取用
        mock_server._SEEN.clear()
        try:
            _, events = await self._run_and_collect(max_tokens=16)
            seen = list(mock_server._SEEN)   # 本次运行录到的输入全文（清空前抄走）
        finally:
            mock_server.CACHE = old_cache
            mock_server._SEEN.clear()        # 不留已见 prompt 给后续用例
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 1)
        sc = SCENARIOS["creative"]
        hint = "\n\n" + sc["out_hint"].format(limit=int(16 * sc["out_cpt"] * 1.2),
                                              aim=int(16 * sc["out_cpt"]))
        # 判据一：引导片段恰好两份——system 注入一份 + 指令末尾复述一份。
        # 从录到的已见 prompt 里按长度认领测量请求那份（引擎记的 sent_chars ==
        # 请求文本长度；探测请求是另一份、不含引导文案）
        sent_chars = points[0]["reqs"][0]["sent_chars"]
        sent_text = next((s for s in seen if len(s) == sent_chars), "")
        self.assertEqual(sent_text.count(hint), 2,
                         "system 与指令末尾应各恰好追加一份输出长度引导文案")
        # 判据二：字符增量大于单份引导长度（两份各自被追加，而非只有一份）
        base = len(sc["system"]) + len(sc["zero_instruction"])
        self.assertGreater(sent_chars - base, len(hint),
                           "相对底稿的字符增量应大于单份引导文案长度")

    async def test_repeats_one_transparency_notice(self):
        """repeats=1 透明度：运行开始播报一次「单次测量无统计效力」提醒；
        repeats≥3 不播报（不淹没复测流程的状态行）。"""
        statuses1 = await self._run_and_collect_status()
        self.assertTrue(any("无统计效力" in s for s in statuses1),
                        f"repeats=1 应有透明度提醒: {statuses1}")
        _, events3 = await self._run_and_collect(repeats=3)
        statuses3 = [e["msg"] for e in events3 if e.get("type") == "status"]
        self.assertFalse(any("无统计效力" in s for s in statuses3),
                         f"repeats=3 不应出现该提醒: {statuses3}")

    async def test_tg_pattern_alternates_decode_speed(self):
        """MOCK_TG_PATTERN（如 "60,120"）：逐请求交替 decode 速度，两档都被
        引擎采到（双峰复现的 mock 基建）。"""
        old_pp, old_pat = mock_server.PP, mock_server.TG_PATTERN
        mock_server.PP = 3000.0
        mock_server.TG_PATTERN = [60.0, 120.0]
        mock_server._REQ_SEQ = 0
        try:
            run = BenchRun(self.std_cfg(ctx_list=[0, 2048], max_tokens=32))
            run.cpt_calib[("mock-llm-7b", "creative")] = 1.8   # 跳过探测：chat 请求恰两个
            q: asyncio.Queue = asyncio.Queue()
            run.subs.add(q)
            task = asyncio.create_task(run.run())
            events = await self._drain(q, task)
        finally:
            mock_server.PP, mock_server.TG_PATTERN = old_pp, old_pat
        points = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(points), 2)
        d0, d1 = points[0]["decode_tok_s"], points[1]["decode_tok_s"]
        self.assertTrue(45 <= d0 <= 85, f"第一请求应落在 60 tok/s 档: {d0}")
        self.assertTrue(85 <= d1 <= 165, f"第二请求应落在 120 tok/s 档: {d1}")


if __name__ == "__main__":
    unittest.main()
