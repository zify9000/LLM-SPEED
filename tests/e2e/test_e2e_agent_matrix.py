"""e2e 组 · agent 缓存×指令矩阵全链路（mock 网关真跑 + 引擎最小矩阵）。

覆盖两簇：
- F 编排：矩阵逐组合出点与点位字段（缓存档/指令档/inst_real_tokens/口径
  版本戳）、预热 quiet 请求发 prime tick、速率闸跟随缓存命中判别口径（毒药
  先验不误杀缓存档）、确定性失败弹悬浮岛、指令真实长度偏离校正补测、
  MOCK_EARLY_STOP_LONG 弃测重测（一次性钩子重测救回 / 持续早停留档判失败）。
- N 缓存判别：MOCK_CACHE 前缀缓存命中回传（增量/全量双口径、逐尝试错位取段、
  先验曲线登记）、无缓存能力诚实退化、网关剥离命中字段时凭 TTFT 走平或全量
  预期佐证通道估算命中、缓存慢读与预热 usage 缺失下仍估算命中且不污染先验
  曲线、预算跳档（C+I 超部署上限整档 point_skipped）、软超窗后矩阵提前收口。

自 tests/test_e2e_mock.py 拆分（原文件已删）。
精简记录：F 簇原 7 例，其中 test_agent_matrix_early_stop_retest 与
test_agent_matrix_persistent_early_stop_marked 的 reps_discarded 断言块几乎
逐字重复（同一 MOCK_EARLY_STOP_LONG 钩子的两种取值，断言同轴），合并为
test_agent_matrix_early_stop_retest_and_persistent_marked 一条、按钩子取值
参数化；N 簇 7 例原样搬运。

运行（任一方式，都要求**从仓库根目录**执行，无需 pytest 也可跑）：
    .venv/bin/python -m pytest tests/e2e/test_e2e_agent_matrix.py -q
    .venv/bin/python -m unittest tests.e2e.test_e2e_agent_matrix -v
    .venv/bin/python -m unittest discover -s tests/e2e -p "test_e2e_agent_matrix.py"
"""
# 顺序是硬约束：_harness 顶部先 import tests._shared（落 MOCK_* 播种），之后
# 才 import mock_server——mock_server 在 import 时读环境变量，反序会采到默认值
# （如 PP=1200 而非 3000），断言口径全错。
from tests.e2e._harness import MockGatewayTest

import bench
import mock_server


class TestAgentMatrixE2E(MockGatewayTest):
    """agent 缓存×指令矩阵：编排（F）与缓存判别（N）两簇。"""

    async def test_agent_matrix_smoke(self):
        """Agent 矩阵端到端（ADR-0042）：缓存×指令阶梯逐组合出点，点位带
        ctx_target=缓存档 / inst_tokens=指令档 / inst_real_tokens（链内差分），
        无旧链的 turn 系字段。"""
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[0, 4096], agent_inst_ladder=[128, 256])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2 缓存档 × 2 指令档 = 4 点（预热不计点）")
        self.assertEqual([(p["ctx_target"], p["inst_tokens"]) for p in pts],
                         [(0, 128), (0, 256), (4096, 128), (4096, 256)],
                         "缓存档升序外层 × 指令档升序内层")
        for p in pts:
            self.assertTrue(p["all_ok"])
            self.assertNotIn("turn", p)
            self.assertNotIn("turn_phase", p)
            self.assertNotIn("turn_delta", p)
            self.assertIsNotNone(p["total_s"], "端到端总时长应产出")
            self.assertIn("inst_real_tokens", p, "指令真实长度字段应在位")
            # 口径版本（ADR-0083）：agent 矩阵点走独立构造点，同样要带版本戳
            self.assertEqual(p.get("metric_version"), bench.METRIC_VERSION)
        self.assertLess(pts[0]["prompt_tokens"], 500, "零缓存档为纯指令小 prompt")
        self.assertGreater(pts[2]["prompt_tokens"], 3500, "缓存档注入足量轨迹语料")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_prime_ticks_visible(self):
        """预热/基线 quiet 请求发实时 tick（带 prime 标记、不入 KPI 语义）+
        批终 done 帧：大缓存档预热 prefill 长达数十秒，全程无帧会让前端实时表
        与「进行中请求」空转（实时组件"未加载"错觉的来源）"""
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[4096], agent_inst_ladder=[128])
        prime_ticks = [e for e in events
                       if e.get("type") == "tick" and e.get("prime")]
        self.assertTrue(prime_ticks, "预热请求应发带 prime 标记的 tick")
        phases = {e.get("phase") for e in prime_ticks}
        self.assertIn("prefill", phases, "预热请求发出即有 prefill 帧")
        self.assertIn("done", phases, "预热批结束应有 done 终帧（前端撤行用）")
        # 预热帧带场景定位字段（前端建行/分组用），且不污染测量点 tick
        for e in prime_ticks:
            self.assertEqual(e.get("scenario"), "agent")
            self.assertEqual(e.get("ctx"), 4096)
        measure_ticks = [e for e in events
                         if e.get("type") == "tick" and not e.get("prime")
                         and e.get("phase") in ("prefill", "decode")]
        self.assertTrue(measure_ticks, "测量请求 tick 不带 prime 标记")

    async def test_agent_matrix_gate_follows_cache_estimate(self):
        """速率闸跟随缓存命中判别口径（ADR-0052）：网关未回传命中但缓存档
        经预热实测断言大概率命中（est_tokens = 指令档 + 块尾）时，闸门按预估
        增量口径评估而非全量——全量口径在缓存加速的短 TTFT 上会误杀诚实点
        （DeepSeek 33K 缓存档全量 8580 tok/s 撞 10× 先验线实例）。毒药先验
        100 tok/s（×10 线 = 1000，mock 全量 ~1200 必杀）下缓存档点仍应通过"""
        def poison(run):
            run.prefill_curve[("mock-llm-7b", "agent")] = [(8192.0, 100.0)]
        _, events = await self._run_agent_matrix_collect(
            preset=poison, agent_cache_ladder=[8192], agent_inst_ladder=[128])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertTrue(pts)
        for p in pts:
            self.assertTrue(p["all_ok"],
                            f"缓存档点不应被速率闸误杀: {p.get('reqs')}")
        self.assertFalse(any("物理上不可能" in (r.get("err") or "")
                             for p in pts for r in p.get("reqs", [])),
                         "缓存档（预估增量口径）不得触发速率闸")

    async def test_failure_toast_on_rate_gate(self):
        """确定性失败弹悬浮岛（ADR-0052）：速率闸判假（非缓存档走全量口径、
        未命中量 ≥8K 先验臂生效）除 error tick 外随发 status toast:true——
        此前只有实时表一个 ✗，界面无任何提醒"""
        def poison(run):
            run.prefill_curve[("mock-llm-7b", "agent")] = [(8192.0, 100.0)]
        _, events = await self._run_agent_matrix_collect(
            preset=poison, agent_cache_ladder=[0], agent_inst_ladder=[8192])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1)
        self.assertFalse(pts[0]["all_ok"], "毒药先验下全量口径应判假成功")
        self.assertTrue(any("物理上不可能" in (r.get("err") or "")
                            for r in pts[0]["reqs"]))
        toasts = [e for e in events
                  if e.get("type") == "status" and e.get("toast")]
        self.assertTrue(any("失败" in e.get("msg", "") for e in toasts),
                        "请求最终失败应弹悬浮岛提醒（status toast:true）")

    async def test_agent_matrix_inst_real_tokens_correction(self):
        """指令真实长度差分与偏离校正补测（ADR-0042）：预置输入系数与 mock
        分词错位（3.5 vs 1.8）放大指令长度偏差 → 触发「实测指令长度偏离目标，
        按实测密度校正补测一次」，补测后 inst_real_tokens 收敛到档位目标
        （±10%+8 内），补测替换点位（不出新点）。"""
        _, events = await self._run_agent_matrix_collect(
            cpt=3.5, agent_cache_ladder=[0], agent_inst_ladder=[1024])
        # 预置输入系数与 mock 分词错位（3.5 vs 1.8）制造构造密度偏差：
        # 1024 档按 3.5 构造 3.5K 字符材料、mock 按 1.8 折 token → 实测 ~2 倍
        # 偏离，必触发校正补测
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 1, "校正补测替换点位，不新增点")
        p = pts[0]
        self.assertTrue(p["all_ok"])
        self.assertIsNotNone(p["inst_real_tokens"])
        self.assertLess(abs(p["inst_real_tokens"] - 1024), 0.1 * 1024 + 8,
                        f"补测后指令真实长度应收敛到 1024±10%: {p['inst_real_tokens']}")
        self.assertTrue(any("校正补测" in e.get("msg", "")
                            for e in events if e.get("type") == "status"),
                        "应发校正补测 status 说明")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_early_stop_retest_and_persistent_marked(self):
        """agent 矩阵输出异常弃测重测与持续早停留档（ADR-0032 口径接入 agent
        矩阵，MOCK_EARLY_STOP_LONG 两种取值参数化：1=一次性钩子、99=持续早停；
        两取值的弃测留痕断言同轴，原先各占一条用例）。

        ①钩子=1（一次性）：首测批 finish=stop 仅 4/32 tokens（< 0.5×max_tokens，
        实测「材料截断」短答形态）→ 整批弃测、换新 nonce+错位指令材料重测——
        重测正常即点干净（anomaly 键缺席）、reps_discarded 恰 1 条留痕、最终
        读数来自重测批、SSE 有弃测播报；预热/基线超短任务不命中钩子（>64
        字符判据），不干扰预热。
        ②钩子=99（持续）：4/32 tokens 低于有效地板 min(100,32)=32，弃测重测
        FLOOR_RETEST_MAX 次后仍低于地板 → 按测量失败留档（2026-09-16 拍板）：
        point err 落档、all_ok=False、读数置空不进聚合；anomaly 标记与
        reps_discarded（逐条留痕每次弃测）保留供复核，出点不中断。"""
        for hook, persistent in ((1, False), (99, True)):
            with self.subTest(early_stop_long=hook):
                old = mock_server.EARLY_STOP_LONG
                mock_server.EARLY_STOP_LONG = hook
                try:
                    _, events = await self._run_agent_matrix_collect(
                        agent_cache_ladder=[0], agent_inst_ladder=[512])
                finally:
                    mock_server.EARLY_STOP_LONG = old
                pts = [e["point"] for e in events if e.get("type") == "point"]
                self.assertEqual(len(pts), 1, "弃测重测替换点位，不新增点")
                p = pts[0]
                if not persistent:
                    # ① 一次性钩子：重测即正常，点干净
                    self.assertEqual(p["reps_discarded"], [{
                        "anomaly": "early_stop", "req": 0, "rep": 1,
                        "out_tokens": 4,
                        "finish": "stop", "ttft_s": p["reps_discarded"][0]["ttft_s"],
                        "decode_tok_s": p["reps_discarded"][0]["decode_tok_s"],
                        "text_sample": "测" * 4, "reason_sample": "",
                        "in_sample": p["reps_discarded"][0]["in_sample"],
                        "in_text": p["reps_discarded"][0]["in_text"]}])
                    self.assertTrue(p["reps_discarded"][0]["in_sample"],
                                    "弃测留痕带输入侧样本（末条消息头部，ADR-0053）")
                    # 矩阵弃测留痕同样带输入全文（预设上下文 + 指令两条 user 消息）
                    it = p["reps_discarded"][0]["in_text"]
                    self.assertGreater(len(it), 200, "in_text 为输入全文，不设采样上限")
                    self.assertTrue(it.startswith("【system】"), f"全文应逐条渲染角色: {it[:80]!r}")
                    self.assertEqual(it.count("【user】"), 2,
                                     "矩阵构造为 system + 两条 user 消息")
                    self.assertNotIn("anomaly", p, "重测成功即正常留档")
                    self.assertEqual(p["out_tokens"], 32, "最终读数来自重测正常批")
                    self.assertTrue(p["all_ok"])
                    statuses = [e.get("msg", "") for e in events if e.get("type") == "status"]
                    self.assertTrue(any("提前停止：仅输出 4/32 tokens" in s
                                        and "弃测重测" in s for s in statuses),
                                    f"SSE 应有弃测重测播报: {[s for s in statuses if '弃测' in s]}")
                else:
                    # ② 持续早停：弃测重测耗尽后按测量失败留档
                    self.assertEqual(p["anomaly"], "early_stop")
                    self.assertEqual(len(p["reps_discarded"]), bench.FLOOR_RETEST_MAX)
                    self.assertEqual(p["reps_discarded"][0]["out_tokens"], 4)
                    self.assertFalse(p["all_ok"], "重测仍低于地板：点按测量失败留档")
                    self.assertIn("早停", p["err"])
                    self.assertIsNone(p["out_tokens"], "垃圾读数应置空不进聚合")
                    self.assertIsNone(p["decode_tok_s"])
                    self.assertIsNone(p["prefill_tok_s"])
                    # 留档样本取自异常轮：肇事 req 带 err + anomaly + in_text（输入
                    # 全文抄本），全部 req 剥离 _in_full 瞬态字段（存档/SSE 不得带全文）
                    culprit = p["reqs"][0]
                    self.assertEqual(culprit["anomaly"], "early_stop")
                    self.assertIn("早停", culprit["err"])
                    self.assertGreater(len(culprit["in_text"]), 200)
                    self.assertTrue(culprit["in_text"].startswith("【system】"))
                    self.assertFalse(any("_in_full" in r for r in p["reqs"]))
                self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_cache_hit_reported(self):
        """agent 矩阵（MOCK_CACHE=1 前缀缓存）：预热请求把上下文写入缓存且
        不计入测点（总点数=组合数）；测量请求命中回传（cache_hit_tokens>0、
        cache_reported 真），prefill 为增量口径（≈ mock PP，远小于全量口径）；
        同缓存档逐尝试错位取段（游标），指令材料不再嵌套前缀——各指令档
        命中同为上下文主体（不再随档增大而命中更多）；指令真实长度
        （测量 prompt − 预热/基线 prompt 链内差分）≈ 档位目标（CPT 对齐时不
        触发偏离校正）。"""
        old_cache = mock_server.CACHE
        mock_server.CACHE = "1"
        mock_server._SEEN.clear()
        try:
            run, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 4096], agent_inst_ladder=[512, 2048])
        finally:
            mock_server.CACHE = old_cache
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4, "2×2 组合应出 4 点（预热不计入测点）")
        by = {(p["ctx_target"], p["inst_tokens"]): p for p in pts}
        self.assertTrue(all(p["all_ok"] for p in pts))
        # 零缓存档：mock 回传命中字段但只命中系统提示级+编号行前缀
        # （基线批与测量同 nonce，命中含编号行与 system 双向 out_hint，
        # ~168 tokens）
        for it in (512, 2048):
            p = by[(0, it)]
            self.assertTrue(p["cache_reported"])
            self.assertLess(p["cache_hit_tokens"], 240,
                            "零缓存档只应命中系统提示级共享前缀")
            # 指令真实长度（链内差分）：CPT 对齐时不触发偏离校正
            self.assertIsNotNone(p["inst_real_tokens"],
                                 "测量/基线批 usage 齐备时应产出指令真实长度")
            self.assertLess(abs(p["inst_real_tokens"] - it), 0.1 * it + 8,
                            f"指令真实长度应贴近档位 {it}")
        # 缓存档：命中 ≈ 已缓存上下文主体（回传真值，非估算口径）
        h512, h2048 = by[(4096, 512)], by[(4096, 2048)]
        for p in (h512, h2048):
            self.assertTrue(p["cache_reported"], "mock 回传命中字段，非估算口径")
            self.assertGreater(p["cache_hit_tokens"], 3500,
                               "应命中预热的上下文主体")
            self.assertIsNotNone(p["inst_real_tokens"])
            # 增量 prefill 口径：未命中 tokens ÷ TTFT ≈ MOCK_PP=3000
            self.assertAlmostEqual(p["prefill_tok_s"], 3000, delta=900,
                                   msg="增量 prefill 偏离 mock 设定")
            # 口径一致性（additive 双口径）：全量口径 ≫ 增量口径——上下文主体
            # 命中，只增量 prefill 指令段（2048 档未命中占比 ~1/3，比值贴 3
            # 以下，阈值取 2）
            self.assertGreater(p.get("prefill_full_tok_s"),
                               p["prefill_tok_s"] * 2)
            for r in p["reqs"]:
                self.assertIsNotNone(r.get("prefill_uncached_tok_s"),
                                     "缓存命中请求应带 req 级未命中口径字段")
            # 端到端总时长 + conc=1 单链总吞吐同单请求 decode 口径
            self.assertIsNotNone(p["total_s"])
            self.assertIsNotNone(p["decode_total_tok_s"])
            self.assertAlmostEqual(p["decode_total_tok_s"], p["decode_tok_s"],
                                   delta=1.0)
        # 指令材料逐尝试错位取段（游标，不再嵌套前缀）：2048 档段不含 512
        # 档段，两档命中同为预热的上下文主体——差异仅指令模板共享前缀
        # （inst_prefix 百余字符）与边界 token，远小于一个指令档
        self.assertAlmostEqual(h2048["cache_hit_tokens"],
                               h512["cache_hit_tokens"], delta=256)
        # TTFT 随自身增量放大：2048 档未命中 ≈ 2K tokens vs 512 档 ≈ 0.5K
        self.assertGreater(h2048["ttft_s"], h512["ttft_s"] * 2)
        # 预热完成 status 带实测 prompt 量（前端展示用）
        self.assertTrue(any("实测 prompt" in e.get("msg", "")
                            for e in events if e.get("type") == "status"))
        # 每点实测速率登记进先验曲线：（0,512) 与 (4096,512) 未命中量同为
        # ~0.5K，同 ctx 坐标互相覆盖 → 4 点落 3 个曲线点
        curve = run.prefill_curve.get(("mock-llm-7b", "agent"), [])
        self.assertGreaterEqual(len(curve), 3,
                                f"4 点应登记 ≥3 个先验点（同 x 覆盖）: {curve}")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_without_cache_degrades_honestly(self):
        """无前缀缓存能力（MOCK_CACHE 关）：缓存档 TTFT 随全量 prompt 增长——
        缓存迹象判别为「无缓存」：命中列 None（前端显示未回传，不展示假设
        命中数），prefill 退回全量口径实测真值（≈ mock PP），不按假设命中
        折算。"""
        mock_server._SEEN.clear()
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[0, 4096], agent_inst_ladder=[512])
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2)
        by = {p["ctx_target"]: p for p in pts}
        base, cached = by[0], by[4096]
        self.assertFalse(cached["cache_reported"], "无缓存字段回传时必须标记非真值")
        self.assertIsNone(cached["cache_hit_tokens"],
                          "无缓存迹象时命中为 None（前端显示未回传）")
        # 缓存档全量 ~4.6K tokens 重算：TTFT ≈ 7× 零缓存档（~0.7K tokens）
        self.assertGreater(cached["ttft_s"], base["ttft_s"] * 3)
        # 全量口径：prompt_tokens ÷ TTFT ≈ MOCK_PP=3000（真实可测速率）
        self.assertAlmostEqual(cached["prefill_tok_s"], 3000, delta=450)
        # 命中 None 时增量口径 = 全量口径（无假设命中折算）
        self.assertAlmostEqual(cached.get("prefill_full_tok_s"),
                               cached["prefill_tok_s"], delta=0.05)

    async def test_agent_matrix_cache_assumed_when_ttft_flat(self):
        """缓存生效但网关剥离命中字段（MOCK_CACHE_NOREPORT）：缓存档测量 TTFT
        相对同指令零缓存档走平 → 判别「有缓存迹象」，命中按预热实测 prompt
        估算（cache_reported=False，前端加 ≈），prefill 为增量口径 ≈ mock PP。"""
        old = mock_server.CACHE_NOREPORT
        mock_server.CACHE_NOREPORT = "1"
        mock_server._SEEN.clear()
        try:
            _, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 4096], agent_inst_ladder=[512])
        finally:
            mock_server.CACHE_NOREPORT = old
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 2)
        by = {p["ctx_target"]: p for p in pts}
        base, cached = by[0], by[4096]
        self.assertFalse(cached["cache_reported"])
        self.assertIsNotNone(cached["cache_hit_tokens"],
                             "有缓存迹象应按预热实测 prompt 估算命中")
        self.assertGreater(cached["cache_hit_tokens"], 3000,
                           "估算命中应为预热实测 prompt 量级")
        self.assertAlmostEqual(cached["prefill_tok_s"], 3000, delta=900,
                               msg="增量 prefill 偏离 mock 设定")
        self.assertLess(cached["ttft_s"], base["ttft_s"] * 2 + 0.3,
                        "缓存档 TTFT 应相对零缓存档走平")

    async def test_agent_matrix_cache_hit_slow_read_estimated(self):
        """缓存命中但读取慢（MOCK_CACHE_READ_TOK_S=N，N≫PP，模拟实测
        DeepSeek 后端 KV 大部在 CPU 的形态）：大缓存档 TTFT 随命中规模线性
        增长、相对零缓存档不走平（走平判别失效），但显著低于同规模全量
        prefill 预期 → 全量预期/佐证通道判别估算命中（cache_reported=False），
        prefill 为增量口径（被缓存读取摊薄，远小于全量口径）；估算命中点
        不登记进 prefill 先验曲线（假全量速率会污染估值锚点）。"""
        mock_server.CACHE_READ_S = 8000.0
        mock_server._SEEN.clear()
        try:
            run, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 8192], agent_inst_ladder=[512, 2048])
        finally:
            mock_server.CACHE_READ_S = 0.0
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4)
        by = {(p["ctx_target"], p["inst_tokens"]): p for p in pts}
        self.assertTrue(all(p["all_ok"] for p in pts))
        for it in (512, 2048):
            base, cached = by[(0, it)], by[(8192, it)]
            # 走平判别失效：缓存读取耗时 ≫ 零缓存档全量 prefill，TTFT 不走平
            self.assertGreater(cached["ttft_s"], base["ttft_s"] + 0.55,
                               f"inst {it} 缓存档 TTFT 应显著高于零缓存档")
            # 仍判出估算命中（全量预期/佐证通道）：非回传、非真值口径
            self.assertFalse(cached["cache_reported"])
            self.assertIsNotNone(cached["cache_hit_tokens"],
                                 "TTFT 显著低于同规模全量预期应估算命中")
            self.assertGreater(cached["cache_hit_tokens"], 7000,
                               "估算命中应为预热实测 prompt 量级")
            # prefill 增量口径：TTFT 含缓存读取耗时，速率被摊薄
            self.assertLess(cached["prefill_tok_s"], 1500,
                            "增量 prefill 应远小于全量口径")
            self.assertGreater(cached.get("prefill_full_tok_s", 0),
                               cached["prefill_tok_s"] * 3)
        # 先验曲线污染防护：估算命中点不登记，曲线只含零缓存档全量正证据
        curve = run.prefill_curve.get(("mock-llm-7b", "agent"), [])
        self.assertEqual(len(curve), 2, f"只应登记 2 个零缓存档先验点: {curve}")
        self.assertTrue(all(x < 4096 for x, _ in curve),
                        f"先验点 x 应为零缓存档 prompt 规模: {curve}")

    async def test_agent_matrix_prime_usage_missing_still_estimated(self):
        """预热 usage 缺失复现（实测 2026-09-16 本地网关事故：MOCK_PRIME_NO_USAGE
        模拟网关间歇丢预热 usage 帧 + MOCK_CACHE_READ_TOK_S 慢读缓存使走平判别
        失效）：旧口径 prime=0 把 proven 通道一并短路、整档虚高速率留档并污染
        先验；现估算基准回退 ptok−inst，全量预期/佐证通道照常判别估算命中
        （cache_reported=False、≈口径），prefill 为增量口径。"""
        mock_server.CACHE_READ_S = 8000.0
        mock_server.PRIME_NO_USAGE = "1"
        mock_server._SEEN.clear()
        try:
            run, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 8192], agent_inst_ladder=[512, 2048])
        finally:
            mock_server.CACHE_READ_S = 0.0
            mock_server.PRIME_NO_USAGE = ""
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        self.assertEqual(len(pts), 4)
        by = {(p["ctx_target"], p["inst_tokens"]): p for p in pts}
        self.assertTrue(all(p["all_ok"] for p in pts))
        for it in (512, 2048):
            cached = by[(8192, it)]
            self.assertFalse(cached["cache_reported"])
            self.assertIsNotNone(
                cached["cache_hit_tokens"],
                "预热 usage 缺失时估算基准应回退 ptok−inst，照常估算命中")
            self.assertGreater(cached["cache_hit_tokens"], 7000,
                               "回退估算命中 ≈ ptok−inst，应为缓存档量级")
            self.assertLess(cached["prefill_tok_s"],
                            cached.get("prefill_full_tok_s", 0) / 3,
                            "prefill 应为增量口径（远小于全量口径）")
        # 先验曲线零污染：估算命中点不登记，只剩零缓存档全量正证据锚点
        curve = run.prefill_curve.get(("mock-llm-7b", "agent"), [])
        self.assertTrue(all(x < 8192 for x, _ in curve),
                        f"虚高缓存档速率不应登记进先验曲线: {curve}")

    async def test_agent_matrix_ctx_limit_skips_rungs(self):
        """预算跳档（ADR-0042）：C+I 连同输出预算超部署上限的组合整档跳过
        （point_skipped 按缓存档聚合 count），其余组合照常出点——矩阵不再
        「提前终止」而是「整档跳过」。"""
        mock_server._SEEN.clear()
        _, events = await self._run_agent_matrix_collect(
            agent_cache_ladder=[0, 4096, 16384], agent_inst_ladder=[512],
            model_max_ctx={"mock-llm-7b": 8192})
        pts = [e["point"] for e in events if e.get("type") == "point"]
        # 上限 8192−32−1024=7136：缓存 16384 档全部组合超预算
        self.assertEqual([(p["ctx_target"], p["inst_tokens"]) for p in pts],
                         [(0, 512), (4096, 512)])
        self.assertTrue(all(p["all_ok"] for p in pts))
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(skipped), 1, f"16384 档应恰好一条 point_skipped: {skipped}")
        self.assertEqual(skipped[0]["ctx"], 16384)
        self.assertEqual(skipped[0]["count"], 1)
        self.assertIn("超出部署上限", skipped[0]["msg"],
                      "point_skipped 须带 msg 供前端常驻展示")
        self.assertIn("done", [e["type"] for e in events])

    async def test_agent_matrix_terminates_on_overflow(self):
        """某组合软超窗（MOCK_SOFT_MAX_CTX 卡在中段组合）矩阵提前收口：失败
        组合照常出点（all_ok=False），其后组合 point_skipped（count=剩余组合
        数，实测事故：超窗后仍继续构造更大组合打爆上限）。"""
        old = mock_server.SOFT_MAX_CTX
        mock_server.SOFT_MAX_CTX = 4500   # 缓存 4096 的预热/128 指令档可过；2048 档（~6K）超限
        try:
            _, events = await self._run_agent_matrix_collect(
                agent_cache_ladder=[0, 4096, 8192],
                agent_inst_ladder=[128, 2048])
        finally:
            mock_server.SOFT_MAX_CTX = old
            mock_server._SEEN.clear()
        pts = [e["point"] for e in events if e.get("type") == "point"]
        combos = [(p["ctx_target"], p["inst_tokens"]) for p in pts]
        self.assertEqual(combos, [(0, 128), (0, 2048), (4096, 128), (4096, 2048)],
                         "矩阵必须在超窗组合处收口，不得跑完")
        self.assertTrue(all(p["all_ok"] for p in pts[:3]), "前三个组合应正常成功")
        failed = pts[3]
        self.assertFalse(failed["all_ok"], "超窗组合必须判失败")
        self.assertIn("prompt too long", failed["reqs"][0].get("err") or "")
        skipped = [e for e in events if e.get("type") == "point_skipped"]
        self.assertEqual(len(skipped), 1, f"剩余组合应恰好一条 point_skipped: {skipped}")
        self.assertEqual(skipped[0]["count"], 2, "剩余组合 (8192,128)+(8192,2048)")
        self.assertEqual(skipped[0]["ctx"], 8192)
        self.assertIn("done", [e["type"] for e in events])
