"""引擎编排打桩回归测试（BenchRun 编排层）：_interruptible 停止即时生效、
上下文构建偏离校正重测（ctx_retry 留痕/换 nonce）、同点位 reps 间密度记账
冻结（_calib_snapshot/_calib_restore）、_run_rep_guarded 弃测重测编排
（异常/退化/首次正常/停表/换盐）、agent 矩阵输出异常守卫与指令段缓存污染
兜底重测、停止中断点收口（_stopped_point：err 点不整档消失）。

全部用例以打桩替换 _one/_run_point/emit 等外部交互，不连网关、不起服务。

运行（均从仓库根目录执行，两种入口等价）：
    python3 -m pytest tests/unit/test_engine_orchestration.py -q
    python3 -m unittest tests.unit.test_engine_orchestration -v
    python3 -m unittest discover -s tests/unit -p "test_engine_orchestration.py"

自 tests/test_engine_unit.py 拆分（原文件已删）。
"""
import asyncio
import os
import re
import time
import unittest

import bench
from bench import (
    ANOMALY_RETEST_MAX,
    FLOOR_RETEST_MAX,
    SCENARIOS,
    BenchRun,
    _stopped_point,
)


class TestInterruptible(unittest.IsolatedAsyncioTestCase):
    """_interruptible：探测/RTT 基线等不经 _run_point 任务集的路径，停止也要即时生效。"""

    async def test_normal_returns_value(self):
        run = BenchRun({})

        async def work():
            await asyncio.sleep(0.01)
            return 42

        self.assertEqual(await run._interruptible(work()), 42)

    async def test_stop_cancels_promptly(self):
        run = BenchRun({})
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def hanging():
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(run._interruptible(hanging()))
        await started.wait()
        run.stop()
        t0 = time.monotonic()
        self.assertIsNone(await asyncio.wait_for(task, timeout=5))
        self.assertLess(time.monotonic() - t0, 2.0,
                        "停止后未及时返回（0.2s 轮询应远小于 2s）")
        self.assertTrue(cancelled.is_set(), "被包装的任务必须收到取消")


class TestCtxDeviationRetest(unittest.IsolatedAsyncioTestCase):
    """上下文构建偏离 >CTX_DEV_TOLERANCE 时按实测密度校正重测：_one 打桩
    控制实测 prompt_tokens（按消息字符数 ÷ 固定密度建模真实分词），校准打桩
    计数。触发/不触发边界、只重测一次、重测后点数据替换、删减点位不触发。"""

    MODEL, SCEN = "m", "creative"

    def _make_run(self, results, cache="bust"):
        """results: 每次请求的覆写项；prompt_tokens 可为 callable(msgs)——
        按字符数/固定密度建模真实分词密度。返回 (run, req_calls, calib_calls,
        statuses)。"""
        run = BenchRun({"cache": cache})
        state = {"n": 0}
        req_calls, calib_calls, statuses = [], [], []
        base = {"err": None, "first_abs": 100.0, "last_abs": 101.0,
                "out_tokens": 100, "stall_s": None, "stall_count": None,
                "max_gap_s": None, "decode_burst": None,
                "decode_tok_s": 100.0, "decode_tok_s_adj": None,
                "tok_per_chunk": 1.0, "ttft_s": 0.1, "ttft_net_s": 0.09,
                "prefill_tok_s": 1000.0, "prefill_net_tok_s": 1100.0,
                "cache_hit": 0, "finish": "stop", "usage_real": True,
                "sent_chars": 10 ** 9, "text_chars": 0, "reason_chars": 0,
                "total_s": 1.0, "cache_reported": False, "cache_miss": 0}

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False,
                           est_tokens=None, img_tokens=0):
            i = min(state["n"], len(results) - 1)
            state["n"] += 1
            r = dict(base)
            r.update(results[i])
            r["req"] = req_i
            r["_msgs"] = msgs
            pt = r.get("prompt_tokens")
            r["prompt_tokens"] = (pt(msgs) if callable(pt) else pt)
            req_calls.append(r)
            return r

        async def fake_calib(key, chars, filler_n, ok, prefix_obs=False):
            calib_calls.append((chars, filler_n, prefix_obs))

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run._one = fake_one
        run._calibrate_usage = fake_calib
        run.emit = fake_emit
        return run, req_calls, calib_calls, statuses

    @staticmethod
    def _density(chars_per_token: float):
        return lambda msgs: round(
            sum(len(m["content"]) for m in msgs) / chars_per_token)

    @staticmethod
    def _nonce(msgs):
        """从构造的 user 消息头部提取 nonce 串（build_messages 把 nonce 放在
        user content 开头的「参考材料（编号 {nonce}）：」行内）。"""
        head = msgs[1]["content"]
        m = re.match(r"参考材料（编号 (\d+)）：", head)
        return m.group(1) if m else None

    async def test_deviation_triggers_and_lands_on_target(self):
        """嵌套记账下新段密度失真（实测/目标 +20%）：触发重测、校正 filler
        按实测密度收缩、重测后实测命中目标 ±5%、点数据整体替换、校准两次
        执行、ctx_retry 留痕、status 播报。"""
        key = (self.MODEL, self.SCEN)
        run, reqs, calibs, statuses = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        run.prefix_kb[key] = (2000.0, 1500.0)   # 前档记账 → 走嵌套构造路径
        run.cpt_marginal[key] = 2.0             # 滑动边际估计（对新段失真）
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 2, "偏离 >10% 应触发整批重测")
        self.assertEqual(len(calibs), 2, "两次尝试都要入校准")
        first_real = reqs[0]["prompt_tokens"]
        self.assertGreater(first_real, 4096 * 1.1, "打桩密度应产出 >10% 偏离")
        self.assertEqual(point["ctx_retry"], first_real, "留痕首次实测均值")
        self.assertEqual(point["prompt_tokens"], reqs[1]["prompt_tokens"])
        self.assertLess(point["prompt_tokens"], 4096 * 1.05,
                        f"校正重测后应命中目标 ±5%（实测 {point['prompt_tokens']}）")
        self.assertGreater(point["prompt_tokens"], 4096 * 0.95)
        # 校正只动 filler：实测偏大 → 重测构造总字符量应收缩（重测换新 nonce，
        # 仅头部编号数字与 filler 变化，指令/系统等固定开销不变，字符量差
        # 几乎全部来自 filler 校正）
        c1 = calibs[0][0]
        c2 = calibs[1][0]
        self.assertLess(c2, c1, "实测偏大 → 重测构造总字符量应收缩")
        # 点级数据整体替换：reqs 是重测批、派生字段取重测值
        self.assertEqual(point["reqs"][0]["prompt_tokens"],
                         reqs[1]["prompt_tokens"])
        # 重测换新 nonce（防前缀缓存命中共享前缀），user 消息头部编号不同
        self.assertNotEqual(self._nonce(reqs[1]["_msgs"]),
                            self._nonce(reqs[0]["_msgs"]),
                            "重测 nonce 必须与首测不同（bust 前缀缓存）")
        self.assertTrue(any("校正重测" in s for s in statuses),
                        f"应有校正重测播报: {statuses}")

    async def test_retry_msgs_prefix_busted(self):
        """回归：重测 msgs 与首测 msgs 不共享可缓存前缀——nonce 在 user
        消息开头，nonce 串不同 → 两条 user 内容从头部编号处即分叉，服务端
        前缀缓存（命中共享前缀而非整条内容）无法把首测缓存带给重测。
        （真实案例：同 nonce 重测 32K 档 cache_hit 30720/32170，TTFT 失真）"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        u1 = reqs[0]["_msgs"][1]["content"]
        u2 = reqs[1]["_msgs"][1]["content"]
        self.assertNotEqual(self._nonce(reqs[0]["_msgs"]),
                            self._nonce(reqs[1]["_msgs"]))
        # 两条 user 内容的公共前缀必须止于头部编号行内（换 nonce 即整段
        # bust），而非共享到 filler 段
        common = os.path.commonprefix([u1, u2])
        self.assertLess(len(common), u1.index("：") + 1,
                        "重测与首测的 user 内容应在头部编号处分叉")

    async def test_within_tolerance_no_retest(self):
        """偏离 ≤10%：不触发，单次请求、ctx_retry 为 None。"""
        run, reqs, calibs, statuses = self._make_run(
            [{"prompt_tokens": self._density(1.8)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(calibs), 1)
        self.assertIsNone(point["ctx_retry"])
        self.assertFalse(any("校正重测" in s for s in statuses))

    async def test_stable_cache_no_retest(self):
        """cache=stable：重测会命中前缀缓存污染读数，跳过。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)}], cache="stable")
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(calibs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_zero_ctx_no_retest(self):
        """ctx=0 档（固定开销主导、无填充）：不触发。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 0, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_no_usage_no_retest(self):
        """usage_real=False（无真实 usage 回传）：无密度样本，不触发。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": 9999, "usage_real": False}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(calibs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_retests_only_once(self):
        """重测后仍超差：不再重试（最多 1 次），接受并留 ctx_retry 痕。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 2, "只重测一次")
        self.assertEqual(len(calibs), 2)
        self.assertEqual(point["ctx_retry"], reqs[0]["prompt_tokens"])

    async def test_trimmed_point_no_retest(self):
        """删减重试成功点位（sent_chars < 构造量）：偏离源于服务端超窗保护，
        重测必然复现，不触发。"""
        run, reqs, calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5), "sent_chars": 100}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1)
        self.assertIsNone(point["ctx_retry"])

    async def test_stop_flag_no_retest(self):
        """stop_flag 置位（首批在途期间停止）：不再发起重测批。"""

        def stop_during_call(msgs):
            run.stop_flag = True   # 模拟首批在途期间置停
            return 99999

        run, reqs, calibs, _ = self._make_run([{"prompt_tokens": stop_during_call}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertEqual(len(reqs), 1, "首批照常收数")
        self.assertEqual(len(calibs), 1, "只有首批校准，无重测批")
        self.assertIsNone(point["ctx_retry"])

    async def test_conc2_batch_retested_together(self):
        """并发 2：整批（conc 个请求）重跑，每个槽位的重测 nonce 与该槽位
        首测 nonce 不同（防前缀缓存污染）、批内槽位间也不重合，替换点数据。"""
        run, reqs, calibs, statuses = self._make_run(
            [{"prompt_tokens": self._density(1.5)}] * 2)
        point = await run._run_point(None, self.MODEL, self.SCEN, 8192, 2, 1.8)
        self.assertEqual(len(reqs), 4, "首批 2 + 重测批 2")
        self.assertEqual(len(calibs), 2)
        self.assertEqual(point["reqs"][0]["prompt_tokens"],
                         reqs[2]["prompt_tokens"])
        self.assertEqual([r["req"] for r in point["reqs"]], [0, 1],
                         "重测批按 req 序号替换原批")
        first_nonces = [self._nonce(r["_msgs"]) for r in reqs[:2]]
        retry_nonces = [self._nonce(r["_msgs"]) for r in reqs[2:]]
        self.assertNotEqual(first_nonces[0], retry_nonces[0],
                            "槽位 0 重测 nonce 应与首测不同")
        self.assertNotEqual(first_nonces[1], retry_nonces[1],
                            "槽位 1 重测 nonce 应与首测不同")
        self.assertEqual(len(set(retry_nonces)), len(retry_nonces),
                         "重测批内各槽位 nonce 互不相同")

    async def test_first_try_chars_lands_with_reading(self):
        """D：首测构造字符量随 ctx_retry 成对落档。只存读数不存字节量时，
        「同一段字节 ↔ 哪个读数」事后无法验证（2026-10-10 计数稳定性定性
        因此只能从其他点位反推）。未触发重测时两键同为 None。"""
        key = (self.MODEL, self.SCEN)
        run, reqs, _calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.5)},
             {"prompt_tokens": self._density(1.5)}])
        run.prefix_kb[key] = (2000.0, 1500.0)
        run.cpt_marginal[key] = 2.0
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertIsNotNone(point["ctx_retry"], "前提：本例触发重测")
        first_chars = sum(len(m["content"]) for m in reqs[0]["_msgs"])
        self.assertEqual(point["ctx_retry_chars"], first_chars,
                         "ctx_retry_chars 应等于首测那一批的构造字符量")
        self.assertNotEqual(point["ctx_retry_chars"],
                            sum(len(m["content"]) for m in reqs[1]["_msgs"]),
                            "校正后的重测批字符量已变，两者不得混淆")

    async def test_no_retest_leaves_retry_keys_none(self):
        """未触发重测（实测落在 ±10% 内）时两个留痕键为 None——它们只在
        真的发生过一次「首测」时才有意义。构造按 cpt=1.8 估字符数，
        故打桩密度取 1.8 即零偏离。"""
        run, _reqs, _calibs, _ = self._make_run(
            [{"prompt_tokens": self._density(1.8)}])
        point = await run._run_point(None, self.MODEL, self.SCEN, 4096, 1, 1.8)
        self.assertIsNone(point["ctx_retry"])
        self.assertIsNone(point["ctx_retry_chars"])


class TestPointCalibFreeze(unittest.IsolatedAsyncioTestCase):
    """A：同一点位的 repeats 之间冻结密度记账（_calib_snapshot/_calib_restore）。

    背景：校准是「边测边学」的滑动记账，同点位 reps 之间这个学习使 rep1 的
    构造量被 rep0 的读数牵动——实测（Lvllm/Flash-Next、64K 单档×repeats=3、
    无回退无重测无异常、all_ok 全 True）逐 rep prompt_tokens = 92608/51530/
    69112，聚合行是三个不同长度的均值（且服务端计数零噪声，漂移全在构造侧）。
    修复后：同点位内各 rep 构造逐位一致；点位之间照常学习（ADR-0005 不削弱）。"""

    KEY = ("m", "code")

    def _run(self):
        run = BenchRun({})
        run.cpt_calib[self.KEY] = 3.4
        run.cpt_calib_w[self.KEY] = 1000.0
        run.prefix_kb[self.KEY] = (111778.0, 32045.0)
        run.cpt_marginal[self.KEY] = 3.2
        run.cpt_marginal_w[self.KEY] = 5000.0
        return run

    def test_snapshot_restore_roundtrip_including_missing_key(self):
        run = self._run()
        snap = run._calib_snapshot(("other", "code"))   # 全缺失键
        run.cpt_calib[("other", "code")] = 9.9          # 模拟 rep0 写入
        run.prefix_kb[("other", "code")] = (1.0, 2.0)
        run._calib_restore(("other", "code"), snap)
        self.assertNotIn(("other", "code"), run.cpt_calib,
                         "快照时不存在的键，恢复后必须仍不存在（而非留 None）")
        self.assertNotIn(("other", "code"), run.prefix_kb)

    def test_restore_only_touches_this_key(self):
        """快照/恢复不得连带清掉其他 (model, scenario) 的记账：同一次运行里
        多模型/多场景交替出点，越界恢复会把别人的学习成果抹掉。"""
        run = self._run()
        other = ("m2", "creative")
        run.prefix_kb[other] = (5.0, 6.0)
        run.cpt_calib[other] = 1.5
        snap = run._calib_snapshot(self.KEY)
        run.prefix_kb[self.KEY] = (999.0, 888.0)   # 本 key 被 rep0 改写
        run.cpt_calib[self.KEY] = 7.7
        run._calib_restore(self.KEY, snap)
        self.assertEqual(run.prefix_kb[self.KEY], (111778.0, 32045.0))
        self.assertEqual(run.cpt_calib[self.KEY], 3.4)
        self.assertEqual(run.prefix_kb[other], (5.0, 6.0), "其他 key 不得受影响")
        self.assertEqual(run.cpt_calib[other], 1.5)

    def test_weight_accumulation_not_triple_counted(self):
        """恢复后末轮观测仍入账一次：repeats=3 的三份相同观测不得把滑动
        平均的字符权重计成 3 倍（否则该点对校准的影响被人为抬高）。"""
        run = self._run()
        snap = run._calib_snapshot(self.KEY)
        w0 = run.cpt_calib_w[self.KEY]
        for _rep in range(3):
            if _rep > 0:
                run._calib_restore(self.KEY, snap)
            run.cpt_calib_w[self.KEY] += 100.0   # 模拟一轮校准写入
        self.assertEqual(run.cpt_calib_w[self.KEY], w0 + 100.0,
                         "三轮只应计一轮的权重（前两轮在恢复时被回退）")

    def test_freeze_call_sites_present(self):
        """源头卡：两个出点循环（文本阶梯 / agent 矩阵）都要有 rep 间冻结，
        漏一个则该路径继续混口径。"""
        src = open(bench.__file__, encoding="utf-8").read()
        self.assertGreaterEqual(src.count("_calib_restore"), 3,
                                "快照/恢复原语定义 1 处 + 两处 rep 循环调用")
        for marker in ("calib_snap = self._calib_snapshot((model, scenario))",
                       "calib_snap = self._calib_snapshot(key)"):
            self.assertIn(marker, src,
                          "缺一条出点路径的快照（文本阶梯或 agent 矩阵）")


class TestRunRepGuarded(unittest.IsolatedAsyncioTestCase):
    """_run_rep_guarded：弃测重测编排——重测预算按异常形态分两档（低于
    有效地板的早停最多 FLOOR_RETEST_MAX 次、其余异常最多
    ANOMALY_RETEST_MAX 次，重测换 nonce 破缓存 + stream_salt 换语料
    落点由 _run_point 内部落实）、预算耗尽仍异常高于地板按实留档
    anomaly、仍低于地板按测量失败留档（读数置空 + err）、首次正常不重测、
    stop_flag 置位不检测不重测。_run_point 打桩脚本化返回，emit 收队列。"""

    ANOM_PT = {   # 实测案例形态：4K 档 echo 模式仅输出 4/512 tokens
        "reqs": [{"err": None, "finish": "stop", "out_tokens": 4,
                  "ttft_s": 0.12, "decode_tok_s": 29.1,
                  "text_sample": "好的，以下是", "reason_sample": "",
                  "in_sample": "指令材料头部",
                  "_in_full": "【user】" + "指令材料" * 150}]}
    OK_PT = {
        "reqs": [{"err": None, "finish": "stop", "out_tokens": 512,
                  "ttft_s": 0.11, "decode_tok_s": 60.0,
                  "text_sample": "", "reason_sample": "",
                  "_in_full": "【user】正常输入全文"}]}

    def _make_run(self):
        run = BenchRun({})
        statuses = []

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run.emit = fake_emit
        return run, statuses

    def _script(self, run, points, post=None):
        """_run_point 打桩：按脚本顺序返回；post(i) 在第 i 次调用返回后执行
        （用于模拟测量期间置位 stop_flag）。reqs 逐个浅拷贝——真实 _run_point
        每批返回全新 req 字典，共享 fixture 的 reqs 列表会让守卫对肇事 req 的
        anomaly/in_text 标记跨用例/跨调用串扰。"""
        calls = []

        async def fake_run_point(*args, **kwargs):
            i = len(calls)
            calls.append((args, kwargs))
            src = points[min(i, len(points) - 1)]
            pt = {**src, "reqs": [dict(r) for r in src["reqs"]]}
            if post:
                post(i)
            return pt

        run._run_point = fake_run_point
        return calls

    async def test_anomaly_then_ok_retest(self):
        """首次 early_stop 异常 → 弃测重测一次 → 重测正常：返回正常 pt 带
        reps_discarded（1 条，含肇事读数与样本、rep 轮次号、in_text 输入全文
        抄本）、无 anomaly 键、播报状态；返回 pt 的 reqs 已剥离 _in_full 瞬态
        字段（正常批存档不得带输入全文）。"""
        run, statuses = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 2, "异常应弃测重测一次")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt.get("reps_discarded"), [{
            "anomaly": "early_stop", "req": 0, "rep": 1,
            "out_tokens": 4,
            "finish": "stop", "ttft_s": 0.12, "decode_tok_s": 29.1,
            "text_sample": "好的，以下是", "reason_sample": "",
            "in_sample": "指令材料头部",
            "in_text": "【user】" + "指令材料" * 150}])
        self.assertGreater(len(pt["reps_discarded"][0]["in_text"]), 200,
                           "in_text 为输入全文抄本，不设 200 字符采样上限")
        self.assertNotIn("_in_full", pt["reqs"][0],
                         "返回 pt 的 reqs 应已剥离 _in_full 瞬态字段")
        self.assertTrue(any("仅输出 4/512 tokens" in s and "弃测重测" in s
                            for s in statuses), f"应有弃测播报: {statuses}")

    async def test_anomaly_persists_marked(self):
        """重测仍异常但高于有效地板：接受结果、pt 带 anomaly='early_stop'
        按实留档、reps_discarded 仅 1 条（不无限重试）；肇事 req 打
        anomaly + in_text（输入全文抄本），全部 req 剥离 _in_full。"""
        run, _ = self._make_run()
        above = {"reqs": [dict(self.ANOM_PT["reqs"][0], out_tokens=200)]}
        calls = self._script(run, [dict(self.ANOM_PT), above])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1 + ANOMALY_RETEST_MAX)
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), ANOMALY_RETEST_MAX)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop",
                         "重测仍异常的肇事 req 应带 anomaly 标记")
        self.assertIsNone(culprit.get("err"),
                          "高于地板的持续异常不按测量失败留档")
        self.assertNotIn("err", pt)
        self.assertEqual(culprit["in_text"], "【user】" + "指令材料" * 150,
                         "肇事 req 应带输入全文 in_text")
        self.assertNotIn("_in_full", culprit, "_in_full 剥离后全文留在 in_text")

    async def test_anomaly_persists_below_floor_failed(self):
        """重测预算（FLOOR_RETEST_MAX=2，2026-09-17 收紧）耗尽仍低于有效
        输出地板（4 < 100）：读数完全不可信——按测量失败留档：pt err 落档、
        all_ok=False、点读数置空不进聚合；肇事 req 带 err + anomaly +
        in_text；reps_discarded 逐条留痕每次弃测（ADR-0032 的按实留档
        宽容止于地板之上）。"""
        run, statuses = self._make_run()
        anom_full = {**self.ANOM_PT, "out_tokens": 4, "decode_tok_s": 29.1,
                     "prefill_tok_s": 300.0, "all_ok": True}
        calls = self._script(run, [dict(anom_full), dict(anom_full)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1 + FLOOR_RETEST_MAX,
                         "低于地板的早停最多重测 FLOOR_RETEST_MAX 次")
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertFalse(pt["all_ok"])
        self.assertIn("早停", pt["err"])
        self.assertIsNone(pt["out_tokens"], "垃圾读数应置空不进聚合")
        self.assertIsNone(pt["decode_tok_s"])
        self.assertIsNone(pt["prefill_tok_s"])
        self.assertEqual(len(pt["reps_discarded"]), FLOOR_RETEST_MAX)
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop")
        self.assertIn("早停", culprit["err"])
        self.assertEqual(culprit["in_text"], "【user】" + "指令材料" * 150)
        self.assertTrue(any("测量失败留档" in s for s in statuses),
                        f"应有失败留档播报: {statuses}")

    async def test_below_floor_recovers_on_later_retest(self):
        """低于地板的早停在第 2 次重测恢复：持续重测直到正常——返回正常
        pt（无 anomaly 键）、reps_discarded 逐条留痕 2 次弃测、读数来自
        恢复批；高于地板的异常无此宽限（仍 1 次，见上例）。"""
        run, _ = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.ANOM_PT),
                                   dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 3, "首测 + 2 次弃测重测后恢复")
        self.assertNotIn("anomaly", pt)
        self.assertNotIn("err", pt)
        self.assertEqual(len(pt["reps_discarded"]), 2)
        self.assertTrue(all(d["anomaly"] == "early_stop"
                            for d in pt["reps_discarded"]))
        self.assertEqual(pt["reqs"][0]["out_tokens"], 512,
                         "最终读数来自恢复批")

    async def test_retest_passes_stream_salt(self):
        """弃测重测换料（2026-09-17 拍板）：首测不带 stream_salt（=0，嵌套
        前缀确定性不变），每次重测带递增 stream_salt 驱动 _run_point 平移
        语料窗口翻 echo 落点——确定型早停与落点绑定，同料重测只会复现
        同一答案。"""
        run, _ = self._make_run()
        calls = self._script(run, [dict(self.ANOM_PT), dict(self.ANOM_PT),
                                   dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("stream_salt", calls[0][1],
                         "首测不传盐（等价 salt=0，保持嵌套前缀口径）")
        self.assertEqual(calls[1][1].get("stream_salt"), 1)
        self.assertEqual(calls[2][1].get("stream_salt"), 2)
        self.assertNotIn("anomaly", pt)

    async def test_ok_no_retest_no_keys(self):
        """首次正常：不重测、不附加 anomaly/reps_discarded 键。"""
        run, statuses = self._make_run()
        calls = self._script(run, [dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("anomaly", pt)
        self.assertNotIn("reps_discarded", pt)
        self.assertEqual(statuses, [], "正常路径不应有弃测播报")

    async def test_stop_flag_no_detect_no_retest(self):
        """stop_flag 置位（测量期间停止）：不检测直接返回，不发起重测。"""
        run, statuses = self._make_run()

        def set_stop(i):
            run.stop_flag = True   # 模拟首批在途期间置停

        calls = self._script(run, [dict(self.ANOM_PT)], post=set_stop)
        pt = await run._run_rep_guarded(None, "m", "creative", 4096, 1, 1.8,
                                        0, 512, True)
        self.assertEqual(len(calls), 1, "停止后不重测")
        self.assertNotIn("anomaly", pt)
        self.assertNotIn("reps_discarded", pt)
        self.assertEqual(statuses, [], "停止路径不应有弃测播报")

    async def test_degenerate_retest_message(self):
        """退化重复路径：播报文案不含 token 数（形态与 early_stop 区分）。"""
        run, statuses = self._make_run()
        degen = {"reqs": [{"err": None, "finish": "stop", "out_tokens": 512,
                           "ttft_s": 0.3, "decode_tok_s": 40.0,
                           "text_sample": "",
                           "reason_sample": "        " + " ".join(
                               f"{i}." for i in range(6, 60))}]}
        calls = self._script(run, [degen, dict(self.OK_PT)])
        pt = await run._run_rep_guarded(None, "m", "creative", 8192, 1, 1.8,
                                        0, 512, False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(pt["reps_discarded"][0]["anomaly"], "degenerate")
        self.assertTrue(any("退化重复" in s for s in statuses), f"{statuses}")


class TestAgentMatrixAnomalyGuard(unittest.IsolatedAsyncioTestCase):
    """_run_agent_matrix 的输出异常弃测重测（ADR-0032 口径接入 agent 矩阵）：
    测量批早停（free 阈值 0.5×）/退化命中 → 整批弃测、换新 nonce+错位指令
    材料重测一次（reps_discarded 留痕、点干净）、重测仍异常按实留档
    anomaly；与指令长度偏离校正的次序为先校正后异常判定（各 1 次、总尝试
    ≤3 次）。_one 打桩脚本化返回（quiet 预热/基线批固定返回），emit 收队列。"""

    MODEL = "m"
    INST = 512

    def _make_run(self):
        run = BenchRun({"max_tokens": 512, "agent_cache_ladder": [0],
                        "agent_inst_ladder": [self.INST]})
        statuses = []

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run.emit = fake_emit
        return run, statuses

    def _script_one(self, run, base_prompt, meas_script):
        """_one 打桩：quiet 预热/基线批固定返回 base_prompt 实测 tokens；
        测量批（quiet=False）按 meas_script 逐次返回（元素为对基准 req 字典
        的键覆写；耗尽后重复最后一项）。calls 记录 (quiet, 末条消息) 供断言
        调用序与 nonce tag。"""
        calls = []

        def base_req(req_i, mt):
            return {"req": req_i, "err": None, "usage_real": True,
                    "finish": "stop", "out_tokens": mt,
                    "prompt_tokens": 0, "ttft_s": 0.05, "decode_tok_s": 60.0,
                    "text_sample": "OK", "reason_sample": "",
                    "in_sample": "指令材料头部",
                    "_in_full": "【user】指令输入全文",
                    "max_gap_s": None, "cache_reported": False}

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False, **kw):
            calls.append((quiet, msgs))   # msgs[1] 带编号 nonce、msgs[2] 为指令消息
            if quiet:
                return {"req": req_i, "err": None,
                        "prompt_tokens": base_prompt, "usage_real": True}
            i = min(len([c for c in calls if not c[0]]) - 1,
                    len(meas_script) - 1)
            r = base_req(req_i, max_tokens)
            r["prompt_tokens"] = base_prompt + self.INST   # 差分恰中档位目标
            r.update(meas_script[i])
            return r

        run._one = fake_one
        return calls

    async def _run_matrix(self, run):
        await run._run_agent_matrix(None, self.MODEL, "agent", 1, 3.5,
                                    None, 1)
        self.assertTrue(run.results, "repeats=1 应即时落盘出点")
        return run.results[0]

    async def test_early_stop_retest_clean(self):
        """首测早停（out 4/512 < 0.5×）→ 弃测整批重测一次 → 点干净（无
        anomaly 键）、reps_discarded 恰 1 条（0032 形状含肇事读数与样本）、
        读数来自重测正常批、播报弃测状态。"""
        run, statuses = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "The task context is incomplete.",
             "decode_tok_s": 30.0},
            {"out_tokens": 512, "decode_tok_s": 60.0},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 4, "零缓存档：基线+测量 ×（首测+重测）")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt["reps_discarded"], [{
            "anomaly": "early_stop", "req": 0, "rep": 1,
            "out_tokens": 4,
            "finish": "stop", "ttft_s": 0.05, "decode_tok_s": 30.0,
            "text_sample": "The task context is incomplete.",
            "reason_sample": "", "in_sample": "指令材料头部",
            "in_text": "【user】指令输入全文"}])
        self.assertNotIn("_in_full", pt["reqs"][0],
                         "点位 reqs 应剥离 _in_full 瞬态字段")
        self.assertEqual(pt["out_tokens"], 512, "最终读数来自重测正常批")
        self.assertEqual(pt["finish"], "stop")
        self.assertTrue(pt["all_ok"])
        self.assertTrue(any("提前停止：仅输出 4/512 tokens" in s
                            and "弃测重测" in s for s in statuses), statuses)

    async def test_persistent_early_stop_marked(self):
        """重测仍早停但高于有效地板（200 ≥ min(100,512)，仍 <0.5×512 判据）：
        anomaly='early_stop' 按实留档、reps_discarded 仅 1 条（最多重测 1 次，
        不无限重试洗掉）、出点不中断；留档样本取自异常轮——肇事 req 带
        anomaly + in_text（输入全文抄本），全部 req 剥离 _in_full。"""
        run, _ = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "short answer"},
            {"out_tokens": 200, "text_sample": "still short but above floor"},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 4, "首测 + 异常重测各一轮（基线+测量）")
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), 1)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop",
                         "重测仍异常的肇事 req 应带 anomaly 标记")
        self.assertIsNone(culprit.get("err"),
                          "高于地板的持续异常不按测量失败留档")
        self.assertEqual(culprit["in_text"], "【user】指令输入全文",
                         "肇事 req 应带输入全文 in_text")
        self.assertFalse(any("_in_full" in r for r in pt["reqs"]),
                         "点位 reqs 应全部剥离 _in_full 瞬态字段")
        self.assertTrue(pt["all_ok"])

    async def test_persistent_early_stop_below_floor_failed(self):
        """重测预算（FLOOR_RETEST_MAX=2，2026-09-17 收紧）耗尽仍低于有效
        输出地板（7 < min(100,512)=100）：读数完全不可信——按测量失败留档：
        pt err 落档、all_ok=False、读数置空不进聚合；anomaly 标记与
        reps_discarded（逐条留痕每次弃测）保留供复核。"""
        run, _ = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "short answer"},
            {"out_tokens": 7, "text_sample": "short answer again"},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 2 * (1 + FLOOR_RETEST_MAX),
                         "首测 + FLOOR_RETEST_MAX 次弃测重测（基线+测量）")
        self.assertEqual(pt["anomaly"], "early_stop")
        self.assertEqual(len(pt["reps_discarded"]), FLOOR_RETEST_MAX)
        self.assertEqual(pt["reps_discarded"][0]["out_tokens"], 4)
        self.assertFalse(pt["all_ok"], "重测仍低于地板：点按测量失败留档")
        self.assertIn("早停", pt["err"])
        self.assertIsNone(pt["out_tokens"], "垃圾读数应置空不进聚合")
        culprit = pt["reqs"][0]
        self.assertEqual(culprit["anomaly"], "early_stop")
        self.assertIn("早停", culprit["err"])
        self.assertEqual(culprit["in_text"], "【user】指令输入全文")
        self.assertFalse(any("_in_full" in r for r in pt["reqs"]))

    async def test_degenerate_retest(self):
        """退化重复（文本通道样本 ≥80 字符且字符集 ≤12）弃测重测：留痕
        anomaly='degenerate'、重测正常即点干净。"""
        run, statuses = self._make_run()
        degen = " " + " ".join(f"{i}." for i in range(6, 60))
        calls = self._script_one(run, 300, [
            {"out_tokens": 512, "text_sample": degen},
            {"out_tokens": 512, "text_sample": "OK"},
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 4)
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt["reps_discarded"][0]["anomaly"], "degenerate")
        self.assertEqual(pt["out_tokens"], 512)
        self.assertTrue(any("退化重复" in s and "弃测重测" in s
                            for s in statuses), statuses)

    async def test_input_correction_then_anomaly(self):
        """次序与有界性：指令长度偏离先触发输入校正（nonce tag 'r'）、校正批
        又早停才触发输出异常重测（tag 逐次 'a0'/'a1'/…，2026-09-16 地板
        多次重试换盐）——共 3 次测量尝试；重测正常即点干净且
        inst_real_tokens 收敛。"""
        run, statuses = self._make_run()
        calls = self._script_one(run, 300, [
            {"prompt_tokens": 600, "out_tokens": 512},    # 差分 300，偏离 -41%
            {"out_tokens": 4, "text_sample": "incomplete"},   # 校正批早停
            {"prompt_tokens": 812, "out_tokens": 512},        # 异常重测正常
        ])
        pt = await self._run_matrix(run)
        self.assertEqual(len(calls), 6, "基线+测量 ×3（首测/校正/异常重测）")
        tags = [re.search(rf"-z{self.INST}(\w*)）：", msgs[1]["content"]).group(1)
                or ""
                for quiet, msgs in calls if not quiet]
        self.assertEqual(tags, ["", "r", "a0"],
                         f"先输入校正（r）后异常重测（a0）: {tags}")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(len(pt["reps_discarded"]), 1)
        self.assertIsNotNone(pt["inst_real_tokens"])
        self.assertLess(abs(pt["inst_real_tokens"] - self.INST),
                        0.1 * self.INST + 8, pt["inst_real_tokens"])

    async def test_retest_shifts_instruction_material(self):
        """异常重测换料：指令材料在语料流中错位取段（seg_salt），长度不变、
        内容不同——防整条 prompt 二次命中缓存。"""
        run, _ = self._make_run()
        calls = self._script_one(run, 300, [
            {"out_tokens": 4, "text_sample": "x"},
            {"out_tokens": 512, "text_sample": "OK"},
        ])
        await self._run_matrix(run)
        meas = [msgs[2]["content"] for quiet, msgs in calls if not quiet]
        seg1 = meas[0][len(SCENARIOS["agent"]["inst_prefix"]):]
        seg2 = meas[1][len(SCENARIOS["agent"]["inst_prefix"]):]
        self.assertEqual(len(seg1), len(seg2), "指令材料长度不变")
        self.assertNotEqual(seg1, seg2, "指令材料内容错位（换料防二次命中缓存）")


class TestAgentMatrixCachePollution(unittest.IsolatedAsyncioTestCase):
    """_run_agent_matrix 的指令段缓存残留污染兜底重测（C>0 回传命中批）：
    实测未命中 tokens 远低于指令档期望（_cache_pollution_verdict）→ 整批
    弃测、游标换新段重测一次（reps_discarded 标 cache_pollution 带实测
    uncached/expected、两次尝试指令材料错位）；重测仍偏差过大
    point["anomaly"] 按实留档不再重试；干净批不触发。_one 打桩脚本化
    cache_hit，emit 收队列。"""

    MODEL = "m"
    PRIME = 15000   # 预热实测 prompt（缓存档上下文 tokens）
    INST = 2048

    def _make_run(self):
        run = BenchRun({"max_tokens": 512, "agent_cache_ladder": [4096],
                        "agent_inst_ladder": [self.INST]})
        statuses = []

        async def fake_emit(ev):
            if ev.get("type") == "status":
                statuses.append(ev["msg"])

        run.emit = fake_emit
        return run, statuses

    def _script_one(self, run, hit_script):
        """_one 打桩：quiet 预热批固定返回 PRIME 实测 prompt；测量批
        （quiet=False）prompt = PRIME + INST（差分恰中档位目标、不触发
        输入校正），cache_hit 按 hit_script 逐次返回（耗尽重复最后一项）。
        calls 记录 (quiet, msgs) 供断言调用序与指令材料错位。"""
        calls = []

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False, **kw):
            calls.append((quiet, msgs))
            if quiet:
                return {"req": req_i, "err": None,
                        "prompt_tokens": self.PRIME, "usage_real": True}
            i = min(len([c for c in calls if not c[0]]) - 1,
                    len(hit_script) - 1)
            return {"req": req_i, "err": None, "usage_real": True,
                    "finish": "stop", "out_tokens": max_tokens,
                    "prompt_tokens": self.PRIME + self.INST,
                    "ttft_s": 0.05, "decode_tok_s": 60.0,
                    "cache_reported": True, "cache_hit": hit_script[i],
                    "text_sample": "OK", "reason_sample": "",
                    "in_sample": "指令材料头部",
                    "_in_full": "【user】指令输入全文",
                    "max_gap_s": None}

        run._one = fake_one
        return calls

    async def _run_matrix(self, run):
        await run._run_agent_matrix(None, self.MODEL, "agent", 1, 3.5,
                                    None, 1)
        self.assertTrue(run.results, "repeats=1 应即时落盘出点")
        return run.results[0]

    async def test_pollution_retest_clean(self):
        """首测批未命中 500（期望 2048，比值 0.24）→ 判污染弃测、游标换段
        重测一次 → 重测干净：点无 anomaly、reps_discarded 恰 1 条带实测
        uncached/expected、两次尝试指令材料错位不重叠、播报换料重测。"""
        run, statuses = self._make_run()
        calls = self._script_one(
            run, [self.PRIME + self.INST - 500, self.PRIME])
        pt = await self._run_matrix(run)
        meas = [msgs[2]["content"] for quiet, msgs in calls if not quiet]
        self.assertEqual(len(meas), 2, "首测 + 污染重测各一轮测量批")
        self.assertNotIn("anomaly", pt)
        self.assertEqual(pt["reps_discarded"], [{
            "anomaly": "cache_pollution", "rep": 1,
            "uncached_tokens": 500, "expected_tokens": self.INST}])
        self.assertNotEqual(meas[0], meas[1],
                            "污染重测指令材料应随游标错位换段")
        pre, suf = (SCENARIOS["agent"]["inst_prefix"],
                    SCENARIOS["agent"]["inst_suffix"])
        seg1 = meas[0][len(pre):len(meas[0]) - len(suf)]
        seg2 = meas[1][len(pre):len(meas[1]) - len(suf)]
        self.assertFalse(seg2.startswith(seg1[:200]),
                         "两次尝试指令材料不互为前缀")
        self.assertEqual(pt["inst_real_tokens"], self.INST)
        self.assertTrue(any("实测未命中 500 tokens 与指令档期望 2048 偏差过大"
                            in s and "换料重测" in s for s in statuses),
                        statuses)

    async def test_pollution_persistent_marked(self):
        """重测仍偏差过大：point["anomaly"]="cache_pollution" 按实留档、
        reps_discarded 仅 1 条（最多重测 1 次，不无限重试洗掉）、出点不
        中断。"""
        run, _ = self._make_run()
        calls = self._script_one(
            run, [self.PRIME + self.INST - 500, self.PRIME + self.INST - 500])
        pt = await self._run_matrix(run)
        self.assertEqual(len([c for c in calls if not c[0]]), 2,
                         "污染重测仅一次")
        self.assertEqual(pt["anomaly"], "cache_pollution")
        self.assertEqual(len(pt["reps_discarded"]), 1)
        self.assertEqual(pt["reps_discarded"][0]["anomaly"], "cache_pollution")
        self.assertTrue(pt["all_ok"])

    async def test_clean_batch_no_retest(self):
        """干净批（未命中 ≈ 指令档期望）不触发：仅一轮测量批、无
        reps_discarded 键。"""
        run, _ = self._make_run()
        calls = self._script_one(run, [self.PRIME])
        pt = await self._run_matrix(run)
        self.assertEqual(len([c for c in calls if not c[0]]), 1)
        self.assertNotIn("reps_discarded", pt)
        self.assertNotIn("anomaly", pt)


class TestStoppedPoint(unittest.TestCase):
    """停止时中断点的收口（_stopped_point）：中断档必须在存档里留下 err 点，
    不能再整档无痕消失（2026-09-24 256K 档事故的直接后果）。"""

    _IDENT = {"model": "m1", "scenario": "code", "kind": "llm",
              "ctx_target": 262144, "concurrency": 1}

    def test_identity_survives_without_completed_reps(self):
        """零已完成轮次（首个复测就被打断，正是 256K 那种）：身份字段仍由
        ident 兜底产出，档位/并发可定位。"""
        pt = _stopped_point([], dict(self._IDENT), "测量被手动停止")
        self.assertEqual(pt["model"], "m1")
        self.assertEqual(pt["ctx_target"], 262144)
        self.assertEqual(pt["concurrency"], 1)
        self.assertFalse(pt["all_ok"])
        self.assertEqual(pt["err"], "测量被手动停止")
        self.assertTrue(pt["stopped_mid_point"])

    def test_readings_voided_not_carried_over(self):
        """半截读数一律置空：已完成轮次的 ttft/prefill/decode 不得残留在
        中断点上（否则该点会混进对比图当真值）。"""
        rep = {**self._IDENT, "ttft_s": 12.3, "prefill_tok_s": 800.0,
               "decode_tok_s": 40.0, "prompt_tokens": 262144, "out_tokens": 512,
               "batch_time_s": 99.9, "all_ok": True, "rtf": 0.4, "speed_x": 2.5,
               "reqs": [{"req": 0, "err": None}], "reps": [{"ttft_s": 1.0}]}
        pt = _stopped_point([rep], dict(self._IDENT), "停止")
        for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "prompt_tokens",
                  "out_tokens", "batch_time_s"):
            self.assertIsNone(pt[k], f"半截读数未清空：{k}")
        # 媒体口径键不在 _VOID_POINT_KEYS 内，靠 _stopped_point 显式清空
        self.assertIsNone(pt["rtf"])
        self.assertIsNone(pt["speed_x"])
        self.assertEqual(pt["reqs"], [])
        self.assertNotIn("reps", pt)

    def test_ident_overrides_rep_shell(self):
        """ident 覆盖骨架：停止发生在贴边裁减档时，ctx_target 记的是裁后实测
        档位，须与已完成点同口径（调用方另以 ctx_edge 留痕原档位）。"""
        rep = {**self._IDENT, "ctx_target": 260608, "ttft_s": 5.0}
        pt = _stopped_point([rep], {**self._IDENT, "ctx_target": 260608}, "停止")
        self.assertEqual(pt["ctx_target"], 260608)

if __name__ == "__main__":
    unittest.main()
