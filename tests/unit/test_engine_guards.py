"""引擎守卫/异常/超窗/重试组（纯逻辑，不起服务）。

覆盖：错误语义化、超窗回退（删减上下文重试）与部署上限贴边裁减判定、
退化重复文本判别、rep 异常（早停/退化）检测、echo 真·预填续写参数、
物理速率闸及其调用点、瞬时失败兜底重试（传输异常/429/5xx/空流分类、
重试留痕、耗尽收口、stop_flag 中止、quiet 静默）。

自 tests/test_engine_unit.py 拆分（原文件已删）。

运行（任一方式，都要求**从仓库根目录**执行，无需 pytest 也可跑）：
    .venv/bin/python -m pytest tests/unit/test_engine_guards.py -q
    .venv/bin/python -m unittest tests.unit.test_engine_guards -v
    .venv/bin/python -m unittest discover -s tests/unit -p test_engine_guards.py
"""
import asyncio
import json
import unittest
import unittest.mock

import httpx

import bench
from bench import (
    CTX_HEADROOM,
    BenchRun,
    RATE_GATE_MIN_TOKENS,
    _classify_http_error,
    _detect_rep_anomaly,
    _is_collapsed_reply,
    _is_ctx_overflow,
    _is_degenerate_text,
    _is_implausible_prefill_rate,
    _is_soft_ctx_overflow,
    _is_transient_exc,
    _is_transient_http_status,
    _plan_ctx_edge,
    _retry_after_s,
    _ctx_short,
    _trim_middle,
    build_messages,
    TRANSIENT_RETRY_MAX,
    EMPTY_STREAM_RETRY_MAX,
)


class TestClassifyHttpError(unittest.TestCase):
    """ADR-0008/0004 错误语义化。"""

    def test_gateway_timeout(self):
        for code in (502, 503, 504):
            msg = _classify_http_error(code, "bad gateway")
            self.assertIn("网关", msg)

    def test_context_exceeded(self):
        self.assertIn("上下文窗口", _classify_http_error(
            400, '{"error":{"message":"context length exceeds limit"}}'))
        # Kimi 用 401 报超窗（ADR-0004）
        self.assertIn("上下文窗口", _classify_http_error(
            401, "invalid_authentication_error: input context exceeds the only allowed"))

    def test_plain_error_passthrough(self):
        msg = _classify_http_error(400, "unknown parameter: xyz")
        self.assertIn("400", msg)
        self.assertIn("xyz", msg)


class TestCtxOverflowRetry(unittest.TestCase):
    """超窗回退（删减上下文重试）的判定与删减辅助。"""

    def test_overflow_detection(self):
        # OpenAI 系：code=context_length_exceeded；message 无 exceed 字样
        self.assertTrue(_is_ctx_overflow(
            400, '{"error":{"message":"This model\'s maximum context length is 8192 '
                 'tokens. However, you requested 9000 tokens",'
                 '"code":"context_length_exceeded"}}'))
        # llama.cpp 系
        self.assertTrue(_is_ctx_overflow(
            400, "the request exceeds the available context size"))
        # 普通 400 不误判
        self.assertFalse(_is_ctx_overflow(400, "unknown parameter: thinking"))
        self.assertFalse(_is_ctx_overflow(500, "internal error"))

    def test_trim_middle_keeps_head_and_tail(self):
        msgs, _, filler_n = build_messages("creative", 4096, 1.5, "nonce-9")
        content = msgs[1]["content"]
        keep = int(len(content) * 0.75)
        trimmed = _trim_middle(content, keep)
        self.assertLessEqual(len(trimmed), keep + 20)   # 删减标记仅十余字符
        self.assertTrue(trimmed.startswith(content[:100]))     # 编号行保留
        self.assertTrue(trimmed.endswith(content[-100:]))      # 任务指令保留
        self.assertIn("已删减", trimmed)
        # keep ≥ 原长时不动
        self.assertEqual(_trim_middle(content, len(content) + 1), content)

    def test_soft_overflow_detection(self):
        """fastllm 系软超窗占位回复：整体输出即 "prompt too long"（≤32 字符），
        content 与 reasoning_content 两个通道的头部文本都参与判定。"""
        self.assertTrue(_is_soft_ctx_overflow("prompt too long", 15))
        self.assertTrue(_is_soft_ctx_overflow("  Prompt Too Long\n", 18))  # 大小写/空白
        self.assertTrue(_is_soft_ctx_overflow("prompt too long", 32))      # 边界长度
        # reasoning 通道：content 头为空/非占位串，思考流头部命中同样识别
        self.assertTrue(_is_soft_ctx_overflow("", 15, "prompt too long"))
        self.assertTrue(_is_soft_ctx_overflow("你好。", 18, "  Prompt Too Long "))
        # 正常输出不误判：前缀相同但有后续内容 / 超长
        self.assertFalse(_is_soft_ctx_overflow("prompt too long, please retry", 29))
        self.assertFalse(_is_soft_ctx_overflow("prompt too long" + "x" * 20, 35))
        self.assertFalse(_is_soft_ctx_overflow("Prompt is too long to process", 29))
        self.assertFalse(_is_soft_ctx_overflow("", 0))
        # reasoning 通道非占位串不误判；超长总量直接短路
        self.assertFalse(_is_soft_ctx_overflow("你好。", 18, "首先分析问题"))
        self.assertFalse(_is_soft_ctx_overflow("", 40, "prompt too long"))

    def test_collapsed_reply_guard(self):
        """输出塌缩守卫（通用保险）：finish=stop 但输出近乎为空而 prompt 很大
        ——占位软拒绝的占位串未必命中已知文案。阈值卡在大 prompt 上，小
        prompt 的合法短输出不误伤。"""
        # 典型占位软拒绝（未知文案、单 token）：判塌缩
        self.assertTrue(_is_collapsed_reply(277867, 1, "stop", 15))
        self.assertTrue(_is_collapsed_reply(8192, 2, "stop", 32))     # 边界
        # 合法短回复不误伤：小 prompt / 输出超阈 / finish 非 stop
        self.assertFalse(_is_collapsed_reply(4096, 1, "stop", 15))    # 小 prompt
        self.assertFalse(_is_collapsed_reply(0, 1, "stop", 5))        # 0K 自由短答
        self.assertFalse(_is_collapsed_reply(20000, 3, "stop", 15))   # 输出超阈
        self.assertFalse(_is_collapsed_reply(20000, 1, "stop", 33))   # 字符超阈
        self.assertFalse(_is_collapsed_reply(20000, 1, "length", 15))  # 非自然收尾
        self.assertFalse(_is_collapsed_reply(None, 1, "stop", 15))    # 无 prompt 数据
        self.assertFalse(_is_collapsed_reply(20000, None, "stop", 15))

    def test_implausible_prefill_rate_gate(self):
        """物理速率闸：prefill 速率超该未命中量的先验曲线 10× 或绝对上限
        10 万 tok/s 即判「物理上不可能」；未命中量 <8K 与无先验时都只按
        绝对上限判（小 prompt 区诚实速率随 TTFT 固定开销摊薄近线性爬升，
        10× 先验带无判别力——实测 agent 零缓存档 inst=1024/4096 的
        569~976 tok/s 被误杀，同运行 306~722 tok/s 的兄弟点反而过闸）。"""
        # 绝对上限（无先验也生效，x 大小无关）
        self.assertTrue(_is_implausible_prefill_rate(387516.0, 73728, []))
        self.assertTrue(_is_implausible_prefill_rate(100000.0, 16384, []))   # 边界
        self.assertTrue(_is_implausible_prefill_rate(100001.0, 100, []),
                        "小 prompt 也受绝对上限约束")
        # 先验倍数（x ≥8K 才走先验带）：5000 > 10×300 → 判；2500 < 3000 → 不判
        self.assertTrue(_is_implausible_prefill_rate(5000.0, 16384, [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(2500.0, 16384, [(2048, 300.0)]))
        # 下限边界：x=8192 恰好启用先验带，8191 只按绝对上限判
        self.assertTrue(_is_implausible_prefill_rate(5000.0, RATE_GATE_MIN_TOKENS,
                                                     [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(5000.0, 8191, [(2048, 300.0)]))
        # 小 prompt 区先验带误杀场景还原（实测 0K 档 inst=1024：速率 722、
        # 先验曲线底端 ~60）：不判
        self.assertFalse(_is_implausible_prefill_rate(722.0, 1024, [(0, 60.0)]))
        self.assertFalse(_is_implausible_prefill_rate(99000.0, 4096, [(2048, 300.0)]),
                         "低于绝对上限即过闸（软拒绝假成功本就发生在长 prompt）")
        # 无先验、低于绝对上限：不判
        self.assertFalse(_is_implausible_prefill_rate(5000.0, 16384, []))
        self.assertFalse(_is_implausible_prefill_rate(None, 16384, [(2048, 300.0)]))
        self.assertFalse(_is_implausible_prefill_rate(0, 16384, [(2048, 300.0)]))


class TestCtxEdgeTrim(unittest.TestCase):
    """部署上限超窗的「贴边裁减」判定（_plan_ctx_edge，LLM 阶梯与媒体阶梯共用）。

    口径：excess = 输入构造目标 + 输出预算 + CTX_HEADROOM − max_ctx；
    excess ≤ 0 照常测量（现状不变）、0 < excess ≤ 10%·max_ctx 裁减贴边
    （ctx_eff = max_ctx − budget − CTX_HEADROOM）、超出量超阈值或贴边量非正
    维持跳过。"""

    MAX_CTX = 81920   # 10% = 8192 整除，便于构造精确边界
    EDGE = MAX_CTX - CTX_HEADROOM   # budget=0 时的贴边输入档

    def test_under_window_unchanged(self):
        """excess ≤ 0：按原档位测，构造量不动（现状行为）。"""
        mode, eff, excess = _plan_ctx_edge(self.EDGE, 0, self.MAX_CTX)
        self.assertEqual((mode, eff, excess), ("ok", self.EDGE, 0))
        mode, eff, excess = _plan_ctx_edge(self.EDGE - 4096, 512, self.MAX_CTX)
        self.assertEqual(mode, "ok")
        self.assertEqual(eff, self.EDGE - 4096)   # 原构造量原样返回
        # 0K 档零输入 + 常规输出预算：永不在窗上
        mode, eff, _ = _plan_ctx_edge(0, 512, self.MAX_CTX)
        self.assertEqual((mode, eff), ("ok", 0))

    def test_exactly_ten_percent_trims(self):
        """0 < excess ≤ 10%·max_ctx：裁减贴边，ctx_eff 落回
        max_ctx − budget − CTX_HEADROOM。"""
        in_tok = self.EDGE + 8192   # excess 恰好 = 10%·max_ctx（含 headroom 口径）
        mode, eff, excess = _plan_ctx_edge(in_tok, 0, self.MAX_CTX)
        self.assertEqual((mode, excess), ("edge", 8192))
        self.assertEqual(eff, self.EDGE)
        # 带输出预算的贴边：ctx_eff = max_ctx − budget − CTX_HEADROOM
        in_tok = self.EDGE - 4096 + 8192   # budget=4096 时 excess 同为 8192
        mode, eff, excess = _plan_ctx_edge(in_tok, 4096, self.MAX_CTX)
        self.assertEqual((mode, excess), ("edge", 8192))
        self.assertEqual(eff, self.MAX_CTX - 4096 - CTX_HEADROOM)

    def test_just_over_ten_percent_skips(self):
        """excess = 10%·max_ctx + 1：超出量超阈值，维持整档跳过。"""
        mode, eff, excess = _plan_ctx_edge(self.EDGE + 8193, 0, self.MAX_CTX)
        self.assertEqual((mode, eff, excess), ("skip", 0, 8193))

    def test_256k_reported_case(self):
        """用户实报场景还原：256K 部署测 256K 档，输出预算 8K——
        裁减为 ≈247K 贴边测量而非整档跳过。"""
        mode, eff, excess = _plan_ctx_edge(262144, 8192, 262144)
        self.assertEqual(mode, "edge")
        self.assertEqual(eff, 262144 - 8192 - CTX_HEADROOM)
        self.assertEqual(excess, 8192 + CTX_HEADROOM)

    def test_budget_gte_max_ctx_still_skips(self):
        """budget + CTX_HEADROOM ≥ max_ctx：贴边量非正（连零输入都放不下），
        即使超窗量在 10% 内也跳过。"""
        mode, eff, _ = _plan_ctx_edge(0, self.MAX_CTX, self.MAX_CTX)
        self.assertEqual(mode, "skip")
        mode, eff, _ = _plan_ctx_edge(100, self.MAX_CTX - CTX_HEADROOM + 10,
                                      self.MAX_CTX)
        self.assertEqual(mode, "skip")   # excess 在 10% 内但贴边量为负


class TestCtxShort(unittest.TestCase):
    """「实际喂进去的长度比档位标签短」判定（`_ctx_short`，ADR-0136）：服务端
    200 不报错，却明显少算 prompt_tokens——档位标签开始说谎，必须留痕。"""

    @staticmethod
    def _req(tok, real=True):
        return {"usage_real": real, "prompt_tokens": tok}

    def test_real_archive_cases_detected(self):
        """存档里真出现过的两例（无 ovf_trim/ctx_edge 留痕、各 rep 一致）：
        2026-10-13 构造 262144 服务端只算 240536（−8.2%）、2026-09-16 的 206524
        （−21.2%）→ 都该判，留痕带构造量/回报量/差额。"""
        self.assertEqual(_ctx_short(262144, [self._req(240536)]),
                         {"ctx": 262144, "reported": 240536, "dropped": 21608})
        self.assertIsNotNone(_ctx_short(262144, [self._req(206524)]))

    def test_normal_rungs_not_flagged(self):
        """同词表的正常计数抖动（真机同轮其余档位 ±1% 内）不得误判。"""
        for ctx, got in [(16384, 16340), (32768, 32772), (65536, 64915),
                         (131072, 131168), (262144, 260619)]:
            self.assertIsNone(_ctx_short(ctx, [self._req(got)]),
                              f"{ctx} → {got} 属正常抖动，不该判")

    def test_threshold_is_five_percent_with_floor(self):
        """判据 = max(5%·ctx, 2048)：卡在阈内不算，越过才算。"""
        self.assertIsNone(_ctx_short(262144, [self._req(262144 - 13107)]))
        self.assertIsNotNone(_ctx_short(262144, [self._req(262144 - 13108)]))
        # 小档位靠绝对下限兜：4096 档少 2000（49%）仍低于 2048 下限，不判
        self.assertIsNone(_ctx_short(4096, [self._req(2096)]))

    def test_requires_server_reported_truth(self):
        """无 usage 真值（字符估算兜底）时不判：估算靶值自指，判不了。"""
        self.assertIsNone(_ctx_short(262144, [self._req(100000, real=False)]))
        # 并发批里有一条没回真值 → 整点不判（判不准不如不判）
        self.assertIsNone(_ctx_short(262144, [self._req(206524),
                                             self._req(100000, real=False)]))

    def test_only_trims_we_know_about_are_exempt(self):
        """我们自己动的刀不重复判（各有角标归口）：超窗删减 ovf_trim、贴边裁减
        ctx_edge（点级已按裁前档位显示）。而按实测密度重建过（ctx_retry）**不**
        豁免——重建后仍短，正是"服务端吃不下这么多"，最该留痕。"""
        self.assertIsNone(_ctx_short(262144, [self._req(206524)], trimmed=True))
        self.assertIsNotNone(_ctx_short(262144, [self._req(240536)]))
        # 0K/小档固定开销主导、无 ok 请求：不判
        self.assertIsNone(_ctx_short(512, [self._req(1)]))
        self.assertIsNone(_ctx_short(262144, []))


class TestDegenerateText(unittest.TestCase):
    """_is_degenerate_text：头部采样退化重复检测（实测三样本 + 正常文本
    不误伤 + 短串保护）。"""

    # 三个真实退化样本（fastllm 部署 ≥8K 档思考流，reason_chars≈512）
    S_BRACE = "}\n" * 60                       # ① "}" 换行循环：2 种字符
    S_NUM = "        " + " ".join(f"{i}." for i in range(6, 60))  # ② 编号列表
    S_NEST = "".join(" " * i + "if (\n" for i in range(20))       # ③ 嵌套递增

    def test_real_samples_true(self):
        for s in (self.S_BRACE, self.S_NUM, self.S_NEST):
            self.assertTrue(_is_degenerate_text(s),
                            f"实测退化样本应命中: {s[:40]!r}")

    def test_normal_chinese_false(self):
        s = ("大语言模型的推理速度受显存带宽与批调度策略共同影响，长上下文档位的"
             "预填充读数尤其依赖服务端的 KV 缓存实现与逐出策略，短文档位则更多"
             "受网络往返与调度排队左右，逐请求对比才能看出真实差异。")
        self.assertGreater(len(s.strip()), 80)
        self.assertFalse(_is_degenerate_text(s))

    def test_normal_code_false(self):
        s = ('def _run_point(self, client, model, scenario, ctx, conc, cpt, rep):\n'
             '    tasks = [asyncio.create_task(self._one(client, model, m)) for m in msgs]\n'
             '    reqs, batch_time = await self._await_batch(tasks)\n')
        self.assertGreater(len(s.strip()), 80)
        self.assertFalse(_is_degenerate_text(s))

    def test_short_false(self):
        self.assertFalse(_is_degenerate_text("}\n" * 20))   # 40 字符 < 80 下限
        self.assertFalse(_is_degenerate_text("好的。"))


class TestDetectRepAnomaly(unittest.TestCase):
    """_detect_rep_anomaly：early_stop 阈值 = max(比例, 有效地板)——echo
    0.8×max_tokens、free 不按比例只按地板（自主收尾合法，但输出几个字
    就停完全不可信，2026-09-16 拍板）、finish=length 不判、全 err 不判、
    退化重复两模式通用且 text/reason 通道任一命中。"""

    def _req(self, **over):
        r = {"err": None, "finish": "stop", "out_tokens": 512,
             "ttft_s": 0.1, "decode_tok_s": 50.0,
             "text_sample": "正文样本", "reason_sample": ""}
        r.update(over)
        return r

    def test_echo_early_stop(self):
        reqs = [self._req(), self._req(out_tokens=4)]   # 肇事在槽位 1
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=True),
                         ("early_stop", 1))

    def test_free_below_floor_early_stop(self):
        # free 模式不按比例罚，但 4 < 有效地板 100：完全不可信同判
        reqs = [self._req(out_tokens=4)]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=False),
                         ("early_stop", 0))

    def test_free_above_floor_no_early_stop(self):
        # free 模式 150 ≥ 地板 100（虽 <0.8×512）：自主收尾合法，不判
        reqs = [self._req(out_tokens=150)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=False))

    def test_echo_above_floor_below_ratio_still_flagged(self):
        # echo 模式 150 ≥ 地板但 <0.8×512=409.6：续写口径仍判
        reqs = [self._req(out_tokens=150)]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=True),
                         ("early_stop", 0))

    def test_echo_within_threshold_none(self):
        # 480/512 超过 0.8×512=409.6：预算内的正常收尾，不判
        reqs = [self._req(out_tokens=480)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_floor_capped_by_budget(self):
        # 小预算地板随预算钳制：max_tokens=32 时地板=32，满预算输出不判
        reqs = [self._req(out_tokens=32)]
        self.assertIsNone(_detect_rep_anomaly(reqs, 32, echo_mode=False))
        reqs = [self._req(out_tokens=31)]
        self.assertEqual(_detect_rep_anomaly(reqs, 32, echo_mode=False),
                         ("early_stop", 0))

    def test_length_finish_none(self):
        reqs = [self._req(out_tokens=4, finish="length")]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_all_err_none(self):
        reqs = [self._req(out_tokens=4, err="http 500")]
        self.assertIsNone(_detect_rep_anomaly(reqs, 512, echo_mode=True))

    def test_degenerate_text_sample(self):
        reqs = [self._req(text_sample="}\n" * 60)]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=False),
                         ("degenerate", 0))

    def test_degenerate_reason_sample(self):
        # 思考流退化走 reasoning 通道（fastllm 实测），text 通道正常也命中
        reqs = [self._req(reason_sample=" ".join(f"{i}." for i in range(6, 60)))]
        self.assertEqual(_detect_rep_anomaly(reqs, 512, echo_mode=True),
                         ("degenerate", 0))


class TestPrefillGateCallSite(unittest.IsolatedAsyncioTestCase):
    """物理速率闸调用点（_one 收口）：x 取未命中 prompt tokens（缓存回传时
    = prompt − cache_hit，否则全量）、未命中量 <8K 只按绝对上限判——走真实
    _one 请求链路（假 client 定时流控制 TTFT→rate），先验曲线预置紧带。"""

    MODEL = "m"
    PRIOR = [(2048, 300.0)]   # 10× 带 = 3000 tok/s

    class _Resp:
        def __init__(self, lines):
            self.status_code = 200
            self.headers = {}
            self._lines = lines   # async 迭代器（定时流控制 TTFT）

        async def aiter_lines(self):
            async for ln in self._lines:
                yield ln

        async def aread(self):
            return b""

    class _Ctx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, resp):
            self.resp = resp

        def stream(self, *a, **k):
            return TestPrefillGateCallSite._Ctx(self.resp)

    @staticmethod
    def _sse(prompt_tokens, completion_tokens, delay_s, cache_hit=0):
        """定时 SSE：首块延迟 delay_s 秒（TTFT≈delay → rate≈prompt/delay）。
        usage 回传 cache_hit（缓存命中口径用例用）。"""
        async def lines():
            await asyncio.sleep(delay_s)
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "测"}}]})
            yield "data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt_tokens,
                           "completion_tokens": completion_tokens,
                           "prompt_cache_hit_tokens": cache_hit}})
            yield "data: [DONE]"
        return lines()

    async def _one_ok(self, usage, delay):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u" * 40}]
        run = BenchRun({})
        run.prefill_curve[(self.MODEL, "agent")] = list(self.PRIOR)
        events = []

        async def fake_emit(ev):
            events.append(ev)

        run.emit = fake_emit
        r = await run._one(self._Client(self._Resp(self._sse(
            usage["prompt_tokens"], usage["completion_tokens"], delay,
            usage.get("prompt_cache_hit_tokens", 0)))),
            self.MODEL, msgs, "agent", 0, 0, 1, 1.8, 512)
        return r, events

    async def test_small_prompt_prior_band_not_applied(self):
        """未命中量 1024 <8K：速率 51200 > 先验带 3000 也过闸（实测 0K 档
        inst=1024 误杀场景）——点干净、无「物理上不可能」播报。"""
        usage = {"prompt_tokens": 1024, "completion_tokens": 3}
        r, events = await self._one_ok(usage, 0.02)   # rate ≈ 51200 tok/s
        self.assertIsNone(r["err"])
        self.assertFalse(any("物理上不可能" in (e.get("msg") or "")
                             for e in events))

    async def test_in_sample_archived(self):
        """输入侧头部采样入档（ADR-0053）：req 级 in_sample = 末条消息文本
        头部 ≤200 字符（agent 矩阵末条即指令、echo 末条为预填）——异常日志
        复盘输入侧；超窗删减重试后反映实际发送内容。agent 为双向引导场景，
        短消息的 200 字符采样窗内可见指令末尾复述约束（ADR-0076）"""
        usage = {"prompt_tokens": 1024, "completion_tokens": 3}
        r, _ = await self._one_ok(usage, 0.02)
        self.assertTrue(
            r.get("in_sample", "").startswith("u" * 40 + "\n\nLength requirement"),
            f"in_sample = 末条消息头部（user 'u'*40 + 指令末尾复述约束）: "
            f"{r.get('in_sample')!r}")

    async def test_large_prompt_prior_band_applies(self):
        """未命中量 16384 ≥8K：速率 8192 > 先验带 3000 → 判「物理上不可能」
        按错误收口（绝对上限 10 万之下，证明走的是先验带分支）。ADR-0052：
        确定性失败除 error tick 外随发 status toast:true 悬浮岛提醒。"""
        usage = {"prompt_tokens": 16384, "completion_tokens": 3}
        r, events = await self._one_ok(usage, 2.0)   # rate ≈ 8192 tok/s
        self.assertIn("物理上不可能", r["err"] or "")
        self.assertIn("先验曲线", r["err"])
        toasts = [e for e in events
                  if e.get("type") == "status" and e.get("toast")]
        self.assertTrue(any("物理上不可能" in (e.get("msg") or "")
                            for e in toasts),
                        "闸门收口应随发 status toast:true 失败提醒（ADR-0052）")

    async def test_cache_hit_x_is_uncached_amount(self):
        """缓存回传 hit>0：x = prompt_tokens − cache_hit（与 gate_rate 未命中
        口径同源）——未命中量 <8K 时先验带不启用，全量 16384 超带不误杀。"""
        usage = {"prompt_tokens": 16384, "completion_tokens": 3,
                 "prompt_cache_hit_tokens": 12000}   # 未命中 4384 <8192
        r, events = await self._one_ok(usage, 0.2)   # rate ≈ 81920（低于绝对上限）
        self.assertIsNone(r["err"])
        self.assertFalse(any("物理上不可能" in (e.get("msg") or "")
                             for e in events))


class TestTransientRetry(unittest.IsolatedAsyncioTestCase):
    """瞬时失败兜底重试：可重试/不可重试分类（传输异常/429/5xx/空流 vs 其他
    4xx 与数据有效性判定）、重试后成功（req 干净且带 retried 留痕）、重试耗尽
    （返回最后一次 err、既有 error tick 不变）、stop_flag 置位立即中止、quiet
    探测/预热静默重试。走真实 _one/_one_media 请求链路（最小假 client 脚本化
    失败注入），退避时长打桩为零避免拖慢测试。"""

    MODEL = "m"
    USAGE = {"prompt_tokens": 1, "completion_tokens": 3}   # 极小 prompt 不触塌缩守卫/速率闸

    # -- 最小假件 -----------------------------------------------------------
    class _Resp:
        """_one/_one_media 所需的最小 httpx.Response 形状。"""
        def __init__(self, status=200, lines=(), headers=None, body=""):
            self.status_code = status
            self._lines = list(lines)
            self.headers = headers or {}
            self._body = body

        @property
        def text(self):
            return self._body

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def aread(self):
            return self._body.encode("utf-8")

        def json(self):
            return json.loads(self._body)

    class _StreamCtx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _ChatClient:
        """chat 流式假客户端：按脚本逐次返回（异常直接抛 = 传输抖动；_Resp
        正常走流；脚本耗尽后重复最后一项）。"""
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        def stream(self, *a, **k):
            i = min(self.calls, len(self.script) - 1)
            self.calls += 1
            item = self.script[i]
            if isinstance(item, Exception):
                raise item
            return TestTransientRetry._StreamCtx(item)

    class _AsrClient:
        """asr 非流式假客户端（脚本化同 _ChatClient）。"""
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        async def post(self, path, **k):
            i = min(self.calls, len(self.script) - 1)
            self.calls += 1
            item = self.script[i]
            if isinstance(item, Exception):
                raise item
            return item

    # -- 脚手架 -------------------------------------------------------------
    MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    @classmethod
    def _sse(cls, chunks=3, usage=None):
        """最小 SSE 行流：chunks 个正文 token + 可选 usage 帧 + [DONE]。"""
        lines = ["data: " + json.dumps(
            {"choices": [{"delta": {"content": "测"}}]}) for _ in range(chunks)]
        if usage:
            lines.append("data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": usage}))
        lines.append("data: [DONE]")
        return lines

    def _make_run(self):
        run = BenchRun({})
        events = []

        async def fake_emit(ev):
            events.append(ev)

        run.emit = fake_emit
        return run, events

    def _statuses(self, events):
        return [e["msg"] for e in events if e.get("type") == "status"]

    def _error_ticks(self, events):
        return [(e["msg"], e.get("req")) for e in events
                if e.get("type") == "tick" and e.get("phase") == "error"]

    async def _call_one(self, run, client, **over):
        with unittest.mock.patch.object(bench, "TRANSIENT_RETRY_BASE_S", 0.01):
            return await run._one(client, self.MODEL, self.MSGS, "creative",
                                  0, 0, 1, 1.8, 128, **over)

    # -- 纯函数分类 ---------------------------------------------------------
    def test_transient_status_classification(self):
        self.assertTrue(_is_transient_http_status(429))
        for s in (500, 502, 503, 504, 599):
            self.assertTrue(_is_transient_http_status(s), s)
        for s in (400, 401, 403, 404, 422):
            self.assertFalse(_is_transient_http_status(s), s)

    def test_transient_exc_classification(self):
        self.assertTrue(_is_transient_exc(httpx.ConnectError("连接被重置")))
        self.assertTrue(_is_transient_exc(httpx.ReadTimeout("read timeout")))
        self.assertTrue(_is_transient_exc(httpx.RemoteProtocolError("裸断连")))
        # 端点探测耗尽/超窗收口等非传输异常不重试
        self.assertFalse(_is_transient_exc(RuntimeError("无可用端点")))
        self.assertFalse(_is_transient_exc(ValueError("x")))

    def test_retry_after_parsing(self):
        self.assertIsNone(_retry_after_s({}))
        self.assertIsNone(_retry_after_s({"retry-after": "abc"}))
        self.assertIsNone(_retry_after_s({"retry-after": "-3"}))
        self.assertEqual(_retry_after_s({"retry-after": "2"}), 2.0)
        self.assertEqual(_retry_after_s({"retry-after": " 0.5 "}), 0.5)
        self.assertEqual(_retry_after_s({"retry-after": "120"}),
                         bench.RETRY_AFTER_CAP_S, "超大值封顶")

    # -- 请求链路 -----------------------------------------------------------
    async def test_transport_error_retried_then_success(self):
        """传输异常（连接重置）兜底重试一次后成功：req 干净、带 retried=1、
        正常读数字段照常产出、status 播报一次兜底重试、无 error tick。"""
        run, events = self._make_run()
        client = self._ChatClient([
            httpx.ConnectError("连接被重置"),
            self._Resp(lines=self._sse(usage=self.USAGE)),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 2, "传输异常应整体重试一次")
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1, "重试后成功应留 retried 痕")
        self.assertEqual(result["out_tokens"], 3, "usage 口径照常")
        self.assertIsNotNone(result["first_abs"])
        statuses = self._statuses(events)
        self.assertEqual(len([s for s in statuses if "兜底重试 1/2" in s]), 1,
                         f"应恰好一条兜底重试播报: {statuses}")
        self.assertIn(self.MODEL, statuses[0])   # 播报含模型/上下文/请求序号
        self.assertIn("r0", statuses[0])
        self.assertEqual(self._error_ticks(events), [], "成功路径不得有 error tick")

    async def test_http_500_retried_with_retry_after(self):
        """HTTP 503 带 Retry-After 头：遵守该头退避（封顶）后重试成功。"""
        run, events = self._make_run()
        client = self._ChatClient([
            self._Resp(status=503, headers={"retry-after": "0.02"},
                       body="upstream unavailable"),
            self._Resp(lines=self._sse(usage=self.USAGE)),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 2)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1)
        statuses = self._statuses(events)
        self.assertTrue(any("0.02s 后兜底重试 1/2" in s for s in statuses),
                        f"Retry-After 头应被遵守: {statuses}")

    async def test_http_4xx_not_retried(self):
        """其他 4xx（鉴权/参数等确定性错误）：不重试，直接按既有错误口径收口。"""
        run, events = self._make_run()
        client = self._ChatClient([
            self._Resp(status=401, body="invalid api key"),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1, "4xx 不得重试")
        self.assertEqual(result["err"], "HTTP 401: invalid api key")
        self.assertNotIn("retried", result)
        self.assertEqual(self._error_ticks(events),
                         [("HTTP 401: invalid api key", 0)],
                         "既有 error tick 口径不变")

    async def test_retry_exhausted_returns_last_err(self):
        """重试耗尽（共 3 次尝试）：返回最后一次 err、retried 键缺席、
        播报 2 次兜底重试、最终既有 error tick 恰一条。"""
        run, events = self._make_run()
        client = self._ChatClient([httpx.ReadTimeout("read timed out")])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1 + TRANSIENT_RETRY_MAX, "共 3 次尝试")
        self.assertEqual(result["err"], "read timed out")
        self.assertNotIn("retried", result)
        statuses = self._statuses(events)
        self.assertTrue(any("兜底重试 1/2" in s for s in statuses), statuses)
        self.assertTrue(any("兜底重试 2/2" in s for s in statuses), statuses)
        self.assertEqual(self._error_ticks(events),
                         [("read timed out", 0)])

    async def test_empty_stream_retried(self):
        """空流（200 但无任何输出 token）：走专属重试预算
        （EMPTY_STREAM_RETRY_MAX——Lvllm「空 EOS」间歇故障，同消息重试
        可能自愈），两次空流后第三次成功。"""
        run, events = self._make_run()
        client = self._ChatClient([
            self._Resp(lines=["data: [DONE]"]),
            self._Resp(lines=["data: [DONE]"]),
            self._Resp(lines=self._sse(usage=self.USAGE)),
        ])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 3)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 2)
        statuses = self._statuses(events)
        self.assertTrue(any(f"兜底重试 2/{EMPTY_STREAM_RETRY_MAX}" in s
                            for s in statuses), statuses)

    async def test_empty_stream_exhausted(self):
        """空流耗尽专属预算（1+EMPTY_STREAM_RETRY_MAX 次尝试）：返回既有的
        「未收到任何输出」错误口径；专属预算独立于通用瞬时重试
        （TRANSIENT_RETRY_MAX）。"""
        run, events = self._make_run()
        client = self._ChatClient([self._Resp(lines=["data: [DONE]"])])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1 + EMPTY_STREAM_RETRY_MAX)
        self.assertIn("未收到任何输出", result["err"])
        self.assertNotIn("retried", result)

    async def test_stop_flag_aborts_retry(self):
        """stop_flag 已置位：不再重试，立即按既有错误口径收口（无兜底播报）。"""
        run, events = self._make_run()

        class _StopClient:
            def __init__(self, run):
                self.run = run
                self.calls = 0

            def stream(self, *a, **k):
                self.calls += 1
                self.run.stop_flag = True   # 模拟请求在途期间用户点停止
                raise httpx.ConnectError("连接被重置")

        client = _StopClient(run)
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1, "停止后不得再重试")
        self.assertEqual(result["err"], "连接被重置")
        self.assertNotIn("retried", result)
        self.assertEqual(self._statuses(events), [], "停止路径不播报兜底重试")

    async def test_quiet_probe_retries_silently(self):
        """quiet=True（探测/预热请求）：同样享受兜底重试但全程静默——不发
        status 播报也不发 tick（agent 矩阵预热依赖重试，且不刷日志）。
        usage 量取不触发系数校准播报的形态（obs/cobs 均越界不采纳），隔离
        重试静默性主题与既有校准播报行为。"""
        run, events = self._make_run()
        client = self._ChatClient([
            httpx.ConnectError("连接被重置"),
            self._Resp(lines=self._sse(
                usage={"prompt_tokens": 1, "completion_tokens": 100})),
        ])
        result = await self._call_one(run, client, quiet=True)
        self.assertEqual(client.calls, 2)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1)
        self.assertEqual([e for e in events if e.get("type") in ("status", "tick")],
                         [], f"quiet 重试必须静默: {events}")

    async def test_clean_success_no_retried_key(self):
        """一次成功（无重试）：结果不带 retried 键，成功路径逐字节不变。"""
        run, events = self._make_run()
        client = self._ChatClient([self._Resp(lines=self._sse(usage=self.USAGE))])
        result = await self._call_one(run, client)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(result["err"])
        self.assertNotIn("retried", result)
        self.assertEqual(result["out_tokens"], 3)

    async def test_media_asr_retried_then_success(self):
        """_one_media（asr）传输异常兜底重试：重试后成功带 retried 留痕。"""
        run, events = self._make_run()
        client = self._AsrClient([
            httpx.ConnectError("连接被重置"),
            self._Resp(body='{"text": "模拟识别结果。"}'),
        ])
        with unittest.mock.patch.object(bench, "TRANSIENT_RETRY_BASE_S", 0.01):
            result = await run._one_media(client, self.MODEL, "asr", "asr",
                                          5, 0, 1, 0)
        self.assertEqual(client.calls, 2)
        self.assertIsNone(result["err"])
        self.assertEqual(result["retried"], 1)
        self.assertGreater(result["out_chars"], 0)
        statuses = self._statuses(events)
        self.assertTrue(any("兜底重试 1/2" in s for s in statuses), statuses)

    async def test_media_asr_500_not_retried_beyond_budget(self):
        """_one_media HTTP 500：重试耗尽后按既有 _fail 口径收口（error tick
        + err 返回），不无限重试。"""
        run, events = self._make_run()
        client = self._AsrClient([self._Resp(status=500, body="boom")])
        with unittest.mock.patch.object(bench, "TRANSIENT_RETRY_BASE_S", 0.01):
            result = await run._one_media(client, self.MODEL, "asr", "asr",
                                          5, 0, 1, 0)
        self.assertEqual(client.calls, 1 + TRANSIENT_RETRY_MAX)
        self.assertEqual(result["err"], "HTTP 500: boom")
        self.assertNotIn("retried", result)
        self.assertEqual(self._error_ticks(events),
                         [("HTTP 500: boom", 0)])


class TestPrefillContinueParams(unittest.IsolatedAsyncioTestCase):
    """echo 真·预填续写参数（ADR-0079）：末条为 assistant 预填时 payload 带
    add_generation_prompt=false + continue_final_message=true（模板不闭合
    轮次、从预填末尾直接续写——闭合形态是 32K+ 确定性早停的根因）；末条
    非 assistant 不下发；网关 400 点名参数时成对省略、按模型锁定、同端点
    重试（回退闭合轮次形态），锁定后不再下发。"""

    MODEL = "m"
    ECHO_MSGS = [{"role": "system", "content": "s"},
                 {"role": "user", "content": "u" * 40},
                 {"role": "assistant", "content": "预填正文"}]

    class _Resp:
        def __init__(self, status=200, lines=(), body=""):
            self.status_code = status
            self._lines = list(lines)
            self.headers = {}
            self._body = body

        @property
        def text(self):
            return self._body

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def aread(self):
            return self._body.encode("utf-8")

    class _Ctx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *exc):
            return False

    class _CapClient:
        """脚本化假客户端：逐次返回脚本项并记录每次请求的 payload。"""
        def __init__(self, script):
            self.script = list(script)
            self.payloads = []

        def stream(self, *a, **k):
            i = min(len(self.payloads), len(self.script) - 1)
            self.payloads.append(k["json"])
            return TestPrefillContinueParams._Ctx(self.script[i])

    @staticmethod
    def _ok_resp():
        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "测"}}]}),
                 "data: " + json.dumps(
                     {"choices": [{"delta": {}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 3}}),
                 "data: [DONE]"]
        return TestPrefillContinueParams._Resp(200, lines)

    def _make_run(self):
        run = BenchRun({})
        events = []

        async def fake_emit(ev):
            events.append(ev)

        run.emit = fake_emit
        return run, events

    async def _call(self, run, client, msgs):
        return await run._one(client, self.MODEL, msgs, "creative",
                              0, 0, 1, 1.8, 128)

    async def test_echo_prefill_sends_continue_params(self):
        """末条 assistant 预填 → payload 带续写参数对。"""
        run, _ = self._make_run()
        client = self._CapClient([self._ok_resp()])
        r = await self._call(run, client, self.ECHO_MSGS)
        self.assertIsNone(r["err"])
        p = client.payloads[0]
        self.assertIs(p.get("add_generation_prompt"), False)
        self.assertIs(p.get("continue_final_message"), True)

    async def test_non_assistant_last_omits_params(self):
        """末条为 user（常规/agent/媒体形态）→ 不下发续写参数。"""
        run, _ = self._make_run()
        client = self._CapClient([self._ok_resp()])
        r = await self._call(run, client, [{"role": "system", "content": "s"},
                                           {"role": "user", "content": "u"}])
        self.assertIsNone(r["err"])
        self.assertNotIn("add_generation_prompt", client.payloads[0])
        self.assertNotIn("continue_final_message", client.payloads[0])

    async def test_400_locks_and_retries_without_params(self):
        """网关 400 点名 add_generation_prompt → 成对省略同端点重试成功、
        模型入 prefill_locked、播报；锁定后的后续请求不再下发。"""
        run, events = self._make_run()
        bad = self._Resp(400, body='{"error":{"message":"unknown parameter: '
                                   'add_generation_prompt"}}')
        client = self._CapClient([bad, self._ok_resp(), self._ok_resp()])
        r = await self._call(run, client, self.ECHO_MSGS)
        self.assertIsNone(r["err"])
        self.assertEqual(len(client.payloads), 2, "400 后应同端点重试")
        self.assertNotIn("add_generation_prompt", client.payloads[1])
        self.assertNotIn("continue_final_message", client.payloads[1])
        self.assertIn(self.MODEL, run.prefill_locked)
        statuses = [e["msg"] for e in events if e.get("type") == "status"]
        self.assertTrue(any("add_generation_prompt" in s for s in statuses),
                        f"应有省略播报: {statuses}")
        # 锁定后后续请求不再下发
        r2 = await self._call(run, client, self.ECHO_MSGS)
        self.assertIsNone(r2["err"])
        self.assertNotIn("add_generation_prompt", client.payloads[2])


class TestDeliveryEstimateNotUsedAsGate(unittest.TestCase):
    """ADR-0145：交付粒度基线只喂显示，不喂判定。

    断流闸（`_est_out`）过去默认走的就是那份「包数 ÷ 事件/token 比」的估算分子。
    把跨运行基线接进来后，若让它一并参与闸口：基线取大了 → 估算 token 数虚高 →
    真卡死的流被算成"还在出东西"，断流漏检。所以闸口按来源分开取值。"""

    def _run(self, ratio):
        run = BenchRun({})
        run.chunk_calib[("A", "code")] = ratio
        run.chunk_calib_w[("A", "code")] = 100.0
        return run

    def test_gate_counts_events_conservatively_when_ratio_is_a_seed(self):
        run = self._run(1 / 4.9)                       # 基线播的：每包 4.9 个 token
        run._chunk_calib_src[("A", "code")] = "ledger"
        # 未实测过的比不参与：按 1 事件/token 直计，卡死时估算涨不动，闸照样触发
        self.assertEqual(run._est_out_cut("A", "code", 10, 0, 1.0, "ledger"), 10)
        self.assertAlmostEqual(run._est_out("A", "code", 10, 0, 1.0), 49.0, places=6,
                               msg="显示侧不得被保守口径拖累")
        run.chunk_calib[("A", "code")] = 1 / 2.0       # 本轮实测后照常生效
        run._chunk_calib_src[("A", "code")] = "run"
        self.assertEqual(run._est_out("A", "code", 10, 0, 1.0), 20)
        self.assertEqual(run._est_out_cut("A", "code", 10, 0, 1.0, "run"), 20)

    def test_display_estimate_still_uses_the_seed(self):
        """显示侧照旧用基线（否则冷启动那一格又变成按包计数，偏差回到 4 倍量级）。"""
        run = self._run(1 / 4.9)
        run._chunk_calib_src[("A", "code")] = "ledger"
        # 显示口径：10 包 × 每包 4.9 个 token
        self.assertAlmostEqual(run._est_out("A", "code", 10, 0, 1.0), 49, places=6)
