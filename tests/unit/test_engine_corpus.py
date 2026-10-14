"""unit 组 · 语料流与消息构造：确定性语料流（嵌套前缀 / 模块块级 / 换料偏移）、
echo 改写区对齐、场景先验与消息构造（含 agent 缓存×指令矩阵）、输出长度治理
的参数面、媒体语料合成与 OCR/TTS 构造、翻译场景构造与预算。

覆盖 ADR-0005 / 0006 / 0016 / 0020 / 0021 / 0034 / 0042 的构造侧口径；
自 tests/test_engine_unit.py 拆分（原文件已删）。

运行（两种入口，均须在仓库根目录执行）：
    .venv/bin/python -m pytest tests/unit/test_engine_corpus.py -q
    .venv/bin/python -m unittest tests.unit.test_engine_corpus -v
    # 按组 discover 的等价形式：
    .venv/bin/python -m unittest discover -s tests/unit -p test_engine_corpus.py
"""
import io
import os
import struct
import tempfile
import unittest
import unittest.mock
import wave

import bench
from bench import (
    CLIENT_CUT_FACTOR,
    OUT_HINT_FACTOR,
    SCENARIOS,
    _align_echo_region,
    _asr_audio,
    _build_ocr_messages,
    _build_translate_messages,
    _fill,
    _load_code_pool,
    _make_module_stream,
    _make_stream,
    _ocr_images,
    _synth_document_png,
    _synth_speech_wav,
    _translate_max_tokens,
    _tts_text,
    _wav_seconds,
    build_agent_messages,
    build_messages,
)


class TestFillStream(unittest.TestCase):
    """ADR-0005 确定性嵌套语料流的关键性质。"""

    POOL = ["甲" * 100, "乙" * 200, "丙" * 300]

    def setUp(self):
        self.stream, self.bounds = _make_stream(self.POOL)

    def test_deterministic(self):
        self.assertEqual((self.stream, self.bounds), _make_stream(self.POOL))

    def test_boundaries_block_starts(self):
        """边界表 = 各语料块在流中的起始偏移（含流首 0，分隔符计入累积）：
        流首为 0；每个非零边界紧随 "\n\n" 之后，且从该处开始截到流尾的内容
        恰为末尾完整块组——echo 改写区对齐的依据。"""
        self.assertEqual(self.bounds[0], 0)
        self.assertLess(self.bounds[1], self.bounds[2])
        for b in self.bounds[1:]:
            self.assertEqual(self.stream[b - 2:b], "\n\n")
        # 末块边界到流尾 = 完整块内容（无尾随分隔符计入）
        self.assertIn(len(self.stream) - self.bounds[-1],
                      {len(p) for p in self.POOL})
        # 任意块边界截断的尾部都是若干完整块（以块内容开头，非半行）
        for b in self.bounds:
            self.assertTrue(self.stream[b:].startswith(("甲", "乙", "丙")))

    def test_nested_prefix(self):
        """小档位填充必须是大档位的精确前缀（嵌套前缀精确记账的前提）。"""
        short = _fill(self.stream, 250)
        long_ = _fill(self.stream, 500)
        self.assertTrue(long_.startswith(short))

    def test_wrap_continuity(self):
        """流不够长时整流循环续接，内容仍确定且衔接无缝。"""
        total = len(self.stream)
        wrapped = _fill(self.stream, total * 2 + 123)
        self.assertEqual(len(wrapped), total * 2 + 123)
        self.assertEqual(wrapped[:total], self.stream)
        self.assertEqual(wrapped[total: total * 2], self.stream)   # 第二遍整段接续
        self.assertEqual(wrapped[total * 2:], self.stream[:123])

    def test_empty_and_nonpositive(self):
        self.assertEqual(_fill(self.stream, 0), "")
        self.assertEqual(_fill("", 10), "")

    def test_offset_window(self):
        """换料偏移（弃测重测翻落点，2026-09-17 拍板）：offset>0 时截取
        窗口循环平移——长度不变、内容不同；offset=0 与原口径逐字节一致；
        offset 超过流长时取模回绕。"""
        total = len(self.stream)
        base = _fill(self.stream, 300)
        self.assertEqual(base, _fill(self.stream, 300, offset=0),
                         "offset=0 必须保持原嵌套前缀口径")
        shifted = _fill(self.stream, 300, offset=150)
        self.assertEqual(len(shifted), 300)
        self.assertEqual(shifted, self.stream[150:450])
        self.assertNotEqual(shifted, base)
        # 回绕：offset 取模；跨流尾时循环续接
        wrapped = _fill(self.stream, 300, offset=total - 100)
        self.assertEqual(wrapped, self.stream[total - 100:] + self.stream[:200])
        self.assertEqual(_fill(self.stream, 50, offset=total + 10),
                         _fill(self.stream, 50, offset=10))


class TestModuleStream(unittest.TestCase):
    """ADR-0005 模块块级语料流：块间乱序（确定性）、块内保序（上下文关联性）。"""

    BLOCKS = [["甲" * 100, "乙" * 200], ["丙" * 300], ["丁" * 150, "戊" * 250]]

    def test_deterministic(self):
        self.assertEqual(_make_module_stream(self.BLOCKS),
                         _make_module_stream(self.BLOCKS))

    def test_blocks_contiguous_and_ordered(self):
        """块内文件必须相邻且保持给定顺序（模块关联性）；块次序允许乱序。"""
        stream, _ = _make_module_stream(self.BLOCKS)
        items = stream.split("\n\n")
        self.assertEqual(len(items), sum(len(b) for b in self.BLOCKS))
        pos = {s: i for i, s in enumerate(items)}
        for block in self.BLOCKS:
            idxs = [pos[s] for s in block]
            self.assertEqual(idxs, sorted(idxs))                     # 块内保序
            self.assertEqual(idxs, list(range(idxs[0], idxs[0] + len(idxs))))   # 块内相邻

    def test_flat_stream_properties_carry_over(self):
        """嵌套前缀前提不变：同一流截取仍互为前缀（校准精确记账不受影响）。"""
        stream, _ = _make_module_stream(self.BLOCKS)
        self.assertTrue(_fill(stream, 500).startswith(_fill(stream, 200)))

    def test_boundaries_file_granularity(self):
        """边界表粒度 = 文件（模块块内每个文件一个起始偏移），偏移落在块
        内容开头（紧随分隔符）——echo 改写区可对齐到单文件粒度。"""
        stream, bounds = _make_module_stream(self.BLOCKS)
        self.assertEqual(bounds[0], 0)
        self.assertEqual(len(bounds), sum(len(b) for b in self.BLOCKS))
        for b in bounds[1:]:
            self.assertEqual(stream[b - 2:b], "\n\n")
        # 边界单调递增且落在流内
        self.assertEqual(bounds, sorted(bounds))
        self.assertTrue(all(0 <= b < len(stream) for b in bounds))

    def test_load_code_pool_groups_by_directory(self):
        """真实语料：同目录文件聚为一块，块内文件名排序，过短文件跳过。"""
        blocks = _load_code_pool()
        self.assertTrue(blocks)
        for block in blocks:
            self.assertTrue(all(len(t) >= 2000 for t in block))

    def test_load_code_pool_fallback(self):
        """corpus 目录缺失时回退内置合成模块池（每条自成一个模块块）。"""
        with unittest.mock.patch.object(bench, "CORPUS_CODE_DIR", "/nonexistent-dir"):
            self.assertEqual(_load_code_pool(), [[s] for s in bench._CODE_POOL])


class TestEchoRegionAlign(unittest.TestCase):
    """echo 改写区起点对齐（_align_echo_region）：三层回退——块边界 →
    段/函数边界（"\\n\\n" 之后）→ 现状任意字符偏移。"""

    def test_hit_block_boundary_tail_full_blocks(self):
        """命中块边界：取满足 [0.8r, 1.6r] 的最后一个边界——改写区 = 材料
        末尾的完整块组，预填从块首开始。"""
        # 块 [0,100) / [100,350) / [350,600)，流长 600；region=300
        # → 目标起点 300；边界尾长：0→600（越上界 480）、100→500（越）、
        # 350→250 ∈ [240, 480] ✓ → 取 350（改写区 = 末尾完整块）
        filler = "A" * 100 + "\n\n" + "B" * 250 + "\n\n" + "C" * 250
        bounds = [0, 102, 354]
        self.assertEqual(_align_echo_region(filler, 300, bounds), 354)
        self.assertEqual(filler[354:354 + 160], "C" * 160, "预填从块首开始")

    def test_block_too_large_falls_to_paragraph(self):
        """块过大（无边界落在 [0.8r, 1.6r]）：落到目标起点后第一个 "\n\n"
        之后的段/函数边界——连续性完整。"""
        filler = "甲" * 500 + "\n\n" + "乙" * 280 + "\n\n" + "丙" * 100
        bounds = [0]   # 唯一边界尾长 = 全长，超 1.6r
        region = 400
        target = len(filler) - region
        idx = filler.find("\n\n", target, target + 160)
        self.assertEqual(_align_echo_region(filler, region, bounds), idx + 2)
        start = _align_echo_region(filler, region, bounds)
        self.assertGreater(start, target, "起点后移到段边界，非任意字符偏移")
        self.assertEqual(filler[start - 2:start], "\n\n", "起点紧随分隔符")

    def test_no_suitable_boundary_keeps_current_behavior(self):
        """无块边界命中且窗口内无 "\n\n"：回退现状（目标起点 = 任意字符
        偏移），行边界收口兜底。"""
        filler = "".join(f"行{i}\n" for i in range(500))   # 无 "\n\n"
        region = 300
        self.assertEqual(_align_echo_region(filler, region, [0]),
                         len(filler) - region)

    def test_filler_shorter_than_region(self):
        """len(filler) < region：region 收口为全长，起点 0（流首 0 是天然
        块边界），预填即材料开头。"""
        filler = "X" * 100
        self.assertEqual(_align_echo_region(filler, 300, [0]), 0)

    def test_empty_boundaries_falls_through(self):
        """边界表为空：直接走段边界/现状层，不炸。"""
        filler = "前" * 450 + "\n\n" + "后" * 100
        region = 300
        target = len(filler) - region
        idx = filler.find("\n\n", target, target + 120)
        self.assertEqual(_align_echo_region(filler, region, []),
                         idx + 2 if idx >= 0 else target)

    def test_boundaries_beyond_filler_ignored(self):
        """边界表超出 len(filler) 的部分不参与（filler 是流的前缀截取，
        循环续接流的后续边界不可见）。"""
        filler = "A" * 100 + "\n\n" + "B" * 250 + "\n\n" + "C" * 250
        bounds = [0, 102, 354, 700]   # 700 > len(filler)
        self.assertEqual(_align_echo_region(filler, 300, bounds), 354)

    def test_deterministic(self):
        """同输入同输出：嵌套前缀 / 多次运行的选择一致。"""
        filler = "A" * 100 + "\n\n" + "B" * 250 + "\n\n" + "C" * 250
        bounds = [0, 102, 354]
        r1 = _align_echo_region(filler, 300, bounds)
        r2 = _align_echo_region(filler, 300, bounds)
        self.assertEqual(r1, r2)
        # 前缀关系：短 filler 的选择只依赖自身长度，不随流整体变化
        self.assertEqual(_align_echo_region(filler[:500], 300, bounds),
                         _align_echo_region(filler[:500], 300, bounds[:2]))


class TestScenarioPriors(unittest.TestCase):
    """ADR-0006 输入系数分场景先验：探测校准失败路径的兜底默认值。"""

    def test_in_cpt_priors(self):
        for sc in SCENARIOS.values():
            if sc.get("kind", "llm") != "llm":
                continue   # 媒体场景（asr/ocr/tts）无文本 prompt 构造，无输入系数
            self.assertGreater(sc["in_cpt"], 0.5)
        # 实测序：C/C++ 代码的字符/token 远高于中文散文（3.9 vs 1.5）
        self.assertGreater(SCENARIOS["code"]["in_cpt"],
                           SCENARIOS["creative"]["in_cpt"])


class TestBuildMessages(unittest.TestCase):
    def test_zero_input_branch(self):
        """ctx=0：不注入参考材料，一句话命题，填充字符数为 0（ADR-0005）。"""
        msgs, est, filler_n = build_messages("creative", 0, 1.8, "n")
        self.assertEqual(filler_n, 0)
        self.assertEqual(len(msgs), 2)
        self.assertIn("800字", msgs[1]["content"])
        self.assertGreater(est, 0)

    def test_explicit_filler_chars(self):
        """显式 filler_chars 精确控制填充长度（嵌套前缀记账路径）。"""
        _, _, n1 = build_messages("code", 4096, 3.4, "n", filler_chars=5000)
        self.assertEqual(n1, 5000)

    def test_stream_offset_shifts_echo_region(self):
        """换料偏移翻 echo 改写区落点（2026-09-17 拍板）：同目标同 nonce
        同 filler_chars 下，offset>0 的构造填充长度不变、内容不同，echo
        预填（改写区开头）随之改变；缺省/0 与嵌套前缀口径逐字节一致。"""
        base, _, n1 = build_messages("code", 4096, 3.4, "nonce-5",
                                     filler_chars=20000, echo=True)
        same, _, _ = build_messages("code", 4096, 3.4, "nonce-5",
                                    filler_chars=20000, echo=True,
                                    stream_offset=0)
        shifted, _, n2 = build_messages("code", 4096, 3.4, "nonce-5",
                                        filler_chars=20000, echo=True,
                                        stream_offset=7919)
        self.assertEqual(base, same, "offset=0 与缺省口径一致")
        self.assertEqual(n1, n2, "换料只平移窗口，填充长度不变")
        self.assertNotEqual(base[1]["content"], shifted[1]["content"],
                            "平移后参考材料内容应不同")
        self.assertEqual(len(base), len(shifted), 3)
        self.assertNotEqual(base[2]["content"], shifted[2]["content"],
                            "echo 预填（改写区开头）应随落点翻牌改变")

    def test_nonzero_injects_filler_and_nonce(self):
        msgs, est, filler_n = build_messages("creative", 4096, 1.8, "nonce-42")
        self.assertGreater(filler_n, 0)
        self.assertIn("nonce-42", msgs[1]["content"])
        self.assertGreater(est, 4000 * 0.9)

    def test_agent_scenario(self):
        """Agent 场景（ADR-0014）：vendored SWE-agent 轨迹语料加载（块 ≥2000
        字符），构造路径与既有场景一致（零输入档无填充、非零档注入轨迹）。"""
        pool = SCENARIOS["agent"]["filler_pool"]
        self.assertGreater(len(pool), 1)   # vendored 轨迹块（corpus/agent/）
        self.assertTrue(all(len(b) >= 2000 for b in pool))
        self.assertGreater(len(SCENARIOS["agent"]["filler_stream"]), 100000)
        msgs, est, filler_n = build_messages("agent", 4096, 3.5, "nonce-7")
        self.assertGreater(filler_n, 0)
        self.assertIn("nonce-7", msgs[1]["content"])
        _, _, filler0 = build_messages("agent", 0, 3.5, "n")
        self.assertEqual(filler0, 0)

    def test_echo_instruction_variant(self):
        """高复用改写口径（ADR-0021）：echo=True 换用 echo_instruction 并在末尾
        追加 assistant 预填（改写区开头原样预填，物理消除聊天式前言）；同
        filler_chars + 同 nonce 下 user 与正常构造仅尾部任务句不同（服务端前缀
        缓存可命中整段参考材料）；无 echo_instruction 的场景回退普通指令且无
        预填；零输入档不受影响。"""
        normal, _, fn1 = build_messages("creative", 4096, 1.5, "nonce-9",
                                        filler_chars=5000)
        echo, _, fn2 = build_messages("creative", 4096, 1.5, "nonce-9",
                                      filler_chars=5000, echo=True)
        self.assertEqual(fn1, fn2, "echo 与正常构造填充长度一致")
        self.assertEqual(normal[0], echo[0], "system 不变")
        self.assertIn("改写润色", echo[1]["content"])
        self.assertNotIn("改写润色", normal[1]["content"])
        cut = normal[1]["content"].rindex("\n\n任务：")
        self.assertEqual(normal[1]["content"][:cut], echo[1]["content"][:cut],
                         "仅尾部任务句不同，head+filler 前缀完全一致")
        # assistant 预填：改写区开头原样预填，内容必出现在参考材料末尾区段
        self.assertEqual(len(echo), 3)
        self.assertEqual(echo[2]["role"], "assistant")
        prefill = echo[2]["content"]
        self.assertTrue(0 < len(prefill) <= 170)
        self.assertIn(prefill.strip()[:60], echo[1]["content"])
        # 改写区起点对齐（ADR-0021 续）：预填 = 改写区开头原样，且改写区起点
        # 落在结构边界——块边界（紧随 "\n\n"）或流首；同参数两次构造一致
        # （确定性，嵌套前缀多次运行不受影响）
        c2, _, _ = build_messages("creative", 4096, 1.5, "nonce-9",
                                  filler_chars=5000, echo=True)
        self.assertEqual(prefill, c2[2]["content"])
        filler = _fill(SCENARIOS["creative"]["filler_stream"], 5000)
        csc = SCENARIOS["creative"]
        region = min(5000, int(csc["max_tokens_default"] * csc["out_cpt"]))
        start = _align_echo_region(filler, region, csc["filler_boundaries"])
        self.assertEqual(filler[start:start + len(prefill)], prefill,
                         "预填 = 改写区开头原样")
        self.assertTrue(start == 0 or filler[start - 2:start] == "\n\n",
                        "改写区起点对齐到块/段边界，不从半个结构单元开始")
        self.assertGreaterEqual(len(filler) - start, 0.6 * region)
        self.assertLessEqual(len(filler) - start, 1.6 * region)
        self.assertEqual(len(normal), 2, "正常构造不带 assistant 消息")
        ec, _, _ = build_messages("code", 8192, 3.9, "n2",
                                  filler_chars=8000, echo=True)
        self.assertIn("重构", ec[1]["content"])
        self.assertEqual(ec[2]["role"], "assistant")
        # agent 无 echo_instruction：echo=True 回退普通 instruction，无预填
        ag_n, _, _ = build_messages("agent", 4096, 3.5, "n3")
        ag_e, _, _ = build_messages("agent", 4096, 3.5, "n3", echo=True)
        self.assertEqual(ag_n[1]["content"], ag_e[1]["content"])
        self.assertEqual(len(ag_e), 2)
        # 零输入档：echo 参数无效果
        z1, _, _ = build_messages("creative", 0, 1.5, "n")
        z2, _, _ = build_messages("creative", 0, 1.5, "n", echo=True)
        self.assertEqual(z1, z2)

    def test_agent_matrix_messages(self):
        """Agent 缓存×指令矩阵构造（ADR-0042）：预设上下文与指令拆开为两条
        user 消息（[system, user(编号行+轨迹上下文), user(指令)]）；同
        cache_chars 下预热与测量的前两条消息逐字节一致（前缀缓存命中口径，
        指令真实长度可链内差分）；指令段长度随 inst_tokens 单调增长、材料
        接续上下文之后的语料段（不与上下文同位）；模板固定开销 ~260
        chars（suffix 为引出长输出而加强，固定开销由链内差分吸收）；
        cache_tokens=0 为零缓存档，上下文
        消息仅编号行；prime=True 的指令消息为超短任务；seg_chars 显式指定
        材料字符数（偏离校正补测用）；SCENARIOS["agent"] 双阶梯默认值存在。"""
        cpt = 3.5
        sc = SCENARIOS["agent"]
        stream = sc["filler_stream"]
        # 同 cache_chars：预热与测量的前两条消息逐字节一致（前缀缓存命中口径）
        prime_m, _ep, ctx_p, seg_p = build_agent_messages(
            4096, 0, cpt, "n-1", cache_chars=5000, prime=True)
        meas_m, _e, ctx_m, seg_m = build_agent_messages(
            4096, 1024, cpt, "n-1", cache_chars=5000)
        self.assertEqual(ctx_p, ctx_m, "同 cache_chars 上下文段长度一致")
        self.assertEqual(ctx_p, 5000)
        self.assertEqual(seg_p, 0, "预热请求无指令材料")
        self.assertEqual([m["role"] for m in prime_m],
                         ["system", "user", "user"])
        self.assertEqual([m["role"] for m in meas_m],
                         ["system", "user", "user"])
        self.assertEqual(prime_m[0], meas_m[0], "system 逐字节一致")
        self.assertEqual(prime_m[1], meas_m[1],
                         "上下文消息逐字节一致（前缀缓存命中口径）")
        self.assertNotEqual(prime_m[2], meas_m[2], "指令消息分开")
        self.assertEqual(prime_m[2]["content"], "Reply with OK.",
                         "预热指令消息为超短任务")
        inst_msg = meas_m[2]["content"]
        self.assertTrue(inst_msg.startswith(sc["inst_prefix"]))
        self.assertTrue(inst_msg.endswith(sc["inst_suffix"]))
        # 指令材料接续上下文之后的语料段（流按长度取模错位，不与上下文同位）
        off = ctx_m % len(stream)
        self.assertEqual(
            inst_msg[len(sc["inst_prefix"]):len(sc["inst_prefix"]) + 40],
            stream[off:off + 40])
        # 模板固定开销（~260 chars ≈ 75 tokens@3.5）：suffix 为引出长输出而
        # 加强（thorough/detailed），64 tokens 小档被固定开销淹没属预期——
        # 指令真实长度走链内差分测量，固定开销在差分中抵消
        self.assertLessEqual(len(sc["inst_prefix"]) + len(sc["inst_suffix"]),
                             400)
        # 指令段长度随 inst_tokens 单调增长
        segs = [build_agent_messages(0, i, cpt, "n")[3]
                for i in (128, 256, 512, 1024, 2048)]
        self.assertTrue(all(a < b for a, b in zip(segs, segs[1:])),
                        f"指令段严格单调增长: {segs}")
        # seg_chars 显式指定指令材料字符数（偏离校正补测用）
        fx_m, _ef, _cz, seg_fx = build_agent_messages(
            0, 1024, cpt, "n", seg_chars=777)
        self.assertEqual(seg_fx, 777)
        self.assertEqual(fx_m[2]["content"], sc["inst_prefix"]
                         + _fill(stream, 777) + sc["inst_suffix"])
        # cache_tokens=0：零缓存档上下文消息仅编号行，无轨迹材料
        z_m, _ez, ctx_z, seg_z = build_agent_messages(0, 512, cpt, "n-2")
        self.assertEqual(ctx_z, 0)
        self.assertGreater(seg_z, 0)
        self.assertEqual(z_m[1]["content"], "参考材料（编号 n-2）：\n")
        # 双阶梯默认值（缓存 0~256K × 指令 128~16K tokens）
        self.assertEqual(sc["cache_ladder"],
                         [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
        self.assertEqual(sc["inst_ladder"],
                         [128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    def test_agent_seg_salt_cursor_nonoverlap(self):
        """同缓存档逐尝试错位取段（_run_agent_matrix 游标推进规则 seg_n +
        seg_n//2 + 1024 的构造侧不变量）：任意两次尝试的指令材料在语料流中
        区间不重叠、内容不互为前缀——否则取段偏移只随 ctx 定，升序测量时
        各指令档材料嵌套前缀，上一档残留被服务端算进 cache_hit（实测 128K
        档 4K 指令未命中仅 1554 的事故）。"""
        cpt = 3.5
        sc = SCENARIOS["agent"]
        stream = sc["filler_stream"]
        cache_chars = 20000
        salt = 0
        segs = []
        for inst in (64, 256, 1024, 4096):
            m, _e, ctx_n, seg_n = build_agent_messages(
                131072, inst, cpt, "n", cache_chars=cache_chars,
                seg_salt=salt)
            seg = m[2]["content"][len(sc["inst_prefix"]):
                                  len(m[2]["content"]) - len(sc["inst_suffix"])]
            self.assertEqual(len(seg), seg_n)
            off = (ctx_n + salt) % len(stream)
            segs.append((off, seg_n, seg))
            salt += seg_n + seg_n // 2 + 1024   # 与 _attempt_track 同推进规则
        for (o1, n1, _s1), (o2, _n2, _s2) in zip(segs, segs[1:]):
            self.assertGreaterEqual(o2, o1 + n1,
                                    "后一次尝试取段起点不落入前一次区间")
        for i, (_o, n1, s1) in enumerate(segs):
            for _o2, n2, s2 in segs[i + 1:]:
                if min(n1, n2) < 100:
                    continue   # 超短材料前缀比较无判别力，跳过
                self.assertFalse(s2.startswith(s1[:200]),
                                 "指令材料不互为前缀（防残留计入 cache_hit）")
                self.assertFalse(s1.startswith(s2[:200]),
                                 "指令材料不互为前缀（防残留计入 cache_hit）")


class TestOutLimitGuards(unittest.TestCase):
    """ADR-0016 输出长度治理的参数面：三场景引导文案模板齐全、容差与断流
    系数固定（行为路径见 test_e2e_mock 的翻转/回退/断流用例）。"""

    def test_hint_templates_per_scenario(self):
        for key, sc in SCENARIOS.items():
            if sc.get("kind", "llm") != "llm":
                continue   # 媒体场景走 _one_media，无输出长度引导（ADR-0020）
            self.assertIn("{limit}", sc["out_hint"], f"{key} 缺输出长度引导模板")
            # _one 统一 format(limit=…, aim=…)：translate/ocr 单向封顶无 {aim}，
            # 多余 kwargs 被 str.format 忽略——同一次调用
            # 全场景可跑（构造断言：不抛 KeyError 即通过）
            self.assertIsInstance(sc["out_hint"].format(limit=1234, aim=1000),
                                  str, key)
        # 中文场景按字/字符引导，英文 agent 场景按 characters
        self.assertIn("字", SCENARIOS["creative"]["out_hint"])
        self.assertIn("字符", SCENARIOS["code"]["out_hint"])
        self.assertIn("characters", SCENARIOS["agent"]["out_hint"])

    def test_creative_code_hint_bidirectional(self):
        """creative/code out_hint 双向引导（ADR-0016，与 agent 同推广）：aim
        目标 + 封顶子句齐备——单向封顶无下限引导，实测模型往远低于预算的
        长度收敛，短输出 decode 窗口不稳定。"""
        for key in ("creative", "code"):
            sc = SCENARIOS[key]
            self.assertIn("{aim}", sc["out_hint"])
            self.assertIn("{limit}", sc["out_hint"])
            hint = sc["out_hint"].format(limit=921, aim=768)
            # aim 目标句在前、封顶子句在后（引导为主、封顶兜底）
            self.assertLess(hint.index("约 768"), hint.index("不得超出"))
            # 封顶子句必须保留（防线 2 本职：服务端不截断时防写超预算跑飞）
            self.assertIn("不得超出", sc["out_hint"])

    def test_agent_hint_bidirectional(self):
        """agent out_hint 双向引导（ADR-0016 + 输出长度治理）：aim 目标 ≈
        limit ÷ OUT_HINT_FACTOR（limit 已含 1.2 余量）——防模型以「材料
        截断」为由如实短答（实测 308/310/401 tokens 就 finish=stop），单向
        封顶无下限引导罩不住。"""
        sc = SCENARIOS["agent"]
        self.assertIn("{aim}", sc["out_hint"])
        self.assertIn("{limit}", sc["out_hint"])
        # aim/limit 同源折算：tokens × out_cpt，limit 再 ×1.2 余量
        tokens = 512
        aim = int(tokens * sc["out_cpt"])
        limit = int(tokens * sc["out_cpt"] * OUT_HINT_FACTOR)
        self.assertEqual(aim, int(limit / OUT_HINT_FACTOR))
        hint = sc["out_hint"].format(limit=limit, aim=aim)
        self.assertIn(f"about {aim} characters", hint)
        self.assertIn(f"exceed {limit} characters", hint)
        # 引导方向：目标句在前、硬上限句在后（thorough 引导为主、封顶兜底）
        self.assertLess(hint.index("aiming"), hint.index("never exceed"))

    def test_factors(self):
        self.assertEqual(OUT_HINT_FACTOR, 1.2)     # 引导上限 = 预设 ×1.2
        self.assertEqual(CLIENT_CUT_FACTOR, 2)     # 断流阈值 = 下发上限 ×2
        # 引导字数折算：512 tokens 创意档 × 1.5 字/token × 1.2 = 921 字
        self.assertEqual(int(512 * SCENARIOS["creative"]["out_cpt"]
                             * OUT_HINT_FACTOR), 921)


class TestMediaCorpus(unittest.TestCase):
    """ADR-0020 媒体语料：合成/解析往返、时长精确性、用户语料优先与确定性。"""

    def test_synth_wav_roundtrip(self):
        wav = _synth_speech_wav(5)
        self.assertAlmostEqual(_wav_seconds(wav), 5.0, places=2)
        wav30 = _synth_speech_wav(30)
        self.assertAlmostEqual(_wav_seconds(wav30), 30.0, places=2)
        # 非 wav 输入返回 None 而不是抛错
        self.assertIsNone(_wav_seconds(b"not a wav"))

    def test_asr_audio_cached_and_deterministic(self):
        bench._AUDIO_CACHE.clear()
        a1, s1 = _asr_audio(5)
        a2, s2 = _asr_audio(5)
        self.assertIs(a1, a2, "同档音频须缓存复用（阶梯/并发共享同一份负载）")
        self.assertAlmostEqual(s1, 5.0, places=2)
        self.assertAlmostEqual(s2, s1)

    def test_asr_audio_prefers_user_corpus(self):
        """corpus/asr/*.wav 存在时循环用户音频到目标时长（帧级精确）。"""
        sr = 16000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"\x01\x00" * sr * 3)   # 3s 用户音频
        with tempfile.TemporaryDirectory() as td:   # 独立临时目录，并行/残留互不干扰
            with unittest.mock.patch.object(bench, "CORPUS_ASR_DIR", td):
                with open(os.path.join(td, "clip.wav"), "wb") as f:
                    f.write(buf.getvalue())
                try:
                    bench._AUDIO_CACHE.clear()
                    audio, actual = _asr_audio(7)   # 3s 循环到 7s
                    self.assertAlmostEqual(actual, 7.0, places=2)
                    self.assertAlmostEqual(_wav_seconds(audio), 7.0, places=2)
                finally:
                    bench._AUDIO_CACHE.clear()

    def test_synth_png_roundtrip_and_size(self):
        png = _synth_document_png(1240, 1754, seed=0)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png[12:16], b"IHDR")
        self.assertEqual(struct.unpack(">II", png[16:24]), (1240, 1754))
        # 确定性：同 seed 同图
        self.assertEqual(png, _synth_document_png(1240, 1754, seed=0))

    def test_ocr_pool_fixed_three_resolutions(self):
        """OCR 图池固定内置合成（ADR-0037）：3 种分辨率文档页、确定性生成——
        跨环境统一负载标准，不再有用户文件通道。"""
        bench._IMG_CACHE = None
        pool = _ocr_images()
        self.assertEqual([(w, h) for _, w, h, _ in pool],
                         [(1240, 1754), (1654, 2339), (2480, 3508)])
        for data, w, h, mime in pool:
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (w, h))
            self.assertEqual(mime, "image/png")
        bench._IMG_CACHE = None
        self.assertEqual([d for d, *_ in _ocr_images()],
                         [d for d, *_ in pool])   # 重建逐字节一致（确定性）
        bench._IMG_CACHE = None

    def test_ocr_messages_shape(self):
        msgs, img_tokens = _build_ocr_messages(4, nonce=7)
        self.assertEqual(msgs[0]["role"], "system")
        content = msgs[1]["content"]
        self.assertEqual(len(content), 5)   # 4 图 + 1 文本指令
        self.assertTrue(all(p["type"] == "image_url" for p in content[:4]))
        self.assertTrue(content[4]["text"].startswith("请识别"))
        self.assertGreater(img_tokens, 0)
        # 同 nonce 确定：不同 nonce 错图（bust 缓存）
        msgs2, _ = _build_ocr_messages(4, nonce=8)
        self.assertNotEqual(content[0]["image_url"]["url"],
                            msgs2[1]["content"][0]["image_url"]["url"])

    def test_tts_text_length_and_nonce(self):
        t1 = _tts_text(200, "a")
        self.assertGreaterEqual(len(t1), 195)   # 编号行 + 语料截取（略超可接受）
        self.assertIn("a", t1[:12])
        t2 = _tts_text(200, "b")
        self.assertNotEqual(t1[:16], t2[:16], "编号行使请求文本错开")


class TestTranslateScenario(unittest.TestCase):
    """翻译场景（中→英，原文字长阶梯替代 ctx_list、矩阵可自定义）：构造形状、
    nonce 错开、输出预算随档伸缩且受全局 max_tokens 上界约束（ADR-0034）。"""

    def test_messages_shape_and_length(self):
        sc = SCENARIOS["translate"]
        msgs = _build_translate_messages(200, "12345")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        user = msgs[1]["content"]
        self.assertIn("12345", user[:20])   # 编号行在原文开头（bust 前缀缓存）
        self.assertTrue(user.endswith(sc["instruction"]))
        # 原文截取命中目标字符数（编号行 + 分隔 + 指令为固定开销）
        self.assertGreaterEqual(len(user), 200 + len(sc["instruction"]))
        self.assertLess(len(user), 200 + 64 + len(sc["instruction"]))

    def test_nonce_staggers_requests(self):
        a = _build_translate_messages(200, "a")[1]["content"]
        b = _build_translate_messages(200, "b")[1]["content"]
        self.assertNotEqual(a[:24], b[:24], "编号行使并发/复测请求文本错开")

    def test_max_tokens_scales_with_rung(self):
        budgets = [_translate_max_tokens(r, 1.5)
                   for r in SCENARIOS["translate"]["ladder"]]
        self.assertEqual(budgets, sorted(budgets))   # 随档单调递增
        # 6400 字原文译文 ~4-8K tokens：预算必须显著超过定长 1024，否则必截断
        self.assertGreater(budgets[-1], 4096)
        # 同档下更高的校准 cpt（每 token 字符更多）→ 原文 tokens 估值更低 → 预算更低
        self.assertLess(_translate_max_tokens(800, 3.0),
                        _translate_max_tokens(800, 1.5))

    def test_max_tokens_capped_by_global(self):
        """全局 max_tokens 对翻译场景同样生效（上界语义）：随档预算超出时按
        cap 截断，低档位预算不足 cap 时不受影响。"""
        uncapped = _translate_max_tokens(6400, 1.5)
        self.assertGreater(uncapped, 1024)
        self.assertEqual(_translate_max_tokens(6400, 1.5, cap=1024), 1024)
        low = _translate_max_tokens(50, 1.5)   # 50 字档预算 ~99，低于 cap
        self.assertEqual(_translate_max_tokens(50, 1.5, cap=1024), low)
        self.assertEqual(_translate_max_tokens(50, 1.5, cap=0), low)   # 无 cap

    def test_ladder_metadata(self):
        sc = SCENARIOS["translate"]
        self.assertEqual(sc.get("kind", "llm"), "llm")   # 与创意/代码同请求路径
        self.assertEqual(sc["ladder"], [50, 100, 200, 400, 800, 1600, 3200, 6400])
        self.assertEqual(sc["unit"], "字原文")
        self.assertIn("{limit}", sc["out_hint"])   # 防线 2 输出长度引导模板
        # 无 echo_instruction：echo 模板续写对翻译无意义，自动回退普通构造
        self.assertNotIn("echo_instruction", sc)
