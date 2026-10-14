"""引擎存档落盘 / 语料出处 / 密度台账回归测试（自 tests/test_engine_unit.py 拆分）。

本组定位（纯逻辑 + 落盘副作用，不依赖 mock 网关与 HTTP 服务）：
语料出处契约（三份 vendored 语料的内容清单哈希与统计区间钉成契约、"目录缺失
回退内置池"不得静默发生、生成器脚本 --verify 与测试判定一致、口径版本表随落盘
记档）、异常输入全文外置侧车（ADR-0053 可精确复跑 / ADR-0082 存档瘦身：超长
in_text 落 <run_id>.anomaly.txt 侧车，存档留头尾摘要 + in_text_ref 字节切片，
预算耗尽截断留痕，侧车写失败绝不拖垮 _save）、存档输出采样累积（_head_sample）、
输入全文渲染（_render_in_full / _strip_in_full）、密度台账纯函数（分段线性
chars↔tokens 反解、子表归户与 LRU 上限、落盘守卫与损坏容忍）。

运行（从仓库根目录，任一方式，无需 pytest 也可跑）：
    python -m pytest tests/unit/test_engine_persistence.py -v
    python -m unittest tests.unit.test_engine_persistence -v
    python -m unittest discover -s tests/unit -p 'test_engine_persistence.py'
"""
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

import bench
from bench import (
    BenchRun,
    _head_sample,
    _render_in_full,
    _strip_in_full,
)

# 语料内容清单（相对路径 + NUL + 内容，按路径排序流式 sha256）——与
# scripts/build_agent_corpus.py 的 manifest_sha256 同算法：两条独立实现互为
# 校验，任一侧被误改都会不一致。
def _manifest_sha256(base: str, rels: list[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(rels):
        h.update(rel.encode())
        h.update(b"\0")
        with open(os.path.join(base, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class TestCorpusProvenance(unittest.TestCase):
    """语料是测速的"刻度"：改语料会让历史存档失去可比性，必须显式发生。

    本类把三份 vendored 语料的**内容清单哈希与统计区间**钉成契约（数字来源与
    核对方式见各目录 PROVENANCE.md）。语料一旦被误改/误重建/静默降级到内置回退
    池，这里立刻变红——而引擎侧的"目录缺失回退内置池"设计本身不会报错。
    故意更新语料时：跑一次下面命令，把新哈希与区间同步到 PROVENANCE 与本节。
        python -m pytest -q tests/unit/test_engine_persistence.py -k corpus -v
    """
    # 本文件在 tests/unit/ 下一层，故比源文件（tests/）多退一级到仓库根
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _dir(self, *parts):
        return os.path.join(self.ROOT, "corpus", *parts)

    def test_agent_corpus_manifest_and_block_bounds(self):
        base = self._dir("agent")
        rels = sorted(f for f in os.listdir(base)
                      if f.startswith("traj-") and f.endswith(".txt"))
        self.assertEqual(len(rels), 100, "agent 语料块数变了（PROVENANCE.md 记载 100）")
        self.assertEqual(
            _manifest_sha256(base, rels),
            "4c85ab95e43d58c782546cc7ec345a859283fdd9f540bb93a8890d1367627778",
            "agent 语料内容变了——若是有意更新，请同步 corpus/agent/PROVENANCE.md")
        chars = [len(open(os.path.join(base, r), encoding="utf-8").read()) for r in rels]
        self.assertGreaterEqual(min(chars), 2000,
                                "存在低于引擎采纳门限（_load_agent_pool ≥2000）的块")
        # 上界用实测值（DOC_MAX_BLOCK=40200）：生成器判定 40000，仓内 3 个块
        # 由早期修订脚本产出、略超——见 PROVENANCE.md 的"不同源"一节
        self.assertLessEqual(max(chars), 40200,
                             "agent 语料块超文档化上界（PROVENANCE.md 记载 40183）")

    def test_agent_corpus_satisfies_documented_contract_via_verify(self):
        """`--verify`（只读、不依赖 pyarrow）必须与测试判定一致。"""
        import subprocess
        r = subprocess.run([sys.executable,
                            os.path.join(self.ROOT, "scripts", "build_agent_corpus.py"),
                            "--verify"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("块长契约 OK", r.stdout)

    def test_code_corpus_manifest(self):
        base = self._dir("code")
        rels = [os.path.relpath(os.path.join(dp, f), base)
                for dp, _, fs in os.walk(base) for f in fs
                if f not in ("LICENSE", "README.md", "PROVENANCE.md")]
        self.assertEqual(len(rels), 15, "code 语料文件数变了（PROVENANCE.md 记载 15）")
        self.assertEqual(
            _manifest_sha256(base, rels),
            "06b408c7b5d00a2c5d8bb0160f9705d500425ca5802086a06feba162dff8d887",
            "code 语料内容变了——若是有意更新，请同步 corpus/code/PROVENANCE.md")

    def test_creative_corpus_manifest_and_size(self):
        base = self._dir("creative")
        path = os.path.join(base, "hongloumeng.txt")
        raw = open(path, encoding="utf-8").read()
        with open(path, "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(),
                "58e75b74f0c51e7863f8aa6871ec349f5c96ffdbecb5bdd5cf83ea016f553826",
                "creative 语料变了——若是有意更新，请同步 corpus/creative/PROVENANCE.md")
        # 去空白字符数 = 语料的"刻度"（设计文档记 83 万字符、循环点推到 512K 之后）
        self.assertEqual(len("".join(raw.split())), 838587,
                         "creative 语料字符尺度变了（PROVENANCE.md 记载 838,587）")

    def test_creative_corpus_has_no_pg_header(self):
        """PG 页眉已剥离（版权声明改存 LICENSE-PG.txt）——正文里不得再出现。"""
        text = open(self._dir("creative", "hongloumeng.txt"), encoding="utf-8").read()
        self.assertNotIn("Project Gutenberg", text)
        self.assertNotIn("\ufffd", text, "残留 U+FFFD 替换字符")

    def test_license_and_provenance_present(self):
        """三份 vendored 语料各有一份溯源文档；creative 另需 PG 许可声明。"""
        for d in ("agent", "code", "creative"):
            self.assertTrue(os.path.isfile(self._dir(d, "PROVENANCE.md")),
                            f"corpus/{d}/PROVENANCE.md 缺失")
        pg = open(self._dir("creative", "LICENSE-PG.txt"), encoding="utf-8").read()
        self.assertIn("Project Gutenberg", pg)
        self.assertIn("24264", pg)
        self.assertTrue(os.path.isfile(self._dir("code", "LICENSE")),
                        "corpus/code/LICENSE（上游 MIT 许可）缺失")

    def test_metric_version_is_documented(self):
        """口径版本表：每个已发布的版本都要有一句"它是什么口径"的说明——
        版本号本身没有意义，读者要能查到怎么解读（ADR-0083）。"""
        import bench as b
        self.assertGreaterEqual(b.METRIC_VERSION, 1)
        for v in range(1, b.METRIC_VERSION + 1):
            self.assertIn(v, b.METRIC_VERSION_NOTES, f"缺少 v{v} 的口径说明")
            self.assertTrue(b.METRIC_VERSION_NOTES[v].strip())
        self.assertEqual(b.LEGACY_METRIC_VERSION, 0)

    def test_save_stamps_metric_version(self):
        """落盘顶层带口径版本；点级版本随点自身保存（拼接/复测可混档）。"""
        import bench as b
        self.assertIn("METRIC_VERSION", b.__dict__)
        with tempfile.TemporaryDirectory() as td:
            run = BenchRun({}, results_dir=td)
            run.results.append({"model": "m1", "scenario": "code", "ctx_target": 0,
                                "metric_version": b.METRIC_VERSION})
            run._save()
            with open(os.path.join(td, f"{run.run_id}.json"), encoding="utf-8") as f:
                d = json.load(f)
        self.assertEqual(d["metric_version"], b.METRIC_VERSION)
        self.assertEqual(d["results"][0]["metric_version"], b.METRIC_VERSION)

    def test_engine_actually_loads_vendored_corpora(self):
        """回归"静默降级"：语料在盘时必须真被引擎加载（而非回退内置池）。"""
        self.assertTrue(bench._load_agent_pool(), "agent 语料未被加载（回退内置池？）")
        self.assertTrue(bench._load_creative_pool(), "creative 语料未被加载")
        self.assertTrue(bench._load_code_pool(), "code 语料未被加载")


class TestAnomalyTextExternalization(unittest.TestCase):
    """异常输入全文外置（ADR-0053 可精确复跑 + ADR-0082 存档瘦身）：
    超长 in_text 落 <run_id>.anomaly.txt 侧车，存档只留头尾摘要 + in_text_ref
    字节切片；短文本内联；预算耗尽截断留痕；侧车写失败绝不拖垮 _save。"""

    def _run(self, td, results):
        run = BenchRun({}, results_dir=td)
        run.results = results
        run._save()   # 同步调用即可（真实调用点在 to_thread 内）
        return run

    @staticmethod
    def _load(td, run):
        with open(os.path.join(td, f"{run.run_id}.json"), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _side(td, run):
        return os.path.join(td, f"{run.run_id}{bench.ANOMALY_TEXT_SUFFIX}")

    def test_long_in_text_externalized_to_sidecar(self):
        """100K 字 reqs.in_text + 120K 字 reps_discarded.in_text：存档远小于输入、
        截断文本带省略标记、ref 的 len/chars/sha1 正确、侧车切片字节 == 原文。"""
        a, b = "甲" * 100_000, "乙" * 120_000
        da, db = a.encode("utf-8"), b.encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            run = self._run(td, [{"model": "m1",
                                  "reqs": [{"req": 0, "in_text": a}],
                                  "reps_discarded": [{"in_text": b}]}])
            path = os.path.join(td, f"{run.run_id}.json")
            side = self._side(td, run)
            self.assertTrue(os.path.exists(side), "侧车应存在")
            self.assertLess(os.path.getsize(path), 20000,
                            "存档应远小于输入全文（截断后仅头尾摘要）")
            d = self._load(td, run)
            pt = d["results"][0]
            raw = open(side, "rb").read()
            for holder, full, data in ((pt["reqs"][0], a, da),
                                       (pt["reps_discarded"][0], b, db)):
                self.assertNotEqual(holder["in_text"], full)
                self.assertIn("……", holder["in_text"], "应有省略标记")
                ref = holder["in_text_ref"]
                self.assertEqual(ref["file"],
                                 f"{run.run_id}{bench.ANOMALY_TEXT_SUFFIX}")
                self.assertEqual((ref["len"], ref["chars"]),
                                 (len(data), len(full)))
                self.assertEqual(ref["sha1"], hashlib.sha1(data).hexdigest())
                sl = raw[ref["off"]:ref["off"] + ref["len"]]
                self.assertEqual(sl, data, "侧车切片应等于原文")
                self.assertEqual(hashlib.sha1(sl).hexdigest(), ref["sha1"])
            trunc = pt["reqs"][0]["in_text"]
            self.assertEqual(trunc[:bench.IN_TEXT_TRUNC_HEAD],
                             a[:bench.IN_TEXT_TRUNC_HEAD])
            self.assertEqual(trunc[-bench.IN_TEXT_TRUNC_TAIL:],
                             a[-bench.IN_TEXT_TRUNC_TAIL:])
            self.assertNotIn("anomaly_text_warning", d, "预算内不应告警")

    def test_short_in_text_stays_inline_no_sidecar(self):
        """短文本（含恰好等于阈值）保持内联原样，且不产生空侧车文件。"""
        short = "S" * 50
        edge = "T" * bench.IN_TEXT_INLINE_MAX_CHARS
        with tempfile.TemporaryDirectory() as td:
            run = self._run(td, [{"reqs": [{"in_text": short}],
                                  "reps_discarded": [{"in_text": edge}]}])
            d = self._load(td, run)
            pt = d["results"][0]
            self.assertEqual(pt["reqs"][0]["in_text"], short)
            self.assertEqual(pt["reps_discarded"][0]["in_text"], edge)
            self.assertNotIn("in_text_ref", pt["reqs"][0])
            self.assertNotIn("in_text_ref", pt["reps_discarded"][0])
            self.assertFalse(os.path.exists(self._side(td, run)),
                             "无超长文本不得留下空侧车")

    def test_reps_reqs_in_text_externalized(self):
        """reps[].reqs[].in_text（69% 体积的来源）也必须走外置，否则静默回归。"""
        big = "R" * 60000
        with tempfile.TemporaryDirectory() as td:
            run = self._run(td, [{"model": "m1",
                                  "reps": [{"reqs": [{"in_text": big},
                                                     {"in_text": "x"}]}]}])
            d = self._load(td, run)
            h = d["results"][0]["reps"][0]["reqs"][0]
            self.assertIn("in_text_ref", h)
            self.assertIn("……", h["in_text"])
            ref = h["in_text_ref"]
            raw = open(self._side(td, run), "rb").read()
            self.assertEqual(raw[ref["off"]:ref["off"] + ref["len"]],
                             big.encode("utf-8"))
            self.assertEqual(d["results"][0]["reps"][0]["reqs"][1]["in_text"], "x",
                             "未超阈值的兄弟字段不受影响")

    def test_budget_exhausted_marks_truncated_and_warns(self):
        """预算耗尽：条目不写侧车、不给 ref（不谎称全文留存），只截断 +
        in_text_truncated 标 + 顶层 anomaly_text_warning；无空侧车。"""
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.object(bench, "ANOMALY_TEXT_BUDGET_BYTES",
                                            1000):
                run = self._run(td, [{"reqs": [{"in_text": "A" * 5000}],
                                      "reps_discarded": [{"in_text": "B" * 5000}]}])
            d = self._load(td, run)
            pt = d["results"][0]
            for h in (pt["reqs"][0], pt["reps_discarded"][0]):
                self.assertTrue(h["in_text_truncated"])
                self.assertNotIn("in_text_ref", h)
                self.assertIn("……", h["in_text"])
            self.assertIn("anomaly_text_warning", d)
            self.assertFalse(os.path.exists(self._side(td, run)),
                             "预算内一条都没写时不得留空侧车")

    def test_budget_stops_writing_after_first_overflow(self):
        """预算只够第一条：第一条外置（有 ref），第二条起全局停写（截断+标），
        不是逐条"见缝插针"继续塞更小的条目。"""
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.object(bench, "ANOMALY_TEXT_BUDGET_BYTES",
                                            6000):
                run = self._run(td, [{"reqs": [{"in_text": "A" * 5000}],
                                      "reps_discarded": [{"in_text": "B" * 5000}]}])
            d = self._load(td, run)
            pt = d["results"][0]
            self.assertIn("in_text_ref", pt["reqs"][0])
            self.assertNotIn("in_text_ref", pt["reps_discarded"][0])
            self.assertTrue(pt["reps_discarded"][0]["in_text_truncated"])
            self.assertIn("anomaly_text_warning", d)
            self.assertEqual(len(open(self._side(td, run), "rb").read()), 5000,
                             "侧车只应含预算内的第一条")

    def test_save_survives_sidecar_write_failure(self):
        """侧车写失败（OSError）：_save 仍成功、全文保持内联、无空侧车——
        瘦身特性不得让存档本身丢失。"""
        big = "X" * 100000
        real_open = open

        def flaky(path, *a, **k):
            if str(path).endswith(bench.ANOMALY_TEXT_SUFFIX):
                raise OSError("模拟侧车不可写")
            return real_open(path, *a, **k)

        with tempfile.TemporaryDirectory() as td:
            run = BenchRun({}, results_dir=td)
            run.results = [{"reqs": [{"in_text": big}]}]
            with unittest.mock.patch("builtins.open", new=flaky):
                run._save()   # 不得抛
            d = self._load(td, run)
            self.assertEqual(d["results"][0]["reqs"][0]["in_text"], big)
            self.assertNotIn("in_text_ref", d["results"][0]["reqs"][0])
            self.assertFalse(os.path.exists(self._side(td, run)))


class TestHeadSample(unittest.TestCase):
    """存档输出采样累积（_head_sample）：超 200 截断、不足全留、截断即停。"""

    def test_boundary_and_empty(self):
        """一条覆盖三种形态：短原文整条保留、200 字边界截断（含单条超长串）、
        空输入与满缓冲（截断即停，收满后不再追加）。"""
        self.assertEqual(_head_sample("abc", "de", 200), "abcde")
        self.assertEqual(_head_sample("a" * 199, "bcd", 200), "a" * 199 + "b")
        self.assertEqual(_head_sample("", "字" * 250, 200), "字" * 200)
        self.assertEqual(_head_sample("", "", 200), "", "空输出采样为空串")
        self.assertEqual(_head_sample("y" * 200, "", 200), "y" * 200)


class TestRenderInFull(unittest.TestCase):
    """输入全文渲染（_render_in_full，瞬态 _in_full 的内容）：每条消息一行
    「【role】文本」、parts 列表拼接 text、连续非 text 部分以「［图片×N］」
    占位、不设长度上限（异常日志要求输入原文全部列出）。"""

    def test_plain_messages_one_line_each(self):
        msgs = [{"role": "system", "content": "系统提示"},
                {"role": "user", "content": "问题"}]
        self.assertEqual(_render_in_full(msgs),
                         "【system】系统提示\n【user】问题")

    def test_parts_list_text_and_image_placeholder(self):
        """OCR 多模态 parts：text 拼接、连续 image_url 部分合并为一个
        「［图片×N］」占位。"""
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:a"}},
            {"type": "image_url", "image_url": {"url": "data:b"}},
            {"type": "text", "text": "请识别"},
            {"type": "image_url", "image_url": {"url": "data:c"}}]}]
        self.assertEqual(_render_in_full(msgs),
                         "【user】［图片×2］请识别［图片×1］")

    def test_no_length_cap(self):
        """不设长度上限：10 万字符输入全文保留（与 in_sample 的 200 上限区分）。"""
        long_input = "长" * 100000
        msgs = [{"role": "user", "content": long_input}]
        self.assertEqual(_render_in_full(msgs), "【user】" + long_input)

    def test_strip_in_full_removes_transient_key(self):
        """_strip_in_full 定稿剥离：reqs 全部去 _in_full，肇事 req 的 in_text
        独立键保留；无 reqs 键/空点不炸。"""
        pt = {"reqs": [{"req": 0}, {"req": 1, "_in_full": "x"},
                       {"req": 2, "anomaly": "early_stop", "in_text": "全文"}]}
        _strip_in_full(pt)
        self.assertFalse(any("_in_full" in r for r in pt["reqs"]))
        self.assertEqual(pt["reqs"][2]["in_text"], "全文",
                         "in_text 独立键不受剥离影响")
        _strip_in_full({})


class TestDensityLedger(unittest.TestCase):
    """密度台账纯函数：分段线性 chars↔tokens 反解。

    为什么钉这个：语料流 token 密度沿流阶跃（代码语料的整数字常量表区边际
    密度 ~1.4 c/t vs 邻段 ~3.2），单锚点外推对第一个跨过它的档位必然失准
    （实测 64K 档首测 +40%）。台账用**包围目标的相邻两段斜率**，把阶跃
    包在表里。"""

    # 实测形状简化的两段曲线：普通段 3.2 c/t，数字表区 1.4 c/t
    PLAIN = [(0.0, 0.0), (100000.0, 31250.0)]        # 100000/3.2
    DIP = [(100000.0, 31250.0), (200000.0, 102678.0)]  # +100000/1.4 ≈ +71428

    def test_below_min_anchor_returns_none(self):
        """目标低于最小锚点：不得用首段斜率向下外推（真机踩过的坑，
        2026-10-10）。台账只有大锚点时，首段斜率跳过了中间整段、被密度
        塔陷拉平（实测 2.25 c/t vs 小档区真实 3.5 c/t），给 8K 档造出
        +89.7% 的 prompt。呓实交回旧公式（旧公式的锚点就在小档区，反而准）。
        """
        anchors = [(113473.0, 32947.0), (336491.0, 132168.0)]
        self.assertIsNone(bench._ledger_interp(anchors, 8192, 1.8),
                          "低于最小锚点必须返回 None，不猜")
        # 区间内仍正常内插（证明不是把整个函数关掉了）
        self.assertIsNotNone(bench._ledger_interp(anchors, 60000, 1.8))

    def test_interpolation_uses_enclosing_segment_not_last_slope(self):
        """关键性质：目标落在两段之间时，用的是它自己那一段的斜率。
        旧单锚点公式拿邻段斜率外推，遇阶跃必偏。"""
        anchors = self.PLAIN + self.DIP[1:]
        # 目标 60000 tokens 落在塌陷段（31250~102678）内
        c = bench._ledger_interp(anchors, 60000, 3.2)
        self.assertIsNotNone(c)
        # 该段斜率 = 100000/71428 ≈ 1.4；按此从 (100000, 31250) 起推
        self.assertAlmostEqual(c, 100000 + (60000 - 31250) * (100000 / 71428),
                               delta=2)
        self.assertLess(c, 100000 + (60000 - 31250) * 3.2,
                        "若误用普通段斜率会构造过长（就是 +40% 的成因）")

    def test_extrapolation_beyond_max_anchor_uses_last_slope(self):
        anchors = self.PLAIN + self.DIP[1:]
        c = bench._ledger_interp(anchors, 110000, None)
        self.assertIsNotNone(c)
        self.assertGreater(c, anchors[-1][0], "目标超过最大锚点，构造必须更长")

    def test_single_anchor_returns_none(self):
        """单锚点没有局部斜率，必须诚实交回旧公式而不是猜。"""
        self.assertIsNone(bench._ledger_interp([(10000.0, 3000.0)], 4096, 3.2))
        self.assertIsNone(bench._ledger_interp([], 4096, 3.2))

    def test_add_keeps_sorted_and_dedups_near_points(self):
        pts = bench._ledger_add([], 10000, 3000)
        pts = bench._ledger_add(pts, 20000, 6000)
        pts = bench._ledger_add(pts, 20010, 6500)   # 同位置重测 → 取新
        self.assertEqual([p[0] for p in pts], [10000.0, 20010.0])
        self.assertEqual(pts[-1][1], 6500.0, "同位置近点应取新观测")

    def test_add_rejects_non_monotonic_observation(self):
        """字更多却 tokens 更少的观测（删减重试/换料窗口）不得让台账折返。"""
        pts = bench._ledger_add([], 10000, 5000)
        pts = bench._ledger_add(pts, 20000, 4000)   # 非单调
        self.assertEqual(len(pts), 1, "非单调观测应丢弃")
        self.assertEqual(pts[0][0], 10000.0)

    def test_add_rejects_out_of_band_density(self):
        pts = bench._ledger_add([], 10000, 100000)  # 0.1 c/t，越下界
        self.assertEqual(pts, [])
        pts = bench._ledger_add([], 1000000, 100)   # 10000 c/t，越上界
        self.assertEqual(pts, [])

    def test_trim_keeps_max_anchor(self):
        """超上限抽稀必须保留最大锚点：它是向上外推的唯一起跳点。"""
        pts = []
        for i in range(bench.DENSITY_LEDGER_MAX_POINTS + 40):
            pts = bench._ledger_add(pts, (i + 1) * 10000.0, (i + 1) * 3000.0)
        self.assertLessEqual(len(pts), bench.DENSITY_LEDGER_MAX_POINTS)
        self.assertEqual(pts[-1][0], (bench.DENSITY_LEDGER_MAX_POINTS + 40) * 10000.0,
                         "最大锚点被抽稀掉了——大档位将无锚点可外推")

    def test_fingerprint_tracks_corpus_content(self):
        fp1 = bench._stream_fingerprint("code")
        self.assertIsNotNone(fp1, "code 场景应有语料指纹（台账可寻址）")
        self.assertEqual(fp1, bench._stream_fingerprint("code"), "指纹必须确定")
        self.assertNotEqual(fp1, bench._stream_fingerprint("creative"),
                            "不同语料不得共用台账")
        self.assertIsNone(bench._stream_fingerprint("ocr"),
                          "无文本语料流的场景不走台账")

    def test_key_includes_model_and_corpus(self):
        """键必须同时含模型与语料指纹：密度由分词器决定（同一段字节在不同
        模型上 tokens 不同），拿一个模型的表切另一个模型的 prompt，错得比
        不用台账更隐蔽。"""
        k1 = bench._ledger_key("m1", "code")
        self.assertNotEqual(k1, bench._ledger_key("m2", "code"), "不同模型不得共用")
        self.assertNotEqual(k1, bench._ledger_key("m1", "creative"), "不同语料不得共用")
        self.assertEqual(k1, bench._ledger_key("m1", "code"), "键必须确定")
        self.assertIsNone(bench._ledger_key("m1", "ocr"), "无流场景无台账")

    def test_trimmed_and_shifted_observations_excluded(self):
        """台账只收「从流首 0 截取、未删减」的观测（真机踩过的坑，现场复现
        于 2026-10-10）：被 _trim_middle 裁过的批次（实发量<构造量）和换料
        批次（stream_offset>0）都不是流首前缀，入表会顶掉同位置的正确锚点，
        下一轮沿假斜率构造——比不加台账更坏。"""
        run = BenchRun({})
        run._probe_density[("m", "code")] = 3.47   # 归户指纹与既有子表同口径
        lk = bench._ledger_key("m", "code")
        run.density[lk] = [{"density": 3.47, "source": {}, "seen": 1.0,
                            "anchors": [[113538.0, 32947.0]]}]

        async def real_ok(sent):
            return [{"usage_real": True, "prompt_tokens": 52748,
                     "sent_chars": sent}]

        import asyncio as _aio

        async def go():
            # 删减过的观测（sent < 构造）：不得入表
            await run._calibrate_usage(("m", "code"), 113857, 113077,
                                       await real_ok(60000), prefix_obs=True)
            self.assertEqual(run.density[lk][0]["anchors"],
                             [[113538.0, 32947.0]],
                             "删减批次不得污染台账，也不得顶掉正确锚点")
            # 换料批次（prefix_obs=False）：同样不得入表
            await run._calibrate_usage(("m", "code"), 113857, 113077,
                                       await real_ok(113857), prefix_obs=False)
            self.assertEqual(run.density[lk][0]["anchors"],
                             [[113538.0, 32947.0]])
            # 干净前缀观测：正常入表
            await run._calibrate_usage(("m", "code"), 197502, 197110,
                                       [{"usage_real": True,
                                         "prompt_tokens": 70397,
                                         "sent_chars": 197502}], prefix_obs=True)
            self.assertIn([197110.0, 70397.0],
                          [[p[0], p[1]] for p in run.density[lk][0]["anchors"]])
        _aio.run(go())

    def test_mock_runs_never_write_ledger(self):
        """mock 运行不写台账：假网关的 usage 是编的，写一次就污染跨运行状态
        （读侧在 run() 开头由 mock_seen 拦，写侧在 _save_density 拦）。"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, bench.DENSITY_LEDGER_FILE)
            seed = {"m|code@x": [{"density": 3.4, "source": {}, "seen": 1.0,
                                 "anchors": [[1000.0, 300.0], [2000.0, 600.0]]}]}
            bench._ledger_save(path, seed)
            run = BenchRun({}, results_dir=td)
            run.mock_seen = True
            # 假网关的密度（两点一条目：_ledger_load 会丢单锚点条目）
            run.density = {"m|code@x": [{"density": 1.0, "source": {},
                                         "seen": 2.0,
                                         "anchors": [[9999.0, 9999.0],
                                                     [19998.0, 19998.0]]}]}
            run._save_density()
            self.assertEqual(bench._ledger_load(path), seed,
                             "mock 运行的密度不得落盘")
            run.mock_seen = False
            run._save_density()
            self.assertEqual(bench._ledger_load(path), run.density,
                             "真运行的密度应正常落盘（守卫不是永关）")

    def test_no_results_dir_skips_density_persistence(self):
        """无 results_dir（内嵌调用/MCP 临时跑）：不写也不炸。"""
        run = BenchRun({})
        run.density = {"m|code@x": [{"density": 3.4, "source": {}, "seen": 1.0,
                                    "anchors": [[1000.0, 300.0],
                                                [2000.0, 600.0]]}]}
        self.assertIsNone(run._density_path)
        run._save_density()      # 不得抛

    def test_load_save_roundtrip_and_tolerates_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, bench.DENSITY_LEDGER_FILE)
            table = {"m|code@abc": [{"density": 3.4,
                                     "source": {"deploy": "x"},
                                     "seen": 1.0,
                                     "anchors": [[1000.0, 300.0],
                                                 [20000.0, 6000.0]]}]}
            bench._ledger_save(path, table)
            self.assertEqual(bench._ledger_load(path), table)
            # 损坏/空/不存在：一律空表，不抛
            with open(path, "w") as f:
                f.write("{truncated")
            self.assertEqual(bench._ledger_load(path), {})
            self.assertEqual(bench._ledger_load(None), {})
            self.assertEqual(bench._ledger_load(os.path.join(td, "nope")), {})
            # 单锚点子表不装载（_ledger_interp 对单点本就返回 None）；
            # 缺 density 的子表无法参与归户匹配，同样丢掉
            with open(path, "w") as f:
                f.write('{"m|code@abc": [{"density": 3.4, "anchors": [[1000, 300]]}]}')
            self.assertEqual(bench._ledger_load(path), {})
            with open(path, "w") as f:
                f.write('{"m|code@abc": [{"anchors": [[1000, 300], [2000, 600]]}]}')
            self.assertEqual(bench._ledger_load(path), {})
            # v1 结构（键→裸锚点列表）不装载：不猜着迁移，自然失效重建
            with open(path, "w") as f:
                f.write('{"m|code@abc": [[1000, 300], [2000, 600]]}')
            self.assertEqual(bench._ledger_load(path), {})

    def test_pick_arbitrates_by_probe_density(self):
        """归户仲裁：probe 密度与子表标定密度相差 ≤5% 归同一张（probe nonce
        固有抖动 ≤1.7%，不会误判）；超 5% 新建子表（真换分词器的量级）。"""
        subs = [{"density": 3.46, "source": {"deploy": "fastllm"},
                 "seen": 1.0, "anchors": [[1000.0, 300.0]]}]
        # 同分词器（探针读数小抖）→ 归户，不新建
        idx, new = bench._ledger_pick(subs, 3.50, {"deploy": "fastllm"}, 5.0)
        self.assertEqual((idx, new), (0, False))
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["seen"], 5.0, "归户要刷新 seen（LRU 依据）")
        # 换分词器（差 20%）→ 新建，**旧表保留**（切回该后端时直接复用）
        idx, new = bench._ledger_pick(subs, 2.87, {"deploy": "lvllm"}, 9.0)
        self.assertEqual((idx, new), (1, True))
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0]["density"], 3.46, "旧表不得被覆写或作废")
        self.assertEqual(subs[1]["source"]["deploy"], "lvllm", "新表记来源")

    def test_pick_source_does_not_overwrite(self):
        """来源只补空缺不覆写：后续换了机器仍归同一表时，覆写会把来源说成
        一个从未建过这张表的地址，事后查污染反而误导。"""
        subs = [{"density": 3.46, "source": {"deploy": "fastllm"},
                 "seen": 1.0, "anchors": [[1.0, 1.0]]}]
        bench._ledger_pick(subs, 3.47, {"deploy": "lvllm"}, 2.0)
        self.assertEqual(subs[0]["source"]["deploy"], "fastllm")

    def test_pick_caps_subtables_lru(self):
        """子表上限 3（拍板值）：满员后淘汰最久未用（seen 最小），保留近期
        真在用的部署——fallback 切走再切回的两边都是热的。"""
        subs = [{"density": d, "source": {"deploy": n}, "seen": s, "anchors": []}
                for d, n, s in ((3.4, "a", 1.0), (2.8, "b", 2.0), (1.9, "c", 3.0))]
        idx, new = bench._ledger_pick(subs, 4.9, {"deploy": "d"}, 9.0)
        self.assertTrue(new)
        self.assertEqual(len(subs), bench.DENSITY_LEDGER_MAX_SUBTABLES)
        self.assertNotIn("a", [s["source"]["deploy"] for s in subs],
                         "seen 最小的应先被淘汰")
        self.assertIn("d", [s["source"]["deploy"] for s in subs])

    def test_note_uses_probe_density_not_observation_density(self):
        """归户只认本轮 probe 密度，不认观测自身密度：实测同机器自己的观测在
        累计密度口径下跨 30~50%（小档被模板稀释、大档遇密度阶跃），用观测密度
        逐条归户会把一轮测速撕成好几张表。"""
        run = BenchRun({"model_info": {"m": {"deploy_label": "dep1"}}})
        run._probe_density[("m", "code")] = 3.46     # probe 指纹归既有表
        lk = bench._ledger_key("m", "code")
        run.density[lk] = [{"density": 3.46, "source": {}, "seen": 1.0,
                            "anchors": [[113473.0, 32947.0]]}]
        # 大档观测自身密度 2.55（与标定差 26%），仍应归同一张表
        run._ledger_note(lk, 334000, 131100)
        self.assertEqual(len(run.density[lk]), 1,
                         "同分词器一轮内的观测不得因自身密度差异拆表")
        self.assertEqual(len(run.density[lk][0]["anchors"]), 2)
        self.assertEqual(run._pending_ledger_notes, [], "归户不应播报")

    def test_note_announces_new_subtable(self):
        """probe 密度与所有已知口径都差超容差 → 新建子表并播报（影响全矩阵
        读数的环境事件不得静默）。"""
        run = BenchRun({"model_info": {"m": {"deploy_label": "lvllm-x"}},
                        "gateway_url": "http://h:1"})
        run._probe_density[("m", "code")] = 2.10     # 与已知 3.46 差 39%
        lk = bench._ledger_key("m", "code")
        run.density[lk] = [{"density": 3.46, "source": {}, "seen": 1.0,
                            "anchors": [[113473.0, 32947.0]]}]
        run._ledger_note(lk, 113473, 54000)
        self.assertEqual(len(run.density[lk]), 2)
        self.assertEqual(len(run._pending_ledger_notes), 1)
        msg = run._pending_ledger_notes[0]
        self.assertIn("新建子表", msg)
        self.assertIn("lvllm-x", msg, "播报要带来源，才能看出是分词器还是改道")

    def test_note_without_probe_density_falls_back_to_observation(self):
        """无 probe 密度的调用路径退回**观测自身密度**归户（宁可多建表不合表），
        而密度 <1 字符/token 的观测直接丢弃——子表的标定密度决定后续所有观测
        归到哪张表，拿一个不合理的数当标定值会把整键的归户带偏。"""
        run = BenchRun({})
        lk = bench._ledger_key("m", "code")
        run.density[lk] = [{"density": 3.47, "source": {}, "seen": 1.0,
                            "anchors": [[113538.0, 32947.0]]}]
        # ① 观测密度 3.47（与既有子表同口径）→ 归户，不新建也不播报
        run._ledger_note(lk, 113857, 32800)
        self.assertEqual(len(run.density[lk]), 1, "同口径不得新建子表")
        self.assertEqual(run._pending_ledger_notes, [])
        # ② 观测密度 2.11（差 39%）→ 新建（无 probe 时只能拿它当口径代表）
        run._ledger_note(lk, 113857, 54000)
        self.assertEqual(len(run.density[lk]), 2, "超容差应新建子表")
        self.assertEqual(len(run._pending_ledger_notes), 1)
        # ③ 密度 0.25（<1）→ 丢弃：不新建、不写入、不再播报
        run._ledger_note(lk, 100, 400)
        self.assertEqual(len(run.density[lk]), 2, "越界密度不配当标定值")
        self.assertEqual(len(run._pending_ledger_notes), 1, "丢弃路径不得发播报")

    def test_construction_prefers_covering_subtable(self):
        """构造取「覆盖目标且锚点最多」的子表，而不是本轮 probe 新建的空表：
        probe 切回旧分词器时，历史同口径表就该直接复用（分表而非作废的目的）。"""
        run = BenchRun({})
        lk = bench._ledger_key("m", "code")
        run.density[lk] = [
            {"density": 2.1, "source": {}, "seen": 9.0, "anchors": []},      # 本轮新表（空）
            {"density": 3.4, "source": {}, "seen": 1.0,
             "anchors": [[100000.0, 31250.0], [200000.0, 102678.0]]},        # 历史同口径
        ]
        run._probe_density[("m", "code")] = 2.1
        got = {}

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False,
                           est_tokens=None, img_tokens=0, **kw):
            got["chars"] = len(msgs[1]["content"])
            return {"req": req_i, "err": None, "first_abs": 1.0, "last_abs": 2.0,
                    "out_tokens": 10, "prompt_tokens": ctx, "usage_real": True,
                    "sent_chars": 10 ** 9, "cache_hit": 0, "cache_reported": False,
                    "finish": "length", "ttft_s": 0.1, "ttft_net_s": 0.1,
                    "decode_tok_s": 10.0, "total_s": 1.0, "stall_s": None,
                    "stall_count": None, "max_gap_s": None, "decode_burst": None,
                    "tok_per_chunk": 1.0, "prefill_tok_s": 100.0,
                    "prefill_net_tok_s": 100.0, "text_chars": 0, "reason_chars": 0}

        async def noop(ev):
            return None
        run._one = fake_one
        run.emit = noop
        import asyncio as _aio
        _aio.run(run._run_point(None, "m", "code", 60000, 1, 1.8))
        self.assertLess(got["chars"], 160000,
                        "应用历史表的塔陷段斜率（旧新表均无料时不得乱猜）")

    def test_construction_prefers_ledger_over_single_anchor(self):
        """有台账时构造走分段内插，而不是旧单锚点+滑动斜率：这是 64K 档
        +40% 的成因所在（旧公式用邻段斜率外推跨不过密度阶跃）。"""
        run = BenchRun({})
        key = ("m", "code")
        lk = bench._ledger_key("m", "code")
        self.assertIsNotNone(lk)
        # 台账：100K 字符=31250 tok，200K 字符=102678 tok（密度塔陷段 1.4 c/t）
        run.density[lk] = [{"density": 3.2, "source": {}, "seen": 1.0,
                            "anchors": [[100000.0, 31250.0],
                                        [200000.0, 102678.0]]}]
        # 旧记账故意设成普通段斜率 3.2：若被采纳，64K 会构造出 ~200K 字符
        run.prefix_kb[key] = (100000.0, 31250.0)
        run.cpt_marginal[key] = 3.2
        got = None

        async def fake_one(client, model, msgs, scenario, req_i, ctx, conc,
                           cpt, max_tokens, rep=0, quiet=False,
                           est_tokens=None, img_tokens=0, **kw):
            nonlocal got
            got = len(msgs[1]["content"])
            return {"req": req_i, "err": None, "first_abs": 1.0, "last_abs": 2.0,
                    "out_tokens": 10, "prompt_tokens": ctx, "usage_real": True,
                    "sent_chars": 10 ** 9, "cache_hit": 0, "cache_reported": False,
                    "finish": "length", "ttft_s": 0.1, "ttft_net_s": 0.1,
                    "decode_tok_s": 10.0, "total_s": 1.0, "stall_s": None,
                    "stall_count": None, "max_gap_s": None, "decode_burst": None,
                    "tok_per_chunk": 1.0, "prefill_tok_s": 100.0,
                    "prefill_net_tok_s": 100.0, "text_chars": 0, "reason_chars": 0}

        async def noop_emit(ev):
            return None

        run._one = fake_one
        run.emit = noop_emit
        asyncio.run(run._run_point(None, "m", "code", 60000, 1, 1.8))
        self.assertIsNotNone(got)
        # 内插值：100000 + (60000-31250)*100000/71428 ≈ 140560（+ 固定开销）
        self.assertLess(got, 160000,
                        "走了台账（塔陷段斜率）；若走旧公式会 >190000")
