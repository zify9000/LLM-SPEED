"""引擎纯逻辑回归测试：上下文构造（嵌套前缀/确定性/wrap/零输入）、复测聚合、错误语义化。

运行（任一方式，无需 pytest 也可跑）：
    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/ -v
"""
import unittest

from bench import (
    _aggregate_reps,
    _classify_http_error,
    _fill,
    _make_stream,
    _median,
    build_messages,
)


class TestFillStream(unittest.TestCase):
    """ADR-0022 确定性嵌套语料流的关键性质。"""

    def setUp(self):
        self.stream = _make_stream(["甲" * 100, "乙" * 200, "丙" * 300])

    def test_deterministic(self):
        self.assertEqual(self.stream, _make_stream(["甲" * 100, "乙" * 200, "丙" * 300]))

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


class TestBuildMessages(unittest.TestCase):
    def test_zero_input_branch(self):
        """ctx=0：不注入参考材料，一句话命题，填充字符数为 0（ADR-0016）。"""
        msgs, est, filler_n = build_messages("creative", 0, 1.8, "n")
        self.assertEqual(filler_n, 0)
        self.assertEqual(len(msgs), 2)
        self.assertIn("800字", msgs[1]["content"])
        self.assertGreater(est, 0)

    def test_explicit_filler_chars(self):
        """显式 filler_chars 精确控制填充长度（嵌套前缀记账路径）。"""
        _, _, n1 = build_messages("code", 4096, 3.4, "n", filler_chars=5000)
        self.assertEqual(n1, 5000)

    def test_nonzero_injects_filler_and_nonce(self):
        msgs, est, filler_n = build_messages("creative", 4096, 1.8, "nonce-42")
        self.assertGreater(filler_n, 0)
        self.assertIn("nonce-42", msgs[1]["content"])
        self.assertGreater(est, 4000 * 0.9)


class TestAggregateReps(unittest.TestCase):
    """ADR-0013 复测取中位：标量中位、decode 中位的一次、all_ok 多数决。"""

    def _rep(self, decode, all_ok=True):
        return {
            "all_ok": all_ok,
            "decode_tok_s": decode,
            "ttft_s": 1.0, "ttft_net_s": 0.9,
            "prefill_tok_s": 100.0, "prefill_net_tok_s": 110.0,
            "prompt_tokens": 4000, "out_tokens": 500,
            "cache_hit_tokens": 0, "reqs": [{"req": 0, "err": None}],
        }

    def test_median_of_odd(self):
        reps = [self._rep(50), self._rep(10), self._rep(30)]
        agg = _aggregate_reps(reps)
        self.assertEqual(agg["decode_tok_s"], 30)
        self.assertTrue(agg["all_ok"])
        self.assertEqual(agg["n_reps"], 3)
        # reqs 明细取 decode 中位的那一次
        self.assertEqual(agg["reqs"][0]["req"], 0)

    def test_majority_rule(self):
        ok = [self._rep(50), self._rep(10)]
        bad = self._rep(5, all_ok=False)
        self.assertTrue(_aggregate_reps(ok + [bad])["all_ok"])       # 2/3 多数
        self.assertFalse(_aggregate_reps([ok[0], bad, bad])["all_ok"])  # 1/3 少数

    def test_median_even(self):
        self.assertEqual(_median([1, 4]), 2.5)
        self.assertEqual(_median([4, 1, 9, 2]), 3.0)


class TestClassifyHttpError(unittest.TestCase):
    """ADR-0014/0020 错误语义化。"""

    def test_gateway_timeout(self):
        for code in (502, 503, 504):
            msg = _classify_http_error(code, "bad gateway")
            self.assertIn("网关", msg)

    def test_context_exceeded(self):
        self.assertIn("上下文窗口", _classify_http_error(
            400, '{"error":{"message":"context length exceeds limit"}}'))
        # Kimi 用 401 报超窗（ADR-0020）
        self.assertIn("上下文窗口", _classify_http_error(
            401, "invalid_authentication_error: input context exceeds the only allowed"))

    def test_plain_error_passthrough(self):
        msg = _classify_http_error(400, "unknown parameter: xyz")
        self.assertIn("400", msg)
        self.assertIn("xyz", msg)


if __name__ == "__main__":
    unittest.main()
