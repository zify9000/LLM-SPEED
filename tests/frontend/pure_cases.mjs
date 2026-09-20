/* 展示层 §pure 区域用例（无 import：test/assert 由 run_frontend_pure.sh 注入）。
   点形状取自真实存档：llm conc=1 / llm conc=5（批级 6 字段齐全）/
   agent 缓存×指令矩阵（inst_tokens），字段名与量级不臆造。
   展示层契约（单/多发分叉、口径三级回退、多数决）见设计文档测量口径一节。 */

// 真实单并发 LLM 点
const CONC1 = {model:"Qwen3.8-Flash-Next", scenario:"code", kind:"llm", ctx_target:0,
  concurrency:1, all_ok:true, prompt_tokens:131, ttft_s:0.78, prefill_tok_s:167.9,
  decode_tok_s:39.2, decode_total_tok_s:39.2, out_tokens:256, decode_time_s:6.54,
  ttft_net_s:0.764, prefill_net_tok_s:171.4, decode_tok_s_adj:39.2,
  decode_total_tok_s_adj:39.2, prefill_span_s:0.78, total_span_s:7.32,
  decode_span_s:6.54, prefill_conc_tok_s:168.2, decode_conc_tok_s:39.2,
  decode_peak_tok_s:48.3, tok_per_chunk:null, decode_burst:null};
// 真实多并发 LLM 点（批级 6 字段齐全）
const CONC5 = {model:"Qwen3.8-Flash-Next", scenario:"code", kind:"llm", ctx_target:0,
  concurrency:5, all_ok:true, prompt_tokens:106, ttft_s:2.56, prefill_tok_s:207.0,
  decode_tok_s:30.4, decode_total_tok_s:130.3, out_tokens:256, decode_time_s:42.06,
  ttft_net_s:2.468, prefill_net_tok_s:214.7, decode_total_tok_s_adj:130.3,
  prefill_span_s:2.56, total_span_s:10.77, decode_span_s:8.21,
  prefill_conc_tok_s:207.4, decode_conc_tok_s:155.9, decode_peak_tok_s:229.3,
  tok_per_chunk:3.57, decode_burst:null};
// 合格投票点工厂：all_ok + out_tokens>=64 + 非 decode_burst + 有 decode_tok_s
const voter = (model, tpc, over = {}) => ({model, all_ok:true, out_tokens:256,
  decode_tok_s:30.4, decode_burst:null, tok_per_chunk:tpc, ...over});

/* ---------------- 闸一致性：hasMultiConc / hiConcOf ---------------- */

test("hasMultiConc：纯 conc=1 点集为 false，hiConcOf 为 null，不变式成立", () => {
  const pts = [CONC1, {...CONC1, ctx_target:512}, {...CONC1, ctx_target:4096}];
  assert.equal(hasMultiConc(pts), false);
  assert.equal(hiConcOf(pts), null);
  assert.equal(hasMultiConc(pts), hiConcOf(pts) != null);
});

test("hasMultiConc：三组不同规模的单并发点集都满足不变式", () => {
  const sets = [
    [{...CONC1}],
    [{...CONC1}, {...CONC1, ctx_target:512}],
    [{...CONC1}, {...CONC1, ctx_target:512}, {...CONC1, ctx_target:4096}],
  ];
  for(const pts of sets){
    assert.equal(hasMultiConc(pts), false);
    assert.equal(hiConcOf(pts), null);
    assert.equal(hasMultiConc(pts), hiConcOf(pts) != null);
  }
});

test("hasMultiConc：[1,1,5] → true 且 hiConcOf 返回 5", () => {
  const pts = [CONC1, {...CONC1, ctx_target:512}, CONC5];
  assert.equal(hasMultiConc(pts), true);
  assert.equal(hiConcOf(pts), 5);
  assert.equal(hasMultiConc(pts), hiConcOf(pts) != null);
});

test("hasMultiConc：混合 [1,5] → true/5，不变式成立", () => {
  const pts = [CONC1, CONC5];
  assert.equal(hasMultiConc(pts), true);
  assert.equal(hiConcOf(pts), 5);
  assert.equal(hasMultiConc(pts), hiConcOf(pts) != null);
});

test("hasMultiConc：纯 conc=5 单档也是多发（防旧 concs.length>1 误判单发）", () => {
  const pts = [CONC5, {...CONC5, ctx_target:512}];
  assert.equal(hasMultiConc(pts), true);
  assert.equal(hiConcOf(pts), 5);
});

test("hasMultiConc/hiConcOf：空集 → false/null", () => {
  assert.equal(hasMultiConc([]), false);
  assert.equal(hiConcOf([]), null);
});

test("hiConcOf：缺 concurrency 字段按 0 处理，不变式仍成立", () => {
  assert.equal(hiConcOf([{}]), null);
  assert.equal(hasMultiConc([{}]), false);
  const pts = [{concurrency:1}, {}, {concurrency:3}];
  assert.equal(hiConcOf(pts), 3);
  assert.equal(hasMultiConc(pts), true);
  assert.equal(hasMultiConc(pts), hiConcOf(pts) != null);
});

/* ---------------- 口径三级回退（按实现的实际优先级断言） ---------------- */

test("preOf：净口径（扣 RTT）优先，缺字段回退毛口径，全缺 undefined", () => {
  assert.equal(preOf({prefill_net_tok_s:214.7, prefill_tok_s:207.0}), 214.7);
  assert.equal(preOf({prefill_tok_s:207.0}), 207.0);
  assert.equal(preOf({prefill_net_tok_s:null, prefill_tok_s:207.0}), 207.0);
  assert.equal(preOf({}), undefined);
});

test("decNetOf：空窗校正 adj 优先，缺则原始窗口 raw", () => {
  assert.equal(decNetOf({decode_tok_s_adj:39.2, decode_tok_s:40.0}), 39.2);
  assert.equal(decNetOf({decode_tok_s:40.0}), 40.0);
  assert.equal(decNetOf({}), undefined);
});

test("decAvgOf：均口径 raw（Σ÷Σ）优先，缺则 adj（与 decNetOf 相反）", () => {
  assert.equal(decAvgOf({decode_tok_s:30.4, decode_tok_s_adj:31.0}), 30.4);
  assert.equal(decAvgOf({decode_tok_s_adj:31.0}), 31.0);
  assert.equal(decAvgOf({decode_tok_s:null, decode_tok_s_adj:31.0}), 31.0);
  assert.equal(decAvgOf({}), undefined);
});

test("decOf：KPI 权威回写别名 = decNetOf（净口径）", () => {
  const p = {decode_tok_s_adj:39.2, decode_tok_s:40.0};
  assert.equal(decOf(p), decNetOf(p));
  assert.equal(decOf(p), 39.2);
});

test("decTotalTimeOf：批级 decode_span_s 优先于 decode_time_s", () => {
  assert.equal(decTotalTimeOf({decode_span_s:8.21, decode_time_s:42.06}), 8.21);
});

test("decTotalTimeOf：无批级字段回退 Σ请求窗口 decode_time_s", () => {
  assert.equal(decTotalTimeOf({decode_time_s:42.06, out_tokens:256, decode_tok_s:30.4}), 42.06);
});

test("decTotalTimeOf：两级都缺时按 输出 ÷ 均速 反推（decTimeOf 路径）", () => {
  // 256 / 30.4 = 8.421… → round1 半步取整 = 8.4
  assert.equal(decTotalTimeOf({out_tokens:256, decode_tok_s:30.4}), 8.4);
  assert.equal(decTotalTimeOf({}), null);
});

test("decColAvgOf：批级 decode_conc_tok_s ÷ 并发数", () => {
  assert.equal(decColAvgOf(CONC5), 31.2);   // round1(155.9/5)
});

test("decColAvgOf：无批级字段或并发<=0 时回退 decAvgOf", () => {
  assert.equal(decColAvgOf({decode_tok_s:50.0}), 50.0);
  assert.equal(decColAvgOf({decode_conc_tok_s:100, concurrency:0, decode_tok_s:20.0}), 20.0);
  assert.equal(decColAvgOf({decode_conc_tok_s:100, decode_tok_s:20.0}), 20.0);
});

test("calcSpanTimeOf：total_span_s 优先（max 口径）", () => {
  assert.equal(calcSpanTimeOf({total_span_s:10.77, prefill_span_s:2.56, decode_time_s:42.06}), 10.77);
});

test("calcSpanTimeOf：无 total_span_s 时回退 prefill_span_s + decode_time_s", () => {
  assert.equal(calcSpanTimeOf({prefill_span_s:2.56, decode_time_s:42.06}), 44.62);
});

test("calcSpanTimeOf：两项不齐则 null（格显 -）", () => {
  assert.equal(calcSpanTimeOf({prefill_span_s:2.56}), null);
  assert.equal(calcSpanTimeOf({decode_time_s:42.06}), null);
  assert.equal(calcSpanTimeOf({}), null);
});

test("decTotalOf：decode_conc_tok_s → adj → raw 三级回退", () => {
  assert.equal(decTotalOf({decode_conc_tok_s:155.9, decode_total_tok_s_adj:130.3, decode_total_tok_s:130.0}), 155.9);
  assert.equal(decTotalOf({decode_total_tok_s_adj:130.3, decode_total_tok_s:130.0}), 130.3);
  assert.equal(decTotalOf({decode_total_tok_s:130.0}), 130.0);
  assert.equal(decTotalOf({}), undefined);
});

/* ---------------- 验算链闭合（design/measurement.md 要求的三条互验） ---------------- */

test("验算链：Decode 均速列 × Decode 总耗时 ≈ 输出 tokens（conc=5 真实点）", () => {
  // 多发表「均速 × Decode 总耗时 = 输出 tokens 列均值」是设计文档钉死的互验
  const lhs = decColAvgOf(CONC5) * decTotalTimeOf(CONC5);
  assert.ok(Math.abs(lhs - CONC5.out_tokens) < 0.5, `期望 ≈256，实际 ${lhs}`);
});

test("验算链：Prefill 总耗时 + Decode 总耗时 ≈ 计算总耗时", () => {
  // prefill_span_s + decode_span_s = total_span_s（批开始→全部请求末 token）
  const lhs = CONC5.prefill_span_s + CONC5.decode_span_s;
  assert.ok(Math.abs(lhs - CONC5.total_span_s) < 0.01, `期望 ≈${CONC5.total_span_s}，实际 ${lhs}`);
});

test("验算链：conc=1 点同样闭合（两式各列同源相等）", () => {
  const lhs = decColAvgOf(CONC1) * decTotalTimeOf(CONC1);
  assert.ok(Math.abs(lhs - CONC1.out_tokens) < 0.5, `期望 ≈256，实际 ${lhs}`);
  assert.ok(Math.abs(CONC1.prefill_span_s + CONC1.decode_span_s - CONC1.total_span_s) < 0.01);
});

/* ---------------- tpcRunVote 多数决 ---------------- */

test("tpcRunVote：3 个合格点里 2 个成批读数 → 严格过半成立", () => {
  const on = tpcRunVote([voter("A", 3.57), voter("A", 3.57), voter("A", 0.5)]);
  assert.ok(on.has("A"));
});

test("tpcRunVote：3 个合格点里仅 1 个成批读数 → 孤立单票不过半，不成立", () => {
  const on = tpcRunVote([voter("A", 3.57), voter("A", 0.5), voter("A", 0.5)]);
  assert.equal(on.has("A"), false);
});

test("tpcRunVote：2:2 平票不成立（必须严格过半）", () => {
  const on = tpcRunVote([voter("A", 3.57), voter("A", 3.57), voter("A", 0.5), voter("A", 0.5)]);
  assert.equal(on.has("A"), false);
});

test("tpcRunVote：all_ok=false / out_tokens<64 / decode_burst / 无 decode_tok_s 均无投票资格", () => {
  const pts = [
    voter("B", 3.57, {all_ok:false}),
    voter("B", 3.57, {out_tokens:63}),
    voter("B", 3.57, {decode_burst:true}),
    voter("B", 3.57, {decode_tok_s:null}),
    voter("B", 0.5),   // 唯一合格点，且不成批
  ];
  assert.equal(tpcRunVote(pts).has("B"), false);
});

test("tpcRunVote：单点运行 n=1 无法构成多数决证据，按实现不压制（读数如实展示）", () => {
  // 与函数注释「边界：单点运行（n=1）无法构成多数决证据，该点读数如实展示不压制」一致
  const on = tpcRunVote([voter("C", 3.57)]);
  assert.ok(on.has("C"));
});

test("tpcRunVote：不同模型独立裁定（Set 成员断言）", () => {
  const on = tpcRunVote([
    voter("A", 3.57), voter("A", 3.57), voter("A", 0.5),   // A 2/3 成立
    voter("B", 3.57), voter("B", 0.5),                     // B 1/2 不成立
  ]);
  assert.ok(on.has("A"));
  assert.equal(on.has("B"), false);
  assert.equal(on.size, 1);
});

test("tpcRunVote：门槛判定用 >=SPEC_MIN_TPC，低于门槛只计入分母", () => {
  const below = SPEC_MIN_TPC - 0.01;
  // 恰好等于门槛算成批票：2/3 → 成立
  assert.ok(tpcRunVote([voter("A", SPEC_MIN_TPC), voter("A", SPEC_MIN_TPC), voter("A", below)]).has("A"));
  // 全部低于门槛 → 0 票 → 不成立
  assert.equal(tpcRunVote([voter("A", below), voter("A", below), voter("A", below)]).has("A"), false);
});

/* ---------------- scenarioStats ---------------- */

test("scenarioStats：空输入不泄漏 -Infinity，各统计字段为 null", () => {
  const s = scenarioStats([]);
  assert.equal(s.preRange, null);
  assert.equal(s.decRange, null);
  assert.equal(s.decAvgRange, null);
  assert.equal(s.prePeak, null);
  assert.equal(s.preAvg, null);
  assert.equal(s.preUntrusted, 0);
  assert.equal(s.hiConc, null);
  assert.equal(s.hiTotal, null);
  assert.equal(s.hiPeak, null);
  assert.equal(s.totRange, null);
  assert.equal(s.peak, null);
  assert.equal(s.baseConc, undefined);
  assert.equal(s.totConc, undefined);
  assert.ok(!JSON.stringify(s).includes("Infinity"), `不应出现 ±Infinity：${JSON.stringify(s)}`);
});

test("scenarioStats：preOf 超 PRE_PHYS_GATE 的点剔出范围/均值并计入 untrusted", () => {
  const s = scenarioStats([
    {...CONC1, prefill_net_tok_s:100.0, prefill_tok_s:100.0},
    {...CONC1, ctx_target:512, prefill_net_tok_s:60000.0, prefill_tok_s:60000.0},
  ]);
  assert.equal(s.preUntrusted, 1);
  assert.deepEqual(s.preRange, [100, 100]);
  assert.equal(s.preAvg, 100);
});

test("scenarioStats：all_ok=false 的点整个不参与统计（其超界 prefill 也不计数）", () => {
  const s = scenarioStats([
    {...CONC1, prefill_net_tok_s:100.0, prefill_tok_s:100.0},
    {...CONC1, ctx_target:512, all_ok:false, prefill_net_tok_s:99999.0, prefill_tok_s:99999.0},
  ]);
  assert.equal(s.preUntrusted, 0);
  assert.deepEqual(s.preRange, [100, 100]);
});

test("scenarioStats：单并发集 hiConc=null、totConc=baseConc=1", () => {
  const s = scenarioStats([CONC1, {...CONC1, ctx_target:512}]);
  assert.equal(s.hiConc, null);
  assert.equal(s.baseConc, 1);
  assert.equal(s.totConc, 1);
});

test("scenarioStats：混合集 hiConc=5，totRange/hiTotal 取最高并发档", () => {
  const mx1 = {...CONC5, ctx_target:0, decode_conc_tok_s:100.0, decode_total_tok_s_adj:100.0, decode_total_tok_s:100.0};
  const mx2 = {...CONC5, ctx_target:512, decode_conc_tok_s:200.0, decode_total_tok_s_adj:200.0, decode_total_tok_s:200.0, decode_peak_tok_s:250.0};
  const s = scenarioStats([CONC1, {...CONC1, ctx_target:512}, mx1, mx2]);
  assert.equal(s.hiConc, 5);
  assert.equal(s.totConc, 5);
  assert.deepEqual(s.totRange, [100, 200]);
  assert.equal(s.hiTotal, 150);
  assert.equal(s.peak.v, 250);
  assert.equal(s.peak.ctx, 512);
});

/* ---------------- hitStateOf 缓存命中态 ---------------- */

test("hitStateOf：cache_reported=true → reported", () => {
  assert.equal(hitStateOf({inst_tokens:64, ctx_target:512, cache_reported:true}, null), "reported");
});

test("hitStateOf：点自身未回传但任一请求回传 cache_reported → reported", () => {
  assert.equal(hitStateOf({inst_tokens:64, ctx_target:512, reqs:[{cache_reported:true}]}, null), "reported");
});

test("hitStateOf：矩阵零缓存档（ctx_target=0）→ cold", () => {
  assert.equal(hitStateOf({inst_tokens:64, ctx_target:0, cache_hit_tokens:0}, null), "cold");
});

test("hitStateOf：有矩阵身份但 cache_hit_tokens 缺 → unknown", () => {
  assert.equal(hitStateOf({inst_tokens:64, ctx_target:512, cache_hit_tokens:null}, null), "unknown");
});

test("hitStateOf：有矩阵身份且 cache_hit_tokens 有值 → est（按预热实测估算）", () => {
  assert.equal(hitStateOf({inst_tokens:64, ctx_target:512, cache_hit_tokens:128}, null), "est");
});

test("hitStateOf：非 agent 点（无 turn 无 inst_tokens）→ null", () => {
  assert.equal(hitStateOf({model:"x", ctx_target:0}, null), null);
});

/* ---------------- 格式化 ---------------- */

test("fmtNum：null/undefined → '-'，其余四舍五入并带千分位", () => {
  assert.equal(fmtNum(null), "-");
  assert.equal(fmtNum(undefined), "-");
  assert.equal(fmtNum(0), "0");
  assert.equal(fmtNum(1234.6), "1,235");
  assert.equal(fmtNum(999), "999");
});

test("round1：保留一位小数（半步向上）", () => {
  assert.equal(round1(1.25), 1.3);
  assert.equal(round1(1.24), 1.2);
  assert.equal(round1(3.14159), 3.1);
});

test("provUrlText：单地址去 scheme；多地址显示「首选 +N」；空址为 ''", () => {
  assert.equal(provUrlText({gateway_url:"https://api.example.com/v1"}), "api.example.com/v1");
  assert.equal(provUrlText({gateway_urls:["https://a.example/v1", "https://b.example/v1", "https://c.example/v1"]}), "a.example/v1 +2");
  assert.equal(provUrlText({}), "");
  assert.equal(provUrlText({gateway_urls:[]}), "");
});

test("cmpLegendKey：单并发档不带后缀，多并发档带并发后缀", () => {
  assert.equal(cmpLegendKey("code", "decode", null), "cmp::code::decode");
  assert.equal(cmpLegendKey("code", "decode", undefined), "cmp::code::decode");
  assert.equal(cmpLegendKey("code", "decode", 5), "cmp::code::decode::5");
});
