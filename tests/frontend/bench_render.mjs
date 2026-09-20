#!/usr/bin/env node
/* 前端明细表渲染基准 + 等价性回归（评审 backlog 性能项②）
   ------------------------------------------------------------------
   被测对象：static/index.html 的内联脚本里 renderResultTable() 的全量重绘路径
   （addPoint 每到达一个测点就调用一次；O(n²) 的来源）。

   本 harness 做三件事：
     1) 从 static/index.html 提取内联脚本（与 tests/check_frontend_js.sh 同口径：
        内容恰好是 <script> 的行 … 内容恰好是 </script> 的行），在 Node 里用
        **极小 DOM 桩**求值，验证整段脚本加载不抛（无 TDZ/ReferenceError）。
     2) 用真实存档的字段形状（读 results/ 下存档的点 key 集）造合成测点，
        对 N ∈ {16,32,64,128,256} 测：
          (a) 单次全量 renderResultTable() 的墙钟耗时；
          (b) O(n²) 累计耗时 = Σ_{k=1..N} 用前 k 个点全量渲染的耗时。
        **只计字符串构建**：#resultTables 是接受 innerHTML 赋值的桩元素，
        不解析 HTML、不做布局——即 DOM parse/layout 时间被排除（无 npm 依赖，
        无法在 Node 里跑真 DOM）。字符串构建是渲染的主要部分，但要如实标注。
     3) 等价性回归（渲染确定性 + 合帧不改变最终 HTML）：
          - 同一测点集渲染两次 → innerHTML 逐字节相同；
          - addPoint 合帧（rAF 合并）后落地的最终 innerHTML == 同步全量渲染；
          - 旧行为（每点同步全量渲染）的最终 innerHTML == 合帧后的最终 innerHTML；
          - 权威同步 renderAll() 作废排队帧后，再排空 rAF，最终 HTML 不变；
          - setPhase() 把排队帧落地后，最终 HTML == 同终态的直接全量渲染。
        任一等价性断言失败 → 退出码非 0。

   计时只用于报告，不做绝对 ms 断言（CI 负载敏感会 flake）。本脚本离线、
   零依赖、确定性（合成点与断言全确定），运行 < 20s。
   运行：bash tests/run_frontend_render.sh
*/
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import vm from "node:vm";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const HTML_PATH = resolve(ROOT, "static/index.html");

/* ------------------------------------------------------------------ 提取 */
/* 与 tests/check_frontend_js.sh 同口径：标签独占一行；恰好一个内联块 */
function extractInlineScript(html){
  const lines = html.split(/\r?\n/);
  const opens = [], closes = [];
  lines.forEach((l, i) => {
    if(l === "<script>") opens.push(i);
    if(l === "</script>") closes.push(i);
  });
  if(opens.length !== 1 || closes.length !== 1 || closes[0] < opens[0])
    throw new Error(`内联 <script> 提取失败：open=${opens.length} close=${closes.length}`);
  return { src: lines.slice(opens[0] + 1, closes[0]).join("\n"),
           firstLine: opens[0] + 2, lastLine: closes[0] };
}

/* -------------------------------------------------------------- DOM 桩 */
class StubEl {
  constructor(tag = "div"){
    this.tagName = String(tag).toUpperCase();
    this.hidden = false;
    this._html = "";
    this.style = {};
    this.dataset = {};
    this.value = "";
    this.options = [];
    this.disabled = false;
    this.checked = false;
    this.textContent = "";
    this.title = "";
    this.offsetWidth = 0;
    this.readyState = 0;
    this.children = [];
    const cls = new Set();
    this.classList = {
      add: (...c) => c.forEach(x => cls.add(x)),
      remove: (...c) => c.forEach(x => cls.delete(x)),
      contains: x => cls.has(x),
      toggle: (x, force) => {
        const on = force === undefined ? !cls.has(x) : !!force;
        on ? cls.add(x) : cls.delete(x);
        return on;
      },
    };
  }
  set innerHTML(v){ this._html = String(v); }
  get innerHTML(){ return this._html; }
  addEventListener(){}
  removeEventListener(){}
  appendChild(c){ this.children.push(c); return c; }
  querySelector(){ return null; }
  querySelectorAll(){ return []; }
  closest(){ return null; }
  remove(){}
  getAttribute(){ return null; }
  setAttribute(){}
  insertAdjacentHTML(){}
}

function createHarness(){
  const bySelector = new Map();
  const sel = s => {
    if(!bySelector.has(s)) bySelector.set(s, new StubEl());
    return bySelector.get(s);
  };
  const rafQueue = new Map();
  let rafNext = 1;
  const sandbox = {
    console,
    document: {
      body: new StubEl("body"),
      querySelector: sel,
      querySelectorAll: () => [],
      getElementById: id => sel(`#${id}`),
      createElement: tag => new StubEl(tag),
      addEventListener(){},
      removeEventListener(){},
    },
    fetch: () => new Promise(() => {}),   // 永不 resolve：init() 停在首个 await
    localStorage: { getItem: () => null, setItem(){}, removeItem(){} },
    getComputedStyle: () => ({ getPropertyValue: () => "" }),
    CSS: { escape: s => String(s) },
    requestAnimationFrame: cb => { const id = rafNext++; rafQueue.set(id, cb); return id; },
    cancelAnimationFrame: id => { rafQueue.delete(id); },
    setTimeout: () => 0,
    clearTimeout(){},
    setInterval: () => 0,
    clearInterval(){},
    EventSource: class { constructor(){ this.readyState = 0; } close(){} },
    confirm: () => false,
    alert(){},
    navigator: { userAgent: "node" },
    addEventListener(){},
    removeEventListener(){},
    dispatchEvent(){},
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  const drainRaf = () => {   // 排空 rAF 队列（回调内再排队的也执行）
    let guard = 0;
    while(rafQueue.size){
      if(++guard > 1000) throw new Error("rAF 队列疑似自激");
      const cbs = [...rafQueue.values()];
      rafQueue.clear();
      for(const cb of cbs) cb();
    }
  };
  return { sandbox, context, sel, rafQueue, drainRaf };
}

/* -------------------------------------------------------- 合成测点 */
/* 字段形状抄自 results/ 下真实存档的 key 并集（模型/场景/上下文档/并发同
   真实矩阵）。点按 allPoints() 的排序键生成，保证追加即为组内末位。 */
const SCEN_ORDER = { creative: 0, code: 1, translate: 2, agent: 3 };
const MODELS = ["DeepSeek-V4-Flash", "Qwen3.8-Flash-Next", "Qwen3.8-27B-v100", "Qwen3.8-Flash-v100"];
const LADDER = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];   // 0…128K
const AGENT_CTX = [0, 4096, 16384, 65536, 131072];
const AGENT_INST = [64, 128, 256];
const CONCS = [1, 5];

function makePoint(model, scenario, ctxTarget, instTokens, conc, modelIdx, seq){
  const isAgent = instTokens != null;
  const prompt = isAgent ? ctxTarget + instTokens + 59 : ctxTarget + 137;
  const out = 256;
  const concTotal = 22 + modelIdx;          // 批级总吞吐（各模型略有差异）
  const dec = +(concTotal / conc).toFixed(1);
  // 多数决会翻转的混合：偶数模型成批交付（MTP 类），奇数模型逐 token
  const tpc = isAgent ? (modelIdx % 2 === 0 ? 2.4 : 1.02) : (modelIdx % 2 === 0 ? 2.7 : 1.01);
  const stalled = seq % 7 === 0;
  const burst = seq % 23 === 0;
  const anomaly = seq > 0 && seq % 61 === 0;
  const p = {
    model, scenario, kind: "llm",
    ctx_target: ctxTarget, concurrency: conc,
    all_ok: true,
    prompt_tokens: prompt, out_tokens: out,
    ttft_s: +(prompt / (900 + modelIdx * 20)).toFixed(3),
    ttft_net_s: +(prompt / (940 + modelIdx * 20)).toFixed(3),
    prefill_tok_s: 900 + modelIdx * 20,
    prefill_net_tok_s: 940 + modelIdx * 20,
    prefill_conc_tok_s: 900 + modelIdx * 20,
    prefill_span_s: +(prompt / (900 + modelIdx * 20)).toFixed(3),
    decode_tok_s: dec, decode_tok_s_adj: dec,
    decode_conc_tok_s: concTotal,
    decode_total_tok_s: concTotal, decode_total_tok_s_adj: concTotal,
    decode_time_s: +(out / dec).toFixed(1),
    decode_span_s: +(out / concTotal).toFixed(1),
    total_span_s: +(prompt / (900 + modelIdx * 20) + out / concTotal).toFixed(2),
    total_s: isAgent ? +(prompt / (900 + modelIdx * 20) + out / concTotal).toFixed(2) : null,
    batch_time_s: +(prompt / (900 + modelIdx * 20) + out / concTotal).toFixed(2),
    tok_per_chunk: tpc,
    decode_peak_tok_s: Math.round(concTotal * 1.4),
    rtt_ms: 12, finish: "stop",
    cache_hit_tokens: isAgent && ctxTarget > 0 ? Math.round(ctxTarget * 0.9) : 0,
    cache_reported: isAgent ? true : null,
    stall_s: stalled ? 1.2 : 0,
    stall_count: stalled ? 2 : 0,
    max_gap_s: stalled ? 1.4 : 0.2,
    decode_burst: burst,
    flush_tok: stalled ? 48 : null,
    flush_count: stalled ? 1 : 0,
    flush_s: stalled ? 1.2 : 0,
    n_reps: 1, reps: [], reps_discarded: [],
    reqs: Array.from({ length: conc }, (_, i) => ({
      req: i + 1, out_tokens: out, prompt_tokens: prompt,
      ttft_s: +(prompt / (900 + modelIdx * 20)).toFixed(3),
      decode_tok_s: dec, finish: "stop", err: null,
      cache_hit: isAgent ? Math.round(ctxTarget * 0.9 / conc) : 0,
      cache_miss: isAgent ? Math.round((ctxTarget * 0.1 + instTokens) / conc) : prompt,
    })),
  };
  if(isAgent){ p.inst_tokens = instTokens; p.inst_real_tokens = instTokens - 8; }
  else p.inst_tokens = null;
  if(anomaly){
    p.anomaly = "early_stop";
    p.reps_discarded = [{ anomaly: "early_stop", rep: 1, out_tokens: 3, finish: "stop",
      decode_tok_s: 4.2, text_sample: "……（截断样本 <b>&amp;</b>）" }];
  }
  return p;
}

function buildUniverse(){
  const out = [];
  let seq = 0;
  MODELS.forEach((m, mi) => {
    for(const [sc, ctxs, insts] of [["creative", LADDER, [null]], ["code", LADDER, [null]], ["agent", AGENT_CTX, AGENT_INST]]){
      for(const c of ctxs) for(const inst of insts) for(const cc of CONCS)
        out.push(makePoint(m, sc, c, inst, cc, mi, seq++));
    }
  });
  out.sort((a, b) =>
    a.model.localeCompare(b.model)
    || (SCEN_ORDER[a.scenario] ?? 9) - (SCEN_ORDER[b.scenario] ?? 9)
    || a.ctx_target - b.ctx_target
    || (a.inst_tokens ?? 0) - (b.inst_tokens ?? 0)
    || a.concurrency - b.concurrency);
  return out;
}

/* ------------------------------------------------------------ 主流程 */
const failures = [];
function check(name, cond, detail){
  if(cond) console.log(`  ✓ ${name}`);
  else { failures.push(name); console.log(`  ✗ ${name}${detail ? " — " + detail : ""}`); }
}
function fp(html){
  return { len: html.length, hash: createHash("sha256").update(html).digest("hex").slice(0, 16) };
}
const MAIN_ROW_RE = /<tr class="(?:err)?">/g;

const html = readFileSync(HTML_PATH, "utf8");
const { src, firstLine, lastLine } = extractInlineScript(html);

const H = createHarness();
/* ---- 整段脚本加载（等价于 tests/check_frontend_js.sh 的 node --check + 真求值） */
try {
  vm.runInContext(src, H.context, { filename: "static/index.html:inline-script" });
  console.log(`load: OK — 内联脚本（行 ${firstLine}–${lastLine}）在 Node + DOM 桩下整段求值无抛出`);
} catch (e) {
  console.error("load: FAILED — 整段脚本求值抛错：");
  console.error(e && e.stack || e);
  process.exit(1);
}
vm.runInContext(`globalThis.__api = {
  state, renderAll, renderResultTable, addPoint, pointKey, allPoints,
  scheduleRender, flushScheduledRender, cancelScheduledRender, setPhase,
  hasMultiConc, tpcRunVote,
  pendingRender: () => _renderRaf !== 0
};`, H.context);
const api = H.sandbox.__api;
const resultEl = H.sel("#resultTables");
const btnAnomalyLog = H.sel("#btnAnomalyLog");

function resetState(phase = "run", pts = [], running = true){
  api.state.points.clear();
  api.state.cmp = null;
  api.state.phase = phase;
  api.state.running = running;
  api.state.runId = "bench-synth";
  api.state.totalPoints = pts.length;
  api.state.donePoints = pts.length;
  api.state.scenMeta = { creative: { kind: "llm" }, code: { kind: "llm" }, agent: { kind: "llm" } };
  for(const p of pts) api.state.points.set(api.pointKey(p), p);
}
function timeMs(fn){
  const t0 = performance.now();
  fn();
  return performance.now() - t0;
}
function median(xs){
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

const universe = buildUniverse();
console.log(`synthetic points: universe=${universe.length}（models=${MODELS.length} scenarios=creative/code/agent ctx=0…128K conc=1/5；字段形状抄自 results/ 存档 key 并集）`);
console.log("measure: renderResultTable() 字符串构建 only（#resultTables 为桩元素，innerHTML 赋值不解析/不布局；DOM parse/layout 未计）");

/* step 1: 计时表 */
const NS = [16, 32, 64, 128, 256];
const REPEAT = 3;
const results = [];
console.log("");
console.log("   N   single(ms)   cumulative(ms)   rows(=Σk)");
for(const N of NS){
  const pts = universe.slice(0, N);
  resetState("run", pts);
  api.renderResultTable();   // 预热
  const single = median(Array.from({ length: REPEAT }, () => {
    resetState("run", pts);
    return timeMs(() => api.renderResultTable());
  }));
  const cumRuns = Array.from({ length: REPEAT }, () => {
    api.state.points.clear();
    let acc = 0;
    for(let k = 1; k <= N; k++){
      const p = pts[k - 1];
      api.state.points.set(api.pointKey(p), p);
      acc += timeMs(() => api.renderResultTable());
    }
    return acc;
  });
  const cumulative = median(cumRuns);
  results.push({ N, single, cumulative });
  console.log(`  ${String(N).padStart(3)}   ${single.toFixed(2).padStart(9)}   ${cumulative.toFixed(1).padStart(13)}   ${N * (N + 1) / 2}`);
}
const INCREMENTAL_THRESHOLD_MS = 150;
const cum128 = results.find(r => r.N === 128).cumulative;

/* ---- 等价性（固定合成点集，确定性） */
const equivPts = universe.filter((_, i) => i % 3 === 0).slice(0, 72);
console.log("");
console.log(`equivalence: fixed set = ${equivPts.length} points（every 3rd of universe，含 creative/code/agent）`);

/* 断言 0：bumpProgress 与 KPI 回写仍在 addPoint 内**同步**生效（不参与合帧） */
resetState("run", []);
api.state.totalPoints = 10;
api.state.donePoints = 0;
const kpiPointsEl = H.sel("#kpiPoints");
const progFillEl = H.sel("#progFill");
const kpiPreEl = H.sel("#kpiPrefill");
api.addPoint(universe[0]);
check("addPoint 同步刷新「已完成测点」KPI（bumpProgress 未被合帧吞掉）",
  kpiPointsEl.innerHTML === "1 <small>/ 10</small>", kpiPointsEl.innerHTML);
check("addPoint 同步推进进度条宽度", progFillEl.style.width === "10%", String(progFillEl.style.width));
check("addPoint 同步回写 Prefill KPI", /tok\/s<\/small>/.test(kpiPreEl.innerHTML), kpiPreEl.innerHTML);
H.drainRaf();   // 清掉上面 addPoint 排的帧，保证后续断言从干净队列开始

/* 断言 1：同一集合渲染两次逐字节相同 */
resetState("run", equivPts);
api.renderResultTable();
const h1 = resultEl.innerHTML;
api.renderResultTable();
const h2 = resultEl.innerHTML;
const f1 = fp(h1);
console.log(`  render #1: len=${f1.len} sha256=${f1.hash}`);
check("同一测点集渲染两次 innerHTML 逐字节相同", h1 === h2);

/* 断言 2：addPoint 合帧落地 == 同步全量渲染；且旧行为逐点同步渲染最终值相同 */
resetState("run", []);
for(const p of equivPts) api.addPoint(p);
const pendingAfterAdd = api.pendingRender();
check("addPoint 后确有排队中的合帧渲染（未被同步渲染）", pendingAfterAdd);
H.drainRaf();
const hCoalesced = resultEl.innerHTML;
const fCoal = fp(hCoalesced);
console.log(`  coalesced rAF final: len=${fCoal.len} sha256=${fCoal.hash}`);
check("合帧落地后队列已清空", !api.pendingRender());

resetState("run", equivPts);
api.renderResultTable();
const hFull = resultEl.innerHTML;
check("合帧最终 innerHTML == 同步全量渲染 innerHTML", hCoalesced === hFull, `coalesced=${fp(hCoalesced).hash} full=${fp(hFull).hash}`);

/* 旧行为（pre-change）：每到达一点同步 renderAll（等价于旧 addPoint → renderAll） */
resetState("run", []);
for(const p of equivPts){
  if(!api.state.points.has(api.pointKey(p))) api.state.donePoints++;
  api.state.points.set(api.pointKey(p), p);
  api.renderAll();
}
const hLegacy = resultEl.innerHTML;
check("旧行为（逐点同步全量）最终 innerHTML == 合帧最终 innerHTML", hLegacy === hCoalesced, `legacy=${fp(hLegacy).hash} coalesced=${fp(hCoalesced).hash}`);
check("每个测点都渲染出且仅一行主行", (hFull.match(MAIN_ROW_RE) || []).length === equivPts.length,
  `rows=${(hFull.match(MAIN_ROW_RE) || []).length} points=${equivPts.length}`);

/* 断言 3：权威同步 renderAll() 作废排队帧，之后排空 rAF 不再改变最终 HTML */
resetState("run", []);
for(const p of equivPts) api.addPoint(p);
check("权威 renderAll 前仍有排队帧", api.pendingRender());
api.renderAll();
const hTerminal = resultEl.innerHTML;
check("权威 renderAll 已作废排队帧", !api.pendingRender());
H.drainRaf();   // 若作废失效，这里会触发一次重绘；结果应仍一致
check("renderAll 后 rAF 不再改写最终 HTML", resultEl.innerHTML === hTerminal && hTerminal === hFull);

/* 断言 4：setPhase 把排队帧落地（终态闸生效） */
resetState("run", []);
for(const p of equivPts) api.addPoint(p);
api.state.running = false;
api.setPhase("done");   // 内部 flushScheduledRender 落地排队帧
check("setPhase 后排队帧已落地", !api.pendingRender());
const hPhase = resultEl.innerHTML;
resetState("done", equivPts, false);
api.renderResultTable();
const hFullDone = resultEl.innerHTML;
check("setPhase 落地结果 == 终态（done，retest 闸开）直接全量渲染", hPhase === hFullDone,
  `phase=${fp(hPhase).hash} full=${fp(hFullDone).hash}`);

/* ------------------------------------------------------------ 报告 */
console.log("");
console.log("============================================================");
const below = cum128 < INCREMENTAL_THRESHOLD_MS;
console.log(`verdict: cumulative O(n²) @N=128 = ${cum128.toFixed(1)} ms（字符串构建，median×${REPEAT}）`
  + ` ${below ? "<" : ">"} ${INCREMENTAL_THRESHOLD_MS} ms 门槛`);
console.log(below
  ? "  → 未过线：O(n²) 在现实规模（多模型/多并发运行 ≈120+ 点）实测可接受；只做 rAF 合帧，不实现增量追加快路径。"
  : "  → 过线：O(n²) 值得做增量追加；本 harness 的等价性断言是回归保护。");
console.log("  说明：仅计字符串构建；innerHTML 赋值的 DOM parse/layout 成本未计（Node 无真 DOM，零依赖约束）。");
console.log("============================================================");
if(failures.length){
  console.log(`equivalence: FAILED — ${failures.length} 项断言未通过：${failures.join("；")}`);
  process.exit(1);
}
console.log("equivalence: all assertions passed — 渲染确定、合帧不改变最终 HTML、终态不被排队帧吞掉。");
