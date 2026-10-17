"""KPI 赛道（`.trk` 横贯轨）几何/类名契约目检替身（可选脚本，**不进回归套件**）。

为什么需要它：赛道是懒建的 DOM + 纯 CSS 档位，只在**真浏览器**里成立，三条 node 侧
门禁都看不到它——渲染 harness 的 DOM 桩里 `querySelector` 返回 null（`trkSet` 对桩
等于无操作，写不炸也验不到），§pure 只测纯函数（`trkFrac` 的分段线性），语法门禁只看
写法。`.trk-rail/.trk-fill/.trk-tick/.trk-num/.trk-actor` 的实际结构、`t0..t4` 档位类名、
`.spark` 可见颗数、升档 pop/冲线 `.burst` 的一次性生命周期，全都只有真 DOM 才有。

它同时守着旧时速表踩过的坑（见 `.memoZ/pitfalls/crosscut/02-findindex-minus-one-folded.md`）：
超量程时 `Array.findIndex` 返回 -1，若被 `Math.max(0, ·)` 折成 0，**超过最大档位反而
退回第一档**（actor 从 🚄 变回 🚶，语义正好反了）——逐值喂到 900（> decode 上界 300）
就是在拦这个回归。

位置契约：`fill.style.width` 与 `trk-actor.style.left` 必须都等于
`(100 * trkFrac(v, tiers)).toFixed(2) + "%"`（数值与位置同源，不许各算各的）。注意 Chrome
的 CSSOM 会把 `"20.00%"` 序列化成 `"20%"`（尾零不保留），所以期望串也先过一遍同一个
CSSOM 再比字符串（脚本里的 `canon`）。

检查站双色：`.trk-tick` 按 `i < now` 切 `.past`（已过站 = 卡色缝 `var(--card)`，未到站 =
灰 `color-mix(ink-4 40%, --track)`），所以已过站条数 == 档位号、空读数为 0；两色的
computed 值只在真渲染里有解析值，脚本挂一个探针量 `--card` 的解析色再逐个刻线比对
（防"灰缝写成同名同色"，那种错类名断言看不出来）。

跑法（需要 playwright + chromium；没装就跳过并返回 0，不影响 tests/ 的任何门禁）：

    python3 scripts/ui_smoke_track.py

只读仓库 `static/index.html`（file:// 直开，不起服务、不碰 results/ 存档）。
"""
import pathlib
import sys

try:
    from playwright.sync_api import sync_playwright
except Exception as _exc:                       # 缺依赖 = 跳过，不算失败
    print(f"SKIP：需要 playwright + chromium（{_exc}）")
    raise SystemExit(0)

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "file://" + str(ROOT / "static" / "index.html")
TIER = {"prefill": [500, 1000, 2000, 5000, 10000], "decode": [10, 30, 50, 100, 300]}
NAMES = ["走路", "跑步", "自行车", "小汽车", "高铁"]
ACTORS = ["🚶", "🏃", "🚴", "🚗", "🚄"]
SPARKS = [0, 2, 4, 5, 6]                        # 可见粒子数：走路无，档越高越多
# 喂值样本写在 CHECKS 里（页面侧算 trkFrac）：decode 逐值到 900（> 上界 300）、prefill 抽测到 20000

CHECKS = r"""() => {
  const TIERS = {prefill: [500,1000,2000,5000,10000], decode: [10,30,50,100,300]};
  const VALS = [0,5,10,20,41,100,150,260,300,900], PRE_VALS = [0,500,1000,7500,20000];
  const probe = document.createElement('span');
  // Chrome 的 CSSOM 序列化会去掉尾零："20.00%" → "20%"。期望串也过一遍同一序列化再比字符串
  const canon = s => { probe.style.width = s; return probe.style.width; };
  const wantPct = (v, ts) => canon((100 * trkFrac(v, ts)).toFixed(2) + "%");
  const tierIdx = (v, ts) => { const i = ts.findIndex(t => v <= t); return i < 0 ? ts.length - 1 : i; };
  const numTxt = v => v >= 1000 ? `${v / 1000}K` : String(v);
  const bursts = el => [...el.querySelectorAll('.burst')];
  const visibleSparks = el => [...el.querySelectorAll('.spark')]
    .filter(s => getComputedStyle(s).display !== 'none').length;
  const ticks = el => [...el.querySelectorAll('.trk-tick')];
  const pastTicks = el => ticks(el).filter(t => t.classList.contains('past')).length;
  /* --card 的解析值（computed 颜色串）：自定义属性只在挂进文档的元素上解析，探针挂进同一子树
     保证继承上下文一致；position:absolute 不扰动轨道布局，量完即删 */
  const cardColor = host => {
    const p = document.createElement('span');
    p.style.cssText = 'position:absolute;left:-999px;top:-999px;background-color:var(--card)';
    host.appendChild(p);
    const c = getComputedStyle(p).backgroundColor;
    p.remove();
    return c;
  };
  const snap = (trk, kpi) => {
    const veh = trk.querySelector('.trk-actor'), actor = trk.querySelector('.actor');
    const tn = kpi.querySelector('.tname');
    return {cls: trk.className, width: trk.querySelector('.trk-fill').style.width,
            left: veh.style.left, actor: actor && actor.textContent, tname: tn && tn.textContent,
            tnameCls: tn && tn.className, pop: veh.classList.contains('pop'),
            sparks: visibleSparks(trk), burst: bursts(trk).length, past: pastTicks(trk)};
  };
  const struct = (trk, ts) => ({
    rail: trk.querySelectorAll('.trk-rail').length,
    fill: trk.querySelectorAll('.trk-fill').length,
    actorNest: !!trk.querySelector('.trk-actor > .aveh > .actor'),
    actorSparks: trk.querySelectorAll('.trk-actor > .aveh > .spark').length,
    streak: trk.querySelectorAll('.trk-actor .streak').length,
    ticks: [...trk.querySelectorAll('.trk-tick')].map(e => +parseFloat(e.style.left).toFixed(2)),
    nums: [...trk.querySelectorAll('.trk-num')].map(e => ({
      txt: e.textContent, left: +parseFloat(e.style.left).toFixed(2),
      l0: e.classList.contains('l0'), l1: e.classList.contains('l1')})),
  });

  const out = {tiers: {}, struct: {}, empty: {}, rows: [], seq: [], prefill: [],
               est: null, estAfter: null, afterReset: null, tickColor: null};
  setPhase('run');        // file:// 直开落在 idle 态，KPI 行是 hidden；量之前先切 run
  const kpiP = document.querySelector('.kpi.prefill'), kpiD = document.querySelector('.kpi.decode');
  const trkP = document.getElementById('trkPrefill'), trkD = document.getElementById('trkDecode');
  // 1) 空读数：boot 的 init() 已 resetKpis 建出压暗空轨（懒建只建一次），量 fresh load 态
  out.tiers = {prefill: trkP.dataset.tiers, decode: trkD.dataset.tiers};
  out.struct = {prefill: struct(trkP, TIERS.prefill), decode: struct(trkD, TIERS.decode)};
  out.empty = {prefill: snap(trkP, kpiP), decode: snap(trkD, kpiD)};
  // 2) 逐值：档位类名/actor/档名/位置/粒子数，全部对着契约解析式核
  resetKpis();
  for(const v of VALS){
    const before = bursts(trkD);
    kpiPush('#kpiDecode', v, '目检');
    const s = snap(trkD, kpiD);
    s.v = v; s.idx = tierIdx(v, TIERS.decode); s.want = wantPct(v, TIERS.decode);
    s.newBurst = bursts(trkD).filter(b => !before.includes(b)).length;
    out.rows.push(s);
  }
  // 4)/5) 升档 pop+burst、同档与降档不庆祝：reset 后首值算「睡醒首发车」
  resetKpis();
  const step = (tag, fn) => {
    const before = bursts(trkD);
    fn();
    const s = snap(trkD, kpiD), fresh = bursts(trkD).filter(b => !before.includes(b));
    s.tag = tag; s.newBurst = fresh.length; s.burstKids = fresh.map(b => b.querySelectorAll('i').length);
    out.seq.push(s);
  };
  step("首发车 5（idle→t0）", () => kpiPush('#kpiDecode', 5, '目检'));
  step("同档 8（t0→t0）", () => kpiPush('#kpiDecode', 8, '目检'));
  step("升档 300（t0→t4）", () => kpiPush('#kpiDecode', 300, '目检'));
  step("降档 12（t4→t1）", () => kpiPush('#kpiDecode', 12, '目检'));
  // 6) 估值帧：降透明靠 .est 类，实测一来必须摘掉
  kpiPush('#kpiDecode', 41, '目检');            // t2 实测打底
  kpiPushEst('#kpiDecode', 88);                 // 估值帧（88 ≤ 100 → t3，位置照走）
  out.est = snap(trkD, kpiD);
  out.est.v = 88; out.est.idx = tierIdx(88, TIERS.decode); out.est.want = wantPct(88, TIERS.decode);
  kpiPush('#kpiDecode', 88, '实测');            // 实测覆盖估值
  out.estAfter = snap(trkD, kpiD);
  // 7) 复位回空读数
  resetKpis();
  out.afterReset = {prefill: snap(trkP, kpiP), decode: snap(trkD, kpiD)};
  // 3) prefill 轨抽测（另一套 data-tiers + 1K 里程碑写法；20000 也超量程）
  resetKpis();
  for(const v of PRE_VALS){
    const before = bursts(trkP);
    kpiPush('#kpiPrefill', v, '目检');
    const s = snap(trkP, kpiP);
    s.v = v; s.idx = tierIdx(v, TIERS.prefill); s.want = wantPct(v, TIERS.prefill);
    s.newBurst = bursts(trkP).filter(b => !before.includes(b)).length;
    out.prefill.push(s);
  }
  // 8) 检查站双色：computedStyle 抽查（灰缝 vs 卡色缝只有真渲染里才有解析值）
  resetKpis();
  kpiPush('#kpiDecode', 41, '目检');            // t2 → 前两个站已过、后两个未到
  const card = cardColor(trkD);
  out.tickColor = {card, pastN: pastTicks(trkD),
    past: ticks(trkD).filter(t => t.classList.contains('past'))
      .map(t => getComputedStyle(t).backgroundColor),
    ahead: ticks(trkD).filter(t => !t.classList.contains('past'))
      .map(t => getComputedStyle(t).backgroundColor)};
  return out;
}"""

# burst 自删要等动画跑完（~550ms），只能出了 evaluate 再等
BURSTS_LEFT = "() => [...document.querySelectorAll('.trk .burst')].length"


def main() -> int:
    fails: list[str] = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL)
        pg.wait_for_timeout(400)
        r = pg.evaluate(CHECKS)
        pg.wait_for_timeout(800)                 # > 动画 550ms，让 burst 自删
        left = pg.evaluate(BURSTS_LEFT)
        b.close()

    print("tiers:", r["tiers"])
    # 结构契约：懒建出来的轨道骨架（node 桩下 querySelector 全 null，只有真浏览器看得见）
    for key in ("prefill", "decode"):
        st, ts = r["struct"][key], TIER[key]
        print(f"  {key}: rail={st['rail']} fill={st['fill']} ticks={st['ticks']}"
              f" nums={[(n['txt'], n['left']) for n in st['nums']]}")
        if st["rail"] != 1 or st["fill"] != 1 or not st["actorNest"]:
            fails.append(f"{key} 轨道骨架缺件：rail={st['rail']} fill={st['fill']} actor 嵌套={st['actorNest']}")
        if st["actorSparks"] != 6 or st["streak"] != 2:
            fails.append(f"{key} actor 里粒子/速度线不是 6+2：{st['actorSparks']}+{st['streak']}")
        if st["ticks"] != [round(100 * i / len(ts), 2) for i in range(1, len(ts))]:
            fails.append(f"{key} 检查站刻线位置不对：{st['ticks']}")
        want_nums = [round(100 * i / len(ts), 2) for i in range(len(ts) + 1)]
        want_txt = [num_txt(i and ts[i - 1]) for i in range(len(ts) + 1)]
        got_txt = [n["txt"] for n in st["nums"]]
        got_left = [n["left"] for n in st["nums"]]
        if got_txt != want_txt or got_left != want_nums:
            fails.append(f"{key} 里程碑数字不对：{list(zip(got_txt, got_left))} 该是 {list(zip(want_txt, want_nums))}")
        if not (st["nums"][0]["l0"] and st["nums"][-1]["l1"]
                and not any(n["l0"] or n["l1"] for n in st["nums"][1:-1])):
            fails.append(f"{key} 两端对齐类 l0/l1 挂错了：{[n['txt'] for n in st['nums'] if n['l0'] or n['l1']]}")

    # 1) 空读数：两轨压暗 + 起点清零 + 😴 待发车 + 零粒子 + 检查站全灰（fresh load，boot 已 resetKpis）
    for key in ("prefill", "decode"):
        e = r["empty"][key]
        if not (e["cls"] == "trk idle" and e["width"] == "0%" and e["left"] == "0%"
                and e["actor"] == "😴" and e["tname"] == "待发车"
                and e["sparks"] == 0 and e["burst"] == 0 and not e["pop"] and e["past"] == 0):
            fails.append(f"空读数没退干净（{key}）：{e}")
    # 2)/3) 逐值：位置/档位/actor/档名/粒子数；900 超量程必须钉最后一档
    prev_idx = -1
    for row in r["rows"]:
        v, idx = row["v"], row["idx"]
        print(f"  v={v:>4}  {row['cls']:<8} left={row['left']:<7} {row['actor']}{row['tname']}"
              f"  粒子{row['sparks']}  过站{row['past']}"
              f"  burst{'新' if row['newBurst'] else '—'}{' POP' if row['pop'] else ''}")
        if row["cls"] != f"trk t{idx}":
            fails.append(f"v={v} 档位类名该是 trk t{idx}，实为 {row['cls']!r}")
        if row["actor"] != ACTORS[idx] or row["tname"] != NAMES[idx]:
            fails.append(f"v={v} 该是 {ACTORS[idx]}{NAMES[idx]}，实为 {row['actor']}{row['tname']}")
        if row["width"] != row["want"] or row["left"] != row["want"]:
            fails.append(f"v={v} 位置不是 trkFrac 契约串：fill={row['width']!r} left={row['left']!r}"
                         f" 该都是 {row['want']!r}")
        if row["sparks"] != SPARKS[idx]:
            fails.append(f"v={v} t{idx} 可见粒子该 {SPARKS[idx]} 颗，实为 {row['sparks']}")
        if row["pop"] != (idx > prev_idx):
            fails.append(f"v={v} pop 给错了：档位 {prev_idx}→{idx}，实为 pop={row['pop']}")
        if row["newBurst"] != (1 if idx > prev_idx else 0):
            fails.append(f"v={v} burst 给错了：档位 {prev_idx}→{idx}，新 burst={row['newBurst']}")
        # 检查站双色：i < now 的刻线带 .past（已过站），故 past 数 == 档位号（t4 时 4 个全过）
        if row["past"] != idx:
            fails.append(f"v={v} 已过站刻线该 {idx} 条（t{idx}），实为 {row['past']}")
        prev_idx = idx
    hi = max(TIER["decode"])
    over = [x for x in r["rows"] if x["v"] > hi]
    for row in over:
        if row["cls"] != "trk t4" or row["left"] != "100%":
            fails.append(f"v={row['v']} 超量程（> {hi}）该钉 t4 + 100%，实为 {row['cls']!r} / {row['left']!r}"
                         "（findIndex 给 -1，别用 Math.max(0, idx) 兜——见 pitfalls/crosscut/02）")
    if not over:
        fails.append("逐值表里没有超量程样本，这条回归没被覆盖")
    # 4)/5) 升档 pop + 冲线 burst（6 颗 <i>）；同档与降档不 pop、不 burst
    want_seq = [("首发车", True, 1), ("同档", False, 0), ("升档", True, 1), ("降档", False, 0)]
    for row, (tag, want_pop, want_burst) in zip(r["seq"], want_seq):
        print(f"  {row['tag']:<20} {row['cls']:<8} pop={row['pop']} 新burst={row['newBurst']}")
        if row["pop"] != want_pop or row["newBurst"] != want_burst:
            fails.append(f"{tag} 帧 {row['tag']}：pop 该 {want_pop} 实为 {row['pop']}，"
                         f"burst 该 {want_burst} 实为 {row['newBurst']}")
        if want_burst and row["burstKids"] != [6]:
            fails.append(f"{tag} 帧 {row['tag']}：burst 里该是 6 颗 <i>，实为 {row['burstKids']}")
    # 6) 估值帧带 est，实测一来摘掉
    est, after = r["est"], r["estAfter"]
    print(f"  est: {est['cls']!r} → 实测后 {after['cls']!r}")
    if est["cls"] != f"trk t{est['idx']} est" or "est" not in (est["tnameCls"] or "").split():
        fails.append(f"估值帧没挂 est：trk={est['cls']!r} tname={est['tnameCls']!r}"
                     f" 该是 trk t{est['idx']} est")
    if est["width"] != est["want"] or est["left"] != est["want"]:
        fails.append(f"估值帧位置不是 trkFrac 契约串：fill={est['width']!r} left={est['left']!r}"
                     f" 该都是 {est['want']!r}")
    if "est" in after["cls"].split() or "est" in (after["tnameCls"] or "").split():
        fails.append(f"实测帧没摘掉 est：trk={after['cls']!r} tname={after['tnameCls']!r}")
    # 7) resetKpis 回空读数（检查站也要回全灰）
    for key in ("prefill", "decode"):
        e = r["afterReset"][key]
        if not (e["cls"] == "trk idle" and e["width"] == "0%" and e["left"] == "0%"
                and e["actor"] == "😴" and e["tname"] == "待发车"
                and e["sparks"] == 0 and not e["pop"] and e["past"] == 0):
            fails.append(f"resetKpis 后没回空读数（{key}）：{e}")
    # 3) prefill 轨（另一套 data-tiers）
    for row in r["prefill"]:
        v, idx = row["v"], row["idx"]
        print(f"  prefill v={v:>5}  {row['cls']:<8} left={row['left']:<7} {row['actor']}{row['tname']}"
              f"  粒子{row['sparks']}  过站{row['past']}  burst{'新' if row['newBurst'] else '—'}")
        if row["cls"] != f"trk t{idx}" or row["actor"] != ACTORS[idx]:
            fails.append(f"prefill v={v} 档位不对：{row['cls']!r} {row['actor']}")
        if row["width"] != row["want"] or row["left"] != row["want"]:
            fails.append(f"prefill v={v} 位置不是 trkFrac 契约串：fill={row['width']!r} left={row['left']!r}"
                         f" 该都是 {row['want']!r}")
        if row["sparks"] != SPARKS[idx]:
            fails.append(f"prefill v={v} t{idx} 可见粒子该 {SPARKS[idx]} 颗，实为 {row['sparks']}")
        if row["past"] != idx:
            fails.append(f"prefill v={v} 已过站刻线该 {idx} 条（t{idx}），实为 {row['past']}")
    # 8) 检查站双色：computedStyle 抽查——已过站 = --card 解析值，未到站 = 灰（两者必须不同色）
    tc = r["tickColor"]
    print(f"  双色 t2：已过站 {tc['past']} / 未到站 {tc['ahead']} / --card={tc['card']}")
    if tc["pastN"] != 2 or len(tc["past"]) != 2 or len(tc["ahead"]) != 2:
        fails.append(f"t2 该 2 条已过站 + 2 条未到站，实为 past={tc['pastN']}"
                     f"（已过 {len(tc['past'])} / 未到 {len(tc['ahead'])}）")
    if any(c != tc["card"] for c in tc["past"]):
        fails.append(f"已过站刻线不是 --card 解析值：{tc['past']} 该都是 {tc['card']}")
    if any(c == tc["card"] for c in tc["ahead"]):
        fails.append(f"未到站刻线与 --card 同色（灰缝写成同名同色了）：{tc['ahead']}")
    # burst 是一次性的：动画（~550ms）跑完必须自删，页面不留残元素
    if left:
        fails.append(f"burst 播完没自删，页面残留 {left} 个")
    if errs:
        fails.append(f"页面 JS 报错：{errs}")

    if fails:
        print("\n".join(["FAIL:"] + [f"  - {x}" for x in fails]))
        return 1
    print("KPI 赛道 OK（结构/逐值档位与位置/超量程钉尾档/粒子数/检查站双色/升档 pop+burst"
          "/降档安静/est/复位/无残留）")
    return 0


def num_txt(v) -> str:
    return f"{v // 1000}K" if v >= 1000 else str(v)


if __name__ == "__main__":
    sys.exit(main())
