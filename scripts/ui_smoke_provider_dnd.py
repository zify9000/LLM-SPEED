"""Provider 管理抽屉（配置编辑器）三套手柄拖拽的真浏览器烟测（可选脚本，**不进回归套件**）。

为什么需要它：抽屉里的拖拽换序只有真 DOM + 真拖拽事件才成立——node 侧门禁（`node --check`、
§pure、渲染 harness）看不见 `dragover`/`drop`/`dragend` 的派发顺序，也看不见"指针出了容器
还收不收得到事件"。这里踩过两个坑（2026-10 用户报，Chromium 实测，均已修）：

1. **出界即失效**：部署环境区把 `dragover`/`drop` 挂在 `#ceDeps` 容器上。指针一滑出组件
   （上面的「模型类型及上下文」区、下面的「+ 添加部署环境」按钮、乃至抽屉外）就收不到
   `dragover`，浏览器判定"此处不接受投放"→ 不发 `drop` → 行悄悄弹回原位，而落点线
   `.drop-marker` 还冻在原地骗人。判据：**出界松手也必须按最后指示的位置落定**（与
   声明行/地址行那两套手柄同语义）。
2. **跨组件投放复制部署**：`_dnd` 句柄的清理原本押在"`#ceDeps` 的委托 `dragend` 一定跑在
   重渲之前"。源节点一旦被中途重渲甩出 DOM，`dragend` 就落在游离节点上（游离节点的事件连
   document 都到不了），句柄长期攥着一个不在 DOM 里的旧条目；此后**任何别的组件**（声明行/
   地址行）投放到部署区都会把它塞回组里，而 `ceStash` 以 DOM 为唯一事实源 → 凭空多出一条
   同名同参的部署环境（症状：拖一行「模型类型及上下文」到部署区＝复制一份未保存的部署环境）。
   判据：**拖拽途中重渲后，再跨组件投放，`deployments` 条数不变**。

跑法（需要 playwright + chromium；没装就跳过并返回 0，不影响 tests/ 的任何门禁）：

    python3 scripts/ui_smoke_provider_dnd.py

只读仓库 `static/index.html`（file:// 直开，不起服务、不碰 config.json 与 results/ 存档）。
视口开得比抽屉高，免得"目标元素在视口外"被误判成拖拽失效。
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

# 两个分组三条部署三条声明两条地址：够摆出"组内换序 / 跨组移动 / 出界夹紧 / 跨组件投放"
SEED = r"""() => {
  state.providers = [{
    name: "local", label: "本地", gateway_urls: ["http://a/v1", "http://b/v1"], local: true,
    deployment_groups: ["A组", "B组"],
    model_profiles: [{model:"m1",kinds:["llm"],max_ctx:32768},
                     {model:"m2",kinds:["llm"],max_ctx:131072},
                     {model:"m3",kinds:["llm"],max_ctx:null}],
    deployments: [{label:"a1",models:["m1"],hardware:"4090",framework:"vLLM",group:"A组"},
                  {label:"a2",models:["m2"],hardware:"4090",framework:"vLLM",group:"A组"},
                  {label:"b1",models:["m3"],hardware:"A100",framework:"SGLang",group:"B组"}]}];
  $("#fProvider").value = "local";
  initCfgEditor("local");
  openCfgDrawer();
}"""

# 现场快照：分组视图（DOM）+ 内存数据 + 句柄 + 类名残留
SNAP = r"""() => {
  const p = ceCurrent();
  const byGroup = {};
  for(const g of document.querySelectorAll("#ceDeps .dep-group"))
    byGroup[g.dataset.group || "未分组"]
      = [...g.querySelectorAll(".dep-items > .dep-edit")].map(el => el.querySelector(".d-label").value);
  let dnd = null;
  try{ dnd = (typeof _dnd !== "undefined" && _dnd)
        ? {kind:_dnd.kind, live:!!(_dnd.el && _dnd.el.isConnected)} : null; }catch(e){ dnd = "ERR"; }
  return {
    byGroup,
    data: (p.deployments||[]).map(d => `${d.label||"(空)"}@${d.group||"未分组"}`),
    prof: [...document.querySelectorAll("#ceProfiles .mp-row")].map(el => el.dataset.model),
    dataProf: (p.model_profiles||[]).map(f => f.model),
    urls: [...document.querySelectorAll("#ceUrls .ce-url")].map(i => i.value),
    dnd, dirty: state.cfgEdit.dirty,
    residue: [...document.querySelectorAll(".drop-marker,.drop-target-zone,.dragging")].map(el => el.className),
  };
}"""


def pt(pg, sel, cx=0.5, cy=0.5):
    r = pg.evaluate("""([s,cx,cy]) => { const e = document.querySelector(s); if(!e) return null;
      const b = e.getBoundingClientRect(); return {x: b.left + b.width*cx, y: b.top + b.height*cy,
      inView: b.top >= 0 && b.bottom <= innerHeight}; }""", [sel, cx, cy])
    if r is None:
        raise AssertionError(f"找不到目标元素：{sel}")
    if not r["inView"]:
        raise AssertionError(f"目标元素不在视口内（烟测环境太矮，不是页面 bug）：{sel} y={r['y']:.0f}")
    return r


def grab(pg, sel):
    loc = pg.locator(sel).first
    loc.scroll_into_view_if_needed()
    pg.wait_for_timeout(90)
    b = loc.bounding_box()
    x, y = b["x"] + b["width"]/2, b["y"] + b["height"]/2
    pg.mouse.move(x, y); pg.wait_for_timeout(50)
    pg.mouse.down(); pg.wait_for_timeout(60)
    return x, y


def drag(pg, src_sel, dst, steps=8):
    """抓起 src_sel 的手柄 → 滑到 dst → 松手。dst 给选择器时**每步重取中心**（抗拖拽
    中途的原生自动滚动），给 {x,y} 时按绝对坐标走。"""
    x, y = grab(pg, src_sel)
    for i in range(1, steps+1):
        t = pt(pg, dst) if isinstance(dst, str) else dst
        pg.mouse.move(x + (t["x"]-x)*i/steps, y + (t["y"]-y)*i/steps)
        pg.wait_for_timeout(22)
    pg.wait_for_timeout(60)
    pg.mouse.up()
    pg.wait_for_timeout(300)


def reset(pg):
    pg.evaluate(SEED)
    pg.wait_for_timeout(250)
    return pg.evaluate(SNAP)


def groups(s):
    return {k: v for k, v in s["byGroup"].items() if v}


def main() -> int:
    fails: list[str] = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 1400})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL)
        pg.wait_for_timeout(400)
        base = reset(pg)
        if groups(base) != {"A组": ["a1", "a2"], "B组": ["b1"]}:
            fails.append(f"起点就不对：{groups(base)}")
        dep_bottom = pg.evaluate("() => document.querySelector('#ceDeps').getBoundingClientRect().bottom")

        # 1) 对照：区内投放（A组首行拖到 B组那行上）——跨组移动 + ceStash 跟上 DOM
        reset(pg)
        drag(pg, "#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(1) .dgrip",
             "#ceDeps .dep-group[data-group='B组'] .dep-edit")
        s = pg.evaluate(SNAP)
        print("1 区内投放（a1 → B组 b1 上）:", groups(s))
        if groups(s) != {"A组": ["a2"], "B组": ["a1", "b1"]}:
            fails.append(f"区内投放没按落点线排：{groups(s)}")
        if s["data"] != ["a2@A组", "a1@B组", "b1@B组"]:
            fails.append(f"DOM 与数据不一致（ceStash 没跟上 DOM）：{s['data']}")
        if s["dnd"] is not None or s["residue"] or not s["dirty"]:
            fails.append(f"1 收尾不干净：dnd={s['dnd']} residue={s['residue']} dirty={s['dirty']}")

        # 2) bug1 出界·向上：指针越过 #ceDeps 顶，落在「模型类型及上下文」区标题上
        reset(pg)
        drag(pg, "#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(2) .dgrip",
             "#ceProfileSec .ds-t")
        s = pg.evaluate(SNAP)
        print("2 出界向上（a2 在声明区标题上松手）:", groups(s))
        if groups(s) != {"A组": ["a2", "a1"], "B组": ["b1"]}:
            fails.append(f"出界向上松手没按最后指示落定（落点线骗人）：{groups(s)}")
        if s["residue"]:
            fails.append(f"2 拖完类名有残留：{s['residue']}")

        # 3) bug1 出界·向下：越过 #ceDeps 底，落在「+ 添加部署环境」按钮上 → 夹紧到末组末尾
        reset(pg)
        drag(pg, "#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(1) .dgrip",
             "#ceDepSec .ce-actions .btn-ghost")
        s = pg.evaluate(SNAP)
        print("3 出界向下（a1 在添加部署按钮上松手）:", groups(s))
        if groups(s) != {"A组": ["a2"], "B组": ["b1", "a1"]}:
            fails.append(f"出界向下没夹紧到末组末尾：{groups(s)}")

        # 4) bug1 出界·整条出抽屉：落在抽屉左侧的遮罩上（x 在抽屉外），仍按末位落定
        reset(pg)
        drag(pg, "#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(1) .dgrip",
             {"x": 300, "y": dep_bottom + 40})
        s = pg.evaluate(SNAP)
        print("4 出界到抽屉外:", groups(s))
        if groups(s) != {"A组": ["a2"], "B组": ["b1", "a1"]}:
            fails.append(f"指针整条出界后松手没落定：{groups(s)}")
        if s["dnd"] is not None:
            fails.append(f"4 拖完句柄没归零：{s['dnd']}")

        # 5) bug2：拖拽途中列表被重渲（源节点甩出 DOM＝句柄过期），再拖声明行进部署区
        reset(pg)
        x, y = grab(pg, "#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(1) .dgrip")
        t = pt(pg, "#ceDeps .dep-group[data-group='B组'] .dep-edit")
        for i in range(1, 7):
            pg.mouse.move(x + (t["x"]-x)*i/7, y + (t["y"]-y)*i/7)
            pg.wait_for_timeout(25)
        pg.evaluate("() => renderCeDeps()")       # 中途重渲：手里的节点变游离节点
        pg.wait_for_timeout(120)
        pg.mouse.up(); pg.wait_for_timeout(300)
        mid = pg.evaluate(SNAP)
        print("5a 中途重渲后松手:", groups(mid), "句柄:", mid["dnd"])
        if groups(mid) != {"A组": ["a1", "a2"], "B组": ["b1"]}:
            fails.append(f"5a 过期句柄不该改动部署区：{groups(mid)}")
        drag(pg, "#ceProfiles .mp-row:nth-child(1) .mp-grip",
             "#ceDeps .dep-group[data-group='B组'] .dep-edit")
        s = pg.evaluate(SNAP)
        print("5b 随后拖声明行进部署区:", groups(s), s["data"])
        if len(s["data"]) != 3 or len([v for v in s["byGroup"].values()]) != 2:
            fails.append(f"跨组件投放复制了部署环境：{s['data']}")
        if s["dnd"] is not None or s["residue"]:
            fails.append(f"5b 跨组件投放后句柄/类名没清干净：dnd={s['dnd']} residue={s['residue']}")

        # 5c) 死句柄自愈：过期句柄必须在下一次碰到它时被识别并丢掉（dragend 到不了 document）
        heal = pg.evaluate("""() => { const ghost = document.querySelector('#ceDeps .dep-edit').cloneNode(true);
            const alive = _dnd; _dnd = {kind:'dep', el: ghost};
            const before = document.querySelectorAll('#ceDeps .dep-edit').length;
            _dndLive(); const cleared = _dnd === null;
            _dnd = alive;
            return {cleared, same: document.querySelectorAll('#ceDeps .dep-edit').length === before}; }""")
        print("5c 死句柄自愈:", heal)
        if not heal["cleared"]:
            fails.append("5c 过期句柄没有就地作废（_dndLive 不认 isConnected）")

        # 6) 声明行本身：拖一行越出 #ceProfiles（落点夹紧到本列表末尾），不新增不丢失
        reset(pg)
        drag(pg, "#ceProfiles .mp-row:nth-child(1) .mp-grip", "#ceDepSec .ds-t")
        s = pg.evaluate(SNAP)
        print("6 声明行出界投放:", s["prof"], s["dataProf"])
        if s["prof"] != ["m2", "m3", "m1"] or len(set(s["dataProf"])) != 3:
            fails.append(f"声明行换序不对或复制了声明：dom={s['prof']} data={s['dataProf']}")

        # 7) 地址行（同一套 bindGripDrag 的回归项）：越出 #ceUrls 也照样落定，地址不丢
        reset(pg)
        drag(pg, "#ceUrls .url-row:nth-child(1) .url-grip", "#ceLocal")
        s = pg.evaluate(SNAP)
        print("7 地址行出界投放:", s["urls"])
        if sorted(s["urls"]) != ["http://a/v1", "http://b/v1"]:
            fails.append(f"地址行换序把地址弄丢或多出来了：{s['urls']}")

        # 8) 键盘等价物仍与鼠标共用一条落盘路径：焦点在手柄上按 ↓ 组内换序
        reset(pg)
        pg.locator("#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(1) .dgrip") \
          .first.press("ArrowDown")
        pg.wait_for_timeout(250)
        s = pg.evaluate(SNAP)
        print("8 键盘 ↓ 组内换序:", groups(s))
        if groups(s) != {"A组": ["a2", "a1"], "B组": ["b1"]}:
            fails.append(f"键盘换序与鼠标口径不一致：{groups(s)}")
        if not s["dirty"]:
            fails.append("换序后没标未保存（dirty）")

        # 9) 分组手柄：出界向上 → 夹紧到首组之前（B组排到 A组前面）
        reset(pg)
        drag(pg, "#ceDeps .dep-group[data-group='B组'] .g-grip", "#ceProfileSec .ds-t")
        s = pg.evaluate(SNAP)
        order = [k for k in s["byGroup"]]
        print("9 分组手柄出界向上:", order)
        if order[:2] != ["B组", "A组"]:
            fails.append(f"分组换序没按出界夹紧落定：{order}")

        # 10) 空分组也能投：新建于不存在的组里只有一条占位，拖进去要改归属
        reset(pg)
        pg.evaluate("() => ceGroupAdd()")          # 新建一个空分组（名字带序号，取末组）
        pg.wait_for_timeout(200)
        newg = pg.evaluate("""() => { const gs = document.querySelectorAll('#ceDeps .dep-group');
            return gs[gs.length-1].dataset.group; }""")
        drag(pg, "#ceDeps .dep-group[data-group='A组'] .dep-edit:nth-child(1) .dgrip",
             "#ceDeps .dep-group:last-child .g-empty")
        s = pg.evaluate(SNAP)
        print(f"10 拖进空分组「{newg}」:", groups(s))
        if newg not in s["byGroup"] or s["byGroup"][newg] != ["a1"] or "A组" not in s["byGroup"] \
           or s["byGroup"]["A组"] != ["a2"]:
            fails.append(f"拖进空分组没改归属：{groups(s)}")
        if s["dnd"] is not None or s["residue"]:
            fails.append(f"10 收尾不干净：dnd={s['dnd']} residue={s['residue']}")

        b.close()

    if errs:
        fails.append(f"页面抛异常：{errs[:3]}")
    for f in fails:
        print("FAIL:", f)
    print("OK：部署区/声明区/地址行三套手柄——区内投放、出界按指示器落定、"
          "跨组件投放不复制部署、死句柄自愈、键盘同口径、空组可投、无类名残留"
          if not fails else f"{len(fails)} 项不过")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
