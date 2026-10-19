"""历史浏览器抽屉（ADR-0132「五面归一」）的浏览器目检替身（可选脚本，**不进回归套件**）。

为什么需要它：抽屉的三条手势——**双击行 = 载入、勾选 = 选集、拖到组名上 = 归组**——只有
在真浏览器里才成立。node 侧三条门禁（`node --check` 语法、注入/令牌/注释静态门禁、
§pure 纯函数）都只看得见"另一半"：HTML5 DnD 的 dragstart/drop、抽屉的滑入类、底栏按
选择集三态换文案、`hdSel` 跨切桶/跨搜索存活、uiDialog 的确认语义、窄屏栅格会不会横向
溢出——这些没有一条能靠静态检查或纯函数测出来。旧脚本测的浮层/弹窗/台账已随重构退役
（`#histPop`/`#histLedgerRoot`/`#cmpModalRoot`/`#spliceModalRoot` 都不存在了），本脚本
跟着换成抽屉的交互契约。

跑法（需要 playwright + httpx；没装就直接跳过并返回 0，不影响 tests/ 的任何门禁）：

    python3 scripts/ui_smoke_history.py            # 用仓库 results/ 里最新的 14 条
    UI_SMOKE_PORT=8601 UI_SMOKE_RECORDS=8 python3 scripts/ui_smoke_history.py

三个细节：①**解释器分两把**——本脚本用系统 `python3` 跑（`.venv` 里没装 playwright），
但它拉起的**服务端**用 `.venv/bin/python`（那里才有 fastapi/uvicorn）；②这是个顶层脚本，
**别 `import` 它探路**（一 import 整套冒烟就跑起来了，浪费几分钟）；③临时目录里**不拷
点开头的文件**（`.history-groups.json` 不进去，初始态才是可预测的"一条组都没有"）。

只读仓库 `results/`：服务端 RESULTS_DIR 被指到临时目录，分组/拼接/删除全落在临时副本上，
真实存档一个字节都不会变（脚本末尾用 BEFORE 快照断言这一点）。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

try:
    import httpx
    from playwright.sync_api import sync_playwright
except Exception as _exc:                       # 缺依赖 = 跳过，不算失败
    print(f"SKIP：需要 playwright + httpx（{_exc}）；前端契约的持久覆盖仍在 tests/frontend/ 三套门禁")
    raise SystemExit(0)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("UI_SMOKE_PORT", "8599"))
N_RECORDS = int(os.environ.get("UI_SMOKE_RECORDS", "14"))
PY = os.path.join(SRC, ".venv/bin/python")
if not os.path.exists(PY):
    PY = sys.executable
ROOT = tempfile.mkdtemp(prefix="hist-smoke.")
RDIR = os.path.join(ROOT, "results")
REAL = os.path.join(SRC, "results")
os.makedirs(RDIR, exist_ok=True)


def dir_snapshot(d):
    """目录快照：{文件名: (size, mtime)}——只算存档，点开头的是组映射等旁路文件。"""
    out = {}
    for f in sorted(os.listdir(d)):
        if f.startswith(".") or not f.endswith(".json"):
            continue
        st = os.stat(os.path.join(d, f))
        out[f] = (st.st_size, round(st.st_mtime, 3))
    return out


def n_archives(d=RDIR):
    return len([f for f in os.listdir(d) if f.endswith(".json") and not f.startswith(".")])


BEFORE = dir_snapshot(REAL)          # 真实仓库存档快照：全程不许变

# 挑一批最近的存档进临时目录（跨天/跨模型，够看出分组又不至于慢）；点开头文件一律不拷
picks = []
for f in sorted(os.listdir(REAL), reverse=True):
    if f.endswith(".json") and not f.startswith(".") and len(picks) < N_RECORDS:
        picks.append(f)
for f in picks:
    shutil.copy(os.path.join(REAL, f), os.path.join(RDIR, f))
N_DIR = len(picks)

LAUNCH = os.path.join(ROOT, "launch.py")
with open(LAUNCH, "w", encoding="utf-8") as fp:
    fp.write(
        "import sys; sys.path.insert(0, %r)\n"
        "import uvicorn, server\n"
        "server.RESULTS_DIR = %r\n"
        "uvicorn.run(server.app, host='127.0.0.1', port=%d, log_level='error')\n"
        % (SRC, RDIR, PORT))

proc = subprocess.Popen([PY, LAUNCH],
                        cwd=SRC, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
url = f"http://127.0.0.1:{PORT}"
for _ in range(60):
    try:
        if httpx.get(url + "/api/bench/history", timeout=1).status_code == 200:
            break
    except Exception:
        time.sleep(0.3)
else:
    proc.kill()
    sys.exit("服务没起来")

fails = []
GMAP = os.path.join(RDIR, ".history-groups.json")


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"　[{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


def groups_file():
    """组映射文件（v2：{v, groups[], assign{}}）。没写过 = 一个组都没有。"""
    if not os.path.exists(GMAP):
        return {"v": 2, "groups": [], "assign": {}}
    d = json.load(open(GMAP, encoding="utf-8"))
    return d if "groups" in d else {"v": 2, "groups": [], "assign": d}


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 950})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url + "/", wait_until="networkidle")

        def js(expr, arg=None):
            return pg.evaluate(expr) if arg is None else pg.evaluate(expr, arg)

        def wait(expr, timeout=8000):
            try:
                pg.wait_for_function(expr, timeout=timeout)
                return True
            except Exception:
                return False

        def click(sel):
            """真实点击优先（证明元素真能点中）；抽屉里偶发 actionability 超时退回 DOM 派发。"""
            try:
                pg.click(sel, timeout=3000)
                return "real"
            except Exception:
                pg.evaluate("s => document.querySelector(s).click()", sel)
                return "dom"

        def dblclick(sel):
            """双击行 = 载入（浏览态）：真实双击优先，退回两次 DOM 派发 click。"""
            try:
                pg.dblclick(sel, timeout=3000)
                return "real"
            except Exception:
                pg.evaluate("s => { const e = document.querySelector(s); e.click(); e.click(); }", sel)
                return "dom"

        def check_box(rid):
            click(f'.hd-row[data-rid="{rid}"] .hd-ck')

        api = httpx.get(url + "/api/bench/history", timeout=10).json()
        rows_all = []

        print("\n[1] 入口与抽屉：一份列表、一个入口（五面归一）")
        check("旧五面（浮层/台账/对比弹窗/拼接弹窗）的根节点都不在了",
              js("() => ['histPop','histLedgerRoot','cmpModalRoot','spliceModalRoot','histMoreBtn']"
                 ".filter(id => document.getElementById(id)).join(',')") == "")
        click("#histPick")
        # .open 挂在根节点上（CSS 是 #histDrawerRoot.open #histDrawer{transform:none}），
        # 所以顺带核一眼抽屉本体的 computed visibility——类挂上了但没滑进来也是坏的
        opened = wait("() => {const r = document.getElementById('histDrawerRoot');"
                      " return !r.hidden && r.classList.contains('open')"
                      " && getComputedStyle(document.getElementById('histDrawer')).visibility === 'visible';}")
        aria = pg.get_attribute("#histPick", "aria-expanded")
        check("抽屉开起来了（根节点 .open 滑入 + 抽屉可见 + aria-expanded）", opened and aria == "true",
              f"open={opened} aria={aria}")
        n_rows = pg.locator("#hdList .hd-row").count()
        check(f"列表条数与目录一致（{n_rows} 行 = {N_DIR} 个存档 = 接口 total {api['total']}）",
              n_rows == N_DIR == api["total"], f"{n_rows}/{N_DIR}/{api['total']}")
        bkts = pg.eval_on_selector_all("#hdGroups .hd-gitem",
                                       "els => els.map(e => [e.dataset.bkt, e.querySelector('.hg-n').innerText, e.querySelector('.hg-c').innerText])")
        check("左栏「全部记录」+「未分组」，计数都对",
              [b[1] for b in bkts] == ["全部记录", "未分组"] and bkts[0][2] == str(n_rows)
              and bkts[1][2] == str(n_rows), str(bkts))
        check("披露行报总数", str(n_rows) in pg.inner_text("#hdDisc"), pg.inner_text("#hdDisc"))
        check("首开不再默认选中任何记录（播种「建议对比对」已移除，选集从零开始）",
              js("() => hdSel.size") == 0)
        check("行上是复选框 + ⋯钮（单击行=选中、双击=载入、勾选=选集三条路并存）",
              pg.locator("#hdList .hd-row .hd-ck").count() == n_rows
              and pg.locator("#hdList .hd-row [data-more]").count() == n_rows)

        print("\n[2] 单击行 = 选中/取消、双击行 = 载入")
        click("#hdSelNone")
        rows_all = js("() => [...document.querySelectorAll('#hdList .hd-row')].map(e => e.dataset.rid)")
        rid = rows_all[0]
        click(f'.hd-row[data-rid="{rid}"]')
        check("单击行不载入：state.runId 不是这条、抽屉还开着",
              js("() => String(state.runId)") != rid
              and not pg.locator("#histDrawerRoot").is_hidden())
        check("单击未选中行 = 立即进选集（无延时）、行上勾选框同步点亮",
              js(f"() => hdSel.has({json.dumps(rid)})")
              and pg.locator(f'.hd-row[data-rid="{rid}"] .hd-ck').is_checked())
        click(f'.hd-row[data-rid="{rid}"]')
        check("单击已选中行 = 待取消（双击窗口内不生效，勾选框还亮着）",
              js(f"() => hdSel.has({json.dumps(rid)})")
              and pg.locator(f'.hd-row[data-rid="{rid}"] .hd-ck').is_checked())
        check("双击窗口（450ms）过后取消兑现：出选集、勾选框同步熄灭",
              wait(f"() => !hdSel.has({json.dumps(rid)})", timeout=2000)
              and not pg.locator(f'.hd-row[data-rid="{rid}"] .hd-ck').is_checked())
        dblclick(f'.hd-row[data-rid="{rid}"]')
        check("双击行（非复选框区）→ 载入：state.runId 就是这条、页面进 done 态（待取消已被双击撤掉）",
              wait(f"() => state.runId === {json.dumps(rid)} && state.phase === 'done'"),
              js("() => String(state.runId) + '/' + state.phase"))
        check("载入后抽屉自己收起", wait("() => document.getElementById('histDrawerRoot').hidden"))
        check("入口按钮换成那条的短标题（不再是「浏览历史记录…」）",
              pg.inner_text("#histPickT").strip() != "浏览历史记录…", pg.inner_text("#histPickT"))

        print("\n[3] 面板快捷气泡：点一下直接载入（常看计数只住 localStorage）")
        chips = js("() => [...document.querySelectorAll('#hdQuick .hd-qchip')].map(e => e.dataset.q)")
        check("渲染出快捷气泡", len(chips) >= 2, f"{len(chips)} 颗")
        check("最多两行（最新 / 常看）、每行至多 3 颗",
              pg.locator("#hdQuick .hd-qrow").count() <= 2
              and max(pg.eval_on_selector_all("#hdQuick .hd-qrow",
                                              "els => els.map(e => e.querySelectorAll('.hd-qchip').length)")) <= 3,
              pg.inner_text("#hdQuick").replace("\n", " / ")[:60])
        check("当前已载入的那条不进气泡（它就在按钮上，重复是噪声）", rid not in chips)
        click(f'#hdQuick .hd-qchip[data-q="{chips[0]}"]')
        check("点气泡直接载入对应记录",
              wait(f"() => state.runId === {json.dumps(chips[0])} && state.phase === 'done'"),
              js("() => String(state.runId)"))
        # 再载入另一条、然后回头把第一条载入第二次 → 「常看」行才该出现（≥2 次）
        for _ in range(3):
            cur = js("() => state.runId")
            nxt = [c for c in js("() => [...document.querySelectorAll('#hdQuick .hd-qchip')].map(e => e.dataset.q)")
                   if c != cur]
            if not nxt:
                break
            click(f'#hdQuick .hd-qchip[data-q="{nxt[0]}"]')
            wait(f"() => state.runId === {json.dumps(nxt[0])}")
        loads = js("() => JSON.parse(localStorage.getItem('llmspeed.histLoads') || '{}')")
        check("载入次数记在 llmspeed.histLoads（本地表，不进存档）",
              any(int(v) >= 2 for v in loads.values()), str(sorted(loads.items(), key=lambda kv: -kv[1])[:3]))
        hints = pg.eval_on_selector_all("#hdQuick .hd-qrow .hd-qhint", "els => els.map(e => e.innerText)")
        check("「常看」行跟着出现（≥2 次的排前面）", "常看" in hints and "最新" in hints, str(hints))
        # 两行是两个独立排序键（新度 / 次数），**允许同现**。这里曾写成「最新行排斥
        # 常看条目」（`!freq.includes(h)`）：一条攒够 2 次载入就从「最新」消失、常看
        # 排名一掉出前三又冒回来，用户读成"记录在两行之间窜"。故把头部两条人为灌成
        # 「常看」，断言它们仍留在「最新」行，且两行确有同现条目
        cur_rid = js("() => state.runId")
        top = [x for x in js("() => histCache.map(h => h.run_id)") if x != cur_rid][:3]
        js("ids => localStorage.setItem('llmspeed.histLoads', JSON.stringify(ids))",
           {**loads, **{rid: 9 for rid in top[:2]}})
        js("() => renderHistQuick()")
        rows_q = pg.eval_on_selector_all(
            "#hdQuick .hd-qrow",
            "els => els.map(e => ({h: e.querySelector('.hd-qhint').innerText,"
            " ids: [...e.querySelectorAll('.hd-qchip')].map(c => c.dataset.q)}))")
        lat = next((r["ids"] for r in rows_q if r["h"].startswith("最新")), [])
        frq = next((r["ids"] for r in rows_q if r["h"].startswith("常看")), [])
        check("「最新」行＝缓存头部三条（上了「常看」也不许被挤掉）",
              lat == top, f"最新={lat} 应为={top}")
        check("同一条允许同现两行（两行各自独立排序，不互相排斥）",
              bool(set(lat) & set(frq)), f"同现={sorted(set(lat) & set(frq))}")

        print("\n[4] 选集：跨切桶 / 跨搜索都存活")
        click("#histPick")
        wait("() => !document.getElementById('histDrawerRoot').hidden")
        click("#hdSelNone")
        check("清空后选集归零（清空吃的是整个 Set，不只当前视图）", js("() => hdSel.size") == 0)
        check("空选集时底栏回到引导文案（不是一排禁用按钮）",
              pg.locator("#hdFoot .hd-hint").count() == 1 and "点一条记录载入" in pg.inner_text("#hdFoot"),
              pg.inner_text("#hdFoot")[:44])
        for r in rows_all[:3]:
            check_box(r)
        check("勾 3 条", js("() => hdSel.size") == 3, js("() => [...hdSel].length"))
        click('.hd-gitem[data-bkt=""]')                     # 切到「未分组」桶
        check("切桶后选集不变（选集独立于视图）", js("() => hdSel.size") == 3)
        key = rows_all[1][-4:]                              # 用记录编号后缀搜一条
        pg.fill("#hdSearch", key)
        wait("() => hdSearched !== null")
        n_hit = pg.locator("#hdList .hd-row").count()
        check(f"搜索「{key}」命中并渲染", n_hit >= 1, f"{n_hit} 行")
        check("搜索后选集仍不变", js("() => hdSel.size") == 3)
        check("披露行说明搜的是全部记录",
              "全部" in pg.inner_text("#hdDisc") and "命中" in pg.inner_text("#hdDisc"),
              pg.inner_text("#hdDisc"))
        pg.fill("#hdSearch", "")
        wait("() => hdSearched === null")

        print("\n[5] 组是实体：新建空组立刻看得见（先有组，再往里放）")
        GNAME = "冒烟组"
        click('.hd-side-foot button[onclick="hdNewGroup()"]')
        wait("() => !document.getElementById('uiDialogRoot').hidden")
        pg.fill("#uiDialogInput", GNAME)
        click("#uiDialogOk")
        check("新建空组建得出来", wait(f"() => histGroupsCache.some(g => g.name === {json.dumps(GNAME)})"))
        gid = js(f"() => (histGroupsCache.find(g => g.name === {json.dumps(GNAME)}) || {{}}).id")
        gitem = js(f"() => {{const e = document.querySelector('.hd-gitem[data-bkt=\"{gid}\"]');"
                   " return e ? [e.querySelector('.hg-n').innerText, e.querySelector('.hg-c').innerText] : null;}")
        check("左栏出现该组且条数是 0", gitem == [GNAME, "0"], str(gitem))
        check("组是实体（有自己的 id，不是只按名字当键）", bool(gid), str(gid))
        # ADR-0138：展示序的唯一事实源是 histLedgerBuckets——左栏读作 根节点 → 未分组 → 各组
        order = pg.eval_on_selector_all(
            "#hdGroups .hd-gitem",
            "els => els.map(e => e.dataset.all === '1' ? 'ALL' : e.dataset.bkt)")
        check("左栏序：全部记录 → 未分组置顶 → 各组（ADR-0138）",
              order[:2] == ["ALL", ""] and gid in order[2:], str(order))
        click(f'.hd-gitem[data-bkt="{gid}"] [data-gop]')
        wait("() => !document.getElementById('hdMenu').hidden")
        glabels = pg.eval_on_selector_all("#hdMenu .hd-mitem", "els => els.map(e => e.textContent.trim())")
        check("组 ⋯ 菜单：改名/备注/并入/解散",
              glabels == ["改名…", "备注…", "并入另一个组…", "解散该组"], str(glabels))
        click("#hdMenu .hd-mitem[data-mi='edit']")           # 开一下又关掉，验证菜单能开
        wait("() => !document.getElementById('uiDialogRoot').hidden")
        click("#uiDialogCancel")
        wait("() => document.getElementById('uiDialogRoot').hidden")

        print("\n[5b] 左栏「合并组」：把两个组并成一个")
        btns = pg.eval_on_selector_all(".hd-side-foot .nav-btn", "els => els.map(e => e.textContent.trim())")
        check("左栏底部按钮排布：新建组 / 合并组 / 自动建组（模型名）",
              btns == ["+ 新建组", "合并组", "自动建组（模型名）"], str(btns))
        # 造两个临时组（各带一段不相交的记录，A=前 1 条 / B=接下来 2 条），
        # 合并后断言目标组条数累加、源组消失；用独立组名，不动 [5] 的“冒烟组”（[6] 拖拽还要用它）
        if N_DIR >= 3:
            seg = 0
            for mk, mv in (("合并源 A", 1), ("合并源 B", 2)):
                click('.hd-side-foot button[onclick="hdNewGroup()"]')
                wait("() => !document.getElementById('uiDialogRoot').hidden")
                pg.fill("#uiDialogInput", mk)
                click("#uiDialogOk")
                wait(f"() => histGroupsCache.some(g => g.name === {json.dumps(mk)})")
                mid = js(f"() => (histGroupsCache.find(g => g.name === {json.dumps(mk)}) || {{}}).id")
                # 用 histGroupOps 把第 seg..seg+mv 条（互不重叠段）移到该组；
                # 不经底栏 UI（js 改 hdSel 不触发底栏重绘，#hdBtnMove 可能没渲染）
                js(f"""async () => {{
                    const ids = hdVisibleItems().slice({seg}, {seg + mv}).map(x => x.run_id);
                    await histGroupOps([{{op:'move', id: {json.dumps(mid)}, run_ids: ids}}]);
                }}""")
                seg += mv
                wait(f"() => hdBuckets().some(b => b.key === {json.dumps(mid)} && b.count === {mv})")
            a_id = js("() => (histGroupsCache.find(g => g.name === '合并源 A') || {}).id")
            b_id = js("() => (histGroupsCache.find(g => g.name === '合并源 B') || {}).id")
            check("合并前两个组都在、条数 1+2", bool(a_id) and bool(b_id), f"{a_id}/{b_id}")
            click('.hd-side-foot button[onclick="hdMergeGroups()"]')
            check("合并弹窗开起来（默认不勾选，两个源组都在列表里）",
                  wait("() => !document.getElementById('gmModalRoot').hidden")
                  and js("() => document.querySelectorAll('#gmList input').length") >= 2
                  and js("() => document.querySelectorAll('#gmList input:checked').length") == 0)
            check("不勾选够 2 个就点合并 → 报错「至少勾选 2 个组」",
                  (pg.click("#btnGmConfirm"), "至少勾选 2 个组" in pg.inner_text("#gmMsg")))
            pg.check(f'#gmList input[data-gid="{a_id}"]')
            pg.check(f'#gmList input[data-gid="{b_id}"]')
            pg.fill("#gmTarget", "")                              # 留空 = 并入第一个勾选组（A，先建先列）
            click("#btnGmConfirm")
            check("留空目标名 → 并入 A：A 剩 3 条（1+2）、B 消失、弹窗收起",
                  wait(f"() => hdBuckets().find(x => x.key === {json.dumps(a_id)})?.count === 3")
                  and js(f"() => !histGroupsCache.some(g => g.id === {json.dumps(b_id)})")
                  and js("() => document.getElementById('gmModalRoot').hidden"))
            # 第二个行为：自定义目标名 = 新建承接。再造一个 C 组（1 条）当第二源，
            # 与 A 合并——不碰「冒烟组」（[6] 拖拽还要用它）
            click('.hd-side-foot button[onclick="hdNewGroup()"]')
            wait("() => !document.getElementById('uiDialogRoot').hidden")
            pg.fill("#uiDialogInput", "合并源 C")
            click("#uiDialogOk")
            wait("() => histGroupsCache.some(g => g.name === '合并源 C')")
            c_id = js("() => (histGroupsCache.find(g => g.name === '合并源 C') || {}).id")
            js(f"""async () => {{
                const ids = hdVisibleItems().slice(3, 4).map(x => x.run_id);
                await histGroupOps([{{op:'move', id: {json.dumps(c_id)}, run_ids: ids}}]);
            }}""")
            wait(f"() => hdBuckets().some(b => b.key === {json.dumps(c_id)} && b.count === 1)")
            click('.hd-side-foot button[onclick="hdMergeGroups()"]')
            wait("() => !document.getElementById('gmModalRoot').hidden")
            pg.check(f'#gmList input[data-gid="{a_id}"]')
            pg.check(f'#gmList input[data-gid="{c_id}"]')
            pg.fill("#gmTarget", "合并目标组")                     # 自定义目标名 = 新建
            click("#btnGmConfirm")
            check("自定义目标名：新建「合并目标组」承接 4 条（A3+C1）、A/C 都消失",
                  wait("() => { const g = histGroupsCache.find(x => x.name === '合并目标组');"
                       " return g && hdBuckets().find(b => b.key === g.id)?.count === 4; }")
                  and js("() => !histGroupsCache.some(g => ['合并源 A','合并源 C'].includes(g.name))"))
            # 清场：把「合并目标组」解散，别影响后续断言的左栏形状
            # （「冒烟组」不动——[6] 拖拽归组还要用它）
            tid = js("() => (histGroupsCache.find(g => g.name === '合并目标组') || {}).id")
            if tid:
                click(f'.hd-gitem[data-bkt="{tid}"] [data-gop]')
                wait("() => !document.getElementById('hdMenu').hidden")
                click("#hdMenu .hd-mitem[data-mi='dissolve']")
                wait("() => !document.getElementById('uiDialogRoot').hidden")
                click("#uiDialogOk")
                wait(f"() => !histGroupsCache.some(g => g.id === {json.dumps(tid)})")
        else:
            print("  （记录不足 3 条，跳过合并组实测；按钮排布已验）")

        print("\n[6] 拖拽归组：抓起一行投到左栏组名上（HTML5 DnD）")
        click("#hdSelNone")                                  # 拖单条：选集为空时只拖这一条
        click('.hd-gitem[data-all="1"]')
        drag_rid = rows_all[2]
        via = "drag_to"
        try:
            pg.locator(f'.hd-row[data-rid="{drag_rid}"]').drag_to(
                pg.locator(f'.hd-gitem[data-bkt="{gid}"]'), timeout=5000)
        except Exception:
            via = "synthetic"
            js("""(a, b) => {
                const src = document.querySelector(a), dst = document.querySelector(b);
                const dt = new DataTransfer();
                const mk = (t, n) => n.dispatchEvent(new DragEvent(t, {bubbles:true, cancelable:true, dataTransfer:dt}));
                mk('dragstart', src); mk('dragover', dst); mk('drop', dst); mk('dragend', src);
            }""", [f'.hd-row[data-rid="{drag_rid}"]', f'.hd-gitem[data-bkt="{gid}"]'])
        check("拖到组名上 → 该记录的 group_id 变成目标组",
              wait(f"() => (histById({json.dumps(drag_rid)}) || {{}}).group_id === {json.dumps(gid)}"),
              f"via {via}")
        gcount = js(f"() => {{const e = document.querySelector('.hd-gitem[data-bkt=\"{gid}\"]');"
                    " return e ? e.querySelector('.hg-c').innerText : null;}")
        check("左栏组计数 0 → 1", gcount == "1", str(gcount))
        check("映射文件里 assign 按组 id 指（组是实体）",
              groups_file()["assign"].get(drag_rid) == gid, str(groups_file()["assign"])[:60])
        print("\n[6b] 底栏「移动分组」下拉：选一个组＝立刻移进去（旧版是逼人输组名的弹窗）")
        mv_rid = js("() => histCache.map(h => h.run_id)"
                    ".find(id => !((histById(id) || {}).group_id)) || ''")
        check("找得到一条未分组记录做样本", bool(mv_rid), mv_rid)
        js(f"() => {{ hdSel.clear(); hdSel.add({json.dumps(mv_rid)}); hdRenderAll(); }}")
        check("底栏是下拉而不是按钮（旧 #hdBtnMove 退役）",
              pg.locator("#hdMoveSel").count() == 1 and pg.locator("#hdBtnMove").count() == 0)
        opts = pg.eval_on_selector_all("#hdMoveSel option",
                                       "els => els.map(o => o.textContent.trim())")
        check("下拉首项当标签、列全部组并带条数、末项新建",
              bool(opts) and opts[0] == "移动分组…"
              and any(t.startswith("冒烟组（") for t in opts)
              and opts[-1].startswith("＋ 新建分组"), str(opts)[:76])
        pg.select_option("#hdMoveSel", value=gid)
        check("选一个组即移动（不再弹确认框）",
              wait(f"() => (histById({json.dumps(mv_rid)}) || {{}}).group_id === {json.dumps(gid)}"))
        check("移动后下拉复位到占位（挂着刚选的组会让人以为没动）",
              js("() => document.querySelector('#hdMoveSel').value") == "")
        check("映射文件同步写入 assign", groups_file()["assign"].get(mv_rid) == gid,
              str(groups_file()["assign"])[:60])

        # 清掉 [5] 建的冒烟组（拖拽目标已验完）：解散后左栏回到只含未分组，后续断言稳定
        if gid:
            click(f'.hd-gitem[data-bkt="{gid}"] [data-gop]')
            wait("() => !document.getElementById('hdMenu').hidden")
            click("#hdMenu .hd-mitem[data-mi='dissolve']")
            wait("() => !document.getElementById('uiDialogRoot').hidden")
            click("#uiDialogOk")
            wait(f"() => !histGroupsCache.some(g => g.id === {json.dumps(gid)})")

        print("\n[7] 「全选当前」只选当前视图里看得见的行")
        click(f'.hd-gitem[data-bkt="{gid}"]')                # 只看这一组（1 条）
        click("#hdSelNone")
        n_vis = pg.locator("#hdList .hd-row").count()
        click("#hdSelVis")
        check(f"组内视图全选 = {n_vis} 条（不是全部 {n_rows} 条）",
              n_vis == 1 and js("() => hdSel.size") == n_vis, f"hdSel={js('() => hdSel.size')}")
        click('.hd-gitem[data-all="1"]')
        click("#hdSelNone")
        click("#hdSelVis")
        check(f"全部视图全选 = {n_rows} 条", js("() => hdSel.size") == n_rows)

        print("\n[8] 底栏动作条：勾 ≥2 → 对比可用 → state.cmp.runs")
        click("#hdSelNone")
        check_box(rows_all[0])
        check("只勾 1 条时「对比」禁用（规则来自 policy，不在按钮上硬编码）",
              pg.locator("#hdBtnCmp").is_disabled())
        check_box(rows_all[1])
        check("勾 2 条时可用，且按钮报数",
              not pg.locator("#hdBtnCmp").is_disabled() and "2 条" in pg.inner_text("#hdBtnCmp"),
              pg.inner_text("#hdBtnCmp"))
        check("底栏芯片把选中的条列出来（看不见的勾选不成立）",
              pg.locator("#hdFoot .hd-chip").count() == 2)
        click("#hdBtnCmp")
        check("生成对比：state.cmp.runs ≥ 2",
              wait("() => state.cmp && state.cmp.runs && state.cmp.runs.length >= 2", timeout=15000),
              js("() => state.cmp ? state.cmp.runs.length : 'no cmp'"))
        check("对比后抽屉关闭", wait("() => document.getElementById('histDrawerRoot').hidden"))
        check("卡片区显式展开（对比不依赖 state.points）",
              js("() => !document.getElementById('cardCard').hidden"))

        print("\n[9] 拼接是抽屉内的模式（旧 spliceModal 退役）")
        click("#histPick")
        wait("() => !document.getElementById('histDrawerRoot').hidden")
        tgt, srcd = rows_all[3], rows_all[4]
        tpath = os.path.join(RDIR, tgt + ".json")
        t_before = json.load(open(tpath, encoding="utf-8"))
        click(f'.hd-row[data-rid="{tgt}"] [data-more]')
        wait("() => !document.getElementById('hdMenu').hidden")
        rlabels = pg.eval_on_selector_all("#hdMenu .hd-mitem", "els => els.map(e => e.textContent.trim())")
        check("行 ⋯ 菜单：备注/改名/拼接来源/删除（载入钮已移除，载入走双击行）",
              rlabels == ["备注…", "存档模型改名…", "拼接来源…", "删除该条"], str(rlabels))
        click("#hdMenu .hd-mitem[data-mi='splice']")
        check("进入拼接模式：模式条出现 + hdMode=splice",
              js("() => hdMode") == "splice" and not pg.locator("#hdModeBar").is_hidden()
              and "拼接模式" in pg.inner_text("#hdModeBar"))
        check("目标自己不在来源列表里", pg.locator(f'#hdList .hd-row[data-rid="{tgt}"]').count() == 0)
        check("未选来源时「拼接进目标」禁用", pg.locator("#hdSpliceGo").is_disabled())
        click(f'.hd-row[data-rid="{srcd}"]')
        check("拼接选源态仍单击 = 选来源：底栏报来源名、按钮可用",
              not pg.locator("#hdSpliceGo").is_disabled()
              and pg.locator(f'#hdList .hd-row[data-rid="{srcd}"].src').count() == 1,
              pg.inner_text("#hdFoot").replace("\n", " ")[:60])
        click("#hdSpliceGo")
        check("确认后回到浏览模式、模式条收起",
              wait("() => hdMode === 'browse'") and pg.locator("#hdModeBar").is_hidden())
        t_after = json.load(open(tpath, encoding="utf-8"))
        check("目标存档真的合并了来源测点（results 只增不减 + splices 留痕）",
              len(t_after.get("results", [])) >= len(t_before.get("results", []))
              and len(t_after.get("splices", [])) == len(t_before.get("splices", [])) + 1,
              f"{len(t_before.get('results', []))}→{len(t_after.get('results', []))} 点 / "
              f"splices {len(t_before.get('splices', []))}→{len(t_after.get('splices', []))}")
        check("回执说出新增/覆盖点数",
              "拼接完成" in pg.inner_text("#statusText"), pg.inner_text("#statusText")[:52])

        print("\n[10] 删除：确认框逐条列名（危险动作不吃看不见的勾选）")
        click("#hdSelNone")
        del_rid = rows_all[-1]
        del_time = pg.inner_text(f'.hd-row[data-rid="{del_rid}"] .hd-time')   # 确认框正文里的短标题前缀
        check_box(del_rid)
        n_before = js("() => histCache.length")
        click("#hdBtnDel")
        wait("() => !document.getElementById('uiDialogRoot').hidden")
        title, body = pg.inner_text("#uiDialogTitle"), pg.inner_text("#uiDialogBody")
        check("确认框标题报条数", "删除 1 条" in title, title)
        check("正文列出要删的是哪条（不是只报个数）",
              "将删除" in body and del_time in body, body[:56].replace("\n", " "))
        click("#uiDialogOk")
        check("确认后记录数减 1", wait(f"() => histCache.length === {n_before - 1}"),
              f"{n_before}→{js('() => histCache.length')}")
        check("目录里的存档也少了 1 个", n_archives() == n_before - 1, str(n_archives()))
        check("被删的那条从列表里消失", pg.locator(f'#hdList .hd-row[data-rid="{del_rid}"]').count() == 0)

        print("\n[11] 真实仓库 results/ 零污染")
        after = dir_snapshot(REAL)
        check("文件集合与 size/mtime 全等（写与删都落在临时副本上）",
              after == BEFORE, f"{len(after)} 个文件")

        print("\n[12] 窄屏 360px：不横向溢出")
        pg.set_viewport_size({"width": 360, "height": 800})
        pg.wait_for_timeout(350)
        check("抽屉在窄屏下仍开着（下面量的是它的溢出，不是空页面的）",
              js("() => !document.getElementById('histDrawerRoot').hidden"))
        over = js("""() => {
            const r = document.documentElement;
            return r.scrollWidth - r.clientWidth;}""")
        check("抽屉开着时 360px 无横向溢出", over <= 0, f"溢出 {over}px")
        pg.set_viewport_size({"width": 1440, "height": 950})
        real = [e for e in errs if "favicon" not in e and "404" not in e]
        check("无 JS 错误", not real, " | ".join(real[:3]))
finally:
    proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass

print("\n截图/临时目录：", ROOT)
print("失败：", fails if fails else "无")
sys.exit(1 if fails else 0)
