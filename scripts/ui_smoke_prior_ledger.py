"""「测速基线」抽屉（ADR-0140 立功能，ADR-0142 改成顶栏入口 + 独立侧栏）的浏览器目检替身
（可选脚本，**不进回归套件**）。

为什么单独一个脚本、不塞进 `ui_smoke_history.py`：那个脚本 [6][7] 有**旧有**失败（拖拽
段解散了要点的组，后面点空组名就崩），在它跑到后半段之前就退出——检查挂在跑不到的地方
等于没有检查（`pitfalls/crosscut/21`）。本脚本只起一个空 results 的服务、只验一件事。

为什么需要真浏览器：入口是「顶栏图标 → 下拉菜单 → 独立抽屉」三层，菜单落位、抽屉滑出、
焦点圈定、确认框、删后重渲与底栏反馈都只在真浏览器里成立，node 侧三套门禁看不到。

跑法（需要 playwright + httpx；没装就跳过并返回 0，不影响 tests/ 的任何门禁）：

    python3 scripts/ui_smoke_prior_ledger.py
    UI_SMOKE_PORT=8611 python3 scripts/ui_smoke_prior_ledger.py

三个细节照旧：①**解释器分两把**——本脚本用系统 `python3`（`.venv` 里没 playwright），
它拉起的**服务端**用 `.venv/bin/python`；②顶层脚本，**别 import 它探路**；③临时目录里
只有我们造的台账，真实 `results/` 一个字节都不动（末尾用快照断言）。
"""
import json
import os
import subprocess
import sys
import tempfile
import time

try:
    import httpx
    from playwright.sync_api import sync_playwright
except Exception as _exc:                       # 缺依赖 = 跳过，不算失败
    print(f"SKIP：需要 playwright + httpx（{_exc}）")
    raise SystemExit(0)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("UI_SMOKE_PORT", "8601"))
PY = os.path.join(SRC, ".venv/bin/python")
if not os.path.exists(PY):
    PY = sys.executable
sys.path.insert(0, SRC)
import bench                                    # noqa: E402  造台账要用引擎自己的写表函数

ROOT = tempfile.mkdtemp(prefix="prior-smoke.")
RDIR = os.path.join(ROOT, "results")
REAL = os.path.join(SRC, "results")
os.makedirs(RDIR, exist_ok=True)


def snap(d):
    return {f: (os.stat(os.path.join(d, f)).st_size,)
            for f in sorted(os.listdir(d)) if f.endswith(".json") and not f.startswith(".")}


BEFORE = snap(REAL)
NOW = time.time()

LAUNCH = os.path.join(ROOT, "launch.py")
with open(LAUNCH, "w", encoding="utf-8") as fp:
    fp.write(
        "import sys; sys.path.insert(0, %r)\n"
        "import uvicorn, server\n"
        "server.RESULTS_DIR = %r\n"
        "uvicorn.run(server.app, host='127.0.0.1', port=%d, log_level='error')\n"
        % (SRC, RDIR, PORT))

proc = subprocess.Popen([PY, LAUNCH], cwd=SRC,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
url = f"http://127.0.0.1:{PORT}"
for _ in range(60):
    try:
        if httpx.get(url + "/api/config", timeout=1).status_code == 200:
            break
    except Exception:
        time.sleep(0.3)
else:
    proc.kill()
    sys.exit("服务没起来")

fails = []


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"　[{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


try:
    # 先从服务端问出「当前登记能算出的指纹」，用它造表才能同时验两件事：
    # ①旧键（带网关尾缀）在界面上归一 ②与登记一致的表会被标成「这次会用上」
    cur = httpx.get(url + "/api/prefill-ledger", timeout=5).json().get("current") or {}
    live_key = next(iter(cur), "gw:local|Qwen.SMOKE|dep-a|ctx262144")
    bogus = "gw:smoke-old|Qwen.SMOKE-OLD|dep-z|ctx8192"
    # 旧口径遗留表（键里没有网关段，ADR-0143 之前的形状）：引擎不采纳，但界面要
    # 报出来、也要删得掉（闷着不报 = 用户以为存过了；删不掉 = 盘上永久堆垃圾）
    legacy_key = "Qwen.SMOKE-LEGACY|dep-old|ctx8192"
    path = os.path.join(RDIR, bench.PREFILL_LEDGER_FILE)
    # 第四位是来源（ADR-0141）：活表两条来自两轮测速，陈旧表那条刻意不带来源，
    # 用来验「旧数据 N 条未标来源」徽标（改存档环境时这类读数搬不动）
    bench._ledger_save(path, {
        f"{live_key}|http://legacy-entry:13080": {
            "code": [[4103.0, 1200.0, NOW - 3600, "run-a"],
                     [65000.0, 900.0, NOW - 7200, "run-a"],
                     [65100.0, 880.0, NOW - 7200, "run-b"]]},
        bogus: {"code": [[4100.0, 500.0, NOW - 86400 * 40, ""]]},
        legacy_key: {"code": [[9000.0, 700.0, NOW - 86400 * 60, ""]]}})

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 950})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url, wait_until="load")
        pg.wait_for_timeout(400)

        def js(code):
            return pg.evaluate(code)

        print("\n[1] 入口：顶栏设置图标 → 菜单「测速基线」→ 独立侧边栏")
        check("历史抽屉里已不再有基线入口（ADR-0142 剥离）",
              pg.locator("#hdPriorBtn").count() == 0)
        pg.click("#btnTopSet")          # 真点一次：入口是顶栏图标按钮，不是内部函数
        pg.wait_for_timeout(250)
        check("下拉菜单出现且带 menu 语义",
              pg.locator("#topSetMenu[role=menu]").count() == 1
              and js("() => document.getElementById('btnTopSet').getAttribute('aria-expanded')") == "true")
        check("菜单挂在按钮下方且在视口内",
              js("() => { const m = document.getElementById('topSetMenu')"
                 "  .getBoundingClientRect(); return m.top > 0 && m.right <= innerWidth; }"))
        pg.click("#topMenuBaseline")
        pg.wait_for_timeout(700)
        check("独立抽屉滑出（自己一个 root，不在历史抽屉里）",
              js("() => !document.getElementById('baseDrawerRoot').hidden"
                 " && document.getElementById('baseDrawerRoot').classList.contains('open')"
                 " && document.getElementById('histDrawerRoot').hidden"))
        check("菜单已收起", pg.locator("#topSetMenu").count() == 0)
        check("列出两张表（遗留表不算表：它不参与预估）",
              pg.locator("#blPanel .pl-tbl").count() == 2)
        check("表头把网关内部名排出来（基线指纹首段，ADR-0143）",
              len(js("() => [...document.querySelectorAll('#blPanel .pl-gw')]"
                     ".map(e => e.textContent)")) >= 2
              and js("() => [...document.querySelectorAll('#blPanel .pl-gw')]"
                     ".some(e => /smoke-old/.test(e.textContent))"),
              str(js("() => [...document.querySelectorAll('#blPanel .pl-gw')]"
                     ".map(e => e.textContent)")))
        check("旧口径遗留表如实列出并给「删掉它」（引擎没用它，但不能闷着）",
              js("() => { const n = document.querySelector('#blPanel .pl-note');"
                 " return !!n && /旧口径遗留表/.test(n.textContent)"
                 "  && !!document.querySelector('[data-pl-del][data-stale]'); }")
              and legacy_key in str(js("() => document.getElementById('blPanel').innerText")))

        print("\n[2] 口径与引擎同源：旧键归一、同档并档、会不会被用上")
        dep = js("() => [...document.querySelectorAll('#blPanel .pl-dep')].map(e => e.textContent)")
        check("地址尾缀已剥掉（界面上没有 http://）", all("http://" not in d for d in dep), str(dep))
        rows0 = js("() => [...document.querySelectorAll('#blPanel .pl-tbl')[0]"
                   "  .querySelectorAll('tbody tr')].map(r => r.innerText)")
        check("同档两条原始读数并成一档（活表两行：4K 与 64K）", len(rows0) == 2, str(rows0))
        check("逐档是带表头的表格（数字列右对齐）",
              pg.locator("#blPanel .pl-tbl").first.locator(".pl-grid thead th").count() == 5)
        check("与当前登记一致的那张标「这次会用上」",
              pg.locator("#blPanel .pl-badge.on").count() == 1)
        check("另一张标「这次不会用上」",
              js("() => [...document.querySelectorAll('#blPanel .pl-badge.off')]"
                 ".some(e => /这次不会用上/.test(e.textContent))"))
        check("陈旧表报出天数（新鲜度升序，它在最后一张）",
              js("() => [...document.querySelectorAll('#blPanel .pl-when')]"
                 ".some(e => /40 天前/.test(e.textContent))"))
        check("来源轮数说出来（活表来自 2 轮）",
              js("() => [...document.querySelectorAll('#blPanel .pl-tbl')]"
                 ".some(e => /来自 2 轮测速/.test(e.textContent))"))
        check("没标来源的旧数据单独挂牌（它们搬不动，得让人看见）",
              js("() => [...document.querySelectorAll('#blPanel .pl-badge.off')]"
                 ".some(e => /旧数据 1 条未标来源/.test(e.textContent))"))

        print("\n[3] 删一档：确认框 → 盘上少一整桶 → 备份 + 审计 + 抽屉内反馈")
        js("() => { const t = document.querySelectorAll('#blPanel .pl-tbl')[0]"
           "  .querySelectorAll('tbody tr'); t[t.length - 1].querySelector('.pl-del').click(); }")
        pg.wait_for_timeout(300)
        check("删除前有确认框（不是点一下就删）",
              js("() => !document.getElementById('uiDialogRoot').hidden"))
        js("() => document.getElementById('uiDialogOk').click()")
        pg.wait_for_timeout(900)
        left = json.load(open(path, encoding="utf-8"))
        rows = [r for k, subs in left.items() if k.startswith("gw:")
                for rr in subs.values() for r in rr]
        check("只剩 4K 与陈旧那条（64K 那一桶整桶没掉）",
              sorted(int(r[0]) for r in rows) == [4100, 4103], str(rows))
        check("删别的表不得顺手把遗留表也抹了（它只是不被采纳，不是该被删）",
              legacy_key in left and len(left[legacy_key]["code"]) == 1)
        check("删前自动备份已生成",
              any(f.startswith(bench.PREFILL_LEDGER_FILE + ".bak-") for f in os.listdir(RDIR)))
        check("审计留痕一行已写", os.path.exists(os.path.join(RDIR, bench.PREFILL_LEDGER_AUDIT)))
        check("面板已重渲（活表剩一行）",
              pg.locator("#blPanel .pl-tbl").first.locator("tbody tr").count() == 1)
        check("反馈写在抽屉自己的底栏（遮罩外的主面板状态灯用户看不见）",
              "已删除" in js("() => document.getElementById('blMsg').textContent")
              and "已删除" not in js("() => document.getElementById('statusText').textContent"))

        print("\n[4] 删整张表 + Escape 关抽屉")
        js("() => document.querySelector('[data-pl-del][data-key^=\"%s\"]').click()" % bogus)
        pg.wait_for_timeout(300)
        check("删整张表是危险确认（红字按钮）",
              js("() => !document.getElementById('uiDialogRoot').hidden"
                 " && /删除/.test(document.getElementById('uiDialogOk').className"
                 "  + document.getElementById('uiDialogOk').textContent)"))
        js("() => document.getElementById('uiDialogOk').click()")
        pg.wait_for_timeout(900)
        left = json.load(open(path, encoding="utf-8"))
        check("陈旧表整张消失", all(bogus not in k for k in left), str(list(left)))
        js("() => document.getElementById('blReload').focus()")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(450)
        check("Escape 关掉抽屉、焦点还给顶栏按钮",
              js("() => document.getElementById('baseDrawerRoot').hidden"
                 " && document.activeElement.id === 'btnTopSet'"))

        print("\n[4b] 删旧口径遗留表：只摘它自己")
        # 上面 Escape 关过抽屉，面板内容随抽屉一起清空了（下次打开要新鲜数据）→ 重开
        pg.click("#btnTopSet")
        pg.wait_for_timeout(250)
        pg.click("#topMenuBaseline")
        pg.wait_for_timeout(700)
        js("() => document.querySelector(`[data-pl-del][data-stale]`).click()")
        pg.wait_for_timeout(300)
        js("() => document.getElementById('uiDialogOk').click()")
        pg.wait_for_timeout(900)
        left2 = json.load(open(path, encoding="utf-8"))
        check("遗留表已从盘上摘掉", legacy_key not in left2, str(list(left2)))
        check("已采纳的表一条没少（前缀替换/整表删都不许误伤别的网关）",
              any(k.startswith("gw:") and k != bogus for k in left2), str(list(left2)))
        check("面板不再报遗留表",
              "旧口径遗留表" not in js("() => document.getElementById('blPanel').innerText"))

        print("\n[5] 真实仓库 results/ 零污染 + 无 JS 错误")
        check("文件集合与 size 全等", snap(REAL) == BEFORE, f"{len(BEFORE)} 个存档")
        real = [e for e in errs if "favicon" not in e and "404" not in e]
        check("无 JS 错误", not real, " | ".join(real[:3]))
        b.close()
finally:
    proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass

print("\n临时目录：", ROOT)
print("失败：", fails if fails else "无")
sys.exit(1 if fails else 0)
