"""「速率先验基线」面板（ADR-0140）的浏览器目检替身（可选脚本，**不进回归套件**）。

为什么单独一个脚本、不塞进 `ui_smoke_history.py`：那个脚本 [6][7] 有**旧有**失败（拖拽
段解散了要点的组，后面点空组名就崩），在它跑到 [10b] 之前就退出，我的断言永远轮不到执行
——检查挂在跑不到的地方等于没有检查。本脚本只起一个空 results 的服务、只验一件事，
十几秒跑完。

为什么需要真浏览器：面板是抽屉里的**就地换模式**（列表隐掉、面板接管主栏、确认框、
删完重渲），这些在 node 侧三条门禁与 §pure 里都只看得见"另一半"。

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
    live_key = next(iter(cur), "Qwen.SMOKE|dep-a|ctx262144")
    bogus = "Qwen.SMOKE-OLD|dep-z|ctx8192"
    path = os.path.join(RDIR, bench.PREFILL_LEDGER_FILE)
    bench._ledger_save(path, {
        f"{live_key}|http://legacy-entry:13080": {
            "code": [[4103.0, 1200.0, NOW - 3600],
                     [65000.0, 900.0, NOW - 7200],
                     [65100.0, 880.0, NOW - 7200]]},
        bogus: {"code": [[4100.0, 500.0, NOW - 86400 * 40]]}})

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 950})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url, wait_until="load")
        pg.wait_for_timeout(400)

        def js(code):
            return pg.evaluate(code)

        print("\n[1] 面板出现在历史抽屉里，且是就地换模式")
        js("() => openHistDrawer()")
        pg.wait_for_timeout(500)
        check("左栏有入口", pg.locator("#hdPriorBtn").count() == 1)
        js("() => hdPriorEnter()")
        pg.wait_for_timeout(700)
        check("模式条出现、记录列表隐掉、面板接管主栏",
              js("() => !document.getElementById('hdModeBar').hidden"
                 " && document.getElementById('hdList').hidden"
                 " && !document.getElementById('hdPrior').hidden"))
        check("列出两张表", pg.locator("#hdPrior .pl-tbl").count() == 2)

        print("\n[2] 口径与引擎同源：旧键归一、同档并档、会不会被用上")
        dep = js("() => [...document.querySelectorAll('#hdPrior .pl-dep')].map(e => e.textContent)")
        check("地址尾缀已剥掉（界面上没有 http://）", all("http://" not in d for d in dep), str(dep))
        check("同档两条原始读数并成一档（活表两枚锚点）",
              pg.locator("#hdPrior .pl-tbl").first.locator(".pl-chip").count() == 2)
        check("与当前登记一致的那张标「这次会用上」",
              pg.locator("#hdPrior .pl-badge.on").count() == 1)
        check("另一张标「这次不会用上」", pg.locator("#hdPrior .pl-badge.off").count() == 1)
        check("陈旧表报出天数（新鲜度升序，它在最后一张）",
              js("() => [...document.querySelectorAll('#hdPrior .pl-when')]"
                 ".some(e => /40 天前/.test(e.textContent))"))

        print("\n[3] 删一档：确认框 → 盘上少一整桶 → 备份 + 审计")
        js("() => { const c = document.querySelectorAll('#hdPrior .pl-tbl')[0]"
           "  .querySelectorAll('.pl-chip button'); c[c.length - 1].click(); }")
        pg.wait_for_timeout(300)
        check("删除前有确认框（不是点一下就删）",
              js("() => !document.getElementById('uiDialogRoot').hidden"))
        js("() => document.getElementById('uiDialogOk').click()")
        pg.wait_for_timeout(900)
        left = json.load(open(path, encoding="utf-8"))
        rows = [r for subs in left.values() for rr in subs.values() for r in rr]
        check("只剩 4K 与陈旧那条（64K 那一桶整桶没掉）",
              sorted(int(r[0]) for r in rows) == [4100, 4103], str(rows))
        check("删前自动备份已生成",
              any(f.startswith(bench.PREFILL_LEDGER_FILE + ".bak-") for f in os.listdir(RDIR)))
        audit_p = os.path.join(RDIR, bench.PREFILL_LEDGER_AUDIT)
        check("审计留痕一行已写", os.path.exists(audit_p))
        check("面板已重渲（活表剩一枚锚点）",
              pg.locator("#hdPrior .pl-tbl").first.locator(".pl-chip").count() == 1)

        print("\n[4] 删整张表 + 退出")
        js("() => document.querySelector('[data-pl-del][data-key^=\"%s\"]').click()" % bogus)
        pg.wait_for_timeout(300)
        check("删整张表是危险确认（红字按钮）",
              js("() => !document.getElementById('uiDialogRoot').hidden"))
        js("() => document.getElementById('uiDialogOk').click()")
        pg.wait_for_timeout(900)
        left = json.load(open(path, encoding="utf-8"))
        check("陈旧表整张消失", all(bogus not in k for k in left), str(list(left)))
        js("() => document.getElementById('hdPriorExit2').click()")
        pg.wait_for_timeout(300)
        check("退出后面板收起、列表回来",
              js("() => document.getElementById('hdPrior').hidden"
                 " && !document.getElementById('hdList').hidden"))

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
