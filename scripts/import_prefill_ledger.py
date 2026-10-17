#!/usr/bin/env python3
"""把历史存档里已测的 prefill 速率点导入 `results/prefill-ledger.v1`（ADR-0134）。

为什么需要：速率先验台账是新增文件，空表时单档复测的先验仍只剩探测点（≈2K tokens，
本机实测 ~1000 tok/s，256K 真值 3700）——等待期读数会被钉在 1/3.5。而同一台部署其实
早就在历次 run 里测过 16K/32K/64K/128K/256K，这些点就躺在 `results/*.json` 里。本脚本
按与引擎**同一套门槛**把它们捞进台账，让下一次复测立刻有真先验，而不是先跑一整条
ladder 才能把曲线养起来。

默认只打印将要导入什么（`--dry-run` 语义），`--write` 才落盘；已有台账不覆盖同档，
只补缺（按 ts 新旧：新锚点优先，与引擎写入侧同规则）。

用法（仓库根目录）：
    python3 scripts/import_prefill_ledger.py                 # 预演，不写
    python3 scripts/import_prefill_ledger.py --write         # 写入台账
    python3 scripts/import_prefill_ledger.py --write --force # 同档也覆盖
"""
from __future__ import annotations

import argparse
import glob
import math
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench   # noqa: E402  复用引擎的归户与入表口径（不复制一份判据）


def _points_of(run: dict) -> list[tuple[str, str, float, float, float]]:
    """一次存档 → [(model, scenario, x, rate, ts)]，门槛与引擎写入侧同源：
    conc=1、usage 真值、点内无超窗删减（`ovf_trim`）/无异常输出、未命中量
    ≥0.5×档位、agent 矩阵点（inst 字段）不要。"""
    ts = time.mktime(time.strptime(run.get("started_at") or "",
                                   "%Y-%m-%d %H:%M:%S"))
    out = []
    for p in run.get("results") or []:
        if p.get("concurrency") != 1 or not p.get("all_ok"):
            continue
        # agent 矩阵点不入表：引擎写入侧只挂在 `_run_point`（泛用路径），矩阵点
        # 的 prefill 是「未命中增量」口径、还混着缓存档，捞进来会得到几 tok/s 的
        # 假先验（实测 256K 档 3 tok/s）
        if (p.get("ovf_trim") or p.get("anomaly") or p.get("agent_inst")
                or p.get("scenario") == "agent" or p.get("kind") == "agent"):
            continue   # ctx_short 不排：横轴 x 取实测 prompt_tokens，短点即实长锚点
        ctx = p.get("ctx_target") or 0
        rate = p.get("prefill_net_tok_s") or p.get("prefill_tok_s")
        ptok = p.get("prompt_tokens") or 0
        if not (ctx > 0 and rate and rate > 0 and ptok >= ctx * 0.5):
            continue
        out.append((p.get("model") or "", p.get("scenario") or "",
                    float(ptok), float(rate), ts))
    return out


def _adopt(model: str, table: dict) -> int:
    """把 `模型|网关`（无部署字段）的旧键并入当前 config.json 解析出的部署指纹。

    旧档案跑在用户填写 deploy_label/quant/硬件之前，指纹天然退化；而引擎现在按
    完整指纹查表，不折叠就播种不上。折叠是**显式 opt-in**（--adopt-model），因为
    它等于宣称「那些 run 就是现在这套部署」——合错表比没表更坏。"""
    try:
        import server   # noqa: PLC0415  只有 --adopt-model 才需要读配置
        providers = server.load_providers()
    except Exception as e:   # noqa: BLE001
        print(f"  跳过 {model}：读 config.json 失败（{e}）")
        return 0
    moved = 0
    for key in [k for k in table if k.startswith(model + "|") and k.count("|") == 1]:
        gw = key.split("|", 1)[1]
        prov = next((p for p in providers
                     if gw in (p.get("gateway_urls") or [])), None)
        if prov is None:
            print(f"  跳过 {key}：config.json 里没有网关 {gw} 的 provider")
            continue
        info = server.match_deployments(prov, [model]) or {}
        if not any((info.get(model) or {}).values()):
            print(f"  跳过 {key}：该模型在 config.json 里没有部署声明可归户")
            continue
        new_key = bench._prefill_fingerprint(model, {
            "model_info": info,
            "model_max_ctx": {m: v["max_ctx"] for m, v in info.items()
                              if v.get("max_ctx")},
            "gateway_url": gw})
        if new_key == key:
            continue
        for scen, rows in table[key].items():
            table.setdefault(new_key, {}).setdefault(scen, []).extend(rows)
        del table[key]
        moved += 1
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--write", action="store_true", help="写入台账（默认只预演）")
    ap.add_argument("--force", action="store_true", help="同档也用存档值覆盖")
    ap.add_argument("--adopt-model", action="append", default=[], metavar="模型名",
                    help="把该模型**没有部署字段**的旧键并入当前 config.json 的指纹"
                         "（旧档案跑在填字段之前，指纹退化成「模型|网关」，不并入就"
                         "播种不上；只对确实是同一套部署的模型用——合错表比没表更坏）")
    args = ap.parse_args()

    path = os.path.join(args.results_dir, bench.PREFILL_LEDGER_FILE)
    table = bench._prefill_ledger_load(path)
    runs = 0
    added: list[str] = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                run = json.load(fh)
        except Exception:   # noqa: BLE001  坏档跳过：导入不该被一个半截文件卡住
            continue
        cfg = run.get("cfg") or {}
        pts = _points_of(run)
        if not pts:
            continue
        runs += 1
        for model, scenario, x, rate, ts in pts:
            key = bench._prefill_fingerprint(model, cfg)
            have = bench._prefill_ledger_seed(
                (table.get(key) or {}).get(scenario) or [])
            dup = any(abs(math.log2(c) - bench.math.log2(x))
                      < bench.PREFILL_LEDGER_BUCKET_LOG2 for c, _ in have)
            if dup and not args.force:
                continue
            bench._prefill_ledger_note(table, key, scenario, x, rate, ts)
            added.append(f"  {model} / {scenario} @{int(x)} tokens → {rate:.0f} tok/s"
                         f"  ({os.path.basename(f)}, {'覆盖' if dup else '新增'})")
    print(f"扫描 {args.results_dir}/*.json：{runs} 次运行产出先验点，"
          f"本次入表 {len(added)} 条")
    for line in added[:40]:
        print(line)
    if len(added) > 40:
        print(f"  …另有 {len(added) - 40} 条")
    for key, subs in sorted(table.items()):
        for scenario, rows in sorted(subs.items()):
            seed = bench._prefill_ledger_seed(rows)
            head = (f"台账 {key} / {scenario}：{len(seed)} 档（"
                    f"{int(seed[0][0])}→{int(seed[-1][0])} tokens）")
            est = bench.rate_prior(seed, 262144) if len(seed) >= 2 else None
            print(head + (f"，256K 先验 → {est:.0f} tok/s" if est else
                          "，档数不足、暂不外推"))
    for model in args.adopt_model:
        n = _adopt(model, table)
        print(f"并入当前部署指纹：{model} → {n} 条场景表（旧「模型|网关」空指纹键已折叠）")
    if not args.write:
        print("（预演：未写盘。加 --write 落盘）")
        return 0
    bench._ledger_save(path, table)
    print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
