#!/usr/bin/env python3
"""把 `config.json` 里的旧版字段（`deployments[].max_ctx` / `kinds` / `kind`）折算成
`model_profiles` 并写盘——**离线手动版**（服务启动时会自动做同样的事）。

背景一句话：上限与类型的唯一事实源是 `provider.model_profiles`（ADR-0118），写在部署环境
里的这两个同名字段已废弃（ADR-0122）：解析不读、保存不写，检测到就自动折算并落盘。落盘
发生在服务启动时（`server._startup_profile_migration()`），所以正常情况下用不着本脚本。

那它还留着做什么：
  - **不起服务就能预演**：想知道某个文件会被改成什么样，跑一下看 diff，不动文件；
  - **在别的机器上手改配置后修一次**：例如运维直接编辑了 config.json 又塞回了旧字段；
  - 转换失败时（配置本身有问题被拒绝写入）这里会**当场把原因打印出来**，而启动路径只留
    一行 warning，容易看漏。

口径**不在本脚本里重写一份**：检测、折算、等价性守卫、备份、原子写全在
`server.migrate_legacy_profiles()`，本文件只负责命令行、diff 打印与退出码。

用法：
    python scripts/migrate_model_profiles.py                  # 预演：打印 diff，不写文件
    python scripts/migrate_model_profiles.py --apply          # 转换并写盘（先备份）
    python scripts/migrate_model_profiles.py --config X.json --apply   # 指定别的 config
退出码：0 无需改动或成功；2 预演发现可转换（给脚本化调用看的信号）；1 被拒绝（原因已打印）。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import server      # noqa: E402  复用服务端的检测/折算/写盘，禁止在本脚本里重写一份


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"),
                    help="要处理的 config.json（默认仓库根那份）")
    ap.add_argument("--apply", action="store_true",
                    help="写盘（默认只预演打印 diff）")
    args = ap.parse_args()
    path = os.path.abspath(args.config)
    if not os.path.exists(path):
        print(f"✗ 找不到配置文件：{path}")
        return 1
    server.CONFIG_PATH = path      # 服务端全部走这一个模块变量，切它就切了配置源

    try:
        result = server.migrate_legacy_profiles(dry_run=not args.apply)
    except Exception as e:         # noqa: BLE001 - 把拒绝原因原样交给人看
        print(f"✗ 拒绝转换：{e}\n  未改动 {path}")
        return 1

    if not result["changed"]:
        print(f"{path}：没有旧版 deployments.max_ctx/kinds，无需改动")
        return 0

    print(f"检测到旧版字段（deployments 里的 max_ctx/kinds/kind）：{', '.join(result['changed'])}")
    with open(path, encoding="utf-8") as f:
        old = f.read()
    diff = list(difflib.unified_diff(old.splitlines(True),
                                     result["text"].splitlines(True),
                                     "当前", "转换后"))
    sys.stdout.write("".join(diff))

    if not args.apply:
        print(f"\n预演结束：未改动 {path}。确认后加 --apply（写盘前会备份成 "
              f"{os.path.basename(path)}.<时间戳>.bak）")
        # 新形态长这样：给一眼能核对的摘要（provider → 模型 上限/类型）
        for p in json.loads(result["text"])["providers"]:
            for f in p.get("model_profiles") or []:
                kinds = "/".join(f.get("kinds") or ["llm"])
                print(f"    [{p['name']}] {f.get('max_ctx') or '不限上限':>8}  {kinds:<9} {f['model']}")
        return 2

    print(f"\n✓ 已写盘 {path}（备份：{os.path.basename(str(result['backup']))}）")
    print("  deployments 里的旧字段已丢弃（值已折算进 model_profiles）；"
          "上限与类型逐项与转换前相同")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
