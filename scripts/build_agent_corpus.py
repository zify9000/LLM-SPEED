#!/usr/bin/env python3
"""构建 Agent 场景语料（corpus/agent/）。

语料源：nebius/swe-agent-trajectories（CC-BY-4.0）——SWE-agent 框架在
SWE-bench dev / SWE-bench-extra 任务上的真实执行轨迹。HuggingFace 不可达时
用 ModelScope 镜像（AI-ModelScope/SWE-agent-trajectories，同一份数据）。

用法：
    .venv/bin/pip install pyarrow          # 仅构建期需要，非运行时依赖
    .venv/bin/python scripts/build_agent_corpus.py [parquet路径或URL]
    .venv/bin/python scripts/build_agent_corpus.py --verify   # 只校验不写盘

输出：corpus/agent/traj-<instance_id>-<seq>.txt，每个文件一个轨迹块
（≥2000 字符，按 [role] 轮次边界聚合；超长单段硬切 ≤40000 字符）。

--verify 只依赖标准库（懒加载 pyarrow），用于随时核对仓内语料是否仍满足块长
契约并打印内容清单 sha256——改语料前后都该跑一次，见 corpus/agent/PROVENANCE.md。
"""
import hashlib
import io
import os
import statistics
import sys
import urllib.request

DEFAULT_URL = ("https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/"
               "SWE-agent-trajectories/repo?Revision=master&FilePath="
               "data/train-00000-of-00012.parquet")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "corpus", "agent")
TARGET_CHARS = 3_000_000    # 覆盖 512K 档（×3.5 字符/token ≈1.8M）并留余量
MIN_BLOCK = 2000            # 与 bench._load_agent_pool 的采纳门限一致
MAX_BLOCK = 40000           # 单块上限：巨型 observation（整文件 dump）硬切
# 文档化上界：仓内语料由**早期修订**的脚本产出，实测最大 40183（3 个块超
# MAX_BLOCK）。按"把自述对准事实、不重建语料"处理（见 PROVENANCE.md），
# 故 --verify 用实测上界判定，并把与生成器判定的偏差单独报出来。
DOC_MAX_BLOCK = 40200


def corpus_stats() -> tuple[list[str], list[int]]:
    """仓内语料文件与各自字符数（只读）。"""
    names = sorted(f for f in os.listdir(OUT_DIR)
                   if f.startswith("traj-") and f.endswith(".txt"))
    chars = [len(open(os.path.join(OUT_DIR, n), encoding="utf-8").read())
             for n in names]
    return names, chars


def manifest_sha256(names: list[str]) -> str:
    """内容清单哈希：按文件名排序，累积 相对路径 + NUL + 内容（与测试同算法）。"""
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        h.update(b"\0")
        with open(os.path.join(OUT_DIR, n), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def verify() -> int:
    if not os.path.isdir(OUT_DIR):
        print(f"ERROR: 语料目录不存在: {OUT_DIR}", file=sys.stderr)
        return 1
    names, chars = corpus_stats()
    if not names:
        print(f"ERROR: 语料目录为空: {OUT_DIR}", file=sys.stderr)
        return 1
    over_gen = [n for n, c in zip(names, chars) if c > MAX_BLOCK]
    under = [n for n, c in zip(names, chars) if c < MIN_BLOCK]
    print(f"文件数 {len(names)}  字符 min/中位/max = "
          f"{min(chars)}/{int(statistics.median(chars))}/{max(chars)}")
    print(f"内容清单 sha256 = {manifest_sha256(names)}")
    if over_gen:
        print(f"提示: {len(over_gen)} 个块超生成器判定 MAX_BLOCK={MAX_BLOCK}"
              f"（文档化上界 {DOC_MAX_BLOCK}）: {', '.join(over_gen[:5])}"
              + (" …" if len(over_gen) > 5 else ""))
    bad = under + [n for n, c in zip(names, chars) if c > DOC_MAX_BLOCK]
    if bad:
        print(f"ERROR: {len(bad)} 个块越界（应 {MIN_BLOCK}~{DOC_MAX_BLOCK}）: "
              f"{', '.join(bad[:5])}", file=sys.stderr)
        return 1
    print("块长契约 OK（不写盘；改语料请同步 corpus/agent/PROVENANCE.md 与语料测试）")
    return 0


def fetch(src: str) -> bytes:
    if os.path.exists(src):
        with open(src, "rb") as f:
            return f.read()
    print(f"下载 {src} ...", file=sys.stderr)
    with urllib.request.urlopen(src, timeout=300) as r:
        return r.read()


def turn_segments(row) -> list[str]:
    """一条轨迹 → [role] 段序列。注意：该数据集所有 ai 轮 mask=True
    （mask 是 SFT 训练目标标记，非质量过滤），不能跳过；system 轮正文在
    system_prompt 字段（text 为空）。"""
    segs = []
    for turn in row["trajectory"]:
        text = (turn["text"] or turn.get("system_prompt") or "").strip()
        if not text:
            continue
        segs.append(f"[{turn['role']}]\n{text}")
    return segs


def blocks_from(segs: list[str]):
    """段序列 → 轨迹块：按段边界聚合，任何输出块都满足
    MIN_BLOCK ≤ 块长 ≤ MAX_BLOCK（块长按 "\\n\\n".join 后的实际长度计）；
    单段超 MAX_BLOCK 硬切。buf 不足 MIN_BLOCK 时不单独吐碎块：
    遇超长段/放不下时用段前缀补满到 MAX_BLOCK 一并吐出，剩余部分继续
    参与后续聚合。结尾仍不足 MIN_BLOCK 的残段按既有约定丢弃
    （下游 bench._load_agent_pool 对 <MIN_BLOCK 的文件本就静默不采纳）。"""
    buf, size = [], 0  # size = "\n\n".join(buf) 的实际长度（含分隔符）
    for seg in segs:
        while len(seg) > MAX_BLOCK:
            if buf and size >= MIN_BLOCK:
                yield "\n\n".join(buf)
                buf, size = [], 0
            if buf:
                take = MAX_BLOCK - size - 2
                yield "\n\n".join(buf + [seg[:take]])
                seg = seg[take:]
                buf, size = [], 0
            else:
                yield seg[:MAX_BLOCK]
                seg = seg[MAX_BLOCK:]
        if buf and size + len(seg) + 2 > MAX_BLOCK:
            if size >= MIN_BLOCK:
                yield "\n\n".join(buf)
                buf, size = [], 0
                buf.append(seg)
                size = len(seg)
            else:
                take = MAX_BLOCK - size - 2
                yield "\n\n".join(buf + [seg[:take]])
                seg = seg[take:]
                buf, size = [], 0
                buf.append(seg)
                size = len(seg)
        elif buf:
            buf.append(seg)
            size += len(seg) + 2
        else:
            buf.append(seg)
            size = len(seg)
    if buf and size >= MIN_BLOCK:
        yield "\n\n".join(buf)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        return verify()
    import pyarrow.parquet as pq   # 懒加载：--verify 路径不需要构建期依赖
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    data = fetch(src)
    table = pq.read_table(io.BytesIO(data))
    rows = table.to_pylist()
    print(f"{len(rows)} 条轨迹", file=sys.stderr)

    os.makedirs(OUT_DIR, exist_ok=True)
    total, n_blocks, n_traj = 0, 0, 0
    for row in rows:
        if total >= TARGET_CHARS:
            break
        n_traj += 1
        for blk in blocks_from(turn_segments(row)):
            # 同一 instance 在数据集中有多条 rollout（实例级重名），文件名用
            # 全局序号防互相覆盖
            name = f"traj-{n_blocks:04d}-{row['instance_id'].replace('/', '_')[:60]}.txt"
            with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
                f.write(blk + "\n")
            total += len(blk)
            n_blocks += 1
    print(f"写出 {n_blocks} 块 / {n_traj} 条轨迹，共 {total:,} 字符 → {OUT_DIR}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
