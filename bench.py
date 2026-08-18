"""测速引擎：对 OpenAI 兼容网关做 prefill / decode 速度测试。

测量原理：
- 用场景语料池把 prompt 撑到目标上下文长度（0K~1M）：语料按固定种子拼成
  确定性长流、各档位从流首截取（嵌套前缀，校准可精确传递；代码场景优先用
  corpus/code/ 下的真实项目源码，按目录聚成模块块、只乱序块的次序，
  保住真实编程上下文的关联性结构），并注入随机 nonce 破坏 prefix cache，
  避免缓存命中与投机采样命中率注水导致的虚高。零输入档（ctx=0）不注入任何
  参考材料，一句话指令直接命题，测纯指令下的生成基线。
- 矩阵前先发探测请求（兼服务端预热），用真实 prompt_tokens 校准输入系数；
  并实测网络 RTT 基线，prefill 净口径 = prompt_tokens ÷ (TTFT − RTT)。
- Prefill 速度 = prompt_tokens / TTFT（另有扣 RTT 的净口径 prefill_net_tok_s）
- Decode 速度  = completion_tokens / (末 token 时间 - 首 token 时间)
- token 数优先取流式返回的 usage（stream_options.include_usage），
  网关/后端不支持时按「字符数 / chars_per_token」估算。
"""
from __future__ import annotations

import asyncio
import bisect
import json
import math
import os
import random
import time
from collections import deque
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# 场景定义：创意写作 / 代码生成
# ---------------------------------------------------------------------------

# 内置回退语料池（corpus/creative/ 存在时被 _load_creative_pool 取代）：
# 多段不同语料，拼接时乱序抽取。单段循环拼接的高重复文本会让模型照抄
# 参考材料，输出 ≈ 输入 → 投机采样（MTP/EAGLE 类）draft 命中率被人为拉满、
# decode 虚高；对 RadixAttention/稀疏注意力后端代表性也差。池化乱序后输出只能
# 正常逐字生成，MTP 按真实口径生效而不被注水（ADR-0005）
_CREATIVE_POOL = [
    "暮色四合，远山在雾霭中只剩下淡淡的轮廓。旅人沿着溪谷独行，鞋底踩过松针与碎石，"
    "发出细碎的声响。溪水在月光下泛着银色的波纹，偶尔有夜鸟掠过水面，惊起一圈圈涟漪。"
    "他想起多年前离开的故乡，想起母亲在灯下缝补的身影，想起那条通往山外的土路。",

    "码头的清晨总是从汽笛声开始的。老周把缆绳绕上桩柱，动作熟练得像是在系一个活结。"
    "集装箱在吊车臂下缓缓移动，海鸥围着桅杆盘旋。他数了数今天的货单，比昨天多了三倍，"
    "这意味着晚饭又要推迟到深夜。海风里有柴油和鱼腥味，那是他闻了三十年的味道。",

    "实验室的灯彻夜未眠。培养皿里的菌落呈现出意外的荧光绿，显微镜下的世界安静而汹涌。"
    "她在记录本上写下第三百七十二次观测结果，笔尖顿了顿——这一次的数据曲线终于偏离了"
    "对照组。窗外天快亮了，走廊尽头的自动售货机发出低沉的嗡嗡声，像某种耐心的陪伴。",

    "古城墙的砖缝里长出了不知名的野草。导游举着小旗子走过，游客们举着手机拍照，"
    "只有角落里的老人一动不动。他是这里的看守人，守了四十年，知道哪块砖在哪年塌过，"
    "哪段墙下埋着清代的瓷片。他说城墙是有记忆的，只是大多数人走得太快，听不见。",

    "雨下了整整一个星期。站台上的积水映着信号灯，红红绿绿地晃动。列车进站的瞬间，"
    "风把雨丝吹成斜的，打湿了她的风衣下摆。车厢里暖气很足，玻璃窗上很快蒙起白雾，"
    "她伸出手指，在雾气里写了一个字，又在列车启动前把它擦掉了。",

    "食堂的蒸汽在中午达到顶峰。大师傅抡着铁勺敲打锅沿，排队的人群向前挪动。"
    "角落那张桌子永远坐着同一个人，面前一碗素面，一本书，吃到汤凉了也不抬头。"
    "据说他是退休的物理教授，正在写一本谁也看不懂的书，已经写了七年。",
]

# 代码语料按"完整文件"粒度池化：每段是一个自洽模块（内部连贯、有标识符复用，
# 逼近真实 repo 拼接），乱序只发生在文件边界上
_CODE_POOL = [
    '''
import time
from collections import OrderedDict

class LRUCache:
    """简单的 LRU 缓存实现，用于演示数据结构与类型注解。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str):
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str, value) -> None:
        if key in self.items:
            self.items.move_to_end(key)
        self.items[key] = value
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
'''.strip(),

    '''
import asyncio
from typing import Awaitable, Callable

async def worker_pool(tasks: list[Callable[[], Awaitable]], workers: int = 4):
    """并发度受限的异步任务池：最多 workers 个任务同时执行。"""
    sem = asyncio.Semaphore(workers)
    results = [None] * len(tasks)

    async def run_one(i: int, fn) -> None:
        async with sem:
            results[i] = await fn()

    await asyncio.gather(*(run_one(i, fn) for i, fn in enumerate(tasks)))
    return results
'''.strip(),

    '''
def binary_search(arr: list[int], target: int) -> int:
    """标准二分查找，返回目标下标，未找到返回 -1。"""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def chunk(iterable, size: int):
    """把可迭代对象按 size 分块，最后一块可能不足。"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
'''.strip(),

    '''
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    """命令行入口：解析输入目录、并发数与输出格式。"""
    p = argparse.ArgumentParser(prog="imgtool", description="批量图片处理工具")
    p.add_argument("src", type=Path, help="输入目录")
    p.add_argument("-j", "--jobs", type=int, default=4, help="并发任务数")
    p.add_argument("--format", choices=["png", "webp", "jpeg"], default="webp")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    return p

def main() -> None:
    args = build_parser().parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"目录不存在: {args.src}")
    print(f"扫描 {args.src}，输出格式 {args.format}，并发 {args.jobs}")
'''.strip(),

    '''
import functools
import time

def retry(times: int = 3, delay: float = 0.1):
    """带线性退避的重试装饰器。"""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    time.sleep(delay * (i + 1))
            raise last_exc
        return wrapper
    return deco
'''.strip(),
]

SCENARIOS: dict[str, dict[str, Any]] = {
    "creative": {
        "label": "创意写作",
        "system": "你是一位才华横溢的小说家，擅长创意写作，文风细腻，想象力丰富，善于刻画场景与人物情感。",
        "instruction": (
            "请基于上面的参考材料展开想象，写一篇约1500字的散文或短篇故事，"
            "要求有场景描写、人物对话和情感递进，语言优美流畅。"
        ),
        "zero_instruction": (
            "请写一篇约800字的散文，题材自定，"
            "要求有场景描写与情感递进，语言优美流畅。"
        ),
        "filler_pool": _CREATIVE_POOL,
        "in_cpt": 1.5,             # 输入系数先验：中文散文实测 ~1.5 字符/token（探测校准前的兜底，ADR-0006）
        "out_cpt": 1.5,            # 输出估算：中文 ~1.5 字符/token（仅 usage 缺失时的兜底）
        "max_tokens_default": 1024,
    },
    "code": {
        "label": "代码生成",
        "system": "你是一位资深软件工程师，精通 C/C++ 与 Python，熟悉推理引擎与系统设计，代码规范，注重注释与边界情况处理。",
        "instruction": (
            "请基于上面的参考材料（某真实开源推理引擎项目的部分源码），"
            "实现一个与参考代码风格一致的功能模块：包含数据结构、核心算法与单元测试，"
            "需有注释并考虑边界情况。"
        ),
        "zero_instruction": (
            "请用 Python 开发一个命令行待办事项（Todo）应用：支持增删改查与"
            "本地持久化存储，代码完整可运行，并有必要注释。"
        ),
        "filler_blocks": None,     # 导入后由 _load_code_pool() 填充（模块块级；corpus 优先，内置池兜底）
        "in_cpt": 3.9,             # 输入系数先验：llama.cpp C/C++ 实测 ~3.9 字符/token（探测校准前的兜底，ADR-0006）
        "out_cpt": 3.4,            # 输出估算：C/C++ ~3.4 字符/token（仅 usage 缺失时的兜底）
        "max_tokens_default": 1024,
    },
}

# 0 = 零输入档：一句话指令直接命题，测纯指令生成基线（ADR-0005）
DEFAULT_CTX_LIST = [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
DEFAULT_CONCURRENCIES = [1, 2]

# 超窗保护：prompt + 输出必须同时装进模型上下文窗。目标档超过
# max_ctx − max_tokens − 本余量时直接跳过——余量兜住分词漂移
# （构造按校准系数命中目标，实测偏差 ≤1%）与系统提示等隐式占用（ADR-0005）
CTX_HEADROOM = 1024

# decode 空窗校正上限（秒）：相邻内容块间隔超过此值视为传输/调度停滞
# （WireGuard 隧道抖动实测空窗 18~90s），校正口径从 decode 窗口中只保留
# 上限内的部分；正常 decode 间隔（200 tok/s → 5ms，5 tok/s → 200ms，
# 云端调度停顿 ≤1~2s）不受影响，只削病理拖尾
DECODE_GAP_CAP = 3.0


def _fmt_ctx(ctx: int) -> str:
    return "0K" if ctx <= 0 else f"≈{ctx // 1024}K"

CORPUS_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "corpus", "code")
CORPUS_CREATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "corpus", "creative")


def _load_code_pool() -> list[list[str]]:
    """代码场景语料：优先用 vendored 的真实项目源码（corpus/code/，llama.cpp b9934，
    MIT License）。真实 repo 的文件结构、标识符复用与跨文件引用让投机采样（MTP 类）
    按真实口径生效——合成片段无论怎么乱序都复现不了这种长程结构。

    返回**模块块**列表（块 = 同目录文件集合，块内按文件名排序，foo.h 与 foo.cpp
    天然相邻）：真实编程上下文的突出特征是关联性——同模块文件、头文件与实现
    互相引用；文件级乱序会把任意窗口切成数十个互不相关的片段，模型无法利用
    上下文复用标识符，MTP 接受率被系统性扭曲（ADR-0005）。
    目录缺失/为空时回退到内置合成模块池（每条自成一个模块块）。"""
    modules: dict[str, list[str]] = {}
    try:
        for root, _dirs, files in os.walk(CORPUS_CODE_DIR):
            for fn in sorted(files):
                if fn.upper().startswith("LICENSE"):
                    continue
                try:
                    with open(os.path.join(root, fn), encoding="utf-8",
                              errors="replace") as f:
                        text = f.read().strip()
                except OSError:
                    continue
                if len(text) >= 2000:   # 过短的文件撑不起长程结构，跳过
                    modules.setdefault(root, []).append(text)
    except OSError:
        pass
    blocks = [texts for _dir, texts in sorted(modules.items()) if texts]
    return blocks or [[s] for s in _CODE_POOL]


def _load_creative_pool() -> list[str]:
    """创意场景语料：优先用 vendored 的公版文学文本（corpus/creative/，清·曹雪芹
    《红楼梦》，Project Gutenberg License）。内置散文池仅 672 字符，4K 档即整池
    循环 ~9 遍、256K 档 ~585 遍——高重复填充让投机采样 draft 命中被人为拉满、
    decode 虚高（ADR-0005 判据）；83 万字符的语料把循环点推到 512K 档之后。
    文件每行一个语料块（150~260 字符、句末切块）；缺失/为空时回退内置池。"""
    pool = []
    try:
        for root, _dirs, files in os.walk(CORPUS_CREATIVE_DIR):
            for fn in sorted(files):
                if fn.upper().startswith(("LICENSE", "README")):
                    continue
                try:
                    with open(os.path.join(root, fn), encoding="utf-8",
                              errors="replace") as f:
                        pool.extend(line.strip() for line in f if len(line.strip()) >= 40)
                except OSError:
                    continue
    except OSError:
        pass
    return pool or list(_CREATIVE_POOL)


SCENARIOS["code"]["filler_blocks"] = _load_code_pool()
SCENARIOS["creative"]["filler_pool"] = _load_creative_pool()


def _make_stream(pool: list[str]) -> str:
    """语料池 → 一条确定性长流：固定种子乱序一次后顺序拼接。各上下文档位都从
    流首截取，点位内容互为嵌套前缀——上一档实测的「字符数 → prompt_tokens」
    对本档前缀是精确已知的，构造只需估算增量段，校准不再被每点随机抽样的
    文件配比抖动带着振荡（code 场景实测偏差曾达 ±30%，ADR-0005）。
    种子固定 → 多次运行内容一致，结果可跨次比较。"""
    parts = list(pool)
    random.Random(0xC0DE).shuffle(parts)
    return "\n\n".join(parts)


def _make_module_stream(blocks: list[list[str]]) -> str:
    """模块块 → 一条确定性长流：固定种子只乱序**块的次序**，块内文件保持
    文件名排序（同模块文件、头文件与实现相邻——真实编程上下文的关联性结构，
    ADR-0005）。任意 ≥模块尺度的窗口内部连贯，整条流仍有跨模块多样性
    （小上下文档位不锁定单一模块）；确定性、完整文件粒度与嵌套前缀性质
    与 _make_stream 一致。"""
    parts = [list(b) for b in blocks]
    random.Random(0xC0DE).shuffle(parts)
    return "\n\n".join(s for b in parts for s in b)


for _sc in SCENARIOS.values():
    if _sc.get("filler_blocks"):
        _sc["filler_stream"] = _make_module_stream(_sc["filler_blocks"])
    else:
        _sc["filler_stream"] = _make_stream(_sc.get("filler_pool") or [])


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _fill(stream: str, need_chars: int) -> str:
    """从确定性语料流头部截取 need_chars；流不够长（超大上下文档）时整流循环
    续接——内容仍确定，嵌套前缀性质保持。"""
    if need_chars <= 0 or not stream:
        return ""
    if need_chars > len(stream):
        stream = stream * (need_chars // len(stream) + 1)
    return stream[:need_chars]


def build_messages(scenario: str, target_tokens: int, cpt: float, nonce: str,
                   filler_chars: int | None = None):
    """构造一条目标长度约 target_tokens 的对话。
    返回 (messages, est_tokens, 填充字符数)。
    target_tokens <= 0 为零输入档：不注入参考材料，一句话指令直接命题。
    filler_chars 显式指定填充长度（嵌套前缀精确记账用，ADR-0005），
    缺省按 target_tokens × cpt 估算。"""
    sc = SCENARIOS[scenario]
    system = sc["system"]
    if target_tokens <= 0:
        user = sc["zero_instruction"]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages, (len(system) + len(user)) / max(cpt, 0.1), 0
    instr = sc["instruction"]
    head = f"参考材料（编号 {nonce}）：\n"
    base_chars = len(system) + len(instr) + len(head) + 16
    need_chars = (filler_chars if filler_chars is not None
                  else max(0, int(target_tokens * cpt) - base_chars))
    filler = _fill(sc["filler_stream"], need_chars)
    user = f"{head}{filler}\n\n任务：{instr}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    est_tokens = (len(system) + len(user)) / max(cpt, 0.1)
    return messages, est_tokens, len(filler)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _net_elapsed(elapsed: float, rtt: float | None) -> float:
    """净耗时 = 实测耗时 − RTT 基线，扣减**封顶为实测值的一半**。
    RTT 基线取自 /models 元数据接口，与 /chat 推理调度路径不完全同构，
    基线抖动（同一网关实测在 75↔368ms 间漂移）可能超过小 prompt 的 TTFT——
    原 1e-3 下界曾把 0K 档净 prefill 放大到 79000 tok/s；封顶后净口径
    相对毛口径最多放大 2×，不再爆炸。"""
    return max(elapsed - (rtt or 0.0), elapsed / 2, 1e-6)


def rate_prior(pts: list[tuple[float, float]], ctx: float) -> float | None:
    """ctx 处的 prefill 速率先验：按实测速率曲线 [(ctx, rate), …]（ctx 升序）
    在 log2(ctx) 轴上做分段线性——区间内相邻点插值；超出上端沿末段斜率
    外推，并钳制在最近速率的 [0.35×, 1.3×] 内。
    prefill 速率随上下文长度先增后减（短档 GPU 利用不足、长档注意力平方
    成本主导），平推最近一个速率不能反映曲线形态；末段斜率外推让估值
    跟随已观测到的上升/下降趋势，钳制兜住噪声与拐点误判。"""
    pts = [p for p in pts if p[0] > 0]   # log 轴不接纳 0K（固定开销主导，非 ctx 标度）
    if not pts:
        return None
    if ctx <= pts[0][0]:
        return pts[0][1]
    for (c1, r1), (c2, r2) in zip(pts, pts[1:]):
        if ctx <= c2:
            if c2 == c1:
                return r2
            t = (math.log2(ctx) - math.log2(c1)) / (math.log2(c2) - math.log2(c1))
            return r1 + (r2 - r1) * t
    if len(pts) < 2:
        return pts[-1][1]
    (c1, r1), (c2, r2) = pts[-2], pts[-1]
    if c2 == c1:
        return r2
    slope = (r2 - r1) / (math.log2(c2) - math.log2(c1))
    est = r2 + slope * (math.log2(ctx) - math.log2(c2))
    return min(max(est, 0.35 * r2), 1.3 * r2)


# 聚合时的字段展示精度，与单点口径一致
_ROUND = {"ttft_s": 3, "ttft_net_s": 3, "prefill_tok_s": 1, "prefill_net_tok_s": 1,
          "decode_tok_s": 1, "decode_tok_s_adj": 1, "decode_total_tok_s": 1,
          "prompt_tokens": 0, "out_tokens": 0, "cache_hit_tokens": 0}


def _aggregate_reps(reps: list[dict]) -> dict:
    """多次复测聚合：标量取中位，reqs 明细保留 decode 中位的那一次（ADR-0006）。
    all_ok 多数决：过半复测成功才视为成功点。"""
    ok_reps = [p for p in reps if p["all_ok"]]
    pool = ok_reps or reps
    rep_point = sorted(pool, key=lambda p: p.get("decode_tok_s") or 0)[len(pool) // 2]
    point = dict(rep_point)
    for f, nd in _ROUND.items():
        vals = [p[f] for p in pool if p.get(f) is not None]
        if vals:
            m = _median(vals)
            point[f] = int(round(m)) if nd == 0 else round(m, nd)
    point["all_ok"] = len(ok_reps) * 2 >= len(reps)
    gaps = [p["max_gap_s"] for p in reps if p.get("max_gap_s")]
    if gaps:
        point["max_gap_s"] = max(gaps)   # 停滞诊断取各次复测最差
    point["n_reps"] = len(reps)
    point["reps"] = [{k: p.get(k) for k in
                      ("ttft_s", "prefill_tok_s", "decode_tok_s",
                       "decode_total_tok_s", "out_tokens", "all_ok")} for p in reps]
    return point


def _classify_http_error(status: int, body: str) -> str:
    """把裸 HTTP 错误翻译成可行动的提示（ADR-0008）。"""
    low = body.lower()
    if status in (502, 503, 504):
        return (f"HTTP {status} 网关错误：网关等后端响应超时或后端不可达"
                f"（此时模型服务通常仍在继续计算。调大网关侧超时：LiteLLM 的 "
                f"request_timeout、nginx 的 proxy_read_timeout；超长 prefill 如 "
                f"192K 档在低配硬件上可能超过 10 分钟）")
    if "context" in low and ("exceed" in low or "only" in low
                             or "context_length" in low or "context size" in low):
        # 各家超窗报文状态码不一：OpenAI 系 400，Kimi 用 401（invalid_authentication_error）
        return (f"HTTP {status} 超出模型上下文窗口：{body[:160]} "
                f"（在 config.json deployments 配 max_ctx 可自动跳过超限档位）")
    return f"HTTP {status}: {body[:300]}"


# ---------------------------------------------------------------------------
# 测速运行器
# ---------------------------------------------------------------------------

class BenchRun:
    """一次测速任务：models × scenarios × ctx_list × concurrencies 的测试矩阵。"""

    def __init__(self, cfg: dict[str, Any], results_dir: str | None = None):
        self.cfg = cfg
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + "".join(
            random.choices("abcdef0123456789", k=4))
        self.results_dir = results_dir
        # SSE 订阅扇出队列：每条事件连接独立队列（多标签页 / 断线重连残留的旧
        # 连接不再从同一队列竞争消费、互相抢事件，ADR-0009）；测试与内嵌调用
        # 同样向 subs 挂队列收事件（ADR-0009：原 self.q 生产路径无人消费、
        # 事件双份驻留内存，已移除）
        self.subs: set[asyncio.Queue] = set()
        self.history: list[dict] = []
        self.stop_flag = False
        self.results: list[dict] = []
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.finished_at: float | None = None   # 供服务端清理过期运行（RUNS 驻留内存）
        self.mock_seen = False   # 命中 mock 网关标记头 → 本次运行不落历史记录
        # 网络往返基线（GET /models 实测，5 次取最小）：TTFT 扣除该固定开销后
        # 才是服务端处理时间，云端短上下文的 prefill 读数否则被 RTT 摊薄失真（ADR-0006）
        self.rtt_s: float | None = None
        # 网关不支持 thinking 参数时置位（400 且报错提及 thinking），后续请求自动省略；
        # 服务端可经 cfg 下发进程级能力记忆（跨测速免重复 400 试探）
        self.thinking_unsupported = bool(cfg.get("thinking_unsupported", False))
        # 默认输出思考流的模型（探测请求不带参数、响应含 reasoning 时置录）：
        # 仅这些模型在 thinking=disabled 模式下需要显式下发禁用参数
        self.thinking_default_on: set[str] = set()
        # 模型级温度限制（400 且报错提及 temperature，如 Kimi K3 仅允许 0.6）：
        # 省略 temperature 参数走服务端默认，按模型锁定（ADR-0004）
        self.temperature_locked: set[str] = set(cfg.get("temperature_locked") or [])
        # (model, scenario) → 实测 chars/token（服务端真实 prompt_tokens 反推），
        # 首点后生效，后续更大上下文按校准值构造（ADR-0005）；按字符量加权滑动
        # 平均（cpt_calib_w 为累计字符权重，封顶保持对下游语料的适应性），
        # 小样本（<2000 字符）不采纳（ADR-0005）
        self.cpt_calib: dict[tuple[str, str], float] = {}
        self.cpt_calib_w: dict[tuple[str, str], float] = {}
        # 嵌套前缀精确记账：(model, scenario) → (已实测前缀的填充字符数, 真实
        # prompt_tokens)。语料流确定性嵌套，下一档构造 = 已知前缀 + 增量段，
        # 只需按增量段的滑动比例（cpt_marginal）估算增量字符数，偏差不再随
        # 档位放大（ADR-0005）
        self.prefix_kb: dict[tuple[str, str], tuple[float, float]] = {}
        self.cpt_marginal: dict[tuple[str, str], float] = {}
        self.cpt_marginal_w: dict[tuple[str, str], float] = {}
        # (model, scenario) → prefill 实测速率曲线 [(ctx, tok/s)]（ctx 升序）：
        # 探测（@实测 prompt_tokens）播种、conc=1 测速点逐点登记。prefill 等待期估值的
        # 先验经 rate_prior() 曲线插值/外推——速率随上下文先增后减，平推
        # 单一速率不能反映形态（用户实测 16K 峰值 vs 192K 减半）
        self.prefill_curve: dict[tuple[str, str], list[tuple[float, float]]] = {}
        # (model, scenario) → 实测输出 chars/token（usage completion_tokens 反推），
        # 实时 tick 与兜底 est_out 按此估算——固定系数与实际输出内容不符时
        # （如按中文 1.5 估、实际输出 ≈4 字符/token）实时读数会数倍虚高（ADR-0005）
        self.out_cpt_calib: dict[tuple[str, str], float] = {}
        # (model, scenario) → 实测 SSE 内容事件/token。llama.cpp/vLLM/DeepSeek 等
        # 每 token 推一个事件，事件数即真实 token 数（与输出内容无关，比字符系数
        # 估算准得多）；成批推送的网关按实测比校准。首个请求由探测播种（ADR-0007）
        self.chunk_calib: dict[tuple[str, str], float] = {}
        # API 路径前缀：LiteLLM/vLLM 用 "/v1"，DeepSeek 官方不带前缀
        # cfg.api_prefix: "auto"(默认探测) / "v1" / "none"
        mode = str(cfg.get("api_prefix") or "auto").lower()
        self.api_prefix = {"v1": "/v1", "none": "", "": None}.get(mode, None)
        self.prefix_locked = mode in ("v1", "none", "")

    def _note_prefill_rate(self, key: tuple[str, str], ctx: float, rate: float):
        """登记一个实测 prefill 速率点到曲线（保持 ctx 升序、同 ctx 覆盖）。"""
        pts = self.prefill_curve.setdefault(key, [])
        i = bisect.bisect_left([c for c, _ in pts], ctx)
        if i < len(pts) and pts[i][0] == ctx:
            pts[i] = (ctx, rate)
        else:
            pts.insert(i, (ctx, rate))

    async def emit(self, ev: dict):
        ev.setdefault("ts", time.time())
        ev["seq"] = len(self.history)     # 单调递增序号，前端用于丢弃重连回放的重复事件
        self.history.append(ev)
        for q in list(self.subs):   # 扇出给全部 SSE 订阅连接
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # 订阅队列有上限（服务端按 maxsize 建队）：慢客户端积压撞顶即摘除，
                # 补终止帧让该流结束——前端 EventSource 重连后按 seq 去重回放兜底
                self.subs.discard(q)
                self._force_terminal(q)

    @staticmethod
    def _force_terminal(q: asyncio.Queue):
        """队列满时挤掉最旧一条腾位，保证终止帧 None 送达（丢了 SSE 端会空转心跳）。"""
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def _close_streams(self):
        """向全部 SSE 订阅连接推送终止标记（gen 收到 None 退出）。"""
        for q in list(self.subs):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                self._force_terminal(q)

    def stop(self):
        self.stop_flag = True

    async def _interruptible(self, coro):
        """等待协程完成，stop_flag 置位时立即取消（0.2s 轮询）。
        探测/RTT 基线等不经 _run_point 任务集的路径也走这里，
        让「停止」在测速全链路即时生效；被中断返回 None。"""
        t = asyncio.create_task(coro)
        cancelled = False
        while True:
            done, _ = await asyncio.wait({t}, timeout=0.2)
            if done:
                break
            if self.stop_flag and not cancelled:
                t.cancel()
                cancelled = True
        try:
            return t.result()
        except asyncio.CancelledError:
            return None

    def _save(self):
        if self.mock_seen:   # 模拟数据不进历史记录
            return
        if not self.results_dir:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        cfg_safe = {k: v for k, v in self.cfg.items() if k not in ("api_key",)}
        path = os.path.join(self.results_dir, f"{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id,
                "started_at": self.started_at,
                "cfg": cfg_safe,
                "results": self.results,
            }, f, ensure_ascii=False, indent=1)

    # -- 主流程 -------------------------------------------------------------

    async def run(self):
        cfg = self.cfg
        # 终态三件套收口入 finally：前置段（系数转换、gateway_url 缺失）抛错也
        # 必须置 finished_at（server 依此回收）、存档、推 SSE 终止帧——否则任务
        # 静默死亡，订阅连接永等终止帧、_purge_finished_runs 永不回收
        try:
            # 权威运行配置随事件流首发（回放即得）：卡片/部署信息不依赖发起标签页的
            # 本地状态，多标签页进入正在运行的任务也能拿到正确的 provider（ADR-0009：cfg 事件）
            await self.emit({"type": "cfg", "started_at": self.started_at,
                             "cfg": {k: v for k, v in cfg.items() if k != "api_key"}})
            uc = cfg.get("url_choice")
            if uc:   # 多地址 provider：服务端已按顺序取首个可达地址，播报本次生效地址
                lat = f"{uc['latency_ms']}ms" if uc.get("latency_ms") is not None else "全部不可达，回退首选"
                await self.emit({"type": "status", "msg":
                    f"多地址自动选择网关 {uc['chosen']}（{lat}，{uc['candidates']} 选 1）"})
            repeats = max(1, min(5, int(cfg.get("repeats") or 1)))
            timeout = httpx.Timeout(connect=15, read=float(cfg.get("timeout_s", 600)),
                                    write=60, pool=30)
            limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
            headers = {"Content-Type": "application/json"}
            if cfg.get("api_key"):
                headers["Authorization"] = f"Bearer {cfg['api_key']}"
            url = cfg["gateway_url"].rstrip("/")
            async with httpx.AsyncClient(base_url=url, headers=headers,
                                         timeout=timeout, limits=limits) as client:
                # 网络往返基线：TTFT 含 RTT + 排队等固定开销，云端短上下文的
                # prefill 读数会被严重摊薄（实测 4K 档低估约一倍）；净口径 =
                # TTFT − RTT（ADR-0006）
                self.rtt_s = await self._measure_rtt(client)
                if self.rtt_s is not None:
                    await self.emit({"type": "status", "msg":
                        f"网络往返基线 {self.rtt_s * 1000:.0f} ms，prefill 净口径已扣除"})
                max_ctx_map = cfg.get("model_max_ctx") or {}
                for model in cfg["models"]:
                    if self.stop_flag:
                        break
                    max_ctx = max_ctx_map.get(model)
                    for scenario in cfg["scenarios"]:
                        if self.stop_flag:
                            break
                        # 输入系数先验按场景细分（ADR-0006）：分词比随语料构成
                        # 差异明显（实测中文散文 ~1.5、C/C++ ~3.9 字符/token），
                        # 单一默认 1.8 对代码场景差 2×；cfg.cpt 显式传入仍优先
                        base_cpt = float(cfg.get("cpt")
                                         or SCENARIOS[scenario]["in_cpt"])
                        # 矩阵前置探测：一次短上下文+极少输出的真实请求，兼作服务
                        # 预热与输入系数校准，首个正式点即按实测系数构造（ADR-0005）
                        if (model, scenario) not in self.cpt_calib:
                            await self._probe(client, model, scenario, base_cpt)
                        max_tokens = int(cfg.get("max_tokens")
                                         or SCENARIOS[scenario]["max_tokens_default"])
                        # 超窗保护上限：prompt + 输出 + 分词漂移余量必须装进而上下文窗，
                        # 超过的档位构造出来必然 400，直接跳过（ADR-0005）
                        ctx_limit = (max_ctx - max_tokens - CTX_HEADROOM) if max_ctx else None
                        for ctx in sorted(cfg["ctx_list"]):
                            if ctx_limit is not None and ctx > ctx_limit:
                                # 目标档位超出部署实际上限：构造出的 prompt 必然 400，
                                # 直接跳过并让前端把进度计为已处理（ADR-0005）
                                await self.emit({"type": "status", "msg":
                                    f"跳过 {model} / 上下文{_fmt_ctx(ctx)}：连同输出预算超出部署上限 "
                                    f"{max_ctx // 1024}K（config.json deployments.max_ctx）"})
                                await self.emit({"type": "point_skipped",
                                    "model": model, "scenario": scenario, "ctx": ctx,
                                    "count": len(cfg["concurrencies"])})
                                continue
                            for conc in sorted(cfg["concurrencies"]):
                                if self.stop_flag:
                                    await self.emit({"type": "stopped"})
                                    return   # 终态收口（finished_at/存档/终止帧）在 finally
                                label = SCENARIOS[scenario]["label"]
                                await self.emit({"type": "status", "msg":
                                    f"测试 {model} / {label} / 上下文{_fmt_ctx(ctx)} / 并发 {conc}"})
                                # 输入系数按实测校准，目标长度命中更准
                                cpt = self.cpt_calib.get((model, scenario)) or base_cpt
                                rep_points = []
                                for rep in range(repeats):
                                    if repeats > 1:
                                        await self.emit({"type": "status", "msg":
                                            f"　复测 {rep + 1}/{repeats}（取中位）"})
                                    pt = await self._run_point(client, model, scenario,
                                                               ctx, conc, cpt, rep)
                                    if self.stop_flag:
                                        break
                                    rep_points.append(pt)
                                if self.stop_flag:   # 停止时丢弃被中断的当前点，交给下一轮判停
                                    continue
                                point = (rep_points[0] if len(rep_points) == 1
                                         else _aggregate_reps(rep_points))
                                self.results.append(point)
                                await self.emit({"type": "point", "point": point})
            # 收尾判定：循环被停止（点在途取消后 continue 跳出）时发 stopped 而非 done
            if self.stop_flag:
                await self.emit({"type": "stopped"})
            else:
                await self.emit({"type": "done"})
        except Exception as e:  # noqa: BLE001
            await self.emit({"type": "error", "msg": f"测速任务异常: {e}"})
        finally:
            self.finished_at = time.time()
            try:
                self._save()   # 存档自身抛错也不能跳过 SSE 终止帧
            finally:
                self._close_streams()

    # -- 前置探测与网络基线 ----------------------------------------------------

    async def _measure_rtt(self, client: httpx.AsyncClient) -> float | None:
        """GET /models 实测网络往返基线（5 次取最小）。/models 是网关注册表查询，
        开销可忽略，测得的几乎纯是 RTT；云端排队走推理调度器、污染不到这个
        元数据接口，方差只剩网络抖动，min 即 RTT 下限的正确估计量（ADR-0006）。
        顺带完成 /v1 前缀探测。"""
        samples = []
        for _ in range(5):
            if self.stop_flag:
                break
            prefix = self.api_prefix if self.prefix_locked else "/v1"
            t = time.perf_counter()
            try:
                r = await self._interruptible(client.get(f"{prefix}/models"))
                if r is None:   # 停止打断基线测量
                    break
                if r.status_code == 404 and not self.prefix_locked and prefix == "/v1":
                    self.api_prefix, self.prefix_locked = "", True
                    t = time.perf_counter()   # 前缀回退重试重新计时：样本只计单次往返
                    r = await self._interruptible(client.get("/models"))
                    if r is None:
                        break
            except Exception:  # noqa: BLE001
                continue
            samples.append(time.perf_counter() - t)
        return min(samples) if samples else None

    async def _probe(self, client: httpx.AsyncClient, model: str, scenario: str,
                     base_cpt: float):
        """矩阵前置探测：一次短上下文 + 极少输出的真实请求，兼作服务端预热与
        输入系数校准。此前首个正式点必用默认系数构造（code 场景实测差 2 倍），
        探测后首点即准；64 个输出 token 同时为输出系数（实时读数口径）播种。
        成本可忽略（~2K 输入 + 64 输出 tokens）（ADR-0005）。"""
        label = SCENARIOS[scenario]["label"]
        cache_mode = str(self.cfg.get("cache") or "bust").lower()
        nonce = "stable" if cache_mode == "stable" else f"probe{random.randint(1000, 9999)}"
        msgs, _, filler_n = build_messages(scenario, 2048, base_cpt, nonce)
        r = await self._interruptible(
            self._one(client, model, msgs, scenario, 0, 2048, 1,
                      base_cpt, 64, quiet=True))
        if r is None:   # 停止打断探测：静默退出，run 循环随即发 stopped
            return
        if r.get("err") or not r.get("usage_real"):
            await self.emit({"type": "status", "msg":
                f"{model} / {label} 探测未完成（{r.get('err') or '无 usage'}），"
                f"沿用默认系数并按点后校准"})
            return
        # 默认思考流检测：探测请求不带 thinking 参数，若响应含 reasoning 内容
        # 说明模型默认思考（DeepSeek V4 类），后续才下发 thinking=disabled——
        # 不支持该参数的网关（vLLM/LiteLLM 系）从此一个 400 都不会收到
        if r.get("reason_chars") and model not in self.thinking_default_on:
            self.thinking_default_on.add(model)
            mode = str(self.cfg.get("thinking") or "auto").lower()
            if mode == "disabled":
                await self.emit({"type": "status", "msg":
                    f"检测到 {model} 默认输出思考流，" +
                    ("网关不支持 thinking 禁用参数，思考流将计入测速"
                     if self.thinking_unsupported else
                     "后续请求将下发 thinking=disabled（只测正文速度）")})
            elif mode == "auto":
                await self.emit({"type": "status", "msg":
                    f"检测到 {model} 默认输出思考流，按模型默认形态测速："
                    "思考流计入测速（只需正文速度请切换「禁用思考」）"})
        # prefill 等待期估值先验：探测实测的净 prefill 速率播种曲线，
        # x 坐标取实测 prompt_tokens（名义 2048 与真实值可有数倍差）
        rate = r.get("prefill_net_tok_s") or r.get("prefill_tok_s")
        if rate:
            self._note_prefill_rate((model, scenario), r["prompt_tokens"], rate)
        chars = sum(len(m["content"]) for m in msgs)
        calib = chars / r["prompt_tokens"]
        self.cpt_calib[(model, scenario)] = calib
        self.cpt_calib_w[(model, scenario)] = chars
        # 顺手播种嵌套前缀记账（探测字符量 ~3.7K 天然过 2000 字符门）：首个非零
        # 小档（<2000 字符、测速点不登记）也能按已知前缀+增量段构造（ADR-0005）
        self.prefix_kb[(model, scenario)] = (filler_n, r["prompt_tokens"])
        await self.emit({"type": "status", "msg":
            f"{model} / {label} 探测完成（兼服务端预热），"
            f"输入系数校准为 {calib:.2f} 字符/token"})

    # -- 单个测试点 -----------------------------------------------------------

    async def _run_point(self, client: httpx.AsyncClient, model: str, scenario: str,
                         ctx: int, conc: int, cpt: float, rep: int = 0) -> dict:
        base_seed = random.randint(100000, 999999)
        cache_mode = str(self.cfg.get("cache") or "bust").lower()
        # 嵌套前缀精确记账（ADR-0005）：语料流确定性嵌套，上一档已实测
        # 「前缀填充字符数 → 真实 prompt_tokens」，本档构造 = 已知前缀 + 增量段
        # × 增量比例滑动估计；无记账（首个非零点）回退全档累计系数估算
        key = (model, scenario)
        kb = self.prefix_kb.get(key)
        filler_chars = None
        if kb and ctx > 0:
            marg = self.cpt_marginal.get(key) or cpt
            filler_chars = max(0, round(kb[0] + (ctx - kb[1]) * marg))
        msgs_list = []
        filler_n = 0
        for i in range(conc):
            # bust=每请求随机编号破坏 prefix cache（冷测真实 prefill）；
            # stable=固定编号，重复/并发请求命中官方上下文缓存以省成本
            nonce = "stable" if cache_mode == "stable" else f"{base_seed + i * 137}"
            msgs, _est, filler_n = build_messages(scenario, ctx, cpt, nonce,
                                                  filler_chars=filler_chars)
            msgs_list.append(msgs)

        max_tokens = int(self.cfg.get("max_tokens")
                         or SCENARIOS[scenario]["max_tokens_default"])
        t_batch = time.perf_counter()
        # 在途请求用任务集管理：stop_flag 置位立即 cancel 全部（含 prefill 等待期），
        # 否则长 prefill 无法中断（ADR-0008）
        tasks = [asyncio.create_task(
                    self._one(client, model, msgs, scenario, i, ctx, conc, cpt,
                              max_tokens, rep=rep))
                 for i, msgs in enumerate(msgs_list)]
        reqs = []
        pending = set(tasks)
        while pending:
            if self.stop_flag:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                break
            # 带超时轮询：阻塞在 wait 期间无法感知 stop_flag（长 prefill 时
            # 首个请求迟迟不完成，停止会一直无反应），0.2s 轮询一次判停
            done, pending = await asyncio.wait(pending, timeout=0.2,
                                               return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    reqs.append(t.result())
                except asyncio.CancelledError:
                    pass
        reqs.sort(key=lambda r: r["req"])
        batch_time = time.perf_counter() - t_batch

        ok = [r for r in reqs if not r.get("err")]
        # prefill 估值先验逐点登记：conc=1 点的实测净速率入曲线（高并发点
        # 的批量效应会扭曲速率曲线，只取单发口径；0K 档是固定开销主导、
        # 非上下文标度，不入曲线）
        if conc == 1 and ctx > 0:
            rates = [(r.get("prefill_net_tok_s") or r.get("prefill_tok_s")) for r in ok]
            rates = [x for x in rates if x]
            if rates:
                self._note_prefill_rate((model, scenario), ctx,
                                        sum(rates) / len(rates))
        point = {
            "model": model, "scenario": scenario, "ctx_target": ctx, "concurrency": conc,
            "reqs": reqs, "all_ok": len(ok) == conc, "batch_time_s": round(batch_time, 3),
            "prompt_tokens": None, "ttft_s": None, "prefill_tok_s": None,
            "decode_tok_s": None, "decode_total_tok_s": None, "out_tokens": None,
            "ttft_net_s": None, "prefill_net_tok_s": None,
            "rtt_ms": round(self.rtt_s * 1000) if self.rtt_s is not None else None,
            "max_gap_s": None,
        }
        if ok:
            gaps = [r["max_gap_s"] for r in ok if r.get("max_gap_s")]
            if gaps:
                point["max_gap_s"] = max(gaps)   # 传输/调度停滞诊断：取请求中最差
            point["prompt_tokens"] = round(sum(r["prompt_tokens"] for r in ok) / len(ok))
            ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
            if ttfts:
                point["ttft_s"] = round(sum(ttfts) / len(ttfts), 3)
            pps = [r["prefill_tok_s"] for r in ok if r.get("prefill_tok_s")]
            if pps:
                point["prefill_tok_s"] = round(sum(pps) / len(pps), 1)
            # 净口径：TTFT 扣除 RTT 基线后的 prefill 速度，云端短上下文看这个
            tns = [r["ttft_net_s"] for r in ok if r.get("ttft_net_s")]
            if tns:
                point["ttft_net_s"] = round(sum(tns) / len(tns), 3)
            pns = [r["prefill_net_tok_s"] for r in ok if r.get("prefill_net_tok_s")]
            if pns:
                point["prefill_net_tok_s"] = round(sum(pns) / len(pns), 1)
            dcs = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
            if dcs:
                point["decode_tok_s"] = round(sum(dcs) / len(dcs), 1)
            dca = [r["decode_tok_s_adj"] for r in ok if r.get("decode_tok_s_adj")]
            if dca:
                point["decode_tok_s_adj"] = round(sum(dca) / len(dca), 1)
            if len(ok) == conc:
                total_out = sum(r.get("out_tokens") or 0 for r in ok)
                firsts = [r["first_abs"] for r in ok if r.get("first_abs")]
                lasts = [r["last_abs"] for r in ok if r.get("last_abs")]
                if total_out and firsts and lasts and max(lasts) > min(firsts):
                    # 聚合吞吐 = 总输出 tokens / (首个 token → 末个 token 的窗口)。
                    # 该定义仅在并发请求的 decode 区间充分重叠时有意义：长上下文
                    # prefill 被服务端串行化时 decode 实际先后发生，窗口混入
                    # prefill 等待期 → 断崖假象（ADR-0006），此时置空
                    decs = [r["last_abs"] - r["first_abs"] for r in ok
                            if r.get("first_abs") and r.get("last_abs")]
                    overlap = min(lasts) - max(firsts)
                    if conc == 1 or (decs and overlap > 0
                                     and overlap >= 0.5 * min(decs)):
                        point["decode_total_tok_s"] = round(
                            total_out / (max(lasts) - min(firsts)), 1)
            point["cache_hit_tokens"] = sum(r.get("cache_hit") or 0 for r in ok)
            fins = [r.get("finish") for r in ok if r.get("finish")]
            if fins:
                uniq = sorted(set(fins))
                point["finish"] = uniq[0] if len(uniq) == 1 else ",".join(uniq)
            point["out_tokens"] = round(sum(r.get("out_tokens") or 0 for r in ok) / len(ok))
        # 输入系数实测校准：用服务端真实 prompt_tokens 反推 chars/token。
        # llama.cpp 等分词与场景先验（in_cpt：创意 1.5 / 代码 3.9）仍可能有差，
        # 校准后后续上下文点按实测值构造（ADR-0005）。按字符量加权滑动平均
        # （权重封顶，保留对语料流下游段落配比变化的适应性）；小样本
        # （<2000 字符，如 0K 档纯指令）分词模板开销占比大、比例严重失真，
        # 采纳会带偏校准——实测 0K 档曾把 4K 点位构造带偏 20%+（ADR-0005）
        chars = sum(len(m["content"]) for m in msgs_list[0])
        reals = [r["prompt_tokens"] for r in ok if r.get("usage_real")]
        if chars >= 2000 and reals:
            real = sum(reals) / len(reals)
            obs = chars / real
            old = self.cpt_calib.get(key)
            w = min(self.cpt_calib_w.get(key, 0.0), 65536.0)
            self.cpt_calib[key] = ((old or obs) * w + obs * chars) / (w + chars)
            self.cpt_calib_w[key] = w + chars
            new = self.cpt_calib[key]
            if old is None or abs(new - old) / old > 0.02:
                await self.emit({"type": "status", "msg":
                    f"{model} / {SCENARIOS[scenario]['label']} 输入系数按实测校准为 "
                    f"{new:.2f} 字符/token"})
            # 嵌套前缀记账 + 增量段比例滑动平均：增量观测 = 增量字符 ÷ 增量
            # tokens，前缀部分的模板/指令开销在差分中自然抵消
            prev = self.prefix_kb.get(key)
            if prev and filler_n > prev[0] and real > prev[1]:
                m_chars = filler_n - prev[0]
                m_obs = m_chars / (real - prev[1])
                if 0.3 <= m_obs <= 20:
                    mw = min(self.cpt_marginal_w.get(key, 0.0), 65536.0)
                    self.cpt_marginal[key] = ((self.cpt_marginal.get(key) or obs) * mw
                                              + m_obs * m_chars) / (mw + m_chars)
                    self.cpt_marginal_w[key] = mw + m_chars
            self.prefix_kb[key] = (filler_n, real)
        return point

    # -- 单条流式请求 ----------------------------------------------------------

    def _est_out(self, model: str, scenario: str, n_chunks: int,
                 n_chars: float, out_cpt: float) -> float:
        """输出 tokens 实时估算：优先按 SSE 内容事件 ÷ 实测事件/token 比
        （每 token 一事件的服务上即真实值，不受中英/代码内容混合影响）；
        未校准时回退字符数 ÷ 输出系数（ADR-0007）。"""
        ratio = self.chunk_calib.get((model, scenario))
        if ratio:
            return n_chunks / ratio
        return n_chars / out_cpt

    async def _note_mock(self, resp: httpx.Response):
        if resp.headers.get("x-mock-server") and not self.mock_seen:
            self.mock_seen = True
            await self.emit({"type": "status",
                             "msg": "检测到 mock 网关，本次运行不写入历史记录"})

    async def _one(self, client: httpx.AsyncClient, model: str, messages: list,
                   scenario: str, req_i: int, ctx: int, conc: int,
                   cpt: float, max_tokens: int, rep: int = 0,
                   quiet: bool = False) -> dict:
        # quiet=True（探测请求）：不发 tick/实时事件，只走完整请求链路并返回结果
        sc = SCENARIOS[scenario]
        # 输出估算系数取实测校准值（首个请求由探测结果播种），固定系数与实际
        # 输出内容不符时实时读数会数倍虚高（实测 70+ vs 25+，ADR-0005）
        out_cpt = self.out_cpt_calib.get((model, scenario)) or sc["out_cpt"]
        tag = f"{model}|{scenario}|{ctx}|{conc}|rep{rep}|r{req_i}"
        est_prompt = round((len(messages[0]["content"]) + len(messages[1]["content"])) / max(cpt, 0.1))
        if not quiet:
            await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                             "ctx": ctx, "conc": conc, "req": req_i, "phase": "prefill",
                             "tokens": 0, "speed": 0, "elapsed": 0,
                             "est_prompt_tokens": est_prompt})

        payload = {
            "model": model, "messages": messages, "stream": True,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
        }
        # 测速用贪心输出：长度方差小、可复现；对 decode 速度本身影响可忽略。
        # 模型级温度限制（如 Kimi K3 仅允许 0.6）锁定后省略，用服务端默认值
        if model not in self.temperature_locked:
            payload["temperature"] = float(self.cfg.get("temperature", 0.0))
        # 思考模式：auto=不发参数（默认，跟随服务端/模型默认形态——有的模型
        # 默认思考效果更好，有的相反，按实际使用形态测速） /
        # disabled=只测正文速度 / enabled=思考流计入测速。
        # disabled 不再无条件下发禁用参数——探测先行（不带参数），响应含
        # reasoning 才说明模型默认思考（DeepSeek V4 类，思考流会占满
        # max_tokens 导致正文为空），仅此时下发 disabled；不支持 thinking
        # 参数的网关（vLLM/LiteLLM 系）因此一次 400 报错都不会收到
        thinking_mode = str(self.cfg.get("thinking") or "auto").lower()
        if not self.thinking_unsupported:
            if thinking_mode == "enabled":
                payload["thinking"] = {"type": "enabled"}
            elif thinking_mode == "disabled" and model in self.thinking_default_on:
                payload["thinking"] = {"type": "disabled"}
        # 端点路径：未锁定时先试 /v1，404 则回退无前缀（DeepSeek 风格）
        candidates = ([self.api_prefix] if self.prefix_locked
                      else ["/v1", ""])
        t0 = time.perf_counter()
        first = last = None
        max_gap = 0.0           # 相邻内容块最大空窗（秒，停滞诊断）
        adj_time = 0.0          # 空窗校正后的 decode 窗口（每段间隔封顶 DECODE_GAP_CAP）
        text_len = 0
        reason_len = 0          # reasoning_content（思考流）字符数，也计入测速
        n_chunks = 0            # 携带正文/思考内容的 SSE 事件数（≈token 事件）
        samples: deque[tuple[float, float]] = deque()   # (t, est_tokens) 滑窗采样
        usage = None
        finished = None
        last_emit = 0.0
        last_est = t0           # prefill 估值节流基准（~1s 一帧）

        async def est_tick():
            """prefill 估值帧（~1s 节流）：「若此刻完成」的速度估值（est_prompt
            ÷ (等待 − RTT)），随等待递减收敛于真实值。覆盖请求发出 → 首个内容
            token 全程（等响应头阶段也算）：有的网关链路（如 LiteLLM→vLLM）
            prefill 期间连响应头都不发，估值若只在读行循环里发，整个 prefill
            一帧都没有，前端 KPI 冻结在「–」直到首 token（ADR-0007）。"""
            nonlocal last_est
            if quiet:
                return
            now0 = time.perf_counter()
            # 阈值取 0.95s 而非 1.0s：与 1s 读行超时同值时，上一帧的 emit/循环
            # 开销会让本次检查以 0.999x 之差踩空，隔拍变 ~2s 一帧
            if now0 - last_est < 0.95:
                return
            net = _net_elapsed(now0 - t0, self.rtt_s)
            rate = rate_prior(self.prefill_curve.get((model, scenario)) or [], ctx)
            if rate:
                # 曲线先验封顶：等待期内估值 ≈ 该 ctx 的曲线速率（est_prompt ÷
                # 预期耗时），超出预期耗时才随等待衰减收敛——速率随上下文
                # 先增后减由曲线斜率外推反映，不再从天文数字双曲起步
                net = max(net, est_prompt / rate)
            await self.emit({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": ctx, "conc": conc,
                             "req": req_i, "phase": "prefill",
                             "speed": round(est_prompt / net, 1),
                             "elapsed": round(now0 - t0, 2)})
            last_est = now0

        async def enter_stream(ctx_mgr):
            """进入流式响应（等响应头），1s 超时轮询兼推估值帧——prefill 等待
            从请求发出那一刻就开始计时，不等响应头到达。"""
            ht = asyncio.ensure_future(ctx_mgr.__aenter__())
            try:
                while True:
                    got, _ = await asyncio.wait({ht}, timeout=1.0)
                    if got:
                        return ht.result()
                    await est_tick()
            finally:
                if not ht.done():
                    ht.cancel()

        try:
            resp_ctx = None
            resp = None
            for prefix in candidates:
                path = f"{prefix}/chat/completions"
                resp_ctx = client.stream("POST", path, json=payload)
                resp = await enter_stream(resp_ctx)
                await self._note_mock(resp)
                if resp.status_code == 404 and not self.prefix_locked and prefix == "/v1":
                    await resp_ctx.__aexit__(None, None, None)
                    self.api_prefix = ""
                    self.prefix_locked = True
                    await self.emit({"type": "status",
                                     "msg": "检测到网关不带 /v1 前缀（DeepSeek 风格），已切换端点"})
                    continue
                # 严格校验参数的网关：400 且报文点名某参数 → 省略该参数同端点
                # 重试并锁定。thinking 为网关级锁定（ADR-0004）；temperature 为
                # 模型级——如 Kimi K3 仅允许 0.6，省略后走服务端默认值（ADR-0004）
                for _param in ("thinking", "temperature"):
                    if resp.status_code != 400 or _param not in payload:
                        continue
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    if _param not in body.lower():
                        continue
                    await resp_ctx.__aexit__(None, None, None)
                    payload.pop(_param, None)
                    if _param == "thinking":
                        self.thinking_unsupported = True
                    else:
                        self.temperature_locked.add(model)
                    await self.emit({"type": "status", "msg":
                        f"网关不接受 {_param} 参数（{body[:80]}），已自动省略并重试"})
                    resp_ctx = client.stream("POST", path, json=payload)
                    resp = await enter_stream(resp_ctx)
                    await self._note_mock(resp)
                break
            if resp is None:
                raise RuntimeError("无可用端点")
            pending_line = None
            try:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise RuntimeError(_classify_http_error(resp.status_code, body))
                lines = resp.aiter_lines()
                while True:
                    # 手动迭代 + 1s 超时读行：prefill 等待期（首 token 前）也发 tick，
                    # 长上下文 prefill 动辄数分钟，不能全程无读数（ADR-0007）
                    if pending_line is None:
                        pending_line = asyncio.ensure_future(lines.__anext__())
                    got, _ = await asyncio.wait({pending_line}, timeout=1.0)
                    # prefill 估值 ~1s 节流（自请求发出全程覆盖，含等响应头阶段）：
                    # 与「有无行到达」解耦——高频注释心跳/无内容帧会饿死超时分支
                    if first is None:
                        await est_tick()
                    if not got:
                        continue
                    try:
                        line = pending_line.result()
                    except StopAsyncIteration:
                        break
                    finally:
                        pending_line = None
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    u = chunk.get("usage")
                    if isinstance(u, dict) and u:
                        usage = u
                    choices = chunk.get("choices") or [{}]
                    ch = choices[0] if choices else {}
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    rpiece = delta.get("reasoning_content")
                    if piece:
                        text_len += len(piece)
                    if rpiece:
                        reason_len += len(rpiece)
                    if piece or rpiece:
                        n_chunks += 1
                        now = time.perf_counter()
                        if first is None:
                            first = now
                            last_emit = now   # 首个 token 后才开始计时，避免速度读数爆炸
                            if not quiet:
                                # 首 token 到达 = 本请求 prefill 完成：TTFT 实测、
                                # prompt 取校准构造值，读数即刻刷新，不等整点
                                # 完成（请求级，ADR-0007）
                                ttft0 = now - t0
                                net0 = _net_elapsed(ttft0, self.rtt_s)
                                await self.emit({"type": "tick", "tag": tag, "model": model,
                                                 "scenario": scenario, "ctx": ctx, "conc": conc,
                                                 "req": req_i, "phase": "prefill",
                                                 "speed": round(est_prompt / net0, 1),
                                                 "ttft": round(ttft0, 3),
                                                 "elapsed": round(ttft0, 2)})
                        elif now - last > max_gap:
                            # 相邻内容块最大空窗：decode 窗口被传输/调度停滞拖尾时
                            # （如隧道拥塞），该点读数不可信——记录供前端警示
                            max_gap = now - last
                        if first is not None and last is not None and now > last:
                            adj_time += min(now - last, DECODE_GAP_CAP)
                        last = now
                        if not quiet and now - last_emit >= 0.4:
                            # token 估算优先按 SSE 内容事件数（每 token 一事件的服务上
                            # 即真实值，与输出内容无关）；未校准回退字符系数（ADR-0007）
                            est = self._est_out(model, scenario, n_chunks,
                                                reason_len + text_len, out_cpt)
                            # 滑窗差分速度（2s）：prefill 后首批事件常成批到达
                            # （服务端缓冲/投机采样），累计均值会把这批在 TTFT 窗口
                            # 生成的 token 摊进 decode 分母 → 起步虚高且收敛慢；
                            # 窗口差分把突发留在基准样本里，只反映当前速率（ADR-0007）
                            samples.append((now, est))
                            cutoff = now - 2.0
                            while len(samples) > 1 and samples[1][0] <= cutoff:
                                samples.popleft()
                            base_t, base_est = samples[0]
                            span = now - base_t
                            speed = ((est - base_est) / span if span >= 0.3
                                     else est / max(now - first, 1e-6))   # 起步兜底
                            await self.emit({
                                "type": "tick", "tag": tag, "model": model,
                                "scenario": scenario, "ctx": ctx, "conc": conc, "req": req_i,
                                "phase": "decode", "tokens": round(est, 1),
                                "speed": round(speed, 1),
                                "elapsed": round(now - t0, 2),
                                "ttft": round(first - t0, 3),
                                "thinking": bool(reason_len)})
                            last_emit = now
                    if ch.get("finish_reason"):
                        finished = ch["finish_reason"]
            finally:
                if pending_line is not None:   # 停止/异常退出时取消挂起的读行任务
                    pending_line.cancel()
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:300]
            if not quiet:
                await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        finally:
            # resp_ctx 创建后即纳入收口：enter_stream/400 重试段抛错或停止取消也
            # 必须关闭连接（404/400 分支的手动 aexit 幂等，重复关闭无害）
            if resp_ctx is not None:
                await resp_ctx.__aexit__(None, None, None)

        t_end = time.perf_counter()
        if first is None:
            msg = "未收到任何输出 token（思考模式下可能被 reasoning 占满，建议禁用思考或调大 max_tokens）"
            if not quiet:
                await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        est_out = (reason_len + text_len) / out_cpt
        out_tokens = (usage.get("completion_tokens")
                      if usage and usage.get("completion_tokens") else round(est_out))
        prompt_tokens = (usage.get("prompt_tokens")
                         if usage and usage.get("prompt_tokens") else est_prompt)
        # 输出系数实测校准：真实 completion_tokens 反推字符/token，后续请求的
        # 实时 tick 与兜底估算按此口径（探测请求已播种，首个正式点即准）
        if usage and usage.get("completion_tokens"):
            obs = (text_len + reason_len) / usage["completion_tokens"]
            key = (model, scenario)
            old = self.out_cpt_calib.get(key)
            if 0.3 <= obs <= 20 and (old is None or abs(obs - old) / old > 0.08):
                self.out_cpt_calib[key] = obs
                await self.emit({"type": "status", "msg":
                    f"{model} / {sc['label']} 输出系数按实测校准为 "
                    f"{obs:.2f} 字符/token（实时读数按此估算）"})
            # SSE 事件/token 比校准：内容事件数 ÷ 真实 completion_tokens。
            # 每 token 一事件的服务上比例为 1，实时读数与真实值完全一致（ADR-0007）
            if n_chunks:
                cobs = n_chunks / usage["completion_tokens"]
                cold = self.chunk_calib.get(key)
                if 0.2 <= cobs <= 5 and (cold is None
                                         or abs(cobs - cold) / cold > 0.05):
                    self.chunk_calib[key] = cobs
                    if cold is None:
                        await self.emit({"type": "status", "msg":
                            f"{model} / {sc['label']} 实时读数切换为 SSE 事件计数口径"
                            f"（{cobs:.2f} 事件/token）"})
        ttft = (first - t0) if first else None
        # 净口径：扣除 RTT 基线，逼近服务端纯处理时间（云端短上下文必看）；
        # 扣减封顶 TTFT 一半——基线抖动超过小 prompt TTFT 时净口径不爆炸
        ttft_net = (_net_elapsed(ttft, self.rtt_s)
                    if (ttft is not None and self.rtt_s is not None) else None)
        decode_time = (last - first) if (first and last and last > first) else None
        # 空窗校正口径：相邻块间隔逐段封顶 DECODE_GAP_CAP 后的窗口。传输/调度
        # 停滞（隧道抖动实测空窗 18~90s）只污染毛窗口，校正窗口仍反映服务端
        # 真实节奏；无停滞时两者相等
        adj = adj_time if (first and last and last > first and adj_time > 0) else None
        # 流末补发权威读数帧：token 数采信服务端 usage、速度为最终窗口均值。
        # 此前 tick 是滑窗估算快照（ADR-0007），与最终值口径不同（ADR-0007，口径收敛），
        # 收尾帧让实时 KPI/实时表停在准确值上
        if not quiet and decode_time and out_tokens:
            await self.emit({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": ctx, "conc": conc, "req": req_i,
                             "phase": "decode", "tokens": round(out_tokens),
                             "speed": round(out_tokens / decode_time, 1),
                             "elapsed": round(t_end - t0, 2),
                             "ttft": round(ttft, 3) if ttft is not None else None,
                             "thinking": bool(reason_len)})
        return {
            "req": req_i, "err": None,
            "first_abs": first, "last_abs": last,
            "cache_hit": (usage or {}).get("prompt_cache_hit_tokens") or 0,
            "cache_miss": (usage or {}).get("prompt_cache_miss_tokens") or 0,
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "ttft_net_s": round(ttft_net, 3) if ttft_net is not None else None,
            "prefill_tok_s": round(prompt_tokens / ttft, 1) if ttft else None,
            "prefill_net_tok_s": (round(prompt_tokens / ttft_net, 1)
                                  if ttft_net else None),
            "decode_tok_s": round(out_tokens / decode_time, 1) if decode_time else None,
            "decode_tok_s_adj": (round(out_tokens / adj, 1)
                                 if adj else None),   # 空窗校正口径
            "prompt_tokens": round(prompt_tokens),
            "out_tokens": round(out_tokens),
            "total_s": round(t_end - t0, 2),
            "max_gap_s": round(max_gap, 2) if max_gap else None,
            "usage_real": bool(usage),
            "finish": finished,
            "text_chars": text_len, "reason_chars": reason_len,   # 存档诊断用
        }
