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
import base64
import bisect
import io
import json
import math
import os
import random
import struct
import time
import wave
import zlib
from array import array
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

# Agent 语料兜底块（corpus/agent/ 缺失时启用）：SWE-agent 风格轨迹——
# [user] 任务 / [assistant] 思考+命令 / [observation] 工具输出 交替
_AGENT_POOL = [
    '''
[user]
We are solving the following GitHub issue:
repo: acme__config-parser-241
Title: Empty input crashes the TOML config parser
When the input file is empty (0 bytes), parse_config() raises IndexError
instead of returning an empty Config object. Steps to reproduce:
  echo -n "" > empty.toml && python -m acme.config empty.toml
Expected: an empty Config with defaults. Actual: IndexError: list index out of range.
[assistant]
I need to locate the parser entry point and reproduce the crash first.
Action: bash
Command: grep -rn "def parse_config" --include="*.py" .
[observation]
./acme/config/__init__.py:12:def parse_config(path):
./acme/config/parser.py:45:def parse_config(text):
[assistant]
Found two entry points. The traceback points at parser.py. Let me read it.
Action: str_replace_editor
Command: view
Path: /repo/acme/config/parser.py
[observation]
Here's the result of running `cat -n` on /repo/acme/config/parser.py:
    40	        return self._defaults.copy()
    41
    42
    43	def parse_config(text):
    44	    """Parse TOML text into a Config."""
    45	    lines = text.splitlines()
    46	    first = lines[0].strip()
    47	    if first.startswith("#"):
    48	        lines = lines[1:]
    49	    table = {}
    50	    for ln in lines:
    51	        key, _, value = ln.partition("=")
    52	        table[key.strip()] = value.strip()
    53	    return Config(table)
[assistant]
The bug is clear: line 46 indexes lines[0] without checking for an empty
file, so a 0-byte input raises IndexError. I'll guard the empty case.
Action: edit
Path: /repo/acme/config/parser.py
<<<
old:    lines = text.splitlines()
        first = lines[0].strip()
new:    lines = text.splitlines()
        if not lines:
        ...    return Config({})
        first = lines[0].strip()
>>>
[observation]
The file /repo/acme/config/parser.py has been edited. Here's what changed:
    45	    lines = text.splitlines()
    46	    if not lines:
    47	        return Config({})
    48	    first = lines[0].strip()
Review the changes and make sure they are as expected.
[assistant]
The guard is in place. Now verify with the reproduction steps.
Action: bash
Command: echo -n "" > /tmp/empty.toml && python -m acme.config /tmp/empty.toml
[observation]
Config({})
[assistant]
Empty input now returns an empty Config. Running the existing test suite to
make sure nothing regressed.
Action: bash
Command: python -m pytest tests/config/ -q
[observation]
43 passed in 1.12s
[assistant]
All tests pass. Submitting the fix.
Action: submit
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
        "echo_instruction": (
            "请输出参考材料末尾约1500字的改写润色版（保持情节与信息不变，"
            "逐句优化表达）。你已经开始作答，请从已输出内容之后直接继续，"
            "保持与原文一致、仅做润色；不要任何前言或说明。"
        ),
        "zero_instruction": (
            "请写一篇约800字的散文，题材自定，"
            "要求有场景描写与情感递进，语言优美流畅。"
        ),
        "filler_pool": _CREATIVE_POOL,
        "in_cpt": 1.5,             # 输入系数先验：中文散文实测 ~1.5 字符/token（探测校准前的兜底，ADR-0006）
        "out_cpt": 1.5,            # 输出估算：中文 ~1.5 字符/token（仅 usage 缺失时的兜底）
        "out_hint": "篇幅要求：全文控制在 {limit} 字以内，不得超出。",   # 输出长度引导（ADR-0016）
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
        "echo_instruction": (
            "请输出参考材料末尾约4000字符代码的完整重构版（修复明显问题、"
            "补充注释与边界处理）。你已经开始作答，请从已输出内容之后直接继续，"
            "与参考材料保持一致、仅做必要修改；不要前言、解释或 markdown 标记。"
        ),
        "zero_instruction": (
            "请用 Python 开发一个命令行待办事项（Todo）应用：支持增删改查与"
            "本地持久化存储，代码完整可运行，并有必要注释。"
        ),
        "filler_blocks": None,     # 导入后由 _load_code_pool() 填充（模块块级；corpus 优先，内置池兜底）
        "in_cpt": 3.9,             # 输入系数先验：llama.cpp C/C++ 实测 ~3.9 字符/token（探测校准前的兜底，ADR-0006）
        "out_cpt": 3.4,            # 输出估算：C/C++ ~3.4 字符/token（仅 usage 缺失时的兜底）
        "out_hint": "篇幅要求：回复总长度控制在 {limit} 字符以内，不得超出。",   # 输出长度引导（ADR-0016）
        "max_tokens_default": 1024,
    },
    "agent": {
        "label": "Agent 调用",
        "system": ("You are an autonomous software engineering agent solving GitHub "
                   "issues. You explore the repository, reason about the problem, and "
                   "edit code, one action at a time."),
        "instruction": (
            "The reference material above is a real execution trajectory of a SWE-agent "
            "solving GitHub issues (task description, tool calls, observations and code "
            "edits so far). Based on this trajectory, decide the next action: briefly "
            "analyze the current state, then output exactly one concrete action "
            "(a bash command or a file edit) with its expected outcome."
        ),
        "zero_instruction": (
            "You are assigned the GitHub issue: 'Empty input crashes the TOML config "
            "parser with IndexError instead of returning an empty Config'. Output your "
            "first action: a single bash command to locate the relevant code, with a "
            "one-sentence rationale."
        ),
        "filler_pool": _AGENT_POOL,  # 导入后由 _load_agent_pool() 覆盖（corpus 优先，内置块兜底）
        "in_cpt": 3.5,             # 输入系数先验：英文代码/日志/JSON 混合轨迹 ~3.5 字符/token（探测校准前的兜底）
        "out_cpt": 3.4,            # 输出估算：命令/补丁类英文 ~3.4 字符/token（仅 usage 缺失时的兜底）
        "out_hint": "Length limit: the entire reply must stay within {limit} characters.",   # 输出长度引导（ADR-0016）
        "max_tokens_default": 512,  # agent 单步动作短，512 足够覆盖思考+一个动作
    },
    "asr": {
        "label": "语音转写",
        "kind": "asr",
        "ladder": [5, 30, 60, 300, 900],   # 音频时长阶梯（秒），替代 ctx_list
        "unit": "秒音频",
        "out_cpt": 1.5,   # 存档诊断：转写输出字符/token 兜底估值
    },
    "ocr": {
        "label": "图像识别",
        "kind": "ocr",
        "ladder": [1, 4, 16],              # 单请求图片张数阶梯，替代 ctx_list
        "unit": "张图片",
        "system": "你是严谨的 OCR 引擎，逐字识别图像中的文字，按图片顺序输出，不遗漏、不臆造。",
        "instruction": "请识别上面 {n} 张图片中的全部文字，按图片顺序分块输出。",
        "out_hint": "篇幅要求：回复总长度控制在 {limit} 字符以内，不得超出。",
        "in_cpt": 3.0,             # 指令文本字符/token 先验（图片 tokens 按 px/750 单独估）
        "out_cpt": 1.5,            # 识别输出多为中文文本
        "max_tokens_default": 1024,
    },
    "tts": {
        "label": "语音合成",
        "kind": "tts",
        "ladder": [50, 200, 800, 3200],    # 合成文本长度阶梯（字符），替代 ctx_list
        "unit": "字文本",
        "speech_rate": 4.5,   # 语速先验（字/秒）：服务端不回可解析 wav 时估音频时长
        "out_cpt": 1.5,
    },
}

# 0 = 零输入档：一句话指令直接命题，测纯指令生成基线（ADR-0005）
DEFAULT_CTX_LIST = [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
DEFAULT_CONCURRENCIES = [1, 2, 5]

# 超窗保护：prompt + 输出必须同时装进模型上下文窗。目标档超过
# max_ctx − max_tokens − 本余量时直接跳过——余量兜住分词漂移
# （构造按校准系数命中目标，实测偏差 ≤1%）与系统提示等隐式占用（ADR-0005）
CTX_HEADROOM = 1024

# 部署上限超窗「贴边裁减」阈值：目标档连同输出预算超出部署上限的量不超过
# max_ctx × 本比例时不再整档跳过，而是把上下文裁减到能放下（贴边）实测一次
# ——256K 部署测 256K 档这类「档位顶着上限」的矩阵点（输出预算只占上限的
# 个位数百分比）不至于整档缺席。超出量超过本比例仍跳过：裁出来的是远低于
# 原档位的另一个点，测了也不代表原档位
CTX_EDGE_TRIM_RATIO = 0.10

# 超窗回退（事中兜底，与 CTX_HEADROOM 的事前跳过互补）：未配 max_ctx 或余量
# 仍不够时，临近上下文上限的点位会因输出随机性被服务端判 context exceeded
# 直接失败。此时按原填充长度逐档保留比例掐掉中段语料重试，而非判失败
CTX_RETRY_KEEP = (0.75, 0.5, 0.3)

# 上下文构建偏离重测阈值（ADR-0005 嵌套记账盲区的现场补救）：语料流 token
# 密度不均时，滑动边际估算（cpt_marginal）对「新进入语料段」的密度只能外推
# ——实测 echo 模式 64K 档偏离 +38.5%（32K→64K 段真实边际 1.83 字符/token，
# 滑动估计用了 3.12）。实测 prompt_tokens 均值与目标档位的相对偏差超过该值时，
# 按实测密度等比校正填充字符量重测一次（bust 模式专属，stable 重测会命中
# 前缀缓存污染读数）
CTX_DEV_TOLERANCE = 0.10

# 输出长度三道防线（ADR-0016，opencode zen 忽略 max_tokens 致输出跑飞至
# 131K tokens 的教训）：
# 1) 参数名实测判定：探测/正式请求观察截断是否生效，旧名 max_tokens 被忽略
#    时改用 max_completion_tokens（五类网关实测四种行为，无法按身份预判）
# 2) system 提示长度引导：把 max_tokens 按 out_cpt 先验折成字数（×本容差）
#    要求模型收敛在预设长度附近——服务端不截断时的软约束
# 3) 客户端断流兜底：输出估算超下发上限 ×CLIENT_CUT_FACTOR 仍未见 finish，
#    主动断流，防止跑飞请求拖垮单点耗时与消耗
OUT_HINT_FACTOR = 1.2
CLIENT_CUT_FACTOR = 2
# chunk_calib（事件/token 比）加权滑动平均的累计权重封顶（单位：事件数）：
# 保住对投机网关接受率变化的适应性（口径参照 cpt_calib 的 65536 字符封顶）
CHUNK_CALIB_W_CAP = 8192.0
# echo 请求的 assistant 预填长度（ADR-0021）：改写区开头原样预填这么多字符，
# 物理消除聊天式前言、强制从正文续写；太短压不住前言习惯，太长会白占输出预算
ECHO_PREFILL_CHARS = 160

# Agent 连续任务链默认参数：agent 场景不以上下文档位为变量，改为按任务轮次
# 推进的会话链——轮 0 冷启动（独立配置，默认 10K 上下文全量 prefill），
# 轮 1..K 暖轮 append-only 增长、命中服务端前缀缓存只增量 prefill，口径对齐
# 真实 agent 循环。
# 暖轮构成固定两阶段：阶段一短文本任务（每轮增量 ~N(1K) 正态、钳制到配置
# 区间），阶段二长文本任务（4K 起逐轮翻倍的 ladder：默认 12 暖轮 = 6 短 +
# 6 长，4K/8K/16K/32K/64K/128K）——短轮测高频小步循环的暖延迟，长轮测偶发
# 大上下文灌入的增量 prefill 能力
AGENT_TURNS_DEFAULT = 12              # 暖轮数（阶段一 + 阶段二），不含冷启动轮
AGENT_COLD_CTX_DEFAULT = 10240        # 冷启动轮上下文 tokens（独立于两阶段配置）
AGENT_TURN_DELTA_DEFAULT = (256, 2048)  # 阶段一短轮增量 tokens 钳制区间
AGENT_TURN_CENTER = 1000              # 阶段一每轮增量正态分布中心（tokens）
AGENT_PHASE2_BASE = 4096              # 阶段二 ladder 起步增量（tokens），逐轮翻倍
AGENT_TURNS_RANGE = (1, 32)
AGENT_TURN_DELTA_RANGE = (256, 32768)
AGENT_COLD_CTX_RANGE = (1024, 262144)
# 阶段二 ladder 单轮增量的绝对上限（与 AGENT_COLD_CTX_RANGE 上界同量级）：
# 部署未配 max_ctx 时 ctx_limit=None，逐轮翻倍 ladder 是唯一防线，n2 配大
# （如 16 档 × 4K 起步）可构造出 GB 级 prompt 触发 MemoryError，故封顶
AGENT_PHASE2_LADDER_CAP = 262144

# 媒体场景（asr/ocr/tts，ADR-0020）计量常数：阶梯在 SCENARIOS[场景].ladder 给出，
# ctx_list 不适用，并发矩阵与 LLM 共用
# whisper 类 ASR 编码器 ~50 tokens/音频秒（30s→1500 tokens）：max_ctx 超窗
# 保护按 时长×本系数 估算 prompt tokens
ASR_TOKENS_PER_SECOND = 50
# VLM 图像 token 估值（OpenAI 口径近似）：w×h ÷ 750
IMAGE_TOKEN_DIVISOR = 750


def _gen_agent_turns(n1: int, n2: int, dmin: int, dmax: int,
                     rng: random.Random, p2_base: int = AGENT_PHASE2_BASE,
                     cold_ctx: int = AGENT_COLD_CTX_DEFAULT
                     ) -> list[tuple[int, int]]:
    """生成各轮（新增 tokens, 阶段）：轮 0 冷启动（cold_ctx，阶段 0，独立配置
    默认 10K）；阶段一 n1 轮短文本——N(1K) 正态采样钳制到 [dmin, dmax] 后
    升序；阶段二 n2 轮长文本——p2_base 起步逐轮翻倍的 ladder（模拟 agent
    循环里偶发的大文件读取/长日志灌入；到 AGENT_PHASE2_LADDER_CAP 封顶，
    防未配 max_ctx 的部署构造出超量 prompt）。
    默认 1+6+6、4K 起步：冷启动测首轮全量 prefill，短轮测高频小步循环的暖
    延迟，长轮（4K/8K/16K/32K/64K/128K）测大上下文灌入的增量 prefill。
    轮目标为增量累积和、天然升序，链平滑增长；若把区间上限抬过 ladder
    首档，短轮增量可能超过它，增量不再全局升序，但 append-only 链语义
    不受影响。"""
    sigma = max(150.0, (dmax - dmin) / 6.0)
    phase1 = sorted(max(dmin, min(dmax, round(rng.gauss(AGENT_TURN_CENTER, sigma))))
                    for _ in range(n1))
    phase2 = [min(p2_base << i, AGENT_PHASE2_LADDER_CAP) for i in range(n2)]
    return [(cold_ctx, 0)] + [(s, 1) for s in phase1] + [(s, 2) for s in phase2]


def _cache_hit_from_usage(usage: dict | None) -> tuple[int, bool]:
    """从 usage 提取前缀缓存命中 tokens。各家字段名不一：DeepSeek 原生
    prompt_cache_hit_tokens；OpenAI/LiteLLM 风格 prompt_tokens_details.
    cached_tokens；Anthropic 风格 cache_read_input_tokens。LiteLLM 等网关
    中转时会剥离上游扩展字段（实测 LiteLLM 只回传 prompt/completion/total
    三个标准字段）——返回 (0, False)，调用方按链内差分估算。"""
    if not usage:
        return 0, False
    v = usage.get("prompt_cache_hit_tokens")
    if v is not None:
        return int(v), True
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return int(details["cached_tokens"]), True
    v = usage.get("cache_read_input_tokens")
    if v is not None:
        return int(v), True
    return 0, False

# decode 空窗校正上限（秒）：相邻内容块间隔超过此值即在净口径中封顶扣除。
# ADR-0017 由 3s 降为 1s（用户拍板，更积极的校正）——不再是「分布分离点」
# 口径：健康尾部的 1~2s 云端调度停顿也会被校正（当前存档实测 1/49 样本）。
# 与 0.5s 停滞诊断闸（STALL_THRESHOLD_S）构成两档：≥0.5s 标注留痕不改数、
# >1s 才在净口径中封顶扣除
DECODE_GAP_CAP = 1.0

# 停滞复合记账门限（秒）：≥该值的相邻块间隔记为一次停滞、满额计入累计
# （门限是检测闸，不是时长折扣）。比 DECODE_GAP_CAP 低一档——亚秒级停顿
# （如 10s decode 里 0.6s）诊断上应留痕但不做读数校正（借鉴 pi-tps）
STALL_THRESHOLD_S = 0.5
# 突发交付甄别：全部 chunk 以亚毫秒平均间隔到达 → 窗口测的是网关缓冲冲刷
# （buffer-flush dispatch）而非真实流式，decode 速率虚高不可信。真流式即使
# 3000 tok/s、5 token/chunk 也有 ~1.7ms 间隔（借鉴 pi-tps）。另增补「少而肥」
# 分支：2~4 个多 token 事件亚毫秒冲刷（网关缓冲合并 + 投机解码并存的真实
# 形态）chunk 数过不了 BURST_MIN_CHUNKS，按每 chunk token 数甄别；真实周期
# 交付（draft+verify）间隔 k/tg ≥10ms，不会误伤
BURST_MIN_CHUNKS = 5      # chunk 太少不足为凭（原判据）
BURST_FAT_MIN_CHUNKS = 3  # 少而肥分支：chunk 数下限（≥原判据豁免不到的 2~4 个）
BURST_FAT_MIN_TPC = 8     # 少而肥分支：每 chunk 平均 token 下限（多 token 事件）
BURST_AVG_GAP_MS = 1.0    # 平均间隔低于该值视为冲刷（两条判据共用）

# 输出塌缩守卫（通用保险，软超窗占位串精确匹配的兜底补充）：finish=stop 但
# 输出近乎为空而 prompt 很大——大上下文下合法短回复概率可忽略，典型形态是
# 占位软拒绝（占位串未必是已知文案）。阈值卡在大 prompt 上：0K/小 ctx 的
# 合法短输出不误伤
COLLAPSE_MIN_PROMPT_TOKENS = 8192   # prompt 低于该值不启用（合法短输出保护）
COLLAPSE_MAX_OUT_TOKENS = 2
COLLAPSE_MAX_OUT_CHARS = 32

# 物理速率闸：prefill 速率超该 ctx 先验曲线 RATE_PRIOR_OVERSHOOT 倍或绝对上限
# 即判「物理上不可能」——占位软拒绝/网关拒答式假成功的通用兜底（实测事故：
# fastllm 占位回复 15 字符 ÷ 超长 prompt 判出 38 万 tok/s 假 prefill 并污染
# 先验曲线与校准）。倍数语境同 rate_prior 的 1.3× 外推钳制
RATE_PRIOR_OVERSHOOT = 10.0
RATE_ABS_CAP_TOK_S = 100000.0

# 异常输出识别 + 弃测重测：单次复测（rep 槽位）命中 early_stop/退化重复时，
# 整批测量已被污染（conc>1 时读数混入异常请求），弃测该次复测、换 nonce 重跑。
# 每个槽位最多重测 1 次——重测仍异常视为该部署在此档位的真实行为，按实留档
# 聚合，不无限重试把异常「洗掉」（与构建偏离校正重测的最多 1 次同口径）
ANOMALY_RETEST_MAX = 1


def _decode_burst(n_chunks: int, decode_time: float | None,
                  out_tokens: float | None = None) -> bool:
    if not decode_time:
        return False
    if (n_chunks >= BURST_MIN_CHUNKS
            and decode_time * 1000 / (n_chunks - 1) < BURST_AVG_GAP_MS):
        return True
    # 少而肥的 chunk：2~4 个多 token 事件亚毫秒冲刷，chunk 数过不了原判据——
    # 按每 chunk token 数增补甄别；out_tokens 缺 usage 时是估算值，仍可用
    return bool(n_chunks >= BURST_FAT_MIN_CHUNKS and out_tokens
                and out_tokens / n_chunks >= BURST_FAT_MIN_TPC
                and decode_time * 1000 / n_chunks < BURST_AVG_GAP_MS)


def _fmt_ctx(ctx: int) -> str:
    return "0K" if ctx <= 0 else f"≈{ctx // 1024}K"


def _plan_ctx_edge(in_tok: int, budget: int, max_ctx: int) -> tuple[str, int, int]:
    """部署上限超窗的「贴边裁减」判定（纯函数，LLM 上下文阶梯与媒体阶梯共用；
    agent 链的轮次收口有自己的口径，不在此列）。in_tok 为该点输入侧构造目标
    （LLM=上下文档位、asr=时长×tokens/秒、ocr=张数×图 token 均值），budget 为
    该点输出预算（max_tokens）。判定口径与既有超窗跳过一致：输出预算外加
    CTX_HEADROOM 分词漂移余量一并对窗——
        excess = in_tok + budget + CTX_HEADROOM − max_ctx
    返回 (mode, in_tok_eff, excess)：
      "ok"   — excess ≤ 0：未超窗，按原构造量测（in_tok_eff = in_tok，现状不变）
      "edge" — 0 < excess ≤ CTX_EDGE_TRIM_RATIO × max_ctx：超窗量在阈值内，
               裁减贴边实测（in_tok_eff = max_ctx − budget − CTX_HEADROOM，恰好
               落回既有跳过边界，输出预算与漂移余量仍被保住）
      "skip" — 超窗量超过阈值，或贴边量非正（budget + CTX_HEADROOM ≥ max_ctx，
               连零输入都放不下）：维持整档跳过
    """
    excess = in_tok + budget + CTX_HEADROOM - max_ctx
    if excess <= 0:
        return "ok", in_tok, excess
    edge = max_ctx - budget - CTX_HEADROOM
    if excess <= CTX_EDGE_TRIM_RATIO * max_ctx and edge > 0:
        return "edge", edge, excess
    return "skip", 0, excess

CORPUS_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "corpus", "code")
CORPUS_CREATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "corpus", "creative")
CORPUS_AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "corpus", "agent")


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


def _load_agent_pool() -> list[str]:
    """Agent 场景语料：优先用 vendored 的 SWE-agent 真实执行轨迹（corpus/agent/，
    nebius/swe-agent-trajectories，CC-BY-4.0；任务取自 SWE-bench dev 与
    SWE-bench-extra，构建脚本 scripts/build_agent_corpus.py）。轨迹块保留
    [user]/[assistant]/[observation] 交替结构——bash 命令、文件查看输出、
    diff 的 token 纹理是 agent 工作负载的真实口径，与创意/代码场景的
    连续散文/源码分布显著不同。
    每个文件一个轨迹块（≥2000 字符，按轮次边界聚合，由构建脚本保证）；
    缺失/为空时回退内置块。"""
    pool = []
    try:
        for root, _dirs, files in os.walk(CORPUS_AGENT_DIR):
            for fn in sorted(files):
                if fn.upper().startswith(("LICENSE", "README")):
                    continue
                try:
                    with open(os.path.join(root, fn), encoding="utf-8",
                              errors="replace") as f:
                        text = f.read().strip()
                except OSError:
                    continue
                if len(text) >= 2000:
                    pool.append(text)
    except OSError:
        pass
    return pool or list(_AGENT_POOL)


SCENARIOS["code"]["filler_blocks"] = _load_code_pool()
SCENARIOS["creative"]["filler_pool"] = _load_creative_pool()
SCENARIOS["agent"]["filler_pool"] = _load_agent_pool()


def _make_stream(pool: list[str]) -> tuple[str, list[int]]:
    """语料池 → 一条确定性长流：固定种子乱序一次后顺序拼接。各上下文档位都从
    流首截取，点位内容互为嵌套前缀——上一档实测的「字符数 → prompt_tokens」
    对本档前缀是精确已知的，构造只需估算增量段，校准不再被每点随机抽样的
    文件配比抖动带着振荡（code 场景实测偏差曾达 ±30%，ADR-0005）。
    种子固定 → 多次运行内容一致，结果可跨次比较。
    返回 (stream, boundaries)：boundaries 为每个语料块在流中的起始偏移
    （含流首 0，"\n\n" 分隔符计入累积）——echo 改写区起点对齐块边界的依据。"""
    parts = list(pool)
    random.Random(0xC0DE).shuffle(parts)
    boundaries: list[int] = []
    pos = 0
    for p in parts:
        boundaries.append(pos)
        pos += len(p) + len("\n\n")
    return "\n\n".join(parts), boundaries


def _make_module_stream(blocks: list[list[str]]) -> tuple[str, list[int]]:
    """模块块 → 一条确定性长流：固定种子只乱序**块的次序**，块内文件保持
    文件名排序（同模块文件、头文件与实现相邻——真实编程上下文的关联性结构，
    ADR-0005）。任意 ≥模块尺度的窗口内部连贯，整条流仍有跨模块多样性
    （小上下文档位不锁定单一模块）；确定性、完整文件粒度与嵌套前缀性质
    与 _make_stream 一致。返回值同 _make_stream（boundaries 粒度 = 文件）。"""
    parts = [list(b) for b in blocks]
    random.Random(0xC0DE).shuffle(parts)
    flat = [s for b in parts for s in b]
    boundaries: list[int] = []
    pos = 0
    for s in flat:
        boundaries.append(pos)
        pos += len(s) + len("\n\n")
    return "\n\n".join(flat), boundaries


for _sc in SCENARIOS.values():
    if _sc.get("filler_blocks"):
        (_sc["filler_stream"],
         _sc["filler_boundaries"]) = _make_module_stream(_sc["filler_blocks"])
    else:
        (_sc["filler_stream"],
         _sc["filler_boundaries"]) = _make_stream(_sc.get("filler_pool") or [])
# tts 场景的输入即文本：直接复用创意语料流（公版红楼梦中文化流，零新增文件）
SCENARIOS["tts"]["filler_stream"] = SCENARIOS["creative"]["filler_stream"]
SCENARIOS["tts"]["filler_boundaries"] = SCENARIOS["creative"]["filler_boundaries"]


# ---------------------------------------------------------------------------
# 媒体场景语料：音频（asr/tts）与文档图（ocr），ADR-0020
# ---------------------------------------------------------------------------
# 测速负载只取决于媒体尺寸（音频时长×采样率 / 图像分辨率×图文密度），与内容
# 语义无关。内置语料确定性合成、零仓库体积、零新增依赖（stdlib wave/zlib/
# struct/base64）；corpus/asr/、corpus/ocr/ 有用户文件时优先用用户的——
# 真实语音/真实扫描页优先级更高（如服务端带 VAD，合成音可能被判定静音短路）。

CORPUS_ASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "corpus", "asr")
CORPUS_OCR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "corpus", "ocr")

_SIN_TABLE = [math.sin(2 * math.pi * i / 2048) for i in range(2048)]
_NOISE_TABLE = [random.Random(0x5EED).uniform(-1, 1) for _ in range(4096)]


def _speech_segment_pcm(sr: int, n: int) -> bytes:
    """生成一段 speech-like PCM16（单声道）：基音 ~120Hz + 谐波，4Hz 音节率
    振幅调制 + 轻底噪——能量包络接近真人朗读，能量型 VAD 不会整段判静音。
    正弦查表 + 相位累加器实现，长音频可秒级生成。"""
    pcm = array("h", bytes(2 * n))
    amp = 10500.0
    p1, p2, p3, pm = 0.0, 0.0, 0.0, 0.0
    i1 = 120.0 / sr * 2048
    i2, i3 = i1 * 2, i1 * 3
    im = 3.8 / sr * 2048     # ~4Hz 音节包络
    for k in range(n):
        env = 0.5 + 0.5 * _SIN_TABLE[int(pm)]
        s = amp * env * (0.62 * _SIN_TABLE[int(p1)]
                         + 0.25 * _SIN_TABLE[int(p2)]
                         + 0.13 * _SIN_TABLE[int(p3)]) + 700 * _NOISE_TABLE[k & 4095]
        pcm[k] = max(-32768, min(32767, int(s)))
        p1 = (p1 + i1) % 2048
        p2 = (p2 + i2) % 2048
        p3 = (p3 + i3) % 2048
        pm = (pm + im) % 2048
    return pcm.tobytes()


def _synth_speech_wav(seconds: float) -> bytes:
    """合成目标时长的 speech-like wav（16kHz 16bit 单声道）：4s 语音段循环
    拼接——ASR 编码器负载只取决于时长×采样率，段重复不改变计算量。"""
    sr = 16000
    n = int(sr * seconds)
    seg = _speech_segment_pcm(sr, sr * 4)
    pcm = (seg * (n // (sr * 4) + 1))[: n * 2]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_seconds(data: bytes) -> float | None:
    """stdlib wave 读 wav 时长（秒）；非 wav/损坏返回 None。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        return None


def _loop_wav(data: bytes, seconds: float) -> tuple[bytes, float]:
    """用户语料音频循环到目标时长：帧级截取保证实际时长精确（阶梯按名义
    时长出点，读数分母必须与实际负载一致）。"""
    with wave.open(io.BytesIO(data), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    sr = params.framerate
    frame_bytes = params.sampwidth * params.nchannels
    total = params.nframes
    if sr <= 0 or total <= 0:
        raise ValueError("无效的 wav 语料（空音频）")
    need = int(sr * seconds)
    pcm = (frames * (need // total + 1))[: need * frame_bytes]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(pcm)
    return buf.getvalue(), need / sr


_AUDIO_CACHE: dict[float, tuple[bytes, float]] = {}


def _asr_audio(seconds: float) -> tuple[bytes, float]:
    """ASR 测试音频：(wav bytes, 实际秒数)。corpus/asr/*.wav 存在时取排序首个
    文件循环拼接到目标时长；否则合成 speech-like 音频。按目标秒数缓存
    （阶梯档位在重复/并发请求间复用同一份负载）。"""
    if seconds not in _AUDIO_CACHE:
        try:
            names = sorted(f for f in os.listdir(CORPUS_ASR_DIR)
                           if f.lower().endswith(".wav"))
        except OSError:
            names = []
        if names:
            with open(os.path.join(CORPUS_ASR_DIR, names[0]), "rb") as f:
                audio, actual = _loop_wav(f.read(), seconds)
        else:
            audio, actual = _synth_speech_wav(seconds), float(seconds)
        _AUDIO_CACHE[seconds] = (audio, actual)
    return _AUDIO_CACHE[seconds]


def _synth_document_png(w: int, h: int, seed: int = 0) -> bytes:
    """合成扫描文档页（最小 PNG 编码器，stdlib zlib）：白底、黑色伪文本行
    （随机词间空隙、行距），视觉编码负载 ≈ 同分辨率真实扫描页。"""
    rnd = random.Random(seed)
    top, bottom = int(h * 0.08), int(h * 0.94)
    margin = int(w * 0.08)
    line_h = max(3, h // 70)
    raw = bytearray()
    y = 0
    while y < h:
        row = bytearray(b"\xff" * (w * 3))
        if top <= y < bottom and y % (line_h + max(2, line_h // 2)) < line_h:
            x = margin
            while x < w - margin:
                run = rnd.randint(max(2, w // 120), max(4, w // 18))
                if rnd.random() < 0.85:   # 词间空隙不涂黑
                    row[x * 3:(x + run) * 3] = b"\x14" * (run * 3)
                x += run + rnd.randint(max(2, w // 150), max(4, w // 40))
        raw += b"\x00" + bytes(row)
        y += 1

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8bit RGB
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _chunk(b"IEND", b""))


def _image_size(data: bytes) -> tuple[int, int] | None:
    """图片尺寸解析（stdlib）：PNG IHDR / JPEG SOF；无法识别返回 None。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 1 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xFF:   # JPEG 规范允许 marker 前任意 fill bytes（连续
                i += 1           # 0xFF），逐字节跳过——把填充当段长会跳过真 SOF
                continue
            if marker in (0xC0, 0xC1, 0xC2):
                if i + 9 > n:
                    break
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            if marker == 0xDA:   # SOS：其后是熵编码数据，不是段结构，停止扫描
                break
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if i + 4 > n:
                break
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


_IMG_CACHE: list[tuple[bytes, int, int, str]] | None = None


def _ocr_images() -> list[tuple[bytes, int, int, str]]:
    """OCR 测试图池：corpus/ocr/ 下的 png/jpg/jpeg 优先；缺省合成 3 种分辨率
    文档页。返回 (bytes, w, h, mime) 列表，token 估值 w×h÷IMAGE_TOKEN_DIVISOR。"""
    global _IMG_CACHE
    if _IMG_CACHE is not None:
        return _IMG_CACHE
    pool: list[tuple[bytes, int, int, str]] = []
    try:
        names = sorted(f for f in os.listdir(CORPUS_OCR_DIR)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
    except OSError:
        names = []
    for fn in names:
        try:
            with open(os.path.join(CORPUS_OCR_DIR, fn), "rb") as f:
                data = f.read()
        except OSError:
            continue
        size = _image_size(data)
        if not size:
            continue
        mime = "image/png" if fn.lower().endswith(".png") else "image/jpeg"
        pool.append((data, size[0], size[1], mime))
    if not pool:
        for i, (w, h) in enumerate(((1240, 1754), (1654, 2339), (2480, 3508))):
            pool.append((_synth_document_png(w, h, seed=i), w, h, "image/png"))
    _IMG_CACHE = pool
    return pool


def _ocr_avg_img_tokens() -> int:
    """图池单张平均 token 估值（超窗保护按 张数×均值 估算 prompt tokens）。"""
    pool = _ocr_images()
    return sum(w * h for _, w, h, _ in pool) // len(pool) // IMAGE_TOKEN_DIVISOR


def _build_ocr_messages(n_img: int, nonce: int) -> tuple[list, int]:
    """OCR 请求消息：N 张图（图池按 nonce 轮转取图，bust 服务端前缀缓存）+
    识别指令。返回 (messages, 图片 token 估值)。"""
    pool = _ocr_images()
    imgs = [pool[(nonce + i) % len(pool)] for i in range(n_img)]
    parts = [{"type": "image_url",
              "image_url": {"url": f"data:{mime};base64,"
                                  + base64.b64encode(data).decode()}}
             for data, _w, _h, mime in imgs]
    img_tokens = sum(w * h for _d, w, h, _m in imgs) // IMAGE_TOKEN_DIVISOR
    parts.append({"type": "text",
                  "text": SCENARIOS["ocr"]["instruction"].format(n=n_img)})
    return ([{"role": "system", "content": SCENARIOS["ocr"]["system"]},
             {"role": "user", "content": parts}], img_tokens)


def _tts_text(chars: int, nonce: str) -> str:
    """TTS 合成文本：编号行 + 创意语料流截取到目标字符数（红楼梦中文化流，
    标点/多音字覆盖真实合成负载；编号行使各请求文本不完全相同）。"""
    head = f"合成片段编号 {nonce}："
    return head + _fill(SCENARIOS["tts"]["filler_stream"],
                        max(0, chars - len(head)))


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


def _trim_middle(content: str, keep: int) -> str:
    """超窗删减重试：把 user 内容掐到约 keep 字符——保留开头（编号行）与
    结尾（任务指令），掐掉中段填充语料，指令结构不被破坏。"""
    if keep >= len(content):
        return content
    half = keep // 2
    return (content[:half] + "\n……（中间内容已删减）……\n"
            + content[-(keep - half):])


def _align_echo_region(filler: str, region: int,
                       boundaries: list[int]) -> int:
    """echo 改写区起点对齐到结构边界，返回对齐后的起点下标（改写区 =
    filler[起点:]，长度 region_aligned = len(filler) − 起点）。region 仍是
    目标口径（max_tokens_default×out_cpt），起点在「改写区 ≈ region」的约束
    下尽量落在完整结构单元开头，三层回退：
    1. 块边界：边界表（限 len(filler) 以内，循环续接流只覆盖首段）中取
       **最后一个** b 使 len(filler) − b ∈ [0.8×region, 1.6×region]——改写区
       = 参考材料末尾的完整文件/语料块组，重写目标结构完备（原任意字符
       偏移曾从「半个宏」开始，重写目标残缺）；
    2. 段/函数边界：[目标起点, 目标起点+0.4×region] 内的第一个 "\n\n" 之后
       ——从一个函数/段落开头开始，连续性完整；
    3. 回退现状：目标起点（任意字符偏移），预填的行边界收口兜底。
    无边界表/空表的场景（filler_boundaries 缺失）直接落第 2/3 层，照样工作。
    选择只依赖 len(filler) 与静态边界表——嵌套前缀与多次运行的确定性不受
    影响（同 len(filler) 同起点）。"""
    n = len(filler)
    region = min(region, n)
    target_start = n - region
    best = None
    for b in boundaries:
        if b > n:
            break
        if 0.8 * region <= n - b <= 1.6 * region:
            best = b
    if best is not None:
        return best
    window = filler[target_start:target_start + int(0.4 * region)]
    idx = window.find("\n\n")
    if idx >= 0:
        return target_start + idx + 2
    return target_start


def build_messages(scenario: str, target_tokens: int, cpt: float, nonce: str,
                   filler_chars: int | None = None, echo: bool = False):
    """构造一条目标长度约 target_tokens 的对话。
    返回 (messages, est_tokens, 填充字符数)。
    target_tokens <= 0 为零输入档：不注入参考材料，一句话指令直接命题。
    filler_chars 显式指定填充长度（嵌套前缀精确记账用，ADR-0005），
    缺省按 target_tokens × cpt 估算。
    echo=True 换用场景的高复用改写指令（echo_instruction，ADR-0021）：同
    filler_chars + 同 nonce 下与 echo=False 版本仅尾部任务句不同，前缀缓存
    可命中整段参考材料；无 echo_instruction 的场景回退普通 instruction。
    echo 且 filler 非空时末尾追加 assistant 预填消息：改写区 = 参考材料末尾
    约 max_tokens_default×out_cpt 字符，**起点对齐到结构边界**——优先取
    末尾的完整语料块组（块边界表命中），否则对齐到段/函数边界（"\n\n"
    之后），都不行才退回任意字符偏移（_align_echo_region 三层回退）；预填
    取改写区开头 ECHO_PREFILL_CHARS 字符并按行边界收口——聊天式前言被
    物理消除，模型从结构完整的正文直接续写，输出与参考材料末尾高重叠
    （实测纯指令约束下 DeepSeek-V4-Flash 仍以「我们根据任务要求……」开头，
    重叠率仅 ~10%，投机采样无从生效）。"""
    sc = SCENARIOS[scenario]
    system = sc["system"]
    if target_tokens <= 0:
        user = sc["zero_instruction"]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages, (len(system) + len(user)) / max(cpt, 0.1), 0
    instr = sc.get("echo_instruction") if echo else None
    instr = instr or sc["instruction"]
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
    if echo and filler and sc.get("echo_instruction"):
        # assistant 预填：改写区 = 参考材料末尾 ~max_tokens_default×out_cpt
        # 字符，起点对齐到结构边界（改写区 = 末尾完整结构单元；块边界 →
        # 段/函数边界 → 任意偏移三层回退，见 _align_echo_region），预填取
        # 改写区开头一段并按行边界收口——对齐后改写区从文件/函数/语料块
        # 开头开始，模型只需照着上下文继续誊写+微调
        region = min(len(filler), int(sc["max_tokens_default"] * sc["out_cpt"]))
        start = _align_echo_region(filler, region,
                                   sc.get("filler_boundaries") or [])
        prefill = filler[start:start + ECHO_PREFILL_CHARS]
        nl = prefill.rfind("\n")
        if nl >= ECHO_PREFILL_CHARS // 2:
            prefill = prefill[:nl + 1]
        if prefill.strip():
            messages.append({"role": "assistant", "content": prefill})
    est_tokens = (len(system) + len(user)) / max(cpt, 0.1)
    return messages, est_tokens, len(filler)


def _correct_filler_chars(filler_n: int, real_mean: float, ctx: int,
                          kb_before: tuple | None, overhead_chars: int,
                          cpt: float) -> int | None:
    """按实测密度等比校正填充字符量（构建偏离重测的校正数学）。
    记账模型：实测 tokens ≈ 固定开销（指令/模板/echo 预填等，不计入 filler
    的全部字符按 cpt 折算）+ filler 字符 ÷ 边际密度 d。校正只动 filler：
    f_new = filler_n + (ctx − real_mean) × d，指令/nonce/echo 预填等固定开销
    不参与收缩/扩张。
    密度 d 分两路取：
    - 有嵌套记账先验（kb_before = (前档 filler_n, 前档实测 tokens)，ADR-0005）：
      本次首测本身就是新语料段的真实密度样本，d = (filler_n − kb[0]) /
      (real_mean − kb[1])——增量段实测边际，比滑动估计（对新段只能外推）准；
    - 无先验（首个非零档）：d = filler_n / (real_mean − overhead_chars/cpt)，
      filler 主导大档位构造时近似成立。
    d 越界（<0.3 或 >20 字符/token，与 _calibrate_usage 的增量观测钳制同界）、
    校正后 filler 为负或不产生变化时返回 None（不重测，接受本次读数）。"""
    d = None
    if kb_before and real_mean > kb_before[1] and filler_n > kb_before[0]:
        d = (filler_n - kb_before[0]) / (real_mean - kb_before[1])
    if not (d and 0.3 <= d <= 20):
        ovh = overhead_chars / max(cpt, 0.1)
        if real_mean <= ovh:
            return None
        d = filler_n / (real_mean - ovh)
    f_new = filler_n + (ctx - real_mean) * d
    if not (0.3 <= d <= 20) or f_new < 0:
        return None
    f_new = round(f_new)
    return f_new if f_new != filler_n else None


def _net_elapsed(elapsed: float, rtt: float | None) -> float:
    """净耗时 = 实测耗时 − RTT 基线，扣减**封顶为实测值的一半**。
    RTT 基线取自 /models 元数据接口，与 /chat 推理调度路径不完全同构，
    基线抖动（同一网关实测在 75↔368ms 间漂移）可能超过小 prompt 的 TTFT——
    原 1e-3 下界曾把 0K 档净 prefill 放大到 79000 tok/s；封顶后净口径
    相对毛口径最多放大 2×，不再爆炸。"""
    return max(elapsed - (rtt or 0.0), elapsed / 2, 1e-6)


def _wma(old: float | None, w: float, obs: float, w_new: float) -> float:
    """加权滑动平均（cpt_calib/chunk_calib 共用口径）：旧值按累计权重 w、
    新观测按本次权重 w_new 混合；无旧值时直接取新观测。"""
    if old is None:
        return obs
    return (old * w + obs * w_new) / (w + w_new)


def _ratio_refold(base_est: float, ratio_old: float | None,
                  ratio_new: float | None) -> float:
    """把滑窗基准样本的 token 估算折算到新校准比例口径：base_est 是按
    ratio_old 估值口径的 token 数（ratio_old 为 None 时未按比例校准，即事件
    数直计），先还原成事件数再按 ratio_new 重算，两侧一致时原样返回——
    校准比例在滑窗内变化时，差分两侧口径错配会产出速度尖峰甚至负值。"""
    if ratio_old == ratio_new:
        return base_est
    events = base_est * ratio_old if ratio_old else base_est
    return events / ratio_new if ratio_new else events


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


def _total_decode_rates(ok: list[dict], conc: int) -> tuple:
    """并发组总吞吐双口径 (raw, adj)，_run_point 与 agent 链共用同公式。

    raw = 总输出 tokens ÷（min(first)→max(last) 的窗口），仅当并发请求的
    decode 区间充分重叠（overlap ≥ 0.5×最短 decode）才有意义——长上下文
    prefill 被服务端串行化时 decode 实际先后发生，窗口混入 prefill 等待期
    → 断崖假象（ADR-0006），置空。
    adj = 空窗校正口径：窗口扣除各请求停滞累计（stall_s，首 token 后块间
    空窗）；首 token 前的 prefill 等待不在 stall 内，由重叠守卫兜底置空
    （128K 全串行案例 stall_s 全空，重叠守卫依旧拦下）。
    两口径共享全部产出守卫：len(ok)==conc、total_out>0、firsts/lasts 齐备、
    max(lasts)>min(firsts)、重叠守卫照旧；adj 另加 eff_window>0。"""
    if len(ok) != conc:
        return None, None
    total_out = sum(r.get("out_tokens") or 0 for r in ok)
    firsts = [r.get("first_abs") for r in ok]
    lasts = [r.get("last_abs") for r in ok]
    if (not total_out or any(f is None for f in firsts)
            or any(l is None for l in lasts)):
        return None, None
    span = max(lasts) - min(firsts)
    if span <= 0:
        return None, None
    overlap = min(lasts) - max(firsts)
    decs = [l - f for f, l in zip(firsts, lasts)]
    if conc > 1 and not (overlap > 0 and overlap >= 0.5 * min(decs)):
        return None, None
    raw = round(total_out / span, 1)
    eff_window = span - sum(r.get("stall_s") or 0 for r in ok)
    adj = round(total_out / eff_window, 1) if eff_window > 0 else None
    return raw, adj


# 聚合时的字段展示精度，与单点口径一致
_ROUND = {"ttft_s": 3, "ttft_net_s": 3, "prefill_tok_s": 1, "prefill_net_tok_s": 1,
          "prefill_uncached_tok_s": 1, "prefill_full_tok_s": 1,
          "decode_tok_s": 1, "decode_tok_s_adj": 1, "decode_total_tok_s": 1,
          "decode_total_tok_s_adj": 1,
          "tok_per_chunk": 2,
          "prompt_tokens": 0, "out_tokens": 0, "cache_hit_tokens": 0,
          # 媒体场景（asr/ocr/tts，ADR-0020）：RTF/倍速/吞吐等点级指标同取均值
          "rtf": 4, "speed_x": 1, "audio_s": 1, "elapsed_s": 2, "elapsed_net_s": 2,
          "ttfa_s": 3, "ms_per_img": 1, "img_per_s": 2, "audio_min_per_min": 2,
          "out_chars": 0, "out_bytes": 0, "n_img": 0}


def _aggregate_reps(reps: list[dict]) -> dict:
    """多次复测聚合：标量取均值（ADR-0017——中位改均值，明细表聚合行即均值行，
    各次实测以复测子行逐次列出）；停滞/空窗诊断同取均值（出现停滞的复测之间），
    聚合行不再有「非均值」特例。reqs 明细保留 decode 居中的一次作代表样本。
    all_ok 多数决：过半复测成功才视为成功点。"""
    ok_reps = [p for p in reps if p["all_ok"]]
    pool = ok_reps or reps
    rep_point = sorted(pool, key=lambda p: p.get("decode_tok_s") or 0)[len(pool) // 2]
    point = dict(rep_point)
    for f, nd in _ROUND.items():
        vals = [p[f] for p in pool if p.get(f) is not None]
        if vals:
            m = sum(vals) / len(vals)   # ADR-0017：复测聚合由中位改均值
            point[f] = int(round(m)) if nd == 0 else round(m, nd)
    point["all_ok"] = len(ok_reps) * 2 > len(reps)
    gaps = [p["max_gap_s"] for p in reps if p.get("max_gap_s")]
    if gaps:
        point["max_gap_s"] = round(sum(gaps) / len(gaps), 2)   # 停滞诊断同取均值
    stall_reps = [p for p in reps if p.get("stall_s")]
    if stall_reps:   # 停滞复合记账同取均值（与 max_gap_s 同口径）
        point["stall_s"] = round(sum(p["stall_s"] for p in stall_reps) / len(stall_reps), 2)
        point["stall_count"] = round(sum(p.get("stall_count") or 0
                                         for p in stall_reps) / len(stall_reps)) or 1
    # 突发交付多数决（与 all_ok 同口径）；dict(rep_point) 可能带入居中次样本的
    # True，需显式覆写
    point["decode_burst"] = (sum(bool(p.get("decode_burst")) for p in reps) * 2
                             > len(reps)) or None
    # 音频时长语速估算标注跨 rep 合并（任一次实测走估算即标注；参照
    # _aggregate_media_reqs 的 any 口径）
    if any(p.get("audio_est") for p in reps):
        point["audio_est"] = True
    point["n_reps"] = len(reps)
    # 异常输出标注（弃测重测机制）：pool 中任一复测按实留档了 anomaly 则取
    # 第一个非 None 值（重测仍异常的 rep 带 anomaly 键进聚合），全为 None
    # 即正常；弃测留痕跨 rep 拼接——弃测的复测本就不在 reps 子行里（没进
    # rep_points），留痕只在聚合行体现。空则两键均不出现在聚合点/子行
    anoms = [p.get("anomaly") for p in pool if p.get("anomaly")]
    point["anomaly"] = anoms[0] if anoms else None
    discards = [d for p in reps for d in (p.get("reps_discarded") or [])]
    if discards:
        point["reps_discarded"] = discards
    # 复测摘要：供前端明细表逐次子行展示（字段与聚合行同口径；旧存档缺新增
    # 字段时前端按空值回退显示）
    point["reps"] = [{k: p.get(k) for k in
                      ("prompt_tokens", "ttft_s", "prefill_tok_s", "prefill_net_tok_s",
                       "prefill_uncached_tok_s", "prefill_full_tok_s",
                      "decode_tok_s", "decode_tok_s_adj", "decode_total_tok_s",
                      "decode_total_tok_s_adj",
                      "tok_per_chunk",
                       "out_tokens", "all_ok", "finish", "stall_s", "stall_count",
                       "max_gap_s", "decode_burst",
                       "cache_hit_tokens", "cache_reported",
                       # 媒体场景逐次复测子行（ADR-0020）
                       "rtf", "speed_x", "audio_s", "elapsed_s", "ttfa_s",
                       "ms_per_img", "img_per_s", "audio_min_per_min",
                       "out_chars", "out_bytes")} for p in reps]
    return point


def _aggregate_media_reqs(kind: str, reqs: list[dict], batch_time: float,
                          conc: int) -> dict:
    """asr/tts 请求级结果 → 点级聚合：净口径 RTF/倍速取成功请求均值；吞吐 =
    总音频量 / 批窗口（音频分钟/分钟，仅全成功并发点才有——个别失败的批
    分母被摊薄、读数无意义）。audio_est 标注音频时长来自语速估算（服务端
    未回可解析 wav，ADR-0020）。"""
    ok = [r for r in reqs if not r.get("err")]
    if not ok:
        return {}

    def _mean(key: str, nd: int):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), nd) if vals else None

    out = {"audio_s": _mean("audio_s", 1),
           "elapsed_s": _mean("total_s", 2),
           "elapsed_net_s": _mean("elapsed_net_s", 2),
           "rtf": _mean("rtf", 4),
           "speed_x": _mean("speed_x", 1)}
    if kind == "asr":
        out["out_chars"] = _mean("out_chars", 0)
    else:
        out["ttfa_s"] = _mean("ttfa_s", 3)
        out["out_bytes"] = _mean("out_bytes", 0)
        if any(r.get("audio_est") for r in ok):
            out["audio_est"] = True
    if len(ok) == conc and batch_time > 0:
        out["audio_min_per_min"] = round(
            sum(r["audio_s"] for r in ok) / batch_time / 60, 2)
    return out


def _is_ctx_overflow(status: int, body: str) -> bool:
    """服务端返回「超出上下文窗口」错误：各家状态码与报文不一（OpenAI 系
    400 / code=context_length_exceeded，Kimi 用 401 invalid_authentication_error，
    llama.cpp 报 available context size），按报文关键词判定。"""
    low = body.lower()
    return "context" in low and ("exceed" in low or "only" in low
                                 or "context_length" in low or "context size" in low
                                 or "maximum context" in low)


def _is_soft_ctx_overflow(head_text: str, total_chars: int,
                          reason_head: str = "") -> bool:
    """服务端「软超限」：HTTP 200 + 正文被替换为占位串（fastllm 系引擎对超长
    prompt 不回 4xx，而是正常流式返回 "prompt too long"、finish=stop、单
    token）。不识别的话该点会记成「拒绝耗时 ÷ prompt 长度」的数十万 tok/s
    假 prefill。按占位串精确匹配（整体输出 ≤32 字符），正常模型输出不会命中；
    占位串可能走 content 也可能走 reasoning_content 通道返回（思考模型的
    fastllm 实测走 reasoning），两个通道的头部文本都参与判定。"""
    if total_chars > 32:
        return False
    for head in (head_text, reason_head):
        if head.strip().lower() == "prompt too long":
            return True
    return False


def _head_sample(buf: str, piece: str, cap: int) -> str:
    """流式输出的头部采样累积：收满 cap 字符后零开销直通，截断即停——
    不回读、不重组，与 head_text 的 32 字符截获同型。"""
    if len(buf) < cap:
        return (buf + piece)[:cap]
    return buf


def _is_collapsed_reply(prompt_tokens, out_tokens, finish,
                        total_chars: int) -> bool:
    """输出塌缩守卫（通用保险）：finish=stop 但输出近乎为空而 prompt 很大
    ——大上下文下合法短回复概率可忽略，典型形态是占位软拒绝（占位串未必
    命中已知文案）。阈值卡在大 prompt 上，0K/小 ctx 的合法短输出不误伤。"""
    if finish != "stop":
        return False
    if prompt_tokens is None or prompt_tokens < COLLAPSE_MIN_PROMPT_TOKENS:
        return False
    if out_tokens is None or out_tokens > COLLAPSE_MAX_OUT_TOKENS:
        return False
    return total_chars <= COLLAPSE_MAX_OUT_CHARS


def _is_degenerate_text(s: str) -> bool:
    """头部采样退化重复检测：文本几乎只用极少数几种字符反复循环。判据
    len(t) >= 80 且 len(set(t)) <= 12，阈值来自三个实测退化样本——fastllm
    部署 ≥8K 档思考流退化（reason_chars≈512 即 1 字符/token）：①"}\\n"×N
    循环（仅 2 种字符）；②"6. 7. 8. 9." 无限编号列表（12 种字符，恰在阈值
    边缘：数字+点+空格+换行）；③嵌套括号/缩进递增（6 种字符）。对照：正常
    中文散文、英文或代码输出的前 200 字符内 distinct 字符数远超 12（汉字、
    标点、关键字、运算符的并集在几十种以上），不会误伤。下限 80 字符挡掉
    「好的。」一类天然短回复——短串字符集小是正常现象，不构成退化证据。"""
    t = s.strip()
    return len(t) >= 80 and len(set(t)) <= 12


def _detect_rep_anomaly(reqs: list[dict], max_tokens: int,
                        echo_mode: bool) -> tuple[str, int] | None:
    """一次复测批（point.reqs）的异常输出识别：返回 (异常类型, 肇事 req
    下标) 或 None。只看无 err 的请求——失败的请求已由错误语义化口径处理，
    不是本机制的对象；全部失败则无异常可判。

    early_stop（仅 echo 模式启用）：finish=stop 且 out_tokens < 0.8×max_tokens。
    echo=模板续写口径下模型本应续写到输出预算上限，提前 stop 即异常
    （实测案例：4K 档某次复测仅输出 4/512 tokens，echo 回复模式下被直接
    平均进聚合行，out_tokens 摊薄、decode 混入读数）。free 模式的提前停止
    是模型自主收尾的合法行为，不启用。

    degenerate（两种回复模式都启用）：text_sample 或 reason_sample 命中
    _is_degenerate_text（思考流退化走 reasoning 通道，头部样本可见）。

    两类同时命中时优先 early_stop：token 数证据是精确的定量口径，不依赖
    采样窗口恰好罩住退化段；弃测留痕本就带 text_sample/reason_sample，
    事后仍可按退化样本复核。"""
    ok = [(i, r) for i, r in enumerate(reqs) if not r.get("err")]
    if not ok:
        return None
    if echo_mode:
        for i, r in ok:
            ot = r.get("out_tokens")
            if r.get("finish") == "stop" and ot is not None \
                    and ot < 0.8 * max_tokens:
                return ("early_stop", i)
    for i, r in ok:
        if _is_degenerate_text(r.get("text_sample") or "") \
                or _is_degenerate_text(r.get("reason_sample") or ""):
            return ("degenerate", i)
    return None


def _is_implausible_prefill_rate(rate: float | None, ctx: float,
                                 curve_pts: list[tuple[float, float]]) -> bool:
    """物理速率闸：prefill 速率超该 ctx 先验曲线 RATE_PRIOR_OVERSHOOT 倍或
    绝对上限即判「物理上不可能」（占位软拒绝/拒答式假成功的通用兜底）。
    无先验时只按绝对上限判——首个请求没有可比基准，不误伤真实快后端。"""
    if not rate or rate <= 0:
        return False
    if rate >= RATE_ABS_CAP_TOK_S:
        return True
    prior = rate_prior(curve_pts, ctx)
    return bool(prior and rate > RATE_PRIOR_OVERSHOOT * prior)


def _classify_http_error(status: int, body: str) -> str:
    """把裸 HTTP 错误翻译成可行动的提示（ADR-0008）。"""
    low = body.lower()
    if status in (502, 503, 504):
        return (f"HTTP {status} 网关错误：网关等后端响应超时或后端不可达"
                f"（此时模型服务通常仍在继续计算。调大网关侧超时：LiteLLM 的 "
                f"request_timeout、nginx 的 proxy_read_timeout；超长 prefill 如 "
                f"192K 档在低配硬件上可能超过 10 分钟）")
    if _is_ctx_overflow(status, body):
        return (f"HTTP {status} 超出模型上下文窗口：{body[:160]} "
                f"（删减上下文重试后仍超限；在 config.json deployments 配 max_ctx "
                f"可自动跳过超限档位）")
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
        # 输出上限参数名实测判定（model → "max_completion_tokens"）：默认旧名
        # max_tokens；「usage 为真且输出超限」或触发客户端断流即判定旧名被网关
        # 忽略，单向改用新名（opencode zen 只认新名、DeepSeek 官方只认旧名、
        # Kimi/本地 vLLM 都认——行为无法按身份预判，只采信实测，ADR-0016）；
        # mt_mct_rejected = 新名被严格网关 400 拒收的模型（回退旧名并锁定，
        # 防翻转振荡）
        self.mt_param: dict[str, str] = {}
        self.mt_mct_rejected: set[str] = set()
        # 网关不回传缓存命中字段的一次性提示（LiteLLM 中转会剥离上游 usage
        # 扩展字段，实测只回传 prompt/completion/total 三字段）
        self._cache_note_done = False
        # (model, scenario) → 服务端前缀缓存 KV 块大小推断（已回传命中值的 gcd，
        # ≥64 才采信）：agent 链暖轮 tick 估值补前轮碎块尾用——命中按块对齐，
        # 前轮不足一块的尾部随本轮增量一起计算，漏估会系统性偏低 ~20%
        self.cache_block: dict[tuple[str, str], int] = {}
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
        # chunk_calib 加权滑动平均的累计权重（事件数，封顶 CHUNK_CALIB_W_CAP）
        self.chunk_calib_w: dict[tuple[str, str], float] = {}
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
            if repeats == 1:
                # 单次测量透明度：无统计效力，读数受瞬时抖动影响——播报一次
                # 提醒（不阻断运行），关键结论建议复测取均值
                await self.emit({"type": "status", "msg":
                    "repeats=1：单次测量无统计效力，读数含瞬时抖动；"
                    "关键结论建议 repeats≥3 复测取均值"})
            timeout = httpx.Timeout(connect=15, read=float(cfg.get("timeout_s", 600)),
                                    write=60, pool=30)
            limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
            # 不设 client 级 Content-Type：它会盖掉 files= 的 multipart 头（ASR
            # 上传会变成 application/json 正文，真实服务端同样拒收）；json=
            # 请求按需自带 application/json
            headers = {}
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
                        sc = SCENARIOS[scenario]
                        kind = sc.get("kind", "llm")
                        if kind != "llm":
                            # 媒体场景（asr/ocr/tts，ADR-0020）：场景自带阶梯（时长/
                            # 张数/字符），无探测/系数校准；模型 kind 不匹配是直调
                            # 引擎时的第二道闸（服务端校验已挡常规入口）
                            mk = (cfg.get("model_kind") or {}).get(model, "llm")
                            if mk != kind:
                                await self.emit({"type": "status", "msg":
                                    f"跳过 {model} / {sc['label']}："
                                    f"模型类型（kind={mk}）与场景（kind={kind}）不匹配"})
                                continue
                            for rung in sc["ladder"]:
                                if self.stop_flag:
                                    await self.emit({"type": "stopped"})
                                    return   # 终态收口在 finally
                                # 超窗保护（与 LLM 同口径）：asr 按 时长×tokens/秒、
                                # ocr 按 张数×图 token 均值+输出预算 估 prompt tokens
                                est_in = 0
                                budget = 0
                                if kind == "asr":
                                    est_in = rung * ASR_TOKENS_PER_SECOND
                                elif kind == "ocr":
                                    est_in = _ocr_avg_img_tokens() * rung
                                    budget = sc["max_tokens_default"]
                                est_tok = est_in + budget
                                rung_edge = None
                                if max_ctx:
                                    mode, in_eff, excess = _plan_ctx_edge(
                                        est_in, budget, max_ctx)
                                    if mode == "skip":
                                        await self.emit({"type": "status", "msg":
                                            f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                            f"估算 {est_tok} tokens 超出部署上限 "
                                            f"{max_ctx // 1024}K（config.json deployments.max_ctx）"})
                                        await self.emit({"type": "point_skipped",
                                            "model": model, "scenario": scenario, "ctx": rung,
                                            "count": len(cfg["concurrencies"]),
                                            "msg": f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                                   f"估算 {est_tok} tokens 超出部署上限 "
                                                   f"{max_ctx // 1024}K"})
                                        continue
                                    if mode == "edge":
                                        # 超窗量在阈值内：按 est→rung 线性反推有效档位
                                        # 贴边实测（asr 取整秒、ocr 取整张，向下取整保证
                                        # 仍在窗内）；裁不出有效档位（不足最小构造量）
                                        # 仍跳过，消息照旧口径
                                        rung_eff = int(in_eff // ASR_TOKENS_PER_SECOND
                                                       if kind == "asr"
                                                       else in_eff // _ocr_avg_img_tokens())
                                        if rung_eff < 1:
                                            await self.emit({"type": "status", "msg":
                                                f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                                f"估算 {est_tok} tokens 超出部署上限 "
                                                f"{max_ctx // 1024}K（config.json deployments.max_ctx）"})
                                            await self.emit({"type": "point_skipped",
                                                "model": model, "scenario": scenario, "ctx": rung,
                                                "count": len(cfg["concurrencies"]),
                                                "msg": f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                                       f"估算 {est_tok} tokens 超出部署上限 "
                                                       f"{max_ctx // 1024}K"})
                                            continue
                                        await self.emit({"type": "status", "msg":
                                            f"{model} / {sc['label']}{rung}{sc['unit']}："
                                            f"估算超部署上限 {excess / max_ctx:.0%}，"
                                            f"已裁减为 {rung_eff}{sc['unit']} 贴边测量"})
                                        rung_edge, rung = rung, rung_eff
                                for conc in sorted(cfg["concurrencies"]):
                                    if self.stop_flag:
                                        await self.emit({"type": "stopped"})
                                        return   # 终态收口在 finally
                                    await self.emit({"type": "status", "msg":
                                        f"测试 {model} / {sc['label']} / "
                                        f"{rung}{sc['unit']} / 并发 {conc}"})
                                    rep_points = []
                                    for rep in range(repeats):
                                        if repeats > 1:
                                            await self.emit({"type": "status", "msg":
                                                f"　复测 {rep + 1}/{repeats}（取均值）"})
                                        if kind == "ocr":
                                            pt = await self._run_ocr_point(
                                                client, model, scenario, rung, conc, rep)
                                        else:
                                            pt = await self._run_media_point(
                                                client, model, scenario, kind, rung, conc, rep)
                                        if self.stop_flag:
                                            break
                                        rep_points.append(pt)
                                    if self.stop_flag:   # 丢弃被中断点，交下一轮判停
                                        continue
                                    point = (rep_points[0] if len(rep_points) == 1
                                             else _aggregate_reps(rep_points))
                                    if rung_edge is not None:
                                        point["ctx_edge"] = rung_edge   # 留痕：贴边前的原档位
                                    self.results.append(point)
                                    await self.emit({"type": "point", "point": point})
                            continue
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
                        # 异常输出守卫的回复模式口径（与 _run_point 内部同款表达式，
                        # 循环外算好逐 rep 传入）：echo=模板续写才有「提前停止」
                        # 判据；退化重复判据两种模式通用，见 _run_rep_guarded
                        echo_mode = str(self.cfg.get("reply_mode")
                                        or "free").lower() == "echo"
                        # 超窗保护上限：prompt + 输出 + 分词漂移余量必须装进而上下文窗，
                        # 超过的档位构造出来必然 400，直接跳过（ADR-0005）
                        ctx_limit = (max_ctx - max_tokens - CTX_HEADROOM) if max_ctx else None
                        if scenario == "agent":
                            # agent 场景不以上下文档位为变量：连续任务链按任务轮次
                            # 推进（前缀缓存口径），逐轮出点，详见 _run_agent_chain
                            for conc in sorted(cfg["concurrencies"]):
                                if self.stop_flag:
                                    await self.emit({"type": "stopped"})
                                    return   # 终态收口在 finally
                                await self._run_agent_chain(client, model, scenario,
                                                            conc, base_cpt, ctx_limit,
                                                            repeats)
                            continue
                        for ctx in sorted(cfg["ctx_list"]):
                            ctx_edge = None
                            if max_ctx:
                                mode, ctx_eff, excess = _plan_ctx_edge(
                                    ctx, max_tokens, max_ctx)
                                if mode == "skip":
                                    # 目标档位超出部署实际上限且超出量超阈值：
                                    # 构造出的 prompt 必然 400，直接跳过并让前端
                                    # 把进度计为已处理（ADR-0005；阈值内的贴边
                                    # 裁减档位不在此列，见下方 edge 分支）
                                    await self.emit({"type": "status", "msg":
                                        f"跳过 {model} / 上下文{_fmt_ctx(ctx)}：连同输出预算超出部署上限 "
                                        f"{max_ctx // 1024}K（config.json deployments.max_ctx）"})
                                    await self.emit({"type": "point_skipped",
                                        "model": model, "scenario": scenario, "ctx": ctx,
                                        "count": len(cfg["concurrencies"]),
                                        # msg 随事件供前端常驻展示（status 行是瞬态的，
                                        # 跳过原因只说一次会被后续状态冲掉）
                                        "msg": f"跳过 {model} / 上下文{_fmt_ctx(ctx)}："
                                               f"连同输出预算超出部署上限 {max_ctx // 1024}K"
                                               f"（config.json deployments.max_ctx）"})
                                    continue
                                if mode == "edge":
                                    # 超窗量在阈值内：把上下文裁减到能放下（贴边）
                                    # 实测，不做整档缺席。point 的 ctx_target 记
                                    # ctx_eff——诚实记录实测档位（构建偏离重测的
                                    # 偏离分母随之也是 ctx_eff），ctx_edge 留痕原档位
                                    await self.emit({"type": "status", "msg":
                                        f"{model} {_fmt_ctx(ctx)} 档连输出预算超部署上限 "
                                        f"{excess / max_ctx:.0%}，已裁减为 "
                                        f"{_fmt_ctx(ctx_eff)} 贴边测量"})
                                    ctx_edge, ctx = ctx, ctx_eff
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
                                            f"　复测 {rep + 1}/{repeats}（取均值）"})
                                    pt = await self._run_rep_guarded(
                                        client, model, scenario, ctx, conc,
                                        cpt, rep, max_tokens, echo_mode)
                                    if self.stop_flag:
                                        break
                                    rep_points.append(pt)
                                if self.stop_flag:   # 停止时丢弃被中断的当前点，交给下一轮判停
                                    continue
                                point = (rep_points[0] if len(rep_points) == 1
                                         else _aggregate_reps(rep_points))
                                if ctx_edge is not None:
                                    point["ctx_edge"] = ctx_edge   # 留痕：贴边前的原档位
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

    async def _await_batch(self, tasks: list) -> tuple[list[dict], float]:
        """等待一批并发请求任务完成，返回 (reqs 按 req 序号排序, 批耗时)。
        在途任务用任务集管理：stop_flag 置位立即 cancel 全部（含 prefill
        等待期，否则长 prefill 无法中断，ADR-0008）；阻塞在 wait 期间无法
        感知 stop_flag，0.2s 轮询一次判停。_run_point 首批/构建偏离校正
        重测批、媒体批、agent 链轮批共用同一套判停结构。"""
        t0 = time.perf_counter()
        reqs = []
        pending = set(tasks)
        while pending:
            if self.stop_flag:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                break
            done, pending = await asyncio.wait(pending, timeout=0.2,
                                               return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    reqs.append(t.result())
                except asyncio.CancelledError:
                    pass
        reqs.sort(key=lambda r: r["req"])
        return reqs, time.perf_counter() - t0

    async def _run_rep_guarded(self, client: httpx.AsyncClient, model: str,
                               scenario: str, ctx: int, conc: int, cpt: float,
                               rep: int, max_tokens: int, echo_mode: bool) -> dict:
        """带异常输出守卫的单次复测：调 _run_point → 识别 early_stop/退化
        重复 → 命中则弃测该次复测、整批重跑一次（ANOMALY_RETEST_MAX）。conc>1
        时整批测量已被肇事请求污染（聚合行均值混入异常读数），弃测按整批
        计。重测换 nonce 破前缀缓存（_run_point 内部重掷 base_seed，bust 模式
        天然不命中缓存），不会复现同一异常停止。重测仍异常则接受结果、按实
        留档 pt["anomaly"]——真实模型行为不是可重试故障，不无限重试洗掉。
        仅 LLM 文本场景调用：媒体路径无 echo 续写口径（asr/tts/ocr 无输出
        预算可对齐），agent 链轮次合法短停（任务完成即收尾，本就 stop 在
        预算内），均不在范围。"""
        pt = await self._run_point(client, model, scenario, ctx, conc, cpt, rep)
        if self.stop_flag:   # 停止打断的半截批不可判：不检测直接返回
            return pt
        discarded = []
        for _ in range(ANOMALY_RETEST_MAX):
            hit = _detect_rep_anomaly(pt["reqs"], max_tokens, echo_mode)
            if hit is None:
                break
            kind, idx = hit
            culprit = pt["reqs"][idx]
            # 弃测留痕：异常类型、肇事请求下标及其关键读数（含输出头部样本，
            # 供事后人工复核退化形态；随聚合行 reps_discarded 逐条列出）
            discarded.append({
                "anomaly": kind, "req": idx,
                "out_tokens": culprit.get("out_tokens"),
                "finish": culprit.get("finish"),
                "ttft_s": culprit.get("ttft_s"),
                "decode_tok_s": culprit.get("decode_tok_s"),
                "text_sample": culprit.get("text_sample") or "",
                "reason_sample": culprit.get("reason_sample") or "",
            })
            await self.emit({"type": "status", "msg":
                (f"检测到异常输出（提前停止：仅输出 {culprit.get('out_tokens')}"
                 f"/{max_tokens} tokens），本次复测弃测重测" if kind == "early_stop"
                 else "检测到异常输出（退化重复），本次复测弃测重测")})
            pt = await self._run_point(client, model, scenario, ctx, conc,
                                       cpt, rep)
            if self.stop_flag:   # 重测在途被停：半截批同样不可判，弃循环
                break
            if _detect_rep_anomaly(pt["reqs"], max_tokens, echo_mode) is not None:
                pt["anomaly"] = kind   # 重测仍异常：按实留档，进聚合不重试
        if discarded:
            pt["reps_discarded"] = discarded   # 无弃测则不设此键
        return pt

    async def _run_point(self, client: httpx.AsyncClient, model: str, scenario: str,
                         ctx: int, conc: int, cpt: float, rep: int = 0,
                         media_msgs: tuple[list, list[int]] | None = None) -> dict:
        """单个测试点（conc 个并发请求一批）。media_msgs 非空为 OCR 媒体路径：
        (msgs_list, 每请求图片 token 估值)——跳过文本构造/校准，计时走 _one 的
        流式路径（VLM 输出天然流式，ADR-0020）。文本路径批完成后做构建偏离
        判定：实测 prompt_tokens 与目标档位偏差 > CTX_DEV_TOLERANCE 时按实测
        密度校正填充字符量整批重测一次（bust 模式，最多 1 次，见重测块注释）。"""
        base_seed = random.randint(100000, 999999)
        cache_mode = str(self.cfg.get("cache") or "bust").lower()
        # 回复模式（ADR-0021）：echo=模板续写——创意/代码场景换用改写指令并
        # 追加 assistant 预填，输出大段复用上下文，decode 对齐真实编辑场景的
        # 投机采样命中率；free（默认）=自由回答纯新生成。agent 无
        # echo_instruction 自动回退普通构造，媒体场景不走文本构造
        echo_mode = str(self.cfg.get("reply_mode") or "free").lower() == "echo"
        sc_kind = SCENARIOS[scenario].get("kind", "llm")
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
        img_tokens_list = [0] * conc
        filler_n = 0
        if media_msgs is not None:
            msgs_list, img_tokens_list = media_msgs
        else:
            for i in range(conc):
                # bust=每请求随机编号破坏 prefix cache（冷测真实 prefill）；
                # stable=固定编号，重复/并发请求命中官方上下文缓存以省成本
                nonce = "stable" if cache_mode == "stable" else f"{base_seed + i * 137}"
                msgs, _est, filler_n = build_messages(scenario, ctx, cpt, nonce,
                                                      filler_chars=filler_chars,
                                                      echo=echo_mode)
                msgs_list.append(msgs)

        max_tokens = int(self.cfg.get("max_tokens")
                         or SCENARIOS[scenario]["max_tokens_default"])
        # 在途请求任务集管理与 0.2s 轮询判停（_await_batch，与重测批/媒体批/
        # agent 链轮批共用）
        tasks = [asyncio.create_task(
                    self._one(client, model, msgs, scenario, i, ctx, conc, cpt,
                              max_tokens, rep=rep, img_tokens=img_tokens_list[i]))
                 for i, msgs in enumerate(msgs_list)]
        reqs, batch_time = await self._await_batch(tasks)
        ok = [r for r in reqs if not r.get("err")]

        # 输入系数校准前置到重测判定之前：失败的那一次测量也是真实密度样本，
        # 无论是否触发校正重测都要入校准（后续档位的嵌套估算会因本次样本
        # 立即变准，ADR-0005）
        if media_msgs is None:
            kb_before = self.prefix_kb.get(key)
            msgs_chars = sum(len(m["content"]) for m in msgs_list[0])
            await self._calibrate_usage(key, msgs_chars, filler_n, ok)
        else:
            kb_before = None
            msgs_chars = 0

        # 上下文构建偏离校正重测（ADR-0005 嵌套记账盲区：新语料段密度未知，
        # 滑动边际估算只能外推）。仅文本路径——agent 链有自己的轮次构造、
        # OCR 媒体路径不经文本构造，均不在范围。触发条件：实测 prompt_tokens
        # 均值与 ctx 相对偏差 > CTX_DEV_TOLERANCE + bust 模式（stable 重测会
        # 命中前缀缓存污染读数）+ 未重测过（最多 1 次）+ 未停止。
        # 删减重试成功点位（sent_chars < 构造量）的偏离源于服务端超窗保护，
        # 重测必然复现，不触发。
        ctx_retry = None
        if (media_msgs is None and ctx > 0 and cache_mode == "bust"
                and not self.stop_flag):
            reals = [r["prompt_tokens"] for r in ok if r.get("usage_real")]
            if reals and not any(r.get("sent_chars", 0) < msgs_chars for r in ok):
                real_mean = sum(reals) / len(reals)
                dev = (real_mean - ctx) / ctx
                if abs(dev) > CTX_DEV_TOLERANCE:
                    f_new = _correct_filler_chars(
                        filler_n, real_mean, ctx, kb_before,
                        msgs_chars - filler_n, cpt)
                    if f_new is not None:
                        await self.emit({"type": "status", "msg":
                            f"{model} / {SCENARIOS[scenario]['label']} "
                            f"{ctx // 1024}K 档构建偏离 {dev * 100:+.1f}%"
                            "（实测/目标），已按实测密度校正重测"})
                        ctx_retry = round(real_mean)
                        # 重测必须换全新 nonce（盐 999983 与步长 137 互质，
                        # 各槽位与首测均无碰撞）：前缀缓存命中的是共享前缀而
                        # 非整条内容，同 nonce 重测会与首测共享 nonce+指令+
                        # filler 前缀 → 大幅命中缓存污染 TTFT/prefill 读数
                        # （真实案例：32K 档重测 cache_hit 30720/32170）。bust
                        # 的 nonce 在 user 消息开头，换 nonce 即整段 bust。
                        # 整批重跑，用重测结果替换本点全部数据
                        retry_msgs = []
                        for i in range(conc):
                            nonce = f"{base_seed + 999983 + i * 137}"
                            msgs, _est, fn = build_messages(
                                scenario, ctx, cpt, nonce, filler_chars=f_new,
                                echo=echo_mode)
                            retry_msgs.append(msgs)
                        retry_tasks = [asyncio.create_task(
                                          self._one(client, model, msgs, scenario,
                                                    i, ctx, conc, cpt, max_tokens,
                                                    rep=rep))
                                       for i, msgs in enumerate(retry_msgs)]
                        reqs, batch_time = await self._await_batch(retry_tasks)
                        ok = [r for r in reqs if not r.get("err")]
                        # 重测样本同样入校准（真实密度观测，口径同首批）
                        await self._calibrate_usage(
                            key, sum(len(m["content"]) for m in retry_msgs[0]),
                            fn, ok)
        # prefill 估值先验逐点登记：conc=1 点的实测净速率入曲线（高并发点
        # 的批量效应会扭曲速率曲线，只取单发口径；0K 档是固定开销主导、
        # 非上下文标度，不入曲线）。媒体场景（图像 token 标度未接入曲线）不登记
        if media_msgs is None and conc == 1 and ctx > 0:
            rates = [(r.get("prefill_net_tok_s") or r.get("prefill_tok_s")) for r in ok]
            rates = [x for x in rates if x]
            if rates:
                self._note_prefill_rate((model, scenario), ctx,
                                        sum(rates) / len(rates))
        point = {
            "model": model, "scenario": scenario,
            "kind": sc_kind,   # 场景类型：llm/asr/ocr/tts，前端按此分叉（ADR-0020）
            "ctx_target": ctx, "concurrency": conc,
            "reqs": reqs, "all_ok": len(ok) == conc, "batch_time_s": round(batch_time, 3),
            "prompt_tokens": None, "ttft_s": None, "prefill_tok_s": None,
            "decode_tok_s": None, "decode_total_tok_s": None, "out_tokens": None,
            "ttft_net_s": None, "prefill_net_tok_s": None,
            "rtt_ms": round(self.rtt_s * 1000) if self.rtt_s is not None else None,
            "max_gap_s": None, "stall_s": None, "stall_count": None,
            "decode_burst": None, "tok_per_chunk": None,
            "decode_total_tok_s_adj": None,
            "ctx_retry": ctx_retry,   # 构建偏离校正重测的首次实测 prompt_tokens 留痕
        }
        if ok:
            gaps = [r["max_gap_s"] for r in ok if r.get("max_gap_s")]
            if gaps:
                point["max_gap_s"] = max(gaps)   # 传输/调度停滞诊断：取请求中最差
            stalls = [r for r in ok if r.get("stall_s")]
            if stalls:
                # 停滞复合记账取请求中最差（与 max_gap_s 同口径）：累计时长
                # 与次数同取自停滞最重的那个请求
                worst = max(stalls, key=lambda r: r["stall_s"])
                point["stall_s"] = worst["stall_s"]
                point["stall_count"] = worst.get("stall_count")
            bursts = [bool(r.get("decode_burst")) for r in ok if r.get("decode_tok_s")]
            if bursts and sum(bursts) * 2 > len(bursts):
                point["decode_burst"] = True   # 多数决（与 all_ok 同口径）
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
            # 平均每事件 token 数：投机解码网关（MTP/EAGLE 类）单事件携带 k 个
            # token 的诊断读数，各请求取均值
            tpc = [r["tok_per_chunk"] for r in ok if r.get("tok_per_chunk")]
            if tpc:
                point["tok_per_chunk"] = round(sum(tpc) / len(tpc), 2)
            if len(ok) == conc:
                # raw + adj 总吞吐（_total_decode_rates，与 agent 链同公式）。
                # 该定义仅在并发请求的 decode 区间充分重叠时有意义：长上下文
                # prefill 被服务端串行化时 decode 实际先后发生，窗口混入
                # prefill 等待期 → 断崖假象（ADR-0006），此时置空
                raw_total, adj_total = _total_decode_rates(ok, conc)
                if raw_total is not None:
                    point["decode_total_tok_s"] = raw_total
                if adj_total is not None:
                    point["decode_total_tok_s_adj"] = adj_total
            point["cache_hit_tokens"] = sum(r.get("cache_hit") or 0 for r in ok)
            fins = [r.get("finish") for r in ok if r.get("finish")]
            if fins:
                uniq = sorted(set(fins))
                point["finish"] = uniq[0] if len(uniq) == 1 else ",".join(uniq)
            point["out_tokens"] = round(sum(r.get("out_tokens") or 0 for r in ok) / len(ok))
        if media_msgs is not None:
            # OCR 媒体口径（ADR-0020）：单张时延 = 总耗时/张数；并发吞吐 =
            # 总张数/批窗口（全成功点才有）；LLM 侧 TTFT/decode 读数照常保留
            totals = [r["total_s"] for r in ok if r.get("total_s")]
            if totals:
                point["ms_per_img"] = round(sum(totals) / len(totals) * 1000 / ctx, 1)
            if len(ok) == conc and batch_time > 0:
                point["img_per_s"] = round(ctx * len(ok) / batch_time, 2)
            point["n_img"] = ctx
            return point
        return point

    async def _run_ocr_point(self, client: httpx.AsyncClient, model: str,
                             scenario: str, n_img: int, conc: int,
                             rep: int = 0) -> dict:
        """OCR 测试点：VLM 读图走 chat 流式路径（_one），消息由图池构造
        （多模态 content parts）。逐槽 nonce 错开取图，bust 服务端前缀缓存。"""
        seed = random.randint(100000, 999999)
        msgs_list, img_tokens_list = [], []
        for i in range(conc):
            msgs, img_tokens = _build_ocr_messages(n_img, seed + i * 137)
            msgs_list.append(msgs)
            img_tokens_list.append(img_tokens)
        return await self._run_point(client, model, scenario, n_img, conc,
                                     SCENARIOS[scenario]["in_cpt"], rep,
                                     media_msgs=(msgs_list, img_tokens_list))

    # -- 媒体请求路径（asr/tts，ADR-0020）-------------------------------------

    async def _run_media_point(self, client: httpx.AsyncClient, model: str,
                               scenario: str, kind: str, rung, conc: int,
                               rep: int = 0) -> dict:
        """asr/tts 测试点：与 _run_point 同构（并发任务集 + 0.2s 轮询判停），
        聚合走媒体口径（_aggregate_media_reqs）。"""
        # 在途请求任务集管理与 0.2s 轮询判停（_await_batch，与 _run_point
        # 首批/重测批、agent 链轮批共用）
        tasks = [asyncio.create_task(
                    self._one_media(client, model, scenario, kind, rung,
                                    i, conc, rep))
                 for i in range(conc)]
        reqs, batch_time = await self._await_batch(tasks)
        ok = [r for r in reqs if not r.get("err")]
        sc = SCENARIOS[scenario]
        point = {
            "model": model, "scenario": scenario, "kind": kind,
            "ctx_target": rung, "concurrency": conc,
            "reqs": reqs, "all_ok": len(ok) == conc,
            "batch_time_s": round(batch_time, 3),
            "rtt_ms": round(self.rtt_s * 1000) if self.rtt_s is not None else None,
        }
        point.update(_aggregate_media_reqs(kind, reqs, batch_time, conc))
        return point

    async def _one_media(self, client: httpx.AsyncClient, model: str,
                         scenario: str, kind: str, rung, req_i: int, conc: int,
                         rep: int) -> dict:
        """asr/tts 单请求：非 chat 端点、无 SSE 文本流。asr=multipart 上传整段
        音频等非流式 JSON；tts=audio/speech 二进制流（首音频字节计时）。RTF 与
        倍速按净耗时（扣 RTT 基线，与 LLM 同口径），错误处理与 _one 对齐。"""
        tag = f"{model}|{scenario}|{rung}|{conc}|rep{rep}|r{req_i}"
        sc = SCENARIOS[scenario]
        await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                         "ctx": rung, "conc": conc, "req": req_i, "phase": "prefill",
                         "tokens": 0, "speed": 0, "elapsed": 0})
        t0 = time.perf_counter()

        async def _fail(msg: str) -> dict:
            await self.emit({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": rung, "conc": conc,
                             "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}

        candidates = ([self.api_prefix] if self.prefix_locked
                      else ["/v1", ""])
        try:
            if kind == "asr":
                wav, audio_s = _asr_audio(rung)
                resp = None
                for prefix in candidates:
                    path = f"{prefix}/audio/transcriptions"
                    resp = await client.post(
                        path, files={"file": ("bench.wav", wav, "audio/wav")},
                        data={"model": model})
                    await self._note_mock(resp)
                    if (resp.status_code == 404 and not self.prefix_locked
                            and prefix == "/v1"):
                        self.api_prefix, self.prefix_locked = "", True
                        await self.emit({"type": "status",
                                         "msg": "检测到网关不带 /v1 前缀（DeepSeek 风格），已切换端点"})
                        continue
                    break
                if resp is None:
                    raise RuntimeError("无可用端点")
                if resp.status_code != 200:
                    return await _fail(_classify_http_error(
                        resp.status_code, resp.text[:300]))
                try:
                    j = resp.json()
                    text = j.get("text") if isinstance(j, dict) else None
                except Exception:  # noqa: BLE001
                    text = None
                if text is None:
                    text = resp.text[:4096]   # 非 JSON 兼容形态（如纯文本回显）
                t_end = time.perf_counter()
                elapsed, net = t_end - t0, _net_elapsed(t_end - t0, self.rtt_s)
                rtf, speed = net / audio_s, audio_s / net
                await self.emit({"type": "tick", "tag": tag, "model": model,
                                 "scenario": scenario, "ctx": rung, "conc": conc,
                                 "req": req_i, "phase": "decode",
                                 "tokens": round(audio_s, 1), "speed": round(speed, 1),
                                 "elapsed": round(elapsed, 2)})
                return {"req": req_i, "err": None, "first_abs": t0, "last_abs": t_end,
                        "audio_s": audio_s, "total_s": round(elapsed, 2),
                        "elapsed_net_s": round(net, 2),
                        "rtf": round(rtf, 4), "speed_x": round(speed, 1),
                        "out_chars": len(text)}

            # kind == "tts"：audio/speech 二进制流，response_format=wav 以解析时长
            text = _tts_text(rung, nonce=f"{rung}-{conc}-{req_i}")
            payload = {"model": model, "input": text, "response_format": "wav",
                       "stream": True}
            resp_ctx = None
            resp = None
            for prefix in candidates:
                path = f"{prefix}/audio/speech"
                resp_ctx = client.stream("POST", path, json=payload)
                resp = await resp_ctx.__aenter__()
                await self._note_mock(resp)
                if (resp.status_code == 404 and not self.prefix_locked
                        and prefix == "/v1"):
                    await resp_ctx.__aexit__(None, None, None)
                    resp_ctx = None
                    self.api_prefix, self.prefix_locked = "", True
                    await self.emit({"type": "status",
                                     "msg": "检测到网关不带 /v1 前缀（DeepSeek 风格），已切换端点"})
                    continue
                break
            if resp is None:
                raise RuntimeError("无可用端点")
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                return await _fail(_classify_http_error(resp.status_code, body))
            buf = bytearray()
            ttfa = None
            last_emit = 0.0
            try:
                async for chunk in resp.aiter_bytes():
                    if ttfa is None:
                        ttfa = time.perf_counter()
                    buf += chunk
                    now = time.perf_counter()
                    if now - last_emit >= 0.4:   # 与 _one decode tick 同节流
                        await self.emit({"type": "tick", "tag": tag, "model": model,
                                         "scenario": scenario, "ctx": rung, "conc": conc,
                                         "req": req_i, "phase": "decode",
                                         "tokens": round(len(buf) / 1024, 1),
                                         "speed": round(len(buf) / 1024
                                                        / max(now - t0, 1e-6), 1),
                                         "elapsed": round(now - t0, 2)})
                        last_emit = now
            finally:
                if resp_ctx is not None:
                    await resp_ctx.__aexit__(None, None, None)
            t_end = time.perf_counter()
            audio_s = _wav_seconds(bytes(buf))
            est = False
            if not audio_s:
                # 服务端忽略 response_format（回 mp3 等）或流被截断：按语速先验估
                # （合法但 0 帧的空 wav 返回 0.0，同样走估算兜底，避免除零）
                audio_s = len(text) / sc["speech_rate"]
                est = True
            elapsed, net = t_end - t0, _net_elapsed(t_end - t0, self.rtt_s)
            rtf, speed = net / audio_s, audio_s / net
            await self.emit({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": rung, "conc": conc,
                             "req": req_i, "phase": "decode",
                             "tokens": round(audio_s, 1), "speed": round(speed, 1),
                             "elapsed": round(elapsed, 2),
                             "ttft": round(ttfa - t0, 3) if ttfa else None})
            return {"req": req_i, "err": None, "first_abs": ttfa or t0,
                    "last_abs": t_end, "audio_s": round(audio_s, 2),
                    "audio_est": est, "total_s": round(elapsed, 2),
                    "elapsed_net_s": round(net, 2),
                    "ttfa_s": round(ttfa - t0, 3) if ttfa else None,
                    "rtf": round(rtf, 4), "speed_x": round(speed, 1),
                    "out_bytes": len(buf)}
        except Exception as e:  # noqa: BLE001
            return await _fail(str(e)[:300])

    async def _calibrate_usage(self, key: tuple[str, str], msgs_chars: int,
                               filler_n: int, ok: list[dict]):
        """输入系数实测校准 + 嵌套前缀记账：用服务端真实 prompt_tokens 反推
        chars/token。llama.cpp 等分词与场景先验（in_cpt：创意 1.5 / 代码 3.9）
        仍可能有差，校准后后续上下文点按实测值构造（ADR-0005）。按字符量加权
        滑动平均（权重封顶，保留对语料流下游段落配比变化的适应性）；小样本
        （<2000 字符，如 0K 档纯指令）分词模板开销占比大、比例严重失真，
        采纳会带偏校准——实测 0K 档曾把 4K 点位构造带偏 20%+（ADR-0005）。
        _run_point 每点调用一次；agent 连续任务链每轮调用一次。"""
        chars = msgs_chars
        # 超窗删减重试成功的点位：实际发送量小于构造量，输入系数校准与嵌套
        # 前缀记账按实际发送字符数对齐（删减只掐中段填充，差值满额计入填充），
        # 否则删减点会把系数/前缀记账带偏
        sent = [r["sent_chars"] for r in ok if r.get("sent_chars")]
        if sent and sum(sent) / len(sent) < chars:
            chars = round(sum(sent) / len(sent))
            filler_n = max(0, filler_n - (msgs_chars - chars))
        reals = [r["prompt_tokens"] for r in ok if r.get("usage_real")]
        if chars >= 2000 and reals:
            real = sum(reals) / len(reals)
            obs = chars / real
            old = self.cpt_calib.get(key)
            w = min(self.cpt_calib_w.get(key, 0.0), 65536.0)
            self.cpt_calib[key] = _wma(old, w, obs, chars)
            self.cpt_calib_w[key] = w + chars
            new = self.cpt_calib[key]
            if old is None or abs(new - old) / old > 0.02:
                await self.emit({"type": "status", "msg":
                    f"{key[0]} / {SCENARIOS[key[1]]['label']} 输入系数按实测校准为 "
                    f"{new:.2f} 字符/token"})
            # 嵌套前缀记账 + 增量段比例滑动平均：增量观测 = 增量字符 ÷ 增量
            # tokens，前缀部分的模板/指令开销在差分中自然抵消
            prev = self.prefix_kb.get(key)
            if prev and filler_n > prev[0] and real > prev[1]:
                m_chars = filler_n - prev[0]
                m_obs = m_chars / (real - prev[1])
                if 0.3 <= m_obs <= 20:
                    mw = min(self.cpt_marginal_w.get(key, 0.0), 65536.0)
                    self.cpt_marginal[key] = _wma(self.cpt_marginal.get(key),
                                                  mw, m_obs, m_chars)
                    self.cpt_marginal_w[key] = mw + m_chars
            self.prefix_kb[key] = (filler_n, real)

    # -- Agent 连续任务链 --------------------------------------------------------

    async def _run_agent_chain(self, client: httpx.AsyncClient, model: str,
                               scenario: str, conc: int, base_cpt: float,
                               ctx_limit: int | None, repeats: int):
        """Agent 连续任务链：K 轮 append-only 增长会话，逐轮产出测速点。

        口径设计（对齐真实 agent 循环，区别于其他场景的单发冷测）：
        - 轮次任务在链开始前全部生成：轮 0 冷启动（独立配置 agent_cold_ctx，
          默认 10K，全量 prefill）；其后暖轮固定两阶段——阶段一短文本任务，
          每轮增量 ~N(1K) 正态采样钳制到配置的增量区间，升序排轮；阶段二
          长文本任务，4K 起步逐轮翻倍的 ladder（默认 12 暖轮 = 6 短 + 6 长：
          4K/8K/16K/32K/64K/128K），模拟偶发的大文件读取/长日志灌入。
          第 k 轮目标 = 前 k 轮增量累积和。语料流确定性嵌套 + 链内 nonce
          固定 → 冷启动轮全量 prefill，暖轮命中服务端前缀缓存、只增量
          prefill；nonce 跨 rep/链/运行随机，冷启动轮不受残留缓存污染。
        - 每轮聚合为一个点：ctx_target = 轮目标 tokens（排序/图表类别轴用），
          turn = 轮号（0 起，0=冷启动），turn_delta = 本轮构造增量 tokens，
          turn_phase = 0/1/2（前端按阶段出统计：0=冷启动看全量 prefill，
          阶段一暖轮看 TTFT 暖延迟，阶段二看增量 prefill）。
          prefill_tok_s 覆写为**增量口径** = 未命中 tokens ÷ TTFT（服务端
          回传 cache_hit 时按回传值，否则按链内实测差分估算；LiteLLM 等网关
          会剥离上游 usage 扩展字段导致无回传）；无缓存能力的服务端增量读数
          随全量 TTFT 同步走低——如实反映其 agent 循环「每轮全量重算」的
          低效，而非粉饰成全量速率。cache_hit_tokens 同理：回传缺失时填
          差分估算值并以 cache_reported=False 标记，前端加 ≈ 展示。
        - conc = 并行会话链数：同轮各链并发在途，跨链同轮聚合；链间 decode
          重叠口径不保证，decode_total_tok_s 不产出（保持诚实）。
        - 轮目标连同输出预算超出部署上限（ctx_limit）时链提前结束，
          point_skipped 补齐进度；某轮出现软/硬超窗错误（req 级 ctx_ovf）时
          同样终止链——失败轮照常出点（all_ok=False），剩余轮次 point_skipped
          计入进度（ADR-0018：按错误点收口，不删减重试、不继续加轮）
        - 超窗删减重试（_one 内）掐断某轮中段后，后续轮的前缀命中会缩短——
          命中读数如实反映，不特殊处理。
        """
        # 两阶段暖轮数：新配置显式给 agent_turns_p1/p2；旧配置 agent_turns
        # 单值对半切（⌈K/2⌉ 短 + ⌊K/2⌋ 长）兼容；链另有 1 轮冷启动（独立
        # 配置 agent_cold_ctx，默认 10K）
        p1_cfg, p2_cfg = self.cfg.get("agent_turns_p1"), self.cfg.get("agent_turns_p2")
        if p1_cfg is not None and p2_cfg is not None:
            n1 = max(0, min(AGENT_TURNS_RANGE[1], int(p1_cfg)))
            n2 = max(0, min(AGENT_TURNS_RANGE[1], int(p2_cfg)))
        else:
            turns = max(AGENT_TURNS_RANGE[0], min(AGENT_TURNS_RANGE[1],
                        int(self.cfg.get("agent_turns") or AGENT_TURNS_DEFAULT)))
            n2 = turns // 2
            n1 = turns - n2
        turns = 1 + n1 + n2   # 轮 0 冷启动 + 轮 1.. 暖轮
        # 增量区间：新配置为 [min, max]（正态采样钳制区间）；兼容旧配置单值
        dv = self.cfg.get("agent_turn_delta")
        if isinstance(dv, (list, tuple)) and len(dv) == 2:
            dmin, dmax = int(dv[0]), int(dv[1])
        elif dv:
            dmin = dmax = int(dv)
        else:
            dmin, dmax = AGENT_TURN_DELTA_DEFAULT
        lo, hi = AGENT_TURN_DELTA_RANGE
        dmin, dmax = max(lo, min(hi, dmin)), max(lo, min(hi, dmax))
        if dmin > dmax:
            dmin, dmax = dmax, dmin
        # 轮次构成在链开始前一次性生成并定序：全部 rep 复用同一构成，复测可比
        p2_base = max(1024, min(65536,
                      int(self.cfg.get("agent_phase2_base") or AGENT_PHASE2_BASE)))
        lo, hi = AGENT_COLD_CTX_RANGE
        cold_ctx = max(lo, min(hi,
                       int(self.cfg.get("agent_cold_ctx") or AGENT_COLD_CTX_DEFAULT)))
        turn_plan = _gen_agent_turns(n1, n2, dmin, dmax, random.Random(),
                                     p2_base, cold_ctx)
        targets: list[int] = []
        acc = 0
        for s, _ph in turn_plan:
            acc += s
            targets.append(acc)
        ladder = [s for s, ph in turn_plan if ph == 2]
        max_tokens = int(self.cfg.get("max_tokens")
                         or SCENARIOS[scenario]["max_tokens_default"])
        key = (model, scenario)
        cpt = self.cpt_calib.get(key) or base_cpt
        ladder_txt = "/".join(f"{s // 1024}K" for s in ladder) or "无"
        await self.emit({"type": "status", "msg":
            f"{model} / Agent 连续任务链 / 并行会话 {conc}：{turns} 轮——"
            f"1 轮冷启动（{cold_ctx} tokens 全量 prefill）+ "
            f"{n1} 轮短文本（每轮增量 ~N(1K) 截断 {dmin}~{dmax} tokens）+ "
            f"{n2} 轮长文本（{ladder_txt} ladder），前缀缓存口径"})

        rep_points: list[list[dict]] = [[] for _ in range(turns)]

        async def _flush_rep_points():
            """已测点落盘收口：聚合各轮跨 rep 的点 → 存档 → 发事件。链提前
            收口（部署上限超窗 / 软硬超窗错误 / 用户停止）时同样执行，否则
            已完成轮次的点会被静默丢弃（既不入存档也不发事件）。repeats==1
            的点在轮内即时落盘（见下方 repeats==1 分支），此处不重复发出。"""
            if repeats == 1:
                return
            for pts in rep_points:
                if not pts:
                    continue
                point = pts[0] if len(pts) == 1 else _aggregate_reps(pts)
                self.results.append(point)
                await self.emit({"type": "point", "point": point})

        for rep in range(repeats):
            if repeats > 1:
                await self.emit({"type": "status", "msg":
                    f"　复测 {rep + 1}/{repeats}（取均值）"})
            base_seed = random.randint(100000, 999999)
            nonces = [f"{base_seed}-c{i}" for i in range(conc)]
            prev_prompt = [0] * conc   # 各链上一轮实测 prompt_tokens（增量差分兜底）
            # 暖轮 tick 估值口径跟随上一轮缓存判别结果（True/False；None=尚无
            # 依据，轮 1（首个暖轮）默认增量口径）——见下方 tick_est 注释
            prev_cache_active: bool | None = None
            # 冷轮（轮 0）TTFT 基准：服务端不回传缓存命中字段时的缓存迹象判别——
            # 暖轮 TTFT 相对冷轮走平（≤1.5×，含固定开销地板的 0.5s 余量）而
            # prompt 在增长，才算「有缓存迹象」，按链内差分估算命中；TTFT 随
            # prompt 增长则判定无缓存——命中显示未回传（None）、prefill 退回
            # 全量口径实测真值，不把假设的命中折算进速度。长文本轮增量占比
            # 过半，缓存收益在 TTFT 上不可分辨，天然落入「未回传」（宁缺毋假）
            cold_ttft: float | None = None
            for k in range(1, turns + 1):
                if self.stop_flag:
                    # 用户主动停止：本 rep 已完成轮次照常聚合落盘（未完成的
                    # 轮次本 rep 无点，聚合按实际参与 rep 数进行）
                    await _flush_rep_points()
                    return
                target = targets[k - 1]
                # 本轮构造增量（tokens）：两阶段构成下逐轮不等长
                turn_delta = target - (targets[k - 2] if k > 1 else 0)
                if ctx_limit is not None and target > ctx_limit:
                    skip_msg = (f"{model} / Agent 链第 {k - 1} 轮起连同输出预算"
                                f"超出部署上限，链提前结束（已完成 {k - 1} 轮）")
                    await self.emit({"type": "status", "msg": skip_msg})
                    # msg 随事件供前端常驻展示（status 行是瞬态的，会被后续状态冲掉）
                    await self.emit({"type": "point_skipped",
                        "model": model, "scenario": scenario, "ctx": target,
                        "count": turns - k + 1, "msg": skip_msg})
                    await _flush_rep_points()
                    return
                # 嵌套前缀记账精确加长（与 _run_point 同公式）
                kb = self.prefix_kb.get(key)
                filler_chars = None
                if kb:
                    marg = self.cpt_marginal.get(key) or cpt
                    filler_chars = max(0, round(kb[0] + (target - kb[1]) * marg))
                msgs_list = []
                filler_n = 0
                for i in range(conc):
                    msgs, _est, filler_n = build_messages(
                        scenario, target, cpt, nonces[i], filler_chars=filler_chars)
                    msgs_list.append(msgs)
                # 暖轮 tick 估值口径跟随上一轮缓存判别：有缓存（回传/判别命中）
                # → 本轮构造增量 + 前轮 prompt 的块对齐碎块尾（前缀缓存按 KV
                # block 命中，前轮不足一块的尾部随本轮一起增量计算；漏掉它会
                # 在 1K 增量上系统性低估 ~20%；块大小取 cache_block 的 gcd
                # 推断，未回传/推不出时不修正）。无缓存迹象 → 全量 prompt
                # 估值（tick_est=None 走 est_prompt），否则分子只按增量估、
                # 读数系统性偏低一半以上。首个暖轮尚无判别依据，默认增量口径
                tick_est = None
                if k > 1 and prev_cache_active is not False:
                    tail = 0
                    blk = self.cache_block.get(key)
                    prev = max(prev_prompt) if prev_prompt else 0
                    if blk and prev:
                        tail = prev - (prev // blk) * blk
                    tick_est = turn_delta + tail
                # 在途请求任务集管理与 0.2s 轮询判停（_await_batch，与
                # _run_point 首批/重测批、媒体批共用）
                tasks = [asyncio.create_task(
                            self._one(client, model, msgs, scenario,
                                      i * turns + (k - 1), target, conc, cpt,
                                      max_tokens, rep=rep, est_tokens=tick_est))
                         for i, msgs in enumerate(msgs_list)]
                reqs, batch_time = await self._await_batch(tasks)
                if self.stop_flag:
                    # 批次被中断：本轮（第 k 轮）req 数据不完整、点未构造，
                    # 本 rep 的第 k 轮不聚合（跳过），已完成轮次照常落盘
                    await _flush_rep_points()
                    return

                ok = [r for r in reqs if not r.get("err")]
                point = {
                    "model": model, "scenario": scenario, "ctx_target": target,
                    "turn": k - 1, "turn_delta": turn_delta,   # 轮号 0 起：0=冷启动
                    "turn_phase": turn_plan[k - 1][1], "concurrency": conc,
                    "reqs": reqs, "all_ok": len(ok) == conc,
                    "batch_time_s": round(batch_time, 3),
                    "prompt_tokens": None, "ttft_s": None, "prefill_tok_s": None,
                    "decode_tok_s": None, "decode_total_tok_s": None,
                    "decode_total_tok_s_adj": None,
                    "out_tokens": None, "ttft_net_s": None,
                    "prefill_net_tok_s": None,
                    "rtt_ms": (round(self.rtt_s * 1000)
                               if self.rtt_s is not None else None),
                    "max_gap_s": None, "stall_s": None, "stall_count": None,
                    "decode_burst": None,
                }
                if ok:
                    gaps = [r["max_gap_s"] for r in ok if r.get("max_gap_s")]
                    if gaps:
                        point["max_gap_s"] = max(gaps)   # 停滞诊断取最差
                    stalls = [r for r in ok if r.get("stall_s")]
                    if stalls:
                        worst = max(stalls, key=lambda r: r["stall_s"])
                        point["stall_s"] = worst["stall_s"]
                        point["stall_count"] = worst.get("stall_count")
                    bursts = [bool(r.get("decode_burst"))
                              for r in ok if r.get("decode_tok_s")]
                    if bursts and sum(bursts) * 2 > len(bursts):
                        point["decode_burst"] = True   # 多数决
                    point["prompt_tokens"] = round(
                        sum(r["prompt_tokens"] for r in ok) / len(ok))
                    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
                    if ttfts:
                        point["ttft_s"] = round(sum(ttfts) / len(ttfts), 3)
                    tns = [r["ttft_net_s"] for r in ok if r.get("ttft_net_s")]
                    if tns:
                        point["ttft_net_s"] = round(sum(tns) / len(tns), 3)
                    # 增量 prefill 口径：未命中 tokens ÷ TTFT。服务端回传缓存
                    # 命中时按回传值；未回传时先做缓存迹象判别（见 cold_ttft
                    # 注释）：TTFT 相对冷轮走平才按链内差分估算（hit≈前轮
                    # prompt），否则 hit=None（前端显示未回传）、uncached=全量，
                    # prefill 为全量口径真值。轮 0 冷启动 hit 恒 0
                    pps, pns, pfs, hits = [], [], [], []
                    for r in ok:
                        ptok, tt = r.get("prompt_tokens"), r.get("ttft_s")
                        if r.get("cache_reported"):
                            hit = r.get("cache_hit") or 0
                        elif ptok and k > 1:
                            tnet = r.get("ttft_net_s") or tt
                            if (cold_ttft and tnet
                                    and tnet <= max(1.5 * cold_ttft, cold_ttft + 0.5)):
                                hit = min(prev_prompt[r["req"] // turns], ptok)
                            else:
                                hit = None   # 无缓存迹象：不假设命中
                        else:
                            hit = 0
                        hits.append(hit)
                        if not ptok or not tt:
                            continue
                        uncached = max(ptok - hit, 1) if hit is not None else ptok
                        pps.append(uncached / tt)
                        pfs.append(ptok / tt)   # 全量口径（对照/换算用）
                        if r.get("ttft_net_s"):
                            pns.append(uncached / r["ttft_net_s"])
                    if k == 1 and ok:
                        # 冷轮 TTFT 基准（净口径优先，取跨链均值）：暖轮缓存迹象判别用
                        ts = [r.get("ttft_net_s") or r.get("ttft_s") for r in ok]
                        ts = [t for t in ts if t]
                        if ts:
                            cold_ttft = sum(ts) / len(ts)
                    # 本轮判别结果供下一轮 tick 估值口径沿用（有命中 = 回传真值
                    # 或判别估算的非零值）。轮 0 冷启动 hit 恒 0，不代表服务端
                    # 无缓存能力，不更新
                    if k > 1 and hits:
                        prev_cache_active = any(h for h in hits if h)
                    if pps:
                        point["prefill_tok_s"] = round(sum(pps) / len(pps), 1)
                    if pfs:
                        # 全量口径字段（additive）：prefill_tok_s 为增量口径，
                        # prompt_tokens ÷ TTFT 的原值供口径换算/对照
                        point["prefill_full_tok_s"] = round(sum(pfs) / len(pfs), 1)
                    if pns:
                        point["prefill_net_tok_s"] = round(sum(pns) / len(pns), 1)
                    dcs = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
                    if dcs:
                        point["decode_tok_s"] = round(sum(dcs) / len(dcs), 1)
                    dca = [r["decode_tok_s_adj"] for r in ok
                           if r.get("decode_tok_s_adj")]
                    if dca:
                        point["decode_tok_s_adj"] = round(sum(dca) / len(dca), 1)
                    # 平均每事件 token 数：agent 链同为 SSE 流式交付，点级取
                    # 各请求均值（与 decode_tok_s 同口径）
                    tpc = [r["tok_per_chunk"] for r in ok
                           if r.get("tok_per_chunk")]
                    if tpc:
                        point["tok_per_chunk"] = round(sum(tpc) / len(tpc), 2)
                    # 总吞吐（conc=1 单链口径，与 _run_point 同公式：总输出 ÷
                    # 首→末 token 窗口，含空窗校正 adj 口径）；conc>1 跨链
                    # decode 重叠口径不保证，保持 null（诚实，见 docstring）
                    if conc == 1 and len(ok) == 1:
                        raw_total, adj_total = _total_decode_rates(ok, 1)
                        if raw_total is not None:
                            point["decode_total_tok_s"] = raw_total
                        if adj_total is not None:
                            point["decode_total_tok_s_adj"] = adj_total
                    # 缓存命中：每轮跨链均值（单次会话口径），前端按轮展示。
                    # 网关未回传字段时 hits 装的是差分估算值（cache_reported=
                    # False，前端加 ≈）；连缓存迹象都没有时为 None——前端显示
                    # 「未回传」，不展示假设的命中数
                    known = [h for h in hits if h is not None]
                    point["cache_hit_tokens"] = (round(sum(known) / len(known))
                                                 if known else None)
                    point["cache_reported"] = all(
                        r.get("cache_reported") for r in ok)
                    if point["cache_reported"]:
                        # 块大小推断：回传命中值按 KV block 对齐（实测 256 倍
                        # 数），逐轮 gcd 收敛；gcd 退化到 <64 说明该后端的命中
                        # 口径不按块对齐（如 DeepSeek 云端），不用于碎块修正
                        g = self.cache_block.get(key, 0)
                        for r in ok:
                            h = r.get("cache_hit") or 0
                            if h > 0:
                                g = math.gcd(g, h)
                        if g >= 64:
                            self.cache_block[key] = g
                    fins = [r.get("finish") for r in ok if r.get("finish")]
                    if fins:
                        uniq = sorted(set(fins))
                        point["finish"] = uniq[0] if len(uniq) == 1 else ",".join(uniq)
                    point["out_tokens"] = round(
                        sum(r.get("out_tokens") or 0 for r in ok) / len(ok))
                    for r in ok:
                        if r.get("prompt_tokens"):
                            prev_prompt[r["req"] // turns] = r["prompt_tokens"]
                # 实测速率登记进先验曲线（conc=1 单链口径，与 _run_point 的高并发
                # 排除同理）：后续轮/rep 的开局估值锚点是相近上下文的真实速率，
                # 而非探测请求的全量速率。轮 0 冷启动登记全量口径、暖轮登记增量
                # 口径，均为 TTFT 端到端口径，与估值帧语义一致（ADR-0007）
                if conc == 1 and point.get("all_ok"):
                    rate = point.get("prefill_net_tok_s") or point.get("prefill_tok_s")
                    if rate:
                        self._note_prefill_rate(key, target, rate)
                await self._calibrate_usage(
                    key, sum(len(m["content"]) for m in msgs_list[0]),
                    filler_n, ok)
                rep_points[k - 1].append(point)
                if repeats == 1:
                    self.results.append(point)
                    await self.emit({"type": "point", "point": point})
                # 软/硬超窗错误（req 级 ctx_ovf 标记）：链就此终止，不再继续
                # 后续轮次——继续加轮只会造出更大的超窗 prompt，逐轮重复同一
                # 错误（实测事故：turn 12 超窗后继续加轮至 27 万 tokens）。
                # 按错误点收口不做删减重试（ADR-0018），剩余轮次以
                # point_skipped 计入进度（与部署上限提前收尾同口径）
                if any(r.get("ctx_ovf") for r in reqs):
                    ovf_msg = (f"{model} / Agent 链第 {k - 1} 轮超出服务端上下文上限，"
                               f"链提前结束（已完成 {k - 1} 轮；config.json deployments "
                               f"配 max_ctx 可自动跳档）")
                    await self.emit({"type": "status", "msg": ovf_msg})
                    if k < turns:
                        await self.emit({"type": "point_skipped",
                            "model": model, "scenario": scenario,
                            "ctx": targets[k], "count": turns - k, "msg": ovf_msg})
                    await _flush_rep_points()
                    return
        await _flush_rep_points()

    # -- 单条流式请求 ----------------------------------------------------------

    def _est_out(self, model: str, scenario: str, n_chunks: int,
                 n_chars: float, out_cpt: float) -> float:
        """输出 tokens 实时估算：按 SSE 内容事件数 ÷ 实测事件/token 比
        （每 token 一事件的服务上即真实值，不受中英/代码内容混合影响）；
        未校准时默认 1 事件/token 直计——OpenAI 兼容服务主流形态，chars/token
        随输出语言在 1.5~4.3 间漂移，字符系数先验不可信（fastllm 英文输出
        曾按 1.5 超发 2.8 倍触发断流、丢 usage、永远无法校准）；无事件可用时
        才回退字符数 ÷ 输出系数（ADR-0019）。"""
        ratio = self.chunk_calib.get((model, scenario))
        if ratio:
            return n_chunks / ratio
        if n_chunks:
            return float(n_chunks)
        return n_chars / out_cpt

    def _mt_name(self, model: str) -> str:
        """当前对 model 下发的输出上限参数名（实测判定，默认 max_tokens，
        ADR-0016）。"""
        return self.mt_param.get(model, "max_tokens")

    async def _note_mock(self, resp: httpx.Response):
        if resp.headers.get("x-mock-server") and not self.mock_seen:
            self.mock_seen = True
            await self.emit({"type": "status",
                             "msg": "检测到 mock 网关，本次运行不写入历史记录"})

    async def _one(self, client: httpx.AsyncClient, model: str, messages: list,
                   scenario: str, req_i: int, ctx: int, conc: int,
                   cpt: float, max_tokens: int, rep: int = 0,
                   quiet: bool = False, est_tokens: int | None = None,
                   img_tokens: int = 0) -> dict:
        # quiet=True（探测请求）：不发 tick/实时事件，只走完整请求链路并返回结果
        sc = SCENARIOS[scenario]
        # 输出估算系数取实测校准值（首个请求由探测结果播种），固定系数与实际
        # 输出内容不符时实时读数会数倍虚高（实测 70+ vs 25+，ADR-0005）
        out_cpt = self.out_cpt_calib.get((model, scenario)) or sc["out_cpt"]
        tag = f"{model}|{scenario}|{ctx}|{conc}|rep{rep}|r{req_i}"
        # messages 拷成 msgs_sent 再入 payload：超窗删减重试只改副本，
        # 调用方（_run_point）持有的原始构造不被改写
        msgs_sent = [dict(m) for m in messages]
        # 输出长度引导（ADR-0016 防线 2）：上限按 out_cpt 先验折成字数（×1.2
        # 容差）写进 system——服务端不截断时，模型自身尽量收在预设长度附近。
        # 先验固定，run 内 system 恒定（前缀缓存/嵌套前缀记账不受影响）
        limit_chars = int(max_tokens * sc["out_cpt"] * OUT_HINT_FACTOR)
        msgs_sent[0] = {**msgs_sent[0], "content":
                        msgs_sent[0]["content"] + "\n\n"
                        + sc["out_hint"].format(limit=limit_chars)}
        if isinstance(msgs_sent[1]["content"], list):
            # OCR 多模态 parts（ADR-0020）：text 部分按 in_cpt 折算，图像按
            # w×h÷750 估值（调用方传 img_tokens，与服务端口径近似、只影响展示）
            text_chars = sum(len(p.get("text", "")) for p in msgs_sent[1]["content"]
                             if isinstance(p, dict) and p.get("type") == "text")
            est_prompt = round(text_chars / max(cpt, 0.1)) + img_tokens
        else:
            # echo 请求带 assistant 预填消息（ADR-0021）：估值计入全部消息
            est_prompt = round(sum(len(m["content"]) for m in msgs_sent) / max(cpt, 0.1))
        # tick 展示估值：agent 链暖轮只增量 prefill，按调用方给的增量估值显示
        # （全量估值在短 TTFT 上会爆出天文数字）；est_prompt 本体仍用于 usage
        # 缺失时的兜底记账，不受展示估值影响
        tick_est = est_tokens or est_prompt
        if not quiet:
            await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                             "ctx": ctx, "conc": conc, "req": req_i, "phase": "prefill",
                             "tokens": 0, "speed": 0, "elapsed": 0,
                             "est_prompt_tokens": est_prompt})

        def _text_chars(content) -> int:
            """content 为 str（LLM）或 parts 列表（OCR image_url + text）。"""
            if isinstance(content, str):
                return len(content)
            if isinstance(content, list):
                return sum(len(p.get("text", "")) for p in content
                           if isinstance(p, dict) and p.get("type") == "text")
            return 0
        sent_chars = sum(_text_chars(m["content"]) for m in msgs_sent)   # 实际发送字符数（删减重试后 < 构造量）
        payload = {
            "model": model, "messages": msgs_sent, "stream": True,
            # 参数名按实测判定（ADR-0016 防线 1）：旧名被忽略的网关改用新名
            self._mt_name(model): max_tokens,
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
        stall_s = 0.0           # 停滞累计时长（≥STALL_THRESHOLD_S 的间隔满额计入）
        stall_count = 0         # 停滞次数：区分「一次大停滞」与「频繁小抖动」
        text_len = 0
        reason_len = 0          # reasoning_content（思考流）字符数，也计入测速
        head_text = ""          # 正文前 32 字符：识别服务端软超限占位回复（prompt too long）
        reason_head = ""        # 思考流前 32 字符：占位串也可能走 reasoning_content 通道返回
        # 存档诊断采样（正文/思考流开头 ≤200 字符）：取证退化输出形态（如
        # fastllm ≥16K 档 ~1 字符/token 的塌缩）与 echo 改写口径核对——
        # 仅入存档定位用，前端不消费；随 head_text 同型截获，零额外开销
        text_sample = ""
        reason_sample = ""
        n_chunks = 0            # 携带正文/思考内容的 SSE 事件数（≈token 事件）
        samples: deque[tuple[float, float, float | None]] = deque()
        # 滑窗采样 (t, est, 采样时的事件/token 比)：比随行记录，供窗口内校准
        # 比例变化时把基准样本折算到新口径再差分
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
            # 并发摊薄口径（保守）：速率曲线只登记 conc=1 实测点，conc>1 的
            # 等待期平台按 rate/conc 显示——服务端串行化时各流实际速率可摊薄
            # 2~8×；并行后端会显示偏低，但比沿用单流速率的虚高诚实
            rate = rate_prior(self.prefill_curve.get((model, scenario)) or [], ctx)
            plat = rate / conc if rate else None
            if plat:
                # 曲线先验封顶：等待期内估值 ≈ 该 ctx 的曲线速率（tick_est ÷
                # 预期耗时），超出预期耗时才随等待衰减收敛——速率随上下文
                # 先增后减由曲线斜率外推反映，不再从天文数字双曲起步
                net = max(net, tick_est / plat)
            # 估值帧过物理闸（与点收口 _is_implausible_prefill_rate 同口径）：
            # 无先验时净耗时下界 elapsed/2 会让首帧读出 ~2× 虚高甚至撞 10 万
            # tok/s 量级，钳到绝对上限/先验×OVERSHOOT
            cap = (RATE_ABS_CAP_TOK_S if not plat
                   else min(RATE_ABS_CAP_TOK_S, plat * RATE_PRIOR_OVERSHOOT))
            await self.emit({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": ctx, "conc": conc,
                             "req": req_i, "phase": "prefill",
                             "speed": round(min(tick_est / net, cap), 1),
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
            orig_user_chars = sent_chars
            ovf_attempt = 0   # 超窗删减重试计数
            while True:
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
                    # 模型级——如 Kimi K3 仅允许 0.6，省略后走服务端默认值（ADR-0004）；
                    # max_completion_tokens 为模型级回退——老规范网关不认新名时
                    # 改回 max_tokens 并锁定（mt_mct_rejected 防翻转振荡，ADR-0016）
                    for _param in ("thinking", "temperature", "max_completion_tokens"):
                        if resp.status_code != 400 or _param not in payload:
                            continue
                        body = (await resp.aread()).decode("utf-8", "replace")[:300]
                        if _param not in body.lower():
                            continue
                        await resp_ctx.__aexit__(None, None, None)
                        payload.pop(_param, None)
                        if _param == "thinking":
                            self.thinking_unsupported = True
                            note = "已自动省略并重试"
                        elif _param == "temperature":
                            self.temperature_locked.add(model)
                            note = "已自动省略并重试"
                        else:
                            payload["max_tokens"] = max_tokens
                            self.mt_param[model] = "max_tokens"
                            self.mt_mct_rejected.add(model)
                            note = "已回退 max_tokens 并重试"
                        await self.emit({"type": "status", "msg":
                            f"网关不接受 {_param} 参数（{body[:80]}），{note}"})
                        resp_ctx = client.stream("POST", path, json=payload)
                        resp = await enter_stream(resp_ctx)
                        await self._note_mock(resp)
                    break
                if resp is None:
                    raise RuntimeError("无可用端点")
                if resp.status_code == 200:
                    break
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                # 超窗回退：prompt 临近模型上下文上限时，输出随机性可能把窗口
                # 顶爆（context exceeded）——掐掉中段填充语料按 CTX_RETRY_KEEP
                # 逐档重试，而非直接判失败；重试仍超限才按错误收口
                if (_is_ctx_overflow(resp.status_code, body)
                        and ovf_attempt < len(CTX_RETRY_KEEP)
                        and orig_user_chars > 2000
                        and isinstance(msgs_sent[1]["content"], str)):
                    await resp_ctx.__aexit__(None, None, None)
                    resp_ctx = None
                    keep = int(orig_user_chars * CTX_RETRY_KEEP[ovf_attempt])
                    ovf_attempt += 1
                    msgs_sent[1] = {**msgs_sent[1],
                                    "content": _trim_middle(msgs_sent[1]["content"], keep)}
                    sent_chars = sum(len(m["content"]) for m in msgs_sent)
                    est_prompt = round(sent_chars / max(cpt, 0.1))
                    await self.emit({"type": "status", "msg":
                        f"{model} 上下文{_fmt_ctx(ctx)} 超出模型上下文窗口，"
                        f"删减填充语料至约 {keep} 字符重试"
                        f"（{ovf_attempt}/{len(CTX_RETRY_KEEP)}）"})
                    continue
                if _is_ctx_overflow(resp.status_code, body):
                    # 删减重试耗尽（或不适用删减的请求形态）仍超限：按错误收口
                    # 不再重试（ADR-0018）；ctx_ovf 标记供 agent 链终止后续轮次
                    msg = _classify_http_error(resp.status_code, body)
                    if not quiet:
                        await self.emit({"type": "tick", "tag": tag, "model": model,
                                         "scenario": scenario, "ctx": ctx, "conc": conc,
                                         "req": req_i, "phase": "error", "msg": msg})
                    return {"req": req_i, "err": msg, "ctx_ovf": True}
                raise RuntimeError(_classify_http_error(resp.status_code, body))
            pending_line = None
            try:
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
                        if len(head_text) < 32:
                            head_text = (head_text + piece)[:32]
                        text_sample = _head_sample(text_sample, piece, 200)
                    if rpiece:
                        reason_len += len(rpiece)
                        if len(reason_head) < 32:
                            reason_head = (reason_head + rpiece)[:32]
                        reason_sample = _head_sample(reason_sample, rpiece, 200)
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
                                                 "speed": round(tick_est / net0, 1),
                                                 "ttft": round(ttft0, 3),
                                                 "elapsed": round(ttft0, 2)})
                        elif now - last > max_gap:
                            # 相邻内容块最大空窗：decode 窗口被传输/调度停滞拖尾时
                            # （如隧道拥塞），该点读数不可信——记录供前端警示
                            max_gap = now - last
                        if first is not None and last is not None and now > last:
                            gap = now - last
                            adj_time += min(gap, DECODE_GAP_CAP)
                            if gap >= STALL_THRESHOLD_S:   # 满额计入，非时长折扣
                                stall_s += gap
                                stall_count += 1
                        last = now
                        if not quiet and now - last_emit >= 0.4:
                            # token 估算优先按 SSE 内容事件数（每 token 一事件的服务上
                            # 即真实值，与输出内容无关）；未校准回退字符系数（ADR-0007）
                            est = self._est_out(model, scenario, n_chunks,
                                                reason_len + text_len, out_cpt)
                            ratio = self.chunk_calib.get((model, scenario))
                            # 滑窗差分速度（3s）：prefill 后首批事件常成批到达
                            # （服务端缓冲/投机采样），累计均值会把这批在 TTFT 窗口
                            # 生成的 token 摊进 decode 分母 → 起步虚高且收敛慢；
                            # 窗口差分把突发留在基准样本里，只反映当前速率（ADR-0007）。
                            # 3s 窗口在响应性与平滑间取中：比 2s 更少受单批推送/
                            # MTP 接受率抖动的瞬时拉扯，与最终 usage 口径的偏差更小
                            samples.append((now, est, ratio))
                            cutoff = now - 3.0
                            while len(samples) > 1 and samples[0][0] <= cutoff:
                                samples.popleft()   # 至少留一个样本作差分基准
                            base_t, base_est, base_ratio = samples[0]
                            span = now - base_t
                            # 窗口内校准比例变化（加权平均仍在缓步移动/首次播种）
                            # 时，基准样本折算到新口径再差分，消除切换瞬间的
                            # 速度尖峰/负值
                            base_est = _ratio_refold(base_est, base_ratio, ratio)
                            # 钳非负：差分口径残余错配/抖动不向实时读数透出负速度
                            speed = (max(0.0, (est - base_est) / span) if span >= 0.3
                                     else max(0.0, est / max(now - first, 1e-6)))
                            await self.emit({
                                "type": "tick", "tag": tag, "model": model,
                                "scenario": scenario, "ctx": ctx, "conc": conc, "req": req_i,
                                "phase": "decode", "tokens": round(est, 1),
                                "speed": round(speed, 1),
                                "elapsed": round(now - t0, 2),
                                "ttft": round(first - t0, 3),
                                "thinking": bool(reason_len)})
                            last_emit = now
                    # 客户端断流兜底（ADR-0016 防线 3）：输出估算超下发上限
                    # ×CLIENT_CUT_FACTOR 仍未见 finish → 网关未执行截断，主动
                    # 断流（resp_ctx 收口时关闭连接），finish 记 client_cut；
                    # 旧名 max_tokens 在用时同时判定其被忽略，改用新名
                    if (piece or rpiece) and self._est_out(
                            model, scenario, n_chunks, reason_len + text_len,
                            out_cpt) > max_tokens * CLIENT_CUT_FACTOR:
                        finished = "client_cut"
                        if (self._mt_name(model) == "max_tokens"
                                and model not in self.mt_mct_rejected):
                            self.mt_param[model] = "max_completion_tokens"
                            await self.emit({"type": "status", "msg":
                                f"{model} 输出超 {max_tokens}×{CLIENT_CUT_FACTOR} "
                                f"仍未被截断：网关忽略 max_tokens 参数，后续请求改用 "
                                f"max_completion_tokens（本次已客户端断流）"})
                        else:
                            await self.emit({"type": "status", "msg":
                                f"{model} 输出超上限 ×{CLIENT_CUT_FACTOR} 仍未被截断，"
                                f"本次已客户端断流兜底（防跑飞）"})
                        break
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
        # 服务端软超限（fastllm 系）：超长 prompt 不报 4xx，而是 200 + 正文替换为
        # 占位串 "prompt too long"（finish=stop、单 token），content 与
        # reasoning_content 两个通道都可能携带。此时 TTFT/prefill 是「拒绝耗时 ÷
        # prompt 长度」，会冒出数十万 tok/s 的假数据——按错误点收口，与硬超限
        # （_is_ctx_overflow）同口径提示（config.json 配 max_ctx 自动跳档）
        if _is_soft_ctx_overflow(head_text, text_len + reason_len, reason_head):
            msg = (f"服务端将超长 prompt 替换为占位回复（prompt too long），该点数据无效："
                   f"{model} 在 {_fmt_ctx(ctx)} 超出服务端上下文上限"
                   f"（config.json deployments 配 max_ctx 可自动跳过超限档位）")
            if not quiet:
                await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg, "ctx_ovf": True}
        # 输出 token 兜底估算与 _est_out 同口径：SSE 事件数优先（未校准时按
        # 1 事件/token 直计），无事件才回退字符数 ÷ 输出系数（ADR-0019）——
        # 字符系数按中文先验 1.5 估英文输出（~4.2 字符/token）会超发 ~2.8 倍，
        # 触发断流丢 usage、永远无法校准的恶性循环
        est_chunks = (n_chunks / self.chunk_calib.get((model, scenario), 1.0)
                      if n_chunks else None)
        est_out = (reason_len + text_len) / out_cpt
        out_tokens = (usage.get("completion_tokens")
                      if usage and usage.get("completion_tokens")
                      else round(est_chunks if est_chunks is not None else est_out))
        prompt_tokens = (usage.get("prompt_tokens")
                         if usage and usage.get("prompt_tokens") else est_prompt)
        # 输出塌缩守卫（通用保险）：finish=stop 但输出近乎为空而 prompt 很大
        # ——占位软拒绝的占位串未必命中已知文案（_is_soft_ctx_overflow 只认
        # "prompt too long"），此类假成功同样会产出「拒绝耗时 ÷ prompt 长度」
        # 的假 prefill。阈值卡在大 prompt 上，0K/小 ctx 合法短输出不误伤
        if _is_collapsed_reply(prompt_tokens, out_tokens, finished,
                               text_len + reason_len):
            msg = (f"empty_reply_suspected：finish=stop 但输出近乎为空"
                   f"（{round(out_tokens)} tokens / {text_len + reason_len} 字符，"
                   f"prompt {round(prompt_tokens)} tokens）——疑似占位软拒绝或空回复，"
                   f"该点数据无效")
            if not quiet:
                await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        # 物理速率闸及其前置（TTFT/缓存解析/prefill 双口径）先于一切校准/
        # 状态播种：漏网的占位假成功（1 chunk/1 token，cobs=1 在界内）若先走
        # 校准块，会把已按投机网关校准的事件比重置回 1，实时读数成倍虚高
        ttft = (first - t0) if first else None
        # 前缀缓存命中：多字段名解析（DeepSeek/OpenAI/Anthropic 风格）；网关
        # 中转剥离扩展字段时记未回传，agent 链按链内差分估算
        cache_hit, cache_rep = _cache_hit_from_usage(usage)
        if (usage and not cache_rep and scenario == "agent"
                and not self._cache_note_done and not quiet):
            self._cache_note_done = True
            await self.emit({"type": "status", "msg":
                "网关未回传缓存命中字段（LiteLLM 等中转会剥离上游 usage 扩展字段）；"
                "agent 链将按 TTFT 自动判别缓存迹象：有迹象按链内差分估算（≈），"
                "无迹象则命中显示未回传、prefill 按全量口径；真值需直连推理后端"})
        # prefill 双口径：全量 = prompt_tokens ÷ TTFT；缓存命中时增出未命中口径
        # = (prompt_tokens − hit) ÷ TTFT（TTFT 只覆盖未命中段的计算，全量口径
        # 被命中率注水）。agent 链点级增量口径与先验曲线登记均基于未命中量，
        # req 级字段供口径换算与一致性校验
        prefill_full = (prompt_tokens / ttft) if ttft else None
        prefill_uncached = ((prompt_tokens - cache_hit) / ttft
                            if ttft and cache_rep and cache_hit else None)
        # 物理速率闸：占位软拒绝/拒答式假成功的通用兜底（实测事故：占位回复
        # ÷ 超长 prompt 判出 38 万 tok/s 假 prefill 入先验曲线并污染校准）。
        # 缓存命中时 TTFT 只覆盖未命中段，闸门按未命中口径评估，不误伤高命中率
        # 暖轮；无命中按全量口径。判假成功即按错误收口——且收口先于下方一切
        # 校准/参数名播种，假速率与假观测都不会入档
        gate_rate = prefill_uncached or prefill_full
        if gate_rate and _is_implausible_prefill_rate(
                gate_rate, ctx, self.prefill_curve.get((model, scenario)) or []):
            msg = (f"prefill 速率 {round(gate_rate)} tok/s 物理上不可能"
                   f"（超该上下文先验曲线 {RATE_PRIOR_OVERSHOOT}× 或绝对上限 "
                   f"{int(RATE_ABS_CAP_TOK_S)} tok/s）——疑似占位软拒绝/拒答式"
                   f"假成功，该点数据无效")
            if not quiet:
                await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        # 截断异常判定（ADR-0016 防线 1 的事中复核）：usage 为真且输出明显超下发
        # 上限（>20% 容差，兜住 reasoning 记账差异）→ 旧名未被服务端执行，单向
        # 改用 max_completion_tokens。超限未达断流阈值的漏网情形由此兜住；
        # 新名也不生效时不回翻（防振荡），由断流兜底
        if (usage and usage.get("completion_tokens")
                and out_tokens > max_tokens * 1.2
                and self._mt_name(model) == "max_tokens"
                and model not in self.mt_mct_rejected):
            self.mt_param[model] = "max_completion_tokens"
            await self.emit({"type": "status", "msg":
                f"{model} 实测输出 {round(out_tokens)} tokens 超出 max_tokens="
                f"{max_tokens} 上限，网关未执行截断，后续请求改用 max_completion_tokens"})
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
            # 每 token 一事件的服务上比例为 1，实时读数与真实值完全一致（ADR-0007）。
            # 下限 0.05 兜住投机解码网关（MTP/EAGLE 类）：单事件携带 k 个 token
            # 时 cobs=1/k，k 大则极小，卡太紧会把校准观测静默丢弃、usage 缺失时
            # out_tokens/decode_tok_s 系统性低估 k 倍
            if n_chunks:
                cobs = n_chunks / usage["completion_tokens"]
                # 加权滑动平均（口径同 cpt_calib）：投机网关接受率逐请求波动，
                # last-write-wins 整体覆写会让并发在途流的下一帧实时速度尖峰
                # 甚至为负（est 按新比例、base_est 按旧比例）
                if 0.05 <= cobs <= 5:
                    cold = self.chunk_calib.get(key)
                    cw = min(self.chunk_calib_w.get(key, 0.0), CHUNK_CALIB_W_CAP)
                    self.chunk_calib[key] = _wma(cold, cw, cobs, n_chunks)
                    self.chunk_calib_w[key] = cw + n_chunks
                    if cold is None:
                        await self.emit({"type": "status", "msg":
                            f"{model} / {sc['label']} 实时读数切换为 SSE 事件计数口径"
                            f"（{self.chunk_calib[key]:.2f} 事件/token）"})
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
            "cache_hit": cache_hit,
            "cache_miss": (usage or {}).get("prompt_cache_miss_tokens") or 0,
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "ttft_net_s": round(ttft_net, 3) if ttft_net is not None else None,
            "prefill_tok_s": round(prefill_full, 1) if prefill_full else None,
            "prefill_net_tok_s": (round(prompt_tokens / ttft_net, 1)
                                  if ttft_net else None),
            "prefill_uncached_tok_s": (round(prefill_uncached, 1)
                                       if prefill_uncached else None),
            "decode_tok_s": round(out_tokens / decode_time, 1) if decode_time else None,
            "decode_tok_s_adj": (round(out_tokens / adj, 1)
                                 if adj else None),   # 空窗校正口径
            "prompt_tokens": round(prompt_tokens),
            "out_tokens": round(out_tokens),
            "total_s": round(t_end - t0, 2),
            "max_gap_s": round(max_gap, 2) if max_gap else None,
            "stall_s": round(stall_s, 2) if stall_s else None,
            "stall_count": stall_count or None,
            # 突发交付：疑似网关缓冲冲刷，decode 速率虚高不可信
            "decode_burst": True if _decode_burst(
                n_chunks, decode_time, out_tokens) else None,
            # 平均每事件 token 数：out_tokens ÷ 交付事件数。投机解码网关
            # （MTP/EAGLE 类）按 draft+verify 周期交付，该值 ≈ 单事件平均
            # 接受长度 k；usage 缺失时 out_tokens 为估算值，字段口径不变
            # （仍是「交付事件平均携带 token 数」）
            "tok_per_chunk": round(out_tokens / n_chunks, 2) if n_chunks else None,
            "usage_real": bool(usage),
            # 服务端是否回传了缓存命中字段（agent 链增量口径：有回传按回传值，
            # 无回传按链内差分估算未命中量）
            "cache_reported": cache_rep,
            "sent_chars": sent_chars,   # 实际发送字符数（超窗删减重试后 < 构造量）
            "finish": finished,
            "text_chars": text_len, "reason_chars": reason_len,   # 存档诊断用
            # 正文/思考流开头 ≤200 字符采样（存档诊断用，前端不消费）：定位
            # 退化输出（~1 字符/token 的塌缩形态）与 echo 改写口径核对
            "text_sample": text_sample, "reason_sample": reason_sample,
        }
