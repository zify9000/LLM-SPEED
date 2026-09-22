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
- Decode 速度（逐请求）= completion_tokens ÷ (末 token 时间 - 首 token
  时间)；点级均速/总耗时为 Σ÷Σ 口径（decode_tok_s = Σout_tokens ÷ Σ各
  请求 decode 窗口、decode_time_s = Σ窗口，可互相验算，conc=1 与逐请求
  严格相等）
- 批级并发口径（_conc_batch_stats）：批起点 = 批内各请求计时起点最早值
  （start_abs）；prefill/total span = 批起点→全批最晚首/末 token；并发均速
  = 批 token 总量 ÷ 对应 span；decode 峰值 = 合并全批 token 交付时间轴的
  1s 滑窗最大速率（decode_peak_tok_s）
- 点级 TTFT/prefill 速度采用批级口径：TTFT = 批开始→全部首 token（即
  prefill_span_s）、prefill 速度 = Σprompt_tokens ÷ 该 span（conc=1 时批
  span 与单请求口径同源，严格相等）
- conc>1 的并发批受服务端资源挤占，decode 停滞/回吐诊断与空窗校正净速
  （decode_tok_s_adj）无判读意义，均不产出
- token 数优先取流式返回的 usage（stream_options.include_usage），
  网关/后端不支持时按「字符数 / chars_per_token」估算。
"""
from __future__ import annotations

import asyncio
import base64
import bisect
import hashlib
import io
import json
import logging
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

# 侧车外置失败只记日志、绝不冒泡（_save 不能因瘦身特性失败）
log = logging.getLogger(__name__)

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
        # 双向输出长度引导（ADR-0016）：目标 ≈ limit ÷ OUT_HINT_FACTOR + 硬上限
        # ——单向封顶无下限引导，实测模型往远低于预算的长度收敛，短输出 decode
        # 窗口不稳定；aim/limit 双占位由 _one 统一 format
        "out_hint": "篇幅要求：全文约 {aim} 字，不得超出 {limit} 字。",
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
        # 双向输出长度引导（ADR-0016）：同 creative，aim/limit 双占位由 _one 统一 format
        "out_hint": "篇幅要求：回复总长度约 {aim} 字符，不得超出 {limit} 字符。",
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
        # 缓存×指令矩阵（ADR-0042）：不以上下文档位为变量——每个缓存档先用一条
        # 预热请求把上下文写入服务端前缀缓存（不计入测点），再在缓存基础上逐
        # 指令档实测 TTFT/decode/总时长，对齐真实 agent「历史已缓存、单步新
        # 输入很短」的形态。ladder 为默认值，测速矩阵可自定义
        # （cfg.agent_cache_ladder / agent_inst_ladder 覆盖），出点循环见
        # _run_agent_matrix
        "cache_ladder": [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144],   # 已缓存上下文（tokens）
        "inst_ladder": [64, 128, 256, 512, 1024, 2048, 4096, 8192],             # 单步指令长度（tokens）
        # 指令模板：夹一段接续上下文之后的轨迹新材料（模拟新到的工具输出/日志），
        # 材料长度随档位伸缩；模板刻意精简（~30 tokens），64 tokens 小档不被
        # 固定开销淹没
        "inst_prefix": "New information from the ongoing task:\n\n",
        # suffix 要求 thorough/detailed/comprehensive：实测模型会以「任务上下文
        # 不完整/文件内容被截断」为由如实短答（308/310/401 tokens 即 finish=
        # stop），任务本身要自然引出长输出——逐步分析 + 具体命令/补丁的下一步。
        # 模板变长无妨：指令真实长度是链内差分测量，固定开销在差分中抵消
        "inst_suffix": ("\n\nRespond thoroughly and in detail: analyze the "
                        "current state step by step, then decide the next "
                        "action — one concrete action (a bash command or a "
                        "file edit) with its exact command or patch and the "
                        "expected outcome."),
        "filler_pool": _AGENT_POOL,  # 导入后由 _load_agent_pool() 覆盖（corpus 优先，内置块兜底）
        "in_cpt": 3.5,             # 输入系数先验：英文代码/日志/JSON 混合轨迹 ~3.5 字符/token（探测校准前的兜底）
        "out_cpt": 3.4,            # 输出估算：命令/补丁类英文 ~3.4 字符/token（仅 usage 缺失时的兜底）
        # 双向输出长度引导（ADR-0016）：目标 ≈ limit ÷ OUT_HINT_FACTOR（out_cpt
        # 先验字数，limit 已含 1.2 余量）+ 硬上限——实测模型以「材料截断」
        # 为由提前短答（308/310/401 tokens 就 finish=stop），单向封顶无下限
        # 引导弹不住。aim/limit 双占位由 _one 统一 format（creative/code
        # 同为双向；translate/ocr 等模板无 {aim}，多余 kwargs 不受影响）
        "out_hint": ("Length requirement: respond thoroughly and in detail, "
                     "aiming for about {aim} characters; never exceed {limit} "
                     "characters."),
        "max_tokens_default": 512,  # agent 单步动作短，512 足够覆盖思考+一个动作
    },
    "translate": {
        "label": "翻译（中→英）",
        # kind 缺省 llm：与创意/代码同一条 chat/completions 文本流路径；但不以
        # ctx_list 为变量——翻译负载短输入短输出（输入几十到几千字、输出随原文
        # 等比伸缩），原文字长阶梯替代上下文档位。ladder 为默认值，测速矩阵可
        # 自定义（cfg.translate_ladder 覆盖），出点循环见 run()
        "ladder": [50, 100, 200, 400, 800, 1600, 3200, 6400],   # 原文长度阶梯（字）
        "unit": "字原文",
        "system": ("你是专业翻译引擎，负责将用户给出的中文原文忠实、完整地翻译为"
                   "英文。只输出英文译文，不输出原文、解释或任何注记。"),
        "instruction": "请将上面的中文原文完整翻译为英文，只输出译文。",
        "zero_instruction": ("请将下面的中文句子翻译为英文，只输出译文："
                           "人生天地间，忽如远行客。"),
        # 语种方向固定中→英：待译原文复用创意语料流（红楼梦中文散文即真实负载，
        # 只测速不测 BLEU），filler_stream 在下方装配段接线
        "in_cpt": 1.5,   # 原文（中文散文）字符/token 先验：超窗保护与输出预算估算用
        "out_cpt": 4.0,  # 译文（英文）字符/token 先验（usage 缺失时的输出兜底）
        "out_ratio": 2.0,   # 输出预算 = 原文 tokens 估值 × 2 + 32（中译英实测
                            # ~1.2-1.4×，2× 留余量）；随档伸缩，全局 max_tokens 不适用
        "out_hint": "篇幅要求：译文与原文篇幅相称，全文控制在 {limit} 字符以内，不追加解释。",   # 输出长度引导（ADR-0016）
        "max_tokens_default": 512,   # 兜底键：正式点按 ladder 档位 × out_ratio 逐档计算
    },
    "asr": {
        "label": "语音转写",
        "kind": "asr",
        "ladder": [5, 15, 60, 300, 1800],   # 音频时长阶梯（秒，默认值），替代 ctx_list
                                           # 测速矩阵可自定义（cfg.asr_ladder 覆盖）
        "unit": "秒音频",
        "out_cpt": 1.5,   # 存档诊断：转写输出字符/token 兜底估值
    },
    "ocr": {
        "label": "图像识别",
        "kind": "ocr",
        "ladder": [1, 2, 4, 8, 16],        # 单请求图片张数阶梯（默认值），替代 ctx_list
                                           # 测速矩阵可自定义（cfg.ocr_ladder 覆盖）
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
        "ladder": [50, 200, 800, 3200],    # 合成文本长度阶梯（字符，默认值），替代 ctx_list
                                           # 测速矩阵可自定义（cfg.tts_ladder 覆盖）
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

# 贴边档未识别失败的回退裁减（与 CTX_RETRY_KEEP 的超窗删减重试互补）：点位
# 估算 prompt + 输出预算 + CTX_HEADROOM 余量达到部署 max_ctx × 本比例以上时
# （贴着窗口上限运行），报文不命中 _is_ctx_overflow 关键词的秒拒（如 fastllm
# 显存/KV 不足直接 500、连接被重置、空流）大概率仍是上下文/显存超限——按
# CTX_RETRY_KEEP 同一阶梯删减填充语料回退重试；阶梯耗尽才按原错误收口并在
# 错误文本追加疑似超限提示
CTX_EDGE_FB_RATIO = 0.85

# 同文本重试耗尽后的强制军备（ADR-0085，2026-09-22 拍板）：上面按「贴边」军备
# 的前提是「失败源于上下文/显存超限」——超限确实只发生在贴边档。但服务端还有
# 一类与档位无关的故障：Lvllm 的 Triton kernel JIT 编译按 shape 触发，编译所需
# 临时显存撞上已划走的显存池即 OOM，返回 200 但零 token（实测 3080×4 部署机
# 2026-09-22：27 次 JIT 告警中 6 次伴随 OOM，触发时点与档位无关）。此类故障下
# 同文本重试是**同一 shape**，必然重走同一条编译路径——反复重试撞同一面墙，
# 而唯一能改变 shape 的 _edge_fallback 因不贴边而未被军备。判据不需要新增探测：
# 「同文本重试已耗尽且仍空流」本身就证明「重试」这个动作无效，此时按同一
# 阶梯删减语料（改 shape）比继续判失败更可能救回该点。仅空流启用——瞬时类
# （超时/5xx）重试确有自愈价值，不得因耗尽而误改 shape 污染口径
EMPTY_STREAM_FORCE_EDGE_FB = True

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


def _out_hint_tail(sc: dict, max_ctx: int | None, ctx: int,
                   budget: int) -> str | None:
    """指令末尾复述的输出长度约束模板（双向引导场景专用，2026-09-16 拍板）：
    system 里的长度引导部分模型不遵守（实测 Qwen3.8-Flash-Next 持续早停
    4~28 tokens），同一约束在末条 user 消息末尾再写一份（近因效应——越
    靠近生成起点的约束越难被忽略）。仅 out_hint 含 {aim} 的双向场景
    （creative/code/agent；translate/ocr 单向封顶，引导写长是错的不复述）；
    返回未格式化的模板（调用方按 aim/limit 格式化），不适用返回 None。
    上下文贴边（ctx + 输出预算 + 余量 + 安全边 256 超 max_ctx）不加——省
    tokens 且防顶窗。闸取点位 ctx_target 而非逐请求 prompt 估值：agent
    矩阵同缓存档的预热/测量请求共用同一 ctx 参数，闸判定必然一致，链内
    差分（real_inst = 测量 − 预热 prompt）不受复述有无分叉影响。"""
    hint = sc.get("out_hint") or ""
    if "{aim}" not in hint:
        return None
    if max_ctx and ctx + budget + CTX_HEADROOM + 256 > max_ctx:
        return None
    return hint
# chunk_calib（事件/token 比）加权滑动平均的累计权重封顶（单位：事件数）：
# 保住对投机网关接受率变化的适应性（口径参照 cpt_calib 的 65536 字符封顶）
CHUNK_CALIB_W_CAP = 8192.0
# echo 请求的 assistant 预填长度（ADR-0021）：改写区开头原样预填这么多字符，
# 物理消除聊天式前言、强制从正文续写；太短压不住前言习惯，太长会白占输出预算
ECHO_PREFILL_CHARS = 160

# 瞬时失败兜底重试（网关抖动自愈）：单个请求失败不再立刻污染测速点
# （all_ok=False），瞬时类失败按短线性退避整体重来（全新计时/流状态），仍失败
# 才按既有错误口径收口。范围：httpx 传输异常（连接失败/读写超时/协议错误）、
# HTTP 429 与 5xx（带 Retry-After 头时封顶遵守）、空流（200 但无任何输出
# token，多为网关瞬时断流）。不重试：ctx_ovf（硬/软超窗，删减重试是另一套
# 机制）、其他 4xx（参数/鉴权等确定性错误）、塌缩回复与物理速率闸（数据有效
# 性判定，确定性形态，重试无法改变）
TRANSIENT_RETRY_MAX = 2       # 额外重试次数（连同首试共 3 次尝试）
TRANSIENT_RETRY_BASE_S = 1.0  # 线性退避：第 n 次重试前等 n 秒
# 空流专属重试预算（ADR-0080 降级、2026-09-22 拍板）：Lvllm 混合架构的
# 「空 EOS」（200 正常但零内容、completion≈1、finish=stop）是间歇性服务端
# 状态故障，同消息重试**可能**自愈，故与瞬时类分开计数、不挤占其预算。
# **次数取 2（与 TRANSIENT_RETRY_MAX 同口径）**：原为 4，依据是「每次约
# 50% 自愈、4 次恢复率 ≈94%」——该 50% 无实测支撑（ADR-0080 引用的现场
# 证据只有一条「败、败、过」序列，即 3 次里 1 次自愈，样本量为 1，不足以
# 支撑任何自愈率）。2026-09-22 复核后按「无依据不臆设恢复率」降为 2：
# 与通用瞬时重试同档，避免用编造的恢复率换取更长的失败等待（每次空流重试
# 白烧一次 prefill，32K 档约 80s）。空流仍保留独立预算的意义在于**不挤占**
# 瞬时重试额度，而非次数更多
EMPTY_STREAM_RETRY_MAX = 2
RETRY_AFTER_CAP_S = 5.0       # Retry-After 遵守上限（秒），防超大值拖死测速

# Agent 缓存×指令矩阵参数（ADR-0042 重构，取代旧「连续任务链」口径）：默认
# 阶梯在 SCENARIOS["agent"] 的 cache_ladder/inst_ladder；这里给预热输出预算
# 与阶梯校验范围（服务端 _validate_bench_cfg 同口径）
AGENT_PRIME_MAX_TOKENS = 16           # 预热请求输出预算（只求写入前缀缓存，内容丢弃）
AGENT_CACHE_RANGE = (0, 4 * 1048576)  # 缓存档取值范围（与自定义上下文档位同界）
AGENT_INST_RANGE = (16, 65536)        # 指令档取值范围
AGENT_LADDER_MAX = 16                 # 两条阶梯各自最大档数（与 ctx_list 同）

# 媒体场景（asr/ocr/tts，ADR-0020）计量常数：默认阶梯在 SCENARIOS[场景].ladder
# 给出，测速矩阵可经 cfg.<场景>_ladder 覆盖；ctx_list 不适用，并发矩阵与 LLM 共用
# whisper 类 ASR 编码器 ~50 tokens/音频秒（30s→1500 tokens）：max_ctx 超窗
# 保护按 时长×本系数 估算 prompt tokens
ASR_TOKENS_PER_SECOND = 50
# VLM 图像 token 估值（OpenAI 口径近似）：w×h ÷ 750
IMAGE_TOKEN_DIVISOR = 750
# 自定义阶梯校验范围（服务端 _validate_bench_cfg 与前端同口径）：最大档数
# 与各场景 (min, max) 取值界限，参照 AGENT_LADDER_MAX / AGENT_*_RANGE 模式
MEDIA_LADDER_MAX = 16
ASR_LADDER_RANGE = (1, 3600)    # 音频时长（秒）：1800s 默认顶档的 2 倍封顶
OCR_LADDER_RANGE = (1, 64)      # 单请求图片张数
TTS_LADDER_RANGE = (1, 65536)   # 合成文本长度（字）


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


# 未回传点缓存迹象判别的全量预期通道阈值：tnet ≤ 0.6×ptok/prior 判有迹象。
# 依据（实测 2026-09-12 DeepSeek-V4-Flash 大缓存档事故）：先验曲线 ±30%
# 不确定度下，真无缓存后端 tnet/全量预期 ≈0.8~1.3；真命中且缓存读取远快
# 于全量 prefill 时 ≲0.4；0.6 居间分划
CACHE_HIT_TTFT_RATIO = 0.6
# hit=None（未回传且无迹象）的 C>0 点登记先验曲线的防污染闸：速率超先验
# 该倍数即判「疑似漏判的缓存命中」不登记（实测 2026-09-16 事故：网关间歇
# 不回传预热 usage 时整档判别 None，4.5K~35K tok/s 虚高速率被当成全量
# prefill 正证据登记，先验污染后全量预期通道对后续点整档失效——真无缓存
# 后端的诚实速率落在先验 ±30% 带内，见 CACHE_HIT_TTFT_RATIO 的依据）
CACHE_SUSPECT_PRIOR_RATIO = 1.5


def _prior_suspect(c: int, reported: bool, hit: int | None,
                   prior_rate: float | None, rate: float) -> bool:
    """先验曲线登记的防污染判别：C>0 未回传且 hit=None（无迹象诚实留档）
    的点，速率超先验 CACHE_SUSPECT_PRIOR_RATIO 倍 → 疑似漏判的缓存命中
    （不是全量 prefill 正证据），不登记。c=0 / 回传点 / 无先验可对照时
    不判（照常登记）。真无缓存后端的诚实速率 ≈ 0.8~1.3× 先验，不会误伤。"""
    return (c > 0 and not reported and hit is None
            and prior_rate is not None
            and rate > CACHE_SUSPECT_PRIOR_RATIO * prior_rate)


def _prefill_prior_sample(c: int, ptok: float | None, reported: bool,
                          hit: int | None, rate: float | None,
                          rate_uncached: float | None,
                          curve: list[tuple[float, float]]
                          ) -> tuple[int, float] | None:
    """prefill 速率先验曲线登记的公共口径（LLM 泛用路径 / 探测播种 / agent
    矩阵三处同源），返回 (x, rate) 或 None（不登记）：
    - 估算命中点（C>0 未回传但判出命中，hit 非 None）不登记：其 TTFT 含
      缓存读取耗时、x≈未命中量，速率被摊薄，入曲线即污染估值锚点与闸门；
    - x 取未命中量 max(prompt_tokens − cache_hit, 1)——命中时用全量 prompt
      作 x 会把「缓存加速的大 prompt」算成虚高全量速率（stable 模式高命中
      档事故：x=ctx、rate=全量口径，先验被抬高后 10× 物理闸与 live 估值
      整轮失真）；
    - 命中回传时优先 req 级未命中口径 rate_uncached（rate 为全量口径）；
      agent 矩阵点级速率本身即未命中增量口径，rate_uncached 传 None；
    - 登记前过 `_prior_suspect` 防污染闸（hit=None 且速率超先验 1.5× 判疑似
      漏判缓存命中，不登记），与 agent 矩阵逐点登记同判据。"""
    if not rate:
        return None
    if c > 0 and not reported and hit is not None:
        return None   # 估算命中：TTFT 含缓存读取耗时，不登记
    x = max((ptok or 0) - (hit or 0), 1)
    r = (rate_uncached if (rate_uncached and reported and hit) else rate)
    if _prior_suspect(c, reported, hit, rate_prior(curve, x), r):
        return None
    return x, r


def _cache_hit_verdict(c: int, ptok: int, tnet: float | None,
                       base: float | None, prior_rate: float | None,
                       prime: int, proven: bool) -> int | None:
    """C>0 未回传点的缓存迹象判别：三通道 OR + 运行级佐证，返回估算命中
    tokens（min(prime, ptok−1)，前端标 ≈）或 None（无迹象，诚实留档）。
    回传/零缓存分支不在此函数内（调用点先分流，本函数只裁未回传点）。
    - 走平：tnet ≤ max(1.5×base, base+0.5)——同指令零缓存档基准（净口径
      优先），小缓存档可靠；base 缺失不判；
    - 全量预期：tnet ≤ CACHE_HIT_TTFT_RATIO×ptok/prior_rate——实测 TTFT
      显著低于同规模全量 prefill 预期。大缓存档必然失效走平（读缓存本身
      耗时随缓存规模线性增长），此通道兜住；无先验不判；
    - 运行级佐证：proven 为真直接估算、不再要求逐点证据——一轮 agent 矩阵
      中预热必然发出，缓存是否生效是后端属性而非逐点属性，一点证实全轮
      继承；
    prime=估算命中基准 tokens：预热实测 prompt 优先；网关间歇不回传预热
    usage 时调用方回退 ptok−inst（指令档目标）——实测 2026-09-16 本地
    网关事故：32K/128K 缓存档两轮预热 usage 全缺，prime=0 把 proven
    通道也短路、整档虚高速率留档。prime=0（两路基准都缺）恒 None。"""
    if c <= 0 or not ptok or not prime:
        return None
    est = min(prime, ptok - 1)
    if base and tnet and tnet <= max(1.5 * base, base + 0.5):
        return est
    if (prior_rate and tnet
            and tnet <= CACHE_HIT_TTFT_RATIO * ptok / prior_rate):
        return est
    if proven:
        return est
    return None

# 缓存档指令段缓存残留污染兜底判据（实测 2026-09-14 事故：128K 缓存档 ×
# 4K 指令档，prompt 134674 / cache_hit 133120——128K 上下文 + 上一档 1K
# 指令残留按块对齐计入命中，实测未命中仅 1554 vs 指令档期望 4056+，比值
# 0.38）：同缓存档指令材料若嵌套前缀，上一指令档残留被服务端算进
# cache_hit，增量 prefill 口径分子失真。均值 <0.6×期望 且绝对差 >512
# tokens 才判——64/256 小档块对齐噪声占比大（差值常 <512），免疫不判
CACHE_POLLUTION_RATIO = 0.6
CACHE_POLLUTION_MIN_GAP = 512


def _cache_pollution_verdict(reqs, real_inst, tail):
    """C>0 回传命中批的指令段缓存残留污染判别：每 ok 请求 uncached =
    prompt_tokens − cache_hit，期望值 = real_inst + 缓存块尾 tail（预热
    不足一块的尾部随测量增量计算，算法同 _run_agent_matrix 的 tick_est）。
    实测均值 < CACHE_POLLUTION_RATIO×期望 且差值 > CACHE_POLLUTION_MIN_GAP
    → 返回 (实测未命中均值, 期望值)；否则 None。仅 cache_reported 且
    cache_hit>0 的 ok 请求参与——未回传命中的网关走 _cache_hit_verdict
    估算路径，估算命中按预热 prompt 对齐、天然免疫此污染，不纳入本校验；
    real_inst 缺失（差分两侧 usage 不齐）不判。"""
    vals = [r["prompt_tokens"] - r["cache_hit"] for r in reqs
            if not r.get("err") and r.get("cache_reported")
            and (r.get("cache_hit") or 0) > 0 and r.get("prompt_tokens")]
    if not vals or real_inst is None:
        return None
    mean = sum(vals) / len(vals)
    expected = real_inst + tail
    if (mean < expected * CACHE_POLLUTION_RATIO
            and expected - mean > CACHE_POLLUTION_MIN_GAP):
        return round(mean), round(expected)
    return None

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
# 先验带只罩得住大 prompt：未命中量低于该值时小 prompt 区的诚实速率随 TTFT
# 固定开销摊薄而近线性爬升（实测事故：agent 矩阵零缓存档 inst=1024/4096 的
# 速率 569~976 tok/s 被误判——同运行小 prompt 兄弟点 307~722 tok/s 本就跨
# 2 倍以上；且闸门 x 曾误传 ctx_target，零缓存档 ctx=0 被 rate_prior 钳到
# 曲线最底端甚至冷启动慢样本），先验 10× 带在此区间没有判别力、只会误杀
# 诚实点——低于该值只按绝对上限判（软拒绝假成功本就发生在长 prompt）
RATE_GATE_MIN_TOKENS = 8192

# 异常输出识别 + 弃测重测：单次复测（rep 槽位）命中 early_stop/退化重复时，
# 整批测量已被污染（conc>1 时读数混入异常请求），弃测该次复测、换 nonce 重跑。
# 每个槽位最多重测 1 次——重测仍异常视为该部署在此档位的真实行为，按实留档
# 聚合，不无限重试把异常「洗掉」（与构建偏离校正重测的最多 1 次同口径）。
# llm 文本场景由 _run_rep_guarded 编排；agent 矩阵在 _run_agent_matrix 内
# 同口径接入——早停阈值放宽为 free 形态的 0.5×max_tokens（agent 单步动作
# 天然短、方差大，echo 的 0.8 判据不适配）
ANOMALY_RETEST_MAX = 1
# 地板失败重试上限（2026-09-16 拍板放宽、09-17 收紧到 2）：重测仍低于有效
# 输出地板（EARLY_STOP_FLOOR_OUT）的早停读数完全不可信、点位直接判测量
# 失败——比按实留档的异常更值得多给一次机会（偶发过载/服务端中断可能在
# 重试后恢复），但重测已换语料落点（ADR-0078）仍复现的多为确定型失效，
# 5 次时间成本太高，2 次足够覆盖瞬时抖动；高于地板的异常仍只重测 1 次
# （真实模型行为不无限重试洗掉）。两条路径同口径：_run_rep_guarded 与
# agent 矩阵
FLOOR_RETEST_MAX = 2

# 异常输入全文外置（ADR-0053 要求异常日志能让人精确复跑，但实测 11MB 级存档里
# in_text 占 99%）：短文本内联保可读性，超长文本落 <run_id>.anomaly.txt 侧车，
# 存档只留头尾摘要 + in_text_ref 字节切片引用（off/len/chars/sha1 → 精确复原）。
IN_TEXT_INLINE_MAX_CHARS = 4096        # 短文本留档内（可读性优先）
IN_TEXT_TRUNC_HEAD = 2000              # 超预算后的保留头
IN_TEXT_TRUNC_TAIL = 500               # 保留尾
# 侧车预算 = **安全阀，不是常规限流**（2026-09-20 定）：需要保持小的是存档 JSON
# ——它会被 json.load 全量解析（历史列表、明细、前端）；侧车只被流式下载、从不
# 解析，因此"把全文留在侧车"的代价只是磁盘（与改造前同一份数据，且不再被解析）。
# 实测最坏存档（results/20260916-182059-c658.json）in_text 共 10.8MB：32MB 阀门
# 下可全部留存、零截断，而存档 JSON 从 11.3MB 降到 ~0.25MB。真撞上阀门（异常轮
# 特别多）时按"该点无需精确复跑"降级并显式标注，宁缺不假。
ANOMALY_TEXT_BUDGET_BYTES = 32 * 1024 * 1024
ANOMALY_TEXT_SUFFIX = ".anomaly.txt"
ANOMALY_TEXT_BUDGET_WARNING = (
    f"异常文本存档超出单档预算 {ANOMALY_TEXT_BUDGET_BYTES} 字节：超预算条目的 "
    f"in_text 已截断且未留存全文（带 in_text_truncated 标），这些异常轮无法按"
    f"原文精确复跑")


def iter_in_text_holders(point: dict):
    """遍历一个测点里所有可能携带 in_text 的字典。

    落档只有三处：reqs[i].in_text、reps_discarded[j].in_text、
    reps[k].reqs[i].in_text（最后一条是 69% 体积的来源，不能漏）。容器/元素
    类型异常（非 list 的 reqs、非 dict 的条目）一律跳过，防御旧档/拼接混入的
    脏数据。
    服务端 splice/复测迁移引用时复用同一遍历，避免两侧漂移。
    """
    def _dicts(v):
        if not isinstance(v, (list, tuple)):
            return []
        return [x for x in v if isinstance(x, dict)]

    for key in ("reqs", "reps_discarded"):
        for h in _dicts(point.get(key)):
            yield h
    for rep in _dicts(point.get("reps")):
        for h in _dicts(rep.get("reqs")):
            yield h


def _truncate_in_text(text: str, has_ref: bool) -> str:
    """超长 in_text 的截断形态：保留头尾 + 省略标记。

    has_ref=False（预算耗尽未留存全文）时标记不得指向不存在的 in_text_ref——
    否则存档会谎称"全文可查"。
    """
    head = text[:IN_TEXT_TRUNC_HEAD]
    tail = text[-IN_TEXT_TRUNC_TAIL:] if IN_TEXT_TRUNC_TAIL else ""
    n = len(text) - len(head) - len(tail)
    if has_ref:
        marker = (f"\n……（中间省略 {n} 字符，全文见异常文本存档 in_text_ref；"
                  f"下载异常日志可导出）……\n")
    else:
        marker = f"\n……（中间省略 {n} 字符，超出异常文本存档预算未留存全文）……\n"
    return head + marker + tail


def _decode_burst(n_chunks: int, decode_time: float | None,
                  out_tokens: float | None = None) -> bool:
    if not decode_time:
        return False
    # 平均事件间隔的分母统一取 n_chunks − 1（首事件到末事件之间的间隔数）；
    # 少而肥分支曾用 n_chunks，3~4 个 chunk 时把间隔高估 1.3~1.5×、把
    # BURST_AVG_GAP_MS 边界推偏
    if (n_chunks >= BURST_MIN_CHUNKS
            and decode_time * 1000 / max(n_chunks - 1, 1) < BURST_AVG_GAP_MS):
        return True
    # 少而肥的 chunk：2~4 个多 token 事件亚毫秒冲刷，chunk 数过不了原判据——
    # 按每 chunk token 数增补甄别；out_tokens 缺 usage 时是估算值，仍可用
    return bool(n_chunks >= BURST_FAT_MIN_CHUNKS and out_tokens
                and out_tokens / n_chunks >= BURST_FAT_MIN_TPC
                and decode_time * 1000 / max(n_chunks - 1, 1)
                    < BURST_AVG_GAP_MS)


# tok_per_batch（均 x/批）口径常量：区分三种交付形态——
# ① 单事件多 token（vLLM/MTP 类，事件自带 k token）：fat = out_tokens/n_chunks
#    直接采信；② 逐 token 事件成批冲刷（llama.cpp 投机解码：事件口径恒 =1，
#    同一 verify 周期接受的 k 个 token 亚毫秒连续到达、下一周期前有明显间隙）：
#    按时隙双峰聚类识别批次、按批数计 k；③ 逐 token 交付记账噪声（无投机服务
#    偶发 2 token 合并进一个事件，比值 1.01~1.02）：无成批交付证据，不入档
SPEC_MIN_TPC = 1.15          # 低于此视为逐 token 交付的记账噪声，不记录
SPEC_GAP_SPLIT_S = 0.002     # 批内冲刷间隔上界：同一 verify 周期交付的事件亚毫秒到达
SPEC_BURST_MIN_N = 3         # 亚毫秒间隙计数下限（太稀不足以为凭）
SPEC_BURST_MIN_SHARE = 0.2   # 亚毫秒间隙占全部间隙比例下限
SPEC_BATCH_MIN_GAP_S = 0.01  # 批界间隔中位数下限：防整段缓冲冲刷（全亚毫秒）误判成一批

# 停滞回吐（stall-flush）甄别常量：≥STALL_THRESHOLD_S 的停滞间隙后紧随一串
# token 交付速率远超健康节奏的回吐事件——服务端把停滞期间积压的 tokens
# 一次性吐出，不是平稳生成（ADR-0054）。三轮实测形态（fastllm 27B，无 MTP，
# 健康 ~34 tok/s）：
# ① 2026-09-13：13.5s 停滞后 430+ tokens 亚毫秒 1-token 串（<2ms/事件），
#    交付 ~1000 tok/s；净速 34 被算成 216~333、tpc 10.5~27；
# ② 2026-09-14：6.0s 停滞后 ~176 tokens 以 ~8ms/事件 3-token 合并交付，
#    ~375 tok/s；tpc 3.08、adj 67.6；
# ③ 2026-09-14：6.61s 停滞（~24 tokens 后）后 ~230 tokens 合并成 ~9 个
#    26-token 事件按 ~22ms 间隔回吐，~1180 tok/s；tpc 7.76、adj 133.9——
#    事件间隔 22ms 既不 <2ms 也不 <0.5×节奏，间隔判据原理性漏检。
# 间隔信号堵不完 → 改看每事件 token 速率：burst-like = est_tok/max(gap,1ms)
# ≥ FLUSH_RATE_RATIO × r_pre。依据：生成速率不可能在停滞后瞬时翻 3 倍
# （三形态交付速率均为健康的 ~11~35 倍），3× 是保守下界，正常交付不误伤
FLUSH_MIN_TOK = 8        # episode 认定门槛：Σ burst est_tok ≥ 8（封顶后计）
FLUSH_RATE_RATIO = 3.0   # burst 起点速率倍数：交付速率 ≥ 3× 健康速率（防误触发）
FLUSH_TAIL_RATIO = 1.2   # burst 延续速率倍数：已入段后 >1.2× 健康速率即延续
                         # （停滞后的积压无论以什么速度排，交付速率高于正常生成
                         # 就是积压排放——拖尾降速也是回吐。无积压真停滞恢复
                         # 正常交付 =r_pre ≤1.2× → 立即停；MTP 批节奏 =r_pre
                         # → 不延续；实测拖尾形态：178 tok @~5× 猛冲后 55 tok
                         # 以 ~2× 拖尾，round-3 的 3× 延续把尾巴 ~55 tokens
                         # 漏剔 → adj 69.8 而真实 ~32）
FLUSH_RPRE_WINDOW = 30   # r_pre 局部速率窗口：停滞前最近 ≤30 个正常事件


def _flush_episodes(gaps: list[float], ev_toks: list[float]) -> list[dict]:
    """停滞回吐（stall-flush）片段识别（纯函数，per-event token 速率判据）。

    gaps[i] = 第 i 与 i+1 个内容事件的间隔（流内全量收集）；ev_toks[k] = 第 k
    个内容事件的 est tokens（字符增量 ÷ 校准输出系数、缺失兜底 1 token/事件，
    再按 usage 真值归一使 Σev_toks = out_tokens，caller 构造见 _one）。
    事件间隔判据堵不完三种回吐形态（①亚毫秒串 ②~8ms 合并 ③~22ms 大合并，
    见常量注释）——③的间隔已接近健康节奏，改看每事件 token 交付速率：
    burst-like = est_tok / max(gap, 1ms) ≥ FLUSH_RATE_RATIO × r_pre。
    r_pre = 停滞前正常事件（到达间隔 <0.5s、未归属先前 episode）的
    Σest_tok/Σgap，取最近 ≤30 个（不足 30 自然取全量）；停滞前无正常事件
    （停滞在流首）时 r_pre 缺失 → 仅按亚毫秒绝对规则（gap < SPEC_GAP_SPLIT_S）
    放行，保守不误伤。

    episode = 停滞间隙 + 紧邻其后的连续 burst-like 事件段：停滞段首事件
    （其到达间隔就是停滞本身，速率无意义）无条件入段；起点事件（第二个）
    须 ≥FLUSH_RATE_RATIO×r_pre（burst 真的开始，防误触发），已入段后延续
    条件放宽为 >FLUSH_TAIL_RATIO×r_pre——停滞后的积压无论以什么速度排，
    交付速率高于正常生成就继续算回吐（拖尾降速也是回吐）；est_tok≈0 的
    零内容/keepalive 事件跳过不中断；段在首个不满足延续条件的内容事件处
    收口。无积压真停滞（hang 后恢复正常交付 =r_pre）≤1.2× 立即停；MTP 批
    节奏 =r_pre 不延续。

    认定门槛：Σ burst est_tok ≥ FLUSH_MIN_TOK（封顶后计；fat 事件下事件数
    无意义，事件数门槛退役）；剔除封顶 r_pre × stall_s（防过剔——停滞期间
    服务端最多生成 r×S 个 token，剔超了会把正常生成误杀）。
    MTP 安全：单 verify 批 k≈3 tokens 即使亚毫秒到达 Σ≈3<8 不认定；≥3 批
    连冲（≥9 tokens）本就是积压，认定正确。

    返回 episode 列表：{stall_idx（停滞间隙下标）, start_idx/end_idx（回吐
    事件区间 [start, end)，end 为收口事件下标）, burst_events = end−start,
    burst_tok（封顶后剔除 tokens）, burst_s（串内间隙总时长 = 交付时间，
    供净口径撤出窗口）}；扫描跳过已归属片段不重复计。"""
    n = len(ev_toks)
    if not gaps or n < 2 or len(gaps) != n - 1 or sum(ev_toks) <= 0:
        return []
    episodes: list[dict] = []
    used: set[int] = set()   # 已归属 episode 的事件下标（r_pre 统计排除）
    gi, ng = 0, len(gaps)
    while gi < ng:
        if gaps[gi] < STALL_THRESHOLD_S:
            gi += 1
            continue
        # 停滞间隙 gi 位于事件 gi 与 gi+1 之间；r_pre 用停滞前的正常事件
        pairs = [(ev_toks[k], gaps[k - 1]) for k in range(1, gi + 1)
                 if gaps[k - 1] < STALL_THRESHOLD_S and k not in used]
        recent = pairs[-FLUSH_RPRE_WINDOW:]
        r_pre = None
        if recent:
            t_sum = sum(t for t, _ in recent)
            g_sum = sum(g for _, g in recent)
            if g_sum > 0:
                r_pre = t_sum / g_sum
        start = gi + 1
        j, burst_est = start, 0.0
        while j < n:
            if ev_toks[j] <= 0:   # 零内容/keepalive：跳过不中断
                j += 1
                continue
            if j > start:
                gap = gaps[j - 1]
                if r_pre is not None:
                    rate = ev_toks[j] / max(gap, 1e-3)
                    # 起点须 ≥3×（burst 真的开始了，防误触发）；已入段后放宽
                    # 为 >1.2×——停滞后的积压无论以什么速度排，交付速率仍高于
                    # 正常生成就继续算回吐（拖尾降速也是回吐），直到回落 ≤1.2×
                    # 或被 r×S 封顶
                    fast = (rate >= FLUSH_RATE_RATIO * r_pre if j == start + 1
                            else rate > FLUSH_TAIL_RATIO * r_pre)
                else:
                    fast = gap < SPEC_GAP_SPLIT_S   # 保守回退：仅亚毫秒
                if not fast:
                    break
            burst_est += ev_toks[j]
            j += 1
        end = j
        burst_tok = burst_est
        if r_pre is not None:
            burst_tok = min(burst_tok, r_pre * gaps[gi])   # 防过剔上限
        if end > start and burst_tok >= FLUSH_MIN_TOK:
            episodes.append({"stall_idx": gi, "start_idx": start,
                             "end_idx": end, "burst_events": end - start,
                             "burst_tok": burst_tok,
                             "burst_s": sum(gaps[start:end - 1])})
            used.update(range(start, end))
        gi = max(end - 1, gi + 1)   # 从收口事件的前一间隙继续扫描
    return episodes


def _tok_per_batch(gaps: list[float], n_chunks: int,
                   out_tokens: float | None) -> float | None:
    """平均每批交付 token 数（tok_per_chunk 新口径，纯函数）。三种形态：
    ① 单事件多 token（vLLM/MTP）：fat = out_tokens/n_chunks ≥ SPEC_MIN_TPC
       直接采信；② 逐 token 事件成批冲刷（llama.cpp 投机解码）：gaps 双峰，
       亚毫秒间隙（< SPEC_GAP_SPLIT_S）为批内冲刷、其余为批界，n_batches =
       1 + 批界数，tpb = out_tokens/n_batches ≥ SPEC_MIN_TPC 才采信——要求
       亚毫秒间隙数量与占比过门槛、批界中位数 ≥ SPEC_BATCH_MIN_GAP_S（防整段
       缓冲冲刷全亚毫秒、无批结构时误判成一批）；③ 其余（逐 token 交付记账
       噪声、无 gaps 证据）→ None（≈1 噪声不入档）。"""
    if not n_chunks or not out_tokens:
        return None
    fat = out_tokens / n_chunks
    if fat >= SPEC_MIN_TPC:   # ① 单事件多 token：保持事件口径现状
        return round(fat, 2)
    # ② 逐 token 事件成批冲刷：按时隙聚类识别 verify 批次
    sub = [g for g in gaps if g < SPEC_GAP_SPLIT_S]
    bounds = [g for g in gaps if g >= SPEC_GAP_SPLIT_S]
    if (len(gaps) >= SPEC_BURST_MIN_N and len(sub) >= SPEC_BURST_MIN_N
            and len(sub) / len(gaps) >= SPEC_BURST_MIN_SHARE and bounds
            and sorted(bounds)[len(bounds) // 2] >= SPEC_BATCH_MIN_GAP_S):
        tpb = out_tokens / (1 + len(bounds))
        if tpb >= SPEC_MIN_TPC:
            return round(tpb, 2)
    return None   # ③ 逐 token 交付记账噪声 / 无成批证据：不入档


def _fmt_ctx(ctx: int) -> str:
    return "0K" if ctx <= 0 else f"≈{ctx // 1024}K"


def _fmt_scenario_ctx(scenario: str, ctx: int) -> str:
    """被测点输入量的场景化标度展示（仅用于用户可见的 status/error 文案）：
    LLM/agent 的 ctx 是 prompt token 目标（走 _fmt_ctx 折 K），但 translate
    的 ctx 是原文字符数、OCR 的 ctx 是图片张数——统一折 K 会产出「≈0K」
    这种无意义标度（4 张图 → ≈0K）。LLM 档位格式保持不变。"""
    if scenario == "ocr":
        return f"{ctx} 张"
    if scenario == "translate":
        return f"{ctx} 字"
    return _fmt_ctx(ctx)


def _plan_ctx_edge(in_tok: int, budget: int, max_ctx: int) -> tuple[str, int, int]:
    """部署上限超窗的「贴边裁减」判定（纯函数，LLM 上下文阶梯与媒体阶梯共用；
    agent 矩阵的组合收口有自己的口径，不在此列）。in_tok 为该点输入侧构造目标
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
# translate 场景的待译原文同复用创意语料流（中文散文 = 真实中译英负载）
SCENARIOS["translate"]["filler_stream"] = SCENARIOS["creative"]["filler_stream"]
SCENARIOS["translate"]["filler_boundaries"] = SCENARIOS["creative"]["filler_boundaries"]


# ---------------------------------------------------------------------------
# 媒体场景语料：音频（asr/tts）与文档图（ocr），ADR-0020
# ---------------------------------------------------------------------------
# 测速负载只取决于媒体尺寸（音频时长×采样率 / 图像分辨率×图文密度），与内容
# 语义无关。内置语料确定性合成、零仓库体积、零新增依赖（stdlib wave/zlib/
# struct/base64）。
# - OCR 图片固定内置合成 3 种分辨率文档页（ADR-0037）：测速结论要跨环境/跨
#   时间可比，用户自备图分辨率各异会让同名档位负载不一、标准不统一；
# - corpus/asr/ 保留用户音频通道：wav 会被帧级循环/截取到精确目标时长，
#   负载按时长归一，不破坏可比性（真实语音对带 VAD 的服务端更保险）。

CORPUS_ASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "corpus", "asr")

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


# 4s 语音段 PCM 缓存（sr → bytes）：确定性内容只合成一次（见 _synth_speech_wav）
_SPEECH_SEG_CACHE: dict[int, bytes] = {}


def _synth_speech_wav(seconds: float) -> bytes:
    """合成目标时长的 speech-like wav（16kHz 16bit 单声道）：4s 语音段循环
    拼接——ASR 编码器负载只取决于时长×采样率，段重复不改变计算量。"""
    sr = 16000
    n = int(sr * seconds)
    # 4s 段确定性（与 sr 绑定、无随机）：只合成一次缓存复用——每次 _AUDIO_CACHE
    # 未命中都重算一遍纯浪费 CPU（~128KB PCM，长短档累计可观）
    seg = _SPEECH_SEG_CACHE.get(sr)
    if seg is None:
        seg = _SPEECH_SEG_CACHE[sr] = _speech_segment_pcm(sr, sr * 4)
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
# ASR 音频缓存字节预算：整段 wav 按阶梯档位常驻（默认阶梯 5~1800s ≈ 70MB，
# 自定义 3600s 可达 ~115MB），mock/多次 run 在进程内长期存活、无界增长。
# 超预算时按插入序逐出最旧档位（dict 有序）——字节预算优于「每矩阵清空」：
# 不依赖 run 生命周期钩子、并发 run 各自档位互相淘汰也不会把在用档位显式清掉。
# 单档自身超预算时单独保留：逐出它意味着每次请求都要重新合成数十 MB，代价
# 远大于内存（1800s 档 57.6MB 正是这种情形）
_AUDIO_CACHE_MAX_BYTES = 48 * 1024 * 1024


def _asr_audio(seconds: float) -> tuple[bytes, float]:
    """ASR 测试音频：(wav bytes, 实际秒数)。corpus/asr/*.wav 存在时取排序首个
    文件循环拼接到目标时长；否则合成 speech-like 音频。按目标秒数缓存
    （阶梯档位在重复/并发请求间复用同一份负载），并按字节预算逐出。"""
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
        while (len(_AUDIO_CACHE) > 1
               and sum(len(v[0]) for v in _AUDIO_CACHE.values())
                   > _AUDIO_CACHE_MAX_BYTES):
            _AUDIO_CACHE.pop(next(iter(_AUDIO_CACHE)))
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


_IMG_CACHE: list[tuple[bytes, int, int, str]] | None = None


# OCR 固定负载（ADR-0037）：3 种分辨率合成文档页（A4 150/200/300dpi 档），
# 跨环境/跨时间统一标准——读数可比性是测速台的核心产出，用户自备图分辨率
# 各异会让同名档位负载不一
_OCR_IMAGE_WH = ((1240, 1754), (1654, 2339), (2480, 3508))


def _ocr_images() -> list[tuple[bytes, int, int, str]]:
    """OCR 测试图池：固定 3 种分辨率的内置合成文档页（确定性生成）。
    返回 (bytes, w, h, mime) 列表，token 估值 w×h÷IMAGE_TOKEN_DIVISOR。"""
    global _IMG_CACHE
    if _IMG_CACHE is None:
        _IMG_CACHE = [(_synth_document_png(w, h, seed=i), w, h, "image/png")
                      for i, (w, h) in enumerate(_OCR_IMAGE_WH)]
    return _IMG_CACHE


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


def _build_translate_messages(n_chars: int, nonce: str) -> list:
    """翻译请求消息：编号行 + 待译原文（红楼梦语料流截取 n_chars 字）+ 翻译
    指令。编号行使 bust 模式各请求互不命中前缀缓存（与 _tts_text 同手法）。"""
    sc = SCENARIOS["translate"]
    body = f"原文片段编号 {nonce}：\n" + _fill(sc["filler_stream"], n_chars)
    return [{"role": "system", "content": sc["system"]},
            {"role": "user", "content": f"{body}\n\n{sc['instruction']}"}]


def _translate_max_tokens(src_chars: int, cpt: float,
                          cap: int | None = None) -> int:
    """翻译场景单档输出预算：译文 tokens 与原文同量级（中译英实测 ~1.2-1.4×），
    按 out_ratio（2×）留余量再补固定 32 兜住短档，并以全局 max_tokens 选项
    为上界（cap 语义与通用场景一致——高档位超界按 length 截断，读数按实
    留档）。预算随档伸缩：无 cap 时 6400 字档译文 ~8K tokens，定长预算必截断。"""
    sc = SCENARIOS["translate"]
    budget = int(src_chars / max(cpt, 0.1) * sc["out_ratio"]) + 32
    return min(budget, cap) if cap else budget


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _fill(stream: str, need_chars: int, offset: int = 0) -> str:
    """从确定性语料流截取 need_chars；流不够长（超大上下文档）时整流循环
    续接——内容仍确定，嵌套前缀性质保持。offset>0 时窗口起点循环平移
    （从流中 offset % len 处起截取，长度不变、内容不同）：仅弃测重测换料
    用——确定型早停与内容落点绑定，同料重测只会复现同一答案；平移打破
    嵌套前缀性质，首测/正常路径一律 offset=0。"""
    if need_chars <= 0 or not stream:
        return ""
    offset %= len(stream)
    total = offset + need_chars
    if total > len(stream):
        stream = stream * (total // len(stream) + 1)
    return stream[offset:total]


def _trim_middle(content: str, keep: int) -> str:
    """超窗删减重试：把 user 内容掐到约 keep 字符——保留开头（编号行）与
    结尾（任务指令），掐掉中段填充语料，指令结构不被破坏。keep <= 0 返回
    空串（此前 content[-0:] 会取回整串，删减反而把文本变长）。"""
    if keep <= 0:
        return ""
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
                   filler_chars: int | None = None, echo: bool = False,
                   stream_offset: int = 0):
    """构造一条目标长度约 target_tokens 的对话。
    返回 (messages, est_tokens, 填充字符数)。
    target_tokens <= 0 为零输入档：不注入参考材料，一句话指令直接命题。
    filler_chars 显式指定填充长度（嵌套前缀精确记账用，ADR-0005），
    缺省按 target_tokens × cpt 估算。stream_offset>0 时填充窗口循环
    平移到语料流另一处（_fill offset 口径）——仅弃测重测换料翻落点用，
    会打破嵌套前缀性质，正常路径恒为 0。
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
    filler = _fill(sc["filler_stream"], need_chars, offset=stream_offset)
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


def build_agent_messages(cache_tokens: int, inst_tokens: int, cpt: float,
                         nonce: str, cache_chars: int | None = None,
                         prime: bool = False, seg_chars: int | None = None,
                         seg_salt: int = 0):
    """构造「已缓存上下文 + 新指令」形态的 Agent 测速对话（ADR-0042）。
    返回 (messages, est_tokens, 上下文字符数, 指令材料字符数)。

    **预设上下文与指令拆开为两条 user 消息**：[system, user(编号行+轨迹
    上下文), user(指令)]。有预设上下文（cache_tokens>0）时先打一轮
    [system, 上下文, 超短任务] 预热请求把上下文写入服务端前缀缓存
    （prime=True），再打一轮 [system, 上下文, 指令] 测量请求——前两条
    消息逐字节一致、前缀缓存命中，只增量 prefill 指令消息；指令真实长度
    可由「测量 prompt − 预热 prompt」链内差分精确测得。上下文段只随
    cache_tokens 变化；指令 = 固定任务模板（inst_prefix/suffix）夹一段
    **接续上下文之后**的轨迹新材料（模拟 agent 循环新到的工具输出/日志），
    材料长度按 inst_tokens × cpt 构造（模板固定开销已扣除）。
    prime=True 构造预热/基线请求：同上下文 + 超短任务（输出预算极小、
    内容丢弃）。cache_tokens=0 为零缓存档：预热退化为 [system, 编号行,
    超短任务] 基线（量系统/模板/指令框架开销，供指令真实长度差分），
    测量为 [system, 编号行, 指令]（逐点换 nonce 破缓存，bust 口径）。
    seg_chars 显式指定指令材料字符数（构建偏离校正的补测用）；缺省按
    inst_tokens × cpt 估算。seg_salt 让指令材料在语料流中错位取段（长度
    不变、内容不同）——弃测重测换料防整条 prompt 二次命中缓存（缓存档
    前缀仍命中预热段）；缺省 0 与既有构造一致。cache_chars 显式指定上下文
    填充长度（嵌套前缀精确记账，与 build_messages 的 filler_chars 同口径）。"""
    sc = SCENARIOS["agent"]
    stream = sc["filler_stream"]
    system = sc["system"]
    head = f"参考材料（编号 {nonce}）：\n"
    n_ctx = 0
    if cache_tokens > 0:
        n_ctx = (cache_chars if cache_chars is not None
                 else max(0, int(cache_tokens * cpt) - len(head) - 16))
    ctx = _fill(stream, n_ctx)
    ctx_msg = {"role": "user", "content": f"{head}{ctx}"}
    if prime:
        messages = [
            {"role": "system", "content": system},
            ctx_msg,
            {"role": "user", "content": "Reply with OK."},
        ]
        est = (len(system) + len(ctx)) / max(cpt, 0.1)
        return messages, est, len(ctx), 0
    pre, suf = sc["inst_prefix"], sc["inst_suffix"]
    n_inst = (seg_chars if seg_chars is not None
              else max(1, int(inst_tokens * cpt) - len(pre) - len(suf)))
    # 指令材料取上下文之后的新段（循环流按长度取模错位；seg_salt 让重测
    # 换料——长度不变、内容不同，防整条 prompt 二次命中缓存），不与上下文同文
    off = ((len(ctx) + seg_salt) % len(stream)) if stream else 0
    seg = _fill(stream[off:] or stream, n_inst)
    messages = [
        {"role": "system", "content": system},
        ctx_msg,
        {"role": "user", "content": f"{pre}{seg}{suf}"},
    ]
    est_tokens = sum(len(m["content"]) for m in messages) / max(cpt, 0.1)
    return messages, est_tokens, len(ctx), len(seg)


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
    """并发组总吞吐双口径 (raw, adj)，_run_point 与 agent 矩阵共用同公式。

    raw = 总输出 tokens ÷（min(first)→max(last) 的窗口），仅当并发请求的
    decode 区间充分重叠（overlap ≥ 0.5×最短 decode）才有意义——长上下文
    prefill 被服务端串行化时 decode 实际先后发生，窗口混入 prefill 等待期
    → 断崖假象（ADR-0006），置空。
    adj = 空窗校正口径：窗口扣除各请求停滞累计（stall_s，首 token 后块间
    空窗）与回吐交付时长（flush_s，req 级剔除段总时长——回吐 tokens 剔出
    分子后其交付时间留在窗口会摊薄读数）；停滞回吐片段的 tokens（flush_tok，
    req 级剔除量）同步从分子扣除——停滞时间全额扣了而回吐 tokens 留在分子
    会双重虚高。剔除后分子 <16 tokens 或有效窗口 <0.2s → adj 置 None（与
    req 级「健康段不足不硬出数」同口径；实测事故：健康段被剔到只剩 8
    tokens/0.15s 时曾读出 51.2 的垃圾总吞吐）。首 token 前的 prefill 等待
    不在 stall 内，由重叠守卫兜底置空（128K 全串行案例 stall_s 全空，重叠
    守卫依旧拦下）。
    两口径共享全部产出守卫：len(ok)==conc、total_out>0、firsts/lasts 齐备、
    max(lasts)>min(firsts)、重叠守卫照旧；adj 另加 eff_window>0、分子 ≥16
    与窗口 ≥0.2s。"""
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
    eff_window = (span - sum(r.get("stall_s") or 0 for r in ok)
                  - sum(r.get("flush_s") or 0 for r in ok))
    adj = None
    if eff_window > 0:
        adj_out = total_out - sum(r.get("flush_tok") or 0 for r in ok)
        if adj_out >= 16 and eff_window >= 0.2:
            adj = round(adj_out / eff_window, 1)
    return raw, adj


# 并发 decode 峰值的滑窗宽度（秒）：合并全批 token 交付时间轴后的聚合窗口
DECODE_PEAK_WINDOW_S = 1.0


def _decode_peak_rate(timelines: list, window: float = DECODE_PEAK_WINDOW_S) -> float | None:
    """并发 decode 峰值速率（decode_peak_tok_s）：合并全批 token 交付时间轴后
    滑窗聚合速率的最大值（tok/s）。

    timelines = [(first_abs, gaps, ev_toks), …]（_one 的瞬态字段 _tl，每请求
    一条）：把每请求事件重建为 (t, tok)——t0 = first_abs、tok0 = ev_toks[0]，
    之后 t_i = t_{i-1} + gaps[i-1]、tok = ev_toks[i]；全批事件合并按 t 排序，
    双指针右对齐滑窗（窗宽 ≤ window），峰值 = max(窗内 tok 合计 ÷ window)。
    突发交付/停滞回吐形成的峰值如实计入——其不可信已由 decode_burst/flush
    字段分别标注，本口径不重复甄别。无有效事件返回 None。"""
    if window <= 0 or not timelines:
        return None
    pts: list[tuple[float, float]] = []
    for first, gaps, ev_toks in timelines:
        t = first
        for i, tok in enumerate(ev_toks):
            if i:
                t += gaps[i - 1]
            pts.append((t, tok))
    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    best = left = 0
    total = 0.0
    for t_r, tok in pts:
        total += tok
        while t_r - pts[left][0] > window:
            total -= pts[left][1]
            left += 1
        best = max(best, total)
    if best <= 0:
        return None
    return round(best / window, 1)


def _conc_batch_stats(ok: list[dict], conc: int) -> dict:
    """批级并发口径（6 指标，_run_point 与 agent 矩阵共用同公式）：

    batch_start = 批内各请求计时起点最早值（min(start_abs)，请求实际入批发
    有先后，串行构造/逐槽启动的错峰如实计入 span 起点）；prefill_span_s =
    max(first_abs) − batch_start（批开始→全部请求首 token 的 prefill 总时
    长）；total_span_s = max(last_abs) − batch_start（批开始→全部请求末
    token 的总时长）；decode_span_s = total − prefill（decode 有效时长）；
    prefill_conc_tok_s = Σprompt_tokens ÷ prefill_span_s（并发均 prefill
    速度）；decode_span_s > 0 时 decode_conc_tok_s = Σout_tokens ÷
    decode_span_s（并发均 decode 速度）；decode_peak_tok_s = 并发 decode
    峰值（_decode_peak_rate，消费各请求瞬态时间轴 _tl）。

    守卫与 _total_decode_rates 同口径：len(ok)==conc、每请求 start_abs /
    first_abs / last_abs 齐备且不为 None，否则对应字段 None；分母 span ≤ 0
    的字段同样置 None（decode_peak 不依赖 span，只看时间轴有无）。span 类
    round(x,2)、速率类 round(x,1)。ok = 批内无 err 请求。"""
    stats = {"prefill_span_s": None, "total_span_s": None, "decode_span_s": None,
             "prefill_conc_tok_s": None, "decode_conc_tok_s": None,
             "decode_peak_tok_s": None}
    starts = [r.get("start_abs") for r in ok]
    firsts = [r.get("first_abs") for r in ok]
    lasts = [r.get("last_abs") for r in ok]
    if (len(ok) != conc or any(s is None for s in starts)
            or any(f is None for f in firsts) or any(l is None for l in lasts)):
        return stats
    batch_start = min(starts)
    prefill_span = max(firsts) - batch_start
    total_span = max(lasts) - batch_start
    if prefill_span > 0:
        stats["prefill_span_s"] = round(prefill_span, 2)
        stats["prefill_conc_tok_s"] = round(
            sum(r.get("prompt_tokens") or 0 for r in ok) / prefill_span, 1)
    if total_span > 0:
        stats["total_span_s"] = round(total_span, 2)
        decode_span = total_span - prefill_span
        stats["decode_span_s"] = round(decode_span, 2)
        if decode_span > 0:
            stats["decode_conc_tok_s"] = round(
                sum(r.get("out_tokens") or 0 for r in ok) / decode_span, 1)
    stats["decode_peak_tok_s"] = _decode_peak_rate(
        [r["_tl"] for r in ok if r.get("_tl")])
    return stats


def _decode_sums(ok: list[dict]) -> tuple[float | None, float | None]:
    """点级 decode Σ÷Σ 口径（_run_point 与 agent 矩阵共用，ADR-0074 追加）：
    decode 总耗时 = Σ各请求 decode 窗口（last_abs − first_abs），decode 均
    速 = Σout_tokens ÷ Σ窗口——均速/总耗时同一分子分母源，可互相验算；
    conc=1 时批内仅单请求，Σ÷Σ 退化为 completion_tokens ÷ (末token−首
    token)，与逐请求口径严格相等。仅取 first_abs / last_abs / out_tokens
    齐备且 last>first 的请求（部分失败批/缺时间轴的请求不计入）；pairs
    空 → (None, None)；Σ窗口 ≤0（守卫下理论不可达）→ decode_tok_s 置
    None。decode_time_s round 2、decode_tok_s round 1（均速分母用未
    round 的窗口和，保证 conc=1 严格相等）。req 级 decode_tok_s 仍逐请
    求，不受此口径影响。"""
    pairs = []
    for r in ok:
        first, last = r.get("first_abs"), r.get("last_abs")
        out = r.get("out_tokens")
        if first is None or last is None or out is None or last <= first:
            continue
        pairs.append((last - first, out))
    if not pairs:
        return None, None
    raw_time = sum(w for w, _ in pairs)
    decode_time = round(raw_time, 2)
    tok_sum = sum(t for _, t in pairs)
    # 均速分母用未 round 的窗口和：conc=1 时与逐请求 decode_tok_s
    # （out ÷ 原始窗口）严格相等，round 2 只作用于产出的总耗时键
    decode_tok_s = round(tok_sum / raw_time, 1) if raw_time > 0 else None
    return decode_time, decode_tok_s


def _batch_prefill_point(tok_sum: float, stats: dict,
                         rtt_s: float | None) -> dict:
    """批级 prefill 点口径（_run_point 与 agent 矩阵共用）：TTFT = 批开始
    →全部首 token（即 stats["prefill_span_s"]），prefill 速度 = Σtokens ÷
    该 span，净口径经 _net_elapsed 扣 RTT 基线。批级口径，conc=1 时批 span
    与单请求口径同源（起点/终点同一 perf_counter 值），严格相等；span 取
    _conc_batch_stats 已 round 的值（0.01s 精度对速率影响可忽略）。

    stats["prefill_span_s"] 为 None（守卫未过/分母 ≤0，如部分失败批）时
    返回 {}，调用方回退逐请求均值口径。rtt_s 为 None 时净口径两键 None。"""
    span = stats.get("prefill_span_s")
    if not span:
        return {}
    ttft_net = _net_elapsed(span, rtt_s) if rtt_s is not None else None
    return {"ttft_s": round(span, 3),
            "prefill_tok_s": round(tok_sum / span, 1),
            "ttft_net_s": round(ttft_net, 3) if ttft_net is not None else None,
            "prefill_net_tok_s": (round(tok_sum / ttft_net, 1)
                                  if ttft_net is not None else None)}


# 聚合时的字段展示精度，与单点口径一致
_ROUND = {"ttft_s": 3, "ttft_net_s": 3, "prefill_tok_s": 1, "prefill_net_tok_s": 1,
          "prefill_uncached_tok_s": 1, "prefill_full_tok_s": 1,
          "decode_tok_s": 1, "decode_tok_s_adj": 1, "decode_total_tok_s": 1,
          "decode_total_tok_s_adj": 1,
          # 点级 decode Σ÷Σ 口径：总耗时（_decode_sums，span 类 2 位）
          "decode_time_s": 2,
          # 批级并发口径：span 类 2 位、速率类 1 位（与单点口径一致）
          "prefill_span_s": 2, "total_span_s": 2, "decode_span_s": 2,
          "prefill_conc_tok_s": 1, "decode_conc_tok_s": 1,
          "decode_peak_tok_s": 1,
          "tok_per_chunk": 2,
          "prompt_tokens": 0, "out_tokens": 0, "cache_hit_tokens": 0,
          "total_s": 2,   # agent 矩阵端到端总时长（TTFT + decode 全程）
          "inst_real_tokens": 0,   # agent 矩阵指令真实长度（差分口径，复测取均值）
          # 媒体场景（asr/ocr/tts，ADR-0020）：RTF/倍速/吞吐等点级指标同取均值
          "rtf": 4, "speed_x": 1, "audio_s": 1, "elapsed_s": 2, "elapsed_net_s": 2,
          "ttfa_s": 3, "ms_per_img": 1, "img_per_s": 2, "audio_min_per_min": 2,
          "out_chars": 0, "out_bytes": 0, "n_img": 0}


# ---------------------------------------------------------------------------
# 结果口径版本（存档顶层与每个测点各写一份）
#
# 为什么需要：ADR-0073/0074/0076/0079 反复改过 decode/TTFT/prefill 的口径，
# 旧存档永远靠前端的隐式"三级回退"猜字段含义——**没有任何机器可读的标记**
# 告诉读者这个点是哪一版口径测出来的。把存档单独发给别人、或把不同时期的点
# 拼接（ADR-0077）到一起时，这一点直接决定结论能不能这么比。
#
# 取值规则：**字段含义**变化时 +1 并在下表追加一行；无该字段的存档一律按
# LEGACY_METRIC_VERSION(0) 处理（本机制引入前的存量，前端走原隐式回退）。
# 只改实现不改口径（重构/修 bug/换语料落点）不递增。
# ---------------------------------------------------------------------------
METRIC_VERSION = 1
LEGACY_METRIC_VERSION = 0
METRIC_VERSION_NOTES = {
    1: "2026-09-20 起：批级 6 字段（ADR-0073）+ 单发/并发双轨口径与点级 Σ÷Σ"
       "（ADR-0074）+ 有效输出地板与指令复述约束（ADR-0076）"
       "+ echo 真续写形态（ADR-0079）",
}


def _first_err(reqs: list[dict] | None) -> str | None:
    """批内首个失败请求的错误文本（点级 err 落档取此为代表；逐请求明细
    仍在 reqs 内）。"""
    for r in reqs or []:
        if r.get("err"):
            return r["err"]
    return None


def _aggregate_reps(reps: list[dict]) -> dict:
    """多次复测聚合：标量取均值（ADR-0017——中位改均值，明细表聚合行即均值行，
    各次实测以复测子行逐次列出）；停滞/空窗诊断同取均值（出现停滞的复测之间），
    聚合行不再有「非均值」特例。reqs 明细取代表样本：pool 中有带 anomaly 的
    复测（重测仍异常按实留档）优先取该次的 reqs——留档样本必须来自异常那次
    （肇事 req 带 anomaly + in_text 供复盘）；无异常维持 decode 居中口径。
    all_ok 多数决：过半复测成功才视为成功点。"""
    ok_reps = [p for p in reps if p["all_ok"]]
    pool = ok_reps or reps
    anom_reps = [p for p in pool if p.get("anomaly")]
    rep_point = (anom_reps[0] if anom_reps else
                 sorted(pool, key=lambda p: p.get("decode_tok_s") or 0)
                 [len(pool) // 2])
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
    flush_reps = [p for p in reps if p.get("flush_tok")]
    if flush_reps:   # 停滞回吐剔除量同取均值（与 stall_s 同口径）
        point["flush_tok"] = round(sum(p["flush_tok"] for p in flush_reps)
                                   / len(flush_reps))
        point["flush_count"] = round(sum(p.get("flush_count") or 0
                                         for p in flush_reps)
                                     / len(flush_reps)) or 1
        point["flush_s"] = round(sum(p.get("flush_s") or 0
                                     for p in flush_reps)
                                 / len(flush_reps), 2)
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
    # 失败原因落档（此前失败轮的 err 只在运行期 toast 瞬态展示，存档无法
    # 回溯）：点级 err = 失败轮次的去重错误文本拼接；逐轮摘要带 err 字段。
    # 全部成功则不设此键（dict(rep_point) 可能带入单轮点的 err 键，需显式清）
    errs: list[str] = []
    for p in reps:
        if p.get("all_ok"):
            continue
        for r in p.get("reqs") or []:
            e = r.get("err")
            if e and e not in errs:
                errs.append(e)
    if errs:
        point["err"] = "；".join(errs)[:500]
    else:
        point.pop("err", None)
    # 复测摘要：供前端明细表逐次子行展示（字段与聚合行同口径；旧存档缺新增
    # 字段时前端按空值回退显示）。err 非点级字段，逐轮从该轮的 reqs 提取
    point["reps"] = [{**{k: p.get(k) for k in
                      ("prompt_tokens", "ttft_s", "prefill_tok_s", "prefill_net_tok_s",
                       "prefill_uncached_tok_s", "prefill_full_tok_s",
                      "decode_tok_s", "decode_tok_s_adj", "decode_total_tok_s",
                      "decode_total_tok_s_adj", "decode_time_s",
                      "prefill_span_s", "total_span_s", "decode_span_s",
                      "prefill_conc_tok_s", "decode_conc_tok_s",
                      "decode_peak_tok_s",
                      "tok_per_chunk",
                       "out_tokens", "all_ok", "anomaly", "finish", "stall_s",
                       "stall_count", "flush_tok", "flush_count", "flush_s",
                       "max_gap_s", "decode_burst",
                       "cache_hit_tokens", "cache_reported", "total_s",
                       "inst_real_tokens",
                       # 媒体场景逐次复测子行（ADR-0020）
                       "rtf", "speed_x", "audio_s", "elapsed_s", "ttfa_s",
                       "ms_per_img", "img_per_s", "audio_min_per_min",
                       "out_chars", "out_bytes")},
              "err": _first_err(p.get("reqs"))} for p in reps]
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


def _render_in_full(messages: list) -> str:
    """输入全文渲染（异常取证字段 _in_full 的内容）：每条消息一行
    「【role】文本」；content 为 parts 列表（OCR 多模态形态）时拼接 text
    部分、连续非 text 部分以「［图片×N］」占位。不设长度上限——异常日志
    要求输入原文全部列出；正常 req 的存档/SSE 在点定稿前剥离该瞬态字段
    （_strip_in_full），全文只随弃测留痕/留档异常轮以 in_text 落档。"""
    lines = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts: list[str] = []
            n_img = 0
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    if n_img:
                        parts.append(f"［图片×{n_img}］")
                        n_img = 0
                    parts.append(p.get("text", ""))
                else:
                    n_img += 1
            if n_img:
                parts.append(f"［图片×{n_img}］")
            body = "".join(parts)
        else:
            body = c if isinstance(c, str) else ""
        lines.append(f"【{m.get('role') or ''}】{body}")
    return "\n".join(lines)


def _strip_in_full(point: dict) -> None:
    """点定稿前剥离瞬态输入全文字段（_in_full）：全文只服务异常取证
    （抄进 reps_discarded.in_text / 肇事 req.in_text），正常 req 的存档与
    SSE 逐点推送都不得携带——100K/1M 上下文场景的全文会让每点负载爆炸。
    肇事 req 的 in_text 是独立键，剥离一并去掉 _in_full 原键。兜底剥离
    token 交付时间轴 _tl（并发批峰值口径的瞬态输入，正常路径已在出点前
    剥离，此处防御未走显式剥离的出点路径）。"""
    for r in point.get("reqs") or []:
        r.pop("_in_full", None)
        r.pop("_tl", None)


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


# 有效输出地板（tokens）：finish=stop 且输出低于该值 = 完全不可信的测量
# （实测 2026-09-16：Qwen3.8-Flash-Next 持续早停 4~28 tokens，重测一次
# 仍短就「按实留档」混进成绩）——弃测重测一次；重测仍低于地板 → 本点按
# 测量失败留档（读数置空、all_ok=False），不再按实留档污染聚合
EARLY_STOP_FLOOR_OUT = 100


def _early_stop_threshold(max_tokens: int, ratio: float | None) -> float:
    """early_stop 判据阈值（tokens）：比例口径与有效输出地板取大。
    ratio=None（free 非 agent 场景）只按地板——自由回答的提前收尾多是
    模型自主收尾的合法行为，不按比例判；但输出几个字就停的读数对测速
    完全不可信，两种回复模式同判（2026-09-16 拍板）。地板随预算钳制
    （预算不足 100 时按预算计，否则小预算点全灭）。"""
    floor = min(EARLY_STOP_FLOOR_OUT, max_tokens)
    return max(ratio * max_tokens, floor) if ratio else floor


# 重测仍低于有效输出地板时的失败留档：点读数置空（镜像全失败点形状——
# 垃圾读数不进图表/聚合/KPI），reqs 与弃测留痕保留供复核
_VOID_POINT_KEYS = (
    "prompt_tokens", "ttft_s", "ttft_net_s", "prefill_tok_s",
    "prefill_net_tok_s", "prefill_full_tok_s", "prefill_uncached_tok_s",
    "decode_tok_s", "decode_tok_s_adj", "decode_total_tok_s",
    "decode_total_tok_s_adj", "decode_time_s", "prefill_span_s",
    "total_span_s", "decode_span_s", "prefill_conc_tok_s",
    "decode_conc_tok_s", "decode_peak_tok_s", "out_tokens", "total_s",
    "max_gap_s", "stall_s", "stall_count", "decode_burst", "tok_per_chunk",
    "flush_tok", "flush_count", "flush_s", "finish",
)


def _void_point_readings(pt: dict, msg: str):
    """把点标记为测量失败：读数字段置空 + all_ok=False + err 落档。
    用于「重测仍低于有效输出地板」的早停——此类读数完全不可信，不再
    「按实留档」污染聚合（ADR-0032 的宽容止于地板之上，2026-09-16 拍板）。"""
    for k in _VOID_POINT_KEYS:
        if k in pt:
            pt[k] = None
    pt["all_ok"] = False
    pt["err"] = msg


def _detect_rep_anomaly(reqs: list[dict], max_tokens: int,
                        echo_mode: bool) -> tuple[str, int] | None:
    """一次复测批（point.reqs）的异常输出识别：返回 (异常类型, 肇事 req
    下标) 或 None。只看无 err 的请求——失败的请求已由错误语义化口径处理，
    不是本机制的对象；全部失败则无异常可判。

    early_stop：finish=stop 且 out_tokens 低于阈值（_early_stop_threshold）。
    echo=模板续写口径下模型本应续写到输出预算上限，阈值 = max(0.8×预算,
    有效地板)（实测案例：4K 档某次复测仅输出 4/512 tokens，echo 回复模式下
    被直接平均进聚合行，out_tokens 摊薄、decode 混入读数）；free 模式只按
    有效地板判——自主收尾不按比例罚，但输出几个字就停的读数完全不可信。
    agent 矩阵的 free 形态单步测量用自己的放宽比例（0.5×）走
    _detect_agent_anomaly，不走本函数。

    degenerate（两种回复模式都启用）：text_sample 或 reason_sample 命中
    _is_degenerate_text（思考流退化走 reasoning 通道，头部样本可见）。

    两类同时命中时优先 early_stop：token 数证据是精确的定量口径，不依赖
    采样窗口恰好罩住退化段；弃测留痕本就带 text_sample/reason_sample，
    事后仍可按退化样本复核。"""
    ok = [(i, r) for i, r in enumerate(reqs) if not r.get("err")]
    if not ok:
        return None
    th = _early_stop_threshold(max_tokens, 0.8 if echo_mode else None)
    for i, r in ok:
        ot = r.get("out_tokens")
        if r.get("finish") == "stop" and ot is not None and ot < th:
            return ("early_stop", i)
    for i, r in ok:
        if _is_degenerate_text(r.get("text_sample") or "") \
                or _is_degenerate_text(r.get("reason_sample") or ""):
            return ("degenerate", i)
    return None


def _detect_agent_anomaly(ok: list[dict], max_tokens: int) -> tuple[str, int] | None:
    """agent 矩阵测量批的异常输出识别（ADR-0032 口径的 agent 变体）：ok 为
    无 err 的测量请求列表，返回 (异常类型, ok 内肇事下标) 或 None。

    early_stop（free 口径启用）：finish=stop 且 out_tokens 低于阈值
    max(0.5×预算, 有效地板)（_early_stop_threshold）。
    agent 单步动作天然短且方差大，echo 的 0.8 判据不适配；实测案例是模型
    以「任务上下文不完整/文件内容被截断」为由如实短答（308/310/401 tokens
    即 finish=stop，改双向引导前 out_hint 单向封顶无下限引导）——此类读数混入聚合会把
    decode 拉低，弃测重测。仍宽松于 0.8：真正的提前收尾（任务已完成）输出
    通常不足预算一半，而预算内正常作答多在其上，0.5 恰在两侧之间；有效
    地板兜底小预算档（输出几个字就停完全不可信，2026-09-16 拍板）。

    degenerate：text_sample 或 reason_sample 命中 _is_degenerate_text（实测
    inst=64 档 `<tool_call>\n` 刷屏到输出预算上限的退化形态）。

    两类同时命中时优先 early_stop（与 _detect_rep_anomaly 同口径）。"""
    th = _early_stop_threshold(max_tokens, 0.5)
    for i, r in enumerate(ok):
        ot = r.get("out_tokens")
        if r.get("finish") == "stop" and ot is not None and ot < th:
            return ("early_stop", i)
    for i, r in enumerate(ok):
        if _is_degenerate_text(r.get("text_sample") or "") \
                or _is_degenerate_text(r.get("reason_sample") or ""):
            return ("degenerate", i)
    return None


def _is_implausible_prefill_rate(rate: float | None, x: float,
                                 curve_pts: list[tuple[float, float]]) -> bool:
    """物理速率闸：prefill 速率超该先验曲线 RATE_PRIOR_OVERSHOOT 倍或绝对
    上限即判「物理上不可能」（占位软拒绝/拒答式假成功的通用兜底）。
    x 为未命中 prompt tokens——与 gate_rate 的未命中/全量口径同源（缓存命中
    时 = prompt_tokens − cache_hit，否则全量），也是先验曲线的登记横轴。
    x 低于 RATE_GATE_MIN_TOKENS 时只按绝对上限判：小 prompt 区诚实速率随
    TTFT 固定开销摊薄近线性爬升，10× 先验带没有判别力、只会误杀诚实点
    （实测 agent 零缓存档 inst=1024/4096 的 569~976 tok/s 被误判，同运行
    306~722 tok/s 的兄弟点反而过闸）。无先验时同样只按绝对上限判——首个
    请求没有可比基准，不误伤真实快后端。"""
    if not rate or rate <= 0:
        return False
    if rate >= RATE_ABS_CAP_TOK_S:
        return True
    if x < RATE_GATE_MIN_TOKENS:
        return False
    prior = rate_prior(curve_pts, x)
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


def _is_transient_http_status(status: int) -> bool:
    """瞬时类 HTTP 状态：429（限流）与 5xx（网关/后端抖动）可兜底重试；
    其余 4xx（参数/超窗/鉴权等确定性错误）重试无意义。"""
    return status == 429 or 500 <= status <= 599


def _is_transient_exc(e: Exception) -> bool:
    """httpx 传输异常（连接失败/读写超时/协议错误等）判为瞬时失败；
    其他异常（端点探测耗尽、超窗收口等）不重试。"""
    return isinstance(e, httpx.TransportError)


def _retry_after_s(headers) -> float | None:
    """Retry-After 头解析（秒数形态；HTTP 日期形态罕见，忽略）。封顶
    RETRY_AFTER_CAP_S，缺失/非法/负值返回 None（退避走线性默认）。"""
    try:
        raw = headers.get("retry-after")
    except AttributeError:
        return None
    if not raw:
        return None
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return None
    if s < 0:
        return None
    return min(s, RETRY_AFTER_CAP_S)


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
        # 事件序号实例单调计数器：history 不再收高频 tick（见 emit），若仍用
        # len(history) 会重复/停滞，前端按 seq 去重（ADR-0009）会丢事件
        self._seq = 0
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
        # 真·预填续写参数（echo 末条 assistant 预填时下发
        # add_generation_prompt=false + continue_final_message=true）不被网关
        # 接受时按模型锁定省略（400 且报错点名参数），回退闭合轮次形态
        # （ADR-0079）
        self.prefill_locked: set[str] = set()
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
        # ≥64 才采信）：agent 矩阵 tick 估值补缓存碎块尾用——命中按块对齐，
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
        # seq 取实例单调计数器（非 len(history)：tick 已不留在 history，长度会
        # 重复/停滞）——前端按 seq 去重重连回放与实时帧（ADR-0009）
        ev["seq"] = self._seq
        self._seq += 1
        # 高频 tick 帧不进 history（SSE 回放缓冲）：长矩阵数十万条 tick 会在
        # 服务端 finished-run 2h TTL 内一直驻留内存；tick 只对实时订阅有意义，
        # 结构性事件（cfg/status/point/skipped/done/stopped/error）照常留档，
        # 重连回放仍能重建状态（ADR-0009）
        if ev.get("type") != "tick":
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

    def _externalize_in_texts(self, sidecar_path: str) -> str | None:
        """把超长 in_text 外置到侧车文件（原地改写 self.results）。

        返回顶层告警串（侧车预算耗尽）或 None。**本方法绝不因侧车 IO 失败而
        抛出**：OSError 只记日志、涉事 in_text 保持全文内联——_save 的成败不能
        由一个可选的瘦身特性决定（写不进去顶多存档大回去，不能丢存档）。
        """
        holders: list[dict] = []
        for p in self.results:
            if isinstance(p, dict):
                holders.extend(iter_in_text_holders(p))
        holders = [h for h in holders
                   if isinstance(h.get("in_text"), str)
                   and len(h["in_text"]) > IN_TEXT_INLINE_MAX_CHARS]
        if not holders:
            return None
        warning = None
        exhausted = False
        f = None
        try:
            # 预算按"整档侧车已写字节"算（append 到既有侧车时从现有大小起算）
            written = (os.path.getsize(sidecar_path)
                       if os.path.exists(sidecar_path) else 0)
        except OSError:
            written = 0
        try:
            for h in holders:
                text = h["in_text"]
                data = text.encode("utf-8")
                if exhausted or written + len(data) > ANOMALY_TEXT_BUDGET_BYTES:
                    # 预算耗尽：不再写全文、也不给 in_text_ref（避免存档暗示
                    # "全文已留存"）；截断 + in_text_truncated 标 + 顶层告警，
                    # 剩余条目一律同待遇（全局停写，不是逐条见缝插针）
                    exhausted = True
                    h["in_text"] = _truncate_in_text(text, has_ref=False)
                    h["in_text_truncated"] = True
                    warning = ANOMALY_TEXT_BUDGET_WARNING
                    continue
                if f is None:
                    # 惰性创建：一条都没真正写入时不留下空侧车文件
                    f = open(sidecar_path, "ab")
                off = f.tell()   # 二进制 append 模式的 tell 即文件末尾字节偏移
                f.write(data)
                written += len(data)
                h["in_text_ref"] = {
                    "file": os.path.basename(sidecar_path),
                    "off": off, "len": len(data), "chars": len(text),
                    "sha1": hashlib.sha1(data).hexdigest(),
                }
                h["in_text"] = _truncate_in_text(text, has_ref=True)
        except OSError as e:
            # 失败点之前的条目已写侧车并改 ref（那些切片是完整可用的），失败点
            # 及之后的条目保持原样——全文仍内联，最坏只是存档大回去
            log.warning("异常输入全文外置失败（涉事 in_text 保持内联）：%s", e)
        finally:
            if f is not None:
                try:
                    f.flush()
                    os.fsync(f.fileno())   # 侧车落盘先于存档替换，防掉电断链
                    f.close()
                except OSError:
                    pass
        return warning

    def _save(self):
        if self.mock_seen:   # 模拟数据不进历史记录
            return
        if not self.results_dir:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        cfg_safe = {k: v for k, v in self.cfg.items() if k not in ("api_key",)}
        path = os.path.join(self.results_dir, f"{self.run_id}.json")
        sidecar = os.path.join(self.results_dir,
                               f"{self.run_id}{ANOMALY_TEXT_SUFFIX}")
        # 原子落盘：同目录临时文件 + os.replace。直接 open(path,"w") 写整档，
        # 进程在中途被杀会留下截断 JSON，服务端只能报「corrupted」；os.replace
        # 在同一文件系统内是原子替换，读者要么看到旧档要么看到新档。
        # server.py 有同名 helper，但跨文件所有权不同，此处本地实现（不改 server）
        tmp = f"{path}.tmp"
        try:
            # 先写侧车、再替换存档：读者看到新存档时，其 in_text_ref 指向的
            # 切片必须已存在（顺序反了就会出现悬空引用 → 精确复跑断链）
            warning = self._externalize_in_texts(sidecar)
            archive = {
                "run_id": self.run_id,
                "started_at": self.started_at,
                # 口径版本（顶层 = 本次运行；点级另有自己的版本，见其注释）
                "metric_version": METRIC_VERSION,
                "cfg": cfg_safe,
                "results": self.results,
            }
            if warning:
                # 预算耗尽时不静默：存档顶层留痕，说明有异常轮原文未存全
                archive["anomaly_text_warning"] = warning
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(archive, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except BaseException:
            # 失败清理临时档；异常不吞（调用方发 error 事件），保证路径上
            # 不会残留 .tmp（server 按 *.json 列历史，不会被误读成历史档）
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

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
                    "关键结论建议 repeats≥3 重复取均值"})
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
                            # 张数/字符），无探测/系数校准；模型能力集不含该 kind 是
                            # 直调引擎时的第二道闸（服务端校验已挡常规入口）
                            mks = (cfg.get("model_kinds") or {}).get(model)
                            if mks is None:   # 兼容旧单值 model_kind（直调引擎的调用方）
                                mks = [(cfg.get("model_kind") or {}).get(model, "llm")]
                            if kind not in mks:
                                await self.emit({"type": "status", "msg":
                                    f"跳过 {model} / {sc['label']}："
                                    f"模型类型（kind={'/'.join(str(k) for k in mks)}）"
                                    f"与场景（kind={kind}）不匹配"})
                                continue
                            # 档位取 cfg.<场景>_ladder（测速矩阵可自定义），
                            # 缺省/空数组回退场景默认 ladder
                            for rung in sorted(cfg.get(f"{scenario}_ladder")
                                               or sc["ladder"]):
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
                                                f"　重复轮次 {rep + 1}/{repeats}（取均值）"})
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
                                    _strip_in_full(point)   # 定稿：OCR 路径 reqs 剥输入全文瞬态字段
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
                            # agent 场景不以上下文档位为变量：缓存×指令矩阵逐
                            # 组合出点（前缀缓存口径），详见 _run_agent_matrix
                            for conc in sorted(cfg["concurrencies"]):
                                if self.stop_flag:
                                    await self.emit({"type": "stopped"})
                                    return   # 终态收口在 finally
                                await self._run_agent_matrix(client, model, scenario,
                                                             conc, base_cpt, ctx_limit,
                                                             repeats)
                            continue
                        if sc.get("ladder"):
                            # translate 场景：原文字长阶梯替代 ctx_list——翻译负载
                            # 短输入短输出。阶梯取 cfg.translate_ladder（测速矩阵
                            # 自定义），缺省用场景默认；输出预算随档伸缩
                            # （_translate_max_tokens）且受全局 max_tokens 上界约束；
                            # echo 判据不启用（无 echo_instruction，early_stop 在
                            # free 口径下会误伤译文收尾），退化重复判据照常
                            mt_cap = int(cfg.get("max_tokens") or 0) or None
                            for rung in sorted(cfg.get("translate_ladder")
                                               or sc["ladder"]):
                                if self.stop_flag:
                                    await self.emit({"type": "stopped"})
                                    return   # 终态收口在 finally
                                cpt = (self.cpt_calib.get((model, scenario))
                                       or base_cpt)
                                rung_budget = _translate_max_tokens(rung, cpt,
                                                                    cap=mt_cap)
                                rung_edge = None
                                if max_ctx:
                                    # 超窗保护（与媒体场景同口径）：est 按校准后
                                    # 输入系数折算原文 tokens + 当档输出预算
                                    est_in = round(rung / cpt)
                                    mode, in_eff, excess = _plan_ctx_edge(
                                        est_in, rung_budget, max_ctx)
                                    if mode == "skip":
                                        await self.emit({"type": "status", "msg":
                                            f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                            f"估算 {est_in + rung_budget} tokens 超出部署上限 "
                                            f"{max_ctx // 1024}K（config.json deployments.max_ctx）"})
                                        await self.emit({"type": "point_skipped",
                                            "model": model, "scenario": scenario, "ctx": rung,
                                            "count": len(cfg["concurrencies"]),
                                            "msg": f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                                   f"估算 {est_in + rung_budget} tokens 超出部署上限 "
                                                   f"{max_ctx // 1024}K"})
                                        continue
                                    if mode == "edge":
                                        # 贴边裁减：按 est→rung 线性反推有效原文长度，
                                        # 裁不出有意义的档位（<8 字）仍整档跳过
                                        rung_eff = int(in_eff * cpt)
                                        if rung_eff < 8:
                                            await self.emit({"type": "status", "msg":
                                                f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                                f"估算 {est_in + rung_budget} tokens 超出部署上限 "
                                                f"{max_ctx // 1024}K（config.json deployments.max_ctx）"})
                                            await self.emit({"type": "point_skipped",
                                                "model": model, "scenario": scenario, "ctx": rung,
                                                "count": len(cfg["concurrencies"]),
                                                "msg": f"跳过 {model} / {sc['label']}{rung}{sc['unit']}："
                                                       f"估算 {est_in + rung_budget} tokens 超出部署上限 "
                                                       f"{max_ctx // 1024}K"})
                                            continue
                                        await self.emit({"type": "status", "msg":
                                            f"{model} / {sc['label']}{rung}{sc['unit']}："
                                            f"估算超部署上限 {excess / max_ctx:.0%}，"
                                            f"已裁减为 {rung_eff}{sc['unit']} 贴边测量"})
                                        rung_edge, rung = rung, rung_eff
                                        rung_budget = _translate_max_tokens(
                                            rung, cpt, cap=mt_cap)
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
                                                f"　重复轮次 {rep + 1}/{repeats}（取均值）"})
                                        pt = await self._run_rep_guarded(
                                            client, model, scenario, rung, conc,
                                            cpt, rep, rung_budget, False)
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
                                            f"　重复轮次 {rep + 1}/{repeats}（取均值）"})
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
                # 存档移出事件循环：多 MB json.dump 是同步阻塞，跑在同一循环上
                # 会卡住正在测量的 TTFT/decode tick；to_thread 只把落盘 I/O 挪走
                await asyncio.to_thread(self._save)
            except Exception as e:  # noqa: BLE001
                # 存档自身抛错也不能跳过 SSE 终止帧（原 finally 嵌套口径不变）：
                # 发 error 事件收口，不让异常从 run() 逃逸
                await self.emit({"type": "error", "msg": f"结果落盘失败: {e}"})
            finally:
                self._close_streams()

    # -- 前置探测与网络基线 ----------------------------------------------------

    async def _measure_rtt(self, client: httpx.AsyncClient) -> float | None:
        """GET /models 实测网络往返基线（5 次取最小）。/models 是网关注册表查询，
        开销可忽略，测得的几乎纯是 RTT；云端排队走推理调度器、污染不到这个
        元数据接口，方差只剩网络抖动，min 即 RTT 下限的正确估计量（ADR-0006）。
        顺带完成 /v1 前缀探测。"""
        samples = []
        fails = 0
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
                fails += 1
                continue
            samples.append(time.perf_counter() - t)
        if not samples and fails and not self.stop_flag:
            # 基线不可得：净口径（ttft_net/prefill_net——前端主 prefill 图）整轮
            # 静默失效。此前 except 只 continue，401/DNS/瞬时超时后所有网络指标
            # 悄悄变 null，用户只看到「净口径消失」。计数后明确播报可行动提示
            await self.emit({"type": "status", "msg":
                f"RTT 基线测量失败 {fails} 次（GET /models 不可达或鉴权失败）——"
                "本轮净口径（TTFT/Prefill 净速度）不可用，仅展示毛口径；"
                "请检查网关地址 / API Key 后重测"})
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
                      base_cpt, 64, quiet=True, prime=True))
        if r is None:   # 停止打断探测：静默退出，run 循环随即发 stopped
            return
        await self._prime_done(model, scenario, 2048, 1, 0)
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
        # prefill 等待期估值先验：探测实测的净 prefill 速率播种曲线，x 坐标
        # 取实测未命中量（命中时 prompt_tokens − cache_hit；名义 2048 与真实
        # 值可有数倍差）。探测 prompt 确定——cache=stable 第二跑可能命中缓存，
        # 命中读数必须走未命中口径 + 防污染闸（与点级登记同一 _prefill_prior_sample）
        sample = _prefill_prior_sample(
            2048, r.get("prompt_tokens"), bool(r.get("cache_reported")),
            (r.get("cache_hit") if r.get("cache_reported") else None),
            r.get("prefill_net_tok_s") or r.get("prefill_tok_s"),
            r.get("prefill_uncached_tok_s"),
            self.prefill_curve.get((model, scenario)) or [])
        if sample:
            self._note_prefill_rate((model, scenario), *sample)
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

    async def _prime_done(self, model: str, scenario: str, ctx: int,
                          conc: int, rep: int):
        """预热/基线批终帧：逐并行链发 done tick，前端据此撤掉「预热中」行
        （这类请求不入测点，markLiveDone 只按点收口，覆盖不到它们）。"""
        for i in range(conc):
            await self.emit({"type": "tick",
                             "tag": f"{model}|{scenario}|{ctx}|{conc}|rep{rep}|r{i}",
                             "model": model, "scenario": scenario, "ctx": ctx,
                             "conc": conc, "req": i, "phase": "done", "prime": True})

    async def _await_batch(self, tasks: list) -> tuple[list[dict], float]:
        """等待一批并发请求任务完成，返回 (reqs 按 req 序号排序, 批耗时)。
        在途任务用任务集管理：stop_flag 置位立即 cancel 全部（含 prefill
        等待期，否则长 prefill 无法中断，ADR-0008）；阻塞在 wait 期间无法
        感知 stop_flag，0.2s 轮询一次判停。_run_point 首批/构建偏离校正
        重测批、媒体批、agent 矩阵批共用同一套判停结构。"""
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
                except Exception:
                    # 非取消异常（任务内部 bug）：先取消并 await 其余在途任务再
                    # 上抛——否则矩阵已抛出、兄弟请求仍在流式推进/占着连接。
                    # 不合成 err req：重测批传入的是 pending 子集，没有可靠的
                    # req 序号可填，伪造下标会污染 reqs 契约；意外异常属编程
                    # 错误、不是请求级服务失败，交由 run() 的 except 发 error
                    # 事件统一收口（请求级失败仍在 _one/_one_media 内转成 err）
                    for p in pending:
                        p.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    raise
        reqs.sort(key=lambda r: r["req"])
        return reqs, time.perf_counter() - t0

    async def _run_rep_guarded(self, client: httpx.AsyncClient, model: str,
                               scenario: str, ctx: int, conc: int, cpt: float,
                               rep: int, max_tokens: int, echo_mode: bool) -> dict:
        """带异常输出守卫的单次复测：调 _run_point → 识别 early_stop/退化
        重复 → 命中则弃测该次复测、整批重跑（预算见下）。conc>1
        时整批测量已被肇事请求污染（聚合行均值混入异常读数），弃测按整批
        计。重测换 nonce 破前缀缓存 + 换语料窗口翻 echo 改写区落点
        （stream_salt，2026-09-17 拍板）——nonce 不在 filler 里，只换
        nonce 时改写区落点不变，确定型早停（输出与内容落点绑定的 greedy
        吸引子）会逐字节复现同一答案；换料后填充段内容不同才有机会翻牌。
        重测预算按当前异常形态分两档（2026-09-16 拍板）：低于有效输出地板（EARLY_STOP_FLOOR_OUT）的早停
        完全不可信，最多重测 FLOOR_RETEST_MAX=2 次，仍低于地板按测量失败
        留档（读数置空、all_ok=False）；高于地板的异常最多重测
        ANOMALY_RETEST_MAX=1 次，仍异常则接受结果、按实留档
        pt["anomaly"]——真实模型行为不是可重试故障，不无限重试洗掉。
        仅 LLM 文本场景调用：媒体路径无 echo 续写口径（asr/tts/ocr 无输出
        预算可对齐）；agent 矩阵另有自己的测量批守卫（free 形态早停阈值
        放宽为 0.5×max_tokens，_run_agent_matrix 内 _detect_agent_anomaly
        编排），不走本函数（agent 单步动作合法短停与 free 口径早停在
        echo_mode=False 下本就不经此处判定）。"""
        pt = await self._run_point(client, model, scenario, ctx, conc, cpt, rep)
        if self.stop_flag:   # 停止打断的半截批不可判：不检测直接返回
            _strip_in_full(pt)
            return pt
        discarded = []
        retries = 0
        while True:
            hit = _detect_rep_anomaly(pt["reqs"], max_tokens, echo_mode)
            if hit is None:
                break
            kind, idx = hit
            culprit = pt["reqs"][idx]
            below_floor = (kind == "early_stop"
                           and (culprit.get("out_tokens") or 0)
                               < min(EARLY_STOP_FLOOR_OUT, max_tokens))
            budget = FLOOR_RETEST_MAX if below_floor else ANOMALY_RETEST_MAX
            if retries >= budget or self.stop_flag:
                if self.stop_flag:   # 重测在途被停：半截批不可判，直接返回
                    break
                # 预算耗尽仍异常：按实留档，进聚合不重试——标记取最新一轮
                # 的异常类型（留档数据是最新批；轮次间类型可不同，如早停↔
                # 退化重复，沿用首次类型会误导复核）。肇事 req 同步打
                # anomaly + in_text（输入全文抄本）：聚合优先取该轮 reqs
                # 入档，剥离 _in_full 后全文仍随 req 落档
                culprit["anomaly"] = kind
                culprit["in_text"] = culprit.get("_in_full") or ""
                pt["anomaly"] = kind
                if below_floor:
                    # 重测预算耗尽仍低于有效输出地板：读数完全不可信——
                    # 按测量失败留档（点 err、读数置空不进聚合，
                    # 2026-09-16 拍板），ADR-0032 的「按实留档」宽容
                    # 止于地板之上
                    culprit["err"] = (
                        f"输出异常早停（仅 {culprit.get('out_tokens')}"
                        f"/{max_tokens} tokens，弃测重测 {retries} 次仍"
                        "复现）——读数不可信，按测量失败留档")
                    _void_point_readings(pt, culprit["err"])
                    await self.emit({"type": "status", "toast": True, "msg":
                        f"{model} / {SCENARIOS[scenario]['label']} / "
                        f"上下文 {_fmt_scenario_ctx(scenario, ctx)}："
                        f"重测 {retries} 次仍仅输出 "
                        f"{culprit.get('out_tokens')} tokens（低于有效地板 "
                        f"{min(EARLY_STOP_FLOOR_OUT, max_tokens)}），"
                        "本点按测量失败留档"})
                break
            # 弃测留痕：异常类型、肇事请求下标及其关键读数（含输出头部样本，
            # 供事后人工复核退化形态；随聚合行 reps_discarded 逐条列出）。
            # in_text = 肇事请求实际发送的输入全文（瞬态 _in_full 抄本，用户
            # 要求异常日志列出输入原文全文），rep = 被弃测轮次的 1-based 轮次号
            discarded.append({
                "anomaly": kind, "req": idx, "rep": rep + 1,
                "out_tokens": culprit.get("out_tokens"),
                "finish": culprit.get("finish"),
                "ttft_s": culprit.get("ttft_s"),
                "decode_tok_s": culprit.get("decode_tok_s"),
                "text_sample": culprit.get("text_sample") or "",
                "reason_sample": culprit.get("reason_sample") or "",
                "in_sample": culprit.get("in_sample") or "",
                "in_text": culprit.get("_in_full") or "",
            })
            await self.emit({"type": "status", "toast": True, "msg":
                (f"检测到异常输出（提前停止：仅输出 {culprit.get('out_tokens')}"
                 f"/{max_tokens} tokens），本轮弃测重测" if kind == "early_stop"
                 else "检测到异常输出（退化重复），本轮弃测重测")
                + f"（第 {retries + 1}/{budget} 次，已换语料落点）"})
            pt = await self._run_point(client, model, scenario, ctx, conc,
                                       cpt, rep, stream_salt=retries + 1)
            retries += 1
            if self.stop_flag:   # 重测在途被停：半截批同样不可判，弃循环
                break
        if discarded:
            pt["reps_discarded"] = discarded   # 无弃测则不设此键
        _strip_in_full(pt)   # 定稿前剥离输入全文瞬态字段（in_text 已抄本）
        return pt

    async def _run_point(self, client: httpx.AsyncClient, model: str, scenario: str,
                         ctx: int, conc: int, cpt: float, rep: int = 0,
                         media_msgs: tuple[list, list[int]] | None = None,
                         stream_salt: int = 0) -> dict:
        """单个测试点（conc 个并发请求一批）。media_msgs 非空为 OCR 媒体路径：
        (msgs_list, 每请求图片 token 估值)——跳过文本构造/校准，计时走 _one 的
        流式路径（VLM 输出天然流式，ADR-0020）。文本路径批完成后做构建偏离
        判定：实测 prompt_tokens 与目标档位偏差 > CTX_DEV_TOLERANCE 时按实测
        密度校正填充字符量整批重测一次（bust 模式，最多 1 次，见重测块注释）。
        conc>1 时偏离判定提前化：全批同料同偏离，首个带真 usage 的请求完成
        即判，超差取消在途余量直接重测，不等全批跑完（2026-09-16 拍板）。
        stream_salt>0 为弃测重测换料口径（2026-09-17 拍板）：填充窗口循环
        平移到语料流另一处翻 echo 改写区落点（确定型早停与落点绑定，同料
        重测逐字节复现同一答案），仅 _run_rep_guarded 的重测调用传入。"""
        base_seed = random.randint(100000, 999999)
        # 换料偏移由 base_seed 派生（与同批 nonce 同源、随重测重掷）；首测
        # salt=0 恒 offset=0，嵌套前缀确定性不受影响。偏移窗口仍是真实的
        # 「字符数→tokens」密度样本，校准照常入（密度位置相关性误差由后续
        # 档位的偏离校正吸收）
        stream_offset = 0
        if stream_salt:
            _stream = SCENARIOS[scenario].get("filler_stream") or ""
            if _stream:
                stream_offset = random.Random(base_seed).randrange(len(_stream))
        cache_mode = str(self.cfg.get("cache") or "bust").lower()
        # 回复模式（ADR-0021）：echo=模板续写——创意/代码场景换用改写指令并
        # 追加 assistant 预填，输出大段复用上下文，decode 对齐真实编辑场景的
        # 投机采样命中率；free（默认）=自由回答纯新生成。agent 无
        # echo_instruction 自动回退普通构造，媒体场景不走文本构造
        echo_mode = str(self.cfg.get("reply_mode") or "free").lower() == "echo"
        sc_kind = SCENARIOS[scenario].get("kind", "llm")
        # 翻译类固定字长阶梯场景（llm kind 但不以 ctx_list 为变量）：原文按字
        # 截取，不走 token 目标构造/输入系数校准/构建偏离重测；输出预算随档伸缩
        tl_mode = sc_kind == "llm" and SCENARIOS[scenario].get("ladder") is not None
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
        elif tl_mode:
            for i in range(conc):
                # bust/stable nonce 口径与文本路径一致（编号行在原文开头）
                nonce = "stable" if cache_mode == "stable" else f"{base_seed + i * 137}"
                msgs_list.append(_build_translate_messages(ctx, nonce))
        else:
            for i in range(conc):
                # bust=每请求随机编号破坏 prefix cache（冷测真实 prefill）；
                # stable=固定编号，重复/并发请求命中官方上下文缓存以省成本
                nonce = "stable" if cache_mode == "stable" else f"{base_seed + i * 137}"
                msgs, _est, filler_n = build_messages(scenario, ctx, cpt, nonce,
                                                      filler_chars=filler_chars,
                                                      echo=echo_mode,
                                                      stream_offset=stream_offset)
                msgs_list.append(msgs)

        # 翻译场景输出预算随原文档位伸缩（译文与原文等比），同时受全局
        # max_tokens 上界约束（与通用场景同语义：超界按 length 截断）
        max_tokens = (_translate_max_tokens(
                          ctx, cpt,
                          cap=int(self.cfg.get("max_tokens") or 0) or None)
                      if tl_mode else
                      int(self.cfg.get("max_tokens")
                          or SCENARIOS[scenario]["max_tokens_default"]))
        # 在途请求任务集管理与 0.2s 轮询判停（_await_batch，与重测批/媒体批/
        # agent 矩阵批共用）
        tasks = [asyncio.create_task(
                    self._one(client, model, msgs, scenario, i, ctx, conc, cpt,
                              max_tokens, rep=rep, img_tokens=img_tokens_list[i]))
                 for i, msgs in enumerate(msgs_list)]
        # 构建偏离早判（仅文本 bust 路径 ctx>0 且 conc>1，2026-09-16 拍板）：
        # 偏离是构造密度问题、全批同料同偏离，首个带真 usage 的 ok 请求完成
        # 即可判定——超差则取消在途余量、直接进校正重测，不再等全批
        # prefill+decode 跑完再全批重测（大上下文多并发实测：等全批 ≈ 白跑
        # 一整批）。无偏离/无可用样本/被停时与旧路径完全一致（等全批）
        if (media_msgs is None and not tl_mode and ctx > 0
                and cache_mode == "bust" and conc > 1):
            t0 = time.perf_counter()
            reqs = []
            pending = set(tasks)
            sample = None
            msgs_chars0 = sum(len(m["content"]) for m in msgs_list[0])
            while pending and sample is None and not self.stop_flag:
                done, pending = await asyncio.wait(
                    pending, timeout=0.2, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    try:
                        r = t.result()
                    except asyncio.CancelledError:
                        continue
                    reqs.append(r)
                    if (not r.get("err") and r.get("usage_real")
                            and not r.get("sent_chars", 0) < msgs_chars0):
                        sample = r
            dev_hit = (sample is not None
                       and abs((sample["prompt_tokens"] - ctx) / ctx)
                           > CTX_DEV_TOLERANCE)
            # 校正数学预审（与下方校正块同公式）：偏离坐实且可解才取消在途
            # 余量——_correct_filler_chars 退化情形返回 None 时不取消，否则
            # 白牺牲一批健康请求
            f_early = (_correct_filler_chars(
                filler_n, sample["prompt_tokens"], ctx,
                self.prefix_kb.get(key), msgs_chars0 - filler_n, cpt)
                if dev_hit else None)
            if f_early is not None:
                # 在途余量取消（其读数随整批废弃），提前进校正重测
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                early_dev = True
            else:
                more = (await self._await_batch(list(pending)))[0] \
                    if pending else []
                reqs.extend(more)
                early_dev = False
            reqs.sort(key=lambda r: r["req"])
            batch_time = time.perf_counter() - t0
        else:
            early_dev = False
            reqs, batch_time = await self._await_batch(tasks)
        ok = [r for r in reqs if not r.get("err")]

        # 输入系数校准前置到重测判定之前：失败的那一次测量也是真实密度样本，
        # 无论是否触发校正重测都要入校准（后续档位的嵌套估算会因本次样本
        # 立即变准，ADR-0005）。翻译场景构造按字截取、不依赖系数估算，不入校准
        if media_msgs is None and not tl_mode:
            kb_before = self.prefix_kb.get(key)
            msgs_chars = sum(len(m["content"]) for m in msgs_list[0])
            await self._calibrate_usage(key, msgs_chars, filler_n, ok)
        else:
            kb_before = None
            msgs_chars = 0

        # 上下文构建偏离校正重测（ADR-0005 嵌套记账盲区：新语料段密度未知，
        # 滑动边际估算只能外推）。仅文本路径——agent 矩阵有自己的构造、
        # OCR 媒体路径不经文本构造，均不在范围。触发条件：实测 prompt_tokens
        # 均值与 ctx 相对偏差 > CTX_DEV_TOLERANCE + bust 模式（stable 重测会
        # 命中前缀缓存污染读数）+ 未重测过（最多 1 次）+ 未停止。
        # 删减重试成功点位（sent_chars < 构造量）的偏离源于服务端超窗保护，
        # 重测必然复现，不触发。翻译场景 ctx 是字符口径（与 prompt_tokens 不同
        # 单位），不做偏离判定
        ctx_retry = None
        if (media_msgs is None and not tl_mode and ctx > 0 and cache_mode == "bust"
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
                        await self.emit({"type": "status", "toast": True, "msg":
                            f"{model} / {SCENARIOS[scenario]['label']} "
                            f"{ctx // 1024}K 档构建偏离 {dev * 100:+.1f}%"
                            "（实测/目标），"
                            + ("并发批在途余量已取消，" if early_dev else "")
                            + "已按实测密度校正重测"})
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
                                echo=echo_mode, stream_offset=stream_offset)
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
        # 非上下文标度，不入曲线）。媒体场景（图像 token 标度未接入曲线）与
        # 翻译场景（字符口径，与曲线的 token 横轴不同单位）不登记。
        # 登记口径与 agent 矩阵同源（_prefill_prior_sample）：x 取未命中量、
        # 命中回传时用 req 级未命中速率、登记前过 _prior_suspect 防污染闸——
        # 否则 cache=stable 的重复运行里高命中档会把「全量口径 ÷ ctx」的虚高
        # 速率写进曲线，抬高 10× 物理闸与 live est_tick 估值整轮失真
        if media_msgs is None and not tl_mode and conc == 1 and ctx > 0:
            for r in ok:
                sample = _prefill_prior_sample(
                    ctx, r.get("prompt_tokens"),
                    bool(r.get("cache_reported")),
                    (r.get("cache_hit") if r.get("cache_reported") else None),
                    r.get("prefill_net_tok_s") or r.get("prefill_tok_s"),
                    r.get("prefill_uncached_tok_s"),
                    self.prefill_curve.get((model, scenario)) or [])
                if sample:
                    self._note_prefill_rate((model, scenario), *sample)
        point = {
            "model": model, "scenario": scenario,
            "kind": sc_kind,   # 场景类型：llm/asr/ocr/tts，前端按此分叉（ADR-0020）
            # 口径版本：点级也写（拼接/复测会把不同时期的点混进同一存档，
            # 顶层字段只代表本次运行的版本；见 METRIC_VERSION 注释）
            "metric_version": METRIC_VERSION,
            "ctx_target": ctx, "concurrency": conc,
            "reqs": reqs, "all_ok": len(ok) == conc, "batch_time_s": round(batch_time, 3),
            "prompt_tokens": None, "ttft_s": None, "prefill_tok_s": None,
            "decode_tok_s": None, "decode_total_tok_s": None, "out_tokens": None,
            # 点级 decode Σ÷Σ 口径（_decode_sums）：批级 decode 总耗时
            "decode_time_s": None,
            "ttft_net_s": None, "prefill_net_tok_s": None,
            "rtt_ms": round(self.rtt_s * 1000) if self.rtt_s is not None else None,
            "max_gap_s": None, "stall_s": None, "stall_count": None,
            "decode_burst": None, "tok_per_chunk": None,
            "decode_total_tok_s_adj": None,
            # 批级并发口径（_conc_batch_stats）：prefill/total/decode span、
            # 并发均 prefill/decode 速度、并发 decode 峰值
            "prefill_span_s": None, "total_span_s": None, "decode_span_s": None,
            "prefill_conc_tok_s": None, "decode_conc_tok_s": None,
            "decode_peak_tok_s": None,
            "flush_tok": None, "flush_count": None, "flush_s": None,
            "ctx_retry": ctx_retry,   # 构建偏离校正重测的首次实测 prompt_tokens 留痕
        }
        if len(ok) != conc:   # 失败原因落档（repeats>1 时聚合层按失败轮拼接覆写）
            point["err"] = _first_err(reqs)
        if ok:
            # 批级并发口径先算好（守卫见 _conc_batch_stats）：批级 prefill
            # 点口径与 6 指标产出都消费
            stats = _conc_batch_stats(ok, conc)
            if conc == 1:
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
                flushes = [r for r in ok if r.get("flush_tok")]
                if flushes:
                    # 停滞回吐剔除量取请求中最差（与 stall_s 同口径）：剔除 tokens
                    # 与次数同取自回吐最重的那个请求
                    worst = max(flushes, key=lambda r: r["flush_tok"])
                    point["flush_tok"] = worst["flush_tok"]
                    point["flush_count"] = worst.get("flush_count")
                    point["flush_s"] = worst.get("flush_s")   # 同源同请求
            bursts = [bool(r.get("decode_burst")) for r in ok if r.get("decode_tok_s")]
            if bursts and sum(bursts) * 2 > len(bursts):
                point["decode_burst"] = True   # 多数决（与 all_ok 同口径）
            point["prompt_tokens"] = round(sum(r["prompt_tokens"] for r in ok) / len(ok))
            # TTFT/prefill 批级口径优先（ADR-0074）：TTFT = 批开始→全部首
            # token（prefill_span_s）、速度 = Σprompt_tokens ÷ span，conc=1
            # 与单请求口径严格相等；批级不可用（守卫未过/分母 ≤0，如部分
            # 失败批）时回退逐请求均值口径
            bp = _batch_prefill_point(sum(r["prompt_tokens"] for r in ok),
                                      stats, self.rtt_s)
            if bp:
                point.update(bp)
            else:
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
                pns = [r["prefill_net_tok_s"] for r in ok
                       if r.get("prefill_net_tok_s")]
                if pns:
                    point["prefill_net_tok_s"] = round(sum(pns) / len(pns), 1)
            # 点级 decode Σ÷Σ 口径（ADR-0074 追加）：总耗时 = Σ各请求
            # decode 窗口、均速 = Σout_tokens ÷ Σ窗口；conc=1 与逐请求
            # 口径严格相等（守卫见 _decode_sums）
            dt, dc = _decode_sums(ok)
            if dt is not None:
                point["decode_time_s"] = dt
            if dc is not None:
                point["decode_tok_s"] = dc
            dca = [r["decode_tok_s_adj"] for r in ok if r.get("decode_tok_s_adj")]
            if dca:
                point["decode_tok_s_adj"] = round(sum(dca) / len(dca), 1)
            # 平均每事件 token 数：投机解码网关（MTP/EAGLE 类）单事件携带 k 个
            # token 的诊断读数，各请求取均值
            tpc = [r["tok_per_chunk"] for r in ok if r.get("tok_per_chunk")]
            if tpc:
                point["tok_per_chunk"] = round(sum(tpc) / len(tpc), 2)
            if len(ok) == conc:
                # raw + adj 总吞吐（_total_decode_rates，与 agent 矩阵同公式）。
                # 该定义仅在并发请求的 decode 区间充分重叠时有意义：长上下文
                # prefill 被服务端串行化时 decode 实际先后发生，窗口混入
                # prefill 等待期 → 断崖假象（ADR-0006），此时置空
                raw_total, adj_total = _total_decode_rates(ok, conc)
                if raw_total is not None:
                    point["decode_total_tok_s"] = raw_total
                if adj_total is not None:
                    point["decode_total_tok_s_adj"] = adj_total
            # 批级并发口径 6 指标（全成功批产出，守卫见 _conc_batch_stats）
            point.update(stats)
            point["cache_hit_tokens"] = sum(r.get("cache_hit") or 0 for r in ok)
            fins = [r.get("finish") for r in ok if r.get("finish")]
            if fins:
                uniq = sorted(set(fins))
                point["finish"] = uniq[0] if len(uniq) == 1 else ",".join(uniq)
            point["out_tokens"] = round(sum(r.get("out_tokens") or 0 for r in ok) / len(ok))
        for r in reqs:   # 剥离 token 交付时间轴瞬态字段（两条 return 路径都不带出）
            r.pop("_tl", None)
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
        # 首批/重测批、agent 矩阵批共用）
        tasks = [asyncio.create_task(
                    self._one_media(client, model, scenario, kind, rung,
                                    i, conc, rep))
                 for i in range(conc)]
        reqs, batch_time = await self._await_batch(tasks)
        ok = [r for r in reqs if not r.get("err")]
        sc = SCENARIOS[scenario]
        point = {
            "model": model, "scenario": scenario, "kind": kind,
            "metric_version": METRIC_VERSION,   # 口径版本（见 METRIC_VERSION 注释）
            "ctx_target": rung, "concurrency": conc,
            "reqs": reqs, "all_ok": len(ok) == conc,
            "batch_time_s": round(batch_time, 3),
            "rtt_ms": round(self.rtt_s * 1000) if self.rtt_s is not None else None,
        }
        if len(ok) != conc:   # 失败原因落档（媒体路径同口径）
            point["err"] = _first_err(reqs)
        point.update(_aggregate_media_reqs(kind, reqs, batch_time, conc))
        return point

    async def _one_media(self, client: httpx.AsyncClient, model: str,
                         scenario: str, kind: str, rung, req_i: int, conc: int,
                         rep: int, _retried: int = 0) -> dict:
        """asr/tts 单请求：非 chat 端点、无 SSE 文本流。asr=multipart 上传整段
        音频等非流式 JSON；tts=audio/speech 二进制流（首音频字节计时）。RTF 与
        倍速按净耗时（扣 RTT 基线，与 LLM 同口径），错误处理与 _one 对齐。"""
        tag = f"{model}|{scenario}|{rung}|{conc}|rep{rep}|r{req_i}"
        sc = SCENARIOS[scenario]
        await self.emit({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                         "ctx": rung, "conc": conc, "req": req_i, "phase": "prefill",
                         "tokens": 0, "speed": 0, "elapsed": 0})

        async def _fail(msg: str) -> dict:
            await self.emit({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": rung, "conc": conc,
                             "req": req_i, "phase": "error", "msg": msg})
            # 与 _one.emit_tick 同契约（ADR-0052）：任何失败都弹悬浮岛提醒；
            # stop_flag 置位（用户主动停止）时不弹——中断是用户动作本身
            if not self.stop_flag:
                await self.emit({"type": "status", "toast": True,
                                 "msg": f"{model} / {sc['label']} 请求 r{req_i} 失败：{msg[:120]}"})
            return {"req": req_i, "err": msg}

        # 瞬时失败兜底重试（与 _one 同口径）：429/5xx 与传输异常退避后整体
        # 重来；返回 None = 额度耗尽或 stop_flag 已置位，交 _fail 按既有
        # 错误口径收口
        async def _retry(msg: str, retry_after: float | None) -> dict | None:
            if self.stop_flag or _retried >= TRANSIENT_RETRY_MAX:
                return None
            wait = retry_after or (_retried + 1) * TRANSIENT_RETRY_BASE_S
            await self.emit({"type": "status", "msg":
                f"{model} / {kind} 请求 r{req_i} 失败（{msg[:60]}），"
                f"{wait:g}s 后兜底重试 {_retried + 1}/{TRANSIENT_RETRY_MAX}",
                "toast": True})
            await asyncio.sleep(wait)
            return await self._one_media(client, model, scenario, kind, rung,
                                         req_i, conc, rep, _retried=_retried + 1)

        candidates = ([self.api_prefix] if self.prefix_locked
                      else ["/v1", ""])
        try:
            if kind == "asr":
                # 音频构造（语料读取/合成 + 按目标时长平铺可达数十 MB，实测
                # 缓存未命中 ~105ms）与 multipart 都在计时窗外完成：t0 推迟到
                # 请求发出前，否则首请求/复测率先背一段与模型无关的本地开销、
                # TTFT/RTF 被系统性抬高。构造失败仍落下方 except → _fail 收口
                wav, audio_s = _asr_audio(rung)
                t0 = time.perf_counter()
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
                    msg = _classify_http_error(resp.status_code, resp.text[:300])
                    if _is_transient_http_status(resp.status_code):
                        retried = await _retry(msg, _retry_after_s(resp.headers))
                        if retried is not None:
                            return retried
                    return await _fail(msg)
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
                result = {"req": req_i, "err": None, "first_abs": t0,
                          "last_abs": t_end,
                          "audio_s": audio_s, "total_s": round(elapsed, 2),
                          "elapsed_net_s": round(net, 2),
                          "rtf": round(rtf, 4), "speed_x": round(speed, 1),
                          "out_chars": len(text)}
                if _retried:   # 存档留痕：兜底重试后成功（前端忽略未知字段）
                    result["retried"] = _retried
                return result

            # kind == "tts"：audio/speech 二进制流，response_format=wav 以解析时长
            text = _tts_text(rung, nonce=f"{rung}-{conc}-{req_i}")
            payload = {"model": model, "input": text, "response_format": "wav",
                       "stream": True}
            resp_ctx = None
            resp = None
            # 计时起点：_tts_text 构造在窗外，起点落在请求发出前（与 asr 同口径）
            t0 = time.perf_counter()
            # resp_ctx 无条件收口（与 _one 同款）：此前只有流式成功路径的
            # finally 会 __aexit__，非 200 返回与通用 except 路径把上下文留给
            # GC。httpx 0.28 读尽正文后恰好释放连接，但释放不应依赖实现细节
            try:
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
                    msg = _classify_http_error(resp.status_code, body)
                    if _is_transient_http_status(resp.status_code):
                        retried = await _retry(msg, _retry_after_s(resp.headers))
                        if retried is not None:
                            return retried
                    return await _fail(msg)
                buf = bytearray()
                ttfa = None
                last_emit = 0.0
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
                result = {"req": req_i, "err": None, "first_abs": ttfa or t0,
                          "last_abs": t_end, "audio_s": round(audio_s, 2),
                          "audio_est": est, "total_s": round(elapsed, 2),
                          "elapsed_net_s": round(net, 2),
                          "ttfa_s": round(ttfa - t0, 3) if ttfa else None,
                          "rtf": round(rtf, 4), "speed_x": round(speed, 1),
                          "out_bytes": len(buf)}
                if _retried:   # 存档留痕：兜底重试后成功（前端忽略未知字段）
                    result["retried"] = _retried
                return result
            finally:
                if resp_ctx is not None:
                    await resp_ctx.__aexit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:300]
            if _is_transient_exc(e):   # httpx 传输异常：兜底重试后再认败
                retried = await _retry(msg, None)
                if retried is not None:
                    return retried
            return await _fail(msg)

    async def _calibrate_usage(self, key: tuple[str, str], msgs_chars: int,
                               filler_n: int, ok: list[dict]):
        """输入系数实测校准 + 嵌套前缀记账：用服务端真实 prompt_tokens 反推
        chars/token。llama.cpp 等分词与场景先验（in_cpt：创意 1.5 / 代码 3.9）
        仍可能有差，校准后后续上下文点按实测值构造（ADR-0005）。按字符量加权
        滑动平均（权重封顶，保留对语料流下游段落配比变化的适应性）；小样本
        （<2000 字符，如 0K 档纯指令）分词模板开销占比大、比例严重失真，
        采纳会带偏校准——实测 0K 档曾把 4K 点位构造带偏 20%+（ADR-0005）。
        _run_point 每点调用一次；agent 矩阵每组合调用一次。"""
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
            # 漂移播报的零值守卫：old 为 0/None（未来若有种子路径写入 0）时
            # 除法会 ZeroDivisionError——测量路径不允许因此崩掉
            if not old or abs(new - old) / old > 0.02:
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

    # -- Agent 缓存×指令矩阵 ----------------------------------------------------

    async def _run_agent_matrix(self, client: httpx.AsyncClient, model: str,
                                scenario: str, conc: int, base_cpt: float,
                                ctx_limit: int | None, repeats: int):
        """Agent 缓存×指令矩阵（ADR-0042）：逐「已缓存上下文 × 单步指令长度」
        组合产出测速点。

        口径设计（对齐真实 agent 循环——历史已在前缀缓存里，单步新输入很短）：
        - 阶梯取 cfg.agent_cache_ladder / agent_inst_ladder（测速矩阵自定义），
          缺省用场景默认（缓存 0/4K…256K × 指令 64…8K tokens）。
        - 每个缓存档 C>0：先发 conc 条 quiet 预热请求（同上下文 + 超短任务、
          输出预算 AGENT_PRIME_MAX_TOKENS、内容丢弃）把上下文写入服务端前缀
          缓存——预热不计入测点；随后逐指令档 I 发测量请求，上下文段与预热
          逐字节一致，命中前缀缓存、只增量 prefill 指令段。C=0 为零缓存档：
          纯指令基线（缓存收益的对照），逐点换 nonce 破缓存（bust 口径）。
        - 每组合聚合为一个点：ctx_target = 缓存档 C（排序/图表类别轴用），
          inst_tokens = 指令档 I；TTFT 与净口径批级优先（批开始→全部首
          token，_batch_prefill_point，conc=1 与单请求口径严格相等，批级
          不可用时回退逐请求均值）；prefill_tok_s 为**增量口径** = 未命中
          tokens ÷ 批级 prefill span（批级不可用回退逐请求均值），全量
          口径 prefill_full_tok_s = Σprompt_tokens ÷ 同一 span——服务端
          回传 cache_hit 按真值；网关剥离字段（LiteLLM 等中转）时按
          「TTFT 相对同指令零缓存档走平」判别缓存迹象，有迹象按预热实测
          prompt 估算命中（cache_reported=False，前端加 ≈），无迹象命中
          None（前端「未回传」）、prefill 退回全量口径实测真值。
          total_s = 请求端到端总时长（TTFT + decode 全程，跨链均值）。
        - 链内 nonce 固定（同档预热/测量请求同前缀）、跨 rep/链/运行随机
          （残留缓存不污染下一 rep 的预热）。conc = 并行会话链数：跨链
          decode 重叠口径不保证，decode_total_tok_s 仅 conc==1 产出；批级
          并发口径（_conc_batch_stats 的 prefill/total/decode span、并发均
          速度与 decode 峰值）不受重叠守卫约束，全并发档产出。conc>1 的
          并发批受服务端资源挤占：停滞/回吐诊断（max_gap_s/stall_s/
          flush_*）与 decode 净速（decode_tok_s_adj）无判读意义，req 级
          恒 None 不产出（ADR-0074）。
        - C>0 缓存档内逐测量尝试错位取段：每次尝试（首测/校正补测/污染
          重测/异常重测）独占一段语料流（seg_salt 取字符游标，用后前进
          seg_n + seg_n//2 + 1024）——任意两次尝试的指令材料不重叠、不
          互为前缀，否则取段偏移只随 ctx 定，同缓存档升序测量时上一指令
          档残留被服务端算进 cache_hit、增量口径分子失真。C=0 零缓存档
          逐点换 nonce 破缓存（nonce 在上下文头部、缓存从头就断），无此
          污染，保持原样。
        - 输出异常弃测重测（ADR-0032 口径接入 agent 矩阵）：测量批 ok 请求
          命中早停（free 口径阈值 0.5×max_tokens，_detect_agent_anomaly）
          或退化重复 → 整批弃测、换新 nonce/指令材料重测一次
          （reps_discarded 留痕）；重测仍异常 point["anomaly"] 按实留档。
          指令段缓存残留污染兜底重测（_cache_pollution_verdict）：C>0
          回传命中批实测未命中 tokens 远低于指令档期望 → 整批弃测、游标
          换新段重测一次（reps_discarded 标 "cache_pollution" 并带实测
          uncached/expected）；重测仍偏差过大按实留档。与指令长度偏离
          校正的次序：先输入校正、再污染校验、后输出异常判定，各自最多
          1 次、总尝试数有界（首测 + 校正补测 + 污染重测 + 异常重测
          ≤ 5 次：异常重测按形态取 FLOOR_RETEST_MAX=2 或
          ANOMALY_RETEST_MAX=1，故上界 1+1+1+2）。
          预热/基线 quiet 请求不判定（非测点）。
        - C + I 连同输出预算超出部署上限（ctx_limit）的组合整档跳过
          （point_skipped 计入进度）；某组合出现软/硬超窗错误（req 级
          ctx_ovf）时矩阵提前收口——更大组合必然同样超窗：失败组合照常
          出点（all_ok=False），剩余组合 point_skipped（ADR-0018 同口径）。
        """
        sc = SCENARIOS[scenario]
        cache_ladder = sorted({max(0, int(v)) for v in
                               (self.cfg.get("agent_cache_ladder")
                                or sc["cache_ladder"])})
        inst_ladder = sorted({max(1, int(v)) for v in
                              (self.cfg.get("agent_inst_ladder")
                               or sc["inst_ladder"])})
        max_tokens = int(self.cfg.get("max_tokens")
                         or sc["max_tokens_default"])
        key = (model, scenario)
        cpt = self.cpt_calib.get(key) or base_cpt
        n_inst = len(inst_ladder)
        fmt = lambda v: str(v) if v < 1024 else _fmt_ctx(v)   # 指令小档不折 K
        combos = [(c, i) for c in cache_ladder for i in inst_ladder]
        # 预算跳档（确定性，全 rep 一致）：C + I 超 ctx_limit 的组合不构造
        skipped = {(c, i) for c, i in combos
                   if ctx_limit is not None and c + i > ctx_limit}
        if skipped:
            by_c: dict[int, list[int]] = {}
            for c, i in sorted(skipped):
                by_c.setdefault(c, []).append(i)
            for c, ils in by_c.items():
                msg = (f"{model} / Agent 调用 / 缓存 {fmt(c)}：指令档 "
                       + "/".join(fmt(i) for i in ils)
                       + " 连同输出预算超出部署上限，整档跳过"
                         "（config.json deployments.max_ctx）")
                await self.emit({"type": "status", "msg": msg})
                await self.emit({"type": "point_skipped", "model": model,
                                 "scenario": scenario, "ctx": c,
                                 "count": len(ils), "msg": msg})
        todo = [cb for cb in combos if cb not in skipped]
        await self.emit({"type": "status", "msg":
            f"{model} / Agent 调用矩阵 / 并行会话 {conc}：缓存 "
            f"{len(cache_ladder)} 档（{'/'.join(fmt(c) for c in cache_ladder)}）× "
            f"指令 {n_inst} 档（{'/'.join(fmt(i) for i in inst_ladder)}）＝ "
            f"{len(combos)} 点——每缓存档先预热写入前缀缓存（不计入测点），"
            "再逐指令档实测 TTFT / Decode / 总时长"})

        rep_points: dict[tuple[int, int], list[dict]] = {cb: [] for cb in todo}

        async def _flush_rep_points():
            """已测点落盘收口：聚合各组合跨 rep 的点 → 存档 → 发事件。矩阵
            提前收口（超窗错误 / 用户停止）时同样执行，否则已完成组合的点
            会被静默丢弃。repeats==1 的点在组合内即时落盘，此处不重复发出。"""
            if repeats == 1:
                return
            for pts in rep_points.values():
                if not pts:
                    continue
                point = pts[0] if len(pts) == 1 else _aggregate_reps(pts)
                self.results.append(point)
                await self.emit({"type": "point", "point": point})

        async def _abort(msg: str):
            """矩阵提前收口（超窗）：剩余组合 point_skipped 计入进度 + 已测点
            落盘（与旧 agent 链终止同口径）。"""
            left = [cb for cb in todo if not rep_points[cb]]
            if left:
                await self.emit({"type": "point_skipped", "model": model,
                                 "scenario": scenario, "ctx": left[0][0],
                                 "count": len(left), "msg": msg})
            await _flush_rep_points()

        for rep in range(repeats):
            if repeats > 1:
                await self.emit({"type": "status", "msg":
                    f"　重复轮次 {rep + 1}/{repeats}（取均值）"})
            base_seed = random.randint(100000, 999999)
            nonces = [f"{base_seed}-c{i}" for i in range(conc)]
            # 零缓存档实测 TTFT（净口径优先）：非回传网关的缓存迹象判别基准，
            # 同 rep 内 0 档先测（阶梯升序 0 恒在最前）
            base_ttft: dict[int, float] = {}
            # 运行级缓存佐证：本轮任一 C>0 点出现「回传命中>0」或「估算命中」
            # 即置 True，之后未回传点直接估算（与 base_ttft 同生命周期）
            cache_proven = False
            for c in cache_ladder:
                rungs = [i for i in inst_ladder if (c, i) not in skipped]
                if not rungs:
                    continue
                if self.stop_flag:
                    await _flush_rep_points()
                    return
                # 上下文构造量：嵌套前缀记账精确加长（与 _run_point 同公式）；
                # 同档预热/测量共用同一 cache_chars——前缀逐字节一致
                cache_chars = None
                if c > 0:
                    kb = self.prefix_kb.get(key)
                    if kb:
                        marg = self.cpt_marginal.get(key) or cpt
                        cache_chars = max(0, round(kb[0] + (c - kb[1]) * marg))
                prime_tokens = [0] * conc   # 各链预热实测 prompt（命中估算用）
                # 指令材料错位取段游标（仅 C>0 用）：本缓存档内每次测量尝试
                # 独占一段语料流，推进规则见 _attempt_track
                salt_cursor = 0
                if c > 0:
                    # 预热：每条并行会话链一条 quiet 请求写入前缀缓存
                    tasks = []
                    for i in range(conc):
                        pmsgs, _e, _cn, _sn = build_agent_messages(
                            c, 0, cpt, nonces[i], cache_chars=cache_chars,
                            prime=True)
                        tasks.append(asyncio.create_task(self._one(
                            client, model, pmsgs, scenario, i, c, conc, cpt,
                            AGENT_PRIME_MAX_TOKENS, rep=rep, quiet=True,
                            hint_tokens=max_tokens, prime=True)))
                    preqs, _bt = await self._await_batch(tasks)
                    await self._prime_done(model, scenario, c, conc, rep)
                    if self.stop_flag:
                        await _flush_rep_points()
                        return
                    for r in preqs:
                        if not r.get("err") and r.get("usage_real"):
                            prime_tokens[r["req"]] = r["prompt_tokens"]
                    pt_txt = (f"实测 prompt {max(prime_tokens)} tokens · "
                              if max(prime_tokens) else "")
                    await self.emit({"type": "status", "msg":
                        f"{model} / Agent 调用 / 缓存 {fmt(c)}："
                        f"前缀缓存已预热（{pt_txt}并行会话 {conc}），开始逐指令档实测"})
                    if any(r.get("ctx_ovf") for r in preqs):
                        msg = (f"{model} / Agent 调用 / 缓存 {fmt(c)} "
                               "预热即超出服务端上下文上限，矩阵提前收口"
                               "（config.json deployments 配 max_ctx 可自动跳档）")
                        await self.emit({"type": "status", "msg": msg})
                        await _abort(msg)
                        return
                for inst in rungs:
                    if self.stop_flag:
                        await _flush_rep_points()
                        return
                    await self.emit({"type": "status", "msg":
                        f"测试 {model} / Agent 调用 / 缓存 {fmt(c)} / "
                        f"指令 {fmt(inst)} / 并行会话 {conc}"})

                    async def _attempt(seg_fix=None, nonce_tag="", seg_salt=0):
                        """一轮测量尝试：预设上下文与指令拆两条 user 消息——
                        先打预热/基线批（缓存档已在上方预热，此处仅零缓存档补
                        [system, 编号行, 超短任务] 基线，量系统/模板/框架开销），
                        再打「上下文 + 指令」测量批；指令真实长度 = 测量 prompt
                        − 预热/基线 prompt（链内差分，固定开销在差分中抵消）。
                        seg_fix 显式指定指令材料字符数（偏离校正补测用）；
                        seg_salt 让指令材料错位取段（弃测重测换料用）。"""
                        primes0 = [0] * conc
                        nonce_of = (lambda i: nonces[i] if c > 0
                                    else f"{nonces[i]}-z{inst}{nonce_tag}")
                        if c == 0:
                            ptasks = [asyncio.create_task(self._one(
                                client, model, build_agent_messages(
                                    0, 0, cpt, nonce_of(i), prime=True)[0],
                                scenario, i, 0, conc, cpt,
                                AGENT_PRIME_MAX_TOKENS, rep=rep, quiet=True,
                                hint_tokens=max_tokens, prime=True))
                                for i in range(conc)]
                            preqs0, _ = await self._await_batch(ptasks)
                            await self._prime_done(model, scenario, 0, conc, rep)
                            if self.stop_flag:
                                return None
                            for r in preqs0:
                                if not r.get("err") and r.get("usage_real"):
                                    primes0[r["req"]] = r["prompt_tokens"]
                        msgs_list, ctx_n, seg_n = [], 0, 0
                        for i in range(conc):
                            msgs, _e, ctx_n, seg_n = build_agent_messages(
                                c, inst, cpt, nonce_of(i),
                                cache_chars=cache_chars, seg_chars=seg_fix,
                                seg_salt=seg_salt)
                            msgs_list.append(msgs)
                        # tick 展示估值 = 指令档 + 缓存块对齐碎块尾（前缀缓存
                        # 按 KV block 命中，预热不足一块的尾部随测量一起增量
                        # 计算；块大小取 cache_block 的 gcd 推断）
                        tick_est = None
                        if c > 0:
                            tail = 0
                            blk = self.cache_block.get(key)
                            prev = max(prime_tokens)
                            if blk and prev:
                                tail = prev - (prev // blk) * blk
                            tick_est = inst + tail
                        tasks = [asyncio.create_task(self._one(
                                    client, model, msgs, scenario,
                                    i * n_inst + inst_ladder.index(inst), c,
                                    conc, cpt, max_tokens, rep=rep,
                                    est_tokens=tick_est))
                                 for i, msgs in enumerate(msgs_list)]
                        reqs, batch_time = await self._await_batch(tasks)
                        if self.stop_flag:
                            return None
                        return msgs_list, ctx_n, seg_n, reqs, batch_time, primes0

                    async def _attempt_track(**kw):
                        """C>0 缓存档的错位取段游标接管：本缓存档内每一次测量
                        尝试（首测/校正补测/污染重测/异常重测）独占一段语料
                        流——seg_salt 取当前游标，尝试返回后游标前进 本次
                        seg_n + seg_n//2 + 1024（冗余兜住校正补测的长度
                        增长），保证任意两次尝试的指令材料在流中不重叠、不
                        互为前缀；否则取段偏移只随 ctx 定（见
                        build_agent_messages），同缓存档按指令档升序测量时
                        各档材料是嵌套前缀，上一档残留被服务端算进
                        cache_hit、增量 prefill 口径分子失真（实测 128K 档
                        4K 指令未命中仅 1554 的事故）。C==0 零缓存档逐点换
                        nonce 破缓存（nonce 在上下文消息头部、缓存从头就
                        断），无此污染，保持现状（异常重测仍用 seg_salt=1
                        换料）。"""
                        nonlocal salt_cursor
                        if c > 0:
                            kw["seg_salt"] = salt_cursor
                        att = await _attempt(**kw)
                        if c > 0 and att is not None:
                            seg_used = att[2]
                            salt_cursor += seg_used + seg_used // 2 + 1024
                        return att

                    def _real_inst(reqs, primes):
                        """指令真实长度（tokens）= 测量批实测 prompt 均值 −
                        预热/基线批实测 prompt 均值（差分基准两侧 usage 缺一
                        不产出，None）。"""
                        ok_u = [r for r in reqs
                                if not r.get("err") and r.get("usage_real")]
                        pr = [p for p in primes if p]
                        if not ok_u or len(pr) != conc:
                            return None
                        return (round(sum(r["prompt_tokens"] for r in ok_u)
                                      / len(ok_u))
                                - round(sum(pr) / len(pr)))

                    att = await _attempt_track()
                    if att is None:
                        await _flush_rep_points()
                        return
                    msgs_list, ctx_n, seg_n, reqs, batch_time, primes0 = att
                    real_inst = _real_inst(reqs, prime_tokens if c > 0
                                           else primes0)
                    # 指令长度构建偏离校正（与 CTX_DEV_TOLERANCE 同思路）：实测
                    # 指令长度偏离目标 >10%（绝对量 >8 tokens 才计）时按实测
                    # 密度校正指令材料字符量补测一次——短指令档被固定模板/框架
                    # 开销淹没是常态，不补测档位名不副实。零缓存档补测换新
                    # nonce（不命中首测缓存）；缓存档上下文已在缓存里，换指令
                    # 材料不影响前缀命中
                    if (real_inst is not None
                            and abs(real_inst - inst) > max(0.1 * inst, 8)):
                        seg_new = max(1, round(seg_n * inst / real_inst))
                        if seg_new != seg_n:
                            await self.emit({"type": "status", "toast": True,
                                             "msg":
                                f"{model} / Agent 调用 / 缓存 {fmt(c)} / "
                                f"指令 {fmt(inst)}：实测指令长度 {real_inst} "
                                "tokens 偏离目标，按实测密度校正补测一次"})
                            att = await _attempt_track(seg_fix=seg_new,
                                                       nonce_tag="r")
                            if att is None:
                                await _flush_rep_points()
                                return
                            (msgs_list, ctx_n, seg_n, reqs, batch_time,
                             primes0) = att
                            real2 = _real_inst(reqs, prime_tokens if c > 0
                                               else primes0)
                            if real2 is not None:
                                real_inst = real2
                    ok = [r for r in reqs if not r.get("err")]
                    anomaly_mark = None
                    reps_discarded = None
                    # 指令段缓存残留污染兜底重测（仅 C>0 回传命中批，判据见
                    # _cache_pollution_verdict）：实测未命中 tokens 远低于
                    # 指令档期望 → 上一指令档材料残留被服务端计入
                    # cache_hit、增量口径分子失真——整批弃测、游标换新段
                    # 重测一次（reps_discarded 标 cache_pollution，带实测
                    # uncached/expected 便于事后复核）；重测仍偏差过大按实
                    # 留档（point["anomaly"]），不无限重试洗掉。未回传命中
                    # 的网关走 _cache_hit_verdict 估算路径（估算命中按预热
                    # prompt 对齐、天然免疫此污染），不纳入本校验
                    if c > 0 and not self.stop_flag:
                        blk = self.cache_block.get(key)
                        prev = max(prime_tokens)
                        tail = (prev - (prev // blk) * blk
                                if (blk and prev) else 0)
                        pol = _cache_pollution_verdict(reqs, real_inst, tail)
                        if pol is not None:
                            uncached_mean, expected = pol
                            reps_discarded = [{
                                "anomaly": "cache_pollution", "rep": rep + 1,
                                "uncached_tokens": uncached_mean,
                                "expected_tokens": expected,
                            }]
                            await self.emit({"type": "status", "toast": True,
                                             "msg":
                                f"{model} / Agent 调用 / 缓存 {fmt(c)} / "
                                f"指令 {fmt(inst)}：实测未命中 {uncached_mean} "
                                f"tokens 与指令档期望 {expected} 偏差过大，"
                                "疑似指令段缓存残留污染，换料重测"})
                            att = await _attempt_track(seg_fix=seg_n,
                                                       nonce_tag="p")
                            if att is None:
                                await _flush_rep_points()
                                return
                            (msgs_list, ctx_n, seg_n, reqs, batch_time,
                             primes0) = att
                            ok = [r for r in reqs if not r.get("err")]
                            # 重测批的指令真实长度差分同样更新（差分基准仍是
                            # 本轮预热实测 prompt）
                            real4 = _real_inst(reqs, prime_tokens)
                            if real4 is not None:
                                real_inst = real4
                            if _cache_pollution_verdict(
                                    reqs, real_inst, tail) is not None:
                                # 重测仍偏差过大：按实留档，不再重试（与异常
                                # 重测「不无限重试洗掉」同口径）
                                anomaly_mark = "cache_pollution"
                    # 输出异常弃测重测（ADR-0032 口径接入 agent 矩阵，仅测点
                    # 判定——预热/基线 quiet 请求不看）：ok 请求命中早停
                    # （阈值 max(0.5×max_tokens, 有效地板)，_early_stop_threshold）
                    # 或退化重复 → 整批测量
                    # 已被肇事请求污染，弃测该批、换新 nonce + 错位指令材料
                    # 整批重测。重测预算按当前异常形态分两档（2026-09-16
                    # 拍板，与 _run_rep_guarded 同口径）：低于有效输出地板
                    # 的早停完全不可信，最多重测 FLOOR_RETEST_MAX=2 次，仍
                    # 低于地板按测量失败留档（肇事 req 标 err，读数不聚合）；
                    # 高于地板的异常最多重测 ANOMALY_RETEST_MAX=1 次，仍异常
                    # 按实留档 point["anomaly"]，不无限重试洗掉。与上方输入
                    # 校正/污染校验的次序：先输入校正、再污染校验、后输出异常
                    # 判定，校正值/污染各最多 1 次、异常重测 ≤2 次、总尝试数
                    # 有界（首测+校正补测+污染重测+异常重测 ≤5 次）
                    ok = [r for r in reqs if not r.get("err")]
                    anom_retries = 0
                    while ok and not self.stop_flag:
                        hit = _detect_agent_anomaly(ok, max_tokens)
                        if hit is None:
                            break
                        kind, aidx = hit
                        culprit = ok[aidx]
                        below_floor = (kind == "early_stop"
                                       and (culprit.get("out_tokens") or 0)
                                           < min(EARLY_STOP_FLOOR_OUT,
                                                 max_tokens))
                        budget = (FLOOR_RETEST_MAX if below_floor
                                  else ANOMALY_RETEST_MAX)
                        if anom_retries >= budget:
                            # 预算耗尽仍异常：按实留档——标记取最新一轮的
                            # 异常类型（留档数据是最新批；轮次间类型可不同，
                            # 如早停↔退化重复，沿用首次类型会误导复核）。
                            # 肇事 req 同步打 anomaly + in_text（输入全文瞬态
                            # 抄本）：留档样本随 req 落档，剥离 _in_full 后
                            # 全文仍在
                            culprit["anomaly"] = kind
                            culprit["in_text"] = culprit.get("_in_full") or ""
                            anomaly_mark = kind
                            if below_floor:
                                # 重测预算耗尽仍低于有效输出地板：肇事 req
                                # 按测量失败留档（标 err——在下方聚合块之前，
                                # 点读数自动按剩余 ok 请求计算、全失败则
                                # 置空），不再按实留档污染聚合
                                # （2026-09-16 拍板）
                                culprit["err"] = (
                                    "输出异常早停（仅 "
                                    f"{culprit.get('out_tokens')}"
                                    f"/{max_tokens} tokens，弃测重测 "
                                    f"{anom_retries} 次仍复现）——读数不可信，"
                                    "按测量失败留档")
                                await self.emit({"type": "status", "toast": True,
                                                 "msg":
                                    f"{model} / Agent 调用 / 缓存 {fmt(c)} / "
                                    f"指令 {fmt(inst)}：重测 {anom_retries} "
                                    f"次仍仅输出 "
                                    f"{culprit.get('out_tokens')} tokens"
                                    "（低于有效地板 "
                                    f"{min(EARLY_STOP_FLOOR_OUT, max_tokens)}"
                                    "），本点按测量失败留档"})
                            break
                        # 弃测留痕（字段形状与 _run_rep_guarded 同口径，
                        # 含输出头部样本供事后人工复核；in_text = 肇事
                        # 请求实际发送的输入全文瞬态抄本，rep = 被弃测的
                        # 1-based 轮次号——矩阵有 rep 循环，取当前轮）
                        reps_discarded = (reps_discarded or []) + [{
                            "anomaly": kind, "req": culprit.get("req"),
                            "rep": rep + 1,
                            "out_tokens": culprit.get("out_tokens"),
                            "finish": culprit.get("finish"),
                            "ttft_s": culprit.get("ttft_s"),
                            "decode_tok_s": culprit.get("decode_tok_s"),
                            "text_sample": culprit.get("text_sample") or "",
                            "reason_sample": culprit.get("reason_sample") or "",
                            "in_sample": culprit.get("in_sample") or "",
                            "in_text": culprit.get("_in_full") or "",
                        }]
                        await self.emit({"type": "status", "toast": True, "msg":
                            f"{model} / Agent 调用 / 缓存 {fmt(c)} / "
                            f"指令 {fmt(inst)}：检测到异常输出"
                            + (f"（提前停止：仅输出 {culprit.get('out_tokens')}"
                               f"/{max_tokens} tokens）" if kind == "early_stop"
                               else "（退化重复）")
                            + f"，整批弃测重测（第 {anom_retries + 1}/{budget}"
                              " 次）"})
                        # C>0 由游标接管错位取段（kw 被覆写）；C==0 保持
                        # seg_salt=1 换料 + nonce_tag 破缓存（逐次重测换不同
                        # tag，同 tag 同材料会命中前缀缓存）
                        att = await _attempt_track(
                            seg_fix=seg_n, seg_salt=1,
                            nonce_tag=f"a{anom_retries}")
                        if att is None:
                            await _flush_rep_points()
                            return
                        (msgs_list, ctx_n, seg_n, reqs, batch_time,
                         primes0) = att
                        anom_retries += 1
                        ok = [r for r in reqs if not r.get("err")]
                        # 重测批的指令真实长度差分同样更新（差分基准仍是
                        # 本轮预热/基线实测 prompt）
                        real3 = _real_inst(reqs, prime_tokens if c > 0
                                           else primes0)
                        if real3 is not None:
                            real_inst = real3

                    ok = [r for r in reqs if not r.get("err")]
                    point = {
                        "model": model, "scenario": scenario, "ctx_target": c,
                        "inst_tokens": inst, "inst_real_tokens": real_inst,
                        "metric_version": METRIC_VERSION,   # 口径版本
                        "concurrency": conc,
                        "reqs": reqs, "all_ok": len(ok) == conc,
                        "batch_time_s": round(batch_time, 3),
                        "prompt_tokens": None, "ttft_s": None,
                        "prefill_tok_s": None, "decode_tok_s": None,
                        "decode_total_tok_s": None,
                        "decode_total_tok_s_adj": None,
                        # 点级 decode Σ÷Σ 口径（_decode_sums）：批级 decode
                        # 总耗时
                        "decode_time_s": None,
                        # 批级并发口径（_conc_batch_stats）：对全并发档产出
                        "prefill_span_s": None, "total_span_s": None,
                        "decode_span_s": None,
                        "prefill_conc_tok_s": None, "decode_conc_tok_s": None,
                        "decode_peak_tok_s": None,
                        "out_tokens": None, "ttft_net_s": None,
                        "prefill_net_tok_s": None, "total_s": None,
                        "rtt_ms": (round(self.rtt_s * 1000)
                                   if self.rtt_s is not None else None),
                        "max_gap_s": None, "stall_s": None, "stall_count": None,
                        "decode_burst": None,
                        "flush_tok": None, "flush_count": None, "flush_s": None,
                    }
                    if anomaly_mark:
                        point["anomaly"] = anomaly_mark   # 重测仍异常留档
                    if reps_discarded:
                        point["reps_discarded"] = reps_discarded   # 弃测留痕
                    if len(ok) != conc:   # 失败原因落档（agent 矩阵同口径）
                        point["err"] = _first_err(reqs)
                    _strip_in_full(point)   # 定稿前剥离输入全文瞬态字段（in_text 已抄本）
                    if ok:
                        # 批级并发口径先算好（守卫见 _conc_batch_stats）：批级
                        # prefill 点口径与 6 指标产出都消费
                        stats = _conc_batch_stats(ok, conc)
                        if conc == 1:
                            gaps = [r["max_gap_s"] for r in ok
                                    if r.get("max_gap_s")]
                            if gaps:
                                point["max_gap_s"] = max(gaps)   # 停滞诊断取最差
                            stalls = [r for r in ok if r.get("stall_s")]
                            if stalls:
                                worst = max(stalls, key=lambda r: r["stall_s"])
                                point["stall_s"] = worst["stall_s"]
                                point["stall_count"] = worst.get("stall_count")
                            flushes = [r for r in ok if r.get("flush_tok")]
                            if flushes:
                                # 停滞回吐剔除量取请求中最差（与 stall_s 同口径）
                                worst = max(flushes, key=lambda r: r["flush_tok"])
                                point["flush_tok"] = worst["flush_tok"]
                                point["flush_count"] = worst.get("flush_count")
                                point["flush_s"] = worst.get("flush_s")   # 同源同请求
                        bursts = [bool(r.get("decode_burst"))
                                  for r in ok if r.get("decode_tok_s")]
                        if bursts and sum(bursts) * 2 > len(bursts):
                            point["decode_burst"] = True   # 多数决
                        point["prompt_tokens"] = round(
                            sum(r["prompt_tokens"] for r in ok) / len(ok))
                        # TTFT 批级口径优先（ADR-0074）：TTFT = 批开始→全部首
                        # token（prefill_span_s），conc=1 与单请求口径严格相等；
                        # 批级不可用（守卫未过/分母 ≤0，如部分失败批）时回退
                        # 逐请求均值口径（净口径同批级，走 _batch_prefill_point）
                        bp = _batch_prefill_point(
                            sum(r["prompt_tokens"] for r in ok), stats,
                            self.rtt_s)
                        if bp:
                            point.update(bp)
                        else:
                            ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
                            if ttfts:
                                point["ttft_s"] = round(sum(ttfts) / len(ttfts), 3)
                            tns = [r["ttft_net_s"] for r in ok
                                   if r.get("ttft_net_s")]
                            if tns:
                                point["ttft_net_s"] = round(sum(tns) / len(tns), 3)
                        # 增量 prefill 口径：未命中 tokens ÷ TTFT。命中三态——
                        # 服务端回传按真值；未回传时做缓存迹象三通道判别
                        # （走平/全量预期/本轮已证实，见 _cache_hit_verdict），
                        # 有迹象按预热实测 prompt 估算命中，无迹象 hit=None
                        # （前端「未回传」）、uncached=全量，prefill 为全量
                        # 口径真值。零缓存档 hit 恒按回传/0
                        pps, pns, pfs, hits = [], [], [], []
                        sum_uncached = 0   # 未命中 tokens 累计（批级速率分子）
                        for r in ok:
                            ptok, tt = r.get("prompt_tokens"), r.get("ttft_s")
                            if c == 0:
                                hit = ((r.get("cache_hit") or 0)
                                       if r.get("cache_reported") else 0)
                            elif r.get("cache_reported"):
                                hit = r.get("cache_hit") or 0
                                if hit > 0:
                                    cache_proven = True   # 回传命中>0：本轮证实
                            elif ptok:
                                hit = _cache_hit_verdict(
                                    c, ptok, r.get("ttft_net_s") or tt,
                                    base_ttft.get(inst),
                                    rate_prior(self.prefill_curve.get(key)
                                               or [], ptok),
                                    # 估算基准：预热实测 prompt 优先；网关
                                    # 间歇不回传预热 usage 时回退 ptok−inst
                                    # （指令档目标）——prime=0 曾把 proven
                                    # 通道也短路、整档虚高留档（2026-09-16
                                    # 事故，见 _cache_hit_verdict docstring）
                                    prime_tokens[r["req"] // n_inst]
                                    or max(ptok - inst, 0),
                                    cache_proven)
                                if hit is not None:
                                    cache_proven = True   # 估算命中：本轮证实
                            else:
                                hit = None
                            hits.append(hit)
                            if not ptok or not tt:
                                continue
                            uncached = (max(ptok - hit, 1)
                                        if hit is not None else ptok)
                            sum_uncached += uncached
                            pps.append(uncached / tt)
                            pfs.append(ptok / tt)   # 全量口径（对照/换算用）
                            if r.get("ttft_net_s"):
                                pns.append(uncached / r["ttft_net_s"])
                        if c == 0:
                            # 零缓存档 TTFT 基准（净口径优先，跨链均值）
                            ts = [r.get("ttft_net_s") or r.get("ttft_s")
                                  for r in ok]
                            ts = [t for t in ts if t]
                            if ts:
                                base_ttft[inst] = sum(ts) / len(ts)
                        # 点级增量 prefill 批级口径优先（ADR-0074）：Σ未命中
                        # tokens ÷ 批级 span，全量口径 = Σprompt_tokens ÷ span；
                        # 批级不可用时回退逐请求均值
                        bp = _batch_prefill_point(sum_uncached, stats,
                                                  self.rtt_s)
                        if bp:
                            point.update(bp)
                            point["prefill_full_tok_s"] = round(
                                sum(r["prompt_tokens"] for r in ok)
                                / stats["prefill_span_s"], 1)
                        elif pps:
                            point["prefill_tok_s"] = round(sum(pps) / len(pps), 1)
                        if not bp and pfs:
                            # 全量口径字段（additive）：prefill_tok_s 为增量口径
                            point["prefill_full_tok_s"] = round(
                                sum(pfs) / len(pfs), 1)
                        if not bp and pns:
                            point["prefill_net_tok_s"] = round(
                                sum(pns) / len(pns), 1)
                        # 点级 decode Σ÷Σ 口径（ADR-0074 追加，同 _run_point）
                        dt, dc = _decode_sums(ok)
                        if dt is not None:
                            point["decode_time_s"] = dt
                        if dc is not None:
                            point["decode_tok_s"] = dc
                        dca = [r["decode_tok_s_adj"] for r in ok
                               if r.get("decode_tok_s_adj")]
                        if dca:
                            point["decode_tok_s_adj"] = round(
                                sum(dca) / len(dca), 1)
                        tpc = [r["tok_per_chunk"] for r in ok
                               if r.get("tok_per_chunk")]
                        if tpc:
                            point["tok_per_chunk"] = round(
                                sum(tpc) / len(tpc), 2)
                        tots = [r["total_s"] for r in ok if r.get("total_s")]
                        if tots:
                            # 端到端总时长（TTFT + decode 全程）：agent 单步
                            # 延迟的直观口径，跨链均值
                            point["total_s"] = round(sum(tots) / len(tots), 2)
                        # 总吞吐（conc=1 单链口径，与 _run_point 同公式）；
                        # conc>1 跨链 decode 重叠口径不保证，保持 null（诚实）
                        if conc == 1 and len(ok) == 1:
                            raw_total, adj_total = _total_decode_rates(ok, 1)
                            if raw_total is not None:
                                point["decode_total_tok_s"] = raw_total
                            if adj_total is not None:
                                point["decode_total_tok_s_adj"] = adj_total
                        # 批级并发口径 6 指标：conc>1 同样产出（这正是批级
                        # 口径对总吞吐口径的补位——不依赖重叠守卫）
                        point.update(stats)
                        # 缓存命中：跨链均值。网关未回传字段时 hits 装的是判别
                        # 估算值（cache_reported=False，前端加 ≈）；连迹象都
                        # 没有时为 None——前端显示「未回传」，不展示假设命中
                        known = [h for h in hits if h is not None]
                        point["cache_hit_tokens"] = (round(sum(known) / len(known))
                                                     if known else None)
                        point["cache_reported"] = all(
                            r.get("cache_reported") for r in ok)
                        if point["cache_reported"]:
                            # 块大小推断：回传命中值按 KV block 对齐，逐点 gcd
                            # 收敛；退化到 <64 说明命中口径不按块对齐，不用于
                            # 碎块修正（与旧 agent 链同口径）
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
                            point["finish"] = (uniq[0] if len(uniq) == 1
                                               else ",".join(uniq))
                        point["out_tokens"] = round(
                            sum(r.get("out_tokens") or 0 for r in ok) / len(ok))
                    # 实测速率登记进先验曲线（conc=1 单链口径）：x 坐标取未命中
                    # 量（≈指令段长度），后续同规模指令档的开局估值锚点。
                    # 公共口径见 _prefill_prior_sample（估算命中不登记、
                    # _prior_suspect 防污染闸）——点级速率本身即未命中增量口径，
                    # 不再传 req 级 rate_uncached
                    if conc == 1 and point.get("all_ok"):
                        sample = _prefill_prior_sample(
                            point["ctx_target"], point.get("prompt_tokens"),
                            bool(point.get("cache_reported")),
                            point.get("cache_hit_tokens"),
                            (point.get("prefill_net_tok_s")
                             or point.get("prefill_tok_s")),
                            None, self.prefill_curve.get(key) or [])
                        if sample:
                            self._note_prefill_rate(key, *sample)
                    await self._calibrate_usage(
                        key, sum(len(m["content"]) for m in msgs_list[0]),
                        ctx_n + seg_n, ok)
                    for r in reqs:   # 剥离 token 交付时间轴瞬态字段（出点定稿）
                        r.pop("_tl", None)
                    rep_points[(c, inst)].append(point)
                    if repeats == 1:
                        self.results.append(point)
                        await self.emit({"type": "point", "point": point})
                    # 软/硬超窗错误（req 级 ctx_ovf 标记）：矩阵就此收口——
                    # 更大缓存/指令组合必然同样超窗。失败组合照常出点，剩余
                    # 组合 point_skipped 计入进度（ADR-0018 同口径）
                    if any(r.get("ctx_ovf") for r in reqs):
                        msg = (f"{model} / Agent 调用 / 缓存 {fmt(c)} / "
                               f"指令 {fmt(inst)} 超出服务端上下文上限，矩阵"
                               "提前收口（config.json deployments 配 max_ctx "
                               "可自动跳档）")
                        await self.emit({"type": "status", "msg": msg})
                        await _abort(msg)
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
                   img_tokens: int = 0, hint_tokens: int | None = None,
                   prime: bool = False, _retried: int = 0,
                   _empty_retried: int = 0,
                   _edge_fb: int = 0, _trim_keep: int | None = None,
                   _orig_chars: int | None = None) -> dict:
        # quiet=True（探测/预热请求）：不发 tick/实时事件，只走完整请求链路并返回
        # 结果。prime=True 例外：保持 status 日志静默但发实时 tick（带 prime 标记）
        # ——大缓存档预热的 prefill 长达数十秒，全程无帧会让前端实时表与「进行中
        # 请求」空转，看起来像组件未加载；prime tick 由前端单独成列、不进 KPI
        sc = SCENARIOS[scenario]
        # 输出估算系数取实测校准值（首个请求由探测结果播种），固定系数与实际
        # 输出内容不符时实时读数会数倍虚高（实测 70+ vs 25+，ADR-0005）
        out_cpt = self.out_cpt_calib.get((model, scenario)) or sc["out_cpt"]
        tag = f"{model}|{scenario}|{ctx}|{conc}|rep{rep}|r{req_i}"
        # messages 拷成 msgs_sent 再入 payload：超窗删减重试只改副本，
        # 调用方（_run_point）持有的原始构造不被改写
        msgs_sent = [dict(m) for m in messages]
        # 贴边回退裁减的重入（_trim_keep 非 None）：掐中段填充语料到指定字符量，
        # 与超窗删减重试同一 _trim_middle 口径（保留开头编号行与结尾任务指令）
        if _trim_keep is not None and isinstance(msgs_sent[1]["content"], str):
            msgs_sent[1] = {**msgs_sent[1],
                            "content": _trim_middle(msgs_sent[1]["content"],
                                                    _trim_keep)}
        # 输出长度引导（ADR-0016 防线 2）：上限按 out_cpt 先验折成字数（×1.2
        # 容差）写进 system——服务端不截断时，模型自身尽量收在预设长度附近。
        # creative/code/agent 为双向引导：目标字数 aim = limit ÷ OUT_HINT_FACTOR
        # （out_cpt 先验字数）防模型往远低于预算的长度收敛、短输出 decode 窗口
        # 不稳定（agent 实测还以「材料截断/信息不足」为由提前短答，EARLY-STOP
        # 案例）；translate/ocr 保持单向封顶（翻译输出应与原文等比，引导写长
        # 是错的）。先验固定，run 内 system 恒定（前缀缓存/嵌套前缀记账不受
        # 影响）；agent 矩阵预热请求（输出预算 16）按测量请求的 max_tokens
        # 折算引导字数（hint_tokens）——system 是首条消息，引导字数不同会让
        # 预热与测量请求在 system 内就分叉，整个前缀缓存预热落空
        aim_chars = int((hint_tokens or max_tokens) * sc["out_cpt"])
        limit_chars = int((hint_tokens or max_tokens) * sc["out_cpt"]
                          * OUT_HINT_FACTOR)
        msgs_sent[0] = {**msgs_sent[0], "content":
                        msgs_sent[0]["content"] + "\n\n"
                        + sc["out_hint"].format(limit=limit_chars,
                                                aim=aim_chars)}
        # 指令末尾复述约束（_out_hint_tail，2026-09-16 拍板）：同一约束在末条
        # user 消息末尾再写一份，防模型忽略 system 引导提前短答。复述文本与
        # system 完全同源（同 aim/limit、同模板）——agent 矩阵预热/测量请求的
        # 末条消息都复述，链内差分互相抵消；缓存档前缀（system+上下文）不含
        # 复述段，前缀缓存命中不受影响。多模态 parts（OCR）跳过
        _tail = _out_hint_tail(sc, (self.cfg.get("model_max_ctx") or {})
                               .get(model), ctx, hint_tokens or max_tokens)
        if _tail:
            _tail_txt = _tail.format(limit=limit_chars, aim=aim_chars)
            for _mi in range(len(msgs_sent) - 1, -1, -1):
                _m = msgs_sent[_mi]
                if _m.get("role") == "user" and isinstance(_m.get("content"), str):
                    msgs_sent[_mi] = {**_m, "content":
                                      _m["content"] + "\n\n" + _tail_txt}
                    break
        if isinstance(msgs_sent[1]["content"], list):
            # OCR 多模态 parts（ADR-0020）：text 部分按 in_cpt 折算，图像按
            # w×h÷750 估值（调用方传 img_tokens，与服务端口径近似、只影响展示）
            text_chars = sum(len(p.get("text", "")) for p in msgs_sent[1]["content"]
                             if isinstance(p, dict) and p.get("type") == "text")
            est_prompt = round(text_chars / max(cpt, 0.1)) + img_tokens
        else:
            # echo 请求带 assistant 预填消息（ADR-0021）：估值计入全部消息
            est_prompt = round(sum(len(m["content"]) for m in msgs_sent) / max(cpt, 0.1))
        # tick 展示估值：agent 矩阵缓存档只增量 prefill，按调用方给的增量估值显示
        # （全量估值在短 TTFT 上会爆出天文数字）；est_prompt 本体仍用于 usage
        # 缺失时的兜底记账，不受展示估值影响
        tick_est = est_tokens or est_prompt
        # prime 请求发 tick：quiet 只挡 status 日志（预热重试/校准不刷日志），
        # 实时帧照常走——tick_off 统一收口本函数内所有 tick 闸
        tick_off = quiet and not prime
        # 贴边回退军备（CTX_EDGE_FB_RATIO）：估算 prompt + 输出预算 + 分词漂移
        # 余量贴着部署上限运行时（含贴边裁减点：裁后恰顶 max_ctx − 预算 − 余量），
        # 未识别形态的失败可走删减回退。_edge_fb > 0 的重入保持军备——裁减后的
        # est_prompt 会跌出比例区，以裁后尺寸重判会让回退阶梯第二档起失效
        _max_ctx = (self.cfg.get("model_max_ctx") or {}).get(model)
        # 贴边军备（原口径）：仅「失败可能源于上下文/显存超限」时才敢删减语料
        edge_near_limit = bool(_max_ctx) and (
            est_prompt + max_tokens + CTX_HEADROOM >= _max_ctx * CTX_EDGE_FB_RATIO)
        # 非贴边军备（EMPTY_STREAM_FORCE_EDGE_FB）：同文本空流重试已耗尽——
        # 重试这个动作被证伪，改 shape 是本点唯一的自救手段。与贴边无关：
        # JIT 类故障不看档位（见常量注释）。_edge_fb > 0 的重入一律保持军备
        # ——裁减后 est_prompt 跌出比例区，若重判会让第二档起失效
        edge_forced = EMPTY_STREAM_FORCE_EDGE_FB and _empty_retried > 0
        edge_armed = _edge_fb > 0 or edge_near_limit or edge_forced

        async def emit_tick(payload: dict):
            """tick 统一出口：prime 请求打标记（前端据此单独成列、不进 KPI）；
            错误帧随发悬浮岛提醒（ADR-0052：任何失败都应有提醒——此前仅兜底
            重试调度弹 toast，速率闸/超窗等确定性失败只在实时表留个 ✗）。"""
            if prime:
                payload["prime"] = True
            await self.emit(payload)
            # stop_flag 置位（用户主动停止）时不弹失败提醒：中断是用户动作本身，
            # 不是需要提醒的失败（停止路径不播报的既有口径不变）
            if payload.get("phase") == "error" and not self.stop_flag:
                await self.emit({"type": "status", "toast": True,
                                 "msg": f"{model} / {sc['label']} / "
                                        f"{_fmt_scenario_ctx(scenario, ctx)} "
                                        f"请求 r{req_i} 失败：{payload.get('msg', '')[:120]}"})

        if not tick_off:
            await emit_tick({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
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
        # echo 预填走真·续写语义（ADR-0079）：末条为 assistant 预填时下发
        # add_generation_prompt=false + continue_final_message=true——模板不再
        # 把预填闭合成已完结轮次再开新轮，而是从未闭合的预填末尾直接续写。
        # 闭合轮次形态在长上下文触发「已作答」吸引子（复述预填几句即 EOS，
        # 32K+ 档确定性早停的根因，2026-09-17 直连对照实验实锤）；真续写
        # 同构造翻牌为满预算，且输出与参考材料重叠率更高（更贴近投机采样
        # 高接受率场景）。不支持的网关 400 后省略两参数并锁定（下方参数锁），
        # 回退闭合轮次形态
        if (msgs_sent and msgs_sent[-1].get("role") == "assistant"
                and model not in self.prefill_locked):
            payload["add_generation_prompt"] = False
            payload["continue_final_message"] = True
        # 端点路径：未锁定时先试 /v1，404 则回退无前缀（DeepSeek 风格）
        candidates = ([self.api_prefix] if self.prefix_locked
                      else ["/v1", ""])
        # 瞬时失败兜底重试：_retried 为当前已重试次数，递归重入 _one 实现
        # 整体重来（全新计时/流状态，msgs_sent/payload 的网关适配锁定随
        # self.* 持久）。返回 None = 不重试（额度耗尽或 stop_flag 已置位），
        # 调用方按既有错误口径收口；quiet 探测/预热同样享受重试但保持静默
        # （不发 tick/status，避免 agent 矩阵预热刷日志）
        async def _retry(msg: str, retry_after: float | None) -> dict | None:
            if self.stop_flag or _retried >= TRANSIENT_RETRY_MAX:
                return None
            wait = retry_after or (_retried + 1) * TRANSIENT_RETRY_BASE_S
            if not quiet:
                await self.emit({"type": "status", "msg":
                    f"{model} / {_fmt_scenario_ctx(scenario, ctx)} 请求 r{req_i} 失败"
                    f"（{msg[:60]}），{wait:g}s 后兜底重试 "
                    f"{_retried + 1}/{TRANSIENT_RETRY_MAX}", "toast": True})
            await asyncio.sleep(wait)
            return await self._one(client, model, messages, scenario, req_i,
                                   ctx, conc, cpt, max_tokens, rep=rep,
                                   quiet=quiet, est_tokens=est_tokens,
                                   img_tokens=img_tokens,
                                   hint_tokens=hint_tokens,
                                   prime=prime, _retried=_retried + 1,
                                   _empty_retried=_empty_retried,
                                   # 贴边回退状态透传：重试保持同一删减档
                                   _edge_fb=_edge_fb, _trim_keep=_trim_keep,
                                   _orig_chars=orig_user_chars)

        async def _retry_empty(msg: str) -> dict | None:
            """空流专属重试（EMPTY_STREAM_RETRY_MAX）：Lvllm 混合架构的「空 EOS」
            （200 正常但零内容、completion≈1）是间歇性服务端状态故障，同消息
            重试**可能**自愈——独立计数不挤占通用瞬时重试预算，留痕合并进
            retried。次数与通用瞬时重试同档（2 次）：原 4 次的「约 50% 自愈」
            依据经复核不成立（见常量注释），已按无依据不臆设恢复率下调。"""
            if self.stop_flag or _empty_retried >= EMPTY_STREAM_RETRY_MAX:
                return None
            wait = (_empty_retried + 1) * TRANSIENT_RETRY_BASE_S
            if not quiet:
                await self.emit({"type": "status", "msg":
                    f"{model} / {_fmt_scenario_ctx(scenario, ctx)} 请求 r{req_i} 失败"
                    f"（{msg[:60]}），{wait:g}s 后兜底重试 "
                    f"{_empty_retried + 1}/{EMPTY_STREAM_RETRY_MAX}",
                    "toast": True})
            await asyncio.sleep(wait)
            return await self._one(client, model, messages, scenario, req_i,
                                   ctx, conc, cpt, max_tokens, rep=rep,
                                   quiet=quiet, est_tokens=est_tokens,
                                   img_tokens=img_tokens,
                                   hint_tokens=hint_tokens,
                                   prime=prime, _retried=_retried,
                                   _empty_retried=_empty_retried + 1,
                                   _edge_fb=_edge_fb, _trim_keep=_trim_keep,
                                   _orig_chars=orig_user_chars)

        async def _edge_fallback(msg: str) -> dict | None:
            """贴边档未识别失败的超窗回退裁减：贴着部署上限运行的点位被秒拒
            （报文不命中 _is_ctx_overflow 关键词，如 fastllm 显存/KV 不足直接
            500、连接重置、空流）大概率仍是上下文/显存超限——按 CTX_RETRY_KEEP
            阶梯删减填充语料递归重入 _one 整体重试（全新计时/流状态，瞬时重试
            额度重置）。返回 None = 未军备/阶梯耗尽/停止，调用方按原错误收口
            （_edge_fb > 0 时调用方在错误文本追加疑似超限提示）。"""
            if (not edge_armed or self.stop_flag
                    or _edge_fb >= len(CTX_RETRY_KEEP)):
                return None
            if (orig_user_chars <= 2000
                    or not isinstance(msgs_sent[1]["content"], str)):
                return None
            keep = int(orig_user_chars * CTX_RETRY_KEEP[_edge_fb])
            if not quiet:
                # 播报原因按军备来源区分：贴边档说「疑似超限」（原口径），
                # 强制军备档不断言超限（JIT/未知服务端故障与档位无关，谎报
                # 原因会把现场判断带偏）
                _why = ("临近部署上下文上限（%s tokens），疑似上下文/显存超限"
                        % _max_ctx) if edge_near_limit else (
                        "同文本重试耗尽仍空流（疑似服务端按 shape 重建/资源"
                        "瞬时限缩，非上下文超限）")
                await self.emit({"type": "status", "toast": True, "msg":
                    f"{model} / {_fmt_scenario_ctx(scenario, ctx)} 请求 r{req_i} 失败"
                    f"（{msg[:60]}）——{_why}，"
                    f"删减填充语料至约 {keep} 字符回退重试 "
                    f"{_edge_fb + 1}/{len(CTX_RETRY_KEEP)}"})
            r = await self._one(client, model, messages, scenario, req_i, ctx,
                                conc, cpt, max_tokens, rep=rep, quiet=quiet,
                                est_tokens=est_tokens, img_tokens=img_tokens,
                                hint_tokens=hint_tokens, prime=prime,
                                _edge_fb=_edge_fb + 1, _trim_keep=keep,
                                _orig_chars=orig_user_chars)
            if r is not None and not r.get("err"):
                # 留痕：贴边回退第几档救回——setdefault 保最深帧的档数，
                # 外层帧逐层返回时不覆写
                r.setdefault("edge_fallback", _edge_fb + 1)
            return r
        t0 = time.perf_counter()
        first = last = None
        max_gap = 0.0           # 相邻内容块最大空窗（秒，停滞诊断）
        adj_time = 0.0          # 空窗校正后的 decode 窗口（每段间隔封顶 DECODE_GAP_CAP）
        gaps: list[float] = []  # 相邻内容块间隔全量（tok_per_batch 批次聚类用）
        ev_chars: list[int] = []   # 每个内容事件的字符增量（text+reason，与
                                   # n_chunks 对齐；停滞回吐的 per-event token
                                   # 速率判据用，ADR-0054）
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
            if tick_off:
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
            await emit_tick({"type": "tick", "tag": tag, "model": model,
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
            # 删减阶梯的比例基准恒为首次构造量（贴边回退重入时 sent_chars 已是
            # 裁后尺寸，经 _orig_chars 透传原值，两道阶梯口径一致）
            orig_user_chars = _orig_chars or sent_chars
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
                    # 改回 max_tokens 并锁定（mt_mct_rejected 防翻转振荡，ADR-0016）；
                    # add_generation_prompt/continue_final_message 为模型级成对
                    # 锁定——不认续写参数的网关回退闭合轮次形态（ADR-0079）
                    for _param in ("thinking", "temperature",
                                   "max_completion_tokens",
                                   "add_generation_prompt",
                                   "continue_final_message"):
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
                        elif _param in ("add_generation_prompt",
                                        "continue_final_message"):
                            # 续写参数成对省略并锁定（ADR-0079）：回退闭合
                            # 轮次形态（旧行为），后续请求不再下发
                            payload.pop("add_generation_prompt", None)
                            payload.pop("continue_final_message", None)
                            self.prefill_locked.add(model)
                            note = "已省略续写参数、按闭合轮次形态重试"
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
                        f"{model} 上下文{_fmt_scenario_ctx(scenario, ctx)} "
                        f"超出模型上下文窗口，"
                        f"删减填充语料至约 {keep} 字符重试"
                        f"（{ovf_attempt}/{len(CTX_RETRY_KEEP)}）"})
                    continue
                if _is_ctx_overflow(resp.status_code, body):
                    # 删减重试耗尽（或不适用删减的请求形态）仍超限：按错误收口
                    # 不再重试（ADR-0018）；ctx_ovf 标记供 agent 矩阵终止后续组合
                    msg = _classify_http_error(resp.status_code, body)
                    if not tick_off:
                        await emit_tick({"type": "tick", "tag": tag, "model": model,
                                         "scenario": scenario, "ctx": ctx, "conc": conc,
                                         "req": req_i, "phase": "error", "msg": msg})
                    return {"req": req_i, "err": msg, "ctx_ovf": True}
                # 瞬时类状态（429/5xx）：兜底重试后仍失败才走 RuntimeError
                # 收口（与其他 4xx 同路径，保持既有 error tick + err 返回）；
                # 贴边档未识别失败先走删减回退（大概率是未自报的超窗/显存超限）
                msg = _classify_http_error(resp.status_code, body)
                if _is_transient_http_status(resp.status_code):
                    retried = await _retry(msg, _retry_after_s(resp.headers))
                    if retried is not None:
                        return retried
                fb = await _edge_fallback(msg)
                if fb is not None:
                    return fb
                if _edge_fb:   # 回退阶梯耗尽仍失败：补疑似超限提示
                    msg += ("（临近部署 max_ctx 上限，删减回退后仍失败"
                            "——大概率上下文/显存超限）")
                raise RuntimeError(msg)
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
                            if not tick_off:
                                # 首 token 到达 = 本请求 prefill 完成：TTFT 实测、
                                # prompt 取校准构造值，读数即刻刷新，不等整点
                                # 完成（请求级，ADR-0007）
                                ttft0 = now - t0
                                net0 = _net_elapsed(ttft0, self.rtt_s)
                                await emit_tick({"type": "tick", "tag": tag, "model": model,
                                                 "scenario": scenario, "ctx": ctx, "conc": conc,
                                                 "req": req_i, "phase": "prefill",
                                                 "speed": round(tick_est / net0, 1),
                                                 "ttft": round(ttft0, 3),
                                                 "elapsed": round(ttft0, 2)})
                        elif now - last > max_gap:
                            # 相邻内容块最大空窗：decode 窗口被传输/调度停滞拖尾时
                            # （如隧道拥塞），该点读数不可信——记录供前端警示
                            max_gap = now - last
                        if first is not None and last is not None:
                            # gap 必须与内容事件一一对应（gaps[i-1] ↔ 第 i 个事件）：
                            # 同刻到达（now == last，粗粒度时钟或极快突发）也要记 0。
                            # 旧判据 `now > last` 会漏记这一条，gaps 比事件少一项，
                            # 消费方 _decode_peak_rate 的 gaps[i-1] 直接 IndexError
                            # 打掉整个测点（2026-09 假时钟回归暴露）
                            gap = max(now - last, 0.0)
                            gaps.append(gap)   # 首块无间隙，逐块全量收集（批次聚类）
                            adj_time += min(gap, DECODE_GAP_CAP)
                            if gap >= STALL_THRESHOLD_S:   # 满额计入，非时长折扣
                                stall_s += gap
                                stall_count += 1
                        ev_chars.append((len(piece) if piece else 0)
                                        + (len(rpiece) if rpiece else 0))
                        last = now
                        if not tick_off and now - last_emit >= 0.4:
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
                            await emit_tick({
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
            if _is_transient_exc(e):   # httpx 传输异常：兜底重试后再认败
                retried = await _retry(msg, None)
                if retried is not None:
                    return retried
            # 贴边档未识别失败（连接重置/读写超时等）：删减回退后再认败
            fb = await _edge_fallback(msg)
            if fb is not None:
                return fb
            if _edge_fb:   # 回退阶梯耗尽仍失败：补疑似超限提示
                msg += ("（临近部署 max_ctx 上限，删减回退后仍失败"
                        "——大概率上下文/显存超限）")
            if not tick_off:
                await emit_tick({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        finally:
            # resp_ctx 创建后即纳入收口：enter_stream/400 重试段抛错或停止取消也
            # 必须关闭连接（404/400 分支的手动 aexit 幂等，重复关闭无害）
            if resp_ctx is not None:
                await resp_ctx.__aexit__(None, None, None)

        t_end = time.perf_counter()
        if first is None:
            # 措辞按实测逻辑写准（2026-09-22）：走到本分支 = 连 reasoning_content
            # 都没有（first 由 content 或 reasoning_content 任一非空置位）——是
            # **彻底的空**，不是「思考占满正文」。旧文案「可能被 reasoning 占满」
            # 与逻辑不符，会把排查方向带偏到 max_tokens/thinking 上（实测故障
            # 实为服务端 JIT 重建期 OOM，见 ADR-0085）
            msg = ("未收到任何输出（正文与思考内容均为空）——非 max_tokens 不足，"
                   "多为服务端故障或上游截断，建议排查推理服务日志")
            # 空流多为间歇性服务端状态故障（Lvllm 混合架构「空 EOS」）：走专属
            # 重试预算（EMPTY_STREAM_RETRY_MAX），不挤占通用瞬时重试；重试
            # 耗尽后走改 shape 回退（ADR-0085），贴边档空流亦可能是服务端被
            # 超长 prompt 压垮，同样走删减回退
            retried = await _retry_empty(msg)
            if retried is not None:
                return retried
            fb = await _edge_fallback(msg)
            if fb is not None:
                return fb
            if _edge_fb:
                # 回退阶梯耗尽仍失败：提示按军备来源区分——贴边档才断言疑似
                # 超限；强制军备档（非贴边空流耗尽）不能谎报超限，否则会把
                # 「服务端按 shape 重建」类故障误导成上下文问题
                msg += (("（临近部署 max_ctx 上限，删减回退后仍失败"
                         "——大概率上下文/显存超限）") if edge_near_limit else
                        ("（非贴边档，同文本重试与改 shape 删减回退均失败"
                         "——疑似服务端故障，建议排查推理服务日志）"))
            if not tick_off:
                await emit_tick({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        # 服务端软超限（fastllm 系）：超长 prompt 不报 4xx，而是 200 + 正文替换为
        # 占位串 "prompt too long"（finish=stop、单 token），content 与
        # reasoning_content 两个通道都可能携带。此时 TTFT/prefill 是「拒绝耗时 ÷
        # prompt 长度」，会冒出数十万 tok/s 的假数据——按错误点收口，与硬超限
        # （_is_ctx_overflow）同口径提示（config.json 配 max_ctx 自动跳档）
        if _is_soft_ctx_overflow(head_text, text_len + reason_len, reason_head):
            msg = (f"服务端将超长 prompt 替换为占位回复（prompt too long），该点数据无效："
                   f"{model} 在 {_fmt_scenario_ctx(scenario, ctx)} 超出服务端上下文上限"
                   f"（config.json deployments 配 max_ctx 可自动跳过超限档位）")
            if not tick_off:
                await emit_tick({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
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
            if not tick_off:
                await emit_tick({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
                                 "ctx": ctx, "conc": conc, "req": req_i, "phase": "error", "msg": msg})
            return {"req": req_i, "err": msg}
        # 物理速率闸及其前置（TTFT/缓存解析/prefill 双口径）先于一切校准/
        # 状态播种：漏网的占位假成功（1 chunk/1 token，cobs=1 在界内）若先走
        # 校准块，会把已按投机网关校准的事件比重置回 1，实时读数成倍虚高
        ttft = (first - t0) if first else None
        # 前缀缓存命中：多字段名解析（DeepSeek/OpenAI/Anthropic 风格）；网关
        # 中转剥离扩展字段时记未回传，agent 矩阵按预热实测 prompt 判别估算
        cache_hit, cache_rep = _cache_hit_from_usage(usage)
        if (usage and not cache_rep and scenario == "agent"
                and not self._cache_note_done and not quiet):
            self._cache_note_done = True
            await self.emit({"type": "status", "msg":
                "网关未回传缓存命中字段（LiteLLM 等中转会剥离上游 usage 扩展字段）；"
                "agent 矩阵未回传时按预热实测 prompt 估算命中（≈）：TTFT 显著低于"
                "同规模全量 prefill 预期 / 相对同指令零缓存档走平 / 本轮已证实缓存"
                "生效——任一成立即估算；仅当 TTFT 与全量 prefill 相符时按未回传留档、"
                "prefill 全量口径；真值需直连推理后端"})
        # prefill 双口径：全量 = prompt_tokens ÷ TTFT；缓存命中时增出未命中口径
        # = (prompt_tokens − hit) ÷ TTFT（TTFT 只覆盖未命中段的计算，全量口径
        # 被命中率注水）。agent 矩阵点级增量口径与先验曲线登记均基于未命中量，
        # req 级字段供口径换算与一致性校验
        prefill_full = (prompt_tokens / ttft) if ttft else None
        prefill_uncached = ((prompt_tokens - cache_hit) / ttft
                            if ttft and cache_rep and cache_hit else None)
        # 物理速率闸：占位软拒绝/拒答式假成功的通用兜底（实测事故：占位回复
        # ÷ 超长 prompt 判出 38 万 tok/s 假 prefill 入先验曲线并污染校准）。
        # 缓存命中时 TTFT 只覆盖未命中段，闸门按未命中口径评估，不误伤高命中率
        # 暖轮；无命中按全量口径。x 同口径取未命中 prompt tokens（先验曲线的
        # 登记横轴，与 gate_rate 同源——旧实现误传 ctx 参数，零缓存档 ctx=0
        # 被 rate_prior 钳到曲线最底端误杀诚实点）。未命中量 <8K 时先验带
        # 无判别力（RATE_GATE_MIN_TOKENS，见函数 docstring），只按绝对上限
        # 判。判假成功即按错误收口——且收口先于下方一切校准/参数名播种，
        # 假速率与假观测都不会入档
        gate_rate = prefill_uncached or prefill_full
        gate_x = ((prompt_tokens - cache_hit) if (cache_rep and cache_hit)
                  else prompt_tokens)
        # 网关未回传命中、但调用方按预热实测断言缓存大概率生效（agent 矩阵缓存
        # 档，ADR-0052）：闸门与命中判别同口径——按预估未命中增量（est_tokens
        # = 指令档 + 缓存块尾）评估；增量 <8K 时先验臂自动跳过、只套绝对上限。
        # 全量口径在缓存加速的短 TTFT 上会误杀诚实点（实测 DeepSeek 33K 缓存档
        # 全量 8580 tok/s 撞 10× 先验线，兄弟点 7~9× 贴线压线通过）
        if not cache_rep and scenario == "agent" and est_tokens and ctx > 0:
            gate_x = est_tokens
            gate_rate = (est_tokens / ttft) if ttft else None
        if gate_rate and _is_implausible_prefill_rate(
                gate_rate, gate_x, self.prefill_curve.get((model, scenario)) or []):
            msg = (f"prefill 速率 {round(gate_rate)} tok/s 物理上不可能"
                   f"（超该上下文先验曲线 {RATE_PRIOR_OVERSHOOT}×，仅未命中量"
                   f" ≥{RATE_GATE_MIN_TOKENS // 1024}K 生效；或绝对上限 "
                   f"{int(RATE_ABS_CAP_TOK_S)} tok/s）——疑似占位软拒绝/拒答式"
                   f"假成功，该点数据无效")
            if not tick_off:
                await emit_tick({"type": "tick", "tag": tag, "model": model, "scenario": scenario,
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
            # 零值守卫：old 为 0/None 时跳过相对漂移比较（除零会崩测量路径）
            if 0.3 <= obs <= 20 and (not old or abs(obs - old) / old > 0.08):
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
        # 停滞回吐（stall-flush）剔除：≥0.5s 停滞间隙后紧随的 token 交付速率
        # 远超健康节奏的回吐段（亚毫秒 1-token 串 / ~8ms 3-token 合并 /
        # ~22ms 26-token 大合并——间隔判据对③原理性漏检，改看每事件 token
        # 速率，ADR-0054）是服务端把停滞期间积压的 tokens 一次性吐出——tokens
        # 既非平稳段产出，又把停滞时间「抵消」进空窗校正分母，净速与
        # tok_per_chunk 双双虚高（实测：34 tok/s 读成 216~333、adj 67.6/133.9
        # 而真实 ~34）。整段剔除：tokens 与停滞/交付时间均不计，净速只反映
        # 平稳生成段
        # 每事件 est_tok：字符增量 ÷ 校准输出系数（out_cpt，与 _est_out 同源；
        # 缺失兜底 1 token/事件），再按 out_tokens 真值归一（scale =
        # out_tokens/Σraw，使 Σest_tok = out_tokens）——flush_tok 由折算估算
        # 升级为实测累计
        ocpt = out_cpt if out_cpt and out_cpt > 0 else None
        raw_est = [(c / ocpt) if ocpt else 1.0 for c in ev_chars]
        raw_sum = sum(raw_est)
        if raw_sum > 0 and out_tokens > 0:
            scale = out_tokens / raw_sum
            ev_toks = [x * scale for x in raw_est]
        else:
            ev_toks = []
        # 停滞回吐剔除：并发批（conc>1）不参与——批内请求共享服务端资源，
        # 逐请求的停滞/回吐只是资源挤占的观测噪声，无判读意义（ADR-0074）；
        # episodes 按空处理（tok_per_chunk 用原始 gaps/chunks），gaps/ev_toks
        # 照常收集（批次聚类与 _decode_peak_rate 时间轴仍需要）
        episodes = _flush_episodes(gaps, ev_toks) if conc == 1 else []
        if episodes:
            flush_tok_f = min(sum(e["burst_tok"] for e in episodes),
                              out_tokens - 1)   # 总剔除量钳制 ≤ out−1（至少留 1 token）
            out_clean = out_tokens - flush_tok_f
            # episode 整段（停滞 + 回吐交付）撤离净窗口：停滞本身不产出、回吐
            # tokens 已从分子剔除，其交付时间留在窗口会摊薄净速（合并回吐串
            # 可达 ~0.5s；亚毫秒串时间 ≈0，与最初口径行为一致）
            adj_clean = adj_time - sum(min(gaps[e["stall_idx"]], DECODE_GAP_CAP)
                                       + e["burst_s"] for e in episodes)
            drop_gaps: set[int] = set()
            for e in episodes:
                # 停滞间隙本身保留（可作批界）；串内间隙 = 回吐事件 start+1..
                # end−1 的到达间隔
                drop_gaps.update(range(e["start_idx"], e["end_idx"] - 1))
            # tok_per_chunk 清洗输入：回吐串间隙剔除、事件数扣回吐事件、
            # tokens 扣回吐实测累计——回吐段（tpc 曾虚高到 3.08~27.6）不再被
            # 当成「投机 verify 批次交付」
            tpc_gaps = [g for gi, g in enumerate(gaps) if gi not in drop_gaps]
            tpc_chunks = n_chunks - sum(e["burst_events"] for e in episodes)
        else:
            flush_tok_f, out_clean = 0.0, out_tokens
            adj_clean, tpc_gaps, tpc_chunks = None, gaps, n_chunks
        # 流末补发权威读数帧：token 数采信服务端 usage、速度为最终窗口均值。
        # 此前 tick 是滑窗估算快照（ADR-0007），与最终值口径不同（ADR-0007，口径收敛），
        # 收尾帧让实时 KPI/实时表停在准确值上
        if not tick_off and decode_time and out_tokens:
            await emit_tick({"type": "tick", "tag": tag, "model": model,
                             "scenario": scenario, "ctx": ctx, "conc": conc, "req": req_i,
                             "phase": "decode", "tokens": round(out_tokens),
                             "speed": round(out_tokens / decode_time, 1),
                             "elapsed": round(t_end - t0, 2),
                             "ttft": round(ttft, 3) if ttft is not None else None,
                             "thinking": bool(reason_len)})
        result = {
            "req": req_i, "err": None,
            "first_abs": first, "last_abs": last,
            # 并发批口径的批起点依据（perf_counter 起点）：_conc_batch_stats
            # 以 min(start_abs) 为批起点
            "start_abs": t0,
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
            # 空窗校正口径：>1s 空窗超出部分封顶扣除；停滞回吐片段（≥0.5s 停滞
            # 后交付速率 ≥3× 健康速率的回吐段）整段剔除——tokens 与停滞/交付
            # 时间均不计，净速只反映平稳生成段；健康段 <16 tokens 或 <0.2s 时
            # 置空（回吐占比过高时净口径无法可靠估计，前端有回退路径；16 的
            # 依据：实测回吐案例健康段仅 ~24 tokens/0.72s，32 会把可用的净读数
            # 全部置空）。conc>1 恒 None：并发批资源挤占下空窗校正无判读意义
            # （ADR-0074）
            "decode_tok_s_adj": (round(out_clean / adj_clean, 1)
                                 if (episodes and adj_clean is not None
                                     and adj_clean >= 0.2 and out_clean >= 16)
                                 else None if episodes
                                 else (round(out_tokens / adj, 1)
                                       if adj and conc == 1 else None)),
            "prompt_tokens": round(prompt_tokens),
            "out_tokens": round(out_tokens),
            "total_s": round(t_end - t0, 2),
            # 停滞诊断（max_gap/stall）：并发批资源挤占下无判读意义，conc>1
            # 恒 None（ADR-0074）
            "max_gap_s": round(max_gap, 2) if (max_gap and conc == 1) else None,
            "stall_s": round(stall_s, 2) if (stall_s and conc == 1) else None,
            "stall_count": (stall_count or None) if conc == 1 else None,
            # 突发交付：疑似网关缓冲冲刷，decode 速率虚高不可信
            "decode_burst": True if _decode_burst(
                n_chunks, decode_time, out_tokens) else None,
            # 平均每批交付 token 数（tok_per_chunk 新口径）：三种交付形态——
            # ① vLLM/MTP 单事件携带 k token（fat 口径直接采信）；② llama.cpp
            # 投机解码逐 token 发事件、同 verify 周期接受的 k 个 token 亚毫秒
            # 成批冲刷（按到达间隔聚类按批计，≈ 接受长度 k）；③ 逐 token 交付
            # 记账噪声（偶发合并，≈1）。仅在有成批交付证据时记录，≈1 噪声不入档
            "tok_per_chunk": _tok_per_batch(tpc_gaps, tpc_chunks, out_clean),
            # usage_real = 服务端确实回传了 prompt_tokens（真值），而非仅有 usage
            # 外壳、prompt_tokens 走字符估算兜底：估算值与同系数推导的靶值自指，
            # 进入构建偏离/prefix_kb/cpt 校准会把估计当真值（永不纠正偏差）
            "usage_real": bool(usage and usage.get("prompt_tokens")),
            # 服务端是否回传了缓存命中字段（agent 矩阵增量口径：有回传按回传值，
            # 无回传按链内差分估算未命中量）
            "cache_reported": cache_rep,
            "sent_chars": sent_chars,   # 实际发送字符数（超窗删减重试后 < 构造量）
            "finish": finished,
            "text_chars": text_len, "reason_chars": reason_len,   # 存档诊断用
            # 输出开头 ≤200 字符采样（存档复盘：前端徽标悬浮/异常日志均消费）：
            # 定位退化输出（~1 字符/token 的塌缩形态）与 echo 改写口径核对
            "text_sample": text_sample, "reason_sample": reason_sample,
            # 输入侧头部采样（同 ≤200 字符口径，ADR-0053）：末条消息文本头部
            # ——agent 矩阵末条即指令（含材料），echo 模式末条为 assistant 预填
            # （续写起点）；删减重试后反映实际发送内容。长上下文本体不采
            # （sent_chars/prompt_tokens 已入档，全文无界）
            "in_sample": ((lambda c: (c if isinstance(c, str) else "".join(
                p.get("text", "") for p in c
                if isinstance(p, dict) and p.get("type") == "text"))[:200])(
                    msgs_sent[-1]["content"] if msgs_sent else "")),
            # token 交付时间轴（瞬态字段，点定稿前剥离）：(首 token 绝对时间,
            # 相邻内容事件间隔全量, 每事件 est tokens)。并发批峰值口径
            # （_decode_peak_rate）消费；正常 req 在点定稿前由 _run_point /
            # _run_agent_matrix 剥离、_strip_in_full 兜底（存档/SSE 不带）
            "_tl": (first, gaps, ev_toks) if ev_toks else None,
            # 输入全文（瞬态字段，仅异常取证）：msgs_sent 的全部消息原文
            # （实际发送内容，含超窗删减重试后的删减版）。正常 req 在点定稿前
            # 由 _strip_in_full 剥离（存档/SSE 不带全文，否则 100K/1M 场景每点
            # 负载爆炸）；仅弃测留痕/留档异常轮抄进 reps_discarded.in_text 与
            # 肇事 req.in_text
            "_in_full": _render_in_full(msgs_sent),
        }
        if _retried or _empty_retried:   # 存档留痕：本请求经兜底/空流重试后成功（前端忽略未知字段）
            result["retried"] = _retried + _empty_retried
        if episodes:   # 停滞回吐剔除留痕：剔除 tokens 折算总量（原始 stall_s 不变）
            result["flush_count"] = len(episodes)
            result["flush_tok"] = round(flush_tok_f)
            # 剔除段交付总时长（含拖尾）：总吞吐口径据此把回吐交付时间撤出窗口
            result["flush_s"] = round(sum(e["burst_s"] for e in episodes), 2)
        return result
