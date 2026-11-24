"""实测 trainer_utils.py 里的工具函数（不训练，只调函数）。

运行：
    D:\\conda_envs\\swift_env\\python.exe trainer_utils_probe.py
"""
import sys

PROJECT = r"E:\learnnotes-vincent\notes\scrip\999-minimind-解读\minimind-master"
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

import torch
from trainer.trainer_utils import (
    get_lr, SkipBatchSampler, safe_math_eval, rep_penalty,
    resolve_dtype, get_model_params,
)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def section(t):
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


def main():
    section("1. get_lr：warmup=0 时的余弦曲线")
    total = 1000
    for s in (0, 1, 250, 500, 750, 999, 1000):
        print(f"  step {s:4d}: lr = {get_lr(s, total, 1e-3):.6f}")

    section("2. get_lr：warmup_ratio=0.1（前 100 步线性升）")
    for s in (0, 50, 99, 100, 500, 1000):
        print(f"  step {s:4d}: lr = {get_lr(s, total, 1e-3, warmup_ratio=0.1):.6f}")

    section("3. SkipBatchSampler：跳过前 2 个 batch")
    s = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=2)
    print(f"  __len__ = {len(s)}")
    print(f"  实际产出 = {list(s)}")

    section("4. safe_math_eval：正常与异常")
    cases = ["2^10", "（1+2）×3", "sqrt(16)", "math.log(e)", "2**-3",
             "10**50", "9**9**9", "1/0", "__import__('os')", "2+"]
    for c in cases:
        try:
            print(f"  {c!r:24s} -> {safe_math_eval(c)}")
        except Exception as e:
            print(f"  {c!r:24s} -> {type(e).__name__}: {e}")

    section("5. rep_penalty：重复度打分")
    texts = [
        "今天天气很好，今天天气很好，今天天气很好",
        "今天天气很好，我们去公园散步吧",
        "好的",
    ]
    for t in texts:
        print(f"  {rep_penalty(t):.4f}  {t}")

    section("6. resolve_dtype：你的 1650 上会发生什么")
    print(f"  bfloat16 -> {resolve_dtype('bfloat16', 'cuda')}")
    print(f"  float16  -> {resolve_dtype('float16', 'cuda')}")

    section("7. get_model_params：Dense 与 MoE 的打印")
    dense = MiniMindForCausalLM(MiniMindConfig())
    get_model_params(dense, dense.config)
    moe = MiniMindForCausalLM(MiniMindConfig(use_moe=True))
    get_model_params(moe, moe.config)


if __name__ == "__main__":
    main()
