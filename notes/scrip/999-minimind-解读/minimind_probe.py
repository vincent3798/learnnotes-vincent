"""在 swift_env 里验证 MiniMind 模型代码（只 import 模型文件，不跑训练脚本）。

运行：
    D:\\conda_envs\\swift_env\\python.exe minimind_probe.py
"""
import sys
import time

PROJECT = r"E:\learnnotes-vincent\notes\scrip\999-minimind-解读\minimind-master"
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def count(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def main() -> None:
    torch.manual_seed(0)

    for use_moe in (False, True):
        cfg = MiniMindConfig(use_moe=use_moe)
        model = MiniMindForCausalLM(cfg).eval()
        tag = "MoE(4 experts/top-1)" if use_moe else "Dense"
        print(f"[{tag}] params={count(model):.2f}M  vocab={cfg.vocab_size} "
              f"layers={cfg.num_hidden_layers} dim={cfg.hidden_size} "
              f"head_dim={cfg.head_dim} kv_heads={cfg.num_key_value_heads} "
              f"ffn={cfg.intermediate_size}")

        ids = torch.randint(0, cfg.vocab_size, (2, 32))
        with torch.inference_mode():
            out = model(ids)
        print(f"  forward: input {tuple(ids.shape)} -> logits {tuple(out.logits.shape)}, "
              f"aux_loss={float(out.aux_loss):.4f}")

        loss = model(ids, labels=ids).loss
        print(f"  loss(labels=ids, 随机权重): {loss.item():.3f}  (ln(vocab)={torch.log(torch.tensor(float(cfg.vocab_size))).item():.3f})")

        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(ids, max_new_tokens=8, do_sample=False)
        print(f"  generate 8 tokens (CPU): {(time.perf_counter() - t0) * 1000:.0f} ms")

    print("\n== 显存/内存估算（64M 全量训练）==")
    p = 64e6
    print(f"  fp16 权重         : {p * 2 / 1024**2:.0f} MB")
    print(f"  训练(AdamW,16B/参): {p * 16 / 1024**3:.2f} GB（不含激活）")

    if torch.cuda.is_available():
        cfg = MiniMindConfig()
        model = MiniMindForCausalLM(cfg).half().cuda().eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 32), device="cuda")
        with torch.inference_mode():
            t0 = time.perf_counter()
            model.generate(ids, max_new_tokens=32, do_sample=False)
            dt = time.perf_counter() - t0
        print(f"\nGPU(fp16) 生成 32 tokens: {dt * 1000:.0f} ms "
              f"({32 / dt:.1f} tok/s), 峰值 {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
