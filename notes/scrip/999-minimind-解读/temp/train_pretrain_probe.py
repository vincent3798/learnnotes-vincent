"""模拟 train_pretrain 的训练步：同样的模型/精度/超参，测本机每步耗时与显存。

不读数据集（仓库未附带），用随机 token 代替文本。
loss 单独用「新建模型 + 一次前向」测，避免固定数据被过拟合后干扰读数。

运行：
    D:\\conda_envs\\swift_env\\python.exe train_pretrain_probe.py
"""
import sys
import time

PROJECT = r"E:\learnnotes-vincent\notes\scrip\999-minimind-解读\minimind-master"
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

SEQ = 340


def initial_loss() -> None:
    torch.manual_seed(0)
    cfg = MiniMindConfig()
    model = MiniMindForCausalLM(cfg).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (2, SEQ), device="cuda")
    with torch.inference_mode():
        loss = model(ids, labels=ids).loss.item()
    print(f"随机权重 + 随机 token 的初始 loss = {loss:.3f}  "
          f"(ln(vocab) = {torch.log(torch.tensor(float(cfg.vocab_size))).item():.3f})")


def bench(batch: int, dtype_name: str, iters: int = 8) -> None:
    torch.manual_seed(0)
    cfg = MiniMindConfig()
    model = MiniMindForCausalLM(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype_name, torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_name == "fp16"))
    autocast = torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype_name != "fp32"))

    ids = torch.randint(0, cfg.vocab_size, (batch, SEQ), device="cuda")
    labels = ids.clone()
    labels[:, -20:] = -100

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast:
            res = model(ids, labels=labels)
            loss = res.loss + res.aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"  batch={batch:2d} {dtype_name:5s} | {ms:8.1f} ms/step | 峰值 {peak:7.1f} MB")

    del model, opt, ids, labels
    torch.cuda.empty_cache()


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("模型: MiniMind-3 Dense 63.91M, seq=340（pretrain 默认）, AdamW lr=5e-4\n")
    initial_loss()
    print()
    for batch in (4, 8, 16):
        for dt in ("fp16", "fp32"):
            try:
                bench(batch, dt)
            except RuntimeError as e:
                print(f"  batch={batch:2d} {dt:5s} | OOM: {str(e)[:60]}")


if __name__ == "__main__":
    main()
