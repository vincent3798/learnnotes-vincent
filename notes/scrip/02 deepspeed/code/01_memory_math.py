"""第 1 章配套：训练显存算账器（不依赖 deepspeed，本地可跑）。

给定模型参数量、ZeRO 阶段、GPU 数量，算出每张卡要放多少显存。

运行：
    D:\\conda_envs\\swift_env\\python.exe code\\01_memory_math.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 每个参数占用的字节数（bf16/fp16 混合精度 + AdamW）
# params 2 + grads 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16
PARAM_B, GRAD_B, MASTER_B, M_B, V_B = 2, 2, 4, 4, 4
OPT_B = MASTER_B + M_B + V_B          # 12


def per_gpu_gb(n_params: float, stage: int, n_gpus: int) -> float:
    """返回每张卡的非激活显存（GB）。"""
    if stage == 0:
        b = PARAM_B + GRAD_B + OPT_B
    elif stage == 1:
        b = PARAM_B + GRAD_B + OPT_B / n_gpus
    elif stage == 2:
        b = PARAM_B + (GRAD_B + OPT_B) / n_gpus
    elif stage == 3:
        b = (PARAM_B + GRAD_B + OPT_B) / n_gpus
    else:
        raise ValueError(stage)
    return n_params * b / 1024 ** 3


def main() -> None:
    models = [
        ("0.5B", 0.5e9), ("1.5B", 1.5e9), ("7B", 7e9),
        ("13B", 13e9), ("70B", 70e9),
    ]

    print("=" * 74)
    print("每张卡的非激活显存（GB）｜bf16 混合精度 + AdamW｜不含激活值")
    print("=" * 74)
    print(f"{'模型':>6}  {'ZeRO-0':>9} {'ZeRO-1':>9} {'ZeRO-2':>9} {'ZeRO-3':>9}"
          f"  {'8卡 Z2':>9} {'8卡 Z3':>9}")
    for name, p in models:
        row = [per_gpu_gb(p, s, 1) for s in (0, 1, 2, 3)]
        row2 = per_gpu_gb(p, 2, 8)
        row3 = per_gpu_gb(p, 3, 8)
        print(f"{name:>6}  {row[0]:>8.1f}G {row[1]:>8.1f}G {row[2]:>8.1f}G "
              f"{row[3]:>8.1f}G  {row2:>8.1f}G {row3:>8.1f}G")

    print("\n单卡上 ZeRO-1/2/3 与非 ZeRO 完全相同（无处可分片）。")
    print("多卡时：ZeRO-1 只分优化器状态，ZeRO-2 再分梯度，ZeRO-3 连参数也分。")

    print("\n" + "=" * 74)
    print("反过来问：4 GB 卡全量训练（AdamW）最多多大模型？")
    print("=" * 74)
    budget_gb = 3.4                      # 扣掉系统与 CUDA 上下文
    bytes_per_param = PARAM_B + GRAD_B + OPT_B   # 单卡时分片无效，恒为 16
    max_params = budget_gb * 1024 ** 3 / bytes_per_param
    print(f"  ZeRO-0/2/3 结果相同: 约 {max_params / 1e9:.2f}B 参数（不含激活）")
    print("  原因：只有 1 张卡，没有任何东西可以分片。")
    print("\n注：这是理论下限。实际还要加激活值、临时张量和 KV cache，")
    print("    能跑通的上限通常要再打五折。")

    print("\n" + "=" * 74)
    print("示例：7B 模型在 8 卡上，ZeRO 节省了多少？")
    print("=" * 74)
    base = per_gpu_gb(7e9, 0, 8)
    for stage in (0, 1, 2, 3):
        g = per_gpu_gb(7e9, stage, 8)
        print(f"  ZeRO-{stage}: {g:6.1f} GB/卡  (相对 ZeRO-0 省 {(1 - g / base) * 100:4.0f}%)")


if __name__ == "__main__":
    main()
