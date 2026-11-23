"""第 2 章配套：ZeRO 分片模拟 + 通信量对照（不依赖 deepspeed）。

运行：
    D:\\conda_envs\\swift_env\\python.exe code\\02_zero_sim.py
"""
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

N_PARAMS = 7e9
N_GPUS = 8


def fmt(n_bytes: float) -> str:
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:7.2f} GB"
    return f"{n_bytes / 1024 ** 2:7.1f} MB"


def zero_layout(stage: int, n_gpus: int) -> dict:
    """每种 ZeRO 阶段下，各个状态是否被分片。"""
    return {
        "参数": stage == 3,
        "梯度": stage >= 2,
        "优化器状态": stage >= 1,
    }


def main() -> None:
    print(f"模型 {N_PARAMS / 1e9:.0f}B 参数，{N_GPUS} 张卡，bf16 + AdamW\n")

    print("== 1. 各阶段分片什么 ==")
    print(f"{'阶段':>6} {'参数':>8} {'梯度':>8} {'优化器状态':>12}   每卡总量")
    for stage in (0, 1, 2, 3):
        lay = zero_layout(stage, N_GPUS)
        total = 0.0
        for name, nbytes in (("参数", 2), ("梯度", 2), ("优化器状态", 12)):
            total += N_PARAMS * nbytes * (1 / N_GPUS if lay[name] else 1)
        mark = lambda b: "分片" if b else "完整"
        print(f"{'ZeRO-' + str(stage):>6} {mark(lay['参数']):>8} {mark(lay['梯度']):>8} "
              f"{mark(lay['优化器状态']):>12}   {fmt(total)}")

    print("\n== 2. 用真实张量演示「分片」就是切块 ==")
    t = torch.arange(100_000_000, dtype=torch.float32)   # 400 MB 的模拟参数
    print(f"  总量: {t.numel():,} 个元素 = {fmt(t.numel() * 4)}")
    for i, shard in enumerate(t.chunk(N_GPUS)):
        if i < 2:
            print(f"  卡{i} 拿到 {shard.numel():,} 个元素，首尾 {shard[0].item()} ~ {shard[-1].item()}")
    print("  所有卡的元素合起来正好是原来的张量 —— ZeRO-3 的参数分片就是这个意思")

    print("\n== 3. 每步通信量（以参数量 P 为单位）==")
    comm = {0: 2, 1: 2, 2: 2, 3: 3}
    print(f"{'阶段':>6} {'通信量':>8} {'按 7B bf16 估算':>20}   说明")
    notes = {
        0: "梯度 all-reduce（2P）",
        1: "reduce-scatter + all-gather（2P）",
        2: "同 ZeRO-1，但省更多显存（2P）",
        3: "再多一次参数 all-gather（3P）",
    }
    for stage, p in comm.items():
        print(f"{'ZeRO-' + str(stage):>6} {p}P{'':>6} "
              f"{fmt(p * N_PARAMS * 2):>20}   {notes[stage]}")

    print("\n  结论：ZeRO-3 显存最省，但每步多传 50% 的数据；")
    print("        显存够用时优先 ZeRO-2，这是工业界的默认选择。")

    print("\n== 4. 显存换来的收益：能不能跑 ==")
    for stage in (0, 2, 3):
        lay = zero_layout(stage, N_GPUS)
        per_card = 0.0
        for name, nbytes in (("参数", 2), ("梯度", 2), ("优化器状态", 12)):
            per_card += N_PARAMS * nbytes * (1 / N_GPUS if lay[name] else 1)
        fits = "24GB 卡 OK" if per_card < 22 * 1024 ** 3 else "24GB 卡放不下"
        print(f"  ZeRO-{stage}: {fmt(per_card)}/卡  ->  {fits}")


if __name__ == "__main__":
    main()
