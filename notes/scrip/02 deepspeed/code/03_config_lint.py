"""第 5 章配套：ds_config.json 体检器（不依赖 deepspeed，本地可跑）。

用法：
    python code/03_config_lint.py configs/ds_zero2.json [--world_size 8]
"""
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

KNOWN_TOP = {
    "train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps",
    "gradient_clipping", "steps_per_print", "wall_clock_breakdown", "fp16", "bf16",
    "optimizer", "scheduler", "zero_optimization", "activation_checkpointing",
    "zero_allow_untested_optimizer", "communication_data_type", "prescale_gradients",
    "gradient_predivide_factor", "sparse_gradients", "ignore_unused_parameters",
    "dump_state", "elasticity", "flops_profiler", "compression", "data_types",
}


def lint(cfg: dict, world_size: int) -> list[str]:
    problems: list[str] = []
    z = cfg.get("zero_optimization", {})
    stage = z.get("stage", 0)

    micro = cfg.get("train_micro_batch_size_per_gpu")
    gas = cfg.get("gradient_accumulation_steps", 1)
    total = cfg.get("train_batch_size")
    if total is not None and micro is not None:
        expect = micro * gas * world_size
        if total != expect:
            problems.append(
                f"train_batch_size={total}，但 micro({micro}) × gas({gas}) × "
                f"world_size({world_size}) = {expect}，DeepSpeed 会报错")

    if cfg.get("fp16", {}).get("enabled") and cfg.get("bf16", {}).get("enabled"):
        problems.append("fp16 与 bf16 同时开启——只能选一个（Ampere 及以上推荐 bf16）")

    if stage not in (0, 1, 2, 3):
        problems.append(f"zero_optimization.stage={stage} 不合法，只能是 0/1/2/3")

    if "offload_param" in z and stage != 3:
        problems.append("offload_param 只在 ZeRO-3 下生效，当前 stage=" + str(stage))

    if "offload_optimizer" in z and stage == 0:
        problems.append("offload_optimizer 需要 ZeRO-1/2/3，当前 stage=0")

    for key in ("offload_param", "offload_optimizer"):
        dev = z.get(key, {}).get("device")
        if dev == "nvme" and not z.get(key, {}).get("nvme_path"):
            problems.append(f"{key}.device=nvme 但没有配置 nvme_path")

    unknown = set(cfg) - KNOWN_TOP
    if unknown:
        problems.append(f"未识别的顶层字段（可能拼错）: {sorted(unknown)}")

    return problems


def describe(cfg: dict, world_size: int) -> None:
    z = cfg.get("zero_optimization", {})
    stage = z.get("stage", 0)
    prec = "bf16" if cfg.get("bf16", {}).get("enabled") else (
        "fp16" if cfg.get("fp16", {}).get("enabled") else "fp32")
    micro = cfg.get("train_micro_batch_size_per_gpu")
    gas = cfg.get("gradient_accumulation_steps", 1)

    print(f"  ZeRO 阶段      : {stage}")
    print(f"  精度           : {prec}")
    print(f"  微批 × 累积    : {micro} × {gas}")
    print(f"  卡数           : {world_size}")
    if micro:
        print(f"  等效总 batch   : {micro * gas * world_size}")
    print(f"  优化器         : {cfg.get('optimizer', {}).get('type', '未配置')}")
    print(f"  调度器         : {cfg.get('scheduler', {}).get('type', '未配置')}")
    if "offload_optimizer" in z:
        print(f"  优化器 offload : {z['offload_optimizer'].get('device')}")
    if "offload_param" in z:
        print(f"  参数 offload   : {z['offload_param'].get('device')}")
    if cfg.get("activation_checkpointing"):
        print("  激活检查点      : 已开启")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--world_size", type=int, default=8)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"== {args.config} ==")
    describe(cfg, args.world_size)

    problems = lint(cfg, args.world_size)
    print("\n体检结果:")
    if not problems:
        print("  ✅ 未发现问题")
    for p in problems:
        print(f"  ❌ {p}")


if __name__ == "__main__":
    main()
