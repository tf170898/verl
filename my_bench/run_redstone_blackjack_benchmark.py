#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_blackjack_common import (
    MODEL_KEYS,
    calc_rate,
    now_tag,
    parse_redstone_infer_reward,
    parse_redstone_train_metrics,
    resolve_blackjack_data,
    run_with_logging,
    write_summary_csv,
    write_summary_md,
)


def _model_list(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_KEYS)
    return [model_arg]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blackjack benchmark for redstone only.")
    parser.add_argument(
        "--verl-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to verl repo root (for blackjack env/data discovery).",
    )
    parser.add_argument(
        "--redstone-root",
        type=Path,
        default=Path.home() / "Downloads" / "redstone",
        help="Path to redstone repo root.",
    )
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_KEYS),
        default="all",
        help="Model key to run. Use 'all' for both.",
    )
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Benchmark output dir. Default: my_bench/results/redstone_<timestamp>",
    )
    args = parser.parse_args()

    verl_root = args.verl_root.resolve()
    redstone_root = args.redstone_root.resolve()
    if args.output_root is None:
        out_root = verl_root / "my_bench" / "results" / f"redstone_{now_tag()}"
    else:
        out_root = args.output_root.resolve()

    logs_dir = out_root / "logs"
    data_dir = out_root / "data"
    summary_csv = out_root / "summary.csv"
    summary_md = out_root / "summary.md"
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Resolve blackjack dataset once for parity with verl benchmarks.
    train_parquet, test_parquet, env_dir = resolve_blackjack_data(
        verl_root=verl_root,
        out_data_dir=data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
    print(f"Using blackjack env dir: {env_dir}")
    print(f"Reference train parquet: {train_parquet}")
    print(f"Reference test parquet:  {test_parquet}")
    print(f"Output root:             {out_root}")

    rows: list[dict[str, str]] = []

    for model_key in _model_list(args.model):
        if model_key == "qwen3-1.7b":
            model_path = "Qwen/Qwen3-1.7B"
            steps = 20
            local_bsz = 4
            seq_len = 4096
            group_size = 16
            rollout_steps = 4
            dp_repl = 4
            trainer_tp = 1
            rollout_gpus = 3
            rollout_tp = 1
            val_gpus = 1
            val_tp = 1
            compile_enable = "true"
            infer_batch = 1024
            infer_tp = 1
            infer_new_tokens = 16
            infer_episode_size = 4
        elif model_key == "qwen3-32b":
            model_path = "Qwen/Qwen3-32B"
            steps = 12
            local_bsz = 1
            seq_len = 2048
            group_size = 8
            rollout_steps = 2
            dp_repl = 1
            trainer_tp = 4
            rollout_gpus = 2
            rollout_tp = 2
            val_gpus = 2
            val_tp = 2
            compile_enable = "false"
            infer_batch = 256
            infer_tp = 4
            infer_new_tokens = 8
            infer_episode_size = 4
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

        run_dir = out_root / "runs" / f"redstone_{model_key}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_log = logs_dir / f"redstone_train_{model_key}.log"
        train_cmd = [
            "uv",
            "run",
            "--env-file",
            ".env.train",
            "--active",
            "python",
            "exps/blackjack-ppo-lora/train.py",
            "--job.config-file",
            "exps/blackjack-ppo-lora/configs/base.toml",
            "--job.dump-folder",
            str(run_dir),
            "--model-download.name",
            model_path,
            "--training.steps",
            str(steps),
            "--training.local-batch-size",
            str(local_bsz),
            "--training.seq-len",
            str(seq_len),
            "--grpo.group-size",
            str(group_size),
            "--grpo.num-rollout-steps",
            str(rollout_steps),
            "--env-worker.num-workers",
            "4",
            "--rollout.sampling.max-new-tokens",
            "8",
            "--policy-validation.freq",
            "100000",
            "--checkpoint.interval",
            str(steps),
            "--parallelism.data-parallel-replicate-degree",
            str(dp_repl),
            "--parallelism.tensor-parallel-degree",
            str(trainer_tp),
            "--rollout.num-gpus",
            str(rollout_gpus),
            "--rollout.parallel.tensor-parallel-degree",
            str(rollout_tp),
            "--policy-validation.rollout.num-gpus",
            str(val_gpus),
            "--policy-validation.rollout.parallel.tensor-parallel-degree",
            str(val_tp),
            "--compile.enable",
            compile_enable,
        ]
        print(f"==> redstone train ({model_key})")
        train_ret, train_elapsed = run_with_logging(
            cwd=redstone_root,
            cmd=train_cmd,
            log_file=train_log,
            env=None,
        )
        train_status = "ok" if train_ret == 0 else "fail"
        train_throughput = f"{calc_rate(steps, train_elapsed):.3f}" if train_ret == 0 else "0.000"
        train_step, train_loss, train_reward = ("na", "na", "na")
        if train_ret == 0:
            train_step, train_loss, train_reward = parse_redstone_train_metrics(run_dir / "metric" / "trainer.jsonl")

        rows.append(
            {
                "framework": "redstone",
                "phase": "train",
                "model": model_key,
                "status": train_status,
                "elapsed_sec": f"{train_elapsed:.3f}",
                "throughput": train_throughput,
                "unit": "steps/s",
                "train_step": train_step,
                "train_loss": train_loss,
                "train_reward": train_reward,
                "extra_metric": "na",
                "log_file": str(train_log),
            }
        )

        infer_log = logs_dir / f"redstone_infer_{model_key}.log"
        infer_cmd = [
            "uv",
            "run",
            "--env-file",
            ".env.train",
            "--active",
            "python",
            "exps/blackjack-ppo-lora/infer.py",
            "--job.config-file",
            "exps/blackjack-ppo-lora/configs/base.toml",
            "--job.dump-folder",
            str(run_dir),
            "--model-download.name",
            model_path,
            "--policy-validation.batch-size",
            str(infer_batch),
            "--policy-validation.rollout.num-gpus",
            "8",
            "--policy-validation.rollout.parallel.tensor-parallel-degree",
            str(infer_tp),
            "--policy-validation.rollout.max-episode-size",
            str(infer_episode_size),
            "--policy-validation.rollout.sampling.max-new-tokens",
            str(infer_new_tokens),
        ]
        print(f"==> redstone infer ({model_key})")
        infer_ret, infer_elapsed = run_with_logging(
            cwd=redstone_root,
            cmd=infer_cmd,
            log_file=infer_log,
            env=None,
        )
        infer_status = "ok" if infer_ret == 0 else "fail"
        infer_throughput = f"{calc_rate(infer_batch, infer_elapsed):.3f}" if infer_ret == 0 else "0.000"
        infer_reward = parse_redstone_infer_reward(infer_log) if infer_ret == 0 else "na"

        rows.append(
            {
                "framework": "redstone",
                "phase": "infer",
                "model": model_key,
                "status": infer_status,
                "elapsed_sec": f"{infer_elapsed:.3f}",
                "throughput": infer_throughput,
                "unit": "traj/s",
                "train_step": "na",
                "train_loss": "na",
                "train_reward": "na",
                "extra_metric": infer_reward,
                "log_file": str(infer_log),
            }
        )

    write_summary_csv(summary_csv, rows)
    write_summary_md(summary_md, rows, title="Redstone Blackjack Benchmark Summary")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD:  {summary_md}")


if __name__ == "__main__":
    main()

