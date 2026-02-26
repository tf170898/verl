#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from benchmark_blackjack_common import (
    INFER_PROMPTS_TARGET,
    MODEL_KEYS,
    calc_rate,
    now_tag,
    parse_verl_infer_accuracy,
    parse_verl_train_metrics,
    prepare_infer_subset,
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
    parser = argparse.ArgumentParser(description="Run blackjack benchmark for verl only.")
    parser.add_argument(
        "--verl-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to verl repo root.",
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
        help="Benchmark output dir. Default: my_bench/results/verl_<timestamp>",
    )
    args = parser.parse_args()

    verl_root = args.verl_root.resolve()
    if args.output_root is None:
        out_root = verl_root / "my_bench" / "results" / f"verl_{now_tag()}"
    else:
        out_root = args.output_root.resolve()

    logs_dir = out_root / "logs"
    data_dir = out_root / "data"
    metrics_dir = out_root / "metrics"
    file_logger_root = metrics_dir / "verl_file_logger"
    reward_fn_path = (Path(__file__).resolve().parent / "blackjack_reward_fn.py").resolve()
    summary_csv = out_root / "summary.csv"
    summary_md = out_root / "summary.md"

    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    file_logger_root.mkdir(parents=True, exist_ok=True)

    train_parquet, test_parquet, env_dir = resolve_blackjack_data(
        verl_root=verl_root,
        out_data_dir=data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
    infer_subset = data_dir / f"blackjack_test_{INFER_PROMPTS_TARGET}.parquet"
    infer_prompts = prepare_infer_subset(test_parquet, infer_subset, INFER_PROMPTS_TARGET)

    print(f"Using blackjack env dir: {env_dir}")
    print(f"Using train parquet: {train_parquet}")
    print(f"Using test parquet:  {test_parquet}")
    print(f"Output root:         {out_root}")

    rows: list[dict[str, str]] = []

    for model_key in _model_list(args.model):
        if model_key == "qwen3-1.7b":
            model_path = "Qwen/Qwen3-1.7B"
            steps = 20
            train_bsz = 64
            mini_bsz = 32
            max_len = 256
            roll_tp = 2
            roll_util = 0.70
            actor_offload = "false"
            rollout_n = 2
            infer_tp = 1
            infer_util = 0.85
            infer_resp_len = 256
        elif model_key == "qwen3-32b":
            model_path = "Qwen/Qwen3-32B"
            steps = 12
            train_bsz = 16
            mini_bsz = 8
            max_len = 128
            roll_tp = 4
            roll_util = 0.60
            actor_offload = "true"
            rollout_n = 1
            infer_tp = 4
            infer_util = 0.70
            infer_resp_len = 128
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

        train_log = logs_dir / f"verl_train_{model_key}.log"
        train_cmd = [
            "python3",
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_parquet}",
            f"data.val_files={test_parquet}",
            "data.train_max_samples=256",
            "data.val_max_samples=64",
            f"data.train_batch_size={train_bsz}",
            f"data.max_prompt_length={max_len}",
            f"data.max_response_length={max_len}",
            "data.filter_overlong_prompts=true",
            "data.truncation=error",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.model.trust_remote_code=true",
            "actor_rollout_ref.model.use_remove_padding=true",
            "actor_rollout_ref.model.enable_gradient_checkpointing=true",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bsz}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.use_kl_loss=false",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={actor_offload}",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={roll_tp}",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={roll_util}",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            "algorithm.use_kl_in_reward=false",
            f"reward.custom_reward_function.path={reward_fn_path}",
            "reward.custom_reward_function.name=bench_reward",
            "trainer.critic_warmup=0",
            'trainer.logger=["console","file"]',
            "trainer.project_name=bench_verl_blackjack",
            f"trainer.experiment_name=verl_train_{model_key}",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=8",
            "trainer.save_freq=-1",
            "trainer.test_freq=100000",
            "trainer.total_epochs=1",
            f"trainer.total_training_steps={steps}",
        ]
        env = os.environ.copy()
        env["VERL_FILE_LOGGER_ROOT"] = str(file_logger_root)
        print(f"==> verl train ({model_key})")
        train_ret, train_elapsed = run_with_logging(
            cwd=verl_root,
            cmd=train_cmd,
            log_file=train_log,
            env=env,
        )
        train_status = "ok" if train_ret == 0 else "fail"
        train_throughput = f"{calc_rate(steps, train_elapsed):.3f}" if train_ret == 0 else "0.000"
        train_step, train_loss, train_reward = ("na", "na", "na")
        if train_ret == 0:
            file_logger_jsonl = (
                file_logger_root
                / "bench_verl_blackjack"
                / f"verl_train_{model_key}.jsonl"
            )
            train_step, train_loss, train_reward = parse_verl_train_metrics(file_logger_jsonl)

        rows.append(
            {
                "framework": "verl",
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

        infer_out = out_root / f"verl_infer_{model_key}.parquet"
        infer_log = logs_dir / f"verl_infer_{model_key}.log"
        infer_cmd = [
            "python3",
            "-m",
            "verl.trainer.main_generation_server",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=8",
            f"data.train_files={infer_subset}",
            "data.prompt_key=prompt",
            f"data.output_path={infer_out}",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.model.trust_remote_code=true",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={infer_tp}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={infer_util}",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.rollout.temperature=0.0",
            "actor_rollout_ref.rollout.top_p=1.0",
            f"actor_rollout_ref.rollout.response_length={infer_resp_len}",
        ]
        print(f"==> verl infer ({model_key})")
        infer_ret, infer_elapsed = run_with_logging(
            cwd=verl_root,
            cmd=infer_cmd,
            log_file=infer_log,
            env=env,
        )
        infer_status = "ok" if infer_ret == 0 else "fail"
        infer_throughput = f"{calc_rate(infer_prompts, infer_elapsed):.3f}" if infer_ret == 0 else "0.000"
        infer_acc = parse_verl_infer_accuracy(infer_out) if infer_ret == 0 else "na"

        rows.append(
            {
                "framework": "verl",
                "phase": "infer",
                "model": model_key,
                "status": infer_status,
                "elapsed_sec": f"{infer_elapsed:.3f}",
                "throughput": infer_throughput,
                "unit": "req/s",
                "train_step": "na",
                "train_loss": "na",
                "train_reward": "na",
                "extra_metric": infer_acc,
                "log_file": str(infer_log),
            }
        )

    write_summary_csv(summary_csv, rows)
    write_summary_md(summary_md, rows, title="Verl Blackjack Benchmark Summary")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD:  {summary_md}")


if __name__ == "__main__":
    main()

