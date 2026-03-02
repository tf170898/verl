#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from benchmark_blackjack_common import (
    MODEL_KEYS,
    calc_rate,
    now_tag,
    parse_verl_train_metrics,
    resolve_blackjack_data,
    run_with_logging,
    write_summary_csv,
    write_summary_md,
)


def _model_list(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_KEYS)
    return [model_arg]


def _preflight_runtime_dependencies() -> None:
    required = ("ray", "transformers", "vllm")
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        return
    missing_csv = ", ".join(missing)
    raise SystemExit(
        "Missing Python package(s): "
        f"{missing_csv}. Install verl with vLLM extras in the same runtime env, e.g. "
        "`python3 -m pip install -e '.[vllm]'` from the verl repo root."
    )


def _find_hf_cache_snapshot(model_id: str) -> Path | None:
    cache_root = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or str(Path.home() / ".cache" / "huggingface" / "hub")
    )
    repo_dir = Path(cache_root) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return None
    cands = [p for p in repo_dir.iterdir() if p.is_dir()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_model_path(model_id: str) -> tuple[str, bool]:
    env_name = {
        "Qwen/Qwen3-1.7B": "VERL_BENCH_QWEN3_1_7B_PATH",
        "Qwen/Qwen3-32B": "VERL_BENCH_QWEN3_32B_PATH",
    }.get(model_id)
    if env_name:
        forced = os.environ.get(env_name)
        if forced:
            forced_path = Path(forced).expanduser()
            if forced_path.exists():
                return str(forced_path.resolve()), True
            print(f"Warning: {env_name} is set but path does not exist: {forced_path}")

    local_cands = [
        Path("/models") / model_id,
        Path("/model") / model_id,
        Path("/verl/models") / model_id,
        Path.home() / "models" / model_id,
    ]
    cache_snapshot = _find_hf_cache_snapshot(model_id)
    if cache_snapshot is not None:
        local_cands.insert(0, cache_snapshot)

    for p in local_cands:
        if p.exists():
            return str(p.resolve()), True
    return model_id, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blackjack benchmark (verl train only).")
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
        "--train-max-samples",
        type=int,
        default=256,
        help="Max training samples consumed by verl trainer from train parquet.",
    )
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=64,
        help="Max validation samples consumed by verl trainer from test parquet.",
    )
    parser.add_argument(
        "--use-env-parquet",
        action="store_true",
        help="Reuse existing parquet found under my_bench/blackjeck_env or my_bench/blackjack_env.",
    )
    parser.add_argument(
        "--attn-impl",
        choices=("sdpa", "eager", "flash_attention_2", "flash_attention_3"),
        default="sdpa",
        help="HF attention implementation for verl model loading.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dir. Default: my_bench/results/verl_train_only_<timestamp>",
    )
    args = parser.parse_args()
    _preflight_runtime_dependencies()

    verl_root = args.verl_root.resolve()
    if args.output_root is None:
        out_root = verl_root / "my_bench" / "results" / f"verl_train_only_{now_tag()}"
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
        prefer_existing_env_parquet=args.use_env_parquet,
    )

    print(f"Using blackjack env dir: {env_dir}")
    print(f"Using train parquet: {train_parquet}")
    print(f"Using test parquet:  {test_parquet}")
    if args.use_env_parquet:
        print("Dataset source:      existing env parquet")
    else:
        print("Dataset source:      generated benchmark parquet")
    print(f"Output root:         {out_root}")
    print(f"Attention impl:      {args.attn_impl}")

    rows: list[dict[str, str]] = []

    for model_key in _model_list(args.model):
        if model_key == "qwen3-1.7b":
            model_id = "Qwen/Qwen3-1.7B"
            steps = 20
            train_bsz = 64
            mini_bsz = 32
            max_len = 256
            roll_tp = 2
            roll_util = 0.70
            actor_offload = "false"
            rollout_n = 2
        elif model_key == "qwen3-32b":
            model_id = "Qwen/Qwen3-32B"
            steps = 12
            train_bsz = 16
            mini_bsz = 8
            max_len = 128
            roll_tp = 4
            roll_util = 0.60
            actor_offload = "true"
            rollout_n = 1
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

        model_path, use_local_model = _resolve_model_path(model_id)
        print(f"Model {model_key}:       {model_path} (local={use_local_model})")

        train_log = logs_dir / f"verl_train_{model_key}.log"
        train_cmd = [
            "python3",
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_parquet}",
            f"data.val_files={test_parquet}",
            f"data.train_max_samples={args.train_max_samples}",
            f"data.val_max_samples={args.val_max_samples}",
            f"data.train_batch_size={train_bsz}",
            f"data.max_prompt_length={max_len}",
            f"data.max_response_length={max_len}",
            "data.filter_overlong_prompts=true",
            "data.truncation=error",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.model.trust_remote_code=true",
            f"+actor_rollout_ref.model.override_config.attn_implementation={args.attn_impl}",
            f"+critic.model.override_config.attn_implementation={args.attn_impl}",
            "actor_rollout_ref.model.use_remove_padding=true",
            "actor_rollout_ref.model.enable_gradient_checkpointing=true",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bsz}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.use_kl_loss=false",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={actor_offload}",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
            "actor_rollout_ref.rollout.name=vllm",
            "++actor_rollout_ref.rollout.free_cache_engine=false",
            "++actor_rollout_ref.rollout.enable_sleep_mode=false",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={roll_tp}",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.logprobs_mode=null",
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
        env["VLLM_USE_V1"] = "1"
        if use_local_model:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        env.setdefault("NCCL_SHM_DISABLE", "1")
        env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

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
            file_logger_jsonl = file_logger_root / "bench_verl_blackjack" / f"verl_train_{model_key}.jsonl"
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
                "extra_metric_name": "na",
                "extra_metric": "na",
                "log_file": str(train_log),
            }
        )

    write_summary_csv(summary_csv, rows)
    write_summary_md(summary_md, rows, title="Verl Blackjack Train-Only Summary")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD:  {summary_md}")


if __name__ == "__main__":
    main()
