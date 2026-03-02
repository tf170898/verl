#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from packaging import version

from benchmark_blackjack_common import (
    check_runtime_shared_memory,
    INFER_PROMPTS_TARGET,
    MODEL_KEYS,
    calc_rate,
    now_tag,
    parse_verl_infer_accuracy,
    parse_verl_train_metrics,
    prepare_infer_subset,
    resolve_blackjack_data,
    run_with_logging,
    validate_otel_histogram_api,
    write_summary_csv,
    write_summary_md,
)


def _model_list(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_KEYS)
    return [model_arg]


def _resolve_rollout_tp(desired_tp: int, n_gpus: int) -> int:
    if n_gpus <= 0:
        return 1
    cap = max(1, min(desired_tp, n_gpus))
    for tp in range(cap, 0, -1):
        if n_gpus % tp == 0:
            return tp
    return 1


def _adjust_train_batch_size(train_bsz: int, rollout_n: int, n_gpus: int) -> int:
    if n_gpus <= 0:
        return train_bsz
    bsz = max(1, train_bsz)
    while (bsz * rollout_n) % n_gpus != 0:
        bsz += 1
    return bsz


def _vllm_supports_bench_engine_overrides() -> bool:
    try:
        from vllm import __version__ as vllm_version
    except Exception:
        return False
    return version.parse(vllm_version) >= version.parse("0.13.0")


def _preflight_runtime_dependencies() -> None:
    required = ("ray", "transformers", "vllm")
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        # vLLM rollout weight sync depends on AsyncLLM.collective_rpc.
        try:
            from vllm import __version__ as vllm_version
            from vllm.v1.engine.async_llm import AsyncLLM
        except Exception as e:
            raise SystemExit(f"Failed to import vLLM runtime APIs: {type(e).__name__}: {e}") from e
        if not hasattr(AsyncLLM, "collective_rpc"):
            raise SystemExit(
                "Incompatible vLLM build detected: AsyncLLM.collective_rpc is missing, "
                "but verl vLLM rollout requires it for weight sync. "
                f"Detected vllm version: {vllm_version}. "
                "Please install a compatible vLLM version (for this repo, setup.py expects <=0.12.0)."
            )
        otel_err = validate_otel_histogram_api()
        if otel_err:
            raise SystemExit(otel_err)
        shm_warn = check_runtime_shared_memory()
        if shm_warn:
            print(f"Warning: {shm_warn}")
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
        help="HF attention implementation for verl model loading. Default avoids flash-attn ABI issues.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Benchmark output dir. Default: my_bench/results/verl_<timestamp>",
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=8,
        help="Number of GPUs used by trainer on one node.",
    )
    parser.add_argument(
        "--rollout-agent-workers",
        type=int,
        default=8,
        help="Number of rollout agent workers.",
    )
    args = parser.parse_args()
    _preflight_runtime_dependencies()
    vllm_supports_bench_engine_overrides = _vllm_supports_bench_engine_overrides()
    if not vllm_supports_bench_engine_overrides:
        print(
            "Info: vLLM < 0.13 detected; forcing safer rollout backend settings "
            "(distributed_executor_backend=uni, V1 multiprocessing off)."
        )

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
    bench_dir = Path(__file__).resolve().parent
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
    infer_subset = data_dir / f"blackjack_test_{INFER_PROMPTS_TARGET}.parquet"
    infer_prompts = prepare_infer_subset(test_parquet, infer_subset, INFER_PROMPTS_TARGET)

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
            # Keep TP=1 for 1.7B to avoid extra intra-engine distributed rendezvous fragility.
            roll_tp = 1
            roll_util = 0.40
            actor_offload = "false"
            optimizer_offload = "false"
            rollout_n = 1
            infer_tp = 1
            infer_util = 0.85
            infer_resp_len = 256
        elif model_key == "qwen3-32b":
            model_id = "Qwen/Qwen3-32B"
            steps = 12
            train_bsz = 16
            mini_bsz = 8
            max_len = 128
            roll_tp = 4
            roll_util = 0.60
            actor_offload = "true"
            optimizer_offload = "false"
            rollout_n = 1
            infer_tp = 4
            infer_util = 0.70
            infer_resp_len = 128
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

        single_gpu_vllm_safe = args.n_gpus_per_node == 1
        if single_gpu_vllm_safe:
            actor_offload = "true"
            optimizer_offload = "true"
            roll_util = min(roll_util, 0.20)
            infer_util = min(infer_util, 0.20)
            print(
                "Info: enabling single-GPU-safe rollout settings "
                "(uni backend, V1 multiprocessing off, offload on, conservative vLLM memory caps)."
            )

        resolved_roll_tp = _resolve_rollout_tp(roll_tp, args.n_gpus_per_node)
        if resolved_roll_tp != roll_tp:
            print(
                f"Warning: adjusted rollout tensor parallel size from {roll_tp} to {resolved_roll_tp} "
                f"to match trainer.n_gpus_per_node={args.n_gpus_per_node}."
            )
            roll_tp = resolved_roll_tp

        resolved_infer_tp = _resolve_rollout_tp(infer_tp, args.n_gpus_per_node)
        if resolved_infer_tp != infer_tp:
            print(
                f"Warning: adjusted infer tensor parallel size from {infer_tp} to {resolved_infer_tp} "
                f"to match trainer.n_gpus_per_node={args.n_gpus_per_node}."
            )
            infer_tp = resolved_infer_tp

        resolved_train_bsz = _adjust_train_batch_size(train_bsz, rollout_n, args.n_gpus_per_node)
        if resolved_train_bsz != train_bsz:
            print(
                f"Warning: adjusted data.train_batch_size from {train_bsz} to {resolved_train_bsz} "
                f"so train_batch_size*rollout_n is divisible by n_gpus_per_node={args.n_gpus_per_node}."
            )
            train_bsz = resolved_train_bsz
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
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optimizer_offload}",
            "actor_rollout_ref.rollout.name=vllm",
            "++actor_rollout_ref.rollout.free_cache_engine=false",
            "++actor_rollout_ref.rollout.enable_sleep_mode=false",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={roll_tp}",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.logprobs_mode=null",
            f"actor_rollout_ref.rollout.max_model_len={max_len * 2}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={max_len * 4}",
            "actor_rollout_ref.rollout.max_num_seqs=8",
            "actor_rollout_ref.rollout.enable_chunked_prefill=false",
            "actor_rollout_ref.rollout.enable_prefix_caching=false",
            "actor_rollout_ref.rollout.enforce_eager=true",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={roll_util}",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            "algorithm.use_kl_in_reward=false",
            f"reward.custom_reward_function.path={reward_fn_path}",
            "reward.custom_reward_function.name=bench_reward",
            "trainer.critic_warmup=0",
            'trainer.logger=["console","file"]',
            "trainer.project_name=bench_verl_blackjack",
            f"trainer.experiment_name=verl_train_{model_key}",
            "++ray_kwargs.ray_init.include_dashboard=false",
            "trainer.nnodes=1",
            f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
            f"actor_rollout_ref.rollout.agent.num_workers={args.rollout_agent_workers}",
            "trainer.save_freq=-1",
            "trainer.test_freq=100000",
            "trainer.total_epochs=1",
            f"trainer.total_training_steps={steps}",
        ]
        if not vllm_supports_bench_engine_overrides:
            train_cmd.append("++actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni")
        elif single_gpu_vllm_safe:
            train_cmd.append("++actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni")
        else:
            train_cmd.extend(
                [
                    "++actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
                    "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_inductor=false",
                    "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_cudagraph=false",
                ]
            )
        env = os.environ.copy()
        env["VERL_FILE_LOGGER_ROOT"] = str(file_logger_root)
        # verl vLLM async server uses v1 AsyncLLM APIs.
        env["VLLM_USE_V1"] = "1"
        env["VLLM_USE_TRITON"] = env.get("VLLM_USE_TRITON", "0")
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if not vllm_supports_bench_engine_overrides:
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        elif single_gpu_vllm_safe:
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env.setdefault("RAY_DISABLE_DASHBOARD", "1")
        env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        env.setdefault("RAY_raylet_start_wait_time_s", "300")
        env.setdefault("VERL_BENCH_TRITON_CONSTEXPR_SHIM", "1")
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{bench_dir}:{py_path}" if py_path else str(bench_dir)
        if use_local_model:
            # Prevent remote hub calls when local model files are available.
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        # Avoid NCCL shared-memory allocation failures on small /dev/shm setups.
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
                "extra_metric_name": "na",
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
            f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
            f"data.train_files={infer_subset}",
            "data.prompt_key=prompt",
            f"+data.output_path={infer_out}",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.model.trust_remote_code=true",
            f"+actor_rollout_ref.model.override_config.attn_implementation={args.attn_impl}",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.logprobs_mode=null",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={infer_tp}",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={infer_util}",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.rollout.temperature=0.0",
            "actor_rollout_ref.rollout.top_p=1.0",
            f"actor_rollout_ref.rollout.response_length={infer_resp_len}",
            f"actor_rollout_ref.rollout.max_model_len={infer_resp_len * 2}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={infer_resp_len * 4}",
            "actor_rollout_ref.rollout.max_num_seqs=8",
            "actor_rollout_ref.rollout.enable_chunked_prefill=false",
            "actor_rollout_ref.rollout.enable_prefix_caching=false",
            "actor_rollout_ref.rollout.enforce_eager=true",
        ]
        if not vllm_supports_bench_engine_overrides:
            infer_cmd.append("++actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni")
        elif single_gpu_vllm_safe:
            infer_cmd.append("++actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni")
        else:
            infer_cmd.extend(
                [
                    "++actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
                    "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_inductor=false",
                    "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_cudagraph=false",
                ]
            )
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
                "extra_metric_name": "action_acc",
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
