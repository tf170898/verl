#!/usr/bin/env bash
set -xeuo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${THIS_DIR}/../../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/tests/special_e2e/blackjack/output}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/blackjack_qwen3_1p7b_8gpu.log}"

TRAIN_SIZE="${TRAIN_SIZE:-256}"
VAL_SIZE="${VAL_SIZE:-64}"
DATASET_SEED="${DATASET_SEED:-42}"

NUM_GPUS="${NUM_GPUS:-8}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"

python3 - <<'PY'
import importlib.util
import sys

required = ["torch", "ray", "transformers", "datasets", "pyarrow", "vllm", "numpy"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required python modules: " + ", ".join(missing))

import torch

num_gpus = int(torch.cuda.device_count())
if num_gpus < 8:
    raise SystemExit(f"Expected at least 8 CUDA GPUs, got {num_gpus}")
print(f"Preflight passed. cuda_gpus={num_gpus}")
PY

python3 "${THIS_DIR}/make_blackjack_dataset.py" \
    --output-dir "${DATA_DIR}" \
    --train-size "${TRAIN_SIZE}" \
    --val-size "${VAL_SIZE}" \
    --seed "${DATASET_SEED}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.return_raw_chat=True \
    data.max_prompt_length=96 \
    data.max_response_length=4 \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    critic.model.override_config.attn_implementation=eager \
    critic.model.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${THIS_DIR}/blackjack_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger=console \
    trainer.project_name='verl-test' \
    trainer.experiment_name='blackjack_qwen3_1p7b_8gpu_smoke' \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    | tee "${LOG_PATH}"

python3 "${THIS_DIR}/check_blackjack_e2e.py" --output-file "${LOG_PATH}"
