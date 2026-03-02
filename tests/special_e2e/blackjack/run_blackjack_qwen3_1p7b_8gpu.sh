#!/usr/bin/env bash
set -xeuo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${THIS_DIR}/../../.." && pwd)"

# Ensure all subprocesses (including Ray/vLLM workers) load local startup shims.
export PYTHONPATH="${THIS_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-mock}"
export ROLLOUT_BACKEND
export VERL_FORCE_NO_FLASH_ATTN="${VERL_FORCE_NO_FLASH_ATTN:-1}"
if [[ "${ROLLOUT_BACKEND}" == "vllm" ]]; then
    export VLLM_USE_V1=1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/tests/special_e2e/blackjack/output}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/blackjack_qwen3_1p7b_8gpu.log}"

TRAIN_SIZE="${TRAIN_SIZE:-256}"
VAL_SIZE="${VAL_SIZE:-64}"
DATASET_SEED="${DATASET_SEED:-42}"

NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_N="${ROLLOUT_N:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-128}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-128}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-auto}"
ROLLOUT_CUDAGRAPH_MODE="${ROLLOUT_CUDAGRAPH_MODE:-PIECEWISE}"

# Prefer local Hugging Face cache rooted at /models and avoid online downloads.
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/models}"
MODEL_REPO_ID="${MODEL_REPO_ID:-Qwen/Qwen3-1.7B}"
export HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}}"
if [[ -d "${MODEL_CACHE_ROOT}/hub" ]]; then
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODEL_CACHE_ROOT}/hub}"
else
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODEL_CACHE_ROOT}}"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

resolve_local_snapshot_path() {
    local repo_id="$1"
    local repo_key="${repo_id//\//--}"
    local model_dir=""
    local snapshot_path=""

    if [[ -d "${MODEL_CACHE_ROOT}/models--${repo_key}" ]]; then
        model_dir="${MODEL_CACHE_ROOT}/models--${repo_key}"
    elif [[ -d "${HF_HUB_CACHE}/models--${repo_key}" ]]; then
        model_dir="${HF_HUB_CACHE}/models--${repo_key}"
    else
        return 1
    fi

    snapshot_path="$(ls -1dt "${model_dir}/snapshots/"* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${snapshot_path}" || ! -d "${snapshot_path}" ]]; then
        return 1
    fi
    printf '%s\n' "${snapshot_path}"
}

if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(resolve_local_snapshot_path "${MODEL_REPO_ID}" || true)"
    if [[ -z "${MODEL_PATH}" ]]; then
        echo "Unable to find local snapshot for ${MODEL_REPO_ID} under ${MODEL_CACHE_ROOT} or ${HF_HUB_CACHE}." >&2
        echo "Set MODEL_PATH to a local model directory (for example: /models/models--Qwen--Qwen3-1.7B/snapshots/<commit>)."
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"
echo "Using MODEL_PATH=${MODEL_PATH}"
echo "HF_HOME=${HF_HOME} HF_HUB_CACHE=${HF_HUB_CACHE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "ROLLOUT_BACKEND=${ROLLOUT_BACKEND} ROLLOUT_N=${ROLLOUT_N} ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE} ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN} ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS} ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS} ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION} ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT} ROLLOUT_CUDAGRAPH_MODE=${ROLLOUT_CUDAGRAPH_MODE}"
echo "VERL_FORCE_NO_FLASH_ATTN=${VERL_FORCE_NO_FLASH_ATTN}"

python3 - <<'PY'
import importlib.util
import os

backend = os.environ.get("ROLLOUT_BACKEND", "mock")
required = ["torch", "ray", "transformers", "datasets", "pyarrow", "numpy"]
if backend == "vllm":
    required.append("vllm")
elif backend == "sglang":
    required.append("sglang")
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
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_remove_padding=False \
    actor_rollout_ref.ref.use_remove_padding=False \
    critic.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="${ROLLOUT_CUDAGRAPH_MODE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    +critic.model.override_config.attn_implementation=eager \
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
