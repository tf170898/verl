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
PERF_SUMMARY_PATH="${PERF_SUMMARY_PATH:-${OUTPUT_DIR}/blackjack_perf_summary.json}"

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

RUN_INFERENCE_STAGE="${RUN_INFERENCE_STAGE:-1}"
INFER_MODEL_PATH="${INFER_MODEL_PATH:-}"
INFER_NUM_SAMPLES="${INFER_NUM_SAMPLES:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-8}"
INFER_MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS:-4}"
INFER_MAX_PROMPT_LEN="${INFER_MAX_PROMPT_LEN:-96}"
INFER_CUDA_DEVICE="${INFER_CUDA_DEVICE:-0}"

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
echo "RUN_INFERENCE_STAGE=${RUN_INFERENCE_STAGE} INFER_NUM_SAMPLES=${INFER_NUM_SAMPLES} INFER_BATCH_SIZE=${INFER_BATCH_SIZE}"

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

TRAIN_START_TS="$(date +%s)"
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
TRAIN_END_TS="$(date +%s)"
TRAIN_DURATION_SEC="$((TRAIN_END_TS - TRAIN_START_TS))"

python3 "${THIS_DIR}/check_blackjack_e2e.py" --output-file "${LOG_PATH}"

if [[ "${RUN_INFERENCE_STAGE}" == "1" ]]; then
    export MODEL_PATH DATA_DIR LOG_PATH PERF_SUMMARY_PATH TRAIN_DURATION_SEC
    export INFER_MODEL_PATH INFER_NUM_SAMPLES INFER_BATCH_SIZE INFER_MAX_NEW_TOKENS INFER_MAX_PROMPT_LEN INFER_CUDA_DEVICE
    python3 - <<'PY'
import json
import os
import re
import time

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blackjack_reward import compute_score


def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _last_metric(log_text: str, metric_key: str):
    pattern = re.compile(rf"{re.escape(metric_key)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    matches = pattern.findall(log_text)
    if not matches:
        return None
    return _to_float(matches[-1])


log_path = os.environ["LOG_PATH"]
data_dir = os.environ["DATA_DIR"]
val_path = os.path.join(data_dir, "val.parquet")
model_path = os.environ.get("INFER_MODEL_PATH") or os.environ["MODEL_PATH"]
summary_path = os.environ["PERF_SUMMARY_PATH"]
train_duration_sec = _to_float(os.environ.get("TRAIN_DURATION_SEC")) or 0.0

num_samples = int(os.environ.get("INFER_NUM_SAMPLES", "64"))
batch_size = int(os.environ.get("INFER_BATCH_SIZE", "8"))
max_new_tokens = int(os.environ.get("INFER_MAX_NEW_TOKENS", "4"))
max_prompt_len = int(os.environ.get("INFER_MAX_PROMPT_LEN", "96"))
infer_cuda_device = int(os.environ.get("INFER_CUDA_DEVICE", "0"))

dataset = datasets.Dataset.from_parquet(val_path)
if len(dataset) == 0:
    raise SystemExit("Validation parquet is empty, cannot run inference stage.")
if num_samples > 0:
    dataset = dataset.select(range(min(num_samples, len(dataset))))
sample_count = len(dataset)

if torch.cuda.is_available():
    device = f"cuda:{infer_cuda_device}"
    torch.cuda.set_device(infer_cuda_device)
    torch.cuda.empty_cache()
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    attn_implementation="eager",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()

score_values = []
acc_values = []
valid_action_values = []
env_reward_values = []
latency_ms_values = []
generated_tokens = 0

inference_start = time.perf_counter()
for start in range(0, sample_count, batch_size):
    end = min(start + batch_size, sample_count)
    batch = dataset[start:end]
    prompts = batch["prompt"]

    prompt_texts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts
    ]
    tokenized = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    prompt_len = tokenized["input_ids"].shape[1]

    batch_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    batch_elapsed = time.perf_counter() - batch_start
    per_sample_ms = (batch_elapsed * 1000.0) / max(1, end - start)
    latency_ms_values.extend([per_sample_ms] * (end - start))

    completion_ids = outputs[:, prompt_len:]
    if tokenizer.pad_token_id is not None:
        generated_tokens += int((completion_ids != tokenizer.pad_token_id).sum().item())
    else:
        generated_tokens += int(completion_ids.numel())
    responses = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    for i, response in enumerate(responses):
        reward_model = batch["reward_model"][i]
        extra_info = batch["extra_info"][i]
        data_source = batch["data_source"][i]
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        score_dict = compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        score_values.append(float(score_dict.get("score", 0.0)))
        acc_values.append(float(score_dict.get("acc", 0.0)))
        valid_action_values.append(float(score_dict.get("valid_action", 0.0)))
        env_reward_values.append(float(score_dict.get("env_reward", 0.0)))

inference_elapsed = time.perf_counter() - inference_start
tokens_per_sec = generated_tokens / max(inference_elapsed, 1e-9)

with open(log_path, encoding="utf-8") as f:
    log_text = f.read()

training_summary = {
    "duration_sec": train_duration_sec,
    "global_step": _last_metric(log_text, "training/global_step"),
    "val_acc_mean_at_1": _last_metric(log_text, "val-core/blackjack/acc/mean@1"),
    "val_valid_action_mean_at_1": _last_metric(log_text, "val-aux/blackjack/valid_action/mean@1"),
    "val_env_reward_mean_at_1": _last_metric(log_text, "val-aux/blackjack/env_reward/mean@1"),
}

inference_summary = {
    "model_path": model_path,
    "num_samples": sample_count,
    "batch_size": batch_size,
    "max_new_tokens": max_new_tokens,
    "duration_sec": inference_elapsed,
    "latency_ms_per_sample_avg": sum(latency_ms_values) / max(1, len(latency_ms_values)),
    "generated_tokens": generated_tokens,
    "tokens_per_sec": tokens_per_sec,
    "score_mean": sum(score_values) / max(1, len(score_values)),
    "acc_mean": sum(acc_values) / max(1, len(acc_values)),
    "valid_action_mean": sum(valid_action_values) / max(1, len(valid_action_values)),
    "env_reward_mean": sum(env_reward_values) / max(1, len(env_reward_values)),
}

summary = {
    "training": training_summary,
    "inference": inference_summary,
}
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print("=== VeRL Blackjack Performance Summary ===")
for k, v in training_summary.items():
    print(f"training/{k}: {v}")
for k, v in inference_summary.items():
    print(f"inference/{k}: {v}")
print(f"Performance summary saved to: {summary_path}")
PY
fi
