#!/usr/bin/env bash
set -u -o pipefail

# Fixed benchmark harness for:
# - Frameworks: verl, redstone
# - Phases: train, infer
# - Models: Qwen/Qwen3-1.7B, Qwen/Qwen3-32B
# - Hardware assumption: 8 GPUs (B200)
#
# This script intentionally hardcodes all settings.

VERL_ROOT="${HOME}/Downloads/verl"
REDSTONE_ROOT="${HOME}/Downloads/redstone"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${VERL_ROOT}/my_bench/verl_redstone_b200_${RUN_TAG}"
LOG_DIR="${OUT_ROOT}/logs"
DATA_DIR="${OUT_ROOT}/data"
METRIC_DIR="${OUT_ROOT}/metrics"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
SUMMARY_MD="${OUT_ROOT}/summary.md"
REWARD_FN_FILE="${OUT_ROOT}/bench_reward_fn.py"
VERL_FILE_LOGGER_ROOT="${METRIC_DIR}/verl_file_logger"
export VERL_FILE_LOGGER_ROOT

TRAIN_SAMPLES=4096
TEST_SAMPLES=512

INFER_PROMPTS_TARGET=256

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${METRIC_DIR}" "${VERL_FILE_LOGGER_ROOT}"

BLACKJACK_ENV_DIR=""
if [ -d "${VERL_ROOT}/my_bench/blackjeck_env" ]; then
  BLACKJACK_ENV_DIR="${VERL_ROOT}/my_bench/blackjeck_env"
elif [ -d "${VERL_ROOT}/my_bench/blackjack_env" ]; then
  BLACKJACK_ENV_DIR="${VERL_ROOT}/my_bench/blackjack_env"
fi

if [ -z "${BLACKJACK_ENV_DIR}" ]; then
  echo "Missing blackjack env dir: expected my_bench/blackjeck_env or my_bench/blackjack_env" >&2
  exit 1
fi

TRAIN_PARQUET="${DATA_DIR}/blackjack_train.parquet"
TEST_PARQUET="${DATA_DIR}/blackjack_test.parquet"

for path in "${VERL_ROOT}" "${REDSTONE_ROOT}" "${BLACKJACK_ENV_DIR}"; do
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

generate_blackjack_dataset() {
  local train_out="$1"
  local test_out="$2"
  local train_n="$3"
  local test_n="$4"
  python3 - "$train_out" "$test_out" "$train_n" "$test_n" <<'PY'
import random
import sys

import numpy as np
import pandas as pd

train_out, test_out, train_n, test_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(20260226)

def basic_action(player_sum: int, dealer_up: int, usable_ace: int) -> str:
    if usable_ace:
        if player_sum >= 19:
            return "STAND"
        if player_sum == 18:
            return "STAND" if 2 <= dealer_up <= 8 else "HIT"
        return "HIT"
    if player_sum >= 17:
        return "STAND"
    if 13 <= player_sum <= 16:
        return "STAND" if dealer_up <= 6 else "HIT"
    if player_sum == 12:
        return "STAND" if dealer_up in (4, 5, 6) else "HIT"
    return "HIT"

def build_prompt(player_sum: int, dealer_up: int, usable_ace: int) -> str:
    return (
        "Blackjack decision task.\n"
        f"Player total: {player_sum}\n"
        f"Dealer upcard: {dealer_up}\n"
        f"Usable ace: {usable_ace}\n"
        "Choose the better action for this state. "
        "Reply with exactly one word: HIT or STAND."
    )

def build_row(idx: int) -> dict:
    player_sum = rng.randint(4, 21)
    dealer_up = rng.randint(1, 10)
    usable_ace = rng.randint(0, 1)
    gt = basic_action(player_sum, dealer_up, usable_ace)
    prompt = np.array(
        [{"role": "user", "content": build_prompt(player_sum, dealer_up, usable_ace)}],
        dtype=object,
    )
    return {
        "prompt": prompt,
        "data_source": "blackjack",
        "reward_model": {"ground_truth": gt},
        "extra_info": {
            "index": idx,
            "state": {
                "player_sum": player_sum,
                "dealer_upcard": dealer_up,
                "usable_ace": usable_ace,
            },
        },
    }

rows = [build_row(i) for i in range(train_n + test_n)]
train_df = pd.DataFrame(rows[:train_n])
test_df = pd.DataFrame(rows[train_n:])
train_df.to_parquet(train_out)
test_df.to_parquet(test_out)
print(f"wrote {len(train_df)} train rows and {len(test_df)} test rows")
PY
}

echo "Generating blackjack dataset from ${BLACKJACK_ENV_DIR}"
generate_blackjack_dataset "${TRAIN_PARQUET}" "${TEST_PARQUET}" "${TRAIN_SAMPLES}" "${TEST_SAMPLES}"

cat > "${REWARD_FN_FILE}" <<'PY'
import re


def _extract_action(text: str) -> str:
    if text is None:
        return ""
    upper = str(text).upper()
    if re.search(r"\bSTAND\b", upper):
        return "STAND"
    if re.search(r"\bHIT\b", upper):
        return "HIT"
    return ""


def bench_reward(data_source, solution_str, ground_truth, extra_info=None):
    # Reward is blackjack action correctness in {-1, +1}.
    pred = _extract_action(solution_str)
    target = str(ground_truth).upper().strip()
    return 1.0 if pred == target else -1.0
PY

now_s() {
  python3 -c 'import time; print(time.time())'
}

calc_elapsed() {
  python3 - "$1" "$2" <<'PY'
import sys
start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"{end - start:.3f}")
PY
}

calc_rate() {
  python3 - "$1" "$2" <<'PY'
import sys
work = float(sys.argv[1])
elapsed = float(sys.argv[2])
if elapsed <= 0:
    print("0.000")
else:
    print(f"{work / elapsed:.3f}")
PY
}

run_with_timing() {
  local workdir="$1"
  local log_file="$2"
  shift 2

  local start end
  start="$(now_s)"
  (
    cd "${workdir}" || exit 1
    "$@"
  ) > >(tee "${log_file}") 2>&1
  RUN_STATUS=$?
  end="$(now_s)"
  RUN_ELAPSED="$(calc_elapsed "${start}" "${end}")"
}

append_summary() {
  local framework="$1"
  local phase="$2"
  local model="$3"
  local status="$4"
  local elapsed="$5"
  local throughput="$6"
  local unit="$7"
  local train_step="$8"
  local train_loss="$9"
  local train_reward="${10}"
  local extra_metric="${11}"
  local log_file="${12}"
  echo "${framework},${phase},${model},${status},${elapsed},${throughput},${unit},${train_step},${train_loss},${train_reward},${extra_metric},${log_file}" >> "${SUMMARY_CSV}"
}

prepare_infer_subset() {
  local src="$1"
  local dst="$2"
  local max_rows="$3"
  python3 - "$src" "$dst" "$max_rows" <<'PY'
import sys
import pyarrow.parquet as pq

src, dst, max_rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
table = pq.read_table(src)
if max_rows < table.num_rows:
    table = table.slice(0, max_rows)
pq.write_table(table, dst)
print(table.num_rows)
PY
}

parse_verl_train_metrics() {
  local metrics_file="$1"
  python3 - "$metrics_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("na,na,na")
    raise SystemExit(0)

last = None
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    last = obj

if last is None:
    print("na,na,na")
    raise SystemExit(0)

step = last.get("step", "na")
data = last.get("data", {})

loss = "na"
for key in ("actor/pg_loss", "critic/vf_loss", "loss"):
    if key in data:
        loss = data[key]
        break

reward = "na"
for key in ("critic/rewards/mean", "reward_mean"):
    if key in data:
        reward = data[key]
        break

print(f"{step},{loss},{reward}")
PY
}

parse_redstone_train_metrics() {
  local metrics_file="$1"
  python3 - "$metrics_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("na,na,na")
    raise SystemExit(0)

best = None
best_step = -1
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    step = obj.get("step")
    if isinstance(step, int) and step >= best_step:
        best = obj
        best_step = step

if best is None:
    print("na,na,na")
    raise SystemExit(0)

step = best.get("step", "na")
loss = best.get("loss", "na")
reward = best.get("reward_mean", "na")
print(f"{step},{loss},{reward}")
PY
}

parse_verl_infer_accuracy() {
  local infer_parquet="$1"
  python3 - "$infer_parquet" <<'PY'
import re
import sys

import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
if len(df) == 0:
    print("na")
    raise SystemExit(0)

def action(x):
    s = str(x).upper()
    if re.search(r"\bSTAND\b", s):
        return "STAND"
    if re.search(r"\bHIT\b", s):
        return "HIT"
    return ""

ok = 0
for _, row in df.iterrows():
    responses = row.get("responses", [])
    if not responses:
        continue
    pred = action(responses[0])
    rm = row.get("reward_model", {})
    gt = str(rm.get("ground_truth", "")).upper().strip() if isinstance(rm, dict) else ""
    ok += int(pred == gt and gt != "")
acc = ok / max(1, len(df))
print(f"{acc:.4f}")
PY
}

parse_redstone_infer_reward() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

txt = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
m = re.findall(r"reward_mean=([0-9eE+\\-\\.]+)", txt)
if not m:
    print("na")
else:
    print(m[-1])
PY
}

echo "framework,phase,model,status,elapsed_sec,throughput,unit,train_step,train_loss,train_reward,extra_metric,log_file" > "${SUMMARY_CSV}"

INFER_SUBSET="${DATA_DIR}/gsm8k_test_${INFER_PROMPTS_TARGET}.parquet"
INFER_PROMPTS="$(prepare_infer_subset "${TEST_PARQUET}" "${INFER_SUBSET}" "${INFER_PROMPTS_TARGET}")"

run_verl_train() {
  local model_key="$1"
  local model_path steps train_bsz mini_bsz max_len roll_tp roll_util actor_offload rollout_n

  case "${model_key}" in
    qwen3-1.7b)
      model_path="Qwen/Qwen3-1.7B"
      steps=20
      train_bsz=64
      mini_bsz=32
      max_len=256
      roll_tp=2
      roll_util=0.70
      actor_offload=false
      rollout_n=2
      ;;
    qwen3-32b)
      model_path="Qwen/Qwen3-32B"
      steps=12
      train_bsz=16
      mini_bsz=8
      max_len=128
      roll_tp=4
      roll_util=0.60
      actor_offload=true
      rollout_n=1
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac

  local log_file="${LOG_DIR}/verl_train_${model_key}.log"
  run_with_timing "${VERL_ROOT}" "${log_file}" \
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="${TRAIN_PARQUET}" \
      data.val_files="${TEST_PARQUET}" \
      data.train_max_samples=256 \
      data.val_max_samples=64 \
      data.train_batch_size="${train_bsz}" \
      data.max_prompt_length="${max_len}" \
      data.max_response_length="${max_len}" \
      data.filter_overlong_prompts=true \
      data.truncation=error \
      actor_rollout_ref.model.path="${model_path}" \
      actor_rollout_ref.model.trust_remote_code=true \
      actor_rollout_ref.model.use_remove_padding=true \
      actor_rollout_ref.model.enable_gradient_checkpointing=true \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size="${mini_bsz}" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.use_kl_loss=false \
      actor_rollout_ref.actor.fsdp_config.param_offload="${actor_offload}" \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.tensor_model_parallel_size="${roll_tp}" \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization="${roll_util}" \
      actor_rollout_ref.rollout.n="${rollout_n}" \
      algorithm.use_kl_in_reward=false \
      reward.custom_reward_function.path="${REWARD_FN_FILE}" \
      reward.custom_reward_function.name=bench_reward \
      trainer.critic_warmup=0 \
      'trainer.logger=["console","file"]' \
      trainer.project_name=bench_verl_redstone \
      trainer.experiment_name="verl_train_${model_key}" \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=8 \
      trainer.save_freq=-1 \
      trainer.test_freq=100000 \
      trainer.total_epochs=1 \
      trainer.total_training_steps="${steps}"

  local status_label throughput train_step train_loss train_reward
  train_step="na"
  train_loss="na"
  train_reward="na"
  if [ "${RUN_STATUS}" -eq 0 ]; then
    status_label="ok"
    throughput="$(calc_rate "${steps}" "${RUN_ELAPSED}")"
    IFS=, read -r train_step train_loss train_reward <<EOF
$(parse_verl_train_metrics "${VERL_FILE_LOGGER_ROOT}/bench_verl_redstone/verl_train_${model_key}.jsonl")
EOF
  else
    status_label="fail"
    throughput="0.000"
  fi
  append_summary "verl" "train" "${model_key}" "${status_label}" "${RUN_ELAPSED}" "${throughput}" "steps/s" "${train_step}" "${train_loss}" "${train_reward}" "na" "${log_file}"
}

run_verl_infer() {
  local model_key="$1"
  local model_path infer_tp infer_util resp_len

  case "${model_key}" in
    qwen3-1.7b)
      model_path="Qwen/Qwen3-1.7B"
      infer_tp=1
      infer_util=0.85
      resp_len=256
      ;;
    qwen3-32b)
      model_path="Qwen/Qwen3-32B"
      infer_tp=4
      infer_util=0.70
      resp_len=128
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac

  local out_parquet="${OUT_ROOT}/verl_infer_${model_key}.parquet"
  local log_file="${LOG_DIR}/verl_infer_${model_key}.log"
  run_with_timing "${VERL_ROOT}" "${log_file}" \
    python3 -m verl.trainer.main_generation_server \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=8 \
      data.train_files="${INFER_SUBSET}" \
      data.prompt_key=prompt \
      data.output_path="${out_parquet}" \
      actor_rollout_ref.model.path="${model_path}" \
      actor_rollout_ref.model.trust_remote_code=true \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.tensor_model_parallel_size="${infer_tp}" \
      actor_rollout_ref.rollout.gpu_memory_utilization="${infer_util}" \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.temperature=0.0 \
      actor_rollout_ref.rollout.top_p=1.0 \
      actor_rollout_ref.rollout.response_length="${resp_len}"

  local status_label throughput action_acc
  action_acc="na"
  if [ "${RUN_STATUS}" -eq 0 ]; then
    status_label="ok"
    throughput="$(calc_rate "${INFER_PROMPTS}" "${RUN_ELAPSED}")"
    action_acc="$(parse_verl_infer_accuracy "${out_parquet}")"
  else
    status_label="fail"
    throughput="0.000"
  fi
  append_summary "verl" "infer" "${model_key}" "${status_label}" "${RUN_ELAPSED}" "${throughput}" "req/s" "na" "na" "na" "${action_acc}" "${log_file}"
}

run_redstone_train() {
  local model_key="$1"
  local model_path steps local_bsz seq_len group_size rollout_steps
  local dp_repl trainer_tp rollout_gpus rollout_tp val_gpus val_tp compile_enable run_dir

  case "${model_key}" in
    qwen3-1.7b)
      model_path="Qwen/Qwen3-1.7B"
      steps=20
      local_bsz=4
      seq_len=4096
      group_size=16
      rollout_steps=4
      dp_repl=4
      trainer_tp=1
      rollout_gpus=3
      rollout_tp=1
      val_gpus=1
      val_tp=1
      compile_enable=true
      ;;
    qwen3-32b)
      model_path="Qwen/Qwen3-32B"
      steps=12
      local_bsz=1
      seq_len=2048
      group_size=8
      rollout_steps=2
      dp_repl=1
      trainer_tp=4
      rollout_gpus=2
      rollout_tp=2
      val_gpus=2
      val_tp=2
      compile_enable=false
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac

  run_dir="${OUT_ROOT}/redstone_${model_key}"
  mkdir -p "${run_dir}"
  REDSTONE_RUN_DIR="${run_dir}"

  local log_file="${LOG_DIR}/redstone_train_${model_key}.log"
  run_with_timing "${REDSTONE_ROOT}" "${log_file}" \
    uv run --env-file .env.train --active python exps/blackjack-ppo-lora/train.py \
      --job.config-file exps/blackjack-ppo-lora/configs/base.toml \
      --job.dump-folder "${run_dir}" \
      --model-download.name "${model_path}" \
      --training.steps "${steps}" \
      --training.local-batch-size "${local_bsz}" \
      --training.seq-len "${seq_len}" \
      --grpo.group-size "${group_size}" \
      --grpo.num-rollout-steps "${rollout_steps}" \
      --env-worker.num-workers 4 \
      --rollout.sampling.max-new-tokens 8 \
      --policy-validation.freq 100000 \
      --checkpoint.interval "${steps}" \
      --parallelism.data-parallel-replicate-degree "${dp_repl}" \
      --parallelism.tensor-parallel-degree "${trainer_tp}" \
      --rollout.num-gpus "${rollout_gpus}" \
      --rollout.parallel.tensor-parallel-degree "${rollout_tp}" \
      --policy-validation.rollout.num-gpus "${val_gpus}" \
      --policy-validation.rollout.parallel.tensor-parallel-degree "${val_tp}" \
      --compile.enable "${compile_enable}"

  local status_label throughput train_step train_loss train_reward
  train_step="na"
  train_loss="na"
  train_reward="na"
  if [ "${RUN_STATUS}" -eq 0 ]; then
    status_label="ok"
    throughput="$(calc_rate "${steps}" "${RUN_ELAPSED}")"
    IFS=, read -r train_step train_loss train_reward <<EOF
$(parse_redstone_train_metrics "${run_dir}/metric/trainer.jsonl")
EOF
  else
    status_label="fail"
    throughput="0.000"
  fi
  append_summary "redstone" "train" "${model_key}" "${status_label}" "${RUN_ELAPSED}" "${throughput}" "steps/s" "${train_step}" "${train_loss}" "${train_reward}" "na" "${log_file}"
}

run_redstone_infer() {
  local model_key="$1"
  local run_dir="$2"
  local model_path infer_batch infer_tp infer_new_tokens infer_episode_size

  case "${model_key}" in
    qwen3-1.7b)
      model_path="Qwen/Qwen3-1.7B"
      infer_batch=1024
      infer_tp=1
      infer_new_tokens=16
      infer_episode_size=4
      ;;
    qwen3-32b)
      model_path="Qwen/Qwen3-32B"
      infer_batch=256
      infer_tp=4
      infer_new_tokens=8
      infer_episode_size=4
      ;;
    *)
      echo "Unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac

  local log_file="${LOG_DIR}/redstone_infer_${model_key}.log"
  run_with_timing "${REDSTONE_ROOT}" "${log_file}" \
    uv run --env-file .env.train --active python exps/blackjack-ppo-lora/infer.py \
      --job.config-file exps/blackjack-ppo-lora/configs/base.toml \
      --job.dump-folder "${run_dir}" \
      --model-download.name "${model_path}" \
      --policy-validation.batch-size "${infer_batch}" \
      --policy-validation.rollout.num-gpus 8 \
      --policy-validation.rollout.parallel.tensor-parallel-degree "${infer_tp}" \
      --policy-validation.rollout.max-episode-size "${infer_episode_size}" \
      --policy-validation.rollout.sampling.max-new-tokens "${infer_new_tokens}"

  local status_label throughput infer_reward
  infer_reward="na"
  if [ "${RUN_STATUS}" -eq 0 ]; then
    status_label="ok"
    throughput="$(calc_rate "${infer_batch}" "${RUN_ELAPSED}")"
    infer_reward="$(parse_redstone_infer_reward "${log_file}")"
  else
    status_label="fail"
    throughput="0.000"
  fi
  append_summary "redstone" "infer" "${model_key}" "${status_label}" "${RUN_ELAPSED}" "${throughput}" "traj/s" "na" "na" "na" "${infer_reward}" "${log_file}"
}

echo "Starting benchmark runs. Output directory: ${OUT_ROOT}"
echo "Using blackjack env directory: ${BLACKJACK_ENV_DIR}"
echo "Using blackjack dataset files: ${TRAIN_PARQUET}, ${TEST_PARQUET}"

for model_key in qwen3-1.7b qwen3-32b; do
  echo "==> verl train (${model_key})"
  run_verl_train "${model_key}"

  echo "==> verl infer (${model_key})"
  run_verl_infer "${model_key}"

  echo "==> redstone train (${model_key})"
  run_redstone_train "${model_key}"

  echo "==> redstone infer (${model_key})"
  run_redstone_infer "${model_key}" "${REDSTONE_RUN_DIR}"
done

python3 - "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))

lines = []
lines.append("# Benchmark Summary")
lines.append("")
lines.append(f"- CSV: `{csv_path}`")
lines.append("")
lines.append("| framework | phase | model | status | elapsed_sec | throughput | unit |")
lines.append("|---|---|---|---:|---:|---:|---|")
for r in rows:
    lines.append(
        f"| {r['framework']} | {r['phase']} | {r['model']} | {r['status']} | {r['elapsed_sec']} | {r['throughput']} | {r['unit']} |"
    )
lines.append("")
lines.append("## Training Metrics")
lines.append("")
lines.append("| framework | model | train_step | train_loss | train_reward |")
lines.append("|---|---|---:|---:|---:|")
for r in rows:
    if r["phase"] != "train":
        continue
    lines.append(
        f"| {r['framework']} | {r['model']} | {r['train_step']} | {r['train_loss']} | {r['train_reward']} |"
    )
lines.append("")
lines.append("## Extra Metrics")
lines.append("")
lines.append("- `verl infer`: `extra_metric` = action accuracy on blackjack test prompts.")
lines.append("- `redstone infer`: `extra_metric` = parsed `reward_mean` from inference log.")
lines.append("")
lines.append("| framework | phase | model | extra_metric |")
lines.append("|---|---|---|---:|")
for r in rows:
    lines.append(
        f"| {r['framework']} | {r['phase']} | {r['model']} | {r['extra_metric']} |"
    )
text = "\n".join(lines) + "\n"
md_path.write_text(text, encoding="utf-8")
print(text)
PY

echo "Done. Summary CSV: ${SUMMARY_CSV}"
echo "Done. Summary MD:  ${SUMMARY_MD}"
