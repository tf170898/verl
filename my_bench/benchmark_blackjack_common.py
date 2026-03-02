from __future__ import annotations

import csv
import inspect
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_KEYS = ("qwen3-1.7b", "qwen3-32b")
INFER_PROMPTS_TARGET = 256


def validate_otel_histogram_api() -> str | None:
    """Return an error message when OpenTelemetry histogram API is incompatible with Ray."""
    try:
        from opentelemetry.sdk.metrics import MeterProvider
    except Exception:
        # If OpenTelemetry is not installed, leave validation to Ray startup.
        return None

    try:
        meter = MeterProvider().get_meter("verl_bench_preflight")
        params = inspect.signature(meter.create_histogram).parameters
    except Exception as e:
        return (
            "Failed to validate OpenTelemetry metrics API used by Ray: "
            f"{type(e).__name__}: {e}"
        )

    if "explicit_bucket_boundaries_advisory" not in params:
        return (
            "Incompatible OpenTelemetry installation detected for Ray dashboard agent: "
            "Meter.create_histogram() does not accept "
            "`explicit_bucket_boundaries_advisory`. "
            "Reinstall aligned telemetry deps, e.g. "
            "`python3 -m pip install -U opentelemetry-api opentelemetry-sdk "
            "opentelemetry-semantic-conventions`."
        )
    return None


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def calc_rate(work: float, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return work / elapsed


def run_with_logging(
    *,
    cwd: Path,
    cmd: list[str],
    log_file: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        ret = proc.wait()
    elapsed = time.time() - start
    return ret, elapsed


def find_blackjack_env_dir(verl_root: Path) -> Path:
    cands = [
        verl_root / "my_bench" / "blackjeck_env",
        verl_root / "my_bench" / "blackjack_env",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("Expected my_bench/blackjeck_env or my_bench/blackjack_env")


def find_existing_blackjack_parquet(env_dir: Path) -> tuple[Path, Path] | None:
    train_cands = sorted(env_dir.glob("**/*train*.parquet"))
    test_cands = sorted(env_dir.glob("**/*test*.parquet"))
    if train_cands and test_cands:
        return train_cands[0], test_cands[0]
    return None


def _basic_strategy_action(player_sum: int, dealer_up: int, usable_ace: int) -> str:
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


def _build_prompt(player_sum: int, dealer_up: int, usable_ace: int) -> str:
    return (
        "Blackjack decision task.\n"
        f"Player total: {player_sum}\n"
        f"Dealer upcard: {dealer_up}\n"
        f"Usable ace: {usable_ace}\n"
        "Choose the better action for this state. "
        "Reply with exactly one word: HIT or STAND."
    )


def generate_blackjack_dataset(
    *,
    train_path: Path,
    test_path: Path,
    train_samples: int,
    test_samples: int,
    seed: int = 20260226,
) -> None:
    rng = random.Random(seed)

    def build_row(idx: int) -> dict:
        player_sum = rng.randint(4, 21)
        dealer_up = rng.randint(1, 10)
        usable_ace = rng.randint(0, 1)
        gt = _basic_strategy_action(player_sum, dealer_up, usable_ace)
        prompt = np.array(
            [{"role": "user", "content": _build_prompt(player_sum, dealer_up, usable_ace)}],
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

    rows = [build_row(i) for i in range(train_samples + test_samples)]
    train_df = pd.DataFrame(rows[:train_samples])
    test_df = pd.DataFrame(rows[train_samples:])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)


def resolve_blackjack_data(
    *,
    verl_root: Path,
    out_data_dir: Path,
    train_samples: int,
    test_samples: int,
    prefer_existing_env_parquet: bool = False,
) -> tuple[Path, Path, Path]:
    env_dir = find_blackjack_env_dir(verl_root)
    if prefer_existing_env_parquet:
        existing = find_existing_blackjack_parquet(env_dir)
        if existing is not None:
            return existing[0], existing[1], env_dir

    train_path = out_data_dir / "blackjack_train.parquet"
    test_path = out_data_dir / "blackjack_test.parquet"
    generate_blackjack_dataset(
        train_path=train_path,
        test_path=test_path,
        train_samples=train_samples,
        test_samples=test_samples,
    )
    return train_path, test_path, env_dir


def prepare_infer_subset(test_parquet: Path, subset_path: Path, max_rows: int) -> int:
    df = pd.read_parquet(test_parquet)
    subset = df.head(max_rows)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(subset_path)
    return int(len(subset))


def parse_verl_train_metrics(file_logger_jsonl: Path) -> tuple[str, str, str]:
    if not file_logger_jsonl.exists():
        return ("na", "na", "na")
    last = None
    with file_logger_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    if last is None:
        return ("na", "na", "na")

    step = str(last.get("step", "na"))
    data = last.get("data", {})
    loss = "na"
    for k in ("actor/pg_loss", "critic/vf_loss", "loss"):
        if k in data:
            loss = str(data[k])
            break
    reward = "na"
    for k in ("critic/rewards/mean", "reward_mean"):
        if k in data:
            reward = str(data[k])
            break
    return (step, loss, reward)


def parse_redstone_train_metrics(trainer_jsonl: Path) -> tuple[str, str, str]:
    if not trainer_jsonl.exists():
        return ("na", "na", "na")
    best = None
    best_step = -1
    with trainer_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
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
        return ("na", "na", "na")
    return (
        str(best.get("step", "na")),
        str(best.get("loss", "na")),
        str(best.get("reward_mean", "na")),
    )


def _extract_action(text: str) -> str:
    upper = str(text).upper()
    if re.search(r"\bSTAND\b", upper):
        return "STAND"
    if re.search(r"\bHIT\b", upper):
        return "HIT"
    return ""


def _extract_first_response_text(responses) -> str:
    if isinstance(responses, np.ndarray):
        responses = responses.tolist()
    if isinstance(responses, (list, tuple)):
        if not responses:
            return ""
        first = responses[0]
    else:
        first = responses

    if first is None:
        return ""
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        for key in ("text", "content", "response"):
            val = first.get(key)
            if isinstance(val, str):
                return val
        return str(first)
    return str(first)


def parse_verl_infer_accuracy(infer_parquet: Path) -> str:
    if not infer_parquet.exists():
        return "na"
    df = pd.read_parquet(infer_parquet)
    if len(df) == 0:
        return "na"

    ok = 0
    for _, row in df.iterrows():
        response_text = _extract_first_response_text(row.get("responses", []))
        if not response_text:
            continue
        pred = _extract_action(response_text)
        reward_model = row.get("reward_model", {})
        gt = ""
        if isinstance(reward_model, dict):
            gt = str(reward_model.get("ground_truth", "")).upper().strip()
        ok += int(pred == gt and gt != "")
    return f"{ok / max(1, len(df)):.4f}"


def parse_redstone_infer_reward(log_file: Path) -> str:
    if not log_file.exists():
        return "na"
    txt = log_file.read_text(encoding="utf-8", errors="ignore")
    found = re.findall(r"reward_mean=([0-9eE+\-\.]+)", txt)
    if not found:
        return "na"
    return found[-1]


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "framework",
        "phase",
        "model",
        "status",
        "elapsed_sec",
        "throughput",
        "unit",
        "train_step",
        "train_loss",
        "train_reward",
        "extra_metric_name",
        "extra_metric",
        "log_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_summary_md(path: Path, rows: list[dict[str, str]], *, title: str) -> None:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| framework | phase | model | status | elapsed_sec | throughput | unit |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['framework']} | {r['phase']} | {r['model']} | {r['status']} | "
            f"{r['elapsed_sec']} | {r['throughput']} | {r['unit']} |"
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
    lines.append("| framework | phase | model | extra_metric_name | extra_metric |")
    lines.append("|---|---|---|---|---:|")
    for r in rows:
        lines.append(
            f"| {r['framework']} | {r['phase']} | {r['model']} | {r['extra_metric_name']} | {r['extra_metric']} |"
        )
    lines.append("")
    lines.append(
        "> Note: `action_acc` and `reward_mean` are different metrics and should not be compared directly."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
