#!/usr/bin/env python3
"""Generate verl-compatible blackjack train/val parquet files."""

from __future__ import annotations

import argparse
import os
from typing import Any

import datasets

from toytext_blackjack import ToyTextBlackjackEnv, build_user_prompt


def _row(split: str, index: int, seed: int) -> dict[str, Any]:
    env = ToyTextBlackjackEnv()
    observation = env.reset(seed=seed)
    observation_list = [int(observation[0]), int(observation[1]), int(observation[2])]
    return {
        "data_source": "blackjack",
        "prompt": [
            {
                "role": "user",
                "content": build_user_prompt(observation),
            }
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "seed": int(seed),
                "observation": observation_list,
            },
        },
        "extra_info": {
            "split": split,
            "index": int(index),
            "seed": int(seed),
            "observation": observation_list,
        },
    }


def _build_rows(split: str, size: int, seed_start: int) -> list[dict[str, Any]]:
    return [_row(split=split, index=i, seed=seed_start + i) for i in range(size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    train_rows = _build_rows(split="train", size=args.train_size, seed_start=args.seed)
    val_rows = _build_rows(split="val", size=args.val_size, seed_start=args.seed + 1_000_000)

    train_ds = datasets.Dataset.from_list(train_rows)
    val_ds = datasets.Dataset.from_list(val_rows)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Wrote train parquet: {train_path}")
    print(f"Wrote val parquet:   {val_path}")
    print(f"train_size={len(train_ds)}, val_size={len(val_ds)}")


if __name__ == "__main__":
    main()
