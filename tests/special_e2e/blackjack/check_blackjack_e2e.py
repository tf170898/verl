#!/usr/bin/env python3
"""Validate blackjack e2e smoke log output."""

from __future__ import annotations

import argparse


REQUIRED_SUBSTRINGS = [
    "Initial validation metrics",
    "Final validation metrics",
    "training/global_step",
    "val-core/blackjack/acc/mean@1",
    "val-aux/blackjack/valid_action/mean@1",
    "val-aux/blackjack/env_reward/mean@1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True, type=str)
    args = parser.parse_args()

    with open(args.output_file, encoding="utf-8") as f:
        content = f.read()

    missing = [s for s in REQUIRED_SUBSTRINGS if s not in content]
    if missing:
        raise AssertionError(
            "Blackjack e2e log check failed. Missing expected substrings:\n"
            + "\n".join(f"- {s}" for s in missing)
        )

    print("Blackjack e2e log check passed.")


if __name__ == "__main__":
    main()
