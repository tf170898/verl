#!/usr/bin/env python3
"""Validate blackjack e2e smoke log output."""

from __future__ import annotations

import argparse


REQUIRED_SUBSTRINGS = [
    "Initial validation metrics",
    "training/global_step",
    "val-core/blackjack/acc/mean@1",
    "val-aux/blackjack/valid_action/mean@1",
    "val-aux/blackjack/env_reward/mean@1",
]

FINAL_VAL_BANNER = "Final validation metrics"
FINAL_VAL_FALLBACK_KEYS = [
    "val-core/blackjack/acc/mean@1",
    "val-aux/blackjack/valid_action/mean@1",
    "val-aux/blackjack/env_reward/mean@1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True, type=str)
    parser.add_argument("--tail-lines", type=int, default=4000)
    args = parser.parse_args()

    with open(args.output_file, encoding="utf-8") as f:
        content = f.read()

    missing = [s for s in REQUIRED_SUBSTRINGS if s not in content]

    # Some trainer/logging combinations do not emit the exact "Final validation metrics" banner.
    # Accept those runs if validation metric keys appear in the log tail.
    if FINAL_VAL_BANNER not in content:
        tail_content = "\n".join(content.splitlines()[-args.tail_lines :])
        if not all(k in tail_content for k in FINAL_VAL_FALLBACK_KEYS):
            missing.append(FINAL_VAL_BANNER)

    if missing:
        raise AssertionError(
            "Blackjack e2e log check failed. Missing expected substrings:\n"
            + "\n".join(f"- {s}" for s in missing)
        )

    print("Blackjack e2e log check passed.")


if __name__ == "__main__":
    main()
