"""Custom blackjack reward function for verl smoke e2e."""

from __future__ import annotations

import os
import re
import sys
from typing import Any


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from toytext_blackjack import ToyTextBlackjackEnv  # noqa: E402


ACTION_RE = re.compile(r"[A-Za-z]+")


def _parse_action_from_text(solution_str: Any) -> tuple[bool, int]:
    if solution_str is None:
        return False, 0
    text = str(solution_str).strip()
    if not text:
        return False, 0

    tokens = ACTION_RE.findall(text.lower())
    if not tokens:
        return False, 0

    first_token = tokens[0]
    if first_token == "hit":
        return True, ToyTextBlackjackEnv.HIT
    if first_token in {"stand", "stick"}:
        return True, ToyTextBlackjackEnv.STAND
    return False, 0


def _to_obs_tuple(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except Exception:
            return None
    return None


def _extract_seed_and_obs(ground_truth: Any, extra_info: Any) -> tuple[int | None, tuple[int, int, int] | None]:
    seed = None
    obs = None

    if isinstance(ground_truth, dict):
        if "seed" in ground_truth:
            try:
                seed = int(ground_truth["seed"])
            except Exception:
                seed = None
        obs = _to_obs_tuple(ground_truth.get("observation"))

    if isinstance(extra_info, dict):
        if seed is None and "seed" in extra_info:
            try:
                seed = int(extra_info["seed"])
            except Exception:
                seed = None
        if obs is None:
            obs = _to_obs_tuple(extra_info.get("observation"))

    return seed, obs


def compute_score(
    data_source: str | None,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    del kwargs

    valid, action = _parse_action_from_text(solution_str)
    seed, expected_obs = _extract_seed_and_obs(ground_truth, extra_info)

    obs_match = 0.0
    env_reward = 0.0
    terminal = 0.0

    try:
        env = ToyTextBlackjackEnv()
        initial_obs = env.reset(seed=seed)
        if expected_obs is not None:
            obs_match = 1.0 if initial_obs == expected_obs else 0.0

        if valid:
            _, env_reward, done = env.step(action)
            terminal = 1.0 if done else 0.0
    except Exception:
        valid = False
        action = 0
        env_reward = 0.0
        terminal = 0.0
        obs_match = 0.0

    valid_action = 1.0 if valid else 0.0
    if valid:
        score = 0.5 * 1.0 + 0.5 * ((float(env_reward) + 1.0) / 2.0)
    else:
        score = 0.0

    if data_source is not None and str(data_source) != "blackjack":
        # Keep smoke test deterministic; no cross-dataset reward logic.
        return {
            "score": 0.0,
            "acc": 0.0,
            "valid_action": 0.0,
            "env_reward": 0.0,
            "terminal": 0.0,
            "obs_match": 0.0,
            "action_id": float(action),
        }

    return {
        "score": float(score),
        "acc": float(valid_action),
        "valid_action": float(valid_action),
        "env_reward": float(env_reward),
        "terminal": float(terminal),
        "obs_match": float(obs_match),
        "action_id": float(action),
    }
