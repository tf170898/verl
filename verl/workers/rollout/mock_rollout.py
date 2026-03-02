# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight async rollout backend for smoke tests.

This backend intentionally bypasses vLLM/sglang/trtllm and serves deterministic
short text actions ("hit"/"stand") from a simple Ray actor.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, Optional

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica, TokenOutput
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)


class MockHttpServer:
    """Minimal token-in/token-out server used by AgentLoopManager."""

    def __init__(self, config: RolloutConfig, model_config: HFModelConfig):
        self.config = config
        self.model_config = model_config

        model_path = (
            model_config.get("path", None) if hasattr(model_config, "get") else getattr(model_config, "path", None)
        )
        trust_remote_code = (
            model_config.get("trust_remote_code", False)
            if hasattr(model_config, "get")
            else getattr(model_config, "trust_remote_code", False)
        )
        self.tokenizer = None
        if model_path:
            local_path = copy_to_local(model_path)
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        self._hit_ids = self._encode_action("hit")
        self._stand_ids = self._encode_action("stand")

    def _encode_action(self, action: str) -> list[int]:
        if self.tokenizer is None:
            return [0]
        for text in (action, f" {action}"):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                return list(ids)
        fallback = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
        return [int(fallback)]

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        del prompt_ids, image_data, video_data, priority

        max_tokens = sampling_params.get(
            "max_tokens", sampling_params.get("max_new_tokens", self.config.response_length)
        )
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = int(self.config.response_length)
        max_tokens = max(1, min(max_tokens, int(self.config.response_length)))

        choose_hit = True
        if request_id:
            try:
                choose_hit = int(request_id[-1], 16) % 2 == 0
            except Exception:
                choose_hit = True
        token_ids = (self._hit_ids if choose_hit else self._stand_ids)[:max_tokens]

        want_logprobs = sampling_params.get("logprobs", None)
        log_probs = [0.0] * len(token_ids) if want_logprobs not in (None, False) else None

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=None,
            stop_reason="completed",
            num_preempted=0,
        )

    async def wake_up(self, tags: Optional[list[str]] = None):
        del tags
        return

    async def sleep(self, level: int = 1):
        del level
        return

    async def clear_kv_cache(self):
        return

    async def wait_for_requests_to_drain(self):
        return

    async def start_profile(self, **kwargs):
        del kwargs
        return

    async def stop_profile(self):
        return

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        del reset_prefix_cache
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        return

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        del reset_prefix_cache
        return {"aborted": False, "request_id": request_id}


class MockReplica(RolloutReplica):
    """RolloutReplica implementation backed by MockHttpServer."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(MockHttpServer)

    async def launch_servers(self):
        name = (
            f"mock_server_{self.replica_rank}"
            if not self.is_reward_model
            else f"mock_server_reward_{self.replica_rank}"
        )
        server = self.server_class.options(name=name).remote(config=self.config, model_config=self.model_config)
        self.servers.append(server)
        self._server_handle = server
        self._server_address = f"mock://{name}"


class ServerAdapter(BaseRollout):
    """No-op training-to-rollout adapter for the mock async backend."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        del replica_rank
        super().__init__(config, model_config, device_mesh)

    async def resume(self, tags: list[str]):
        del tags
        return

    async def release(self):
        return

    @torch.no_grad()
    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        del kwargs
        async for _ in ensure_async_iterator(weights):
            pass
        return

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        del prompts
        raise NotImplementedError("mock rollout only supports async server mode generation.")
