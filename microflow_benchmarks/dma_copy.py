"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
import torch


def copy_rs(source_gpu: int, num_gpus: int,
                            intermediate_cache3_ext: torch.Tensor,
                            buddys_intermediate_cache3_ext: List[torch.Tensor],
                            num_tokens: int,
                            hidden_size: int,
                            copy_stream: Optional[torch.cuda.Stream] = None,) -> None:
    current_stream = torch.cuda.current_stream()
    if copy_stream is None:
        copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(current_stream)
    with torch.cuda.stream(copy_stream):
        for iter_ in range(1, num_gpus):
            dst_r = (source_gpu + iter_) % num_gpus
            start_idx = dst_r * num_tokens
            ops.copy_2d(
                intermediate_cache3_ext,
                buddys_intermediate_cache3_ext[dst_r],
                hidden_size,
                start_idx,
                num_tokens,
            )
    current_stream.wait_stream(copy_stream)


def copy_ag(source_gpu: int, num_gpus: int,
                            intermediate_cache3_ext: torch.Tensor,
                            buddys_intermediate_cache3_ext: List[torch.Tensor],
                            num_tokens: int,
                            hidden_size: int,
                            copy_stream: Optional[torch.cuda.Stream] = None,) -> None:
    start_idx = source_gpu * num_tokens
    current_stream = torch.cuda.current_stream()
    if copy_stream is None:
        copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(current_stream)
    with torch.cuda.stream(copy_stream):
        for iter_ in range(1, num_gpus):
            dst_r = (source_gpu + iter_) % num_gpus
            ops.copy_2d(
                buddys_intermediate_cache3_ext[dst_r],
                intermediate_cache3_ext,
                hidden_size,
                start_idx,
                num_tokens,
            )
    current_stream.wait_stream(copy_stream)