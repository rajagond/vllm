import argparse
from functools import partial
import os
import sys
import time
from typing import Any, Dict, List, Tuple, TypedDict, Union
import torch
import numpy as np
import datetime
import torch.distributed
import torch.nn.functional as F
from contextlib import nullcontext
from vllm.utils import FlexibleArgumentParser
from transformers import AutoConfig
import prettytable
from ipc import init_ipc
from dma_copy import copy_rs, copy_ag

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE
torch.cuda.set_device(LOCAL_RANK)

# print(f"RANK={RANK}, LOCAL_RANK={LOCAL_RANK}, LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE}, WORLD_SIZE={WORLD_SIZE}, NNODES={NNODES}")

os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
np.random.seed(42)

torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
# print(f"TP_GROUP: {TP_GROUP}", flush=True)
# print = partial(print, flush=True)

def get_p2p_comms():
    subgroup = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    device = torch.device(f"cuda:{LOCAL_RANK}")
    return init_ipc(subgroup, device)

def _rand(shape, dtype):
    return torch.randn(shape, dtype=dtype).cuda()


def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype).cuda()

def _zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype).cuda()

@torch.no_grad()
def benchmark(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    warmup_iters: int,
    test_iters: int,
    tp_size: int,
) -> Tuple[float, float, float, float, float, float]:
    torch.distributed.barrier()
    torch.cuda.synchronize()
    num_gpus = WORLD_SIZE
    rank = LOCAL_RANK
    num_iters = test_iters
    num_warmup_iters = warmup_iters

    H2_per_gpu = intermediate_size // num_gpus

    # Case 1 tensors
    input_c1 = _rand((num_tokens, hidden_size), dtype)
    mlp1_weight = _rand((hidden_size, H2_per_gpu), dtype)
    mlp2_weight = _rand((H2_per_gpu, hidden_size), dtype)
    mlp1_out_c1 = _rand((num_tokens, H2_per_gpu), dtype)
    mlp2_out_c1 = _rand((num_tokens, hidden_size), dtype)

    # Case 2 tensors
    input_c2 = _rand((num_tokens // 2, hidden_size), dtype)
    mlp1_out_c2 = _rand((num_tokens // 2, H2_per_gpu), dtype)
    mlp2_out_c2 = _rand((num_tokens // 2, hidden_size), dtype)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    def run_half():
        torch.distributed.all_reduce(mlp2_out_c2, op = torch.distributed.ReduceOp.SUM, group = TP_GROUP)

    def run_ar():
        torch.distributed.all_reduce(mlp2_out_c1, op = torch.distributed.ReduceOp.SUM, group = TP_GROUP)

    def run_mlp1_case1():
        torch.matmul(input_c1, mlp1_weight, out=mlp1_out_c1)

    def run_mlp1_case2():
        torch.matmul(input_c2, mlp1_weight, out=mlp1_out_c2)

    def run_mlp2_case1():
        torch.matmul(mlp1_out_c1, mlp2_weight, out=mlp2_out_c1)

    def run_mlp2_case2():
        torch.matmul(mlp1_out_c2, mlp2_weight, out=mlp2_out_c2)

    # JIT compilation & warmup -- AG
    run_half()
    run_ar()
    run_mlp1_case1()
    run_mlp1_case2()
    run_mlp2_case1()
    run_mlp2_case2()
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # mlp1
    # case 1
    for _ in range(num_warmup_iters):
        run_mlp1_case1()
    start_event.record()
    for _ in range(num_iters):
        run_mlp1_case1()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    mlp1_c1 = (start_event.elapsed_time(end_event) / num_iters ) * 1000.0 # us

    # case 2
    for _ in range(num_warmup_iters):
        run_mlp1_case2()
    start_event.record()
    for _ in range(num_iters):
        run_mlp1_case2()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    mlp1_c2 = (start_event.elapsed_time(end_event) / num_iters ) * 1000.0 # us

    # mlp2
    # case 1
    for _ in range(num_warmup_iters):
        run_mlp2_case1()
    start_event.record()
    for _ in range(num_iters):
        run_mlp2_case1()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()

    mlp2_c1 = (start_event.elapsed_time(end_event) / num_iters) * 1000.0 # us

    # case 2
    for _ in range(num_warmup_iters):
        run_mlp2_case2()
    start_event.record()
    for _ in range(num_iters):
        run_mlp2_case2()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()

    mlp2_c2 = (start_event.elapsed_time(end_event) / num_iters) * 1000.0 # us

    # All Reduce -- AG
    for _ in range(num_warmup_iters):
        run_half()
    start_event.record()
    for _ in range(num_iters):
        run_half()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    elapsed_time_half_ar = (start_event.elapsed_time(end_event) / num_iters) * 1000.0 # us

    for _ in range(num_warmup_iters):
        run_ar()
    start_event.record()
    for _ in range(num_iters):
        run_ar()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    elapsed_time_ar = (start_event.elapsed_time(end_event) / num_iters) * 1000 # us

    return elapsed_time_ar, elapsed_time_half_ar, mlp1_c1, mlp1_c2, mlp2_c1, mlp2_c2

@torch.no_grad()
def benchmark_copy(
    num_tokens_total: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_warmup: int = 20,
    num_iters: int = 100,
) -> float:
    torch.distributed.barrier()
    rank = LOCAL_RANK
    num_gpus = WORLD_SIZE
    p2p_comms = get_p2p_comms()

    copy_stream = torch.cuda.Stream(priority = -1)

    input_c1 = _rand([num_tokens_total, hidden_size], dtype=dtype)

    staging_buffer = _rand(
        [num_tokens_total, hidden_size], 
        dtype=dtype,
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    for r, conn in enumerate(p2p_comms):
        if conn is not None:
            conn.send(staging_buffer)
    
    buddys_staging_buffer = [
       _empty((0,), dtype=dtype) if conn is None else conn.recv()
        for r, conn in enumerate(p2p_comms)
    ]
    torch.cuda.synchronize()
    torch.distributed.barrier()

    def run_rs():
        copy_rs(
            rank,
            num_gpus,
            input_c1,
            staging_buffer,
            buddys_staging_buffer,
            (num_tokens_total) // num_gpus,
            hidden_size,
            copy_stream=copy_stream,
        )
    
    def run_ag():
        copy_ag(
            rank,
            num_gpus,
            input_c1,
            staging_buffer,
            buddys_staging_buffer,
            (num_tokens_total) // num_gpus,
            hidden_size,
            copy_stream=copy_stream,
        )

    # Benchmark logic remains the same
    run_rs()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    run_ag()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    torch.distributed.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    for ii in range(num_warmup):
        run_rs()
    start_event.record()
    for ii in range(num_iters):
        run_ag()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    avg_rs = (start_event.elapsed_time(end_event) / num_iters) * 1000
    
    for ii in range(num_warmup):
        run_ag()
    start_event.record()
    for ii in range(num_iters):
        run_ag()
    end_event.record()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    avg_ag = (start_event.elapsed_time(end_event) / num_iters) * 1000
    
    return avg_rs, avg_ag

def parse_args():
    parser = argparse.ArgumentParser()
    parser = FlexibleArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="mistralai/Mixtral-8x22B-Instruct-v0.1")
    parser.add_argument("--tp-size", "-tp", type=int, default=8)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8"],
                        default="auto")
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--plot-dir", default=".", type=str, help="plot directory")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()
    torch.cuda.synchronize()

    hidden_size = 12288
    intermediate_size = 4 * hidden_size
    dtype = torch.bfloat16
    if args.batch_size is None:
        batch_sizes = [
           16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        ]
    else:
        batch_sizes = [args.batch_size]
    
    if args.batch_size is None:
        table = prettytable.PrettyTable()
        table.field_names = ["NumTokens", "MLP1 (us)", "MLP1 (half, us)", "MLP2 (us)", "MLP2 (half, us)", "All Reduce (Half, us)", "All Reduce (us)", "dma-rs (us)", "dma-ag (us)", "% (2*half/org)", "comm% (AR/mlp1+mlp2)"]
        for batch_size in batch_sizes:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            elapsed_time_ar, elapsed_time_half_ar, mlp1_c1, mlp1_c2, mlp2_c1, mlp2_c2 = benchmark(
                num_tokens=batch_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                warmup_iters=args.warmup,
                test_iters=args.iters,
                tp_size=args.tp_size,
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            dma_rs, dma_ag = benchmark_copy(
                num_tokens_total=batch_size,
                hidden_size=hidden_size,
                dtype=dtype,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()

            table.add_row(
                [
                    batch_size,
                    f"{mlp1_c1:.2f}",
                    f"{mlp1_c2:.2f}",
                    f"{mlp2_c1:.2f}",
                    f"{mlp2_c2:.2f}",
                    f"{elapsed_time_half_ar:.2f}",
                    f"{elapsed_time_ar:.2f}",
                    f"{dma_rs:.2f}",
                    f"{dma_ag:.2f}",
                    f"{(((mlp1_c2 + mlp2_c2) - (mlp1_c1 + mlp2_c1)) / (mlp1_c1 + mlp2_c1)) * 100.0:.2f}",
                    f"{(elapsed_time_ar / (mlp1_c1 + mlp2_c1)) * 100.0:.2f}",
                ]
            )
        print(table, flush=True)
        csv_path = f"{args.plot_dir}/gpt3_mlps_allreduce_rank{LOCAL_RANK}.csv"
        with open(csv_path, "w") as f:
            f.write(table.get_csv_string())
    else:
        ctx = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
            if args.profile
            else nullcontext()
        )
        with ctx:
            pass
