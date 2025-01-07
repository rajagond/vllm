import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def matmul_loop(rank, size):
    # Set the seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize the distributed process group
    dist.init_process_group("nccl", rank=rank, world_size=size)
    
    # Set the device for the current process
    device = torch.device(f'cuda:{rank}')
    
    # Create random matrices for multiplication
    A = torch.randn(1024, 1024, device=device)
    B = torch.randn(1024, 1024, device=device)
    
    # Infinite loop for matrix multiplication
    while True:
        result = torch.matmul(A, B)

def main():
    size = 8  # Number of GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Spawn processes for each GPU
    mp.spawn(matmul_loop, args=(size,), nprocs=size, join=True)

if __name__ == '__main__':
    main()
