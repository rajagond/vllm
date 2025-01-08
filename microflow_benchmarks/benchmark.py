import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from prettytable import PrettyTable

def generate_batch_lengths() -> List[int]:
    """Generate batch lengths from 2 to 65536, doubling each time."""
    return [2 * 2**i for i in range(16)]  # 2 to 65536


def benchmark_matmul(batch_length: int, 
                    H: int = 12 * 1024, 
                    ngpus: int = 8,
                    warmup_iters: int = 10,
                    test_iters: int = 100) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Benchmark matrix multiplication operations for both cases.
    
    Returns:
        Tuple of (mlp1_case1_time, mlp1_case2_time, mlp2_case1_time, mlp2_case2_time)
    """
    H2 = 4 * H
    H2_per_gpu = H2 // ngpus
    
    # Case 1 tensors
    input_c1 = torch.randn(batch_length, H, device='cuda', dtype=torch.bfloat16)
    mlp1_weight = torch.randn(H, H2_per_gpu, device='cuda', dtype=torch.bfloat16)
    mlp2_weight = torch.randn(H2_per_gpu, H, device='cuda', dtype=torch.bfloat16)
    mlp1_out_c1 = torch.zeros(batch_length, H2_per_gpu, device='cuda', dtype=torch.bfloat16)
    mlp2_out_c1 = torch.zeros(batch_length, H, device='cuda', dtype=torch.bfloat16)
    
    # Case 2 tensors
    input_c2 = torch.randn(batch_length // 2, H, device='cuda', dtype=torch.bfloat16)
    mlp1_out_c2 = torch.zeros(batch_length // 2, H2_per_gpu, device='cuda', dtype=torch.bfloat16)
    mlp2_out_c2 = torch.zeros(batch_length // 2, H, device='cuda', dtype=torch.bfloat16)
    
    # CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # mlp1
    # case 1
    for _ in range(warmup_iters):
        torch.matmul(input_c1, mlp1_weight, out=mlp1_out_c1)
    start_event.record()
    for _ in range(test_iters):
        torch.matmul(input_c1, mlp1_weight, out=mlp1_out_c1)
    end_event.record()
    torch.cuda.synchronize()
    mlp1_case1_time = (start_event.elapsed_time(end_event) / test_iters) * 1000.0 # Convert to microseconds

    # case 2
    for _ in range(warmup_iters):
        torch.matmul(input_c2, mlp1_weight, out=mlp1_out_c2)
    start_event.record()
    for _ in range(test_iters):
        torch.matmul(input_c2, mlp1_weight, out=mlp1_out_c2)
    end_event.record()
    torch.cuda.synchronize()
    mlp1_case2_time = (start_event.elapsed_time(end_event) / test_iters) * 2000.0 # Convert to microseconds

    # mlp2
    # case 1
    for _ in range(warmup_iters):
        torch.matmul(mlp1_out_c1, mlp2_weight, out=mlp2_out_c1)
    start_event.record()
    for _ in range(test_iters):
        torch.matmul(mlp1_out_c1, mlp2_weight, out=mlp2_out_c1)
    end_event.record()
    torch.cuda.synchronize()
    mlp2_case1_time = (start_event.elapsed_time(end_event) / test_iters) * 1000.0 # Convert to microseconds

    # case 2
    for _ in range(warmup_iters):
        torch.matmul(mlp1_out_c2, mlp2_weight, out=mlp2_out_c2)
    start_event.record()
    for _ in range(test_iters):
        torch.matmul(mlp1_out_c2, mlp2_weight, out=mlp2_out_c2)
    end_event.record()
    torch.cuda.synchronize()
    mlp2_case2_time = (start_event.elapsed_time(end_event) / test_iters) * 2000.0 # Convert to microseconds

    return mlp1_case1_time, mlp1_case2_time, mlp2_case1_time, mlp2_case2_time
    
    # # Warmup
    # for _ in range(warmup_iters):
    #     # Case 1
    #     mlp1_out_c1 = torch.matmul(input_c1, mlp1_weight)
    #     mlp2_out_c1 = torch.matmul(mlp1_out_c1, mlp2_weight)
    #     # Case 2
    #     mlp1_out_c2 = torch.matmul(input_c2, mlp1_weight)
    #     mlp2_out_c2 = torch.matmul(mlp1_out_c2, mlp2_weight)
    
    # # Timing arrays
    # mlp1_times_c1 = []
    # mlp2_times_c1 = []
    # mlp1_times_c2 = []
    # mlp2_times_c2 = []
    
    # for _ in range(test_iters):
    #     # Case 1 MLP1
    #     start_event.record()
    #     mlp1_out_c1 = torch.matmul(input_c1, mlp1_weight)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     mlp1_times_c1.append(start_event.elapsed_time(end_event))
        
    #     # Case 1 MLP2
    #     start_event.record()
    #     mlp2_out_c1 = torch.matmul(mlp1_out_c1, mlp2_weight)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     mlp2_times_c1.append(start_event.elapsed_time(end_event))
        
    #     # Case 2 MLP1
    #     start_event.record()
    #     mlp1_out_c2 = torch.matmul(input_c2, mlp1_weight)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     mlp1_times_c2.append(start_event.elapsed_time(end_event))
        
    #     # Case 2 MLP2
    #     start_event.record()
    #     mlp2_out_c2 = torch.matmul(mlp1_out_c2, mlp2_weight)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     mlp2_times_c2.append(start_event.elapsed_time(end_event))
    
    # # Calculate average times
    # mlp1_c1_time = np.mean(mlp1_times_c1) * 1000.0 # Convert to microseconds
    # mlp1_c2_time = np.mean(mlp1_times_c2) * 2000.0 # Convert to microseconds
    # mlp2_c1_time = np.mean(mlp2_times_c1) * 1000.0 # Convert to microseconds
    # mlp2_c2_time = np.mean(mlp2_times_c2) * 2000.0 # Convert to microseconds
    
    # return (
    #     mlp1_c1_time, mlp1_c2_time, mlp2_c1_time, mlp2_c2_time,
    # )

def create_results_table(results: List[dict]) -> PrettyTable:
    """Create a pretty table from the results."""
    table = PrettyTable()
    table.field_names = [
        "BL", 
        "MLP1 C1 (us)", "MLP1 C2 (us)", "MLP2 C1 (us)", "MLP2 C2 (us)",
    ]
    table.float_format = ".3"
    
    for result in results:
        table.add_row([
            result["batch_length"],
            result["mlp1_case1"],
            result["mlp1_case2"],
            result["mlp2_case1"],
            result["mlp2_case2"]
        ])
    
    return table

def plot_results(results: List[dict]):
    """Create seaborn plots for both time and TFLOPS results."""
    """Create seaborn plots for the results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Melt the DataFrame for seaborn plotting
    df_melted = pd.melt(
        df,
        id_vars=['batch_length'],
        value_vars=['mlp1_case1', 'mlp1_case2', 'mlp2_case1', 'mlp2_case2'],
        var_name='operation',
        value_name='time_us'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df_melted,
        x='batch_length',
        y='time_us',
        hue='operation',
        marker='o'
    )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Length')
    plt.ylabel('Time (us)')
    plt.title('MatMul Performance Comparison')
    plt.legend(title='Operation')
    
    # Save the plot
    plt.savefig('matmul_benchmark_results.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    
    # Generate batch lengths
    batch_lengths = generate_batch_lengths()
    
    # Run benchmarks
    results = []
    print("Running benchmarks...")
    
    for bl in batch_lengths:
        (mlp1_c1, mlp1_c2, mlp2_c1, mlp2_c2) = benchmark_matmul(bl)
        results.append({
            "batch_length": bl,
            "mlp1_case1": mlp1_c1,
            "mlp1_case2": mlp1_c2,
            "mlp2_case1": mlp2_c1,
            "mlp2_case2": mlp2_c2,
        })
    
    # Create and print pretty table
    table = create_results_table(results)
    print("\nBenchmark Results (times in us):")
    print(table)
    
    # Create plot
    plot_results(results)
    print("\nPlot saved as 'matmul_benchmark_results.png'")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('matmul_benchmark_results.csv', index=False)
    print("Results saved to 'matmul_benchmark_results.csv'")

if __name__ == "__main__":
    main()