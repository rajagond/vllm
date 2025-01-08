#!/bin/bash

sudo nvidia-smi --gpu-reset || true
sudo nvidia-smi -pm ENABLED || true
sudo nvidia-smi -lgc tdp,tdp || true

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/ /_/g')
echo "GPU name: $gpu_name"
date=$(date +"%Y_%m_%d")
dir="./${gpu_name}"
mkdir -p $dir
time_stamp=$(date +"%Y_%m_%d_%H_%M_%S")

python3 benchmark.py > $dir/${time_stamp}.log 2>&1

nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port="23456"
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"
hidden_size=12288
dtype="bfloat16"
## with communication
nproc_per_node=8
CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${additional_args} benchmark_allreduce.py --plot-dir ${dir}"

echo ${CMD}
${CMD} > $dir/ar_mlp_${nproc_per_node}_${time_stamp}.txt 2>&1
sudo nvidia-smi -rgc || true
exit $ret
