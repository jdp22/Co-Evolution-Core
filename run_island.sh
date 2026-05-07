#!/bin/bash
# Multi-Island fine-tuning for Countdown
# K=5 islands, 1 engine each, 5 GPUs total
# Marginal-contribution reward: lambda=0.5, no EMA

set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate coevo

# --- Config ---
cuda_devices="0,1,2,3,4"
num_islands=5
engines_per_island=1
migration_freq=500000   # effectively disabled
migration_lr=0.1
marginal_lambda=1.0
coverage_ema=1.0        # raw prev-iter, no EMA
eval_tasks="math500,gsm8k"  # dataset names only; each uses handler.default_test_path
experiment_dir=island-ft-experiment
log_file=${experiment_dir}/es_island_${num_islands}isl_marginal_$(date +%Y%m%d_%H%M%S).log

mkdir -p ${experiment_dir}
cp "$0" ${experiment_dir}/run_es_island_$(date +%Y%m%d_%H%M%S).sh

nohup python -u island_finetuning.py \
  --ray_port 6480 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --cuda_devices ${cuda_devices} \
  --num_islands ${num_islands} \
  --engines_per_island ${engines_per_island} \
  --population_size 30 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --num_iterations 1000 \
  --save_freq 50 \
  --eval_freq 50 \
  --migration_freq ${migration_freq} \
  --migration_lr ${migration_lr} \
  --marginal_lambda ${marginal_lambda} \
  --coverage_ema ${coverage_ema} \
  --eval_tasks ${eval_tasks} \
  --experiment_dir ${experiment_dir} \
  --train_task uspto50k \
  --eval_tasks uspto50k \
  >> ${log_file} 2>&1 &
echo "Started with PID $!"
