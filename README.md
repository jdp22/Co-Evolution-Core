
## Setup

Requires Python ≥ 3.10, CUDA, and one GPU per island engine.

```bash
python -m venv es && source es/bin/activate
pip install vllm==0.11.0 tensorboard
pip install -r requirement.txt
```

## Usage

```bash
python es_fine-tuning_island.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --cuda_devices 0,1,2,3,4 \
  --num_islands 5 \
  --engines_per_island 1 \
  --marginal_lambda 0.5 \
  --population_size 30 \
  --num_iterations 1000 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --train_task countdown \
  --eval_tasks math500,gsm8k \
  --experiment_dir es-ft-island-experiment
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model_name` | — | HuggingFace model path or name |
| `--cuda_devices` | `0,1,2,3` | Comma-separated GPU IDs; must equal `num_islands × engines_per_island` |
| `--num_islands` | 2 | Number of islands K |
| `--engines_per_island` | 2 | vLLM engines per island |
| `--marginal_lambda` | 0.5 | Marginal reward blend weight (0 = individual only, 1 = pure marginal) |
| `--train_task` | `countdown` | Training task; see `data_handlers/` for available tasks |
| `--eval_tasks` | `olympiadbench` | Comma-separated eval tasks |
| `--migration_freq` | 50 | Weight migration interval in iterations (0 to disable) |
| `--save_freq` | 10 | Checkpoint save frequency |
| `--eval_freq` | 50 | Eval frequency |
