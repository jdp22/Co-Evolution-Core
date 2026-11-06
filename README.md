# Co-Evolve: A One-to-many LLM Fine-tuning Paradigm

This repository contains the core implementation of **Co-Evolve**, a method
that fine-tunes one pretrained language model into a cooperative population of
persistent descendants. Each island receives more credit for solving examples
that are under-covered by its peers. The trainer implements this objective with
evolution strategies and records the seed-based parameter updates used by each
island.

Paper: [**Co-Evolution: A "One-to-many" LLM Fine-Tuning Paradigm**](https://github.com/jdp22/Co-evolution-A-One-to-many-LLM-Fine-Tuning-Paradigm)
(NeurIPS 2026, accepted as a poster). The camera-ready source and compiled
paper are maintained in that paper repository.

The public core implementation is this repository:
[github.com/jdp22/Co-Evolution-Core](https://github.com/jdp22/Co-Evolution-Core).

The repository is intentionally focused on the released training core:

- multi-island ES fine-tuning with marginal-coverage credit;
- vLLM worker extensions for seeded perturbations and weight synchronization;
- dataset handlers and reward functions for the supported tasks;
- checkpoint and replay-log writing during training.

The main algorithm entry point is [`island_finetuning.py`](island_finetuning.py).
It implements the multi-island perturbation loop, peer-coverage reweighting,
seed-based updates, checkpointing, and replay-log emission used by the paper's
core method.

Later exploratory experiments and large raw experiment directories are not part
of this repository.

## Installation

The trainer requires Python 3.10 or newer, CUDA, and one GPU per island engine.
The tested setup uses vLLM 0.11.0.

```bash
python -m venv coevo
source coevo/bin/activate

pip install "vllm==0.11.0" "ray[default]" tensorboard datasets pyarrow
pip install -r requirement.txt
```

The model weights and benchmark data are not included. Download them from
their original sources and place the prepared files under the paths expected by
the dataset handlers, or pass an explicit `--train_data_path`.

## Data layout

The handlers use local prepared files. A typical layout is:

```text
data/
├── uspto_50k/train.parquet
├── uspto_50k/test.parquet
├── olympiadbench/train.parquet
├── olympiadbench/test.parquet
├── rocstories/train.parquet
└── rocstories/test.parquet

baseline/
└── data/countdown/
    ├── train_200.parquet
    └── test.parquet
```

The Countdown handler keeps its historical default under
`baseline/data/countdown/`:

```text
baseline/data/countdown/train_200.parquet
baseline/data/countdown/test.parquet
```

For the paper's low-data setting, `train_200.parquet` should contain the first
200 examples of the selected training split. Evaluation uses the complete
official test split. The exact prompt and reward implementation lives in
`data_handlers/` and `utils/reward_score/`.

## Paper-aligned training example

The main paper configuration uses five descendants, 30 perturbations per
island, 250 ES rounds, and full marginal credit (`lambda=1`). The following
command assumes five visible GPUs and a prepared Countdown training file with
200 examples:

```bash
python island_finetuning.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --cuda_devices 0,1,2,3,4 \
  --num_islands 5 \
  --engines_per_island 1 \
  --population_size 30 \
  --num_iterations 250 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --marginal_lambda 1.0 \
  --migration_freq 0 \
  --train_task countdown \
  --train_data_path baseline/data/countdown/train_200.parquet \
  --eval_tasks countdown \
  --experiment_dir runs/countdown_qwen25_3b
```

`--cuda_devices` must contain exactly
`num_islands × engines_per_island` device IDs. Set `--eval_tasks ""` to skip
evaluation during training.

The included `run_island.sh` is a convenience launcher. For a paper result,
prefer an explicit command or save the exact command and environment alongside
the run directory.

## Outputs and replay logs

Each run writes a timestamped directory containing the shared base model,
periodic island checkpoints, TensorBoard events, and replay logs:

```text
runs/countdown_qwen25_3b/
└── island_run_YYYYMMDD_HHMMSS/
    ├── model_saves/
    │   ├── base_model/
    │   └── island*_checkpoint_iter_*/
    └── replay_logs/
        ├── replay_meta.json
        └── island*.jsonl
```

Each replay record stores the perturbation seeds, update coefficients, update
order, checkpoint events, migration events, and a digest of the record. The
logs are intended to support seed-based reconstruction and audit of island
updates. Full model checkpoints can be disabled only if an external replay and
materialization workflow is available for the run.

## Supported handlers

The repository includes handlers for Countdown, OlympiadBench, ROCStories,
USPTO-50K, GQA, GSM8K, MATH-500, MBPP, and the boxed Countdown variant. The
generic trainer is intended for text models. Vision-language experiments using
Qwen2.5-VL require a processor-aware multimodal inference path and are not
covered by the text-only command above.

## License

The code in this repository is released under the MIT License. Model weights,
benchmark data, and third-party dependencies remain subject to their own
licenses.

## Citation

```bibtex
@inproceedings{jiang2026coevolution,
  title     = {Co-Evolution: A ``One-to-many'' LLM Fine-Tuning Paradigm},
  author    = {Jiang, Dapeng and Tan, Haichuan and Lei, Dianqiao and
               Song, Wenxuan and Zong, Shenqi and Lan, Yanyan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
