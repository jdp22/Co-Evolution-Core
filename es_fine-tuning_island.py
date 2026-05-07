"""
Multi-Island ES fine-tuning for the Countdown task.

Topology:
  K islands, each with `engines_per_island` vLLM engines.
  Each island maintains its own ES population independently.
  Every `migration_freq` iterations migrations happen:
    - Virtual Seed Migration: treat (θ_other - θ_self)/σ as an extra
      virtual seed with normalized reward weight = migration_lr,
      i.e. engine-0 of each island receives the CPU weights from the
      other island's engine-0 and does: θ += migration_lr*(θ_other - θ).
      Then broadcast within the island (NCCL).

No diversity reward (clean island-model baseline).
"""

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import threading

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
SIGMA          = 0.001
ALPHA          = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES    = 4          # total engines = num_islands * engines_per_island
NUM_ITERATIONS = 1000
NUM_ISLANDS    = 2
ENGINES_PER_ISLAND = 2
MIGRATION_FREQ = 50
MIGRATION_LR   = 0.1
EXPERIMENT_DIR = "es-ft-island-experiment"


# ---------------------------------------------------------------------------
# vLLM state-dict conversion
# ---------------------------------------------------------------------------
def _convert_vllm_to_hf_state_dict(vllm_sd, hf_config):
    num_heads    = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim     = hf_config.hidden_size // num_heads
    q_size  = num_heads    * head_dim
    kv_size = num_kv_heads * head_dim
    hf_sd = {}
    for key, val in vllm_sd.items():
        if key.endswith("qkv_proj.weight"):
            prefix = key[: -len("qkv_proj.weight")]
            hf_sd[prefix + "q_proj.weight"] = val[:q_size].clone()
            hf_sd[prefix + "k_proj.weight"] = val[q_size: q_size + kv_size].clone()
            hf_sd[prefix + "v_proj.weight"] = val[q_size + kv_size:].clone()
        elif key.endswith("qkv_proj.bias"):
            prefix = key[: -len("qkv_proj.bias")]
            hf_sd[prefix + "q_proj.bias"] = val[:q_size].clone()
            hf_sd[prefix + "k_proj.bias"] = val[q_size: q_size + kv_size].clone()
            hf_sd[prefix + "v_proj.bias"] = val[q_size + kv_size:].clone()
        elif key.endswith("gate_up_proj.weight"):
            prefix = key[: -len("gate_up_proj.weight")]
            half = val.shape[0] // 2
            hf_sd[prefix + "gate_proj.weight"] = val[:half].clone()
            hf_sd[prefix + "up_proj.weight"]   = val[half:].clone()
        else:
            hf_sd[key] = val
    return hf_sd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Island ES Fine-tuning for Countdown Task"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma",          type=float, default=SIGMA)
    parser.add_argument("--alpha",          type=float, default=ALPHA)
    parser.add_argument("--population_size",type=int,   default=POPULATION_SIZE)
    parser.add_argument("--num_iterations", type=int,   default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str,   default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices",   type=str,   default="0,1,2,3",
                        help="Comma-separated GPU IDs; must have num_islands*engines_per_island entries")
    parser.add_argument("--num_islands",         type=int, default=NUM_ISLANDS)
    parser.add_argument("--engines_per_island",  type=int, default=ENGINES_PER_ISLAND)
    parser.add_argument("--migration_freq", type=int,   default=MIGRATION_FREQ,
                        help="Migrate weights between islands every N iters (0 to disable)")
    parser.add_argument("--migration_lr",   type=float, default=MIGRATION_LR,
                        help="Step size for virtual-seed migration update")
    parser.add_argument("--verbose",        action="store_true")
    parser.add_argument("--save_freq",      type=int, default=10)
    parser.add_argument("--eval_freq",      type=int, default=50)
    parser.add_argument("--global_seed",    type=int, default=None)
    parser.add_argument("--marginal_lambda",     type=float, default=0.5,
                        help="Blend weight for marginal-contribution reward (0=off, 0.5=half-half). "
                             "Reward = (1-λ)*individual + λ*marginal. marginal[s] = weighted-avg "
                             "reward on questions the other islands don't cover well.")
    parser.add_argument("--coverage_ema",        type=float, default=1.0,
                        help="EMA decay for other_coverage (1.0 = raw prev-iter, no smoothing).")
    parser.add_argument("--resume_from",         type=str,   default=None,
                        help="Path to a previous logging_dir (e.g. es-ft-island-experiment/island_es_YYYYMMDD_HHMMSS) "
                             "to resume from. Each island will load islandX_checkpoint_iter_<start_iter>/pytorch_model.pth.")
    parser.add_argument("--start_iter",          type=int,   default=0,
                        help="Iteration number to resume from (0-based). "
                             "E.g. if last saved checkpoint was iter 70, pass --start_iter 70.")
    parser.add_argument("--disable_replay_log", action="store_true",
                        help="Disable writing per-iteration replay logs (enabled by default).")
    parser.add_argument("--ray_port", type=int, default=6379,
                        help="Ray GCS port. Use different ports to run multiple independent jobs concurrently.")
    from data_handlers import list_datasets
    parser.add_argument("--train_task",          type=str,   default="countdown",
                        choices=list_datasets(),
                        help="Primary training task. Available: " + str(list_datasets()))
    parser.add_argument("--train_data_path",     type=str,   default="",
                        help="Path to training data file. Empty = use task default.")
    parser.add_argument("--eval_tasks",          type=str,   default="olympiadbench",
                        help="Comma-separated eval task list using dataset names only. "
                             "Each task loads its own default_test_path from its handler. Empty disables eval.")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    total_engines = args.num_islands * args.engines_per_island
    num_devices = len(args.cuda_devices.split(","))
    if num_devices != total_engines:
        raise ValueError(
            f"cuda_devices has {num_devices} entries but "
            f"num_islands({args.num_islands}) * engines_per_island({args.engines_per_island}) = {total_engines}"
        )
    return args


# ---------------------------------------------------------------------------
# ESNcclLLM
# ---------------------------------------------------------------------------
class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark helpers (shared)
# ---------------------------------------------------------------------------
def _resolve_eval_tasks(args):
    from data_handlers import list_datasets
    available = set(list_datasets())
    seen = set()
    eval_tasks = []
    for raw in args.eval_tasks.split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        name = tok
        if name not in available:
            raise ValueError(f"Unknown eval task '{tok}'. Available: {sorted(available)}")
        if name in seen:
            continue
        seen.add(name)
        eval_tasks.append(name)
    return eval_tasks


def _eval_task_and_log(task_name, task_handler, task_datas, engine, tokenizer, writer, step,
                       prefix="", temperature=0.0, seed=42):
    print(f"--- Evaluating {task_name} ({len(task_datas)} samples) {prefix}---")
    t0 = time.time()
    eval_handle, _ = _evaluate_handle(engine, task_handler, task_datas, tokenizer,
                                      temperature=temperature, seed=seed)
    outputs = ray.get(eval_handle)
    metrics = _postprocess_outputs(outputs, task_handler, task_datas)
    elapsed = time.time() - t0
    print(f"  [{task_name}] avg_reward={metrics['avg_reward']:.4f}  answer_acc={metrics['avg_answer_reward']:.4f}  ({elapsed:.1f}s)")
    writer.add_scalar(f"{prefix}test/{task_name}_avg_reward", metrics["avg_reward"], step)
    writer.add_scalar(f"{prefix}test/{task_name}_answer_reward", metrics["avg_answer_reward"], step)
    writer.add_scalar(f"{prefix}time/{task_name}_eval", elapsed, step)
    return outputs, metrics


# ---------------------------------------------------------------------------
# Majority vote helpers
# ---------------------------------------------------------------------------
def _majority_vote(answers, island_rewards=None):
    """Pick the most frequent non-None answer; break ties by highest island train reward, then randomly."""
    from collections import Counter
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    winners = [a for a, c in counts.items() if c == max_count]
    if len(winners) == 1:
        return winners[0]
    # Tie-breaking: prefer the answer whose supporting islands have the highest max train reward
    if island_rewards is not None:
        def best_reward_for(ans):
            return max(
                (island_rewards[k] for k, a in enumerate(answers)
                 if a == ans and island_rewards[k] is not None),
                default=-float('inf')
            )
        winners.sort(key=best_reward_for, reverse=True)
        top_r = best_reward_for(winners[0])
        winners = [a for a in winners if best_reward_for(a) == top_r]
    return random.choice(winners)


def _extract_answer_for_voting(handler, response, sample_data):
    """Normalize per-handler voting extraction to a single answer value or None."""
    # Countdown handler returns (answer, is_valid, reject_info), and we only vote
    # on valid formulas that use the correct numbers.
    if handler.name == "countdown":
        answer, is_valid, _ = handler.extract_answer_for_voting(
            response,
            numbers=sample_data.get("numbers"),
        )
        return answer if is_valid and answer else None

    # Other handlers should return an answer string.
    ans = handler.extract_answer_for_voting(response)
    if isinstance(ans, tuple):
        ans = ans[0]
    if ans is None:
        return None
    ans = str(ans).strip()
    return ans if ans else None


def _is_voted_answer_correct(handler, voted_answer, ground_truth):
    if voted_answer is None:
        return False
    if hasattr(handler, "is_voted_answer_correct"):
        return bool(handler.is_voted_answer_correct(voted_answer, ground_truth))
    formatted = handler.format_answer_for_check(str(voted_answer))
    return bool(handler.is_answer_correct(formatted, ground_truth))


def _majvote_task_accuracy(all_island_outputs, task_datas, task_handler, island_rewards=None):
    """Generic majority-vote score for any dataset handler.

    Accumulates the raw score returned by the handler so that handlers
    using continuous rewards (e.g. ROCStories lenient score) report the
    mean continuous score, while handlers returning bool report binary
    accuracy.  float(True)==1.0, float(False)==0.0 so bool handlers are
    unaffected.
    """
    scores = []
    for i, data in enumerate(task_datas):
        answers = [
            _extract_answer_for_voting(task_handler, all_island_outputs[k][i].outputs[0].text, data)
            for k in range(len(all_island_outputs))
        ]
        voted = _majority_vote(answers, island_rewards=island_rewards)
        if voted is None:
            scores.append(0.0)
        elif hasattr(task_handler, "is_voted_answer_correct"):
            scores.append(float(task_handler.is_voted_answer_correct(voted, data["ground_truth"])))
        else:
            formatted = task_handler.format_answer_for_check(str(voted))
            scores.append(float(task_handler.is_answer_correct(formatted, data["ground_truth"])))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Engine launcher
# ---------------------------------------------------------------------------
def launch_engines(num_engines, model_name):
    """Launch `num_engines` isolated vLLM actors, each pinned to one GPU."""
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


# ---------------------------------------------------------------------------
# Task handler helpers
# ---------------------------------------------------------------------------
from data_handlers import get_dataset_handler


def _evaluate_handle(llm, handler, task_datas, tokenizer, temperature=0.0, seed=42):
    """Generic async vLLM generate call; builds prompts from data["messages"]."""
    prompts = []
    for d in task_datas:
        prompts.append(tokenizer.apply_chat_template(
            d["messages"], tokenize=False, add_generation_prompt=True))
    sampling_params = SamplingParams(temperature=temperature, seed=seed, max_tokens=handler.default_max_tokens)
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


def _postprocess_outputs(outputs, handler, task_datas):
    """Generic postprocessing: uses handler.compute_reward per sample."""
    rewards, per_q_correct, per_q_behavior, output_lens = [], [], [], []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        r = float(handler.compute_reward(response, data["ground_truth"]))
        b = float(handler.compute_behavior(response, data["ground_truth"]))
        rewards.append({"reward": r, "reward_info": {}})
        per_q_correct.append(1 if r > 0.5 else 0)
        per_q_behavior.append(b)  # handler-defined behavior signal for diversity tracking
        output_lens.append(len(output.outputs[0].token_ids))
    avg_r = float(np.mean([x["reward"] for x in rewards])) if rewards else 0.0
    avg_answer_r = float(np.mean(per_q_correct)) if per_q_correct else 0.0
    return {
        "rewards": rewards,
        "avg_reward": avg_r,
        "avg_answer_reward": avg_answer_r,   # pure binary accuracy (r > 0.5)
        "per_q_correct": per_q_correct,       # list[int] length M
        "per_q_behavior": per_q_behavior,      # list[float] handler-defined behavior signal
        "output_lens": output_lens,            # list[int] output token count per question
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _save_training_state(ckpt_path, iteration, island_rng_states):
    state = {
        "iteration":        iteration,
        "rng_python":       random.getstate(),
        "rng_numpy":        np.random.get_state(),
        "rng_torch":        torch.random.get_rng_state(),
        "island_rng_states": island_rng_states,
    }
    torch.save(state, os.path.join(ckpt_path, "training_state.pth"))


def _load_training_state(ckpt_path):
    state = torch.load(os.path.join(ckpt_path, "training_state.pth"), weights_only=False)
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.random.set_rng_state(state["rng_torch"])
    return state["iteration"], state.get("island_rng_states", {})


def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
        f.write("\n")


def _finalize_replay_record(record):
    digest_obj = {k: v for k, v in record.items() if k != "update_digest"}
    payload = json.dumps(digest_obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    record["update_digest"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return record


# ---------------------------------------------------------------------------
# Per-island ES step
# ---------------------------------------------------------------------------
def run_island_es_step(island_idx, engines, task_datas, tokenizer, seeds, args, writer, iteration,
                       other_coverage=None, handler=None):
    """
    Run one ES generation for a single island.

    other_coverage : np.ndarray [M] float32 or None
        EMA-smoothed fraction of OTHER islands' population answering each question correctly.
        When not None and args.marginal_lambda > 0, the NES fitness becomes:
            r_nes = (1-λ)*r_indiv + λ*r_marginal
        where r_marginal[s] rewards covering questions the other islands DON'T cover.

    Returns (seeds_perf, mean_r, behavior_mat, per_seed_coeffs):
        seeds_perf   : {seed -> {avg_reward, avg_answer_reward, norm_reward, per_q_correct}}
        mean_r       : float, mean individual reward (for logging / eval tie-breaking)
        behavior_mat : np.ndarray [N, M] float32, per-seed raw reward per question
        per_seed_coeffs : list[(seed, coeff)] exact update order/scalars applied on engine-0
    """
    prefix = f"island{island_idx}/"
    seeds_perf = {}
    behavior_rows = {}   # seed -> list[float] handler-defined behavior per question
    seed_iter = iter(seeds)
    inflight = {}
    results_this_gen = []

    # Kick off initial batch
    for eng_idx, llm in enumerate(engines):
        try:
            seed = next(seed_iter)
        except StopIteration:
            break
        ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
        handle, start_ts = _evaluate_handle(llm, handler, task_datas, tokenizer)
        inflight[handle] = {"engine": llm, "engine_idx": eng_idx, "seed": seed, "start_ts": start_ts}

    while inflight:
        done, _ = ray.wait(list(inflight.keys()), num_returns=1)
        h = done[0]
        meta = inflight.pop(h)

        outputs = ray.get(h)
        metrics = _postprocess_outputs(outputs, handler, task_datas)
        elapsed = time.time() - meta["start_ts"]

        seeds_perf[meta["seed"]] = metrics
        behavior_rows[meta["seed"]] = metrics["per_q_behavior"]
        results_this_gen.append({"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed})

        llm = meta["engine"]
        ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

        try:
            next_seed = next(seed_iter)
        except StopIteration:
            continue

        ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
        handle, start_ts = _evaluate_handle(llm, handler, task_datas, tokenizer)
        inflight[handle] = {
            "engine": llm, "engine_idx": meta["engine_idx"],
            "seed": next_seed, "start_ts": start_ts
        }

    # Individual rewards (always computed for logging)
    all_avg_rewards = [seeds_perf[s]["avg_reward"] for s in seeds]
    mean_r = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
    std_r  = float(np.std(all_avg_rewards))  if all_avg_rewards else 0.0

    writer.add_scalar(f"{prefix}reward/mean", mean_r, iteration)
    writer.add_scalar(f"{prefix}reward/std",  std_r,  iteration)
    # Log output token length (mean across all seeds × questions)
    all_output_lens = [l for s in seeds for l in seeds_perf[s].get("output_lens", [])]
    if all_output_lens:
        tok_mean = float(np.mean(all_output_lens))
        tok_p90  = float(np.percentile(all_output_lens, 90))
        tok_max  = float(np.max(all_output_lens))
        writer.add_scalar(f"{prefix}output/token_mean", tok_mean, iteration)
        writer.add_scalar(f"{prefix}output/token_p90",  tok_p90,  iteration)
        writer.add_scalar(f"{prefix}output/token_max",  tok_max,  iteration)
        writer.add_histogram(f"{prefix}output/token_dist", np.array(all_output_lens), iteration)
    print(f"[Island {island_idx}] mean_reward={mean_r:.4f}  std={std_r:.4f}" +
          (f"  tok_mean={tok_mean:.0f}  tok_p90={tok_p90:.0f}" if all_output_lens else ""))

    # -----------------------------------------------------------------
    # Build NES fitness: blend individual + marginal-contribution reward
    #
    # Convergence note: using a fixed other_coverage (EMA-smoothed, from
    # the previous iteration) keeps the landscape quasi-stationary so NES
    # gradient estimates remain consistent within a generation. The 1-λ
    # individual component ensures each island continues to improve even
    # when other islands cover everything, preventing reward collapse.
    # -----------------------------------------------------------------
    indiv_rewards = np.array(all_avg_rewards, dtype=np.float32)   # [N]
    lam = float(getattr(args, "marginal_lambda", 0.0))
    if other_coverage is not None and lam > 0.0:
        bmat  = np.array([behavior_rows[s] for s in seeds], dtype=np.float32)  # [N, M]
        w     = np.maximum(0.0, 1.0 - other_coverage)   # [M]: 1 = other can't cover, 0 = covered
        w_sum = float(w.sum())
        if w_sum > 1e-8:
            marginal_r = (bmat * w[np.newaxis, :]).sum(axis=1) / w_sum  # [N]
        else:
            # Other islands cover everything → marginal signal is flat; fall back to individual
            marginal_r = indiv_rewards
        nes_rewards = (1.0 - lam) * indiv_rewards + lam * marginal_r
        uncovered_frac = float(np.mean(w))
        marginal_mean  = float(np.mean(marginal_r))
        writer.add_scalar(f"{prefix}reward/marginal_mean",      marginal_mean,  iteration)
        writer.add_scalar(f"{prefix}reward/marginal_max",       float(np.max(marginal_r)),  iteration)
        writer.add_scalar(f"{prefix}reward/marginal_std",       float(np.std(marginal_r)),  iteration)
        writer.add_scalar(f"{prefix}coverage/uncovered_frac",   uncovered_frac, iteration)
        writer.add_scalar(f"{prefix}coverage/uncovered_count",  float(np.sum(w > 0.1)), iteration)
        writer.add_histogram(f"{prefix}coverage/per_q_other",   other_coverage, iteration)
        writer.add_histogram(f"{prefix}coverage/weight_w",      w,              iteration)
        writer.add_histogram(f"{prefix}reward/marginal_dist",   marginal_r,     iteration)
        print(f"[Island {island_idx}] marginal_mean={marginal_mean:.4f}  "
              f"uncovered_frac={uncovered_frac:.4f}", flush=True)
    else:
        nes_rewards = indiv_rewards

    nes_mean = float(np.mean(nes_rewards))
    nes_std  = float(np.std(nes_rewards))
    # Log NES fitness distribution
    writer.add_scalar(f"{prefix}reward/nes_mean", nes_mean, iteration)
    writer.add_scalar(f"{prefix}reward/nes_std",  nes_std,  iteration)
    writer.add_histogram(f"{prefix}reward/indiv_dist",  indiv_rewards, iteration)
    writer.add_histogram(f"{prefix}reward/nes_dist",    nes_rewards,   iteration)
    for i, s in enumerate(seeds):
        seeds_perf[s]["norm_reward"] = (float(nes_rewards[i]) - nes_mean) / (nes_std + 1e-8)

    # Apply ES update on engine 0
    per_seed_coeffs = [
        (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
        for seed in seeds
    ]
    handles = [
        engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False))
        for seed, coeff in per_seed_coeffs
    ]
    ray.get(handles)

    # Broadcast from engine 0 to all engines in this island
    ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])

    behavior_mat = np.array([behavior_rows[s] for s in seeds], dtype=np.float32)  # [N, M]
    # Per-question coverage of this island's population
    pop_coverage = behavior_mat.astype(np.float32).mean(axis=0)   # [M]
    writer.add_scalar(f"{prefix}coverage/self_mean",  float(pop_coverage.mean()), iteration)
    writer.add_scalar(f"{prefix}coverage/self_max",   float(pop_coverage.max()),  iteration)
    writer.add_histogram(f"{prefix}coverage/per_q_self", pop_coverage, iteration)
    return seeds_perf, mean_r, behavior_mat, per_seed_coeffs


# ---------------------------------------------------------------------------
# Timestamped stdout: every print line is prefixed with wall-clock time.
# ---------------------------------------------------------------------------
class _TimestampedStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._at_bol  = True   # True when next write starts a new line
        self._lock    = threading.Lock()   # guards _at_bol across threads

    def write(self, text):
        if not text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        lines = text.split("\n")
        out = []
        with self._lock:
            for i, segment in enumerate(lines):
                if self._at_bol and segment:
                    out.append(f"[{ts}] {segment}")
                else:
                    out.append(segment)
                if i < len(lines) - 1:
                    out.append("\n")
                    self._at_bol = True
                else:
                    self._at_bol = segment == "" or text.endswith("\n")
            self._wrapped.write("".join(out))

    def flush(self):
        self._wrapped.flush()

    def fileno(self):
        return self._wrapped.fileno()

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    # Install timestamp wrapper so every print line carries wall-clock time
    sys.stdout = _TimestampedStream(sys.stdout)
    sys.stderr = _TimestampedStream(sys.stderr)


    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    # Each job gets its own isolated Ray cluster via a unique temp dir.
    # Do NOT call ray.shutdown() here — it would kill another job's cluster.
    ray_temp_dir = f"/tmp/ray_job_{args.ray_port}"
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True,
             _temp_dir=ray_temp_dir)

    logging_dir = f"{args.experiment_dir}/island_es_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)
    replay_logs_dir = f"{logging_dir}/replay_logs"
    if not args.disable_replay_log:
        os.makedirs(replay_logs_dir, exist_ok=True)

    # Load base model once, save as HF checkpoint for vLLM to load
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to("cpu")
    tokenizer  = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load training data
    handler = get_dataset_handler(args.train_task)
    train_path = args.train_data_path or handler.default_train_path
    task_datas = handler.load_data(train_path)
    print(f"[{args.train_task}] Train samples: {len(task_datas)}")

    # Build eval task configs from CLI. All eval and majvote paths are handler-driven.
    eval_task_configs = {}   # {task_name: {"handler": handler, "datas": [...]}}
    eval_tasks = _resolve_eval_tasks(args)
    for eval_task in eval_tasks:
        eval_handler = get_dataset_handler(eval_task)
        eval_path = eval_handler.default_test_path
        if not eval_path or not os.path.exists(eval_path):
            print(f"[WARN] Skip eval task '{eval_task}': data path not found: {eval_path}")
            continue
        eval_datas = eval_handler.load_data(eval_path, split="test")
        eval_task_configs[eval_task] = {"handler": eval_handler, "datas": eval_datas}
        print(f"[{eval_task}] Eval samples: {len(eval_datas)} (path={eval_path})")

    if not eval_task_configs:
        print("[WARN] No eval tasks loaded; eval/majvote will be skipped.")

    # Launch all engines
    total_engines = args.num_islands * args.engines_per_island
    all_engines, all_pgs = launch_engines(total_engines, base_model_path)

    # Partition engines into islands
    islands = [
        all_engines[i * args.engines_per_island: (i + 1) * args.engines_per_island]
        for i in range(args.num_islands)
    ]

    # Init intra-island NCCL group for each island (independent master ports)
    master_address = get_ip()
    for isl_idx, island_engines in enumerate(islands):
        master_port = get_open_port()
        ray.get([
            island_engines[rank].collective_rpc.remote(
                "init_inter_engine_group",
                args=(master_address, master_port, rank, args.engines_per_island)
            )
            for rank in range(args.engines_per_island)
        ])
        print(f"Island {isl_idx}: NCCL group initialized (port {master_port})")

    # Per-island RNG state (seeded with island index for reproducibility if global_seed set)
    island_rngs = [random.Random(args.global_seed + isl_idx if args.global_seed is not None else None)
                   for isl_idx in range(args.num_islands)]

    def cleanup():
        for llm in all_engines:
            try: ray.kill(llm)
            except Exception: pass
        for pg in all_pgs:
            try: remove_placement_group(pg)
            except Exception: pass
        ray.shutdown()

    def sig_handler(sig, frame):
        sys.stdout._wrapped.write(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] [MAIN] Received signal {sig} — shutting down\n"
        )
        sys.stdout._wrapped.flush()
        # Abort barriers so island threads don't hang at wait()
        try: iter_barrier.abort()
        except Exception: pass
        try: migration_barrier.abort()
        except Exception: pass
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # -----------------------------------------------------------------------
    # Shared coordination state
    # -----------------------------------------------------------------------
    # For majority vote: last island to finish eval at a given iteration triggers it
    eval_lock    = threading.Lock()
    eval_store   = {}   # {iteration_key: {isl_idx: (eval_outputs_by_task, mean_r)}}

    # For migration: two-phase barrier so all islands deposit weights before any reads
    migration_barrier     = threading.Barrier(args.num_islands)
    migration_weights_store = {}
    migration_weights_lock  = threading.Lock()

    # For marginal-contribution reward: each island deposits its latest behavior_mat
    # after each step so other islands can read it.
    behavior_store = {}   # {isl_idx: np.ndarray [N, M] int8}
    behavior_lock  = threading.Lock()

    # Iteration barrier: all islands sync at the TOP of each iteration so that
    # behavior_store reads are always from a complete previous iteration.
    iter_barrier = threading.Barrier(args.num_islands)

    # -----------------------------------------------------------------------
    # Resume: load checkpoint weights into each island engine
    # -----------------------------------------------------------------------
    if args.resume_from and args.start_iter > 0:
        print(f"[RESUME] Loading checkpoints from {args.resume_from}, start_iter={args.start_iter}")
        for isl_idx, isl_engines in enumerate(islands):
            ckpt_pth = os.path.join(
                args.resume_from, "model_saves",
                f"island{isl_idx}_checkpoint_iter_{args.start_iter}",
                "pytorch_model.pth"
            )
            if not os.path.exists(ckpt_pth):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_pth}")
            ray.get(isl_engines[0].collective_rpc.remote("load_weights_from_disk", args=(ckpt_pth,)))
            print(f"[RESUME] Island {isl_idx} loaded {ckpt_pth}")

    if not args.disable_replay_log:
        replay_meta = {
            "logging_dir": logging_dir,
            "created_at": datetime.now().isoformat(),
            "num_islands": int(args.num_islands),
            "start_iter": int(args.start_iter),
            "num_iterations": int(args.num_iterations),
            "sigma": float(args.sigma),
            "alpha": float(args.alpha),
            "population_size": int(args.population_size),
            "migration_freq": int(args.migration_freq),
            "migration_lr": float(args.migration_lr),
            "resume_from": args.resume_from,
        }
        with open(f"{replay_logs_dir}/replay_meta.json", "w", encoding="utf-8") as f:
            json.dump(replay_meta, f, ensure_ascii=True, indent=2)

    # -----------------------------------------------------------------------
    # Per-island training loop (runs in its own thread)
    # -----------------------------------------------------------------------
    def island_loop(isl_idx):
        isl_rng      = island_rngs[isl_idx]
        isl_engines  = islands[isl_idx]
        prefix       = f"island{isl_idx}/"
        ema_coverage = None   # EMA-smoothed other_coverage, local to this thread

        try:
            # -----------------------------------------------------------
            # Step-0 eval: evaluate before any weight update
            # -----------------------------------------------------------
            STEP0_KEY = args.start_iter - 1   # unique key in eval_store (-1 for fresh start)
            step0_tb  = args.start_iter       # TensorBoard step (0 for fresh, resume_iter for resumed)
            if args.eval_freq > 0 and eval_task_configs and args.start_iter == 0:
                print(f"--- [Island {isl_idx}] Step-0 eval ({', '.join(eval_task_configs.keys())}) ---")
                eval_outputs_by_task = {}
                train_metrics = None
                for task_name, cfg in eval_task_configs.items():
                    outputs, metrics = _eval_task_and_log(
                        task_name,
                        cfg["handler"],
                        cfg["datas"],
                        isl_engines[0],
                        tokenizer,
                        writer,
                        step0_tb,
                        prefix,
                        temperature=1.0,
                        seed=isl_idx,
                    )
                    eval_outputs_by_task[task_name] = outputs
                    if task_name == args.train_task:
                        train_metrics = metrics
                if train_metrics is not None:
                    writer.add_scalar(f"{prefix}test/avg_reward",    train_metrics["avg_reward"],        step0_tb)
                    writer.add_scalar(f"{prefix}test/answer_reward", train_metrics["avg_answer_reward"], step0_tb)
                do_majvote = False
                with eval_lock:
                    if STEP0_KEY not in eval_store:
                        eval_store[STEP0_KEY] = {}
                    eval_store[STEP0_KEY][isl_idx] = (eval_outputs_by_task, 0.0)
                    if len(eval_store[STEP0_KEY]) == args.num_islands:
                        do_majvote = True
                if do_majvote:
                    for task_name, cfg in eval_task_configs.items():
                        all_task_outputs = [eval_store[STEP0_KEY][i][0][task_name] for i in range(args.num_islands)]
                        mv = _majvote_task_accuracy(all_task_outputs, cfg["datas"], cfg["handler"])
                        writer.add_scalar(f"test/majvote_{task_name}_accuracy", mv, step0_tb)
                        print(f"  [MajVote@step0] {task_name}={mv:.4f}")
                    with eval_lock:
                        del eval_store[STEP0_KEY]

            for iteration in range(args.start_iter, args.num_iterations):
                iter_start = time.time()
                seeds = [isl_rng.randint(0, 1_000_000) for _ in range(args.population_size)]

                # ----------------------------------------------------------
                # Sync point: all islands wait here so behavior_store holds
                # a complete snapshot from the previous iteration before any
                # island reads it. This ensures other_coverage and ensemble
                # coverage are computed from consistent (same-iteration) data.
                # ----------------------------------------------------------
                iter_barrier.wait()

                # Read other islands' behavior from the PREVIOUS iteration
                # (clean, because we just passed the barrier).
                other_coverage = None
                if args.marginal_lambda > 0.0 and args.num_islands > 1:
                    with behavior_lock:
                        pieces = [
                            behavior_store[j].astype(np.float32).mean(axis=0)  # [M]
                            for j in range(args.num_islands)
                            if j != isl_idx and j in behavior_store
                        ]
                    if pieces:
                        raw_cov = np.mean(np.stack(pieces, axis=0), axis=0)  # [M]
                        if ema_coverage is None:
                            ema_coverage = raw_cov
                        else:
                            alpha_ema = float(args.coverage_ema)
                            # alpha_ema=1.0 → use latest value (no EMA)
                            # alpha_ema=0.1 → slow EMA (90% old, 10% new)
                            ema_coverage = (1.0 - alpha_ema) * ema_coverage + alpha_ema * raw_cov
                        other_coverage = ema_coverage

                # Ensemble coverage using last iteration's data (only island 0 logs to avoid duplicates)
                if isl_idx == 0 and iteration > 0:
                    with behavior_lock:
                        all_mats = [behavior_store[j] for j in range(args.num_islands)
                                    if j in behavior_store]
                    if len(all_mats) == args.num_islands:
                        ensemble_cov = np.zeros(all_mats[0].shape[1], dtype=np.float32)
                        for mat in all_mats:
                            ensemble_cov = np.maximum(ensemble_cov,
                                                      mat.astype(np.float32).mean(axis=0))
                        writer.add_scalar("ensemble/coverage_mean",
                                          float(ensemble_cov.mean()), iteration)
                        writer.add_scalar("ensemble/coverage_max",
                                          float(ensemble_cov.max()), iteration)
                        writer.add_scalar("ensemble/fully_covered_frac",
                                          float(np.mean(ensemble_cov > 0.5)), iteration)

                _, mean_r, behavior_mat, per_seed_coeffs = run_island_es_step(
                    isl_idx, isl_engines, task_datas, tokenizer, seeds, args, writer, iteration,
                    other_coverage=other_coverage, handler=handler
                )
                replay_record = None
                if not args.disable_replay_log:
                    replay_record = {
                        "iteration": int(iteration),
                        "island_idx": int(isl_idx),
                        "update_order": [int(seed) for seed in seeds],
                        "seeds": [int(seed) for seed in seeds],
                        "coeffs": [float(coeff) for _, coeff in per_seed_coeffs],
                        "sigma": float(args.sigma),
                        "alpha": float(args.alpha),
                        "population_size": int(args.population_size),
                        "migration_event": {
                            "happened": False,
                            "partner_idx": None,
                            "migration_lr": None,
                        },
                        "checkpoint_event": {
                            "saved": False,
                            "path": None,
                        },
                        "update_digest": None,
                    }
                # Write behavior AFTER step so next iteration's barrier reads fresh data
                with behavior_lock:
                    behavior_store[isl_idx] = behavior_mat

                iter_elapsed = time.time() - iter_start
                writer.add_scalar(f"{prefix}time/iteration", iter_elapsed, iteration)
                print(f"[Island {isl_idx}] iter {iteration} done | mean_r={mean_r:.4f} "
                      f"| wall={iter_elapsed:.1f}s")

                # -----------------------------------------------------------
                # Independent checkpoint
                # -----------------------------------------------------------
                if args.save_freq > 0 and (iteration + 1) % args.save_freq == 0:
                    ckpt_path = f"{model_saves_dir}/island{isl_idx}_checkpoint_iter_{iteration+1}"
                    os.makedirs(ckpt_path, exist_ok=True)
                    ray.get(isl_engines[0].collective_rpc.remote(
                        "save_self_weights_to_disk", args=(f"{ckpt_path}/pytorch_model.pth",)
                    ))
                    tokenizer.save_pretrained(ckpt_path)
                    if replay_record is not None:
                        replay_record["checkpoint_event"] = {
                            "saved": True,
                            "path": f"island{isl_idx}_checkpoint_iter_{iteration+1}/pytorch_model.pth",
                        }
                    print(f"[Island {isl_idx}] Checkpoint saved at iter {iteration+1}")

                # -----------------------------------------------------------
                # Independent eval; last island triggers majority vote
                # -----------------------------------------------------------
                if args.eval_freq > 0 and eval_task_configs and (iteration + 1) % args.eval_freq == 0:
                    print(f"--- [Island {isl_idx}] Test eval ({', '.join(eval_task_configs.keys())}) ---")
                    eval_outputs_by_task = {}
                    train_metrics = None
                    for task_name, cfg in eval_task_configs.items():
                        outputs, metrics = _eval_task_and_log(
                            task_name,
                            cfg["handler"],
                            cfg["datas"],
                            isl_engines[0],
                            tokenizer,
                            writer,
                            iteration,
                            prefix,
                        )
                        eval_outputs_by_task[task_name] = outputs
                        if task_name == args.train_task:
                            train_metrics = metrics
                    if train_metrics is not None:
                        writer.add_scalar(f"{prefix}test/avg_reward",    train_metrics["avg_reward"],        iteration)
                        writer.add_scalar(f"{prefix}test/answer_reward", train_metrics["avg_answer_reward"], iteration)

                    # Deposit results; check if we are the last island
                    do_majvote = False
                    with eval_lock:
                        if iteration not in eval_store:
                            eval_store[iteration] = {}
                        eval_store[iteration][isl_idx] = (eval_outputs_by_task, mean_r)
                        if len(eval_store[iteration]) == args.num_islands:
                            do_majvote = True

                    if do_majvote:
                        island_rewards = [eval_store[iteration][i][1] for i in range(args.num_islands)]
                        for task_name, cfg in eval_task_configs.items():
                            all_task_outputs = [eval_store[iteration][i][0][task_name] for i in range(args.num_islands)]
                            mv = _majvote_task_accuracy(
                                all_task_outputs,
                                cfg["datas"],
                                cfg["handler"],
                                island_rewards=island_rewards,
                            )
                            writer.add_scalar(f"test/majvote_{task_name}_accuracy", mv, iteration)
                            print(f"  [MajVote@{iteration+1}] {task_name}={mv:.4f}")

                        with eval_lock:
                            del eval_store[iteration]   # free memory

                # -----------------------------------------------------------
                # Migration: all islands synchronize here
                # -----------------------------------------------------------
                if args.migration_freq > 0 and (iteration + 1) % args.migration_freq == 0:
                    print(f"[Island {isl_idx}] Arrived at migration barrier (iter {iteration+1})")

                    # Phase 1: each island deposits its own CPU weights
                    weights = ray.get(
                        isl_engines[0].collective_rpc.remote("get_weights_to_cpu", args=())
                    )[0]
                    with migration_weights_lock:
                        migration_weights_store[isl_idx] = weights

                    migration_barrier.wait()    # wait until all islands have deposited

                    # Phase 2: each island independently applies delta from its partner
                    other_idx = (isl_idx + 1) % args.num_islands
                    if replay_record is not None:
                        replay_record["migration_event"] = {
                            "happened": True,
                            "partner_idx": int(other_idx),
                            "migration_lr": float(args.migration_lr),
                        }
                    print(f"  Island {isl_idx} <- Island {other_idx} (lr={args.migration_lr})")
                    ray.get(
                        isl_engines[0].collective_rpc.remote(
                            "apply_delta_from_cpu",
                            args=(migration_weights_store[other_idx], args.migration_lr)
                        )
                    )
                    ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,))
                             for e in isl_engines])

                    migration_barrier.wait()    # wait until all islands are done applying

                    with migration_weights_lock:
                        if isl_idx == 0:
                            migration_weights_store.clear()
                            gc.collect()
                    print(f"[Island {isl_idx}] Migration done at iter {iteration+1}")

                if replay_record is not None:
                    replay_record = _finalize_replay_record(replay_record)
                    _append_jsonl(
                        f"{replay_logs_dir}/island{isl_idx}.jsonl",
                        replay_record,
                    )

        except Exception as exc:
            import traceback
            print(f"[Island {isl_idx}] FATAL exception at iter {locals().get('iteration', '?')}: {exc}", flush=True)
            traceback.print_exc()
            # Abort all barriers so other threads don't hang forever
            try: iter_barrier.abort()
            except Exception: pass
            try: migration_barrier.abort()
            except Exception: pass
            raise

        # Final model save
        final_path = f"{model_saves_dir}/island{isl_idx}_final_iter_{args.num_iterations}"
        os.makedirs(final_path, exist_ok=True)
        ray.get(isl_engines[0].collective_rpc.remote(
            "save_self_weights_to_disk", args=(f"{final_path}/pytorch_model.pth",)
        ))
        tokenizer.save_pretrained(final_path)
        print(f"[Island {isl_idx}] Final model saved to {final_path}")

    # -----------------------------------------------------------------------
    # Watchdog: every 120 s log which threads are still alive so silent
    # hangs (e.g. barrier deadlock, Ray actor timeout) are visible in the log.
    # -----------------------------------------------------------------------
    _watchdog_stop = threading.Event()
    def _watchdog():
        while not _watchdog_stop.wait(120):
            alive = [t.name for t in threads if t.is_alive()]
            print(f"[WATCHDOG] alive={alive}", flush=True)
    _wd_thread = threading.Thread(target=_watchdog, name="watchdog", daemon=True)

    # -----------------------------------------------------------------------
    # Launch all island threads and wait
    # -----------------------------------------------------------------------
    threads = [
        threading.Thread(target=island_loop, args=(isl_idx,), name=f"island-{isl_idx}", daemon=True)
        for isl_idx in range(args.num_islands)
    ]
    _wd_thread.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _watchdog_stop.set()

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
