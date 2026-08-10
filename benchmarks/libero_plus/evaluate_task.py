import contextlib
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hydra
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig

from benchmarks.libero.evaluate_task import (
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_episode,
)
from benchmarks.libero.simulation import (
    LIBERO_ENV_RESOLUTION,
    get_libero_env,
    save_rollout_video,
)
from vid2wam.data.processor import Vid2WAMProcessor
from vid2wam.data.normalization import load_dataset_stats_from_json
from vid2wam.support.tensors import set_global_seed
from libero.libero import benchmark


def _load_manifest_task(cfg: DictConfig) -> tuple[dict, list[dict]]:
    raw_path = cfg.EVALUATION.get("perturbation_manifest_path")
    if raw_path is None:
        raise ValueError("EVALUATION.perturbation_manifest_path must be provided.")
    manifest_path = Path(str(raw_path)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Perturbation manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite = str(cfg.EVALUATION.task_suite_name)
    base_task_id = str(int(cfg.EVALUATION.task_id))
    try:
        task_entry = manifest["tasks"][suite][base_task_id]
    except KeyError as error:
        raise KeyError(f"Manifest has no entry for {suite} task {base_task_id}.") from error

    variants = task_entry["variants"]
    requested_all = bool(cfg.EVALUATION.get("all_trials", False))
    manifest_all = manifest.get("evaluation_mode") == "all"
    if requested_all != manifest_all:
        raise ValueError(
            f"Manifest mode is {manifest.get('evaluation_mode')!r}, "
            f"but EVALUATION.all_trials={requested_all}."
        )
    expected = None if requested_all else int(cfg.EVALUATION.num_trials)
    if expected is not None and len(variants) != expected:
        raise ValueError(
            f"Manifest contains {len(variants)} variants for {suite} task {base_task_id}, "
            f"but EVALUATION.num_trials={expected}."
        )
    return task_entry, variants


def _build_runtime(cfg: DictConfig):
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: Vid2WAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    return model, processor, model_device, action_horizon, int(video_size[1]), int(video_size[0])


@hydra.main(version_base="1.3", config_path="../../settings", config_name="libero_plus.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    if int(cfg.EVALUATION.get("env_num", 1)) != 1:
        raise ValueError("LIBERO-Plus evaluation supports env_num=1 per worker.")
    _validate_visualize_future_video_cfg(cfg)
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError("LIBERO-Plus currently supports visualize_future_video=false only.")

    task_entry, variants = _load_manifest_task(cfg)
    model, processor, model_device, action_horizon, input_w, input_h = _build_runtime(cfg)

    suite_name = str(cfg.EVALUATION.task_suite_name)
    base_task_id = int(cfg.EVALUATION.task_id)
    with contextlib.redirect_stdout(io.StringIO()):
        task_suite = benchmark.get_benchmark_dict()[suite_name]()

    output_root = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    suite_dir = output_root / suite_name
    video_dir = suite_dir / "videos"
    if bool(cfg.EVALUATION.get("save_rollout_video", False)):
        video_dir.mkdir(parents=True, exist_ok=True)
    suite_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "benchmark": "LIBERO-Plus",
        "task_suite": suite_name,
        "task_id": base_task_id,
        "task_description": task_entry["base_task_name"],
        "successes": 0,
        "total_episodes": len(variants),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "category_stats": {},
        "trials": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
    }
    category_stats = defaultdict(lambda: {"successes": 0, "total_episodes": 0})

    for trial_idx, variant in enumerate(variants):
        variant_task_id = int(variant["variant_task_id"])
        task = task_suite.get_task(variant_task_id)
        if task.name != variant["name"]:
            raise ValueError(
                f"LIBERO-Plus task mapping mismatch at {suite_name}/{variant_task_id}: "
                f"benchmark={task.name!r}, manifest={variant['name']!r}."
            )

        initial_states = task_suite.get_task_init_states(variant_task_id)
        if len(initial_states) == 0:
            raise ValueError(f"No init states for {suite_name} variant task {variant_task_id}.")
        state_index = trial_idx % len(initial_states)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
        try:
            success, replay_images, _, _ = run_single_episode(
                env=env,
                initial_state=initial_states[state_index],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
        finally:
            env.close()

        category = str(variant["category"])
        category_stats[category]["total_episodes"] += 1
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
            category_stats[category]["successes"] += 1
        else:
            results["failure_episodes"].append(trial_idx)

        if bool(cfg.EVALUATION.get("save_rollout_video", False)):
            save_rollout_video(
                video_dir,
                replay_images,
                f"base{base_task_id}_trial{trial_idx}_variant{variant_task_id}",
                success=success,
                task_description=task_description,
            )

        results["trials"].append(
            {
                "trial_index": trial_idx,
                "variant_task_id": variant_task_id,
                "variant_name": variant["name"],
                "category": category,
                "difficulty_level": int(variant["difficulty_level"]),
                "repeat_index": int(variant.get("repeat_index", 0)),
                "init_state_index": state_index,
                "success": bool(success),
            }
        )
        print(
            f"{suite_name} base_task={base_task_id} trial={trial_idx + 1}/{len(variants)} "
            f"category={category} variant={variant_task_id} success={bool(success)}"
        )

    for category, stats in category_stats.items():
        stats["success_rate"] = stats["successes"] / stats["total_episodes"]
    results["category_stats"] = dict(category_stats)
    results["duration"] = time.time() - start_time

    output_file = suite_dir / f"gpu{cfg.gpu_id}_task{base_task_id}_results.json"
    output_file.write_text(json.dumps(results, indent=2, cls=NumpyEncoder) + "\n", encoding="utf-8")
    print(
        f"Completed {suite_name} base task {base_task_id}: "
        f"{results['successes']}/{results['total_episodes']} successes in {results['duration']:.2f}s"
    )
    return results


if __name__ == "__main__":
    eval_single_process()
