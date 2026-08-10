import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf

from benchmarks.libero.launch_suite import (
    _resolve_worker_task_choice,
    collect_worker_overrides,
    run_evaluation,
)


PERTURBATION_CATEGORIES = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]

ORIGINAL_TASKS = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_category(
    candidates: list[dict],
    count: int,
    *,
    seed: int,
) -> list[dict]:
    if not candidates:
        raise ValueError("Cannot sample from an empty perturbation category.")

    rng = random.Random(seed)
    selected = []
    repeat_index = defaultdict(int)
    while len(selected) < count:
        cycle = list(candidates)
        rng.shuffle(cycle)
        for row in cycle:
            if len(selected) >= count:
                break
            item = dict(row)
            item["repeat_index"] = repeat_index[int(row["variant_task_id"])]
            repeat_index[int(row["variant_task_id"])] += 1
            selected.append(item)
    return selected


def build_manifest(
    classification_path: Path,
    *,
    num_trials: int,
    seed: int,
    all_trials: bool = False,
) -> dict:
    if not all_trials and num_trials <= 0:
        raise ValueError(f"EVALUATION.num_trials must be positive, got {num_trials}")

    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    manifest = {
        "version": 1,
        "source": str(classification_path.resolve()),
        "seed": seed,
        "evaluation_mode": "all" if all_trials else "sampled",
        "num_trials_per_original_task": None if all_trials else num_trials,
        "perturbation_categories": PERTURBATION_CATEGORIES,
        "tasks": {},
    }

    for suite, base_names in ORIGINAL_TASKS.items():
        rows = classification.get(suite)
        if not isinstance(rows, list):
            raise ValueError(f"Missing classification rows for suite: {suite}")

        normalized_rows = []
        for index, row in enumerate(rows):
            if int(row["id"]) != index + 1:
                raise ValueError(
                    f"Expected one-based sequential IDs in {suite}; got id={row['id']} at row {index}."
                )
            item = dict(row)
            item["variant_task_id"] = index
            normalized_rows.append(item)

        grouped = {base_id: [] for base_id in range(len(base_names))}
        for row in normalized_rows:
            matches = [
                base_id
                for base_id, base_name in enumerate(base_names)
                if row["name"] == base_name or row["name"].startswith(base_name + "_")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one original task match for {suite}/{row['name']}, got {matches}."
                )
            grouped[matches[0]].append(row)

        suite_manifest = {}
        for base_id, base_name in enumerate(base_names):
            by_category = defaultdict(list)
            for row in grouped[base_id]:
                by_category[row["category"]].append(row)

            missing = [category for category in PERTURBATION_CATEGORIES if not by_category[category]]
            if missing:
                raise ValueError(f"{suite} task {base_id} has no variants for: {missing}")

            if all_trials:
                selected = []
                for row in sorted(grouped[base_id], key=lambda item: int(item["variant_task_id"])):
                    item = dict(row)
                    item["repeat_index"] = 0
                    selected.append(item)
            else:
                quotas = {
                    category: num_trials // len(PERTURBATION_CATEGORIES)
                    for category in PERTURBATION_CATEGORIES
                }
                remainder = num_trials % len(PERTURBATION_CATEGORIES)
                offset = _stable_seed(seed, suite, base_id, "quota") % len(PERTURBATION_CATEGORIES)
                for index in range(remainder):
                    category = PERTURBATION_CATEGORIES[(offset + index) % len(PERTURBATION_CATEGORIES)]
                    quotas[category] += 1

                selected = []
                for category in PERTURBATION_CATEGORIES:
                    candidates = sorted(
                        by_category[category], key=lambda row: int(row["variant_task_id"])
                    )
                    selected.extend(
                        _sample_category(
                            candidates,
                            quotas[category],
                            seed=_stable_seed(seed, suite, base_id, category),
                        )
                    )
                random.Random(_stable_seed(seed, suite, base_id, "order")).shuffle(selected)
            category_counts = Counter(item["category"] for item in selected)
            suite_manifest[str(base_id)] = {
                "base_task_id": base_id,
                "base_task_name": base_name,
                "available_variants": len(grouped[base_id]),
                "available_by_category": dict(Counter(row["category"] for row in grouped[base_id])),
                "sampled_by_category": dict(category_counts),
                "variants": selected,
            }

        manifest["tasks"][suite] = suite_manifest

    manifest["total_trials"] = sum(
        len(task["variants"])
        for suite in manifest["tasks"].values()
        for task in suite.values()
    )
    return manifest


def create_task_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for suite in ORIGINAL_TASKS:
            for base_task_id in range(len(ORIGINAL_TASKS[suite])):
                handle.write(f"{suite},{base_task_id}\n")
    return path


@hydra.main(version_base="1.3", config_path="../../settings", config_name="libero_plus.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")
    if cfg.EVALUATION.libero_plus_root is None:
        raise ValueError("Pass EVALUATION.libero_plus_root=/path/to/third_party/LIBERO-plus.")

    plus_root = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.libero_plus_root)))).resolve()
    classification_path = plus_root / "libero/libero/benchmark/task_classification.json"
    if not classification_path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus classification file not found: {classification_path}")

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "perturbation_manifest.json"
    all_trials = bool(cfg.EVALUATION.get("all_trials", False))
    manifest = build_manifest(
        classification_path,
        num_trials=int(cfg.EVALUATION.num_trials),
        seed=int(cfg.EVALUATION.perturbation_seed),
        all_trials=all_trials,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    task_file = create_task_file(output_dir / "tasks.txt")
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    print(f"Perturbation manifest: {manifest_path}")
    print(f"Original tasks: {sum(len(tasks) for tasks in ORIGINAL_TASKS.values())}")
    if all_trials:
        print(f"Evaluation mode: all variants ({manifest['total_trials']} trials)")
    else:
        print(f"Trials per original task: {cfg.EVALUATION.num_trials}")

    if bool(cfg.MULTIRUN.get("create_only", False)):
        return

    overrides = [
        override
        for override in collect_worker_overrides()
        if not override.lstrip("+~").startswith("EVALUATION.all_trials=")
    ]
    overrides.append(f"EVALUATION.all_trials={str(all_trials).lower()}")
    overrides.append(f"EVALUATION.perturbation_manifest_path={manifest_path}")
    run_evaluation(
        task_file=task_file,
        task_choice=_resolve_worker_task_choice(),
        ckpt=str(cfg.ckpt),
        num_gpus=int(cfg.MULTIRUN.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(cfg.MULTIRUN.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=overrides,
        eval_entrypoint="benchmarks/libero_plus/evaluate_task.py",
        summary_entrypoint="benchmarks/libero_plus/summarize.py",
        trial_plan=(
            f"{manifest['total_trials']} total variants"
            if all_trials
            else f"{cfg.EVALUATION.num_trials} per original task"
        ),
    )


if __name__ == "__main__":
    main()
