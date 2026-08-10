#!/usr/bin/env python3
"""Apply compatibility edits required by the RoboTwin release evaluator."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _module_file(module: str, relative: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"Python package is not installed: {module}")
    return Path(next(iter(spec.submodule_search_locations))) / relative


def _patch(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in replacements:
        if new in updated:
            continue
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print(f"patched: {path}")
    else:
        print(f"ready:   {path}")
    for _, expected in replacements:
        if expected not in updated:
            raise RuntimeError(
                f"Unsupported file layout; expected text not found in {path}: {expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, default=Path("third_party/RoboTwin"))
    args = parser.parse_args()

    _patch(
        _module_file("sapien", "wrapper/urdf_loader.py"),
        [
            (
                'with open(urdf_file, "r") as f:',
                'with open(urdf_file, "r", encoding="utf-8") as f:',
            ),
            ('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"'),
            (
                'with open(srdf_file, "r") as f:',
                'with open(srdf_file, "r", encoding="utf-8") as f:',
            ),
        ],
    )
    _patch(
        _module_file("mplib", "planner.py"),
        [
            (
                "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
                "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
            )
        ],
    )
    _patch(
        _module_file("curobo", "geom/sdf/world_mesh.py"),
        [("wp.torch.device_from_torch", "wp.device_from_torch")],
    )
    base_task = args.robotwin_root.expanduser().resolve() / "envs" / "_base_task.py"
    if not base_task.is_file():
        raise FileNotFoundError(f"RoboTwin base task not found: {base_task}")
    _patch(
        base_task,
        [
            (
                r'        print(f"step: \033[92m{self.take_action_cnt} / {self.step_lim}\033[0m", end="\r")',
                "        # Keep per-action steps silent; episode summaries report evaluation progress.",
            )
        ],
    )


    eval_policy = args.robotwin_root.expanduser().resolve() / "script" / "eval_policy.py"
    if not eval_policy.is_file():
        raise FileNotFoundError(f"RoboTwin evaluator not found: {eval_policy}")
    _patch(
        eval_policy,
        [
            (
                "    test_num = 100\n",
                '    test_num = int(usr_args.get("eval_num_episodes", 100))\n',
            ),
            (
                '    args["ckpt_setting"] = ckpt_setting\n\n'
                "    embodiment_type = args.get(\"embodiment\")",
                '    args["ckpt_setting"] = ckpt_setting\n'
                '    args["environment_test"] = bool(usr_args.get("environment_test", False))\n\n'
                "    embodiment_type = args.get(\"embodiment\")",
            ),
            (
                "            except Exception as e:\n"
                "                # stack_trace = traceback.format_exc()",
                "            except Exception as e:\n"
                "                if args.get(\"environment_test\", False):\n"
                "                    raise\n"
                "                # stack_trace = traceback.format_exc()",
            ),
            (
                '    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")\n'
                "    save_dir.mkdir(parents=True, exist_ok=True)",
                '    requested_output_dir = usr_args.get("eval_output_dir")\n'
                "    if requested_output_dir:\n"
                "        manager_output_dir = Path(requested_output_dir)\n"
                "        save_dir = manager_output_dir / task_config\n"
                "    else:\n"
                "        manager_output_dir = None\n"
                '        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")\n'
                "    save_dir.mkdir(parents=True, exist_ok=True)",
            ),
            (
                '    file_path = os.path.join(save_dir, f"_result.txt")',
                '    result_name = {"demo_clean": "_result_clean.txt", "demo_randomized": "_result_random.txt"}.get(task_config, "_result.txt")\n'
                "    result_dir = manager_output_dir if manager_output_dir is not None else save_dir\n"
                "    file_path = os.path.join(result_dir, result_name)",
            ),
        ],
    )


if __name__ == "__main__":
    main()
