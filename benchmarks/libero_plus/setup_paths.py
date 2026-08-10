import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated LIBERO-Plus path configuration.")
    parser.add_argument("--libero-plus-root", type=Path, default=Path("third_party/LIBERO-plus"))
    parser.add_argument("--config-dir", type=Path, default=Path(".libero_plus"))
    args = parser.parse_args()

    plus_root = args.libero_plus_root.expanduser().resolve()
    benchmark_root = plus_root / "libero/libero"
    required = [
        benchmark_root / "benchmark/task_classification.json",
        benchmark_root / "bddl_files",
        benchmark_root / "init_files",
        benchmark_root / "assets",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LIBERO-Plus files:\n" + "\n".join(missing))

    config_dir = args.config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(plus_root / "libero/datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"Wrote {config_path}")
    print(f'export LIBERO_CONFIG_PATH="{config_dir}"')


if __name__ == "__main__":
    main()
