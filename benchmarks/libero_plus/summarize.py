import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _empty_stats() -> dict:
    return {"successes": 0, "total_episodes": 0, "duration": 0.0}


def _finish(stats: dict) -> dict:
    total = int(stats["total_episodes"])
    stats["success_rate"] = float(stats["successes"] / total) if total else 0.0
    return stats


def summarize(output_dir: Path) -> dict:
    category_stats = defaultdict(lambda: {"successes": 0, "total_episodes": 0})
    suite_stats = defaultdict(_empty_stats)
    task_stats = {}
    overall = _empty_stats()

    for suite in SUITES:
        suite_dir = output_dir / suite
        if not suite_dir.is_dir():
            continue
        for result_path in sorted(suite_dir.glob("gpu*_task*_results.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            successes = int(result["successes"])
            total = int(result["total_episodes"])
            duration = float(result["duration"])

            for target in (suite_stats[suite], overall):
                target["successes"] += successes
                target["total_episodes"] += total
                target["duration"] += duration

            task_key = f"{suite}_task{int(result['task_id'])}"
            task_stats[task_key] = {
                "task_description": result.get("task_description", ""),
                "successes": successes,
                "total_episodes": total,
                "success_rate": successes / total if total else 0.0,
                "duration": duration,
            }

            for category, stats in result.get("category_stats", {}).items():
                category_stats[category]["successes"] += int(stats["successes"])
                category_stats[category]["total_episodes"] += int(stats["total_episodes"])

    summary = {
        "benchmark": "LIBERO-Plus",
        "suite_stats": {key: _finish(value) for key, value in suite_stats.items()},
        "category_stats": {key: _finish(value) for key, value in category_stats.items()},
        "task_stats": task_stats,
        "overall": _finish(overall),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    summary = summarize(output_dir)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "category_success_rates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Perturbation", "Successes", "Trials", "Success Rate (%)"])
        for category, stats in sorted(summary["category_stats"].items()):
            writer.writerow(
                [
                    category,
                    stats["successes"],
                    stats["total_episodes"],
                    f"{100.0 * stats['success_rate']:.2f}",
                ]
            )

    print("\nLIBERO-Plus category results:")
    for category, stats in sorted(summary["category_stats"].items()):
        print(
            f"- {category}: {stats['successes']}/{stats['total_episodes']} "
            f"({100.0 * stats['success_rate']:.2f}%)"
        )
    overall = summary["overall"]
    print(
        f"Overall: {overall['successes']}/{overall['total_episodes']} "
        f"({100.0 * overall['success_rate']:.2f}%)"
    )
    print(f"Summary: {summary_path}")
    print(f"Category CSV: {csv_path}")


if __name__ == "__main__":
    main()
