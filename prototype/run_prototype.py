"""Inspect a local dataset directory and emit a terminal or JSON report."""

import argparse
from collections import Counter
from html import escape
import json
import math
from pathlib import Path
import sys

from .metadata_reader import InspectionError, read_info


def inspect(root, episode_index, batch_size, tolerance):
    from .episode_reader import LeRobotV3Adapter

    info = read_info(root)
    if episode_index >= info["total_episodes"]:
        raise InspectionError(
            "EPISODE_INDEX_MISMATCH", f"Episode {episode_index}: outside total_episodes"
        )
    adapter = LeRobotV3Adapter(root.resolve(), info)
    return inspect_episode(adapter, adapter.episodes(episode_index)[0], batch_size, tolerance)


def inspect_episode(adapter, episode, batch_size, tolerance):
    from .index_validator import validate_episode

    info = adapter.info
    episode_index = episode["episode_index"]
    if episode["dataset_to_index"] > info["total_frames"]:
        raise InspectionError(
            "EPISODE_INDEX_MISMATCH", f"Episode {episode_index}: range exceeds total_frames"
        )
    features = {
        key: info["features"][key]
        for key in ("action", "observation.state") if key in info["features"]
    }
    availability = adapter.video_availability(episode)
    batches = adapter.batches(episode, features, batch_size)
    result = validate_episode(episode, batches, info["fps"], features, tolerance)
    return {
        "status": "PASS",
        "dataset": str(adapter.root),
        "format": info["codebase_version"],
        "episode_index": episode_index,
        "fps": info["fps"],
        **result,
        "timeline_validation": "PASS",
        "numeric_features": list(features),
        "numeric_validation": "PASS" if features else "SKIPPED",
        "video_availability": availability,
        "video_validation": "SKIPPED",
    }


def inspect_dataset(root, batch_size, tolerance):
    from .episode_reader import LeRobotV3Adapter, validate_episode_metadata

    info = read_info(root)
    adapter = LeRobotV3Adapter(root.resolve(), info)
    episodes = adapter.episodes()
    by_id = {}
    for episode in episodes:
        validate_episode_metadata(episode)
        episode_id = episode["episode_index"]
        if episode_id in by_id:
            raise InspectionError("EPISODE_INDEX_MISMATCH", f"Duplicate Episode {episode_id}")
        by_id[episode_id] = episode
    if len(by_id) != info["total_episodes"] or any(
        episode_id != expected for expected, episode_id in enumerate(sorted(by_id))
    ):
        raise InspectionError("EPISODE_INDEX_MISMATCH", "Episode IDs do not match total_episodes")

    reports = []
    issues = []
    expected_counts = Counter()
    next_index = 0
    for episode_id, episode in sorted(by_id.items()):
        if episode["dataset_from_index"] != next_index:
            issues.append({"code": "EPISODE_RANGE_ERROR",
                           "message": f"Episode {episode_id}: expected start index {next_index}"})
        next_index = episode["dataset_to_index"]
        try:
            expected_counts[adapter.data_path(episode)] += episode["length"]
            reports.append(inspect_episode(adapter, episode, batch_size, tolerance))
        except InspectionError as error:
            reports.append({"episode_index": episode_id, "status": "ERROR",
                            "code": error.code, "message": str(error)})
    if next_index != info["total_frames"] or sum(ep["length"] for ep in episodes) != info["total_frames"]:
        issues.append({"code": "DATASET_FRAME_COUNT_ERROR",
                       "message": "Episode ranges or lengths do not match total_frames"})
    issues.extend({"code": code, "message": message}
                  for code, message in adapter.shard_issues(expected_counts))
    failed = sum(report["status"] == "ERROR" for report in reports)
    return {"status": "ERROR" if failed or issues else "PASS", "dataset": str(root),
            "format": info["codebase_version"], "fps": info["fps"],
            "total_episodes": info["total_episodes"], "total_frames": info["total_frames"],
            "passed_episodes": len(reports) - failed, "failed_episodes": failed,
            "issues": issues, "episodes": reports}


def main(argv=None, *, default_episode=1):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("info", "inspect"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--episode", type=int, default=default_episode)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Timestamp tolerance in seconds")
    parser.add_argument("--json", action="store_true", help="Print a JSON report")
    parser.add_argument("--output", type=Path, help="Write report.json and report.html to this directory")
    args = parser.parse_args(argv)
    if (args.episode is not None and args.episode < 0) or args.batch_size < 1:
        parser.error("Episode must be non-negative and batch size must be positive")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("Timestamp tolerance must be finite and non-negative")

    try:
        if args.command == "info":
            info = read_info(args.root)
            report = {"status": "PASS", "dataset": str(args.root),
                      **{key: info[key] for key in ("codebase_version", "fps", "total_episodes",
                                                    "total_frames")},
                      "features": list(info["features"])}
        elif args.episode is not None:
            report = inspect(args.root, args.episode, args.batch_size, args.tolerance)
        else:
            report = inspect_dataset(args.root, args.batch_size, args.tolerance)
    except InspectionError as error:
        report = {"status": "ERROR", "code": error.code, "message": str(error)}

    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "report.json").write_text(serialized + "\n", encoding="utf-8")
            html = '<!doctype html><meta charset="utf-8"><title>RoboDataCheck report</title>'
            html += f"<h1>RoboDataCheck report</h1><pre>{escape(serialized)}</pre>"
            (args.output / "report.html").write_text(html + "\n", encoding="utf-8")
        except OSError as error:
            parser.exit(1, f"Could not write reports: {error}\n")
    if args.json:
        print(serialized)
    else:
        stream = sys.stderr if report["status"] == "ERROR" else sys.stdout
        for key, value in report.items():
            if key in ("episodes", "issues"):
                continue
            print(f"{key.replace('_', ' ').capitalize()}: {value}", file=stream)
        for item in [*report.get("issues", []), *report.get("episodes", [])]:
            if "code" in item:
                print(f"{item['code']}: {item['message']}", file=stream)
    return int(report["status"] == "ERROR")


def cli():
    return main(default_episode=None)


if __name__ == "__main__":
    raise SystemExit(main())
