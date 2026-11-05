"""Inspect a local dataset directory and emit a terminal or JSON report."""

import argparse
import json
import math
from pathlib import Path
import sys

from .metadata_reader import InspectionError, read_info


def inspect(root, episode_index, batch_size, tolerance):
    from .episode_reader import iter_frame_batches, read_episode, video_availability
    from .index_validator import validate_episode

    info = read_info(root)
    if episode_index >= info["total_episodes"]:
        raise InspectionError(
            "EPISODE_INDEX_MISMATCH", f"Episode {episode_index}: outside total_episodes"
        )
    episode = read_episode(root, episode_index)
    if episode["dataset_to_index"] > info["total_frames"]:
        raise InspectionError(
            "EPISODE_INDEX_MISMATCH", f"Episode {episode_index}: range exceeds total_frames"
        )
    features = [key for key in ("action", "observation.state") if key in info["features"]]
    availability = video_availability(root, info, episode)
    batches = iter_frame_batches(root, info, episode, features, batch_size)
    result = validate_episode(episode, batches, info["fps"], features, tolerance)
    return {
        "status": "PASS",
        "dataset": str(root),
        "format": info["codebase_version"],
        "episode_index": episode_index,
        "fps": info["fps"],
        **result,
        "timeline_validation": "PASS",
        "numeric_features": features,
        "numeric_validation": "PASS" if features else "SKIPPED",
        "video_availability": availability,
        "video_validation": "SKIPPED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("info", "inspect"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Timestamp tolerance in seconds")
    parser.add_argument("--json", action="store_true", help="Print a JSON report")
    args = parser.parse_args(argv)
    if args.episode < 0 or args.batch_size < 1:
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
        else:
            report = inspect(args.root, args.episode, args.batch_size, args.tolerance)
    except InspectionError as error:
        report = {"status": "ERROR", "code": error.code, "message": str(error)}

    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        stream = sys.stderr if report["status"] == "ERROR" else sys.stdout
        for key, value in report.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}", file=stream)
    return int(report["status"] == "ERROR")


if __name__ == "__main__":
    raise SystemExit(main())
