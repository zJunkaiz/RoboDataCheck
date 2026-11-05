"""Create a tiny local LeRobot v3.0 example in a new directory."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True)
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    frame_path = root / "data/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    frame_path.parent.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "fps": 10, "total_episodes": 2, "total_frames": 4,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {"action": {"dtype": "float32", "shape": [2]},
                     "observation.state": {"dtype": "float32", "shape": [2]}},
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    pq.write_table(pa.table({
        "episode_index": [0, 1], "length": [2, 2],
        "dataset_from_index": [0, 2], "dataset_to_index": [2, 4],
        "data/chunk_index": [0, 0], "data/file_index": [0, 0],
    }), episode_path)
    vectors = pa.array([[0, 1], [1, 1], [2, 0], [3, 0]], type=pa.list_(pa.float32(), 2))
    pq.write_table(pa.table({
        "episode_index": [0, 0, 1, 1], "index": [0, 1, 2, 3],
        "frame_index": [0, 1, 0, 1], "timestamp": [0.0, 0.1, 0.0, 0.1],
        "action": vectors, "observation.state": vectors,
    }), frame_path)


if __name__ == "__main__":
    main()
