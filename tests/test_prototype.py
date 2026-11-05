import contextlib
import io
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from prototype.metadata_reader import InspectionError
from prototype.run_prototype import inspect, main


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(TemporaryDirectory())
        self.root = Path(tmp)
        self.network = self.enterContext(
            patch.object(socket, "socket", side_effect=AssertionError("Local mode opened a socket"))
        )
        info = {
            "codebase_version": "v3.0",
            "fps": 10,
            "total_episodes": 2,
            "total_frames": 5,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"action": {"dtype": "float32"}, "observation.state": {"dtype": "float32"},
                         "observation.images.front": {"dtype": "video"}},
        }
        meta = self.root / "meta/episodes/chunk-000/file-000.parquet"
        meta.parent.mkdir(parents=True)
        (self.root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        pq.write_table(pa.table({
            "episode_index": [0, 1], "length": [2, 3],
            "dataset_from_index": [0, 2], "dataset_to_index": [2, 5],
            "data/chunk_index": [0, 0], "data/file_index": [0, 0],
            "videos/observation.images.front/chunk_index": [0, 0],
            "videos/observation.images.front/file_index": [0, 0],
        }), meta)
        self.shard = self.root / "data/chunk-000/file-000.parquet"
        self.shard.parent.mkdir(parents=True)
        self.write_frames()
        self.addCleanup(self.network.assert_not_called)

    def write_frames(self, timestamps=None, episode_ids=None, actions=None):
        pq.write_table(pa.table({
            "episode_index": episode_ids or [0, 0, 1, 1, 1],
            "index": [0, 1, 2, 3, 4],
            "frame_index": [0, 1, 0, 1, 2],
            "timestamp": timestamps or [0.0, 0.1, 0.0, 0.1, 0.2],
            "action": actions or [[0.0]] * 5,
            "observation.state": [[1.0]] * 5,
        }), self.shard)

    def test_info_reads_metadata_without_data_shard(self):
        self.shard.unlink()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["info", str(self.root), "--json"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["total_episodes"], 2)

    def test_nonzero_episode_passes_after_removing_video(self):
        video = self.root / "videos/observation.images.front/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
        self.assertEqual(inspect(self.root, 1, 2, 1e-4)["video_availability"], "PRESENT")
        video.unlink()
        report = inspect(self.root, 1, 2, 1e-4)
        self.assertEqual(report["frames"], 3)
        self.assertEqual(report["first_frame"],
                         {"view_row_index": 0, "index": 2, "frame_index": 0, "timestamp": 0.0})
        self.assertEqual(report["last_frame"],
                         {"view_row_index": 2, "index": 4, "frame_index": 2, "timestamp": 0.2})
        self.assertEqual(report["numeric_validation"], "PASS")
        self.assertEqual(report["video_availability"], "MISSING")
        self.assertEqual(report["video_validation"], "SKIPPED")

    def test_missing_shard_reports_path(self):
        self.shard.unlink()
        with self.assertRaises(InspectionError) as error:
            inspect(self.root, 1, 2, 1e-4)
        self.assertEqual(error.exception.code, "MISSING_DATA_SHARD")
        self.assertEqual(str(error.exception), str(self.shard.resolve()))

    def test_duplicate_timestamp_across_batches(self):
        self.write_frames(timestamps=[0.0, 0.1, 0.0, 0.1, 0.1])
        with self.assertRaisesRegex(InspectionError, "view row 2: duplicate timestamp") as error:
            inspect(self.root, 1, 2, 1e-4)
        self.assertEqual(error.exception.code, "TIMESTAMP_ERROR")

    def test_conflicting_episode_id_is_visible_after_filtering(self):
        self.write_frames(episode_ids=[0, 0, 1, 0, 1])
        with self.assertRaisesRegex(InspectionError, "view row 1: wrong Episode ID") as error:
            inspect(self.root, 1, 2, 1e-4)
        self.assertEqual(error.exception.code, "EPISODE_INDEX_MISMATCH")

    def test_nonfinite_action_reports_feature_and_row(self):
        self.write_frames(actions=[[0.0]] * 4 + [[float("nan")]])
        with self.assertRaisesRegex(InspectionError, "view row 2: action") as error:
            inspect(self.root, 1, 2, 1e-4)
        self.assertEqual(error.exception.code, "NUMERIC_ERROR")
