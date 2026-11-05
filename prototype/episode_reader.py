"""Project the relevant columns from a single local Parquet shard."""

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .metadata_reader import InspectionError, shard_path


INDEX_COLUMNS = ("episode_index", "index", "frame_index", "timestamp")
EPISODE_COLUMNS = (
    "episode_index", "length", "dataset_from_index", "dataset_to_index",
    "data/chunk_index", "data/file_index",
)


@dataclass
class FrameBatch:
    episode_index: int
    view_row_start: int
    data: dict


def read_episodes(root, episode_index=None):
    directory = root / "meta/episodes"
    shards = sorted(directory.glob("chunk-*/file-*.parquet"))
    if not shards:
        raise InspectionError("MISSING_METADATA", f"No Episode shards in {directory}")

    matches = []
    for shard in shards:
        try:
            with pq.ParquetFile(shard) as parquet:
                missing = set(EPISODE_COLUMNS) - set(parquet.schema_arrow.names)
                if missing:
                    raise InspectionError(
                        "METADATA_READ_ERROR", f"{shard}: missing columns {sorted(missing)}"
                    )
                columns = [
                    name for name in parquet.schema_arrow.names
                    if name in EPISODE_COLUMNS or name.startswith("videos/")
                ]
            filters = None if episode_index is None else [("episode_index", "=", episode_index)]
            table = pq.read_table(shard, columns=columns, filters=filters)
            matches.extend(table.to_pylist())
        except (OSError, pa.ArrowException) as error:
            raise InspectionError("METADATA_READ_ERROR", f"{shard}: {error}") from error

    return matches


def read_episode(root, episode_index):
    matches = read_episodes(root, episode_index)
    if len(matches) != 1:
        raise InspectionError(
            "EPISODE_INDEX_MISMATCH",
            f"Expected one metadata record for Episode {episode_index}, found {len(matches)}",
        )
    return validate_episode_metadata(matches[0])


def validate_episode_metadata(episode):
    start, end, length = (
        episode["dataset_from_index"], episode["dataset_to_index"], episode["length"]
    )
    if (
        any(type(episode[key]) is not int or episode[key] < 0 for key in EPISODE_COLUMNS)
        or length == 0
        or end - start != length
    ):
        raise InspectionError(
            "EPISODE_INDEX_MISMATCH", f"Episode {episode['episode_index']!r}: invalid index range or length"
        )
    return episode


def data_path(root, info, episode):
    path = shard_path(
        root, info.get("data_path"),
        chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"],
    )
    if not path.is_file():
        raise InspectionError("MISSING_DATA_SHARD", str(path))
    return path


def iter_frame_batches(root, info, episode, features, batch_size):
    path = data_path(root, info, episode)

    episode_index = episode["episode_index"]
    # Include rows claimed by either identifier so contradictory IDs remain visible.
    index = ds.field("index")
    in_range = (index >= episode["dataset_from_index"]) & (index < episode["dataset_to_index"])
    selected = (ds.field("episode_index") == episode_index) | in_range
    try:
        table = ds.dataset(str(path), format="parquet")
        columns = [*INDEX_COLUMNS, *features]
        missing = set(columns) - set(table.schema.names)
        if missing:
            raise InspectionError("DATA_READ_ERROR", f"{path}: missing columns {sorted(missing)}")
        for name, spec in features.items():
            dtype = table.schema.field(name).type
            while pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
                dtype = dtype.value_type
            if str(dtype) != {"float32": "float", "float64": "double", "float16": "halffloat"}.get(
                spec["dtype"], spec["dtype"]
            ):
                raise InspectionError(
                    "FEATURE_SCHEMA_ERROR", f"{path}: {name} declares {spec['dtype']}, found {dtype}"
                )
        view_row = 0
        for batch in table.to_batches(
            columns=columns, filter=selected, batch_size=batch_size, use_threads=False
        ):
            if batch.num_rows:
                yield FrameBatch(episode_index, view_row, batch.to_pydict())
                view_row += batch.num_rows
    except (OSError, pa.ArrowException) as error:
        raise InspectionError("DATA_READ_ERROR", f"{path}: {error}") from error


def video_availability(root, info, episode):
    video_keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    if not video_keys:
        return "NOT_DECLARED"
    for key in video_keys:
        chunk = episode.get(f"videos/{key}/chunk_index")
        file = episode.get(f"videos/{key}/file_index")
        if chunk is None or file is None:
            return "UNKNOWN"
        if not shard_path(
            root, info.get("video_path"), video_key=key, chunk_index=chunk, file_index=file
        ).is_file():
            return "MISSING"
    return "PRESENT"


class LeRobotV3Adapter:
    """Local LeRobot storage access used by the inspection engine."""

    def __init__(self, root, info):
        self.root = root
        self.info = info

    def episodes(self, episode_index=None):
        if episode_index is not None:
            return [read_episode(self.root, episode_index)]
        if self.info["total_episodes"] == 0 and not (self.root / "meta/episodes").exists():
            return []
        return read_episodes(self.root)

    def batches(self, episode, features, batch_size):
        return iter_frame_batches(self.root, self.info, episode, features, batch_size)

    def video_availability(self, episode):
        return video_availability(self.root, self.info, episode)

    def data_path(self, episode):
        return data_path(self.root, self.info, episode)

    def shard_issues(self, expected_counts):
        paths = {path.resolve() for path in (self.root / "data").glob("chunk-*/file-*.parquet")}
        for path in sorted(paths - expected_counts.keys()):
            yield ("UNREFERENCED_DATA_SHARD", str(path))
        for path, expected in sorted(expected_counts.items()):
            try:
                actual = pq.read_metadata(path).num_rows
            except (OSError, pa.ArrowException) as error:
                yield ("DATA_READ_ERROR", f"{path}: {error}")
                continue
            if actual != expected:
                yield ("SHARD_FRAME_COUNT_ERROR", f"{path}: expected {expected} rows, found {actual}")
