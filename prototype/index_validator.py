"""Validate the selected Episode in view order across batch boundaries."""

from math import isclose, isfinite

from .metadata_reader import InspectionError


def finite_numeric(value):
    if isinstance(value, list):
        return bool(value) and all(finite_numeric(item) for item in value)
    return type(value) in (int, float) and isfinite(value)


def has_shape(value, shape):
    if not shape:
        return not isinstance(value, list)
    return isinstance(value, list) and len(value) == shape[0] and all(
        has_shape(item, shape[1:]) for item in value
    )


def validate_episode(episode, batches, fps, features, tolerance=1e-4):
    episode_id = episode["episode_index"]
    start = episode["dataset_from_index"]
    frame_count = 0
    previous_timestamp = None
    first = last = None
    action_min = action_max = None
    for batch in batches:
        columns = batch.data
        for offset, timestamp in enumerate(columns["timestamp"]):
            view_row = batch.view_row_start + offset
            location = f"Episode {episode_id}, view row {view_row}"
            row_episode = columns["episode_index"][offset]
            if type(row_episode) is not int or row_episode != episode_id:
                raise InspectionError("EPISODE_INDEX_MISMATCH", f"{location}: wrong Episode ID")
            index = columns["index"][offset]
            frame = columns["frame_index"][offset]
            if (
                type(frame) is not int or frame != frame_count
                or type(index) is not int or index != start + frame_count
            ):
                raise InspectionError(
                    "FRAME_INDEX_ERROR",
                    f"{location}: expected index={start + frame_count}, frame_index={frame_count}; "
                    f"got index={index!r}, frame_index={frame!r}",
                )
            if type(timestamp) not in (int, float) or not isfinite(timestamp):
                raise InspectionError("TIMESTAMP_ERROR", f"{location}: invalid timestamp")
            if previous_timestamp is not None:
                gap = timestamp - previous_timestamp
                if gap <= 0:
                    reason = "duplicate timestamp" if gap == 0 else "timestamp went backwards"
                    raise InspectionError("TIMESTAMP_ERROR", f"{location}: {reason}")
                if not isclose(gap, 1 / fps, rel_tol=0, abs_tol=tolerance):
                    raise InspectionError(
                        "TIMESTAMP_ERROR", f"{location}: interval {gap:g} s differs from 1/FPS"
                    )
            if not isclose(timestamp, frame / fps, rel_tol=0, abs_tol=tolerance):
                raise InspectionError(
                    "TIMESTAMP_ERROR", f"{location}: timestamp differs from frame_index/FPS"
                )
            for feature, spec in features.items():
                value = columns[feature][offset]
                if not finite_numeric(value):
                    raise InspectionError(
                        "NUMERIC_ERROR", f"{location}: {feature} contains a missing or non-finite value"
                    )
                shape = spec.get("shape")
                if shape is not None and not (
                    has_shape(value, shape) or (shape == [1] and not isinstance(value, list))
                ):
                    raise InspectionError(
                        "FEATURE_SCHEMA_ERROR", f"{location}: {feature} expected shape {shape}"
                    )
                if feature == "action":
                    vector = value if isinstance(value, list) else [value]
                    if all(type(item) in (int, float) for item in vector):
                        if action_min is None:
                            action_min, action_max = vector.copy(), vector.copy()
                        if len(vector) != len(action_min):
                            raise InspectionError(
                                "FEATURE_SCHEMA_ERROR", f"{location}: inconsistent action dimensions"
                            )
                        for dimension, item in enumerate(vector):
                            action_min[dimension] = min(action_min[dimension], item)
                            action_max[dimension] = max(action_max[dimension], item)
            last = {"view_row_index": view_row, "index": index, "frame_index": frame,
                    "timestamp": timestamp}
            if first is None:
                first = last
            frame_count += 1
            previous_timestamp = timestamp

    if frame_count != episode["length"]:
        raise InspectionError(
            "FRAME_INDEX_ERROR",
            f"Episode {episode_id}: expected {episode['length']} frames, found {frame_count}",
        )
    result = {"frames": frame_count, "first_frame": first, "last_frame": last}
    if action_min is not None:
        result["action_statistics"] = {
            "min": action_min,
            "max": action_max,
            "constant_dimensions": [
                i for i, (low, high) in enumerate(zip(action_min, action_max)) if low == high
            ],
        }
    return result
