"""Read the metadata required by the local LeRobot v3.0 prototype."""

import json
import math
from pathlib import Path


class InspectionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def read_info(root: Path) -> dict:
    path = root / "meta/info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InspectionError("MISSING_METADATA", str(path)) from error
    except (OSError, ValueError) as error:
        raise InspectionError("METADATA_READ_ERROR", f"{path}: {error}") from error

    if not isinstance(info, dict):
        raise InspectionError("METADATA_READ_ERROR", f"{path}: expected a JSON object")
    version = info.get("codebase_version")
    if version is None:
        raise InspectionError("METADATA_READ_ERROR", f"{path}: missing codebase_version")
    if version != "v3.0":
        raise InspectionError("UNSUPPORTED_FORMAT", f"{path}: expected v3.0, got {version!r}")

    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise InspectionError("METADATA_READ_ERROR", f"{path}: invalid fps")
    for key in ("total_episodes", "total_frames"):
        if type(info.get(key)) is not int or info[key] < 0:
            raise InspectionError("METADATA_READ_ERROR", f"{path}: invalid {key}")
    features = info.get("features")
    if not isinstance(features, dict) or any(
        not isinstance(spec, dict) or not isinstance(spec.get("dtype"), str)
        for spec in features.values()
    ):
        raise InspectionError("METADATA_READ_ERROR", f"{path}: invalid features")
    return info


def shard_path(root, template, **values):
    try:
        relative = Path(template.format(**values))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("shard path must stay inside the dataset")
        return path
    except (AttributeError, KeyError, IndexError, OSError, ValueError) as error:
        raise InspectionError("METADATA_READ_ERROR", f"Invalid shard path: {error}") from error
