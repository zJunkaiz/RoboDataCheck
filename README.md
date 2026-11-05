# RoboDataCheck prototype

RoboDataCheck verifies the stored frame indices, timeline, and finite action and state values for a single Episode in a local LeRobot v3.0 dataset. It reads the target Parquet shard in batches using PyArrow. It reports video file availability and skips video validation.

## Run

From the project directory with Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m prototype.run_prototype info D:/path/to/dataset
python -m prototype.run_prototype inspect D:/path/to/dataset --episode 1 --batch-size 64 --json
python -m unittest discover -s tests -v
```

`info` reads `meta/info.json`. `inspect` also reads the Episode metadata and target data shard. It exits with code 1 and a diagnostic code on the first failed check. The timestamp check compares stored values and adjacent intervals with the declared FPS, using a configurable absolute tolerance in seconds. The default is 0.0001 s.

This prototype reads LeRobot's storage format directly and does not require the LeRobot Python package or a Hub client. Local inspection makes no downloads. Timeline results describe the stored sequence, whose timestamps can be generated from frame numbers, and do not measure synchronization between physical sensors.

## Reproduce the public-data check

Use the [pinned SO-101 pick-and-place snapshot](https://huggingface.co/datasets/lerobot/svla_so101_pickplace/tree/f641879e22172be7e8161d5e6c1503c2d2feb657). Download the three files below and keep their relative paths under the dataset directory.

- `meta/info.json`
- `meta/episodes/chunk-000/file-000.parquet`
- `data/chunk-000/file-000.parquet`

Episode 1 contains 266 frames. The selected view spans rows 0 through 265, the stored global indices span 303 through 568, and the stored Episode frame indices span 0 through 265. With this snapshot's video files omitted, the expected result is `numeric_validation: PASS`, `video_availability: MISSING`, and `video_validation: SKIPPED`.
