# RoboDataCheck v0.1.0

RoboDataCheck inspects local LeRobot v3.0 datasets before training. It checks dataset and Episode counts, index ranges, shard row counts, frame continuity, and timestamp cadence. For declared action and state features it checks Parquet types, per-frame shapes, and finite values. Each passing Episode includes action minima, maxima, and constant dimensions for scalar and vector actions.

## Run

From the project directory with Python 3.11 or newer:

```powershell
python -m pip install .
python examples/make_demo.py demo_dataset
robodatacheck info demo_dataset
robodatacheck inspect demo_dataset --output reports
robodatacheck inspect demo_dataset --episode 1 --batch-size 2 --json
python -m unittest discover -s tests -v
```

`info` reads `meta/info.json`. `inspect` checks all Episodes by default and continues after individual Episode failures. Each Episode reports its first failing check. An unreadable or inconsistent Episode catalog stops the scan. The command returns exit code 1 for failed checks and writes `report.json` and `report.html` when given `--output`. Use `--episode` to inspect a single Episode, or `--tolerance` to change the timestamp tolerance from its default of 0.0001 seconds.

Inspection uses local Parquet reads through PyArrow. Video checks are limited to file availability. Timeline checks cover the stored sequence. Cross-sensor synchronization, video decoding, and statistical outlier detection are outside the v0.1.0 scope.

## Reproduce the public-data check

Use the [pinned SO-101 pick-and-place snapshot](https://huggingface.co/datasets/lerobot/svla_so101_pickplace/tree/f641879e22172be7e8161d5e6c1503c2d2feb657). Download the three files below and keep their relative paths under the dataset directory.

- `meta/info.json`
- `meta/episodes/chunk-000/file-000.parquet`
- `data/chunk-000/file-000.parquet`

Episode 1 contains 266 frames. The selected view spans rows 0 through 265, the stored global indices span 303 through 568, and the stored Episode frame indices span 0 through 265. With this snapshot's video files omitted, the expected result is `numeric_validation: PASS`, `video_availability: MISSING`, and `video_validation: SKIPPED`.

A full scan of this snapshot should pass all 50 Episodes and 11,939 frames.
