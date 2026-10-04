import json

import numpy as np

from smc_pipeline.annotations import build_annotation, remove_small_objects
from smc_pipeline.export import write_tracks_csv
from smc_pipeline.tracking import TrackerConfig, load_frames, track


def disk_mask(centres, radius=8, size=128):
    yy, xx = np.mgrid[:size, :size]
    mask = np.zeros((size, size), np.uint8)
    for cx, cy in centres:
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = 1
    return mask


def test_remove_small_objects_drops_specks():
    mask = disk_mask([(30, 30)])
    mask[100, 100] = 1
    cleaned = remove_small_objects(mask, min_area=80)
    assert cleaned[100, 100] == 0 and cleaned[30, 30] == 1


def test_annotation_of_disk():
    ann = build_annotation("x.tif", disk_mask([(40, 40)], radius=10))
    (cell,) = ann["cells"]
    assert abs(cell["centroid"][0] - 40) < 0.5
    assert cell["eccentricity"] < 0.1 and cell["roundness"] > 0.8


def test_tracking_follows_moving_nuclei(tmp_path):
    for frame in range(5):
        centres = [(30 + 2 * frame, 40), (90, 90 - frame)]
        ann = build_annotation(f"f{frame}", disk_mask(centres))
        (tmp_path / f"W1_{frame:03d}.json").write_text(json.dumps(ann))

    frames = load_frames(tmp_path, "W1")
    tracks = track(frames, TrackerConfig())
    assert len(tracks) == 2
    assert all(None not in t for t in tracks.values())

    out = tmp_path / "tracks.csv"
    assert write_tracks_csv(tracks, list(frames), out) == 10
