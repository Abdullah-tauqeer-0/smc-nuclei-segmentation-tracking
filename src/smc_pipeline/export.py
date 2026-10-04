"""Write tracks to a tidy CSV (one row per track and frame)."""
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

FIELDS = ["track_id", "frame", "x", "y", "area", "eccentricity", "perimeter", "roundness"]


def write_tracks_csv(tracks: Dict[int, List[Optional[dict]]], frame_numbers: Sequence[int], path) -> int:
    """Save tracks; frames where a track has no detection are omitted. Returns row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for tid, trk in tracks.items():
            for i, cell in enumerate(trk):
                if cell is None:
                    continue
                x, y = cell["centroid"]
                writer.writerow(
                    [tid, frame_numbers[i], x, y, cell["area"], cell["eccentricity"],
                     cell["perimeter"], cell["roundness"]]
                )
                rows += 1
    return rows
