"""Frame-to-frame nucleus tracking with a greedy, feature-weighted linking score."""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class TrackerConfig:
    """Linking parameters.

    radius_px: candidate search radius around the last known centroid (original
        code: 4 x magnification = 16 px at 4x).
    gap_tolerance: consecutive frames a track may go unmatched before it ends.
    w_*: weights of the linking score terms (see ``link_score``).
    """

    radius_px: float = 16.0
    gap_tolerance: int = 2
    w_centroid: float = 0.3
    w_area: float = 1.0
    w_shape: float = 1.0
    w_ecc: float = 0.3
    w_perimeter: float = 0.3
    w_roundness: float = 0.3


def load_frames(json_dir, prefix: str) -> Dict[int, dict]:
    """Load ``<prefix>_<NNN>.json`` annotation files, keyed by frame number (sorted)."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.json$", re.I)
    frames = {}
    for path in Path(json_dir).iterdir():
        m = pattern.match(path.name)
        if m:
            frames[int(m.group(1))] = json.loads(path.read_text())
    if not frames:
        raise FileNotFoundError(f"No files matching '{prefix}_<frame>.json' in {json_dir}")
    for ann in frames.values():
        for cell in ann["cells"]:
            outline = cell.get("outline")
            if outline:
                pts = np.asarray(outline, dtype=np.float32).reshape(-1, 1, 2)
                cell["hu_moments"] = cv2.HuMoments(cv2.moments(pts)).flatten()
            else:
                cell["hu_moments"] = None
    return dict(sorted(frames.items()))


def link_score(a: dict, b: dict, cfg: TrackerConfig) -> float:
    """Weighted dissimilarity between a cell and a candidate in the next frame (lower = better).

    Terms: centroid distance (px), relative area difference, Euclidean distance between
    Hu-moment vectors, and absolute differences in eccentricity, relative perimeter
    and roundness. Features are used unnormalised, as in the original analysis.
    """
    d_pos = float(np.linalg.norm(np.asarray(a["centroid"]) - np.asarray(b["centroid"])))
    d_area = abs(a["area"] - b["area"]) / (a["area"] or 1)
    d_ecc = abs(a["eccentricity"] - b["eccentricity"])
    d_per = abs(a["perimeter"] - b["perimeter"]) / (a["perimeter"] or 1)
    d_round = abs(a.get("roundness", 0) - b.get("roundness", 0))
    ha, hb = a.get("hu_moments"), b.get("hu_moments")
    d_shape = 0.0 if ha is None or hb is None else float(np.linalg.norm(ha - hb))
    return (
        cfg.w_centroid * d_pos
        + cfg.w_area * d_area
        + cfg.w_shape * d_shape
        + cfg.w_ecc * d_ecc
        + cfg.w_perimeter * d_per
        + cfg.w_roundness * d_round
    )


def track(frames: Dict[int, dict], cfg: Optional[TrackerConfig] = None) -> Dict[int, List[Optional[dict]]]:
    """Link cells across frames.

    Tracks are processed in creation order; each takes the lowest-scoring unclaimed
    candidate within ``radius_px`` (greedy assignment). Unmatched cells start new
    tracks. Returns {track_id: [cell or None per frame]}, where None marks frames
    before the track began or frames missed within the gap tolerance.
    """
    cfg = cfg or TrackerConfig()
    keys = list(frames)
    active = {}  # tid -> (track, gap_count, last_cell)
    finished = {}
    next_tid = 0

    for cell in frames[keys[0]]["cells"]:
        active[next_tid] = ([cell], 0, cell)
        next_tid += 1

    for f_idx in range(1, len(keys)):
        cells = frames[keys[f_idx]]["cells"]
        claimed = [False] * len(cells)
        tree = cKDTree(np.asarray([c["centroid"] for c in cells])) if cells else None

        survivors = {}
        for tid, (trk, gap, last) in active.items():
            if gap >= cfg.gap_tolerance:
                finished[tid] = trk
                continue
            best, best_score = None, None
            if tree is not None:
                for idx in tree.query_ball_point(last["centroid"], r=cfg.radius_px):
                    if claimed[idx]:
                        continue
                    score = link_score(last, cells[idx], cfg)
                    if best is None or score < best_score:
                        best, best_score = idx, score
            if best is not None:
                trk.append(cells[best])
                claimed[best] = True
                survivors[tid] = (trk, 0, cells[best])
            else:
                trk.append(None)
                survivors[tid] = (trk, gap + 1, last)

        for idx, taken in enumerate(claimed):
            if not taken:
                survivors[next_tid] = ([None] * f_idx + [cells[idx]], 0, cells[idx])
                next_tid += 1
        active = survivors

    tracks = {tid: trk for tid, (trk, _, _) in active.items()}
    tracks.update(finished)
    return dict(sorted(tracks.items()))
