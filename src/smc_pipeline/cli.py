"""Command-line interface: ``smc-pipeline segment`` and ``smc-pipeline track``."""
import argparse

from .export import write_tracks_csv
from .tracking import TrackerConfig, load_frames, track


def _segment(args) -> None:
    from .segment import segment_folder

    n = segment_folder(
        args.input, args.output, args.weights,
        threshold=args.threshold, min_area=args.min_area,
        batch_size=args.batch_size, device=args.device, overwrite=args.overwrite,
    )
    print(f"Segmented {n} images -> {args.output}")


def _track(args) -> None:
    cfg = TrackerConfig(radius_px=args.radius, gap_tolerance=args.gap)
    frames = load_frames(args.json_dir, args.prefix)
    tracks = track(frames, cfg)
    rows = write_tracks_csv(tracks, list(frames), args.output)
    print(f"{len(tracks)} tracks over {len(frames)} frames ({rows} rows) -> {args.output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="smc-pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="segment bright-field TIFFs into per-image JSON")
    seg.add_argument("--input", required=True, help="folder of .tif images (searched recursively)")
    seg.add_argument("--output", required=True, help="folder for JSON annotations")
    seg.add_argument("--weights", required=True, help="path to model checkpoint (.pth)")
    seg.add_argument("--threshold", type=float, default=0.5, help="probability threshold")
    seg.add_argument("--min-area", type=int, default=80, help="remove objects smaller than this (px)")
    seg.add_argument("--batch-size", type=int, default=8)
    seg.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: auto)")
    seg.add_argument("--overwrite", action="store_true", help="redo images that already have JSON")
    seg.set_defaults(func=_segment)

    trk = sub.add_parser("track", help="link nuclei across the frames of one well/field")
    trk.add_argument("--json-dir", required=True, help="folder with <prefix>_<frame>.json files")
    trk.add_argument("--prefix", required=True, help="file prefix shared by one well/field")
    trk.add_argument("--output", required=True, help="output CSV path")
    trk.add_argument("--radius", type=float, default=16.0, help="linking radius in pixels")
    trk.add_argument("--gap", type=int, default=2, help="max consecutive missed frames")
    trk.set_defaults(func=_track)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
