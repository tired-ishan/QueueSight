"""Version 2 - Tracking-based queue counter.

This version adds:
- centroid tracking
- reduced duplicate counting
- smoothing over a short window
- clearer status labels
"""

from __future__ import annotations

import argparse

from processor import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="QueueSight v2 - tracked queue counter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/v2_tracked_output.mp4")
    parser.add_argument("--csv", default="outputs/v2_tracked_log.csv")
    args = parser.parse_args()

    process_video(args.input, args.output, csv_path=args.csv, show_progress=True)


if __name__ == "__main__":
    main()
