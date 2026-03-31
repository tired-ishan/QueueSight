"""Version 1 - Basic queue counter.

This version demonstrates the first baseline:
- detect people in the frame
- keep only detections inside the queue polygon
- count them frame by frame
- display the count on video

It is intentionally simpler than the later versions.
"""

from __future__ import annotations

import argparse

from processor import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="QueueSight v1 - basic queue counter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/v1_basic_output.mp4")
    args = parser.parse_args()

    process_video(args.input, args.output, csv_path=None, show_progress=True)


if __name__ == "__main__":
    main()
