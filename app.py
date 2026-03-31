from __future__ import annotations

import argparse
from pathlib import Path

from processor import plot_history, process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QueueSight - queue estimation from video")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="outputs/annotated_output.mp4", help="Path to annotated output video")
    parser.add_argument("--csv", default="outputs/queue_log.csv", help="Path to CSV log output")
    parser.add_argument("--chart", default="outputs/queue_chart.png", help="Path to chart PNG output")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV logging")
    parser.add_argument("--no-chart", action="store_true", help="Disable chart generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = None if args.no_csv else args.csv

    summary = process_video(
        input_path=args.input,
        output_path=args.output,
        csv_path=csv_path,
        config_path=args.config,
        show_progress=True,
    )

    print("\nProcessing complete")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if csv_path and not args.no_chart:
        chart = plot_history(csv_path, args.chart)
        print(f"chart_png: {chart}")


if __name__ == "__main__":
    main()
