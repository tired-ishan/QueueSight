# QueueSight - Smart Queue Length Estimation Using Computer Vision

QueueSight is a Computer Vision BYOP project that estimates queue crowding from video input. It detects people in a marked queue region, counts how many are present, and classifies the queue as Low, Medium, or High.

## Why this project

Queues are a daily problem in canteens, helpdesks, libraries, and service counters. Manual observation is slow and often inaccurate. This project shows how computer vision can turn a raw camera feed into useful operational information.

## Versions included in this repository

### Version 1 - Basic counter
- Detects people in a queue region
- Counts them frame by frame
- Draws the queue area and count on the video

### Version 2 - Tracking and smoothing
- Adds centroid-based tracking
- Reduces double counting
- Smooths noisy counts across frames

### Version 3 - Final version
- Adds CSV logging
- Generates a count-over-time chart
- Saves an annotated output video
- Includes cleaner configuration and documentation

## Tech stack

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML
- ReportLab

## Repository structure

```text
queuesight_project/
├── app.py
├── detector.py
├── tracker.py
├── processor.py
├── utils.py
├── config.yaml
├── requirements.txt
├── README.md
├── report.pdf
├── report.md
├── CHANGELOG.md
├── versions/
│   ├── v1_basic_counter.py
│   ├── v2_tracking_counter.py
│   └── v3_final_app.py
└── outputs/
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/queuesight_project.git
cd queuesight_project
```

### 2. Create and activate a virtual environment
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to run

### Final version
```bash
python app.py --input path/to/video.mp4 --output outputs/annotated.mp4
```

### Choose a custom CSV output path
```bash
python app.py --input path/to/video.mp4 --output outputs/annotated.mp4 --csv outputs/queue_log.csv
```

### Show processing progress
The script prints frame-level progress to the terminal.

## What the output contains

- Annotated video with queue polygon
- Bounding boxes and object IDs
- Queue count for each frame
- Crowd level label: Low / Medium / High
- CSV log of the count over time
- PNG chart generated from the CSV log

## Configuration

You can edit `config.yaml` to adjust:
- queue region
- crowd thresholds
- detector parameters
- tracking parameters
- display options

## Notes on accuracy

This project works best when:
- the camera is fixed
- the queue area is clearly visible
- lighting is reasonably stable
- people are not heavily occluded

## Limitations

- HOG-based person detection is lighter than deep learning but less accurate
- Heavy crowding and occlusion can reduce detection quality
- Results depend on camera angle and scene clarity

## Project report

The detailed report is included as `report.pdf`.

## Version tracking

The repository includes three versioned scripts under `versions/` so the development path is visible:
- `v1_basic_counter.py`
- `v2_tracking_counter.py`
- `v3_final_app.py`

## License

For academic use.
