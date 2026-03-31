# QueueSight - Smart Queue Length Estimation Using Computer Vision

QueueSight is a Computer Vision BYOP project that estimates queue crowding from video input. It detects people in a marked queue region, counts how many are present, and classifies the queue as Low, Medium, or High.

## Why this project

Queues are a daily problem in canteens, helpdesks, libraries, and service counters. Manual observation is slow and often inaccurate. This project shows how computer vision can turn a raw camera feed into useful operational information.

## Versions included in this repository

### Version 1 - Basic counter
- Detects people in a queue region
- Counts them frame by frame
- Draws the queue area and count on the video

## Version 2 - Improved Stability
- Added object tracking
- Reduced duplicate counting
- Added smoothing for stable output

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
├── utils.py
├── tracker.py
├── config.yaml
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── versions/
│   ├── v1_basic_counter.py
│   ├── v2_tracking_counter.py
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

## License

For academic use.
