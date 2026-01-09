# YOLOv8 + DeepSORT Aircraft Tracking System

Real-time aircraft detection and tracking using YOLOv8 for detection and DeepSORT for persistent ID management across 46 aircraft classes.

## Demo

[▶️ View Demo Video](Results/1_tracked_20260109_153132.mp4)

## Quick Start

```bash
pip install -r requirements.txt
python run_tracker.py
```

## Features

- **Persistent ID Tracking**: Same aircraft maintains same ID throughout video
- **46 Aircraft Classes**: F22, Su57, B2, C17, F15, F16, F18, and more
- **Real-time Processing**: GPU-accelerated detection and tracking
- **Detection-Only Display**: IDs shown only when actively detected
- **CSV Analytics**: Export tracking statistics with timestamps

## Supported Aircraft

A10, A400M, AG600, AV8B, B1, B2, B52, Be200, C130, C17, C2, C5, E2, E7, EF2000, F117, F14, F15, F16, F18, F22, F35, F4, J10, J20, JAS39, KC135, MQ9, Mig31, Mirage2000, P3, RQ4, Rafale, SR71, Su24, Su34, Su57, Tornado, Tu160, Tu95, U2, US2, V22, Vulcan, XB70, YF23

## Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_age` | 60 | Frames to keep track alive |
| `conf_threshold` | 0.25 | Detection confidence |
| `max_iou_distance` | 0.4 | Spatial matching |
| `max_cosine_distance` | 0.3 | Appearance matching |

## Usage

### Interactive Mode
```bash
python run_tracker.py
```

### Programmatic
```python
from yolov8_deepsort import get_tracker

tracker = get_tracker(conf_threshold=0.25)
results = tracker.process_video("video.mp4")
```

## Output

**Video**: `Results/{name}_tracked_{timestamp}.mp4`  
**CSV**: `Results/{name}_tracking_{timestamp}.csv`

CSV Format:
```csv
Track_ID,Aircraft_Type,First_Frame,Last_Frame,Total_Frames
1,F22,15,450,436
2,Su57,28,445,418
```

## Requirements

- Python 3.8+
- CUDA GPU (recommended)
- 8GB RAM minimum

## Installation

```bash
pip install ultralytics opencv-python torch deep-sort-realtime
```

## Project Structure

```
├── yolov8_deepsort.py     # Core library
├── run_tracker.py         # Main script
├── aircraft2/weights/     # Model weights
└── Results/               # Output folder
```

## Contact

**Email**: jaikishannishad33@gmail.com  
**GitHub**: [Jaikishan1228](https://github.com/Jaikishan1228)

## License

Open-source project using YOLOv8 (AGPL-3.0) and DeepSORT (GPL-3.0)
