# YOLOv8 + DeepSORT Aircraft Tracking System

## Overview

This repository implements a real-time aircraft detection and tracking system utilizing YOLOv8 for object detection and DeepSORT for multi-object tracking. The system is designed to maintain persistent object identities across video frames while ensuring accurate tracking of multiple aircraft simultaneously.

## System Architecture

The tracking system comprises two primary components:

1. **Detection Module**: YOLOv8-based aircraft detection trained on 46 aircraft classes
2. **Tracking Module**: DeepSORT algorithm with optimized parameters for persistent ID management

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- NVIDIA CUDA Toolkit (optional, for GPU acceleration)

### Dependencies Installation

```bash
pip install -r requirements.txt
```

Required packages:
- ultralytics (YOLOv8)
- opencv-python (Computer Vision operations)
- torch (Deep Learning framework)
- deep-sort-realtime (Tracking algorithm)

## Project Structure

```
Yolov8_DeepSort_Tracker/
├── yolov8_deepsort.py          # Core tracking library module
├── run_tracker.py              # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── aircraft2/
│   └── weights/
│       └── best.pt            # Trained YOLOv8 model weights
└── Results/                    # Output directory
    ├── *_tracked_*.mp4        # Processed video files
    └── *_tracking_*.csv       # Tracking statistics
```

## Key Features

### 1. Persistent Identity Management
The system employs advanced matching algorithms to ensure consistent object identification across frames, preventing identity switches and maintaining tracking continuity.

### 2. Detection-Based Visualization
Track identifiers are displayed exclusively when objects are actively detected, eliminating false positives from predictive tracking.

### 3. Multi-Aircraft Support
Supports detection and classification of 46 distinct aircraft types with real-time class label display.

### 4. Comprehensive Analytics
Generates detailed CSV reports containing track identifiers, aircraft classifications, temporal information, and frame-level tracking statistics.

## Supported Aircraft Classes

The system is trained to recognize the following 46 aircraft types:

**Attack & Fighter Aircraft**: A10, AV8B, F14, F15, F16, F18, F22, F35, F4, F117, J10, J20, EF2000, Rafale, JAS39, Mirage2000, MQ9, Su24, Su34, Su57, Tornado, YF23

**Bombers & Strategic Aircraft**: B1, B2, B52, Tu95, Tu160, SR71, Vulcan, XB70

**Transport & Cargo Aircraft**: A400M, C130, C17, C2, C5, AG600

**Special Purpose Aircraft**: Be200, E2, E7, KC135, P3, RQ4, U2, US2, V22

## Configuration Parameters

### DeepSORT Tracking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_age` | 60 | Maximum frames to maintain track without detection |
| `n_init` | 1 | Frames required for track confirmation |
| `max_iou_distance` | 0.4 | Spatial overlap threshold for track association |
| `max_cosine_distance` | 0.3 | Feature similarity threshold for matching |
| `nn_budget` | 300 | Feature vector memory allocation per class |
| `embedder` | mobilenet | Neural network for feature extraction |

### Detection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `conf_threshold` | 0.25 | Minimum confidence for valid detections |
| `iou` | 0.5 | Non-maximum suppression threshold |
| `time_since_update` | 0 | Display only active detections |

## Usage

### Method 1: Interactive Command-Line Interface

Execute the main runner script:

```bash
python run_tracker.py
```

The system provides three operational modes:

1. **Single Video Processing**: Process individual video files with configurable parameters
2. **Real-Time Webcam Tracking**: Live tracking from camera input
3. **Batch Processing**: Automated processing of multiple video files

### Method 2: Programmatic Integration

```python
from yolov8_deepsort import get_tracker

# Initialize tracker instance
tracker = get_tracker(conf_threshold=0.25)

# Process video file
results = tracker.process_video("path/to/video.mp4")

# Access results
print(f"Frames Processed: {results['frames_processed']}")
print(f"Unique Tracks: {results['total_tracks']}")
print(f"Output Location: {results['output_path']}")
```

### Method 3: Custom Configuration

```python
from yolov8_deepsort import AircraftTracker
from pathlib import Path

# Initialize with custom parameters
tracker = AircraftTracker(
    model_path=Path("aircraft2/weights/best.pt"),
    results_dir=Path("Results"),
    conf_threshold=0.30
)

# Execute tracking
results = tracker.process_video(
    video_path="input_video.mp4",
    output_path="custom_output.mp4",
    display=True
)
```

## Output Format

### Video Output
Processed videos are saved with the following naming convention:
```
{original_filename}_tracked_{timestamp}.mp4
```

### CSV Statistics
Tracking statistics are exported in CSV format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| Track_ID | Integer | Unique track identifier |
| Aircraft_Type | String | Classified aircraft model |
| First_Frame | Integer | Initial detection frame number |
| Last_Frame | Integer | Final detection frame number |
| Total_Frames | Integer | Cumulative frames tracked |

Example:
```csv
Track_ID,Aircraft_Type,First_Frame,Last_Frame,Total_Frames
1,F22,15,450,436
2,Su57,28,445,418
3,B2,45,380,336
```

## System Requirements

### Minimum Requirements
- CPU: Intel Core i5 (8th Gen) or equivalent
- RAM: 8 GB
- Storage: 2 GB available space
- Operating System: Windows 10/11, Linux, macOS

### Recommended Specifications
- CPU: Intel Core i7 (10th Gen) or equivalent
- GPU: NVIDIA GTX 1060 or higher with 6GB VRAM
- RAM: 16 GB
- Storage: 5 GB available space (SSD recommended)

## Performance Optimization

### GPU Acceleration
The system automatically detects and utilizes CUDA-compatible GPUs for accelerated processing. GPU acceleration provides:
- 5-10x faster detection inference
- Real-time processing at 30+ FPS on 1080p video
- Efficient feature extraction for tracking

### Processing Speed
- CPU-only: 5-15 FPS (depending on resolution)
- GPU-accelerated: 25-60 FPS (depending on GPU model and resolution)

## Troubleshooting

### Common Issues

**Issue**: High number of unique track IDs
- **Solution**: Increase `max_iou_distance` to 0.5 or `max_cosine_distance` to 0.4 for more flexible matching

**Issue**: Tracks disappearing prematurely
- **Solution**: Increase `max_age` parameter to 90 or 120 frames

**Issue**: Multiple IDs on single aircraft
- **Solution**: Decrease `iou` threshold to 0.4 and increase `conf_threshold` to 0.30

**Issue**: Slow processing speed
- **Solution**: Ensure GPU acceleration is enabled; reduce video resolution; disable display during processing

## Technical Details

### Detection Pipeline
1. Frame acquisition from video source
2. YOLOv8 inference with NMS post-processing
3. Confidence filtering and class assignment
4. Bounding box extraction in [x, y, w, h] format

### Tracking Pipeline
1. DeepSORT feature extraction using MobileNet embedder
2. Track-to-detection association via Hungarian algorithm
3. Kalman filter-based motion prediction
4. Track state management (confirmed/tentative/deleted)
5. Identity preservation across temporal gaps

## License

This project utilizes open-source components with their respective licenses:
- YOLOv8: AGPL-3.0 License
- DeepSORT: GPL-3.0 License

## Acknowledgments

This implementation builds upon:
- **Ultralytics YOLOv8**: Object detection framework
- **DeepSORT**: Simple Online and Realtime Tracking with a Deep Association Metric
- **OpenCV**: Computer vision library for image processing

## Contact & Support

For technical inquiries, bug reports, or feature requests, please refer to the project repository documentation or contact the development team.

## Version History

- **v1.0.0** (2026-01-09): Initial release with simplified tracking implementation
  - YOLOv8 detection integration
  - DeepSORT tracking with persistent IDs
  - 46 aircraft class support
  - CSV analytics export
