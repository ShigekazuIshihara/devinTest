# Facial Landmark Detection with MediaPipe and PERCLOS

This Python program uses MediaPipe to detect facial landmarks in real-time from a webcam feed or from static images. It also implements PERCLOS (PERcentage of eye CLOSure) to detect eye closure and includes optional data logging functionality.

## Features

- Real-time facial landmark detection using webcam
- Static image facial landmark detection
- Visualization of facial mesh, contours, and irises
- Support for multiple faces (configurable)
- PERCLOS calculation for eye closure detection
- Visual indicators for eye closure status
- **NEW**: Optional CSV logging of PERCLOS data with timestamps

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

Install the required dependencies:

```bash
pip install mediapipe opencv-python numpy
```

## Usage

### For webcam-based real-time detection:

```bash
# Use default webcam (ID 0)
python facial_landmark_detection.py --webcam 0

# Use secondary webcam (ID 1)
python facial_landmark_detection.py --webcam 1

# Use any other webcam by specifying its ID
python facial_landmark_detection.py --webcam 2
```

### For webcam detection with data logging:

```bash
# Use default webcam with logging
python facial_landmark_detection.py --webcam 0 --log

# Use secondary webcam with logging
python facial_landmark_detection.py --webcam 1 --log
```

- Press 'q' to exit the program
- When logging is enabled, data is automatically saved to a timestamped CSV file

### For processing a static image:

```bash
python facial_landmark_detection.py --image path/to/your/image.jpg
```

The processed image will be saved with the prefix "processed_" added to the original filename.

**Note**: Logging functionality is only available in webcam mode.

### Webcam Selection

- **Webcam ID 0**: Usually the default/built-in camera
- **Webcam ID 1**: Usually the first external USB camera
- **Webcam ID 2+**: Additional external cameras

If you're unsure which webcam ID to use, try starting with 0 (most common), then 1, etc. The program will display an error message if the specified webcam cannot be accessed.

## Logging Feature

When the `--log` option is used with webcam mode, the program will:

1. Create a CSV file with timestamp in the filename (format: `perclos_log_YYYYMMDD_HHMMSS.csv`)
2. Log timestamp, avg_left_perclos, and avg_right_perclos values every second
3. Display "LOGGING ENABLED" indicator on the video feed
4. Automatically close the log file when the program is terminated with 'q'

### Log File Format

The CSV file contains the following columns:
- `timestamp`: Date and time with millisecond precision (YYYY-MM-DD HH:MM:SS.mmm)
- `avg_left_perclos`: Average PERCLOS value for the left eye (2 decimal places)
- `avg_right_perclos`: Average PERCLOS value for the right eye (2 decimal places)

Example log entry:
```csv
timestamp,avg_left_perclos,avg_right_perclos
2025-07-15 14:30:15.123,45.67,43.21
2025-07-15 14:30:16.124,46.12,44.03
```

## How It Works

The program uses MediaPipe's Face Mesh solution which can detect up to 468 facial landmarks. It processes each frame (from webcam or static image) to:

1. Detect faces in the image
2. Identify facial landmarks
3. Draw a mesh overlay on the face
4. Highlight facial contours and irises
5. Calculate PERCLOS values for both eyes
6. Display eye closure status when PERCLOS is below threshold
7. **Optional**: Log PERCLOS data to CSV file every second

## PERCLOS Implementation

PERCLOS (PERcentage of eye CLOSure) is a measure used to detect drowsiness by monitoring eye closure. In this implementation:

1. The iris diameter is calculated and used as a reference (100%)
2. The distance between upper and lower eyelids is measured
3. The percentage of eye opening relative to iris diameter is calculated
4. If the percentage falls below 20%, the eye is considered closed
5. "Eye Closed" is displayed on the screen when an eye is detected as closed
6. A visual bar indicator shows the current PERCLOS value for each eye

The implementation uses a temporal smoothing approach by averaging PERCLOS values over multiple frames to reduce noise and provide more stable detection.

## Customization

You can modify the following parameters in the code:

- `max_num_faces`: Maximum number of faces to detect (default: 1)
- `min_detection_confidence`: Minimum confidence value for face detection (default: 0.5)
- `min_tracking_confidence`: Minimum confidence value for face tracking (default: 0.5)
- `threshold`: PERCLOS threshold for eye closure detection (default: 20%)
- `history_size`: Number of frames to average PERCLOS values over (default: 10)
- Logging interval: Currently set to 1 second (can be modified in the code)

## Command Line Options

- `--webcam <ID>`: Use webcam for real-time detection (specify webcam ID: 0, 1, 2, etc.)
- `--image <path>`: Process a single image file
- `--log`: Enable CSV logging (only works with --webcam option)

## Data Analysis

The generated CSV log files can be easily imported into data analysis tools like:
- Excel or Google Sheets for basic analysis
- Python with pandas for advanced analysis
- R for statistical analysis
- MATLAB for signal processing

This makes it easy to perform longitudinal studies, fatigue analysis, or integration with other physiological monitoring systems.