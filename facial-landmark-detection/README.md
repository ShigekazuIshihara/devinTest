# Facial Landmark Detection with MediaPipe and PERCLOS

This Python program uses MediaPipe to detect facial landmarks in real-time from a webcam feed or from static images. It also implements PERCLOS (PERcentage of eye CLOSure) to detect eye closure.

## Features

- Real-time facial landmark detection using webcam
- Static image facial landmark detection
- Visualization of facial mesh, contours, and irises
- Support for multiple faces (configurable)
- PERCLOS calculation for eye closure detection
- Visual indicators for eye closure status

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
python facial_landmark_detection.py --webcam
```

- Press 'q' to exit the program

### For processing a static image:

```bash
python facial_landmark_detection.py --image path/to/your/image.jpg
```

The processed image will be saved with the prefix "processed_" added to the original filename.

## How It Works

The program uses MediaPipe's Face Mesh solution which can detect up to 468 facial landmarks. It processes each frame (from webcam or static image) to:

1. Detect faces in the image
2. Identify facial landmarks
3. Draw a mesh overlay on the face
4. Highlight facial contours and irises
5. Calculate PERCLOS values for both eyes
6. Display eye closure status when PERCLOS is below threshold

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
