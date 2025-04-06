# Facial Landmark Detection with MediaPipe

This Python program uses MediaPipe to detect facial landmarks in real-time from a webcam feed or from static images.

## Features

- Real-time facial landmark detection using webcam
- Static image facial landmark detection
- Visualization of facial mesh, contours, and irises
- Support for multiple faces (configurable)

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

## Installation

Install the required dependencies:

```bash
pip install mediapipe opencv-python
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

## Customization

You can modify the following parameters in the code:

- `max_num_faces`: Maximum number of faces to detect (default: 1)
- `min_detection_confidence`: Minimum confidence value for face detection (default: 0.5)
- `min_tracking_confidence`: Minimum confidence value for face tracking (default: 0.5)
