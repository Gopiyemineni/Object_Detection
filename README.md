🎥 Video Analyzer (YOLOv11 + BoT-SORT Tracking)

This project provides a real-time video analysis system that detects and tracks multiple objects in a video using YOLOv11 and BoT-SORT tracker.
It saves the processed video with bounding boxes and generates a detailed JSON report of detections.

✨ Features

✅ Object detection with YOLOv11
✅ Multi-object tracking using BoT-SORT
✅ Saves processed video with bounding boxes
✅ Exports detection report as JSON
✅ Configurable thresholds and resolutions
✅ Optional SocketIO integration for live updates
✅ Local display with progress monitoring

📂 Project Structure
video_analyzer/
│── video_analyzer.py       # Main analyzer script
│── results/                # Processed videos & JSON reports (auto-created)
│── multi_object_detector.pt # YOLOv11 model (example)
│── input_video.mp4          # Example input video
│── README.md                # Documentation

⚙️ Installation

Clone the repository

git clone https://github.com/yourusername/video-analyzer.git
cd video-analyzer


Install dependencies

pip install ultralytics opencv-python torch numpy

🚀 Usage
1. Run Standalone
python video_analyzer.py

