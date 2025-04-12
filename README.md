# DDTA
Droplet Detection and Tracking in Complex Motions: A Deep Learning-based Approach

- **Demo Preview**
  ![Demo Results](reults/reults.png)

## Features

- **Image Analysis**
  - Multiple format support (JPG/PNG/TIFF)
  - Pixel-to-real-world calibration
  - Size distribution histograms
  - CSV results export

- **Video Processing**
  - Real-time detection mode
  - Object tracking with trajectory visualization
  - Velocity vector analysis
  - Frame-by-frame navigation

- **Customization**
  - Adjustable detection parameters (confidence/IOU)
  - GPU/CPU support
  - Multiple model weight support
  - Custom display ranges

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) + (CUDA 11.3+)
- PyTorch 1.12.0+
- OpenCV 4.5.4+

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/droplet-analysis-system.git
cd droplet-analysis-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
mkdir weights
wget -P weights https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
