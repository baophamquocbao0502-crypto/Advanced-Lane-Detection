# Advanced Lane Detection

A full modular lane detection pipeline built with Python and OpenCV, inspired by Udacity's Self-Driving Car Nanodegree.

![Demo](output/images/pipeline_result.png)

---

## Pipeline Overview

```
[Video Frame]
     │
     ▼
Camera Calibration (undistortion)
     │
     ▼
Thresholding (Sobel + HLS + Morphology + ROI)
     │
     ▼
Perspective Transform (Birds-Eye View)
     │
     ▼
Lane Detection (Sliding Window + Polynomial Fit)
     │
     ▼
Curvature + Vehicle Offset
     │
     ▼
[Video Result]
```

---

## Features

- **Camera Calibration** — removes lens distortion using chessboard images
- **Thresholding** — combines Sobel gradient + HLS color threshold to detect lane pixels
- **Perspective Transform** — converts to Birds-Eye View for accurate polynomial fitting
- **Sliding Window Search** — finds lane pixels from scratch each frame
- **Search Around Poly** — faster search around previous polynomial (after first detection)
- **Polynomial Fit** — fits degree-2 polynomial: `x = Ay² + By + C`
- **EMA Smoothing** — exponential moving average across frames to reduce jitter
- **Curvature** — calculates radius of curvature in meters
- **Vehicle Offset** — calculates how far vehicle is from lane center in meters

---

## Project Structure

```
AdvancedLaneDetection/
│
├── src/
│   ├── calibration.py      # Camera calibration + undistortion
│   ├── thresholding.py     # Binary image creation
│   ├── perspective.py      # Perspective transform
│   ├── lane_detection.py   # Sliding window + polynomial fit
│   ├── curvature.py        # Curvature + offset calculation
│   └── pipeline.py         # Full video pipeline
│
├── data/
│   ├── camera_cal/         # Chessboard calibration images
│   ├── test_images/        # Test images
│   └── test_videos/        # Input videos
│
├── output/
│   ├── images/             # Debug output images
│   └── videos/             # Processed videos
│
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AdvancedLaneDetection.git
cd AdvancedLaneDetection

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Camera Calibration (run once)
```bash
python src/calibration.py
```
Place chessboard images in `data/camera_cal/` first.  
Download sample images from: [Udacity Camera Calibration](https://github.com/udacity/CarND-Camera-Calibration)

### Step 2 — Run the full pipeline
```bash
python main.py
```

Or run individual modules:
```bash
python src/thresholding.py    # Test thresholding
python src/perspective.py     # Test perspective transform
python src/lane_detection.py  # Test lane detection
python src/curvature.py       # Test curvature calculation
python src/pipeline.py        # Run full pipeline
```

---

## Results

| Metric | Value |
|---|---|
| Curvature detection | ✅ Real-time in meters |
| Vehicle offset | ✅ Real-time in meters |
| Smoothing | ✅ EMA + N-frame average |
| Detection method | Sliding Window → Search Around Poly |

---

## Key Technical Details

### Why fit x as a function of y?
Lane lines are nearly vertical — multiple y values share the same x. Fitting y(x) fails for vertical lines. Fitting x(y) works perfectly.

### Why Birds-Eye View?
Parallel lanes appear to converge in the camera view (perspective distortion). After perspective transform, they are truly parallel — enabling accurate polynomial fitting and curvature calculation.

### Why EMA smoothing?
Each frame processed independently causes jitter. EMA gives higher weight to recent frames while retaining history:
```
EMA_new = (1 - α) × EMA_old + α × fit_new
α = 0.25
```

---

## Comparison: Advanced Pipeline vs Lightweight Approach

| | Advanced Pipeline | Lightweight (fitLine) |
|---|---|---|
| Curve handling | Polynomial (degree 2) | Linear only |
| Output | Curvature + offset | Lines only |
| Thresholding | Sobel + HLS white + yellow | HLS white only |
| Complexity | 6 modules | 1 file |
| Best for | Standard camera | Fisheye/IPG camera |

---

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- MoviePy

---

## References

- [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)
- [CarND-Advanced-Lane-Lines](https://github.com/papiot/CarND-Advanced-Lane-Lines)
- Sjafrie, H. — *Introduction to Self-Driving Vehicle Technology*, CRC Press

---

## Author

Learning Autonomous Driving Engineering — based in Ingolstadt, Germany 🇩🇪  
Currently studying: Sensor Fusion · SLAM · Vehicle Control (MPC, Stanley)
