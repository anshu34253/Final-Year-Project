# 🎬 AI Automatic Video Highlight Detection & Short Video Generation

An end-to-end Machine Learning pipeline built with **PyTorch**, **torchvision**, **OpenCV**, and **MoviePy/FFmpeg** to automatically detect exciting key moments in long videos (such as the TVSum dataset) and compile them into a short highlight video.

---

## 📌 Project Architecture

```
                               ┌────────────────────────┐
                               │       INPUT VIDEO      │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │    Frame Extraction    │  (OpenCV)
                               │  (1 frame per second)  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │   Feature Extractor    │  (ResNet-50)
                               │  (2048-dim vectors)    │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │     Sequence Model     │  (PyTorch Bidirectional LSTM)
                               │  (Learns time context) │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  Importance Scoring    │  (Score range: 0.0 - 1.0)
                               │  & Top Clip Selection  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  Highlight Generation  │  (MoviePy & FFmpeg)
                               │  (Stitches top clips)  │
                               └────────────────────────┘
```

---

## 📁 Directory & File Structure Explanation

```
video_highlight_project/
│
├── data/                         --> Workspace for dataset files
│   ├── raw/                      --> Place input video files (.mp4) here (e.g. TVSum videos)
│   ├── processed/                --> Stores extracted ResNet-50 feature matrices (.npy files)
│   └── annotations/              --> Holds ground-truth importance scores from TVSum dataset
│
├── models/                       --> Holds saved PyTorch model weights (e.g., best_model.pth)
│
├── outputs/                      --> Holds all pipeline outputs
│   ├── frames/                   --> Sample extracted video frame images (.jpg)
│   ├── predictions/              --> Output segment importance score CSV files and plots (.png)
│   └── highlights/               --> Final rendered summary highlight videos (.mp4)
│
├── src/                          --> Core Python source modules
│   ├── dataset.py                --> Custom PyTorch Dataset class for loading TVSum feature data & scores
│   ├── extract_frames.py         --> OpenCV utility to sample and save video frames at regular intervals
│   ├── feature_extractor.py      --> Pre-trained ResNet-50 model to extract 2048-d feature vectors per frame
│   ├── lstm_model.py             --> PyTorch Bidirectional LSTM architecture to predict importance scores
│   ├── train.py                  --> Model training loop with MSE loss, Adam optimizer, and check-pointing
│   ├── predict.py                --> Inference engine to compute segment importance scores for new videos
│   ├── highlight_generator.py    --> Cut and stitch top-scoring clips using MoviePy and FFmpeg
│   └── utils.py                  --> Common helper functions (seeds, device detection, directory setup, plotting)
│
├── notebooks/                    --> Place Jupyter Notebooks here for data exploration & visualizations
│
├── requirements.txt              --> List of required Python dependencies
├── README.md                     --> Project documentation and guide
└── main.py                       --> Main entry point to run the entire pipeline or individual steps
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Copy your raw `.mp4` video files into the `data/raw/` folder.

### 3. Run Pipeline Steps via `main.py`

- **Display help and options:**
  ```bash
  python main.py --help
  ```

- **Run full pipeline on a video:**
  ```bash
  python main.py --video data/raw/sample.mp4 --step all
  ```

- **Run individual steps:**
  ```bash
  # Step 1: Extract frames from video
  python main.py --video data/raw/sample.mp4 --step extract-frames

  # Step 2: Extract ResNet-50 features
  python main.py --video data/raw/sample.mp4 --step extract-features

  # Step 3: Predict segment scores using LSTM
  python main.py --video data/raw/sample.mp4 --step predict

  # Step 4: Generate short highlight video (top 20% moments)
  python main.py --video data/raw/sample.mp4 --step highlight --top_k 0.20
  ```

---

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch, torchvision (ResNet-50 pre-trained)
- **Computer Vision**: OpenCV (`cv2`)
- **Video Editing**: MoviePy, FFmpeg
- **Data Processing & Viz**: NumPy, pandas, scikit-learn, matplotlib
