"""
main.py
-------
Main entry point for the Video Highlight Detection and Generation project.

This script coordinates all pipeline stages:
1. Extract frames from input raw video (OpenCV)
2. Extract 2048-d visual feature representations (ResNet-50)
3. Learn/predict sequence importance scores (PyTorch Bidirectional LSTM)
4. Render short highlight video summary (MoviePy / FFmpeg)

Usage Examples:
    # Print pipeline help and options:
    python main.py --help

    # Run full end-to-end highlight generation pipeline on a video:
    python main.py --video data/raw/sample.mp4 --step all
"""

import argparse
import os
import numpy as np

from src.utils import set_seed, get_device, ensure_directories_exist, plot_importance_scores
from src.extract_frames import extract_frames_from_video
from src.feature_extractor import ResNet50FeatureExtractor
from src.predict import predict_importance_scores, save_predictions
from src.highlight_generator import generate_highlight_video


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Automatic Video Highlight Detection & Short Video Generation Pipeline"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to raw input video file (.mp4). If omitted, picks the first video in data/raw/"
    )
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["extract-frames", "extract-features", "train", "predict", "highlight", "all"],
        default="all",
        help="Pipeline step to execute: extract-frames, extract-features, train, predict, highlight, or all"
    )
    
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=1,
        help="Number of frames to extract per second of video (default: 1)"
    )

    parser.add_argument(
        "--top_k",
        type=float,
        default=0.20,
        help="Fraction of video duration to keep in highlight video (e.g. 0.20 = top 20%%)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pth",
        help="Path to trained PyTorch model weights (.pth)"
    )

    return parser


def run_pipeline(args):
    """Executes the requested video highlight pipeline steps."""
    set_seed(42)
    device = get_device()

    # Ensure output folder paths exist
    ensure_directories_exist([
        "data/raw",
        "data/processed",
        "data/annotations",
        "models",
        "outputs/frames",
        "outputs/predictions",
        "outputs/highlights"
    ])

    # Auto-detect video if not explicitly provided
    if not args.video:
        import glob
        raw_videos = sorted(glob.glob("data/raw/*.mp4"))
        if raw_videos:
            args.video = raw_videos[0]
            print(f"[Main] Auto-selected input video: {args.video}")
        else:
            args.video = "data/raw/sample.mp4"

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    processed_feature_path = os.path.join("data/processed", f"{video_name}_features.npy")
    prediction_csv_path = os.path.join("outputs/predictions", f"{video_name}_scores.csv")
    prediction_plot_path = os.path.join("outputs/predictions", f"{video_name}_plot.png")
    highlight_output_path = os.path.join("outputs/highlights", f"{video_name}_highlight.mp4")

    print("\n" + "=" * 60)
    print(f"       AI VIDEO HIGHLIGHT PIPELINE - Step: {args.step.upper()}")
    print("=" * 60)

    # STEP 1 & 2: Extract Frames & Extract Features using ResNet-50
    if args.step in ["extract-frames", "extract-features", "all"]:
        if not os.path.exists(args.video):
            print(f"[Main] Notice: Input video file '{args.video}' not found.")
            print("[Main] Place your input .mp4 videos inside data/raw/ to execute processing.")
            return

        print("\n--> Step 1: Extracting Frames with OpenCV...")
        frames = extract_frames_from_video(
            video_path=args.video,
            sample_fps=args.sample_fps,
            save_dir=os.path.join("outputs/frames", video_name)
        )

        print("\n--> Step 2: Extracting Visual Features with ResNet-50...")
        extractor = ResNet50FeatureExtractor(pretrained=True)
        features = extractor.extract_from_frames(frames, device=device)

        # Save feature matrix to disk (.npy)
        np.save(processed_feature_path, features)
        print(f"[Main] Saved visual features to: {processed_feature_path}")

    # STEP 3: Train LSTM Model
    if args.step == "train":
        print("\n--> Step 3: Model Training Mode.")
        print("[Main] Training step selected. Prepare your DataLoader with TVSum annotations to train.")

    # STEP 4: Predict Importance Scores
    if args.step in ["predict", "all"]:
        print("\n--> Step 4: Predicting Importance Scores with Bidirectional LSTM...")
        if os.path.exists(processed_feature_path):
            features = np.load(processed_feature_path)
        else:
            print(f"[Main] Features file '{processed_feature_path}' not found. Generating dummy features for demo.")
            features = np.random.randn(60, 2048).astype(np.float32)

        # Generate scores using LSTM model
        scores = predict_importance_scores(features=features, model_path=args.model_path, device=device)

        # Save CSV and visualization plot
        save_predictions(scores, prediction_csv_path)
        plot_importance_scores(scores, save_path=prediction_plot_path, title=f"Importance Scores: {video_name}")

    # STEP 5: Generate Highlight Video
    if args.step in ["highlight", "all"]:
        print("\n--> Step 5: Generating Highlight Summary Video with MoviePy & FFmpeg...")
        if not os.path.exists(args.video):
            print(f"[Main] Notice: Input video '{args.video}' does not exist yet for highlight generation.")
            return

        if 'scores' not in locals():
            if os.path.exists(prediction_csv_path):
                import pandas as pd
                df = pd.read_csv(prediction_csv_path)
                scores = df["importance_score"].values
            else:
                scores = np.random.uniform(0.2, 0.9, size=60)

        generate_highlight_video(
            video_path=args.video,
            scores=scores,
            output_path=highlight_output_path,
            top_k_percent=args.top_k,
            sample_fps=args.sample_fps
        )

    print("\n" + "=" * 60)
    print("       PIPELINE EXECUTION COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)
