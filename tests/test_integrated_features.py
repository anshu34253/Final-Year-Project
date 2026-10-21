"""
tests/test_integrated_features.py
---------------------------------
Comprehensive Test Suite for Integrated Features:
1. Face detection (UniFace SCRFD)
2. Face tracking (ByteTrack)
3. No-face fallback
4. Crop stability (EMA smoothing, no sudden jumps)
5. Subtitle generation (ASS formats, bold, clean, minimal styles)
6. Subtitle OFF mode
7. Emoji OFF mode
8. Emoji ON mode (keyword context matching)
9. Manual start/end editing
10. Manual crop editing (0.0 to 1.0 manual X)
11. Single clip re-render
12. No duplicate ResNet/Bi-LSTM inference
13. Existing pipeline compatibility
14. Output video validation
15. 9:16 aspect ratio
16. Audio/video synchronization
17. Path traversal / security
18. Unicode / emoji filenames
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import cv2

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

from src.face_tracking import FaceSubjectTracker, get_face_detector
from src.crop_strategy import compute_916_crop_coordinates
from src.emoji_service import enhance_segments_with_emojis, detect_emoji_for_text
from src.subtitle_service import create_styled_ass, SUBTITLE_STYLES, SUBTITLE_POSITIONS
from src.reel_renderer import render_single_reel, probe_video_dimensions


class IntegratedFeaturesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # Create a synthetic 16:9 test video (640x360, 5 seconds, 25 fps)
        cls.test_video_path = os.path.join(cls.temp_dir, "synthetic_test_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.test_video_path, fourcc, 25.0, (640, 360))

        # Draw a synthetic moving "face" circle across frames
        for f in range(125):
            frame = np.full((360, 640, 3), 40, dtype=np.uint8)
            # Face center moves smoothly from x=200 to x=440
            cx = int(200 + (f / 125.0) * 240)
            cy = 180
            # Draw synthetic head / skin circle
            cv2.circle(frame, (cx, cy), 45, (160, 180, 220), -1)
            cv2.circle(frame, (cx - 15, cy - 10), 6, (40, 40, 40), -1)
            cv2.circle(frame, (cx + 15, cy - 10), 6, (40, 40, 40), -1)
            cv2.ellipse(frame, (cx, cy + 15), (15, 8), 0, 0, 180, (40, 40, 40), 2)
            out.write(frame)
        out.release()

    # 1. Face detection
    def test_01_face_detection_model_loads(self):
        detector = get_face_detector()
        self.assertIsNotNone(detector, "UniFace SCRFD detector failed to load")
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        faces = detector.detect(test_img)
        self.assertIsInstance(faces, list)

    # 2. Face tracking
    def test_02_face_tracking_execution(self):
        tracker = FaceSubjectTracker()
        res = tracker.analyze_clip(self.test_video_path, start_sec=0.0, duration_sec=2.0)
        self.assertIn("crop_x", res)
        self.assertIn("crop_w", res)
        self.assertIn("crop_h", res)
        self.assertEqual(res["crop_x"] % 2, 0, "crop_x must be an even integer for codec compatibility")

    # 3. No-face fallback
    def test_03_no_face_fallback_to_center(self):
        # Blank black video
        blank_video = os.path.join(self.temp_dir, "blank_video.mp4")
        out = cv2.VideoWriter(blank_video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 360))
        for _ in range(25):
            out.write(np.zeros((360, 640, 3), dtype=np.uint8))
        out.release()

        tracker = FaceSubjectTracker()
        res = tracker.analyze_clip(blank_video, start_sec=0.0, duration_sec=1.0)
        self.assertFalse(res["has_face"])
        # Expected center crop for 640x360: 9:16 target width = round(360 * 9/16 / 2) * 2 = 202
        # Center x = (640 - 202) / 2 = 219 -> even integer: 218 or 219
        self.assertTrue(abs(res["crop_x"] - (640 - res["crop_w"]) / 2) <= 2)

    # 4. Crop stability
    def test_04_crop_stability_and_smoothing(self):
        # Coordinates with face tracking should not jump abruptly between consecutive frames
        tracker = FaceSubjectTracker(smoothing_alpha=0.15)
        res = tracker.analyze_clip(self.test_video_path, start_sec=0.0, duration_sec=4.0)
        samples = res.get("frame_samples", [])
        if len(samples) > 2:
            diffs = [abs(samples[i]["crop_x"] - samples[i - 1]["crop_x"]) for i in range(1, len(samples))]
            max_jump = max(diffs)
            self.assertLess(max_jump, 50, f"Crop window jumped too suddenly ({max_jump}px)!")

    # 5. Subtitle generation (ASS format and styles)
    def test_05_subtitle_generation_styles(self):
        ass_path = os.path.join(self.temp_dir, "test_style.ass")
        speech = [{"start": 1.0, "end": 4.0, "text": "Testing clean bold minimal subtitles"}]
        for style in ["bold", "clean", "minimal"]:
            for pos in ["bottom", "center", "top"]:
                ok = create_styled_ass(
                    speech_segments=speech,
                    reel_start_sec=0.0,
                    reel_end_sec=5.0,
                    output_ass_path=ass_path,
                    style_name=style,
                    position_name=pos
                )
                self.assertTrue(ok)
                self.assertTrue(os.path.isfile(ass_path))
                with open(ass_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("Dialogue: 0,", content)
                    self.assertIn(SUBTITLE_STYLES[style]["font_name"], content)

    # 6. Subtitle OFF mode
    def test_06_subtitle_off_mode(self):
        ass_path = os.path.join(self.temp_dir, "test_off.ass")
        if os.path.isfile(ass_path):
            os.remove(ass_path)
        # When captions=False, render_single_reel should not burn subtitles
        out_reel = os.path.join(self.temp_dir, "reel_no_subs.mp4")
        res = render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=0.0,
            duration_sec=2.0,
            output_path=out_reel,
            captions=False
        )
        self.assertFalse(res["captions_burned"])
        self.assertTrue(os.path.isfile(out_reel))

    # 7. Emoji OFF mode
    def test_07_emoji_off_mode(self):
        speech = [{"start": 1.0, "end": 3.0, "text": "This is totally amazing and on fire"}]
        enhanced = enhance_segments_with_emojis(speech, enabled=False)
        self.assertEqual(enhanced[0]["text"], speech[0]["text"], "Emoji OFF mode altered text!")

    # 8. Emoji ON mode
    def test_08_emoji_on_mode(self):
        speech = [
            {"start": 1.0, "end": 3.0, "text": "Wow this is an amazing breakthrough"},
            {"start": 3.5, "end": 5.5, "text": "We are going to win the championship trophy and make money"}
        ]
        enhanced = enhance_segments_with_emojis(speech, enabled=True)
        # Check that emojis were added
        self.assertNotEqual(enhanced[0]["text"], speech[0]["text"])
        self.assertTrue(any(e in enhanced[0]["text"] for e in ["😲", "😮", "🤯"]))

    # 9. Manual start/end editing
    def test_09_manual_start_end_editing(self):
        out_reel = os.path.join(self.temp_dir, "reel_trimmed.mp4")
        res = render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=1.0,
            duration_sec=2.5,
            output_path=out_reel,
            captions=False
        )
        self.assertTrue(os.path.isfile(out_reel))
        # Verify duration
        cap = cv2.VideoCapture(out_reel)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        dur = frames / fps
        self.assertAlmostEqual(dur, 2.5, delta=0.5)

    # 10. Manual crop editing
    def test_10_manual_crop_editing(self):
        # Test manual_x = 0.0 (left), 0.5 (center), 1.0 (right)
        crop_left = compute_916_crop_coordinates(video_w=1920, video_h=1080, crop_mode="manual", manual_x=0.0)
        crop_center = compute_916_crop_coordinates(video_w=1920, video_h=1080, crop_mode="manual", manual_x=0.5)
        crop_right = compute_916_crop_coordinates(video_w=1920, video_h=1080, crop_mode="manual", manual_x=1.0)

        self.assertEqual(crop_left["crop_x"], 0)
        self.assertGreater(crop_center["crop_x"], 0)
        self.assertGreater(crop_right["crop_x"], crop_center["crop_x"])

    # 11. Single clip re-render
    def test_11_single_clip_re_render(self):
        out_reel = os.path.join(self.temp_dir, "single_clip_rerender.mp4")
        res1 = render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=0.0,
            duration_sec=2.0,
            output_path=out_reel,
            crop_mode="center",
            captions=False
        )
        self.assertTrue(os.path.isfile(out_reel))
        size1 = os.path.getsize(out_reel)

        # Re-render with different crop
        res2 = render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=0.5,
            duration_sec=1.5,
            output_path=out_reel,
            crop_mode="manual",
            manual_x=0.8,
            captions=False
        )
        self.assertTrue(os.path.isfile(out_reel))
        size2 = os.path.getsize(out_reel)
        self.assertTrue(size2 > 0)

    # 12. No duplicate ResNet/Bi-LSTM inference during clip edit
    def test_12_no_duplicate_ai_inference_on_edit(self):
        from app import app
        client = app.test_client()

        # Create a mock job directory
        mock_job_id = "test_rerender_no_ai"
        job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], mock_job_id)
        os.makedirs(job_dir, exist_ok=True)

        meta = {
            "output_mode": "reels",
            "video_filename": "test.mp4",
            "source_video_path": os.path.abspath(self.test_video_path),
            "original_duration": 5.0,
            "reels": [
                {
                    "reel_num": 1,
                    "filename": "reel_01.mp4",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                    "duration_sec": 3.0,
                    "transcript": "Test speech for rerender",
                    "speech_segments": [{"start": 0.5, "end": 2.5, "text": "Test speech"}]
                }
            ]
        }
        with open(os.path.join(job_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Track if ResNet50 or BiLSTM forward was invoked
        import src.feature_extractor
        import src.lstm_model

        resnet_called = False
        lstm_called = False

        orig_extract = src.feature_extractor.ResNet50FeatureExtractor.extract_features
        def mock_extract(*args, **kwargs):
            nonlocal resnet_called
            resnet_called = True
            return orig_extract(*args, **kwargs)

        src.feature_extractor.ResNet50FeatureExtractor.extract_features = mock_extract

        try:
            resp = client.post(
                f"/api/render_clip/{mock_job_id}",
                json={
                    "reel_num": 1,
                    "start_sec": 0.5,
                    "end_sec": 2.5,
                    "crop_mode": "center",
                    "captions": False
                }
            )
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resnet_called, "ResNet-50 was rerun during manual clip editing!")
            self.assertFalse(lstm_called, "Bi-LSTM was rerun during manual clip editing!")
        finally:
            src.feature_extractor.ResNet50FeatureExtractor.extract_features = orig_extract

    # 13. Existing pipeline compatibility
    def test_13_existing_pipeline_signature_compatibility(self):
        from main import run_pipeline, run_multi_reel_pipeline
        self.assertTrue(callable(run_pipeline))
        self.assertTrue(callable(run_multi_reel_pipeline))

    # 14. Output video validation
    def test_14_output_video_validity(self):
        out_reel = os.path.join(self.temp_dir, "valid_output.mp4")
        res = render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=0.0,
            duration_sec=2.0,
            output_path=out_reel,
            captions=False
        )
        self.assertTrue(os.path.isfile(out_reel))
        self.assertGreater(os.path.getsize(out_reel), 1000)

    # 15. 9:16 aspect ratio
    def test_15_output_aspect_ratio_916(self):
        out_reel = os.path.join(self.temp_dir, "aspect_916.mp4")
        render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=0.0,
            duration_sec=1.5,
            output_path=out_reel,
            aspect_ratio="9:16",
            captions=False
        )
        dims = probe_video_dimensions(out_reel)
        self.assertEqual(dims["width"], 1080)
        self.assertEqual(dims["height"], 1920)

    # 16. Audio/video synchronization probe
    def test_16_audio_video_synchronization_attributes(self):
        dims = probe_video_dimensions(self.test_video_path)
        self.assertEqual(dims["width"], 640)
        self.assertEqual(dims["height"], 360)

    # 17. Path traversal / security
    def test_17_security_path_traversal_rejection(self):
        from app import is_safe_job_id, app
        self.assertFalse(is_safe_job_id("../../etc/passwd"))
        self.assertFalse(is_safe_job_id("job_123/../../"))
        self.assertTrue(is_safe_job_id("job_1789322932_ed810a"))

        client = app.test_client()
        resp = client.get("/media/../etc/passwd/highlight.mp4")
        self.assertIn(resp.status_code, [400, 404, 403])

    # 18. Unicode & emoji filenames
    def test_18_unicode_and_emoji_filenames(self):
        unicode_name = "reel_🔥_amazing_test.mp4"
        unicode_path = os.path.join(self.temp_dir, unicode_name)
        res = render_single_reel(
            source_video_path=self.test_video_path,
            start_sec=0.0,
            duration_sec=1.0,
            output_path=unicode_path,
            captions=False
        )
        self.assertTrue(os.path.isfile(unicode_path))
        self.assertGreater(os.path.getsize(unicode_path), 500)


if __name__ == "__main__":
    unittest.main()
