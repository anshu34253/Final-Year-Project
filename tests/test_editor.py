"""
tests/test_editor.py
--------------------
Comprehensive Test Suite for Phase 8: Browser-Based Video Editor.

Tests:
1. Editor page access (valid job and clip ID)
2. Invalid job and clip IDs (rejection & security)
3. Project state retrieval and initialization
4. Trim timestamp validation and boundary clamping
5. Crop settings and aspect-ratio presets (9:16, 16:9, 1:1, 4:5)
6. Subtitle editing, styles, and persistence
7. Emoji and text overlay editing
8. Audio volume, mute, transitions, and audio upload validation
9. Playback speed settings
10. Visual adjustments (brightness, contrast, saturation, filters)
11. Project saving, restoration, and JSON persistence
12. Project reset to original reel state
13. Path traversal protection & job isolation
14. Duplicate concurrent export prevention
15. Re-editing without AI model inference (no ResNet/Bi-LSTM rerun)
16. Original generated clip preservation
17. Real FFmpeg export rendering and output video validation
18. Failed render recovery & error reporting
"""

import os
import sys
import io
import json
import time
import shutil
import tempfile
import unittest
import numpy as np
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

from app import app, is_safe_job_id
from src.editor_service import (
    get_or_create_project, save_project, reset_project,
    validate_project_payload, save_uploaded_audio, is_safe_id
)
from src.editor_renderer import (
    start_editor_render, get_render_status, execute_editor_render,
    generate_editor_ass, build_render_filtergraph, probe_video_dimensions,
    RENDER_TASKS, RENDER_LOCK
)


class VideoEditorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.app.test_client()

        cls.test_job_id = "test_editor_job_phase8"
        cls.job_dir = os.path.join(BASE_DIR, "web_outputs", cls.test_job_id)
        os.makedirs(cls.job_dir, exist_ok=True)

        # Generate a small synthetic 16:9 source video (640x360, 4 seconds, 25 fps)
        cls.source_video_path = os.path.join(cls.job_dir, "test_source.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.source_video_path, fourcc, 25.0, (640, 360))
        for i in range(100):
            frame = np.full((360, 640, 3), 50, dtype=np.uint8)
            # Add a moving colored circle
            cv2.circle(frame, (100 + i * 4, 180), 40, (0, 200, 255), -1)
            out.write(frame)
        out.release()

        # Copy as reel_01.mp4 to represent original generated clip
        cls.original_reel_path = os.path.join(cls.job_dir, "reel_01.mp4")
        shutil.copyfile(cls.source_video_path, cls.original_reel_path)
        cls.original_reel_bytes = os.path.getsize(cls.original_reel_path)

        # Create segments.json
        cls.segments_meta = {
            "job_id": cls.test_job_id,
            "output_mode": "reels",
            "video_filename": "test_source.mp4",
            "source_video_path": cls.source_video_path,
            "original_duration": 4.0,
            "num_reels": 1,
            "reels": [
                {
                    "reel_num": 1,
                    "filename": "reel_01.mp4",
                    "output_path": cls.original_reel_path,
                    "start_sec": 0.5,
                    "end_sec": 3.5,
                    "duration_sec": 3.0,
                    "start_formatted": "00:00",
                    "end_formatted": "00:03",
                    "score": 92.5,
                    "transcript": "Test speech dialogue for reel editor",
                    "speech_segments": [
                        {"start": 0.5, "end": 2.0, "text": "Test speech dialogue"},
                        {"start": 2.0, "end": 3.5, "text": "for reel editor"}
                    ],
                    "crop_mode": "smart",
                    "manual_x": 0.5,
                    "captions": True,
                    "emojis": False
                }
            ]
        }

        with open(os.path.join(cls.job_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(cls.segments_meta, f, indent=2)

    @classmethod
    def tearDownClass(cls):
        # Clean up test output directory
        if os.path.isdir(cls.job_dir):
            shutil.rmtree(cls.job_dir, ignore_errors=True)

    # 1. Editor page access (valid job and clip ID)
    def test_01_editor_page_access_valid(self):
        resp = self.client.get(f"/editor/{self.test_job_id}/1")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("stageVideo", html)
        self.assertIn("timelineScroll", html)
        self.assertIn("renderFinalBtn", html)
        self.assertIn("Trim & Split", html)
        self.assertIn("Aspect Ratio Presets", html)

    # 2. Invalid job and clip IDs (rejection & security)
    def test_02_invalid_job_and_clip_ids(self):
        # Path traversal in job ID
        resp = self.client.get("/editor/..%2F..%2Fetc/1")
        self.assertIn(resp.status_code, [400, 404])

        # Clip ID < 1
        resp = self.client.get(f"/editor/{self.test_job_id}/0")
        self.assertEqual(resp.status_code, 400)

        # Non-existent job
        resp = self.client.get("/editor/non_existent_job_xyz/1", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)

    # 3. Project state retrieval and initialization
    def test_03_project_get_and_initialization(self):
        resp = self.client.get(f"/api/editor/{self.test_job_id}/1/project")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        proj = data["project"]
        self.assertEqual(proj["clip_id"], 1)
        self.assertIn("trim", proj)
        self.assertIn("crop", proj)
        self.assertIn("subtitles", proj)
        self.assertIn("audio", proj)
        self.assertIn("speed", proj)
        self.assertIn("adjustments", proj)

    # 4. Trim timestamp validation and boundary clamping
    def test_04_trim_validation(self):
        # Normal trim
        payload = {"trim": {"start": 1.0, "end": 2.5}}
        res = validate_project_payload(payload, max_duration=4.0)
        self.assertEqual(res["trim"]["start"], 1.0)
        self.assertEqual(res["trim"]["end"], 2.5)
        self.assertAlmostEqual(res["trim"]["duration"], 1.5)

        # Inverted trim (start >= end)
        payload_inv = {"trim": {"start": 3.0, "end": 1.0}}
        res_inv = validate_project_payload(payload_inv, max_duration=4.0)
        self.assertTrue(res_inv["trim"]["start"] < res_inv["trim"]["end"])

        # Beyond source duration
        payload_over = {"trim": {"start": 0.0, "end": 100.0}}
        res_over = validate_project_payload(payload_over, max_duration=4.0)
        self.assertLessEqual(res_over["trim"]["end"], 4.0)

    # 5. Crop settings and aspect-ratio presets (9:16, 16:9, 1:1, 4:5)
    def test_05_crop_settings_and_aspect_ratios(self):
        for aspect in ["9:16", "16:9", "1:1", "4:5"]:
            payload = {"crop": {"aspect_ratio": aspect, "mode": "center", "manual_x": 0.6}}
            res = validate_project_payload(payload)
            self.assertEqual(res["crop"]["aspect_ratio"], aspect)
            self.assertEqual(res["crop"]["mode"], "center")
            self.assertEqual(res["crop"]["manual_x"], 0.6)

        # Invalid aspect ratio defaults safely to 9:16
        payload_invalid = {"crop": {"aspect_ratio": "21:9", "mode": "invalid_mode"}}
        res_invalid = validate_project_payload(payload_invalid)
        self.assertEqual(res_invalid["crop"]["aspect_ratio"], "9:16")
        self.assertEqual(res_invalid["crop"]["mode"], "smart")

    # 6. Subtitle editing, styles, and persistence
    def test_06_subtitle_editing(self):
        payload = {
            "subtitles": {
                "enabled": True,
                "style": "clean",
                "position": "top",
                "font_family": "Impact",
                "font_size": 52,
                "color": "#00FF00",
                "bold": True,
                "segments": [
                    {"id": "sub_1", "start": 0.5, "end": 1.8, "text": "Hello edited world"},
                    {"id": "sub_2", "start": 1.8, "end": 3.0, "text": "Second line here"}
                ]
            }
        }
        res = validate_project_payload(payload)
        self.assertEqual(res["subtitles"]["style"], "clean")
        self.assertEqual(res["subtitles"]["position"], "top")
        self.assertEqual(res["subtitles"]["font_family"], "Impact")
        self.assertEqual(len(res["subtitles"]["segments"]), 2)
        self.assertEqual(res["subtitles"]["segments"][0]["text"], "Hello edited world")

    # 7. Emoji and text overlay editing
    def test_07_emoji_and_text_overlays(self):
        payload = {
            "overlays": [
                {
                    "id": "ov_t1",
                    "type": "text",
                    "content": "VIRAL HOOK",
                    "font_size": 40,
                    "color": "#FFCC00",
                    "x_pct": 50,
                    "y_pct": 20,
                    "start_sec": 0.5,
                    "end_sec": 2.5
                },
                {
                    "id": "ov_e1",
                    "type": "emoji",
                    "content": "🔥",
                    "font_size": 60,
                    "color": "#FFFFFF",
                    "x_pct": 50,
                    "y_pct": 50,
                    "start_sec": 1.0,
                    "end_sec": 3.0
                }
            ]
        }
        res = validate_project_payload(payload)
        self.assertEqual(len(res["overlays"]), 2)
        self.assertEqual(res["overlays"][0]["type"], "text")
        self.assertEqual(res["overlays"][0]["content"], "VIRAL HOOK")
        self.assertEqual(res["overlays"][1]["type"], "emoji")
        self.assertEqual(res["overlays"][1]["content"], "🔥")

    # 8. Audio settings and upload validation
    def test_08_audio_settings_and_upload(self):
        payload = {
            "audio": {
                "original_enabled": True,
                "original_volume": 1.5,
                "music_volume": 0.4,
                "fade_in": 0.5,
                "fade_out": 0.8
            }
        }
        res = validate_project_payload(payload)
        self.assertEqual(res["audio"]["original_volume"], 1.5)
        self.assertEqual(res["audio"]["music_volume"], 0.4)
        self.assertEqual(res["audio"]["fade_in"], 0.5)

        # Test audio upload route with valid dummy mp3
        dummy_audio = io.BytesIO(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 100)
        resp = self.client.post(
            f"/api/editor/{self.test_job_id}/1/upload_audio",
            data={"audio_file": (dummy_audio, "background_beat.mp3")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        upload_data = resp.get_json()
        self.assertEqual(upload_data["status"], "success")
        self.assertTrue(upload_data["filename"].endswith(".mp3"))

        # Test rejected extension (.exe)
        bad_file = io.BytesIO(b"bad executable content")
        resp_bad = self.client.post(
            f"/api/editor/{self.test_job_id}/1/upload_audio",
            data={"audio_file": (bad_file, "malware.exe")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp_bad.status_code, 400)

    # 9. Playback speed settings
    def test_09_playback_speed_settings(self):
        for spd in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            payload = {"speed": spd}
            res = validate_project_payload(payload)
            self.assertEqual(res["speed"], spd)

        # Snaps unknown speed to nearest supported
        res_snap = validate_project_payload({"speed": 1.1})
        self.assertIn(res_snap["speed"], [1.0, 1.25])

    # 10. Visual adjustments
    def test_10_visual_adjustments(self):
        payload = {
            "adjustments": {
                "brightness": 0.2,
                "contrast": 1.3,
                "saturation": 1.4,
                "filter_preset": "vivid"
            }
        }
        res = validate_project_payload(payload)
        self.assertEqual(res["adjustments"]["brightness"], 0.2)
        self.assertEqual(res["adjustments"]["contrast"], 1.3)
        self.assertEqual(res["adjustments"]["saturation"], 1.4)
        self.assertEqual(res["adjustments"]["filter_preset"], "vivid")

    # 11. Project save and restoration
    def test_11_project_save_and_restoration(self):
        save_payload = {
            "trim": {"start": 0.8, "end": 2.8},
            "crop": {"aspect_ratio": "1:1", "mode": "center", "manual_x": 0.5},
            "speed": 1.25,
            "adjustments": {"brightness": 0.1, "contrast": 1.1, "saturation": 1.2, "filter_preset": "warm"}
        }
        resp = self.client.post(
            f"/api/editor/{self.test_job_id}/1/save",
            data=json.dumps(save_payload),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)

        # Retrieve and verify persistence
        resp_get = self.client.get(f"/api/editor/{self.test_job_id}/1/project")
        self.assertEqual(resp_get.status_code, 200)
        data = resp_get.get_json()["project"]
        self.assertEqual(data["trim"]["start"], 0.8)
        self.assertEqual(data["crop"]["aspect_ratio"], "1:1")
        self.assertEqual(data["speed"], 1.25)
        self.assertEqual(data["adjustments"]["filter_preset"], "warm")

    # 12. Project reset to original reel state
    def test_12_project_reset(self):
        resp = self.client.post(f"/api/editor/{self.test_job_id}/1/reset")
        self.assertEqual(resp.status_code, 200)
        reset_data = resp.get_json()["project"]
        # Verify reset reverted trim start to original 0.5
        self.assertEqual(reset_data["trim"]["start"], 0.5)
        self.assertEqual(reset_data["crop"]["aspect_ratio"], "9:16")

    # 13. Path traversal protection & job isolation
    def test_13_security_path_traversal(self):
        self.assertFalse(is_safe_id("../../etc"))
        self.assertFalse(is_safe_id("job_id/../secret"))
        self.assertFalse(is_safe_id("job_id\\secret"))
        self.assertTrue(is_safe_id("valid_job-123_abc"))

        # Save with malicious job_id rejected
        resp = self.client.post("/api/editor/..%2F..%2Fhack/1/save", json={})
        self.assertIn(resp.status_code, [400, 404])

    # 14. Duplicate concurrent export prevention
    def test_14_duplicate_export_prevention(self):
        proj = get_or_create_project(self.test_job_id, 1)
        task_key = f"{self.test_job_id}_1"

        with RENDER_LOCK:
            RENDER_TASKS[task_key] = {"status": "rendering", "progress": 50, "stage": "Rendering..."}

        try:
            res = start_editor_render(self.test_job_id, 1, proj)
            self.assertEqual(res["status"], "already_running")
        finally:
            with RENDER_LOCK:
                RENDER_TASKS.pop(task_key, None)

    # 15. Re-editing avoids AI model inference
    def test_15_re_editing_avoids_ai_inference(self):
        import src.feature_extractor as fe_mod
        import src.lstm_model as lstm_mod

        called_fe = False
        called_lstm = False

        orig_fe_init = fe_mod.ResNet50FeatureExtractor.__init__
        orig_lstm_forward = lstm_mod.VideoHighlightBiLSTM.forward

        def mock_fe_init(self, *args, **kwargs):
            nonlocal called_fe
            called_fe = True
            orig_fe_init(self, *args, **kwargs)

        def mock_lstm_forward(self, *args, **kwargs):
            nonlocal called_lstm
            called_lstm = True
            return orig_lstm_forward(self, *args, **kwargs)

        fe_mod.ResNet50FeatureExtractor.__init__ = mock_fe_init
        lstm_mod.VideoHighlightBiLSTM.forward = mock_lstm_forward

        try:
            # Perform complete project load, edit, save, reset
            self.client.get(f"/editor/{self.test_job_id}/1")
            self.client.post(f"/api/editor/{self.test_job_id}/1/save", json={"trim": {"start": 0.5, "end": 2.5}})
            self.client.post(f"/api/editor/{self.test_job_id}/1/reset")
            self.assertFalse(called_fe, "ResNet50FeatureExtractor was unnecessarily invoked during editing!")
            self.assertFalse(called_lstm, "VideoHighlightBiLSTM was unnecessarily invoked during editing!")
        finally:
            fe_mod.ResNet50FeatureExtractor.__init__ = orig_fe_init
            lstm_mod.VideoHighlightBiLSTM.forward = orig_lstm_forward

    # 16. Original generated clip preservation
    def test_16_original_clip_preservation(self):
        self.assertTrue(os.path.isfile(self.original_reel_path))
        current_bytes = os.path.getsize(self.original_reel_path)
        self.assertEqual(current_bytes, self.original_reel_bytes, "Original reel_01.mp4 was modified or corrupted!")

    # 17. Real FFmpeg export rendering and output video validation
    def test_17_real_ffmpeg_export_rendering(self):
        proj = get_or_create_project(self.test_job_id, 1)
        proj["trim"] = {"start": 0.5, "end": 2.5, "duration": 2.0}
        proj["crop"] = {"aspect_ratio": "9:16", "mode": "center", "manual_x": 0.5}
        proj["adjustments"] = {"brightness": 0.05, "contrast": 1.05, "saturation": 1.1, "filter_preset": "normal"}
        proj["speed"] = 1.0

        # Execute synchronous render
        execute_editor_render(self.test_job_id, 1, proj)

        edited_path = os.path.join(self.job_dir, "reel_01_edited.mp4")
        self.assertTrue(os.path.isfile(edited_path), "Exported reel_01_edited.mp4 was not generated")
        self.assertGreater(os.path.getsize(edited_path), 0, "Exported video file is empty")

        # Validate video using OpenCV
        cap = cv2.VideoCapture(edited_path)
        self.assertTrue(cap.isOpened(), "Could not open exported MP4 with OpenCV")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur = total_frames / fps if fps > 0 else 0
        cap.release()

        # Target 9:16 resolution is 1080x1920
        self.assertEqual(w, 1080)
        self.assertEqual(h, 1920)
        self.assertAlmostEqual(dur, 2.0, delta=0.5)

        # Original clip must remain untouched
        self.assertTrue(os.path.isfile(self.original_reel_path))
        self.assertEqual(os.path.getsize(self.original_reel_path), self.original_reel_bytes)

    # 18. Failed render recovery
    def test_18_failed_render_recovery(self):
        # Pass non-existent source video to simulate failure
        broken_proj = {
            "source_video_path": "non_existent_fake_path.mp4",
            "trim": {"start": 0, "end": 10},
            "crop": {"aspect_ratio": "9:16", "mode": "center"}
        }
        execute_editor_render(self.test_job_id, 99, broken_proj)
        status = get_render_status(self.test_job_id, 99)
        self.assertEqual(status["status"], "error")
        self.assertIsNotNone(status["error"])


if __name__ == "__main__":
    unittest.main()
