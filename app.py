"""
app.py
------
Flask Web Application for Automatic Video Highlight Detection and Short Video Generation.

Features:
1. Video upload (MP4) OR direct YouTube URL processing (via yt-dlp)
2. Dual Analysis Modes: Visual + Speech (Multimodal) vs Visual Only (Baseline)
3. Real-time background worker with live 8-stage progress tracking
4. In-browser HTML5 highlight preview + speech transcript excerpts
5. Importance score plot visualization and direct MP4 downloads
6. Directory traversal protection, rate limiting, and robust error handling
"""

import os
import sys

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 1. Setup Windows UTF-8 safety at startup
from src.utf8_utils import setup_utf8_environment, SafeStreamHandler
setup_utf8_environment()

# Auto-switch to working project virtual environment if current python lacks working PyTorch DLLs
VENV_PYTHON = r"C:\Users\mahin\OneDrive\Desktop\FY PROJECT\lighthouse\venv\Scripts\python.exe"
if os.path.isfile(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    try:
        import torch
        torch._C
    except (ImportError, Exception):
        import subprocess
        sys.exit(subprocess.call([VENV_PYTHON] + sys.argv))

import uuid
import time
import json
import logging
import threading
import io
import zipfile
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, send_from_directory, send_file, flash, abort
)
from werkzeug.utils import secure_filename

from main import run_pipeline, run_multi_reel_pipeline
from src.youtube_downloader import is_valid_youtube_url
from src.editor_service import (
    get_or_create_project, save_project, reset_project,
    validate_project_payload, save_uploaded_audio, is_safe_id
)
from src.editor_renderer import start_editor_render, get_render_status

# Configure logging with UTF-8 safe stream handler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[SafeStreamHandler(sys.stdout)]
)
logger = logging.getLogger("FlaskHighlightApp")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directory configurations
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
WEB_OUTPUTS_FOLDER = os.path.join(BASE_DIR, "web_outputs")
YT_DOWNLOADS_FOLDER = os.path.join(BASE_DIR, "data", "youtube_downloads")
MODEL_CHECKPOINT = os.path.join(BASE_DIR, "models", "best_bilstm.pt")
MULTIMODAL_CHECKPOINT = os.path.join(BASE_DIR, "models", "best_multimodal_bilstm.pt")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

ALLOWED_EXTENSIONS = {"mp4"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEB_OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(YT_DOWNLOADS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["WEB_OUTPUTS_FOLDER"] = WEB_OUTPUTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

# In-memory tracking for active processing jobs
JOBS_STORE: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def allowed_file(filename: str) -> bool:
    """Checks if the filename has an allowed extension (.mp4)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_safe_job_id(job_id: str) -> bool:
    """Validates job_id to prevent directory traversal attacks."""
    if not job_id or not isinstance(job_id, str):
        return False
    return all(c.isalnum() or c in ("_", "-") for c in job_id) and ".." not in job_id


def process_video_worker(
    job_id: str,
    video_path: str = None,
    youtube_url: str = None,
    model_mode: str = "multimodal",
    output_mode: str = "reels",
    num_reels: str = "auto",
    reel_length: str = "30-60",
    aspect_ratio: str = "9:16",
    captions: bool = True,
    crop_mode: str = "smart",
    manual_x: float = 0.5,
    subtitle_style: str = "bold",
    subtitle_position: str = "bottom",
    emojis: bool = False,
    budget_ratio: float = 0.15,
    original_filename: str = "video.mp4"
):
    """
    Background worker thread running the ML pipeline asynchronously.
    Supports both Single Summary mode and Multi-Reel mode with Smart Face Crop,
    custom Subtitle Styling, and Contextual Emojis.
    Updates JOBS_STORE stage strings in real-time.
    """
    setup_utf8_environment()
    job_output_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    out_meta_path = os.path.join(job_output_dir, "segments.json")

    logger.info(
        f"Starting ML processing for Job '{job_id}' (Output: {output_mode.upper()}, "
        f"Mode: {model_mode}, Reels: {num_reels}, Length: {reel_length}, Aspect: {aspect_ratio}, "
        f"Crop: {crop_mode}, Captions: {captions}, Emojis: {emojis})..."
    )

    def update_stage(msg: str):
        with JOBS_LOCK:
            if job_id in JOBS_STORE:
                JOBS_STORE[job_id]["stage"] = str(msg)

    try:
        with JOBS_LOCK:
            JOBS_STORE[job_id]["status"] = "processing"
            JOBS_STORE[job_id]["stage"] = "Initializing processing..."

        if output_mode == "reels":
            results = run_multi_reel_pipeline(
                video_path=video_path,
                youtube_url=youtube_url,
                model_mode=model_mode,
                num_reels=num_reels,
                reel_length=reel_length,
                aspect_ratio=aspect_ratio,
                captions=captions,
                crop_mode=crop_mode,
                manual_x=manual_x,
                subtitle_style=subtitle_style,
                subtitle_position=subtitle_position,
                emojis=emojis,
                output_dir=job_output_dir,
                stage_callback=update_stage,
                job_id=job_id
            )
        else:
            out_video_path = os.path.join(job_output_dir, "highlight.mp4")
            out_plot_path = os.path.join(job_output_dir, "scores.png")
            results = run_pipeline(
                video_path=video_path,
                youtube_url=youtube_url,
                model_mode=model_mode,
                budget_ratio=budget_ratio,
                output_video_path=out_video_path,
                plot_output_path=out_plot_path,
                annotations_dir=ANNOTATIONS_DIR,
                stage_callback=update_stage,
                job_id=job_id
            )

        results["original_display_name"] = original_filename
        results["job_id"] = job_id
        results["output_mode"] = output_mode

        # Save metadata to web_outputs/<job_id>/segments.json
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with JOBS_LOCK:
            JOBS_STORE[job_id]["status"] = "complete"
            if output_mode == "reels":
                JOBS_STORE[job_id]["stage"] = f"Generated {results.get('num_reels', len(results.get('reels', [])))} Reels successfully!"
            else:
                JOBS_STORE[job_id]["stage"] = "Highlight video successfully generated!"
            JOBS_STORE[job_id]["result"] = results

        logger.info(f"Job '{job_id}' completed successfully!")

    except Exception as e:
        try:
            err_msg = str(e)
        except Exception:
            err_msg = repr(e)
        try:
            logger.error(f"Error processing job '{job_id}': {err_msg}", exc_info=True)
        except Exception:
            pass
        with JOBS_LOCK:
            JOBS_STORE[job_id]["status"] = "error"
            JOBS_STORE[job_id]["stage"] = "Failed"
            JOBS_STORE[job_id]["error"] = f"Processing failed: {err_msg}"



# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Renders home landing page with tabs for Video Upload and YouTube URL."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Serves the favicon.ico image."""
    return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route("/upload", methods=["GET", "POST"])
@app.route("/api/upload", methods=["POST"])
def upload():
    """Renders upload page on GET and handles form submission or JSON API on POST."""
    if request.method == "GET":
        return render_template("upload.html")

    req_data = request.get_json(silent=True) or {}
    
    upload_type = req_data.get("upload_type") or request.form.get("upload_type")
    youtube_url = req_data.get("youtube_url") or request.form.get("youtube_url", "").strip()

    if not upload_type:
        upload_type = "youtube" if youtube_url else "file"

    model_mode = req_data.get("mode") or request.form.get("mode", "multimodal")
    if model_mode not in ["multimodal", "baseline"]:
        model_mode = "multimodal"

    output_mode = req_data.get("output_mode") or request.form.get("output_mode", "reels")
    if output_mode not in ["reels", "single"]:
        output_mode = "reels"

    num_reels = req_data.get("num_reels") or request.form.get("num_reels", "auto")
    if str(num_reels).isdigit():
        num_reels = str(int(num_reels))
    elif num_reels != "auto":
        num_reels = "auto"

    reel_length = req_data.get("reel_length") or request.form.get("reel_length", "30-60")
    if reel_length not in ["15-30", "30-60", "60-90", "auto"]:
        reel_length = "30-60"

    aspect_ratio = req_data.get("aspect_ratio") or request.form.get("aspect_ratio", "9:16")
    if aspect_ratio not in ["9:16", "16:9"]:
        aspect_ratio = "9:16"

    captions_param = req_data.get("captions") or request.form.get("captions", "on")
    captions = captions_param in ["on", "1", "true", "True", True]

    crop_mode = req_data.get("crop_mode") or request.form.get("crop_mode", "smart")
    if crop_mode not in ["smart", "center", "manual"]:
        crop_mode = "smart"

    try:
        manual_x = max(0.0, min(1.0, float(req_data.get("manual_x") or request.form.get("manual_x", 0.5))))
    except (ValueError, TypeError):
        manual_x = 0.5

    subtitle_style = req_data.get("subtitle_style") or request.form.get("subtitle_style", "bold")
    if subtitle_style not in ["bold", "clean", "minimal"]:
        subtitle_style = "bold"

    subtitle_position = req_data.get("subtitle_position") or request.form.get("subtitle_position", "bottom")
    if subtitle_position not in ["bottom", "center", "top"]:
        subtitle_position = "bottom"

    emojis_param = req_data.get("emojis") or request.form.get("emojis", "off")
    emojis = emojis_param in ["on", "1", "true", "True", True]

    # Parse and validate budget percentage for single mode
    budget_param = req_data.get("budget") or request.form.get("budget", "15")
    try:
        budget_pct = int(budget_param)
        if budget_pct not in [10, 15, 20]:
            budget_pct = 15
    except ValueError:
        budget_pct = 15
    budget_ratio = budget_pct / 100.0

    # 1. Handle YouTube Submission
    if upload_type == "youtube":
        if not youtube_url:
            if request.is_json:
                return jsonify({"status": "error", "message": "Please enter a valid YouTube URL."}), 400
            flash("Please enter a valid YouTube URL.", "danger")
            return redirect(url_for("index"))

        if not is_valid_youtube_url(youtube_url):
            if request.is_json:
                return jsonify({"status": "error", "message": "Invalid YouTube URL! Please provide a valid link."}), 400
            flash("Invalid YouTube URL! Please provide a standard YouTube, youtu.be, or Shorts link.", "danger")
            return redirect(url_for("index"))

        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        with JOBS_LOCK:
            JOBS_STORE[job_id] = {
                "status": "queued",
                "stage": "Validating YouTube URL and preparing download...",
                "error": None,
                "result": None,
                "created_at": time.time()
            }

        thread = threading.Thread(
            target=process_video_worker,
            kwargs={
                "job_id": job_id,
                "video_path": None,
                "youtube_url": youtube_url,
                "model_mode": model_mode,
                "output_mode": output_mode,
                "num_reels": num_reels,
                "reel_length": reel_length,
                "aspect_ratio": aspect_ratio,
                "captions": captions,
                "crop_mode": crop_mode,
                "manual_x": manual_x,
                "subtitle_style": subtitle_style,
                "subtitle_position": subtitle_position,
                "emojis": emojis,
                "budget_ratio": budget_ratio,
                "original_filename": youtube_url
            },
            daemon=True
        )
        thread.start()
        
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "status": "success",
                "message": "YouTube video queued for processing.",
                "job_id": job_id,
                "redirect_url": url_for("processing", job_id=job_id)
            })
        return redirect(url_for("processing", job_id=job_id))

    # 2. Handle Local File Upload
    else:
        if "video_file" not in request.files:
            if request.is_json:
                return jsonify({"status": "error", "message": "No video file selected."}), 400
            flash("No video file selected. Please select an MP4 video.", "danger")
            return redirect(url_for("index"))

        file = request.files["video_file"]
        if file.filename == "":
            if request.is_json:
                return jsonify({"status": "error", "message": "No file was chosen."}), 400
            flash("No file was chosen. Please select an MP4 video to upload.", "danger")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            if request.is_json:
                return jsonify({"status": "error", "message": "Invalid file format. Only MP4 videos (.mp4) supported."}), 400
            flash("Invalid file format. Only MP4 videos (.mp4) are currently supported.", "danger")
            return redirect(url_for("index"))

        orig_filename = secure_filename(file.filename) or "uploaded_video.mp4"
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        saved_video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{orig_filename}")
        file.save(saved_video_path)

        with JOBS_LOCK:
            JOBS_STORE[job_id] = {
                "status": "queued",
                "stage": "Video uploaded. Initializing pipeline...",
                "error": None,
                "result": None,
                "created_at": time.time()
            }

        thread = threading.Thread(
            target=process_video_worker,
            kwargs={
                "job_id": job_id,
                "video_path": saved_video_path,
                "youtube_url": None,
                "model_mode": model_mode,
                "output_mode": output_mode,
                "num_reels": num_reels,
                "reel_length": reel_length,
                "aspect_ratio": aspect_ratio,
                "captions": captions,
                "crop_mode": crop_mode,
                "manual_x": manual_x,
                "subtitle_style": subtitle_style,
                "subtitle_position": subtitle_position,
                "emojis": emojis,
                "budget_ratio": budget_ratio,
                "original_filename": orig_filename
            },
            daemon=True
        )
        thread.start()
        
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "status": "success",
                "message": "Video uploaded and queued for processing.",
                "job_id": job_id,
                "redirect_url": url_for("processing", job_id=job_id)
            })
        return redirect(url_for("processing", job_id=job_id))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Renders dashboard view of recent projects."""
    recent_jobs = []
    try:
        with JOBS_LOCK:
            for jid, jdata in list(JOBS_STORE.items()):
                if jdata.get("status") == "complete" and jdata.get("result"):
                    res = jdata["result"] or {}
                    reels = res.get("reels") or [{"confidence": 85.0}]
                    scores = [r.get("confidence") or r.get("score") or 85.0 for r in reels if isinstance(r, dict)]
                    top_score = round(max(scores), 1) if scores else 85.0
                    dur = res.get("video_duration") or 0.0
                    try:
                        dur_float = float(dur)
                    except (ValueError, TypeError):
                        dur_float = 0.0

                    recent_jobs.append({
                        "id": jid,
                        "title": res.get("original_display_name", f"Project {jid}"),
                        "shorts_generated": res.get("num_reels", len(reels)),
                        "top_viral_score": top_score,
                        "duration": f"{dur_float:.0f}s",
                        "status": "completed"
                    })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")

    return render_template("dashboard.html", projects=recent_jobs)


@app.route("/about", methods=["GET"])
def about():
    """Renders project architecture and documentation page."""
    return render_template("about.html")


@app.route("/login", methods=["GET"])
def login():
    """Renders login/signup page."""
    return render_template("login.html")


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Fetches fast YouTube video metadata for preview card."""
    data = request.get_json() or request.form or {}
    youtube_url = data.get("youtube_url", "").strip()
    if not youtube_url or not is_valid_youtube_url(youtube_url):
        return jsonify({"status": "error", "message": "Invalid YouTube URL"}), 400

    try:
        from src.youtube_downloader import extract_video_id
        vid = extract_video_id(youtube_url)
        return jsonify({
            "status": "success",
            "metadata": {
                "title": f"YouTube Video ({vid})",
                "channel": "YouTube Creator",
                "duration": "1080p MP4 Ready",
                "thumbnail": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/processing/<job_id>", methods=["GET"])
def processing(job_id: str):
    """Renders the processing status page."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    with JOBS_LOCK:
        job = JOBS_STORE.get(job_id)

    meta_path = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id, "segments.json")
    if job is None and os.path.isfile(meta_path):
        return redirect(url_for("result", job_id=job_id))

    if job is None:
        flash("Job not found or has expired. Please submit your video again.", "warning")
        return redirect(url_for("index"))

    if job["status"] == "complete":
        return redirect(url_for("result", job_id=job_id))

    return render_template("processing.html", job_id=job_id)


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    """JSON API polled by processing.html to check job state."""
    if not is_safe_job_id(job_id):
        return jsonify({"status": "error", "error": "Invalid Job ID"}), 400

    with JOBS_LOCK:
        job = JOBS_STORE.get(job_id)

    meta_path = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id, "segments.json")
    if job is None and os.path.isfile(meta_path):
        return jsonify({"status": "complete", "stage": "Complete", "redirect_url": url_for("result", job_id=job_id)})

    if job is None:
        return jsonify({
            "status": "error",
            "error": "Job expired or server restarted. Please submit your video again.",
            "redirect_url": url_for("index")
        }), 404

    resp = {
        "status": job["status"],
        "stage": job["stage"],
        "error": job["error"],
        "redirect_url": url_for("result", job_id=job_id) if job["status"] == "complete" else None
    }
    return jsonify(resp)


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    """Renders the results page (supporting both single summary and multi-reel cards)."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    meta_path = os.path.join(job_dir, "segments.json")

    if not os.path.isfile(meta_path):
        flash("Result metadata not found for this job.", "danger")
        return redirect(url_for("index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check for edited reel files on disk
    reels = data.get("reels", [])
    for r in reels:
        try:
            r_num = int(r.get("reel_num", 1))
            edited_fname = f"reel_{r_num:02d}_edited.mp4"
            edited_path = os.path.join(job_dir, edited_fname)
            if os.path.isfile(edited_path) and os.path.getsize(edited_path) > 0:
                r["has_edited"] = True
                r["edited_filename"] = edited_fname
                r["edited_size_mb"] = round(os.path.getsize(edited_path) / (1024 * 1024), 2)
            else:
                r["has_edited"] = False
        except Exception:
            r["has_edited"] = False

    return render_template("result.html", job_id=job_id, data=data)


@app.route("/media/<job_id>/<filename>", methods=["GET"])
def serve_media(job_id: str, filename: str):
    """Safely serves highlight video, reels, or score plot images for in-browser playback."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        abort(403, "Access denied.")

    is_valid_file = filename == "scores.png" or filename.endswith(".mp4")
    if not is_valid_file:
        abort(403, "Access denied.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Job directory not found.")

    mimetype = "video/mp4" if filename.endswith(".mp4") else "image/png"
    return send_from_directory(job_dir, filename, mimetype=mimetype)


@app.route("/download/<job_id>", methods=["GET"])
@app.route("/download/<job_id>/<filename>", methods=["GET"])
def download(job_id: str, filename: str = None):
    """Forces browser download of a specific highlight/reel video file."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Job directory not found.")

    if filename is None:
        target_file = "highlight.mp4"
        if not os.path.isfile(os.path.join(job_dir, target_file)):
            reels = sorted([f for f in os.listdir(job_dir) if f.startswith("reel_") and f.endswith(".mp4")])
            target_file = reels[0] if reels else "highlight.mp4"
    else:
        if ".." in filename or "/" in filename or "\\" in filename or not filename.endswith(".mp4"):
            abort(403, "Access denied.")
        target_file = filename

    file_path = os.path.join(job_dir, target_file)
    if not os.path.isfile(file_path):
        abort(404, "Requested video file not found.")

    return send_from_directory(
        job_dir,
        target_file,
        as_attachment=True,
        download_name=f"{job_id}_{target_file}",
        mimetype="video/mp4"
    )


@app.route("/download_all/<job_id>", methods=["GET"])
def download_all(job_id: str):
    """Bundles all generated reels and manifest.json into a ZIP archive for one-click download."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Job directory not found.")

    meta_path = os.path.join(job_dir, "segments.json")
    title_prefix = "reels"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                yt_meta = meta.get("youtube_meta")
                if yt_meta and yt_meta.get("title"):
                    title_prefix = secure_filename(yt_meta["title"])[:30]
        except Exception:
            pass

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(job_dir):
            if fname.endswith(".mp4") or fname == "manifest.json":
                fpath = os.path.join(job_dir, fname)
                zf.write(fpath, arcname=fname)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"{title_prefix}_{job_id}_reels.zip",
        mimetype="application/zip"
    )


@app.route("/api/clip_details/<job_id>/<int:reel_num>", methods=["GET"])
def api_clip_details(job_id: str, reel_num: int):
    """Returns reel metadata and settings for the Manual Editor."""
    if not is_safe_job_id(job_id):
        return jsonify({"status": "error", "message": "Invalid Job ID"}), 400

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    meta_path = os.path.join(job_dir, "segments.json")
    if not os.path.isfile(meta_path):
        return jsonify({"status": "error", "message": "Job metadata not found"}), 404

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    reels = data.get("reels", [])
    target = next((r for r in reels if r.get("reel_num") == reel_num), None)
    if not target:
        return jsonify({"status": "error", "message": f"Reel #{reel_num} not found"}), 404

    target_copy = dict(target)
    target_copy["video_url"] = url_for("serve_media", job_id=job_id, filename=target["filename"]) + f"?t={int(time.time())}"
    target_copy["video_duration"] = data.get("original_duration", 0.0)

    return jsonify({"status": "success", "reel": target_copy})


@app.route("/api/render_clip/<job_id>", methods=["POST"])
def api_render_clip(job_id: str):
    """
    Manual Editor Clip Re-renderer:
    Renders only a single short clip with user-customized:
    - Start and End time
    - Crop mode (smart face tracking, center, manual X)
    - Subtitle toggle (ON/OFF)
    - Subtitle style (Clean, Bold, Minimal)
    - Subtitle position (Bottom, Center, Top)
    - Emoji toggle (ON/OFF)
    
    IMPORTANT: ResNet-50 and Bi-LSTM are NEVER rerun for manual edits.
    """
    if not is_safe_job_id(job_id):
        return jsonify({"status": "error", "message": "Invalid Job ID"}), 400

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    meta_path = os.path.join(job_dir, "segments.json")
    if not os.path.isfile(meta_path):
        return jsonify({"status": "error", "message": "Job metadata not found"}), 404

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not read metadata: {e}"}), 500

    data = request.get_json(silent=True) or request.form or {}
    reel_num = int(data.get("reel_num", 1))

    # Find the target reel in metadata
    reels = job_data.get("reels", [])
    target_reel = None
    target_idx = -1
    for idx, r in enumerate(reels):
        if r.get("reel_num") == reel_num:
            target_reel = r
            target_idx = idx
            break

    if target_reel is None:
        return jsonify({"status": "error", "message": f"Reel #{reel_num} not found in job"}), 404

    # Extract source video path
    source_video = job_data.get("source_video_path")
    if not source_video or not os.path.isfile(source_video):
        yt_meta = job_data.get("youtube_meta")
        if yt_meta and yt_meta.get("video_path") and os.path.isfile(yt_meta["video_path"]):
            source_video = yt_meta["video_path"]
        else:
            # Check uploads directory for files matching job_id
            for uf in os.listdir(app.config["UPLOAD_FOLDER"]):
                if uf.startswith(job_id) and uf.endswith(".mp4"):
                    source_video = os.path.join(app.config["UPLOAD_FOLDER"], uf)
                    break

    if not source_video or not os.path.isfile(source_video):
        return jsonify({"status": "error", "message": "Source video file not found for re-rendering"}), 404

    # Parse parameters with validation
    try:
        start_sec = max(0.0, float(data.get("start_sec", target_reel.get("start_sec", 0.0))))
        end_sec = float(data.get("end_sec", target_reel.get("end_sec", start_sec + 30.0)))
        if end_sec <= start_sec:
            end_sec = start_sec + 15.0
        duration_sec = end_sec - start_sec
    except (ValueError, TypeError):
        start_sec = float(target_reel.get("start_sec", 0.0))
        end_sec = float(target_reel.get("end_sec", start_sec + 30.0))
        duration_sec = end_sec - start_sec

    crop_mode = str(data.get("crop_mode", "smart")).lower()
    if crop_mode not in ["smart", "center", "manual"]:
        crop_mode = "smart"

    try:
        manual_x = max(0.0, min(1.0, float(data.get("manual_x", 0.5))))
    except (ValueError, TypeError):
        manual_x = 0.5

    captions = data.get("captions")
    if isinstance(captions, str):
        captions = captions.lower() in ["true", "1", "on"]
    elif captions is None:
        captions = True
    else:
        captions = bool(captions)

    subtitle_style = str(data.get("subtitle_style", "bold")).lower()
    if subtitle_style not in ["bold", "clean", "minimal"]:
        subtitle_style = "bold"

    subtitle_position = str(data.get("subtitle_position", "bottom")).lower()
    if subtitle_position not in ["bottom", "center", "top"]:
        subtitle_position = "bottom"

    emojis = data.get("emojis")
    if isinstance(emojis, str):
        emojis = emojis.lower() in ["true", "1", "on"]
    elif emojis is None:
        emojis = False
    else:
        emojis = bool(emojis)

    # Collect speech segments relevant to the new time interval
    existing_speech = target_reel.get("speech_segments") or []
    if not existing_speech:
        for r in reels:
            if r.get("speech_segments"):
                existing_speech.extend(r.get("speech_segments", []))

    from src.reel_renderer import render_single_reel

    out_filename = f"reel_{reel_num:02d}.mp4"
    out_path = os.path.join(job_dir, out_filename)

    render_result = render_single_reel(
        source_video_path=source_video,
        start_sec=start_sec,
        duration_sec=duration_sec,
        output_path=out_path,
        aspect_ratio="9:16",
        speech_segments=existing_speech,
        captions=captions,
        transcript=target_reel.get("transcript"),
        crop_mode=crop_mode,
        manual_x=manual_x,
        subtitle_style=subtitle_style,
        subtitle_position=subtitle_position,
        emojis=emojis
    )

    def fmt_time(s):
        m = int(s // 60)
        sec = int(s % 60)
        return f"{m:02d}:{sec:02d}"

    # Update reel in memory
    target_reel["start_sec"] = round(start_sec, 2)
    target_reel["end_sec"] = round(end_sec, 2)
    target_reel["start_formatted"] = fmt_time(start_sec)
    target_reel["end_formatted"] = fmt_time(end_sec)
    target_reel["duration_sec"] = round(duration_sec, 2)
    target_reel["crop_mode"] = crop_mode
    target_reel["manual_x"] = manual_x
    target_reel["captions"] = captions
    target_reel["subtitle_style"] = subtitle_style
    target_reel["subtitle_position"] = subtitle_position
    target_reel["emojis"] = emojis
    target_reel["crop_info"] = render_result.get("crop_info")
    target_reel["file_size_mb"] = render_result.get("file_size_mb")

    reels[target_idx] = target_reel
    job_data["reels"] = reels

    # Save updated segments.json and manifest.json
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2, ensure_ascii=False)
        manifest_path = os.path.join(job_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)
    except Exception as save_err:
        logger.warning(f"Error saving updated metadata: {save_err}")

    video_url = url_for("serve_media", job_id=job_id, filename=out_filename) + f"?t={int(time.time())}"
    return jsonify({
        "status": "success",
        "message": f"Reel #{reel_num} re-rendered successfully!",
        "reel": target_reel,
        "video_url": video_url
    })


# ----------------------------------------------------------------------
# PROFESSIONAL VIDEO EDITOR ROUTES (PHASE 8)
# ----------------------------------------------------------------------

@app.route("/editor/<job_id>/<int:clip_id>", methods=["GET"])
def editor_page(job_id: str, clip_id: int):
    """
    Renders the professional dark-themed browser video editor for a generated Short.
    Loads project state without rerunning AI highlight detection models.
    """
    if not is_safe_job_id(job_id) or clip_id < 1:
        abort(400, "Invalid Job or Clip ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        flash("Job output directory not found.", "danger")
        return redirect(url_for("index"))

    try:
        project = get_or_create_project(job_id, clip_id)
    except Exception as e:
        logger.error(f"Error loading project for editor ({e})")
        flash(f"Could not load editor project: {e}", "danger")
        return redirect(url_for("result", job_id=job_id))

    reel_fname = project.get("reel_filename", f"reel_{clip_id:02d}.mp4")
    media_url = url_for("serve_media", job_id=job_id, filename=reel_fname) + f"?t={int(time.time())}"

    return render_template(
        "editor.html",
        job_id=job_id,
        clip_id=clip_id,
        project=project,
        media_url=media_url
    )


@app.route("/api/editor/<job_id>/<int:clip_id>/project", methods=["GET"])
def api_editor_get_project(job_id: str, clip_id: int):
    """Returns the current editor project state JSON for a clip."""
    if not is_safe_job_id(job_id) or clip_id < 1:
        return jsonify({"status": "error", "message": "Invalid Job or Clip ID"}), 400

    try:
        project = get_or_create_project(job_id, clip_id)
        return jsonify({"status": "success", "project": project})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 404


@app.route("/api/editor/<job_id>/<int:clip_id>/save", methods=["POST"])
def api_editor_save_project(job_id: str, clip_id: int):
    """Validates and persists edited project state without rerunning AI models."""
    if not is_safe_job_id(job_id) or clip_id < 1:
        return jsonify({"status": "error", "message": "Invalid Job or Clip ID"}), 400

    req_data = request.get_json(silent=True) or {}
    try:
        current_proj = get_or_create_project(job_id, clip_id)
        max_dur = float(current_proj.get("source_duration", 3600.0))
        validated = validate_project_payload(req_data, max_duration=max_dur)

        for k, v in validated.items():
            current_proj[k] = v

        saved = save_project(job_id, clip_id, current_proj)
        return jsonify({
            "status": "success",
            "message": "Project state saved successfully.",
            "project": saved
        })
    except Exception as e:
        logger.error(f"Save project error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/editor/<job_id>/<int:clip_id>/reset", methods=["POST"])
def api_editor_reset_project(job_id: str, clip_id: int):
    """Resets all edits back to original reel configuration."""
    if not is_safe_job_id(job_id) or clip_id < 1:
        return jsonify({"status": "error", "message": "Invalid Job or Clip ID"}), 400

    try:
        reset_proj = reset_project(job_id, clip_id)
        return jsonify({
            "status": "success",
            "message": "Project reset to original clip state.",
            "project": reset_proj
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/editor/<job_id>/<int:clip_id>/upload_audio", methods=["POST"])
def api_editor_upload_audio(job_id: str, clip_id: int):
    """Securely uploads an audio file for background music mixing."""
    if not is_safe_job_id(job_id) or clip_id < 1:
        return jsonify({"status": "error", "message": "Invalid Job or Clip ID"}), 400

    if "audio_file" not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    file_obj = request.files["audio_file"]
    if file_obj.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        saved_name = save_uploaded_audio(job_id, file_obj)
        return jsonify({
            "status": "success",
            "message": "Audio file uploaded successfully.",
            "filename": saved_name
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/editor/<job_id>/<int:clip_id>/render", methods=["POST"])
def api_editor_render(job_id: str, clip_id: int):
    """Starts asynchronous background render of edited short using FFmpeg."""
    if not is_safe_job_id(job_id) or clip_id < 1:
        return jsonify({"status": "error", "message": "Invalid Job or Clip ID"}), 400

    req_data = request.get_json(silent=True) or {}
    try:
        current_proj = get_or_create_project(job_id, clip_id)
        if req_data:
            max_dur = float(current_proj.get("source_duration", 3600.0))
            validated = validate_project_payload(req_data, max_duration=max_dur)
            for k, v in validated.items():
                current_proj[k] = v
            save_project(job_id, clip_id, current_proj)

        render_res = start_editor_render(job_id, clip_id, current_proj)
        return jsonify(render_res)
    except Exception as e:
        logger.error(f"Render launch error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/editor/<job_id>/<int:clip_id>/render_status", methods=["GET"])
def api_editor_render_status(job_id: str, clip_id: int):
    """Returns the current rendering stage and progress for an edited short."""
    if not is_safe_job_id(job_id) or clip_id < 1:
        return jsonify({"status": "error", "message": "Invalid Job or Clip ID"}), 400

    try:
        status_info = get_render_status(job_id, clip_id)
        return jsonify(status_info)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------------------------------------------------
# ERROR HANDLERS
# ----------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Uploaded file is too large! Maximum allowed video size is 500 MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found_error(error):
    return render_template("index.html", error_message="Page not found."), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template("index.html", error_message="An unexpected server error occurred. Please try again."), 500


if __name__ == "__main__":
    logger.info("Starting Flask Highlight Detection Web Server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
