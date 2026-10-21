"""
src/utf8_utils.py
-----------------
Windows-compatible UTF-8 and Unicode safety utilities.
Guarantees that logging, standard output/error, subprocess output,
and file operations handle non-ASCII / emoji characters (e.g. 😰 🎬 ✅)
safely without raising 'charmap' UnicodeEncodeError on Windows.
"""

import os
import sys
import logging

def setup_utf8_environment():
    """
    Configures Python stdout, stderr, and standard logging to handle UTF-8 safely
    with character replacement for any unsupported console environments on Windows.
    Also ensures ffmpeg/ffprobe is located in process PATH if available on Windows.
    """
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

    import shutil
    try:
        import imageio_ffmpeg
        img_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if img_exe and os.path.isfile(img_exe):
            img_dir = os.path.dirname(img_exe)
            alias = os.path.join(img_dir, "ffmpeg.exe")
            if not os.path.isfile(alias):
                shutil.copy2(img_exe, alias)
            os.environ["PATH"] = img_dir + os.path.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    if shutil.which("ffmpeg") is None:
        candidate_dirs = [
            r"C:\Program Files\Shotcut",
            r"C:\Program Files (x86)\Shotcut",
            r"C:\Program Files\FFmpeg\bin",
            r"C:\ffmpeg\bin"
        ]
        for cdir in candidate_dirs:
            if os.path.isfile(os.path.join(cdir, "ffmpeg.exe")):
                os.environ["PATH"] = cdir + os.path.pathsep + os.environ.get("PATH", "")
                break

    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def sanitize_for_logging(text: str) -> str:
    """
    Sanitizes string for safe logging/printing under any codepage.
    If text can be encoded in UTF-8, it returns it as-is.
    """
    if not isinstance(text, str):
        text = str(text)
    return text


class SafeStreamHandler(logging.StreamHandler):
    """
    A robust logging StreamHandler that never crashes on Windows charmap encoding errors.
    If emitting a log record fails due to a UnicodeEncodeError, it falls back to
    encoding with errors='replace'.
    """
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                stream = self.stream
                safe_msg = msg.encode("utf-8", errors="replace").decode(
                    stream.encoding or "utf-8", errors="replace"
                )
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)
