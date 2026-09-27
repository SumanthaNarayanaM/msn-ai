"""FFmpeg integration: scene detection, frame sampling and audio extraction.

``ffmpeg``/``ffprobe`` are external binaries and are never installed or
downloaded automatically. When they are missing the provider reports the
missing binary by name so the UI can tell the user exactly what to install.

Frame sampling prefers *real* scene-change detection (``-vf select='gt(scene,
threshold)'``) so a talk or a slideshow is summarised by the frames that actually
changed, and only falls back to fixed-interval sampling when the container
cannot be seeked cheaply.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.ingestion.providers.base import CapabilityProvider, resolve_binary

logger = get_logger("msn_ai.ingestion.ffmpeg")

#: Per-frame timeout when extracting a single frame.
_FRAME_TIMEOUT = 120

#: ``[Parsed_showinfo_1 @ ...] n:   0 pts:  12345 pts_time:12.345 ...
_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def _parse_pts_times(ffmpeg_stderr: str) -> list[float]:
    """Extract the presentation timestamps printed by the ``showinfo`` filter."""
    times: list[float] = []
    for match in _PTS_RE.finditer(ffmpeg_stderr or ""):
        try:
            times.append(float(match.group(1)))
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
    return times


class FFmpegProvider(CapabilityProvider):
    """Frame/audio extraction backed by the local ffmpeg binaries."""

    name = "ffmpeg"

    def __init__(self, ffmpeg_cmd: str = "ffmpeg", ffprobe_cmd: str = "ffprobe") -> None:
        super().__init__()
        self.ffmpeg_cmd = ffmpeg_cmd or "ffmpeg"
        self.ffprobe_cmd = ffprobe_cmd or "ffprobe"
        self.ffmpeg_bin = resolve_binary(self.ffmpeg_cmd)
        self.ffprobe_bin = resolve_binary(self.ffprobe_cmd)
        if self.ffmpeg_bin:
            self.set_available()
        else:
            self.set_unavailable(
                f"FFmpeg is not installed ('{self.ffmpeg_cmd}' was not found on "
                "PATH). Without it MSN AI cannot sample video frames or extract "
                "audio for transcription. Install FFmpeg and reopen MSN AI."
            )

    def status(self) -> dict[str, Any]:
        data = super().status()
        data.update(
            {
                "ffmpeg": self.ffmpeg_bin,
                "ffprobe": self.ffprobe_bin,
            }
        )
        return data

    # --- probing ---------------------------------------------------------

    def probe(self, media_path: str | Path) -> dict[str, Any]:
        """Return ffprobe stream/format metadata, or ``{}`` when unavailable."""
        if not self.ffprobe_bin:
            return {}
        cmd = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=90, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("ffprobe failed for %s: %s", media_path, exc)
            return {}
        if completed.returncode != 0:
            return {}
        try:
            return json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return {}

    # --- frame sampling --------------------------------------------------

    def sample_frames(
        self,
        media_path: str | Path,
        output_dir: str | Path,
        duration: float = 0.0,
        interval: float = 5.0,
        max_frames: int = 120,
        scene_threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Extract representative frames.

        Tries scene detection first; if that yields nothing (static video, or a
        stream that cannot be seeked) it falls back to fixed-interval sampling.
        Each returned dict has ``path``, ``index``, ``timestamp`` and
        ``timestamp_human``.
        """
        if not self.ffmpeg_bin:
            raise RuntimeError(self.unavailable_reason)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("frame_*.jpg", "scene_*.jpg"):
            for stale in out_dir.glob(pattern):
                try:
                    stale.unlink()
                except OSError:  # pragma: no cover - defensive
                    pass

        frames = self._sample_scenes(media_path, out_dir, duration, max_frames, scene_threshold)
        if not frames:
            frames = self._sample_interval(media_path, out_dir, duration, interval, max_frames)
        return frames

    def _sample_scenes(
        self,
        media_path: str | Path,
        out_dir: Path,
        duration: float,
        max_frames: int,
        scene_threshold: float,
    ) -> list[dict[str, Any]]:
        if max_frames <= 0:
            return []
        pattern = str(out_dir / "scene_%04d.jpg")
        # ``showinfo`` prints each kept frame's presentation timestamp on
        # stderr. Guessing a timestamp from the frame index would make every
        # citation point at the wrong moment in the video.
        vf = f"select='gt(scene,{scene_threshold})',showinfo"
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(media_path),
            "-vf",
            vf,
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "3",
            "-y",
            pattern,
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Scene detection failed: %s", exc)
            return []
        if completed.returncode != 0:
            logger.debug(
                "Scene detection exited %s: %s", completed.returncode, completed.stderr[:300]
            )
        stamps = _parse_pts_times(completed.stderr)
        produced = sorted(out_dir.glob("scene_*.jpg"))
        frames: list[dict[str, Any]] = []
        for index, path in enumerate(produced):
            # Fall back to no timestamp (reported as unknown) rather than
            # inventing one when showinfo gave us nothing.
            frames.append(
                self._frame_entry(path, index, stamps[index] if index < len(stamps) else None)
            )
        return frames

    def _sample_interval(
        self,
        media_path: str | Path,
        out_dir: Path,
        duration: float,
        interval: float,
        max_frames: int,
    ) -> list[dict[str, Any]]:
        if max_frames <= 0:
            return []
        step = max(float(interval or 5.0), 0.5)
        count = max_frames
        if duration > 0:
            count = min(max_frames, max(int(duration / step), 1))
        timestamps: list[float] = []
        for i in range(count):
            timestamps.append(min(i * step, max(duration - 0.05, 0.0)) if duration > 0 else i * step)

        frames: list[dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps):
            target = out_dir / f"frame_{index:04d}.jpg"
            cmd = [
                self.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(media_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-y",
                str(target),
            ]
            try:
                completed = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=_FRAME_TIMEOUT, check=False
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if completed.returncode == 0 and target.exists() and target.stat().st_size > 0:
                frames.append(self._frame_entry(target, index, timestamp))
        return frames

    def _frame_entry(self, path: Path, index: int, timestamp: float | None) -> dict[str, Any]:
        known = timestamp is not None
        return {
            "path": str(path),
            "name": path.name,
            "index": index,
            "timestamp": round(float(timestamp), 3) if known else None,
            "timestamp_human": format_timestamp(float(timestamp)) if known else "unknown",
            "timestamp_known": known,
        }

    # --- audio -----------------------------------------------------------

    def extract_audio(self, media_path: str | Path, output_path: str | Path) -> bool:
        """Extract a 16 kHz mono WAV suitable for Whisper. True on success."""
        if not self.ffmpeg_bin:
            raise RuntimeError(self.unavailable_reason)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(target),
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return False
        return completed.returncode == 0 and target.exists() and target.stat().st_size > 44


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS`` or ``HH:MM:SS`` (mirrors the video parser)."""
    if not seconds or seconds < 0:
        return "00:00"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_ffmpeg_provider(settings) -> CapabilityProvider:
    return FFmpegProvider(
        ffmpeg_cmd=getattr(settings, "ffmpeg_cmd", "ffmpeg"),
        ffprobe_cmd=getattr(settings, "ffprobe_cmd", "ffprobe"),
    )
