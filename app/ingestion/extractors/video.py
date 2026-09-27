"""Video ingestion: real container metadata, optional frames, optional transcript.

Two layers:

1. A dependency-free container parser (ISO-BMFF/MP4/MOV, Matroska/WebM, RIFF/AVI)
   that always works, yielding duration, resolution, frame rate and track info.
2. Optional ffmpeg-driven frame sampling and a speech provider for transcripts.

When ffmpeg or a speech model is missing, the extraction still succeeds with the
metadata it genuinely read and records an explicit ``unavailable`` entry, which
the API and UI surface verbatim. MSN AI never invents a summary of frames or
speech it did not process.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.errors import ExtractionError
from app.core.logging import get_logger
from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

logger = get_logger("msn_ai.ingestion.video")

#: Read at most this much of the container header; enough for moov/Segment.
MAX_HEADER_BYTES = 8 * 1024 * 1024


@dataclass
class VideoMetadata:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str = ""
    audio_codec: str = ""
    has_audio: bool = False
    container: str = ""
    frame_count: int = 0
    bitrate: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "duration_seconds": round(self.duration, 3),
            "duration_human": format_timestamp(self.duration),
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}" if self.width and self.height else "",
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "has_audio": self.has_audio,
            "container": self.container,
        }
        data.update(self.extra)
        return data


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS`` or ``HH:MM:SS``."""
    if seconds is None or seconds < 0 or math.isnan(seconds):
        return "00:00"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# --------------------------------------------------------------------------
# ISO base media file format (MP4 / MOV / 3GP)
# --------------------------------------------------------------------------

_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}


def _iter_boxes(data: bytes, start: int, end: int):
    """Yield ``(type, payload_start, payload_end)`` for ISO-BMFF boxes."""
    pos = start
    while pos + 8 <= end:
        try:
            size = struct.unpack(">I", data[pos : pos + 4])[0]
            box_type = data[pos + 4 : pos + 8]
        except struct.error:
            return
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield box_type, pos + header, pos + size
        pos += size


def _find_box(data: bytes, path: tuple[bytes, ...], start: int, end: int) -> tuple[int, int] | None:
    """Depth-first search for a box path, returning its payload range."""
    if not path:
        return (start, end)
    head, rest = path[0], path[1:]
    for box_type, p_start, p_end in _iter_boxes(data, start, end):
        if box_type == head:
            found = _find_box(data, rest, p_start, p_end)
            if found:
                return found
    return None


def parse_mp4(path: Path) -> VideoMetadata | None:
    """Parse duration, resolution and codecs from an MP4/MOV container."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_HEADER_BYTES)
    except OSError:
        return None
    if len(data) < 16:
        return None
    if not (data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide")):
        return None

    meta = VideoMetadata(container="mp4/mov")
    brand = data[8:12]
    if brand:
        meta.extra["major_brand"] = brand.decode("ascii", "replace")

    # mvhd: movie timescale + duration
    mvhd = _find_box(data, (b"moov", b"mvhd"), 0, len(data))
    movie_timescale = 0
    if mvhd:
        s, e = mvhd
        payload = data[s:e]
        if len(payload) >= 20:
            version = payload[0]
            if version == 1 and len(payload) >= 32:
                timescale = struct.unpack(">I", payload[20:24])[0]
                duration = struct.unpack(">Q", payload[24:32])[0]
            else:
                timescale = struct.unpack(">I", payload[12:16])[0]
                duration = struct.unpack(">I", payload[16:20])[0]
            movie_timescale = timescale or 0
            if movie_timescale:
                meta.duration = duration / movie_timescale

    # Track boxes live *inside* moov, not at the top level.
    moov = _find_box(data, (b"moov",), 0, len(data))
    if not moov:
        # A file that announces itself as MP4 but has no moov is not a video.
        return None
    trak_ranges = [
        (start, end)
        for box_type, start, end in _iter_boxes(data, moov[0], moov[1])
        if box_type == b"trak"
    ]

    duration_seconds_by_track: list[tuple[str, float]] = []
    tracks_seen = 0
    for trak_start, trak_end in trak_ranges:
        tracks_seen += 1
        # tkhd: display width/height (16.16 fixed point)
        width = height = 0
        tkhd = _find_box(data, (b"tkhd",), trak_start, trak_end)
        if tkhd:
            s, e = tkhd
            payload = data[s:e]
            if len(payload) >= 84:
                width = struct.unpack(">I", payload[-8:-4])[0] >> 16
                height = struct.unpack(">I", payload[-4:])[0] >> 16

        handler = ""
        hdlr = _find_box(data, (b"mdia", b"hdlr"), trak_start, trak_end)
        if hdlr:
            s, e = hdlr
            payload = data[s:e]
            if len(payload) >= 12:
                handler = payload[8:12].decode("ascii", "replace")

        track_duration = 0.0
        mdhd = _find_box(data, (b"mdia", b"mdhd"), trak_start, trak_end)
        if mdhd:
            s, e = mdhd
            payload = data[s:e]
            if len(payload) >= 20:
                version = payload[0]
                if version == 1 and len(payload) >= 32:
                    ts = struct.unpack(">I", payload[20:24])[0]
                    dur = struct.unpack(">Q", payload[24:32])[0]
                else:
                    ts = struct.unpack(">I", payload[12:16])[0]
                    dur = struct.unpack(">I", payload[16:20])[0]
                if ts:
                    track_duration = dur / ts

        codec = ""
        stsd = _find_box(data, (b"mdia", b"minf", b"stbl", b"stsd"), trak_start, trak_end)
        if stsd:
            s, e = stsd
            payload = data[s:e]
            if len(payload) >= 16:
                codec = payload[12:16].decode("ascii", "replace").strip()

        if handler == "vide":
            if width and height and not (meta.width and meta.height):
                meta.width, meta.height = width, height
            if codec and not meta.video_codec:
                meta.video_codec = codec
        elif handler == "soun":
            meta.has_audio = True
            if codec and not meta.audio_codec:
                meta.audio_codec = codec
        if track_duration:
            duration_seconds_by_track.append((handler or "?", track_duration))

    if duration_seconds_by_track:
        meta.extra["track_durations"] = {
            h: round(d, 3) for h, d in duration_seconds_by_track
        }
    if meta.duration <= 0 and duration_seconds_by_track:
        meta.duration = max(d for _, d in duration_seconds_by_track)

    # A container with no tracks and no duration is not a usable video; the
    # caller should report the file as unparseable rather than describing a
    # zero-by-zero "video".
    if not tracks_seen and not meta.duration:
        return None
    if not meta.width and not meta.height and not meta.duration and not meta.has_audio:
        return None
    return meta


# --------------------------------------------------------------------------
# Matroska / WebM (EBML)
# --------------------------------------------------------------------------

def _read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int]:
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not (first & mask):
        length += 1
        mask >>= 1
    if length > 8:
        return 0, 0
    value = first if keep_marker else (first & (mask - 1))
    for i in range(1, length):
        value = (value << 8) | data[pos + i]
    return value, length


def _ebml_children(data: bytes, start: int, end: int):
    pos = start
    while pos < end:
        elem_id, id_len = _read_vint(data, pos, keep_marker=True)
        if not id_len:
            return
        pos += id_len
        size, size_len = _read_vint(data, pos, keep_marker=False)
        if not size_len:
            return
        pos += size_len
        unknown = size == (1 << (7 * size_len)) - 1
        if unknown:
            return
        payload_start = pos
        payload_end = min(pos + size, end)
        yield elem_id, payload_start, payload_end
        pos = payload_end


def _uint(data: bytes, start: int, end: int) -> int:
    return int.from_bytes(data[start:end], "big") if end > start else 0


def _float(data: bytes, start: int, end: int) -> float:
    raw = data[start:end]
    if len(raw) == 4:
        return struct.unpack(">f", raw)[0]
    if len(raw) == 8:
        return struct.unpack(">d", raw)[0]
    return 0.0


def parse_matroska(path: Path) -> VideoMetadata | None:
    """Parse duration/resolution from a Matroska or WebM container."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_HEADER_BYTES)
    except OSError:
        return None
    if len(data) < 4 or data[0:4] != b"\x1a\x45\xdf\xa3":
        return None

    meta = VideoMetadata(container="matroska/webm")
    timecode_scale = 1_000_000  # ns per tick, Matroska default
    duration_ticks = 0.0

    segment = None
    for elem_id, start, end in _ebml_children(data, 0, len(data)):
        if elem_id == 0x18538067:  # Segment
            segment = (start, end)
            break
    if not segment:
        return None

    for elem_id, start, end in _ebml_children(data, segment[0], segment[1]):
        if elem_id == 0x1549A966:  # Info
            for sub_id, sub_start, sub_end in _ebml_children(data, start, end):
                if sub_id == 0x2AD7B1:  # TimecodeScale
                    timecode_scale = _uint(data, sub_start, sub_end) or 1_000_000
                elif sub_id == 0x4489:  # Duration (float, in timecode ticks)
                    duration_ticks = _float(data, sub_start, sub_end)
        elif elem_id == 0x1654AE6B:  # Tracks
            for sub_id, sub_start, sub_end in _ebml_children(data, start, end):
                if sub_id == 0xAE:  # TrackEntry
                    track_type = 0
                    codec = ""
                    width = height = 0
                    for t_id, t_start, t_end in _ebml_children(data, sub_start, sub_end):
                        if t_id == 0x83:
                            track_type = _uint(data, t_start, t_end)
                        elif t_id == 0x86:
                            codec = data[t_start:t_end].decode("ascii", "replace").rstrip("\x00")
                        elif t_id == 0xE0:  # Video
                            for v_id, v_start, v_end in _ebml_children(data, t_start, t_end):
                                if v_id == 0xB0:
                                    width = _uint(data, v_start, v_end)
                                elif v_id == 0xBA:
                                    height = _uint(data, v_start, v_end)
                    if track_type == 1:  # video
                        if width and not meta.width:
                            meta.width, meta.height = width, height
                        if codec and not meta.video_codec:
                            meta.video_codec = codec
                    elif track_type == 2:  # audio
                        meta.has_audio = True
                        if codec and not meta.audio_codec:
                            meta.audio_codec = codec

    if duration_ticks:
        meta.duration = duration_ticks * timecode_scale / 1_000_000_000
    meta.extra["timecode_scale_ns"] = timecode_scale
    return meta


# --------------------------------------------------------------------------
# RIFF / AVI
# --------------------------------------------------------------------------

def parse_avi(path: Path) -> VideoMetadata | None:
    """Parse duration/resolution/fps from a RIFF AVI container."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_HEADER_BYTES)
    except OSError:
        return None
    if len(data) < 12 or data[0:4] != b"RIFF":
        return None

    meta = VideoMetadata(container="avi")
    try:
        riff_type = data[8:12]
    except Exception:  # pragma: no cover - defensive
        return None
    if riff_type not in (b"AVI ", b"AVIX"):
        return None

    # Walk RIFF LIST chunks looking for 'avih'.
    def walk(start: int, end: int) -> None:
        pos = start
        while pos + 8 <= end:
            chunk_id = data[pos : pos + 4]
            size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
            body = pos + 8
            body_end = min(body + size, end)
            if chunk_id == b"LIST":
                if data[body : body + 4] == b"hdrl" and meta.frame_count == 0:
                    walk(body + 4, body_end)
            elif chunk_id == b"avih" and body_end - body >= 40:
                usec_per_frame, _, _, total_frames = struct.unpack("<IIII", data[body : body + 16])
                width, height = struct.unpack("<II", data[body + 32 : body + 40])
                meta.width, meta.height = width, height
                meta.frame_count = int(total_frames)
                if usec_per_frame:
                    meta.fps = round(1_000_000.0 / usec_per_frame, 3)
                    meta.duration = meta.frame_count / meta.fps
            pos = body + size + (size & 1)

    walk(12, len(data))
    return meta


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------

class VideoExtractor(Extractor):
    """Extract genuine video metadata, then optional frames and transcript."""

    name = "video"
    extensions = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("reading video container", 0.15)
        warnings: list[str] = []
        unavailable: dict[str, str] = {}
        blocks: list[ExtractedBlock] = []
        artifacts: dict = {}

        meta = parse_mp4(path) or parse_matroska(path) or parse_avi(path)
        if meta is None:
            # No readable container and therefore no content at all: this is a
            # genuine failure, not a degraded success.
            raise ExtractionError(
                "The video container could not be parsed. The file may be "
                "corrupt, incompletely uploaded, or use a container MSN AI does "
                "not understand yet."
            )

        data = meta.to_dict()

        summary = (
            f"Video file: {data.get('resolution') or 'unknown resolution'}, "
            f"duration {data.get('duration_human')} ({meta.duration:.2f}s), "
            f"container {meta.container}, video codec {meta.video_codec or 'unknown'}"
            + (f", audio codec {meta.audio_codec}" if meta.has_audio else ", no audio track")
            + (f", {meta.fps} fps" if meta.fps else "")
            + "."
        )
        blocks.append(
            ExtractedBlock(
                text=summary,
                kind=SourceKind.VIDEO,
                locator="video properties",
                order=0,
                metadata=data,
            )
        )

        # --- transcript (speech provider) ---
        transcript_blocks: list[ExtractedBlock] = []
        if ctx.speech is not None and ctx.speech.available and meta.has_audio:
            ctx.progress("transcribing audio", 0.3)
            try:
                segments = ctx.speech.transcribe(str(path))
                for index, segment in enumerate(segments):
                    text = (getattr(segment, "text", "") or "").strip()
                    if not text:
                        continue
                    start = float(getattr(segment, "start", 0.0) or 0.0)
                    end = float(getattr(segment, "end", start) or start)
                    transcript_blocks.append(
                        ExtractedBlock(
                            text=text,
                            kind=SourceKind.TRANSCRIPT,
                            locator=f"{format_timestamp(start)}-{format_timestamp(end)}",
                            order=index,
                            metadata={"start": start, "end": end, "t": start},
                        )
                    )
                if transcript_blocks:
                    data["transcript_segments"] = len(transcript_blocks)
                    data["transcript_human"] = format_timestamp(transcript_blocks[-1].metadata.get("end", 0.0))
            except Exception as exc:
                warnings.append(f"Transcription failed: {exc}")
                unavailable["speech"] = str(exc)
        else:
            reason = (
                ctx.speech.unavailable_reason
                if ctx.speech is not None
                else "No speech provider is configured."
            )
            if meta.has_audio:
                unavailable["speech"] = reason
                warnings.append(
                    "No local speech-to-text model is installed, so the audio track "
                    "was not transcribed."
                )
        if transcript_blocks:
            blocks.extend(transcript_blocks)

        # --- frames (ffmpeg) ---
        frame_count = 0
        ffmpeg = ctx.ffmpeg
        if ffmpeg is not None and ffmpeg.available:
            ctx.progress("sampling frames", 0.55)
            try:
                out_dir = ctx.settings.video_frames_path() / ctx.file_record.id
                frames = ffmpeg.sample_frames(
                    str(path),
                    str(out_dir),
                    duration=meta.duration,
                    interval=float(getattr(ctx.settings, "video_frame_interval", 5.0)),
                    max_frames=int(getattr(ctx.settings, "video_max_frames", 120)),
                    scene_threshold=float(getattr(ctx.settings, "video_scene_threshold", 0.35)),
                )
                frame_count = len(frames)
                if frame_count:
                    data["frames_sampled"] = frame_count
                    artifacts["video_frames_dir"] = ctx.file_record.id
                    for index, frame in enumerate(frames):
                        note = ""
                        if ctx.ocr is not None and ctx.ocr.available:
                            try:
                                ocr = ctx.ocr.image_to_text(
                                    str(frame["path"]),
                                    source=f"{ctx.file_record.filename} {frame['timestamp_human']}",
                                )
                                note = (ocr.text or "").strip()
                            except Exception:
                                note = ""
                        if ctx.vision is not None and ctx.vision.available:
                            try:
                                described = ctx.vision.describe(str(frame["path"]))
                                if described.text:
                                    note = (note + "\n" + described.text.strip()).strip()
                            except Exception:
                                pass
                        body = f"Frame at {frame['timestamp_human']}."
                        if note:
                            body += "\n" + note
                        timestamp = frame.get("timestamp")
                        locator = (
                            frame["timestamp_human"]
                            if frame.get("timestamp_known", timestamp is not None)
                            else f"frame {frame['index'] + 1}"
                        )
                        blocks.append(
                            ExtractedBlock(
                                text=self.block_text_limit(ctx, body),
                                kind=SourceKind.FRAME,
                                locator=locator,
                                order=len(blocks),
                                metadata={
                                    "t": timestamp,
                                    "frame": frame["index"],
                                    "path": str(Path(frame["path"]).name),
                                    "ocr_or_vision": bool(note),
                                },
                            )
                        )
            except Exception as exc:
                warnings.append(f"Frame extraction failed: {exc}")
                unavailable["ffmpeg"] = str(exc)
        else:
            reason = ffmpeg.unavailable_reason if ffmpeg is not None else "ffmpeg is not installed."
            if meta.duration > 0:
                unavailable["video_frames"] = reason
                warnings.append(
                    "FFmpeg is not available, so no frames were sampled; MSN AI can "
                    "only report this video's container metadata."
                )

        if not transcript_blocks and frame_count == 0 and not meta.has_audio:
            warnings.append(
                "This video has no audio and no frames were sampled, so only "
                "container metadata is available."
            )

        full_text = "\n\n".join(b.text for b in blocks)
        return ExtractionResult(
            text=self.block_text_limit(ctx, full_text),
            blocks=blocks,
            document_metadata=data,
            unavailable=unavailable,
            warnings=warnings,
            artifacts=artifacts,
        )
