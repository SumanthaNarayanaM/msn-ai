"""Pydantic settings backed by .env support."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="MSN_AI_",
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True

    # Application
    app_name: str = "MSN AI"
    env: str = "development"
    debug: bool = True
    version: str = "0.2.0"
    log_level: str = "INFO"

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    model_storage_dir: Path = PROJECT_ROOT / "model_storage"
    conversations_dir: Path | None = None

    # Inference
    provider: str = "mock"
    model_name: str = "mock-msn-ai-v0.2"
    max_tokens: int = 1024
    temperature: float = 0.7
    request_timeout_seconds: float = 60.0

    # Conversation / memory
    session_persist: bool = True
    memory_dir: Path | None = None
    system_prompt: str = (
        "You are MSN AI, a friendly, natural and helpful local assistant. "
        "You are comfortable with casual conversation AND detailed technical "
        "help, switching between the two naturally. Be conversational and warm, "
        "not robotic. Do not constantly say 'As an AI...'. Be concise when a "
        "short answer is enough, and go into detail when the user asks for it. "
        "You are technically capable: you can write and explain code (Python, "
        "JavaScript, React and more), explain errors, and reason about problems. "
        "Never invent personal facts about the user; remember only what they "
        "explicitly share. Use the conversation history to stay consistent."
    )

    # Real local model (llama.cpp backend)
    model_path: Path | None = None
    n_ctx: int = 2048
    n_threads: int = 0  # 0 == auto (logical CPU count)
    model_hf_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    model_hf_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    # --- Uploads / ingestion (v0.2) ---
    # Every uploaded byte and every generated artifact stays under data_dir.
    uploads_dir: Path | None = None
    extracted_dir: Path | None = None
    knowledge_dir: Path | None = None
    embeddings_dir: Path | None = None
    thumbnails_dir: Path | None = None
    video_frames_dir: Path | None = None
    files_metadata_dir: Path | None = None

    #: Hard ceiling for a normal upload (MiB). Configurable via MSN_AI_MAX_UPLOAD_MB.
    max_upload_mb: int = 100
    #: Videos get a separate, larger ceiling (MSN_AI_MAX_VIDEO_UPLOAD_MB).
    max_video_upload_mb: int = 512
    #: Reject archives that would expand beyond this size (zip bomb guard).
    max_archive_expanded_mb: int = 512
    #: Number of files extracted from a single .zip (structure explanation).
    max_archive_entries: int = 400
    #: Max extracted characters kept per source block before truncation.
    max_block_chars: int = 200_000
    #: Files larger than this are truncated with an explicit marker.
    max_text_chars: int = 2_000_000
    #: Enable the background ingestion worker.
    ingestion_enabled: bool = True
    #: Maximum characters of retrieved context injected into a chat prompt.
    max_context_chars: int = 6000
    #: Chunks retrieved per chat turn.
    retrieval_top_k: int = 5
    #: Minimum cosine similarity for a chunk to be considered relevant.
    retrieval_min_score: float = 0.12

    # --- Chunking ---
    chunk_size: int = 1200
    chunk_overlap: int = 180

    # --- Embeddings ---
    #: "hashing" (built-in, zero download) or "sentence-transformers" (optional).
    embedding_provider: str = "hashing"
    embedding_model: str = "local-hashing-v1"
    embedding_dim: int = 384

    # --- Optional capability providers (never auto-downloaded) ---
    #: Local vision model (llava-style GGUF). Empty => vision reports unavailable.
    vision_model_path: Path | None = None
    #: Multimodal projector that must accompany the vision GGUF.
    vision_mmproj_path: Path | None = None
    vision_model_name: str = "none"
    vision_n_ctx: int = 4096
    vision_max_tokens: int = 512
    vision_n_gpu_layers: int = 0

    #: Local OCR. "tesseract" uses the Tesseract binary; "rapidocr" uses RapidOCR.
    ocr_provider: str = "tesseract"
    tesseract_cmd: str = "tesseract"
    ocr_languages: str = "eng"
    ocr_enabled: bool = True

    #: Local speech-to-text. "faster-whisper" or "whisper"; empty => unavailable.
    speech_provider: str = "faster-whisper"
    speech_model: str = "base"
    speech_compute_type: str = "int8"

    #: External ffmpeg/ffprobe binaries for frame + audio extraction.
    ffmpeg_cmd: str = "ffmpeg"
    ffprobe_cmd: str = "ffprobe"

    # --- Video ---
    #: Interval (seconds) between sampled frames when scene detection is off.
    video_frame_interval: float = 5.0
    #: Absolute cap on frames extracted from one video.
    video_max_frames: int = 120
    video_scene_threshold: float = 0.35

    @field_validator("conversations_dir", mode="before")
    @classmethod
    def _default_conversations_dir(cls, value: object) -> object:
        return value

    @field_validator("memory_dir", mode="before")
    @classmethod
    def _default_memory_dir(cls, value: object) -> object:
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()

    def conversations_path(self) -> Path:
        return self.conversations_dir or (self.data_dir / "conversations")

    def memory_path(self) -> Path:
        return self.memory_dir or (self.data_dir / "memory")

    # --- v0.2 ingestion paths -------------------------------------------
    # All ingestion output is confined to data_dir; nothing is written
    # outside the project root.

    def _sub(self, override: Path | None, name: str) -> Path:
        return override if override is not None else (self.data_dir / name)

    def uploads_path(self) -> Path:
        return self._sub(self.uploads_dir, "uploads")

    def extracted_path(self) -> Path:
        return self._sub(self.extracted_dir, "extracted")

    def knowledge_path(self) -> Path:
        return self._sub(self.knowledge_dir, "knowledge")

    def embeddings_path(self) -> Path:
        return self._sub(self.embeddings_dir, "embeddings")

    def thumbnails_path(self) -> Path:
        return self._sub(self.thumbnails_dir, "thumbnails")

    def video_frames_path(self) -> Path:
        return self._sub(self.video_frames_dir, "video_frames")

    def files_metadata_path(self) -> Path:
        return self._sub(self.files_metadata_dir, "metadata")

    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    def max_video_upload_bytes(self) -> int:
        return int(self.max_video_upload_mb) * 1024 * 1024

    def upload_limit_for(self, is_video: bool) -> int:
        """Return the byte ceiling that applies to a given upload kind."""
        return self.max_video_upload_bytes() if is_video else self.max_upload_bytes()

    def resolved_model_path(self) -> Path:
        """Explicit model path, or the default GGUF location under model_storage."""
        return self.model_path or (self.model_storage_dir / "base" / self.model_hf_file)

    def resolved_n_threads(self) -> int | None:
        """Return the thread count for llama.cpp (None means auto)."""
        return None if self.n_threads <= 0 else self.n_threads

    def ensure_directories(self) -> None:
        """Create all runtime directories that the application expects."""
        from app.utils.paths import ensure_standard_directories

        ensure_standard_directories(
            data_dir=self.data_dir,
            model_storage_dir=self.model_storage_dir,
            conversations_dir=self.conversations_path(),
        )
    def resolved_options(self) -> dict:
        """Flatten settings into a plain dict (useful for dependency injection)."""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "app_name": self.app_name,
            "env": self.env,
            "debug": self.debug,
            "version": self.version,
            "log_level": self.log_level,
            "data_dir": str(self.data_dir),
            "model_storage_dir": str(self.model_storage_dir),
            "conversations_dir": str(self.conversations_path()),
            "session_persist": self.session_persist,
            "memory_dir": str(self.memory_path()),
            "system_prompt": self.system_prompt,
            "provider": self.provider,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model_path": str(self.resolved_model_path()),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "model_hf_repo": self.model_hf_repo,
            "model_hf_file": self.model_hf_file,
        }


@lru_cache
def get_settings() -> Settings:
    """Return a cached, lazily-initialized Settings instance."""
    return Settings()


settings = get_settings()