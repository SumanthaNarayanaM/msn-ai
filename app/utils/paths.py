"""Filesystem helpers and standard directory layout."""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger

logger = get_logger("msn_ai.utils.paths")

STANDARD_DIRS = (
    "data/datasets",
    "data/conversations",
    "data/generated/images",
    "data/generated/videos",
    "model_storage/base",
    "model_storage/checkpoints",
    "model_storage/cache",
    "training",
    "tests",
    "scripts",
    "config",
    "docs",
    "web",
)

#: v0.2 multimodal ingestion layout. Every uploaded byte and every generated
#: artifact lives under one of these directories, which are themselves under
#: ``data_dir`` -- nothing is ever written outside the project root.
INGESTION_DIRS = (
    "uploads",
    "extracted",
    "knowledge",
    "embeddings",
    "thumbnails",
    "video_frames",
    "metadata",
)


def ensure_ingestion_directories(settings) -> list[Path]:
    """Create the v0.2 ingestion directory tree from a Settings instance."""
    created: list[Path] = []
    for path in (
        settings.uploads_path(),
        settings.extracted_path(),
        settings.knowledge_path(),
        settings.embeddings_path(),
        settings.thumbnails_path(),
        settings.video_frames_path(),
        settings.files_metadata_path(),
    ):
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    logger.debug("Ensured %d ingestion directories.", len(created))
    return created


def ensure_standard_directories(
    data_dir: Path,
    model_storage_dir: Path,
    conversations_dir: Path,
) -> list[Path]:
    """Create the MSN AI directory layout, making parents as needed."""
    created: list[Path] = []

    def _mkdir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)

    _mkdir(data_dir)
    _mkdir(model_storage_dir)
    _mkdir(conversations_dir)
    _mkdir(data_dir / "datasets")
    _mkdir(data_dir / "generated" / "images")
    _mkdir(data_dir / "generated" / "videos")
    _mkdir(model_storage_dir / "base")
    _mkdir(model_storage_dir / "checkpoints")
    _mkdir(model_storage_dir / "cache")

    logger.debug("Ensured %d standard directories.", len(created))
    return created