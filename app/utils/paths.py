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