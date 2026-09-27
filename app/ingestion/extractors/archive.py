"""ZIP archive extraction focused on explaining project structure.

The archive is **never** unpacked to disk. Members are listed, text members are
read in memory with strict caps, and every entry name is validated so a crafted
zip cannot escape the project or exhaust memory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from app.core.logging import get_logger
from app.ingestion.extractors.base import Extractor, ExtractionContext
from app.ingestion.models import ExtractedBlock, ExtractionResult, SourceKind

logger = get_logger("msn_ai.ingestion.archive")

#: Only these members are read as text.
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".py", ".js", ".jsx", ".ts", ".tsx", ".json",
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".env.example", ".csv",
    ".xml", ".html", ".htm", ".css", ".scss", ".sh", ".bat", ".ps1", ".sql",
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".rb", ".php", ".swift",
    ".kt", ".gradle", ".gitignore", ".dockerfile", ".makefile", ".lock",
}
#: Per-member read cap.
MAX_MEMBER_BYTES = 512 * 1024
#: Total bytes read from text members.
MAX_TOTAL_MEMBER_BYTES = 4 * 1024 * 1024
#: Source files that best reveal a project's purpose.
_PRIORITY_NAMES = ("readme", "license", "package.json", "pyproject.toml", "setup.py", "requirements.txt")


def _is_text_member(name: str) -> bool:
    suffix = PurePosixPath(name).suffix.lower()
    base = PurePosixPath(name).name.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    return base in _PRIORITY_NAMES or base.startswith("readme")


class ZipExtractor(Extractor):
    """Describe an archive's structure and index its readable text members."""

    name = "zip"
    extensions = frozenset({".zip"})

    def extract(self, path: Path, ctx: ExtractionContext) -> ExtractionResult:
        ctx.progress("opening archive", 0.2)
        warnings: list[str] = []
        max_entries = int(getattr(ctx.settings, "max_archive_entries", 400))
        max_expanded = int(getattr(ctx.settings, "max_archive_expanded_mb", 512)) * 1024 * 1024

        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            return ExtractionResult(
                text="",
                warnings=[f"The ZIP archive is malformed: {exc}"],
                unavailable={"zip": str(exc)},
            )

        blocks: list[ExtractedBlock] = []
        artifacts: dict = {}
        over_expansion_limit = False
        try:
            infos = archive.infolist()
            declared_total = sum(i.file_size for i in infos)
            if declared_total > max_expanded:
                over_expansion_limit = True
                warnings.append(
                    f"Archive expands to {declared_total / 1_048_576:.0f} MB which is over "
                    f"the {max_expanded / 1_048_576:.0f} MB limit; only the file list was "
                    "indexed and no file contents were read."
                )

            entries = [i for i in infos if not i.is_dir()]
            unsafe: list[str] = []
            for info in entries:
                name = info.filename
                normalised = name.replace("\\", "/")
                if normalised.startswith("/") or ".." in PurePosixPath(normalised).parts or ":" in normalised[:3]:
                    unsafe.append(name)
            if unsafe:
                warnings.append(
                    f"{len(unsafe)} archive entries contain absolute or traversal paths "
                    "and were skipped."
                )

            safe_entries = [i for i in entries if i.filename not in set(unsafe)]
            shown = safe_entries[:max_entries]
            if len(safe_entries) > max_entries:
                warnings.append(
                    f"Listing truncated to the first {max_entries} of {len(safe_entries)} files."
                )

            structure_lines = [f"Archive '{ctx.file_record.filename}' contains "
                               f"{len(safe_entries)} files:"]
            by_folder: dict[str, int] = {}
            for info in shown:
                folder = PurePosixPath(info.filename.replace("\\", "/")).parent.as_posix()
                folder = "." if folder in ("", ".") else folder
                by_folder[folder] = by_folder.get(folder, 0) + 1
                structure_lines.append(f"  {info.filename} ({info.file_size} bytes)")

            folder_summary = "\n".join(
                f"{folder}: {count} file(s)" for folder, count in sorted(by_folder.items())
            )
            structure_text = (
                "Project structure\n" + folder_summary + "\n\nFile listing\n" + "\n".join(structure_lines)
            )
            blocks.append(
                ExtractedBlock(
                    text=self.block_text_limit(ctx, structure_text),
                    kind=SourceKind.ARCHIVE,
                    locator="archive structure",
                    order=0,
                    metadata={"file_count": len(safe_entries)},
                )
            )

            # Read the most informative text members. When the declared
            # expansion is over the limit nothing is read at all, so a
            # compression bomb cannot make us do the work we just refused.
            ctx.progress("reading archive text files", 0.5)
            budget = MAX_TOTAL_MEMBER_BYTES
            candidates = [] if over_expansion_limit else sorted(
                shown,
                key=lambda i: (0 if PurePosixPath(i.filename).name.lower().startswith("readme") else 1, i.file_size),
            )
            for info in candidates:
                if budget <= 0 or not _is_text_member(info.filename):
                    continue
                if info.file_size > MAX_MEMBER_BYTES:
                    warnings.append(
                        f"Skipped '{info.filename}' ({info.file_size} bytes); above the "
                        f"{MAX_MEMBER_BYTES // 1024} KB per-file read limit."
                    )
                    continue
                try:
                    with archive.open(info) as handle:
                        data = handle.read(MAX_MEMBER_BYTES)
                except Exception as exc:
                    warnings.append(f"Could not read '{info.filename}': {exc}")
                    continue
                budget -= len(data)
                text = data.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                blocks.append(
                    ExtractedBlock(
                        text=self.block_text_limit(ctx, text),
                        kind=SourceKind.ARCHIVE,
                        locator=info.filename,
                        order=len(blocks),
                        metadata={"member": info.filename, "size": info.file_size},
                    )
                )

            full_text = "\n\n".join(b.text for b in blocks)
            return ExtractionResult(
                text=self.block_text_limit(ctx, full_text),
                blocks=blocks,
                document_metadata={
                    "file_count": len(safe_entries),
                    "folder_count": len(by_folder),
                    "folders": sorted(by_folder),
                    "declared_expanded_bytes": declared_total,
                    "skipped_unsafe_entries": len(unsafe),
                },
                warnings=warnings,
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.warning("ZIP extraction failed for %s: %s", path.name, exc)
            return ExtractionResult(
                text="",
                warnings=[f"The ZIP archive could not be read: {exc}"],
                unavailable={"zip": str(exc)},
            )
        finally:
            try:
                archive.close()
            except Exception:  # pragma: no cover - defensive
                pass
