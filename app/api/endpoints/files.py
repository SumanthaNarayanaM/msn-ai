"""File upload, management, memory-scope and knowledge-search endpoints."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import FileResponse as FastFileResponse
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_ingestion_service
from app.core.errors import FileNotFoundError_, ValidationError
from app.core.logging import get_logger
from app.ingestion.models import FileRecord, FileStatus
from app.ingestion.service import IngestionService
from app.models.schemas import (
    FileListResponse,
    FileResponse,
    KnowledgeSource,
    RememberRequest,
    SearchRequest,
    SearchResponse,
)

logger = get_logger("msn_ai.api.files")

router = APIRouter(prefix="/files", tags=["files"])

ServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def to_response(record: FileRecord) -> FileResponse:
    """Map a record onto the API schema, adding UI-friendly derived fields."""
    data = record.to_public_dict()
    return FileResponse(
        **data,
        has_thumbnail=bool(record.artifacts.get("thumbnail")),
        readable=record.status is not FileStatus.FAILED,
    )


@router.post(
    "/upload",
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file for ingestion",
)
async def upload_file(
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="The file to upload.")],
    session_id: Annotated[str, Form(description="Owning chat session.")] = "",
    scope: Annotated[str, Form(description="session | project | personal")] = "session",
    project: Annotated[str, Form(description="Project name for scope='project'.")] = "",
    wait: Annotated[
        bool,
        Form(
            description=(
                "Process inline and return the finished record instead of "
                "queueing. Convenient for small files and scripts; the response "
                "then already has status 'ready' or 'failed'."
            )
        ),
    ] = False,
) -> FileResponse:
    """Accept an upload, validate it, store it, and queue extraction.

    Returns immediately with ``status='uploaded'``; the client polls
    ``GET /files/{id}`` for progress until it reaches ``ready`` or ``failed``.
    Pass ``wait=true`` to process inline and get the final state in the reply.
    """
    if not file.filename:
        raise ValidationError("No filename was supplied with the upload.")
    # The stream is read in chunks with a hard byte ceiling, so the whole body
    # is never buffered in memory. That read is blocking, so it runs in a
    # worker thread rather than stalling the event loop.
    record = await run_in_threadpool(
        service.save_upload,
        file.file,
        file.filename,
        declared_mime=file.content_type,
        session_id=session_id or "",
        scope=scope or "session",
        project=project or "",
        # Inline processing and the background queue must not both handle the
        # same file, so ``wait`` picks exactly one of them.
        enqueue=not wait,
    )
    if wait:
        record = await run_in_threadpool(service.process, record.id)
    return to_response(record)


@router.get("", response_model=FileListResponse, summary="List uploaded files")
def list_files(
    service: ServiceDep,
    session_id: Optional[str] = Query(default=None),
    scope: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> FileListResponse:
    records = service.store.list(session_id=session_id, scope=scope, limit=limit)
    return FileListResponse(
        files=[to_response(r) for r in records], total=len(records)
    )


@router.get(
    "/{file_id}",
    response_model=FileResponse,
    summary="Get one file and its ingestion state",
    responses={404: {"description": "Unknown file id"}},
)
def get_file(service: ServiceDep, file_id: str) -> FileResponse:
    return to_response(service.store.get(file_id))


@router.get(
    "/{file_id}/content",
    summary="Download or preview the stored original",
    responses={404: {"description": "Unknown file id"}},
)
def get_file_content(service: ServiceDep, file_id: str):
    """Serve the stored bytes back. Always the id-keyed original, never a path."""
    record = service.store.get(file_id)
    path = service.store.storage_path(record)
    if not path.exists():
        raise FileNotFoundError_(
            f"The stored file for '{record.display_name}' is no longer on disk."
        )
    return FastFileResponse(
        path=str(path),
        media_type=record.mime_type,
        filename=record.display_name,
    )


@router.get(
    "/{file_id}/extracted",
    summary="Get the extracted plain text",
    responses={404: {"description": "Unknown file id"}},
)
def get_extracted_text(service: ServiceDep, file_id: str):
    """Return the text MSN AI actually extracted, for inspection/debugging."""
    record = service.store.get(file_id)
    path = service.store.extracted_path(record)
    if not path.exists():
        raise FileNotFoundError_(
            f"No extracted text for '{record.display_name}' yet; it may still be processing."
        )
    return {
        "id": record.id,
        "filename": record.display_name,
        "status": record.status.value,
        "char_count": record.char_count,
        "warnings": record.warnings,
        "unavailable": record.unavailable,
        "text": path.read_text(encoding="utf-8", errors="replace"),
    }


@router.get(
    "/{file_id}/thumbnail",
    summary="Get the generated preview image",
    responses={404: {"description": "No thumbnail, or unknown file id"}},
)
def get_thumbnail(service: ServiceDep, file_id: str):
    record = service.store.get(file_id)
    name = record.artifacts.get("thumbnail")
    if not name:
        raise FileNotFoundError_(
            f"No preview image for '{record.display_name}' (previews are generated for images)."
        )
    path = service.settings.thumbnails_path() / str(name)
    if not path.exists():
        raise FileNotFoundError_("The preview image is no longer on disk.")
    return FastFileResponse(path=str(path), media_type="image/jpeg")


@router.post(
    "/{file_id}/analyze",
    response_model=FileResponse,
    summary="Re-run extraction (after installing a capability)",
)
def analyze_file(service: ServiceDep, file_id: str) -> FileResponse:
    """Re-extract a file, e.g. after installing Tesseract or a vision model.

    Runs synchronously so the response reflects the new result; the frontend
    disables the button while it is in flight.
    """
    return to_response(service.reanalyze(file_id))


@router.post(
    "/{file_id}/remember",
    response_model=FileResponse,
    summary="Promote a file's knowledge to project or personal memory",
)
def remember_file(
    service: ServiceDep, file_id: str, body: RememberRequest
) -> FileResponse:
    return to_response(
        service.remember(file_id, scope=body.scope, project=body.project)
    )


@router.post(
    "/{file_id}/forget",
    response_model=FileResponse,
    summary="Remove indexed knowledge but keep the original file",
)
def forget_file(service: ServiceDep, file_id: str) -> FileResponse:
    return to_response(service.forget(file_id))


@router.delete(
    "/{file_id}",
    response_model=FileResponse,
    summary="Delete a file, its knowledge and its stored bytes",
)
def delete_file(service: ServiceDep, file_id: str) -> FileResponse:
    """Permanent delete. Use ``/forget`` to drop knowledge but keep the file."""
    return to_response(service.delete(file_id, remove_original=True))


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the uploaded knowledge base",
)
def search_knowledge(service: ServiceDep, body: SearchRequest) -> SearchResponse:
    """Explicit retrieval, used by the Files panel and by tests."""
    hits = service.search(
        body.query,
        scope=body.scope,
        session_id=body.session_id or "",
        project=body.project,
        top_k=body.top_k,
        file_ids=body.file_ids,
    )
    return SearchResponse(
        query=body.query,
        sources=[KnowledgeSource(**hit.to_dict()) for hit in hits],
        scope=body.scope,
        used_knowledge=bool(hits),
    )
