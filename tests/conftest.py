"""MSN AI automated test suite.

Tests always run against the mock provider so they never require a model
download or the llama.cpp runtime to be present.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MSN_AI_PROVIDER"] = "mock"
os.environ["MSN_AI_LOG_LEVEL"] = "WARNING"
os.environ["MSN_AI_SESSION_PERSIST"] = "false"
os.environ["MSN_AI_MEMORY_DIR"] = "data/memory/test"

# Force a fresh Settings next time it is requested.
from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

# --- v0.2 ingestion fixtures ---------------------------------------------
# Every ingestion fixture is scoped to a temporary directory so the suite never
# touches the real data/ tree, and all of it runs against the zero-download
# hashing embedder so no model is required.

from app.core.config import Settings  # noqa: E402
from app.ingestion.embeddings import build_embedding_provider  # noqa: E402
from app.ingestion.knowledge import KnowledgeBase  # noqa: E402
from app.ingestion.service import IngestionService  # noqa: E402
from app.ingestion.store import FileStore  # noqa: E402
from app.utils.paths import ensure_ingestion_directories  # noqa: E402

#: Small, realistic fixtures reused across the extraction tests.
SAMPLE_NOTE = b"""Launch Checklist

Revenue in Q3 reached 4.2 million dollars, mostly from the Berlin office.

Risks
The vendor contract expires in December and must be renegotiated before then.
The Chicago team shipped the new ingestion pipeline last month.
"""

SAMPLE_JSON = b"""{
  "service": "msn-ai",
  "port": 8000,
  "features": ["chat", "files", "video"],
  "limits": {"max_upload_mb": 100, "max_video_upload_mb": 512}
}
"""

SAMPLE_CSV = b"name,role,team\nAda,engineer,platform\nGrace,architect,platform\nLinus,maintainer,core\n"


@pytest.fixture()
def ingest_settings(tmp_path: Path) -> Settings:
    """Isolated settings rooted in ``tmp_path``.

    ``ingestion_enabled=False`` keeps the background worker out of the way so
    tests that call ``process()`` directly are deterministic; the background
    queue has its own dedicated test with it switched on.
    """
    return Settings(
        data_dir=tmp_path / "data",
        model_storage_dir=PROJECT_ROOT / "model_storage",
        log_level="WARNING",
        provider="mock",
        model_name="mock-msn-ai-v0.1",
        session_persist=False,
        ingestion_enabled=False,
        embedding_provider="hashing",
        embedding_dim=384,
    )


@pytest.fixture()
def settings(ingest_settings: Settings) -> Settings:
    """Alias with the ingestion directory tree already created."""
    ensure_ingestion_directories(ingest_settings)
    ingest_settings.memory_path().mkdir(parents=True, exist_ok=True)
    return ingest_settings


@pytest.fixture()
def store(settings: Settings) -> FileStore:
    return FileStore(settings)


@pytest.fixture()
def embedder(settings: Settings):
    return build_embedding_provider(settings)


@pytest.fixture()
def knowledge(settings: Settings, embedder) -> KnowledgeBase:
    return KnowledgeBase(settings, embedder)


@pytest.fixture()
def service(settings: Settings, embedder, store, knowledge) -> IngestionService:
    return IngestionService(settings, embedder, store, knowledge)


@pytest.fixture()
def upload(service: IngestionService):
    """Upload a payload and run extraction synchronously."""

    def _upload(
        payload: bytes = SAMPLE_NOTE,
        filename: str = "launch.txt",
        session_id: str = "s1",
        **kwargs,
    ):
        record = service.save_upload(
            io.BytesIO(payload), filename, session_id=session_id, **kwargs
        )
        return service.process(record.id)

    return _upload