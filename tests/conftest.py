"""MSN AI automated test suite.

Tests always run against the mock provider so they never require a model
download or the llama.cpp runtime to be present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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