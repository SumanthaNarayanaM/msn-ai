# MSN AI v0.2

A fully local AI platform with file ingestion, local knowledge/RAG, and citations.
v0.2 ships a working FastAPI backend with:

- **Real local model** via `llamacpp` (llama.cpp) — fully offline, CPU-only, no paid APIs
- **Local file ingestion** — upload documents, images, video; MSN AI reads them locally
- **Local knowledge/RAG** — deterministic hashing embeddings + SQLite FTS5; no external vector DB
- **Citations** — every answer backed by file excerpts with source links
- **Remember/Forget** — persist knowledge across sessions or drop it instantly

## Platform roadmap

| Capability      | Status in v0.2 |
|-----------------|----------------|
| MSN AI Chat     | Real local model (Qwen2.5-1.5B-Instruct Q4_K_M) |
| MSN AI Files    | Upload, extract, index, search, cite — **Live** |
| MSN AI Image    | OCR ready (needs Tesseract binary) |
| MSN AI Video    | Frame sampling + speech-to-text ready (needs FFmpeg + faster-whisper) |
| MSN AI Vision   | Multimodal GGUF support wired (needs local mmproj model) |
| MSN AI Agent    | Not implemented |
| MSN AI Tools    | Not implemented |

## Architecture

```
 User
   |
   v
 MSN AI API (FastAPI)
   |
   +-- Chat / Sessions        (app/core/session.py)
   +-- Files / Ingestion      (app/ingestion/)
   |     |-- Extractors       (pdf, docx, xlsx, pptx, txt, md, csv, html, zip, image, video)
   |     |-- Chunking         (citation-preserving, block-aware)
   |     |-- Embeddings       (deterministic local-hashing-v1, 384-dim)
   |     |-- Knowledge Store  (SQLite FTS5 + vector index)
   |     \-- Providers        (OCR, vision, speech, ffmpeg — all optional, no auto-download)
   |
   \-- ModelProvider          (app/inference/base.py)
         |-- MockLocalModelProvider
         \-- RealLocalModelProvider (llama.cpp GGUF)
```

Everything above `ModelProvider` depends only on that interface, so the mock
can always be swapped back in without touching API or session code.

The default local model is **Qwen2.5-1.5B-Instruct** (Q4_K_M). Do not claim it
is "trained by MSN AI" — it is an existing openly available local model
([Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)) being
used as MSN AI's initial language engine.

## Requirements

- Python >= 3.10 (tested on 3.13), Windows PowerShell
- 16 GB RAM (comfortable); 8 GB is enough for the 1.5B model
- ~1.2 GB free disk for the model file
- No GPU required — runs on CPU (Intel i5-1135G7 tested)
- Optional: Tesseract OCR, FFmpeg, faster-whisper, multimodal GGUF for extended capabilities

## Installation

```powershell
Set-Location "D:\MSN AI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the environment template and edit if needed:

```powershell
Copy-Item .env.example .env
```

## Model download

The model ships as a single ~1.1 GB GGUF file and is stored in
`model_storage/base` (on D:, not C:):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model.ps1
```

This downloads `qwen2.5-1.5b-instruct-q4_k_m.gguf` from Hugging Face into
`D:\MSN AI\model_storage\base`. Re-run to resume an interrupted download.

Install the real inference runtime (llama.cpp Python bindings):

```powershell
pip install -r requirements-local.txt
```

## How to select the provider

Edit `.env`:

```ini
# Real local model:
MSN_AI_PROVIDER=llamacpp
MSN_AI_MODEL_NAME=qwen2.5-1.5b-instruct

# Optional overrides:
# MSN_AI_MODEL_PATH=D:/MSN AI/model_storage/base/qwen2.5-1.5b-instruct-q4_k_m.gguf
# MSN_AI_N_CTX=2048
# MSN_AI_N_THREADS=0        (0 == auto)
```

To switch back to mock mode (no model needed):

```ini
MSN_AI_PROVIDER=mock
MSN_AI_MODEL_NAME=mock-msn-ai-v0.2
```

`MSN_AI_PROVIDER=local` is an alias for `llamacpp`.

## Optional capabilities (no auto-download)

| Capability | What it enables | Requirements |
|------------|-----------------|--------------|
| OCR | Text extraction from images, scanned PDFs, video frames | Install Tesseract binary (`tesseract` on PATH) |
| Vision | Image understanding (describe, OCR++) | Place multimodal GGUF + mmproj in `model_storage/vision`; set `MSN_AI_VISION_MODEL_PATH` |
| Speech | Video/audio transcription | `pip install faster-whisper` (base model downloads on first use) |
| Video frames | Sample frames, extract audio | Install FFmpeg (`ffmpeg` + `ffprobe` on PATH) |
| Embeddings | Semantic search | Built-in `local-hashing-v1` (always available, no download) |

**IMPORTANT:** MSN AI never downloads models or binaries automatically. Every optional capability requires explicit user action (install binary, place model file, install Python package). If a capability is unavailable, the backend reports `available: false` with a clear `reason`.

## Start MSN AI

```powershell
python -m app.main
# or: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On first start, the real provider loads the GGUF into memory (a few seconds;
roughly 1-2 GB RAM). Watch the log line `Model ready in ...s`.

The web UI is served at http://127.0.0.1:3000 (static files in `web/`).
API docs: http://127.0.0.1:8000/docs

## Hardware expectations

| Hardware           | Expected experience                             |
|--------------------|-------------------------------------------------|
| i5-1135G7 (4C/8T)  | ~8-12 tokens/s — fine for friendly chat        |
| 16 GB RAM          | comfortable (model uses ~2 GB)                 |
| 8 GB RAM           | usable; reduce `MSN_AI_N_CTX` to 1024         |
| GPU (Iris Xe etc.) | optional; llama.cpp runs on CPU by default     |

## File ingestion — Quick start

```powershell
# 1. Health check (shows ingestion capabilities)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/ingestion/health"

# 2. Upload a file (wait=true blocks until extraction finishes)
$form = @{ file = Get-Item ".\my_notes.txt"; wait = "true" }
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/files/upload" -Method Post -Form $form

# 3. Remember it across sessions
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/files/$($resp.id)/remember" -Method Post -ContentType "application/json" -Body '{"scope": "personal"}'

# 4. Ask a question — MSN AI will cite the file
$body = '{"message": "Summarize my notes.", "use_knowledge": true}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat" -Method Post -ContentType "application/json" -Body $body
```

**Supported file types:** `.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.json`, `.xml`, `.xsd`, `.csv`, `.tsv`, `.html`, `.htm`, `.xhtml`, `.atom`, `.rss`, `.pdf`, `.docx`, `.xlsx`, `.xlsm`, `.pptx`, `.zip`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.svg`, `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`

**Upload limits:** 100 MB per file (512 MB for video/archives). Files larger than the limit are rejected.

**Lifecycle:** `uploaded` → `processing` → `ready` | `failed`. Failed files show `error` and `warnings`; nothing is hidden.

**Memory scopes:**
- `session` — file indexed only for the current chat session (default)
- `personal` — file remembered across all your sessions
- `project` — reserved for future multi-user projects

`Remember` (scope: `personal`) keeps the file in the knowledge base forever.
`Forget` removes indexed chunks but **keeps the original file on disk**.
`Delete` removes the file and all indexed data.

**Uploaded files are NOT used to retrain or fine-tune Qwen.** They are indexed into the local knowledge store (SQLite + deterministic hashing embeddings) and retrieved only when relevant to a query. The model weights never change.

## API surface (v1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info banner |
| GET | `/api/v1/health` | Health + active provider/model |
| GET | `/api/v1/ingestion/health` | Ingestion capabilities + file/knowledge stats |
| POST | `/api/v1/chat` | Send a message (creates/reuses session) |
| GET | `/api/v1/sessions` | List active sessions |
| GET | `/api/v1/sessions/{id}` | Session history |
| DELETE | `/api/v1/sessions/{id}` | Delete a session |
| POST | `/api/v1/files/upload` | Upload a file (multipart) |
| GET | `/api/v1/files` | List files (optionally filtered by session) |
| GET | `/api/v1/files/{id}` | File record |
| GET | `/api/v1/files/{id}/content` | Download original file |
| GET | `/api/v1/files/{id}/thumbnail` | Thumbnail (images/video) |
| GET | `/api/v1/files/{id}/extracted` | Full extracted text |
| POST | `/api/v1/files/{id}/analyze` | Re-run extraction |
| POST | `/api/v1/files/{id}/remember` | Persist in knowledge base (`scope: personal|project`) |
| POST | `/api/v1/files/{id}/forget` | Remove from knowledge base, keep file |
| DELETE | `/api/v1/files/{id}` | Delete file + indexed data |
| POST | `/api/v1/knowledge/search` | Raw knowledge search |

Interactive docs: http://127.0.0.1:8000/docs

## Chat payload (v0.2 additive fields)

```json
{
  "message": "string",
  "session_id": "optional string",
  "attachments": ["file_id_1", "file_id_2"],
  "use_knowledge": true
}
```

Response includes:
```json
{
  "reply": "string",
  "sources": [
    { "file_id": "...", "filename": "...", "locator": "lines 1-3", "kind": "document", "text": "..." }
  ],
  "used_knowledge": true
}
```

## Configuration

All settings come from environment variables with the `MSN_AI_` prefix,
loaded from `.env` (see `.env.example`).

| Variable | Default | Meaning |
|----------|---------|---------|
| `MSN_AI_HOST` | `127.0.0.1` | Bind address |
| `MSN_AI_PORT` | `8000` | Bind port |
| `MSN_AI_PROVIDER` | `mock` | `mock` / `llamacpp` / `local` |
| `MSN_AI_MODEL_NAME` | `mock-msn-ai-v0.2` | Model identifier |
| `MSN_AI_MAX_TOKENS` | `1024` | Generation budget |
| `MSN_AI_TEMPERATURE` | `0.7` | Sampling temperature |
| `MSN_AI_MODEL_PATH` | (derived) | ABSOLUTE path to the .gguf file |
| `MSN_AI_N_CTX` | `2048` | Context window for llama.cpp |
| `MSN_AI_N_THREADS` | `0` | 0 == auto |
| `MSN_AI_MODEL_HF_REPO` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Download source |
| `MSN_AI_MODEL_HF_FILE` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Download file |
| `MSN_AI_VISION_MODEL_PATH` | (none) | Path to multimodal GGUF + mmproj |
| `MSN_AI_LOG_LEVEL` | `INFO` | Logging level |
| `MSN_AI_VERSION` | `0.2.0` | Application version |

## Tests

```powershell
python -m pytest -q
```

Tests always run against the mock provider (`tests/conftest.py` forces it), so
they never need the model download or the llama.cpp runtime.

## Project layout

```
D:\MSN AI\
├── app\               Python application
│   ├── api\           FastAPI routers / endpoints / dependencies
│   ├── core\          settings, logging, errors, session manager
│   ├── inference\     ModelProvider abstraction + providers (mock, local_llm)
│   ├── ingestion\     File ingestion, extraction, knowledge, providers
│   ├── models\        Pydantic schemas
│   ├── tools\         reserved (MSN AI Tools)
│   └── utils\         path helpers
├── data\              conversations, knowledge index (SQLite), generated artifacts
├── model_storage\     base (GGUF), vision (mmproj), checkpoints, cache
├── training\          reserved
├── tests\             pytest suite
├── scripts\           fetch_model.ps1
├── config\            configuration files
├── docs\              documentation
└── web\               Static frontend (ES modules, no build step)
```

## Remaining work to connect the real local coding model

Finding, wiring and running a friendly local chat model is done in v0.1.
Future work for MSN AI Code:

1. Pick a larger/quantized coding LLM and drop its GGUF into `model_storage/base`
   (or point `MSN_AI_MODEL_PATH` at it).
2. That is it — `RealLocalModelProvider` already works with any GGUF that
   stores a chat template; just change `MSN_AI_MODEL_PATH`/`MSN_AI_MODEL_NAME`.
3. Optionally add a code-specific system prompt and stop sequences in the
   provider later.