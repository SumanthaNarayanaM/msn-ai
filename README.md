# MSN AI v0.1

A fully local AI platform. v0.1 ships a working FastAPI backend with two model
providers:

- a **mock** provider so the whole stack runs even before any model exists, and
- a **real local provider** (`llamacpp`) that serves a small GGUF chat model
  via the mature [llama.cpp](https://github.com/ggerganov/llama.cpp) runtime —
  fully offline, CPU-only, no paid APIs.

## Platform roadmap

| Capability      | Status in v0.1 |
|-----------------|----------------|
| MSN AI Code     | Real local model wired (code features later) |
| MSN AI Image    | Not implemented (directories reserved) |
| MSN AI Video    | Not implemented (directories reserved) |
| MSN AI Agent    | Not implemented |
| MSN AI RAG      | Not implemented |
| MSN AI Tools    | Not implemented |

## Architecture

```
 User
  |
  v
 MSN AI API (FastAPI)
  |
  v
 Conversation Manager (app/core/session.py)
  |
  v
 ModelProvider (app/inference/base.py)
  |-- MockLocalModelProvider (app/inference/providers/mock.py)
  \-- RealLocalModelProvider (app/inference/providers/local_llm.py)
                   |
                   v
               llama.cpp (local GGUF model ~1.1 GB)
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
MSN_AI_MODEL_NAME=mock-msn-ai-v0.1
```

`MSN_AI_PROVIDER=local` is an alias for `llamacpp`.

## Start MSN AI

```powershell
python -m app.main
# or: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On first start, the real provider loads the GGUF into memory (a few seconds;
roughly 1-2 GB RAM). Watch the log line `Model ready in ...s`.

## Hardware expectations

| Hardware           | Expected experience                             |
|--------------------|-------------------------------------------------|
| i5-1135G7 (4C/8T)  | ~8-12 tokens/s — fine for friendly chat        |
| 16 GB RAM          | comfortable (model uses ~2 GB)                 |
| 8 GB RAM           | usable; reduce `MSN_AI_N_CTX` to 1024         |
| GPU (Iris Xe etc.) | optional; llama.cpp runs on CPU by default     |

## Verify end to end

```powershell
# Health (should now report provider="llamacpp")
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -Method Get

# Chat (creates a session)
$body = '{"message":"Hey MSN AI, introduce yourself."}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat" -Method Post -ContentType "application/json" -Body $body

# Continue the conversation (pass session_id from the previous response)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat" -Method Post `
  -ContentType "application/json" `
  -Body '{"message":"What did you just tell me?","session_id":"<id>"}'
```

## API surface (v1)

| Method | Path                        | Description                          |
|--------|-----------------------------|--------------------------------------|
| GET    | `/`                         | Service info banner                  |
| GET    | `/api/v1/health`            | Health + active provider/model       |
| POST   | `/api/v1/chat`              | Send a message (creates/reuses session) |
| GET    | `/api/v1/sessions`          | List active sessions                 |
| GET    | `/api/v1/sessions/{id}`     | Session history                      |
| DELETE | `/api/v1/sessions/{id}`     | Delete a session                     |

Interactive docs: http://127.0.0.1:8000/docs

## Configuration

All settings come from environment variables with the `MSN_AI_` prefix,
loaded from `.env` (see `.env.example`).

| Variable                 | Default                        | Meaning                          |
|--------------------------|--------------------------------|----------------------------------|
| `MSN_AI_HOST`            | `127.0.0.1`                    | Bind address                     |
| `MSN_AI_PORT`            | `8000`                         | Bind port                        |
| `MSN_AI_PROVIDER`        | `mock`                         | `mock` / `llamacpp` / `local`    |
| `MSN_AI_MODEL_NAME`      | `mock-msn-ai-v0.1`             | Model identifier                 |
| `MSN_AI_MAX_TOKENS`      | `1024`                         | Generation budget                |
| `MSN_AI_TEMPERATURE`     | `0.7`                          | Sampling temperature             |
| `MSN_AI_MODEL_PATH`      | (derived)                      | ABSOLUTE path to the .gguf file  |
| `MSN_AI_N_CTX`           | `2048`                         | Context window for llama.cpp     |
| `MSN_AI_N_THREADS`       | `0`                            | 0 == auto                        |
| `MSN_AI_MODEL_HF_REPO`   | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Download source              |
| `MSN_AI_MODEL_HF_FILE`   | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Download file              |
| `MSN_AI_LOG_LEVEL`       | `INFO`                         | Logging level                    |

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
│   ├── models\        Pydantic schemas
│   ├── tools\         reserved (MSN AI Tools)
│   └── utils\         path helpers
├── data\              datasets, conversations, generated images/videos
├── model_storage\     base (GGUF lives here), checkpoints, cache
├── training\          reserved
├── tests\             pytest suite
├── scripts\           fetch_model.ps1 (model download)
├── config\            configuration files
├── docs\              documentation
└── web\               reserved web frontend
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