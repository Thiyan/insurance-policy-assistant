# Insurance Policy Assistant

An AI-powered insurance policy analysis system that uses semantic search and contextual reasoning to answer questions grounded in actual policy documents. It processes complex insurance PDFs, understands coverage details, exclusions, and clauses, and returns precise, source-backed responses.

## Features

- **RAG pipeline** — ingests insurance PDFs, chunks and embeds them, then retrieves the most relevant context at query time
- **LangChain + OpenAI** — uses LangChain orchestration and OpenAI models for embeddings and answer generation
- **FastAPI** — exposes a REST API for querying the assistant
- **Configurable ingestion** — skip re-ingestion on subsequent startups by toggling a single env flag

## Project Structure

```
insurance-policy-assistant/
├── app/
│   ├── api/v1/           # API route definitions
│   ├── config/           # App configuration and constants
│   ├── exception/        # Custom exceptions and error codes
│   ├── model/            # Pydantic data models
│   ├── observation/      # Logging setup
│   ├── persistence/      # ChromaDB client
│   ├── service/
│   │   ├── generation/   # LLM response generation
│   │   ├── ingestion/    # PDF pre-processing, chunking, embedding
│   │   └── retrieval/    # Vector search and retrieval
│   └── main.py           # FastAPI entrypoint
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docker-run/           # Host-side mount directories for Docker
│   ├── resources/        # Place your insurance PDF here
│   ├── data/             # ChromaDB vector store (persisted)
│   └── logs/             # Application log files
├── resources/            # Insurance PDF(s) for local development
├── tests/
├── .env.example
└── pyproject.toml
```

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) — Python package manager used for dependency management
- An [OpenAI API key](https://platform.openai.com/api-keys)
- An insurance policy PDF (the `.env.example` references `Insurance_Handbook_20103.pdf`)

---

## Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/Thiyan/insurance-policy-assistant.git
cd insurance-policy-assistant
```

### 2. Install dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
pip install uv
uv sync
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
OPENAI_API_KEY=your_openai_api_key_here   # Required
PDF_PATH=./resources/your_policy.pdf      # Path to your insurance PDF
LLM_MODEL=gpt-4o                          # OpenAI model to use
EMBEDDING_MODEL=text-embedding-3-small    # Embedding model
RUN_INGESTION=true                        # Set to false after first run to reuse existing DB
```

### 4. Start the API

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. On first startup with `RUN_INGESTION=true`, the PDF will be parsed, chunked, and indexed into ChromaDB automatically.

---

## Running with Docker

### 1. Prepare the PDF

The repo includes a `docker-run/` directory that serves as the host-side mount point for the container. Place your insurance PDF inside `docker-run/resources/` before starting the container.

The three subdirectories map to the following container paths:

| Host path (`docker-run/`) | Container path | Purpose |
|---|---|---|
| `docker-run/resources/` | `/app/resources/` | Insurance PDF(s) to be ingested |
| `docker-run/data/` | `/app/data/` | ChromaDB vector store (persisted across restarts) |
| `docker-run/logs/` | `/app/logs/` | Application log files |

The paths configured in `.env.example` (`PDF_PATH`, `DB_PATH`, `LOG_DIR`) refer to paths **inside the container** and should be left as-is.

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Build and run

```bash
cd docker
docker compose up --build
```

The service starts on port `8000`. On first startup, the PDF is parsed and indexed into ChromaDB at the mounted `docker-run/data/` path, so the vector store survives container restarts and rebuilds.

To stop the containers:

```bash
docker compose down
```

> **Tip:** After the initial ingestion completes, set `RUN_INGESTION=false` in your `.env` to skip re-processing the PDF on subsequent restarts. The existing ChromaDB data in `docker-run/data/` will be reused.

---

## API Usage

Once running, send questions to the assistant:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is covered under the general liability policy?"}'
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## Configuration Reference

All configuration is done via environment variables. Copy `.env.example` to `.env` and adjust as needed.

### Application

| Variable | Default | Description |
|---|---|---|
| `PDF_PATH` | Local: `./resources/<file>.pdf` · Docker: `/app/resources/<file>.pdf` | Path to the insurance PDF to ingest |
| `DB_PATH` | Local: `./data/chroma_db` · Docker: `/app/data/chroma_db` | Directory where ChromaDB persists the vector store |
| `COLLECTION_NAME` | `rag_collection` | Name of the ChromaDB collection |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Number of tokens per text chunk when splitting the PDF |
| `CHUNK_OVERLAP` | `50` | Overlapping tokens between adjacent chunks to preserve context across boundaries |
| `N_RESULTS` | `3` | Number of top chunks to retrieve per query |

### LLM & Embeddings

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (**required**) |
| `LLM_MODEL` | `gpt-5.5` | OpenAI model used to generate answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model used to embed chunks and queries |
| `EMBEDDING_DIMS` | `1536` | Dimensionality of the embedding vectors — must match the chosen embedding model |
| `EMBED_BATCH_SIZE` | `100` | Number of chunks to embed in a single API call |
| `TEMPERATURE` | `0` | LLM sampling temperature — `0` means deterministic, factual responses |

### Vector DB

| Variable | Default | Description |
|---|---|---|
| `HNSW_SPACE` | `cosine` | Distance metric used by ChromaDB for similarity search (`cosine`, `l2`, or `ip`) |

### Ingestion

| Variable | Default | Description |
|---|---|---|
| `RUN_INGESTION` | `true` | Set to `true` on first run to parse and index the PDF. Set to `false` on subsequent startups to skip ingestion and reuse the existing vector store |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_DIR` | Local: `./logs` · Docker: `/app/logs` | Directory where log files are written |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_MAX_BYTES` | `500000000` | Maximum size of a single log file in bytes before rotation |
| `LOG_BACKUP_COUNT` | `3` | Number of rotated log files to retain |
| `LOG_CONSOLE_OUTPUT` | `true` | Whether to also print logs to stdout |

---

## License

[Apache 2.0](./LICENSE)