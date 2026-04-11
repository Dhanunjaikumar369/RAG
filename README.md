# SafetyRAG — Workplace Safety Q&A

Ask any question about **industrial safety, fire regulations, emergency action plans,
and employee rights** — and get grounded, citation-aware answers backed by official
**OSHA guidelines**.

Built with **LangChain + FAISS + HuggingFace BGE embeddings + Groq (LLaMA-3)**.

---

## Architecture

```
Employee Question
      │
      ▼
┌──────────────────────┐
│   Query Rewriter     │  ← Groq LLaMA-3 rewrites question for better retrieval
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  FAISS Retriever     │  ← Semantic search over OSHA PDF chunks (BGE embeddings)
│  (top-k chunks)      │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Answer Generator    │  ← Groq LLaMA-3 synthesises a grounded answer
│  (with citations)    │     citing source document + page number
└──────────────────────┘
```

### Indexed Documents (OSHA Public Domain)

| File | Description | Pages |
|---|---|---|
| `emergency_action_plans.pdf` | OSHA Emergency Action Plans guide | 34 |
| `employee_safety_handbook.pdf` | OSHA Employee Safety & Health Handbook | 31 |
| `fire_safety.pdf` | OSHA Fire Safety in the Workplace | — |

---

## Project Structure

```
RAG/
├── src/safetyrag/
│   ├── config.py                  # pydantic-settings — loads from .env
│   ├── ingest.py                  # PDF → chunks → FAISS index
│   ├── retriever.py               # FAISS retriever factory
│   ├── chain.py                   # Adaptive RAG chain (rewrite → retrieve → answer)
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   └── routers/
│   │       ├── ingest.py          # POST /ingest/upload, POST /ingest/rebuild
│   │       └── query.py           # POST /query
│   └── cli/
│       └── cli.py                 # safetyrag-ingest, safetyrag-query
├── docs/                          # OSHA PDF source documents
│   ├── emergency_action_plans.pdf
│   ├── employee_safety_handbook.pdf
│   └── fire_safety.pdf
├── tests/
│   ├── test_ingest.py
│   ├── test_chain.py
│   └── test_api.py
├── .github/workflows/ci.yml       # Lint → Test → Docker CI
├── Dockerfile                     # Multi-stage production image
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Dhanunjaikumar369/RAG.git
cd RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=your_groq_api_key
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Build the knowledge base

```bash
safetyrag-ingest
# Loads all PDFs from ./docs/, chunks them, embeds with BGE, saves FAISS index
```

### 4. Ask a question

```bash
# Single question
safetyrag-query "What should I do if a fire alarm goes off?"

# Interactive REPL
safetyrag-query
```

---

## Usage

### CLI

#### `safetyrag-ingest`

```
Usage: safetyrag-ingest [OPTIONS]

  Index all PDFs in DOCS_DIR into the FAISS vector store.

Options:
  --docs-dir PATH                 PDF folder  [default: docs]
  --force                         Force rebuild even if index exists
  --log-level [DEBUG|INFO|...]    [default: INFO]
```

#### `safetyrag-query`

```
Usage: safetyrag-query [OPTIONS] [QUESTION]

  Ask a workplace safety question. Omit QUESTION for interactive REPL.

Options:
  --log-level [DEBUG|INFO|...]    [default: INFO]
```

**Example session:**

```
$ safetyrag-query
SafetyRAG — Workplace Safety Q&A  (type 'exit' to quit)

Question > What PPE is required when handling hazardous chemicals?

Answer:
Based on OSHA guidelines, the required Personal Protective Equipment (PPE)
when handling hazardous chemicals includes:

• Chemical-resistant gloves (nitrile or neoprene)
• Safety goggles or face shield
• Chemical-resistant apron or coveralls
• Respiratory protection if vapors are present (N95 or supplied-air respirator)

[Source: employee_safety_handbook.pdf | Page: 12]
```

---

### REST API

```bash
safetyrag-api
# → http://localhost:8000
# → http://localhost:8000/docs   (Swagger UI)
```

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/ingest/status` | Check if index exists |
| `POST` | `/ingest/upload` | Upload a new PDF to extend the knowledge base |
| `POST` | `/ingest/rebuild` | Rebuild index from all PDFs in `./docs/` |
| `POST` | `/query` | Ask a workplace safety question |

#### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements for fire extinguisher placement?"}'
```

```json
{
  "question": "What are the requirements for fire extinguisher placement?",
  "answer": "According to OSHA fire safety guidelines:\n• Extinguishers must be..."
}
```

#### Example: Upload a new document

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@my_safety_policy.pdf"
```

---

### Docker

```bash
cp .env.example .env      # set GROQ_API_KEY
docker compose up --build
# Auto-ingests ./docs/ PDFs on first start
# API at http://localhost:8000
```

---

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health                             PASSED
tests/test_api.py::test_ingest_status_no_index             PASSED
tests/test_api.py::test_query_no_index_returns_503         PASSED
tests/test_api.py::test_query_with_mocked_chain            PASSED
tests/test_api.py::test_query_missing_body                 PASSED
tests/test_chain.py::test_ask_returns_string               PASSED
tests/test_chain.py::test_build_llm_raises_without_key     PASSED
tests/test_ingest.py::test_load_and_split_calls_loader     PASSED
tests/test_ingest.py::test_load_index_raises_when_missing  PASSED
tests/test_ingest.py::test_build_index_skips_if_exists     PASSED
```

---

## Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key *(required)* | — |
| `LLM_MODEL` | Groq model ID | `llama3-8b-8192` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.0` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-small-en-v1.5` |
| `VECTOR_STORE_PATH` | Path to persist FAISS index | `vector_store` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K` | Chunks retrieved per query | `4` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| RAG framework | [LangChain](https://langchain.com/) 0.2 |
| LLM | Groq — LLaMA-3 8B (free tier available) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (local, no API key) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) (persisted to disk) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| CLI | [Click](https://click.palletsprojects.com/) |
| Config | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| Source documents | OSHA public-domain safety publications |

---

## Author

**Dhanunjaikumar** — [github.com/Dhanunjaikumar369](https://github.com/Dhanunjaikumar369)
