# Semantic Text-to-SQL — Optimize Tokens

A token-efficient, local-first conversational Text-to-SQL application. Deterministic hybrid schema
retrieval and metadata grounding build a compact context before the SQL-generation model is called.
The implementation is organized around its own retrieval, grounding, validation, execution, and
evaluation contracts.

<p align="center">
  <img src="docs/assets/text-to-sql-architecture.png" alt="Semantic Text-to-SQL architecture" width="1000">
</p>

_Visual overview of the query workflow. The detailed architecture and validation boundaries below
are authoritative._

## Current architecture

```text
              User Question
                   |
                   v
          Conversation Resolver
                   |
                   v
         Trusted Evidence Merge
                   |
                   v
   +---- Hybrid Schema Retrieval ----+
   | BM25                             |
   | Dense embeddings                 |
   | Value matching                   |
   | Reciprocal Rank Fusion (RRF)     |
   | Production LightGBM reranking    |
   +---------------+------------------+
                   |
                   v
         Dependency Restoration
      PK / FK / formulas / bridges
                   |
                   v
          Verified Context
     + exact selected tables
     + exact selected columns
     + physical types
     + date formats
     + grain and cardinality
     + exact glossary concepts
                   |
                   v
              SQL Model
          reasoning and generation
                   |
                   v
         Thin SQLGlot Safety
                   |
                   v
          Read-only Execution
             |             |
          success        failure
             |             |
             |       focused repair
             |         max 3 tries
             |             |
             +------+------+
                    |
                    v
        Final SQL, result, context,
           and attempt history
```

The web application has one model selector. The selected model performs SQL reasoning, generation,
and focused repair.

## Model responsibilities

### Deterministic hybrid retrieval

BM25 searches names, descriptions, aliases, directly relevant glossary terms, and profile metadata.
FastEmbed supplies local dense similarity, while value matching compares explicit question literals
with profiled values. Each source produces an independent ranking; RRF combines ranks without mixing
incompatible raw scores. Retrieval targets five tables and roughly five columns per table, then
restores primary keys, relationship keys, glossary formula dependencies, and necessary bridge tables.
The dense model is downloaded locally on first use; no vector database or remote embedding service is
used.

### Production ML schema reranker

A versioned LightGBM learning-to-rank model can reorder only the RRF candidate pool. It cannot add
schema objects or metadata, and deterministic key/formula/bridge restoration still runs after it.
If the artifact is absent, incompatible, or fails during inference, retrieval falls back to RRF.
The production configuration enables it with `TEXT2SQL_SCHEMA_RERANKER_ENABLED=true`; install the
runtime dependency with `uv sync --extra ml`.
Its features include BM25, embedding, value-match and RRF ranks, key roles, plus similarity-weighted
table/column usage from up to three compatible successful historical queries. These internal
historical signals do not cause SQL to be copied into the prompt.

Production behavior is deliberately fail-safe:

- The model artifact is loaded once during API startup, not once per request.
- Feature order is frozen and versioned with the artifact.
- Inference is restricted to candidates already retrieved by BM25, embeddings, values, and RRF.
- Deterministic PK/FK, formula-dependency, and bridge restoration runs after ML scoring.
- A missing or incompatible artifact logs a warning and automatically falls back to RRF.
- Retrieval telemetry records ML scores and the final selected schema for operational diagnosis.

The bundled v1 artifact was trained on 290 questions from the leakage-screened 399-record BIRD
history corpus and evaluated on a fixed 69-question holdout (seed 42); 40 records with SQL that
could not be reliably mapped to live schema labels were skipped. The protected 100-question test
split was not used. Historical features for holdout questions come only from the training partition,
and training questions exclude themselves. RRF continues to rank tables because that preserves
held-out table recall; ML reranks columns only:

| Held-out retrieval metric | RRF | RRF tables + ML columns |
|---|---:|---:|
| Exact gold-table recall at 5 | 85.51% | 85.51% |
| Mean gold-column recall at 5 per selected table | 74.84% | 83.68% |

These are retrieval metrics, not SQL execution accuracy. Reproduce training with:

```bash
uv sync --extra ml
uv run python scripts/train_schema_reranker.py \
  --database-root /path/to/BIRD/dev_databases \
  --output models/schema_reranker/v1/model.txt
```

### Deterministic grounding

After retrieval, deterministic code verifies identifiers and adds required database
facts:

- Table primary keys, unique keys, relationship keys, and row grain
- Relationship path, cardinality, key uniqueness, and fanout risk for selected tables
- Physical database type for every selected column
- `observed_nulls`: whether profiling found any SQL `NULL`
- Up to five observed `top_values`, `allowed_values`, and examples for selected text/categorical
  columns
- Mandatory `observed_format` only for every selected date/datetime column
- Numeric min/max only when the question contains a numeric threshold/range intent
- Directly matched, approved business definitions

Example Model 2 context:

```json
{
  "question": "How many records are in each category?",
  "dialect": "sqlite",
  "tables": {
    "items": {
      "grain": "one row per ItemID",
      "primary_key": ["ItemID"],
      "unique_keys": [["ItemID"]],
      "columns": {
        "ItemID": {
          "type": "INTEGER",
          "observed_nulls": false
        },
        "Category": {
          "type": "TEXT",
          "semantic_type": "categorical",
          "observed_nulls": false,
          "top_values": ["Books", "Music"]
        }
      }
    }
  }
}
```

The web interface exposes this object in an expanded **Information passed to Model 2** panel.
Retrieval telemetry remains available in the API response for diagnostics but is not displayed in
the conversational interface.

### Model 2: SQL generator

Model 2 receives the question, dialect, selected live schema, verified context, optional trusted
evidence, at most one strongly matched validated historical query, and focused repair information
after a failure. Historical retrieval is same-database and fail-closed at the configured threshold;
no example is the normal result when a strong analogue is unavailable. Model 2 returns one SQL
statement only:

```sql
SELECT COUNT(*) AS customer_count
FROM customers;
```

It does not return a semantic-plan JSON object, Markdown, commentary, or multiple alternatives.

## Validation boundary

The active generation path deliberately keeps only high-confidence validation:

- SQL parses with SQLGlot
- Exactly one statement
- `SELECT` or `WITH ... SELECT` only
- No `INSERT`, `UPDATE`, `DELETE`, DDL, administrative commands, or `SELECT INTO`
- Execution uses a read-only connection/transaction

The runtime does **not** reject SQL using exact formula strings, exact aggregation structures,
specific join/CTE strategies, grain/cardinality AST patterns, or glossary formula matching.

Statuses have narrow meanings:

- `SQL_SAFETY_VALID`: syntax and read-only safety checks passed
- `ACCEPTED`: read-only execution also succeeded

Neither status proves business correctness. Returning rows—or returning a non-empty result—is never
treated as proof that the query correctly answers the question.

When an attempt fails, the web application displays its attempt number, error code, explanation,
and rejected SQL. Parse and database failures receive focused repair feedback. Generation
is bounded to three total SQL attempts.

## Models

The current SQL model catalog exposes:

- Local Ollama: `qwen3.5:9b`
- AgentRouter: `gpt-5.6-sol`
- AgentRouter: `claude-opus-5`
- AgentRouter: `claude-opus-4-7`
- Groq: `qwen/qwen3.6-27b`

The model endpoint reports whether each option is currently configured. AgentRouter credentials
and Groq credentials remain server-side and are never sent to the browser. The selected provider
and model are used for Model 2 and any focused repair unless an explicit server-side override is set.

## Install

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Ollama when using the local model

```bash
git clone https://github.com/meisamgh/semantic_text2sql_optimize_tokens.git
cd semantic_text2sql_optimize_tokens
uv sync --dev
cp .env.example .env
```

For local generation:

```bash
ollama pull qwen3.5:9b
ollama serve
```

For AgentRouter, put the token in the ignored `.env` file:

```dotenv
AGENTROUTER_API_KEY=your-token-here
AGENTROUTER_BASE_URL=https://agentrouter.org
```

For Qwen 3.6 27B on Groq:

```dotenv
GROQ_API_KEY=your-groq-key-here
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

Never commit `.env`; it is ignored by Git.

## Database layout

SQLite databases are discovered under `TEXT2SQL_DATABASE_ROOT`. Each database uses this layout:

```text
data/
  books/
    books.sqlite
```

For BIRD databases, point the root to the directory containing database-ID folders:

```dotenv
TEXT2SQL_DATABASE_ROOT=../bird-bench/llm/mini_dev_data/minidev/MINIDEV/dev_databases
TEXT2SQL_PROFILE_ROOT=profiles
TEXT2SQL_GLOSSARY_ROOT=data/business_glossaries
```

Profiles are generated offline:

```bash
uv run python scripts/profile_database.py \
  --dialect sqlite \
  --db-id books \
  --output profiles
```

Historical examples are disabled by default. The implementation remains available behind
`TEXT2SQL_HISTORY_ENABLED=false`; enable it only after a paired evaluation demonstrates benefit.
The bundled BIRD seed and evaluation tools live under `benchmarks/` and are not part of the runtime
pipeline.

## Run

```bash
set -a
source .env
set +a
uv run uvicorn semantic_text2sql.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"status":"online"}
```

## API

- `POST /api/chat/jobs`: start a cancellable conversational query
- `GET /api/chat/jobs/{job_id}`: inspect progress or retrieve the response
- `DELETE /api/chat/jobs/{job_id}`: cancel an active request
- `POST /api/chat`: synchronous conversation endpoint
- `POST /api/check`: check and optionally execute supplied read-only SQL
- `GET /api/health`: health status
- `GET /api/models`: configured model discovery
- `GET /api/databases`: configured database discovery

Example synchronous request:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "db_id": "books",
    "message": "How many books are there in each category?",
    "provider": "ollama",
    "model": "qwen3.5:9b"
  }'
```

Conversation state is process-local. Failed turns preserve the last accepted state. The web UI
supports follow-ups, corrections, explanation, optimization, cancellation, SQL copying, result
tables, token usage, exact Model 2 context, and validation-attempt inspection.

## Configuration

Important environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `TEXT2SQL_DATABASE_ROOT` | SQLite database root | `data` |
| `TEXT2SQL_PROFILE_ROOT` | Offline profile root | `profiles` |
| `TEXT2SQL_GLOSSARY_ROOT` | Approved glossary root | `data/business_glossaries` |
| `OLLAMA_BASE_URL` | Ollama endpoint | `http://127.0.0.1:11434` |
| `AGENTROUTER_API_KEY` | AgentRouter token | unset |
| `AGENTROUTER_BASE_URL` | AgentRouter gateway | `https://agentrouter.org` |
| `GROQ_API_KEY` | Groq API key | unset |
| `GROQ_BASE_URL` | Groq OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `TEXT2SQL_EMBEDDING_MODEL` | Local FastEmbed model | `BAAI/bge-small-en-v1.5` |
| `TEXT2SQL_RETRIEVAL_TABLES` | High-recall table budget before bridge expansion | `5` |
| `TEXT2SQL_RETRIEVAL_COLUMNS` | Approximate columns per table before dependency restoration | `5` |
| `TEXT2SQL_SCHEMA_RERANKER_ENABLED` | Enable production LightGBM reranking after RRF | `true` in `.env.example` |
| `TEXT2SQL_SCHEMA_RERANKER_MODEL` | Versioned LightGBM text-model artifact | `models/schema_reranker/v1/model.txt` |
| `TEXT2SQL_SCHEMA_RERANKER_POOL` | Maximum RRF candidates sent to the ML reranker | `30` |
| `TEXT2SQL_SQL_MODEL` | Optional Model 2 override | selected model |
| `TEXT2SQL_HISTORY_ENABLED` | Enable one strong, validated historical example | `true` |
| `TEXT2SQL_HISTORY_MIN_SCORE` | Minimum historical score | `0.85` |
| `TEXT2SQL_HISTORY_ML_MIN_SCORE` | Minimum match used as an internal ML schema feature | `0.65` |

## Verification

```bash
uv run ruff format --check src
uv run ruff check src
uv run mypy src/semantic_text2sql
```

The previous v3/v4 benchmark numbers are intentionally omitted because this pipeline changed
materially. A new execution-accuracy claim requires a frozen, matched evaluation of this exact
Optimize Tokens
pipeline.

## Security notes

- `.env`, virtual environments, generated profiles, and local runtime files are ignored by Git.
- API keys remain server-side.
- SQL is parsed and restricted to one read-only query before execution.
- SQLite uses query-only connections; PostgreSQL uses read-only transactions.
- Rows and retry attempts are bounded.
- The bundled `books.sqlite` is demonstration data, not a production database.

## Project layout

```text
src/semantic_text2sql/
  api.py              FastAPI and web endpoints
  conversation.py     turn classification and session state
  hybrid_retrieval.py BM25, dense, value matching, RRF, and bridge expansion
  context_planner.py  Deterministic context verification and dependency restoration
  context.py          deterministic context assembly
  linker.py           table-first and column-level retrieval
  profiling.py        offline database profiles
  glossary.py         direct-match glossary retrieval
  llm.py              Ollama, AgentRouter, and Groq adapters plus Model 2 prompt
  validator.py        SQLGlot syntax and read-only safety checks
  agent.py            bounded repair and read-only execution
web/
  index.html
  app.js
  styles.css
scripts/
  create_demo_db.py
  profile_database.py
  train_schema_reranker.py
models/schema_reranker/v1/
  model.txt          versioned production LightGBM artifact
  metrics.json       frozen training and holdout retrieval metrics
benchmarks/
  evaluate_bird.py
  build_bird_history.py
  run_chat_queue.py
  data/bird_history_seed42_400.json
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party references and licenses.
