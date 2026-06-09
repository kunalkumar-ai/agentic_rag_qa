# Agentic RAG QA

## What This Is

An agentic RAG system for multi-company financial due diligence. Ask questions across Tesla, GM, and Ford 10-K filings (2022–2025) and get cited, source-grounded answers. Built from scratch using OpenAI function calling with a ReAct agent loop.

---

## Corpus

12 PDFs in `sec_docs/` — Tesla, GM, Ford × 2022, 2023, 2024, 2025 10-K filings.

---

## Architecture

**Ingestion (offline):**
```
PDFs → pdfplumber → hierarchical chunker (parent sections + child paragraphs)
  → OpenAI embeddings → ChromaDB + unified BM25 index
  — every chunk tagged: company + year + parent_id
```

**Query (agentic, ReAct loop):**
```
Question → Agent (GPT-4o + function calling)
  → calls tools one at a time, MAX_ITERATIONS = 5
  → writes final cited answer when done
```

---

## Tools

| Tool | What it does |
|---|---|
| `retrieve(query, companies, years, top_k)` | Hybrid search → RRF → rerank → parent lookup |
| `get_document_index()` | Returns available companies + years from config |

No router, no rewriter — the agent handles these natively.

---

## Files

| File | Responsibility |
|---|---|
| `config.py` | Constants (MAX_ITERATIONS=5, models) + DOCUMENTS registry |
| `chunker.py` | Hierarchical chunking — company/year tags, 6000-char ceiling |
| `extract.py` | PDF → txt via pdfplumber, saves to `extracted_texts/` |
| `ingest.py` | txt → chunk → embed → ChromaDB + BM25 |
| `retriever.py` | Hybrid search → RRF → rerank → parent lookup |
| `tool_schemas.py` | OpenAI function call JSON schemas |
| `tools.py` | Tool implementations wrapping retriever |
| `agent.py` | ReAct loop — tool dispatch, MAX_ITERATIONS cap |
| `query.py` | Single-question entry point |
| `chat.py` | Conversational loop (deque maxlen=6) |
| `logger.py` | JSON trace logging per query |
| `sec_docs/` | 12 10-K PDFs |
| `extracted_texts/` | Extracted .txt files (one per PDF, output of extract.py) |
| `docs/superpowers/specs/` | Design spec |
| `docs/superpowers/plans/` | Implementation plan |

---

## How to Run

```bash
# Step 1: extract PDFs to text (one-time)
python3 extract.py

# Step 2: ingest into ChromaDB + BM25 (one-time)
python3 ingest.py

# Single question
python3 query.py "How did GM revenue change from 2022 to 2024?"

# Conversational
python3 chat.py
```

---

## Key Decisions

| Decision | Why |
|---|---|
| No framework | Learn ReAct internals before abstracting them |
| MAX_ITERATIONS = 5 | Enough for multi-company questions; hard cap prevents runaway loops |
| Two tools only | retrieve + get_document_index covers all query types |
| No rewriter | Agent resolves references using conversation history natively |
| No calculator | GPT-4o handles arithmetic in context; add tool if it fails |
| No RAGAS yet | Add after core loop works |

---

## Reference

`CascadeProjects/rag` — predecessor project, fixed pipeline, Tesla only. Reference for chunker, retriever, and logging patterns.

---

## Production Hardening Roadmap

Goal: turn this learning project into a production-grade artifact for the job hunt. Each phase produces a measurable, demonstrable improvement — not new features, but production discipline. Apply to jobs in parallel; let interview feedback reprioritize phases 3–4.

### Phase 1 — Eval harness (week 1)

Build measurement before anything else. Without a number, every later change is a vibe.

- `eval/golden_dataset.jsonl` — start with 20 hand-written questions, grow to 100+. Schema per line:
  ```json
  {"id": 1, "question": "...", "expected_companies": ["Tesla"], "expected_years": [2024], "category": "...", "ground_truth_answer": "...", "ground_truth_source_chunks": ["chunk-id-1"]}
  ```
- Categories to cover: `single-company-single-year`, `multi-company-single-year`, `single-company-multi-year`, `multi-company-multi-year`, `out-of-corpus`, `ambiguous`, `numeric-comparison`, `adversarial` (e.g. asking about Honda).
- `eval/run_eval.py` — runs full pipeline on every question, records: retrieval recall@k, faithfulness, answer correctness, latency, $/query, tool calls made, iterations used.
- `eval/results/<timestamp>.json` — versioned results. Commit them. Track metrics over time.
- Deliverable: a graph showing baseline → improvements as you iterate.

### Phase 2 — Deploy (week 2)

A public URL beats a GitHub repo 10:1 for hiring managers.

- FastAPI backend wrapping `agent.py` — `/query`, `/chat`, `/health`, `/stats`.
- Minimal frontend (Next.js or Streamlit) — chat UI with cited sources rendered.
- Host on Modal / Fly / Railway.
- Put the URL in the README and resume.

### Phase 3 — Observability + failure handling (week 3)

Per-query structured logs (extend `logger.py`): tool calls, retrieval hit/miss, MAX_ITERATIONS hit, rate-limit retries, latency p50/p95, $/query.

Failure modes to handle explicitly and write down:
- MAX_ITERATIONS reached without final answer
- Retrieval confidence below threshold
- Out-of-corpus questions (e.g. Honda when only Tesla/GM/Ford loaded)
- OpenAI rate-limits / transient errors
- Concurrent requests

Tiny `/dashboard` (or Streamlit sidecar) reading the logs: p50/p95 latency, $/query, eval score over time.

### Phase 4 — Cost engineering + interview-driven iteration (week 4+)

- Add prompt caching. Measure $/query before and after. One-line resume bullet.
- Consider model routing (Haiku for tool-arg generation, Sonnet/4o for final answer).
- From here on, prioritize based on what interviewers actually probe on. Candidates: LangGraph rewrite, guardrails, structured-data RAG, voice, multi-tenant auth.

### Tracking

Each phase ends with a commit that includes: code change + new eval results JSON + one-line note in this README explaining what moved and by how much.
