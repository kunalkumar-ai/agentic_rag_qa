# Agentic RAG QA

## What This Is

A learning project — evolving the fixed RAG pipeline from `CascadeProjects/rag` into an agentic system built from scratch. Goal is to understand agentic patterns before using frameworks (LangGraph, LlamaIndex) or Claude.

**Learning progression:** raw OpenAI function calling → frameworks → Claude

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
| `ingest.py` | Chunk → embed → ChromaDB + BM25 |
| `retriever.py` | Hybrid search → RRF → rerank → parent lookup |
| `tools.py` | Tool schemas + implementations for the agent |
| `agent.py` | ReAct loop — tool dispatch, MAX_ITERATIONS cap |
| `query.py` | Single-question entry point |
| `chat.py` | Conversational loop (deque maxlen=6) |
| `logger.py` | JSON trace logging per query |
| `sec_docs/` | 12 10-K PDFs |
| `docs/superpowers/specs/` | Design spec |

---

## How to Run

```bash
# One-time ingestion
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

## Previous Project

`CascadeProjects/rag` — fixed pipeline, Tesla only, 3 years. Reference it for chunker, retriever, and logging patterns.
