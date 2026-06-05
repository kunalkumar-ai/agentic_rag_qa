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
