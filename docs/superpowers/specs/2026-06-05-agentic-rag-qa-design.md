# Agentic RAG QA — Design Spec

**Date:** 2026-06-05
**Project:** agentic_rag_qa
**Purpose:** Learning project — evolve a fixed RAG pipeline into a from-scratch agentic system using OpenAI function calling (ReAct loop), then migrate to frameworks/Claude afterward.

---

## Context

The previous project (`CascadeProjects/rag`) was a fixed pipeline for Tesla 10-K filings:
`Question → Router → Hybrid Retrieve → Rerank → Parent Lookup → GPT-4o → Answer`

Every question ran the same steps in the same order. This project replaces that fixed pipeline with an agent that decides what to do at each step.

---

## Corpus

12 PDFs in `sec_docs/`:
- Companies: Tesla, GM, Ford
- Years: 2022, 2023, 2024, 2025
- Same hierarchical chunking strategy as the previous project (parent sections + child paragraphs, company + year metadata on every chunk)

---

## Architecture

### Ingestion (offline, run once)

```
12 PDFs (Tesla/GM/Ford × 2022–2025)
  → pdfplumber extraction
  → hierarchical chunker
      — parent chunks: full sections
      — child chunks: individual paragraphs
      — every chunk tagged: company + year + parent_id
      — child chunk ceiling: 6000 chars (carried over from previous project)
  → OpenAI text-embedding-3-large → ChromaDB (local)
  → unified BM25 index (all child chunks, all companies, all years)
```

### Query (online, agentic)

```
Question
  → Agent loop (GPT-4o + function calling, ReAct pattern)
      → calls tools one at a time, up to MAX_ITERATIONS = 5
      → builds up context across tool call results
      → writes final cited answer when it has enough context
  → Cited answer printed to console
```

---

## Tools

The agent has access to exactly two tools:

### `retrieve(query, companies, years, top_k)`

Runs the full retrieval pipeline for one focused sub-query:
1. Dense search (ChromaDB, filtered by company + year)
2. Sparse search (BM25, filtered by company + year)
3. Reciprocal Rank Fusion merge
4. Cross-encoder reranking (BAAI/bge-reranker-v2-m3, local)
5. Parent section lookup (swap child chunks for full parent sections)

Returns top-k chunks with source labels: `[Source: Tesla 2024 10-K — Section Name]`

**When the agent calls this:** Once per focused sub-query. For a multi-company question, the agent calls retrieve multiple times — one per company/year combination it needs.

**Parameters:**
- `query`: str — the focused sub-question to retrieve for
- `companies`: list[str] — filter to these companies (e.g., `["tesla", "gm"]`)
- `years`: list[int] — filter to these years (e.g., `[2023, 2024]`)
- `top_k`: int — number of parent chunks to return (default: 5)

### `get_document_index()`

Returns the full list of available companies and years from config. No parameters.

**When the agent calls this:** When the question is ambiguous about which companies/years are relevant, or as a first step on broad questions. Prevents the agent from hallucinating company/year combinations that don't exist in the corpus.

---

## Agent Loop

```python
messages = [system_prompt] + conversation_history + [{"role": "user", "content": question}]

for _ in range(MAX_ITERATIONS):  # MAX_ITERATIONS = 5
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    if response has tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            append tool_call + result to messages
    else:
        return response.content  # final answer, exit loop

raise MaxIterationsError("Agent did not finish within 5 iterations")
```

**System prompt instructs the agent to:**
- Call `get_document_index` first if unsure what's available
- Call `retrieve` once per focused sub-query, not one broad query
- Cite every factual sentence inline: *(Tesla 2024 10-K — Section Name)*
- Stop and write the final answer once it has sufficient context

**MAX_ITERATIONS = 5** — hard cap. If hit, raises `MaxIterationsError` (visible error, not a silent wrong answer).

---

## Entry Points

### `query.py` — single question

```bash
python3 query.py "How did GM's revenue change from 2022 to 2024?"
```

- Calls agent loop with no conversation history
- Prints full tool call trace + final answer to console
- Writes JSON log to `logs/`

### `chat.py` — conversational

```bash
python3 chat.py
```

- Maintains conversation history as a `deque(maxlen=6)` — 3 turns × (user message + assistant answer)
- Each turn appends to `messages` before calling the agent loop
- No separate rewriter — the agent resolves references ("which of those...") using conversation context natively
- Commands: `exit` to quit, `clear` to reset history

---

## Logging

Every query writes a JSON log to `logs/` containing:
- `question`: original user question
- `agent_trace`: list of tool calls in order — `{tool, args, result}` per call
- `total_tool_calls`: int
- `answer`: final answer text
- `timing_ms`: total elapsed time

For chat sessions, each turn is logged individually. The log is the primary way to inspect agent reasoning during development.

---

## Files

| File | Responsibility |
|---|---|
| `config.py` | Constants (MAX_ITERATIONS, models, thresholds) + DOCUMENTS registry |
| `chunker.py` | Hierarchical chunking — company/year tags, 6000-char ceiling |
| `ingest.py` | Ingestion pipeline — chunk → embed → ChromaDB + BM25 |
| `retriever.py` | Hybrid search → RRF → rerank → parent lookup (called by retrieve tool) |
| `tools.py` | Tool definitions: `retrieve`, `get_document_index` — schemas + implementations |
| `agent.py` | ReAct loop — messages list, tool dispatch, MAX_ITERATIONS cap |
| `query.py` | Single-question entry point |
| `chat.py` | Conversational loop with sliding window history |
| `logger.py` | JSON trace logging per query |

---

## What Was Dropped vs. Previous Project

| Previous | Now | Why |
|---|---|---|
| `router.py` (separate routing step) | Agent decides routing via tool calls | Agent does this natively |
| `rewriter.py` (query rewriting) | Agent resolves references in context | Agent has full history |
| `evaluator.py` (RAGAS) | Deferred | Add after core loop works |
| Calculator tool | Deferred | GPT-4o handles simple arithmetic in context |

---

## Build Order

1. `config.py` + `chunker.py` + `ingest.py` — extend ingestion to all 12 docs
2. `retriever.py` — carry over from previous project, minor param changes
3. `tools.py` — wrap retriever as callable tool with OpenAI function schema
4. `agent.py` — ReAct loop, tool dispatch, MAX_ITERATIONS cap
5. `query.py` — single-question entry point
6. `logger.py` — JSON trace logging
7. `chat.py` — conversational loop
8. Evaluate on golden dataset questions across all 3 companies
