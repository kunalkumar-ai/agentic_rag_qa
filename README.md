# Financial Due Diligence Assistant

**Live demo →** [sec10kfinancialassistance.streamlit.app](https://sec10kfinancialassistance.streamlit.app)  
**Source** → [github.com/kunalkumar-ai/agentic_rag_qa](https://github.com/kunalkumar-ai/agentic_rag_qa)

An agentic RAG system for multi-company financial due diligence. Ask questions across Tesla, GM, and Ford 10-K filings (2022–2025) and get cited, source-grounded answers — with full retrieval transparency.

Built from scratch without LangChain or LlamaIndex. Every layer of the stack is written and understood before any abstraction.

---

## What It Does

- Answers questions across 12 SEC 10-K filings — Tesla, GM, Ford × 2022–2025
- Handles multi-company, multi-year, and follow-up questions natively
- Every factual sentence cites its source section inline
- Correctly refuses out-of-corpus questions (wrong company, wrong year, live data)
- Full agent trace visible in the UI — every tool call, every retrieved chunk, every reranker score

---

## Pipeline

**Ingestion (offline, run once):**
```
PDFs → pdfplumber → hierarchical chunker (parent sections + child paragraphs)
  → OpenAI text-embedding-3-large → ChromaDB + BM25 index
  — every chunk tagged: company + year + parent_id
```

**Query (agentic, ReAct loop):**
```
Question → Agent (GPT-4o + function calling)
  → retrieve(query, companies, years)
    → Dense search (ChromaDB) + Sparse search (BM25)
    → Reciprocal Rank Fusion → top 20 candidates
    → Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
    → Parent section lookup
  → repeat up to MAX_ITERATIONS = 5
→ Cited answer
```

---

## Key Engineering Decisions

| Decision | Why |
|---|---|
| **Built from scratch, no LangChain** | LangChain abstracts away the agent loop. Building it raw forced understanding of how ReAct actually works — message history, tool dispatch, iteration caps, failure modes. |
| **Hierarchical chunking** | Flat chunks split sentences mid-paragraph. Two-level structure: small child chunks for retrieval precision, large parent sections for LLM context. |
| **Hybrid search (dense + BM25)** | Dense embeddings fail on exact numbers and ticker symbols. BM25 fails on paraphrase. RRF merge of both gives better coverage than either alone. |
| **Cross-encoder reranking** | Bi-encoder embeddings rank candidates fast but roughly. Cross-encoder re-scores top 20 candidates with full query-document attention — better precision at the cost of latency. |
| **No router, no rewriter** | The ReAct agent handles query routing and follow-up reference resolution natively using conversation history — fewer moving parts, same capability. |
| **MAX_ITERATIONS = 5** | Hard cap prevents runaway loops and unbounded cost. If the agent can't answer in 5 tool calls, it says so — better than hallucinating. |
| **LLM-as-judge eval** | Single GPT-4o-mini call per question scores refusal, fact coverage, and correctness simultaneously — more robust to phrasing variation than regex pattern matching. |

---

## Evaluation

Validated against a 20-question golden dataset across 8 categories:
`single-company`, `multi-company`, `multi-year`, `out-of-corpus`, `ambiguous`, `numeric`, `adversarial`

| Metric | Score |
|---|---|
| Routing | **1.00** — always retrieves the right company/year |
| Faithfulness | **1.00** — no hallucinated citations |
| Refusal | **1.00** — correctly declines out-of-corpus questions |
| Fact Coverage | **0.71** |
| Correctness | **0.74** |

Results are versioned in `eval/results/` — every pipeline change is measured.

---

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | GPT-4o (generation) + GPT-4o-mini (eval judge) |
| Embeddings | text-embedding-3-large (OpenAI) |
| Vector DB | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| Reranker | BAAI/bge-reranker-v2-m3 (local, sentence-transformers) |
| Frontend | Streamlit |
| Language | Python 3.13 |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# One-time ingestion
python3 extract.py   # PDF → text
python3 ingest.py    # text → ChromaDB + BM25

# Run the app
streamlit run app.py

# Or CLI
python3 query.py "What were Tesla's main risks in 2024?"
python3 chat.py  # conversational mode
```

---

## Project Structure

| File | Responsibility |
|---|---|
| `agent.py` | ReAct loop — tool dispatch, MAX_ITERATIONS cap |
| `retriever.py` | Hybrid search → RRF → rerank → parent lookup |
| `chunker.py` | Hierarchical chunking — company/year tags |
| `tools.py` | Tool implementations wrapping retriever |
| `tool_schemas.py` | OpenAI function call JSON schemas |
| `ingest.py` | Ingestion pipeline |
| `eval/run_eval.py` | Eval harness — runs golden dataset, writes versioned results |
| `eval/llm_judge.py` | LLM-as-judge — refusal, fact coverage, correctness |
| `eval/golden_dataset.jsonl` | 20-question ground-truth dataset |
| `app.py` | Streamlit UI |
