# RAG vs Agentic RAG — How the Two Projects Differ

This document explains how the fixed RAG pipeline (`CascadeProjects/rag`) and the agentic RAG system (`CascadeProjects/agentic_rag_qa`) work, where they are the same, and where they fundamentally differ.

---

## 1. The Core Difference — Fixed Pipeline vs Agent Loop

This is the most important thing to understand first.

**Previous project (fixed pipeline):**

Every question — no matter what it asks — always runs the exact same steps in the exact same order:

```
Question
  → Router         (always runs)
  → Hybrid Search  (always runs)
  → RRF Merge      (always runs)
  → Reranker       (always runs)
  → Parent Lookup  (always runs)
  → Generator      (always runs)
  → Answer
```

The code decides everything. The LLM only does two things: routing (which years?) and final answer generation. It has no say in how retrieval is done, how many times to search, or whether a search is even needed.

**New project (agent loop):**

The LLM drives the process. GPT-4o reads the question, decides what to do, calls a tool, reads the result, decides what to do next, and so on — until it has enough context to write an answer.

```
Question
  → Agent (GPT-4o)
      → thinks: "I need Tesla 2024 risks"
      → calls retrieve(query, companies=["tesla"], years=["2024"])
      → reads result
      → thinks: "I also need GM 2024 risks"
      → calls retrieve(query, companies=["gm"], years=["2024"])
      → reads result
      → thinks: "I have enough, I'll write the answer now"
      → writes answer
  → Answer
```

The LLM is now the orchestrator. The tools are just functions the agent can call. Nothing is hardcoded except the tools available and the iteration cap (MAX_ITERATIONS=5).

---

## 2. What Is the Same in Both Projects

The retrieval pipeline — the actual mechanics of finding relevant content — is identical in both projects. This is important: the "RAG" part didn't change. What changed is who controls it.

Both projects do:

1. **Hierarchical chunking** — documents split into parent sections and child paragraphs
2. **Dense search** — child paragraphs embedded with OpenAI `text-embedding-3-large` and stored in ChromaDB
3. **Sparse search** — BM25 keyword index across all child paragraphs
4. **Reciprocal Rank Fusion (RRF)** — merge dense and BM25 ranked lists
5. **Cross-encoder reranking** — `BAAI/bge-reranker-v2-m3` scores each candidate against the question
6. **Parent lookup** — swap top-ranked child chunks for their full parent sections
7. **Labeled context** — each parent section labeled with `[Source: Company Year 10-K — Section]`
8. **Cited answer** — GPT-4o generates an answer with inline citations

The difference is **who calls step 1–8 and when**.

---

## 3. The Router — Hardcoded Logic vs Agent Reasoning

**Previous project — `router.py` (separate module, always runs first):**

Every question goes through a dedicated GPT-4o-mini call whose only job is to return which companies and years to search. This runs before any retrieval.

```python
# router.py — separate LLM call, always happens
def route_query(question: str) -> dict:
    response = client.chat.completions.create(model="gpt-4o-mini", ...)
    return {"companies": [...], "years": [...], "reasoning": "..."}
```

The router output is then used to filter ChromaDB and BM25. One routing decision, one retrieval, done.

**New project — no router:**

The agent itself decides which companies and years to search — it's part of its reasoning, not a separate step. When the agent calls `retrieve`, it passes `companies` and `years` as arguments. It can also call `get_document_index()` first if it's unsure what's available.

```python
# agent decides this itself — no separate module
retrieve(query="Tesla risks", companies=["tesla"], years=["2024"])
```

The critical difference: in the old project, routing happens once before retrieval. In the new project, the agent can change its mind between tool calls — if the first retrieval result suggests it needs more data, it can call retrieve again with different parameters.

---

## 4. The Rewriter — Separate Module vs Native Context

**Previous project — `rewriter.py` (separate module, runs in chat mode):**

In `chat.py`, before any retrieval, a separate GPT-4o-mini call takes the follow-up question + last 3 turns of history and rewrites the question into a standalone question.

```
You: "Which of those were new in 2024?"
Rewriter → "Which Tesla 2024 risk factors were not present in the 2023 10-K?"
```

This rewritten question then enters the pipeline. The original question is stored in history (not the rewritten one), so future rewrites see natural language.

**New project — no rewriter:**

The agent receives the full conversation history directly in its `messages` list. GPT-4o resolves references like "which of those" naturally because it can see the previous question and answer in context. No separate module, no extra API call.

```python
# chat.py — history passed directly to agent
messages = [system_prompt] + history + [current_question]
```

This works because GPT-4o is a much stronger model than GPT-4o-mini. It doesn't need a pre-processing step to understand "those" — it can figure it out from the conversation.

---

## 5. The Generator — Separate Module vs Agent's Final Step

**Previous project — `generator.py` (separate module, always last):**

After retrieval, a dedicated function sends the question + retrieved parent sections to GPT-4o and returns the answer. It has its own system prompt. It is always the last step.

```python
# generator.py — always called after retrieve()
def generate_answer(question, context_chunks, chunk_metas):
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
```

**New project — no generator:**

The agent writes the final answer itself when it decides it has enough context. The system prompt tells GPT-4o how to cite and structure the answer. There is no separate generation step — when the model responds without a tool call, that response is the answer.

```python
# agent.py — if no tool_calls, the message IS the answer
if not msg.tool_calls:
    return {"answer": msg.content, ...}
```

---

## 6. How the Conversation History Works Differently

**Previous project:**

History is stored as a list of `{"question": str, "answer": str}` dicts in a `deque(maxlen=3)`. The rewriter converts this into a text block and passes it to GPT-4o-mini to produce a standalone question.

```python
history = deque(maxlen=3)  # 3 Q&A pairs
# history → rewriter → standalone question → pipeline
```

**New project:**

History is stored as a list of OpenAI message dicts (role: user / role: assistant) in a `deque(maxlen=6)` — 3 turns × 2 messages each. This is passed directly into the agent's messages list.

```python
history = deque(maxlen=6)  # 3 turns × (user + assistant)
# history → messages list → agent loop
```

The format difference matters: the old project stored Q&A pairs and formatted them for the rewriter. The new project stores raw OpenAI messages and passes them directly to the API — no formatting needed.

---

## 7. What the Logs Show — Then vs Now

**Previous project logs:**
```json
{
  "question": "...",
  "route": {"companies": ["tesla"], "years": ["2024"], "reasoning": "..."},
  "top_child_chunks": [{"chunk_id": "...", "dense_score": 0.82, "bm25_score": 6.1, "reranker_score": 0.94, ...}],
  "dense_candidates": 30,
  "bm25_candidates": 30,
  "total_candidates": 55,
  "answer": "...",
  "ragas": {"faithfulness": 0.91, ...}
}
```

You could see every retrieval score (dense, BM25, reranker), the routing decision, and RAGAS evaluation scores.

**New project logs:**
```json
{
  "question": "...",
  "agent_trace": [
    {
      "tool": "retrieve",
      "args": {"query": "...", "companies": ["tesla"], "years": ["2024"]},
      "chunks_metadata": [{"section_name": "...", "reranker_score": 0.99, ...}],
      "result": "full context text..."
    }
  ],
  "total_tool_calls": 2,
  "answer": "...",
  "timing_ms": 22000
}
```

You can see the agent's reasoning sequence — which tools it called, in what order, with what arguments, and what it got back. The dense and BM25 scores are not surfaced (they happen inside retriever.py) but the reranker scores per chunk are visible.

---

## 8. The Confidence Check — Hardcoded vs Gone

**Previous project:**

After retrieval, `query.py` and `chat.py` explicitly check the top reranker score:

```python
if top_score < CONFIDENCE_THRESHOLD:  # 0.4
    print("⚠️ Low confidence warning")
```

The answer is still generated, but the user is warned.

**New project:**

No explicit confidence check. The agent handles this implicitly — if the reranker scores are low, the retrieved content will be weak, and GPT-4o is instructed to say "I don't have enough information" if context is insufficient. The reranker scores are visible in the log so you can inspect them manually.

---

## 9. Multi-Year Retrieval — How Each Project Handles It

This is where the architectural difference is most visible.

**Previous project:**

The router returns e.g. `years: ["2022", "2023", "2024"]`. A single hybrid search runs across all three years simultaneously, filtered by those years. RRF merges everything. The dynamic top-k formula (`5 + (num_years - 1) * 2`) allocates more reranker slots for multi-year queries to ensure all years are represented.

One search, all years together.

**New project:**

For a question like "How did Tesla's gross margin change from 2022 to 2024?", the agent made 4 tool calls:

```
[1] get_document_index()               ← checked what's available
[2] retrieve(query, years=["2022"])    ← searched 2022 specifically
[3] retrieve(query, years=["2023"])    ← searched 2023 specifically
[4] retrieve(query, years=["2024"])    ← searched 2024 specifically
```

The agent decomposed the question on its own and searched each year separately. This is more precise — each retrieval is focused on one year — but uses more API calls. The previous project would have done this in one retrieval call.

---

## 10. Summary Table

| Aspect | Previous Project | New Project |
|---|---|---|
| **Who controls flow** | Python code (fixed) | GPT-4o (dynamic) |
| **Routing** | Separate `router.py` module, GPT-4o-mini call | Agent decides via tool arguments |
| **Retrieval count** | Always once per question | Agent decides — 1 to 5 calls |
| **Multi-year search** | Single search, all years together | Separate search per year/company |
| **Query rewriting** | Separate `rewriter.py` module, GPT-4o-mini call | Agent resolves references from history natively |
| **Answer generation** | Separate `generator.py` module | Agent writes answer directly |
| **Conversation memory** | deque of Q&A pairs → rewriter | deque of OpenAI messages → agent |
| **Confidence check** | Explicit check, prints warning | Implicit — GPT-4o says "insufficient context" |
| **Corpus** | Tesla only, 3 years (2022–2024) | Tesla + GM + Ford, 4 years (2022–2025) |
| **Evaluation** | RAGAS (faithfulness, correctness, recall) | Manual golden dataset (20 questions) |
| **Models used** | GPT-4o-mini (routing, rewriting) + GPT-4o (answer) | GPT-4o only |
| **Core retrieval pipeline** | Hybrid search → RRF → rerank → parent lookup | Identical |
| **Logs show** | Dense/BM25/reranker scores, route decision | Agent trace, tool args, reranker scores per chunk |

---

## 11. Which Approach Is Better?

Neither is universally better — they make different trade-offs.

**Fixed pipeline is better when:**
- The question type is always the same (single company, financial Q&A)
- Latency matters — one retrieval call is faster than four
- You want full control and predictability — every question takes the same path
- You want RAGAS evaluation — fixed output is easier to score automatically

**Agentic is better when:**
- Questions vary widely — some need one company, some need three
- You want the system to adapt — call retrieve more times for complex questions
- You want to add new tools later (calculator, web search) without restructuring
- The LLM is strong enough to reason about what it needs

The agentic approach trades **predictability and speed** for **flexibility and reasoning**. The fixed pipeline trades **flexibility** for **control and efficiency**.

In practice, most production RAG systems start fixed and add agentic behavior incrementally — exactly the path this project took.
