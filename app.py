import streamlit as st
from download_index import download_index
from agent import run_agent, MaxIterationsError

download_index()

st.set_page_config(
    page_title="Financial Due Diligence Assistant",
    page_icon="📊",
    layout="wide",
)

HISTORY_WINDOW = 6
GITHUB_URL = "https://github.com/kunalkumar-ai/agentic_rag_qa"

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def build_history() -> list[dict]:
    recent = st.session_state.messages[-HISTORY_WINDOW:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


# ---------- tabs ----------
tab_chat, tab_about = st.tabs(["💬 Ask Questions", "🏗️ How It Works"])


# ================================================================
# TAB 1 — CHAT
# ================================================================
with tab_chat:
    st.markdown("## 📊 Financial Due Diligence Assistant")
    st.markdown(
        "Ask anything across **Tesla, GM, and Ford 10-K filings (2022–2025)**. "
        "Every answer is grounded in the source filings and cited inline."
    )
    st.divider()

    # example questions
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    examples = [
        "What were Tesla's main risk factors in 2024?",
        "Compare GM and Ford revenue in 2024",
        "How did Tesla's gross margin change from 2022 to 2024?",
        "What is GM's autonomous driving strategy?",
        "How has Ford's EV strategy evolved from 2022 to 2024?",
        "Which company had the highest R&D spend in 2023?",
    ]
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

    st.divider()

    # render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                with st.expander(f"🔍 Agent trace — {msg['tool_calls']} tool call(s) · {msg['timing_ms']}ms"):
                    for i, step in enumerate(msg["trace"], 1):
                        st.markdown(f"**Step {i} · `{step['tool']}`**")
                        st.json(step["args"], expanded=False)
                        if step.get("chunks_metadata"):
                            st.markdown("Retrieved chunks:")
                            for c in step["chunks_metadata"]:
                                st.markdown(
                                    f"&nbsp;&nbsp;`{c['reranker_score']:.3f}` &nbsp;"
                                    f"**{c['company'].upper()} {c['year']}** — {c['section_name'][:80]}"
                                )

    # handle example button trigger
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user = st.session_state.messages[-1]["content"]
        if not (len(st.session_state.messages) > 1 and st.session_state.messages[-2]["role"] == "user"):
            with st.chat_message("assistant"):
                with st.spinner("Searching filings…"):
                    try:
                        result = run_agent(last_user, history=build_history()[:-1])
                        answer = result["answer"]
                        trace = result["agent_trace"]
                        tool_calls = result["total_tool_calls"]
                        timing = result["timing_ms"]
                        st.markdown(answer)
                        with st.expander(f"🔍 Agent trace — {tool_calls} tool call(s) · {timing}ms"):
                            for i, step in enumerate(trace, 1):
                                st.markdown(f"**Step {i} · `{step['tool']}`**")
                                st.json(step["args"], expanded=False)
                                if step.get("chunks_metadata"):
                                    for c in step["chunks_metadata"]:
                                        st.markdown(
                                            f"&nbsp;&nbsp;`{c['reranker_score']:.3f}` &nbsp;"
                                            f"**{c['company'].upper()} {c['year']}** — {c['section_name'][:80]}"
                                        )
                        st.session_state.messages.append({
                            "role": "assistant", "content": answer,
                            "trace": trace, "tool_calls": tool_calls, "timing_ms": timing,
                        })
                    except MaxIterationsError:
                        msg = "⚠️ Could not find a confident answer within the iteration limit. Try rephrasing."
                        st.warning(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    except Exception as e:
                        msg = f"⚠️ Something went wrong: {e}"
                        st.error(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})

    # chat input
    if question := st.chat_input("Ask a question about Tesla, GM, or Ford…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching filings…"):
                try:
                    result = run_agent(question, history=build_history())
                    answer = result["answer"]
                    trace = result["agent_trace"]
                    tool_calls = result["total_tool_calls"]
                    timing = result["timing_ms"]
                    st.markdown(answer)
                    with st.expander(f"🔍 Agent trace — {tool_calls} tool call(s) · {timing}ms"):
                        for i, step in enumerate(trace, 1):
                            st.markdown(f"**Step {i} · `{step['tool']}`**")
                            st.json(step["args"], expanded=False)
                            if step.get("chunks_metadata"):
                                for c in step["chunks_metadata"]:
                                    st.markdown(
                                        f"&nbsp;&nbsp;`{c['reranker_score']:.3f}` &nbsp;"
                                        f"**{c['company'].upper()} {c['year']}** — {c['section_name'][:80]}"
                                    )
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "trace": trace, "tool_calls": tool_calls, "timing_ms": timing,
                    })
                except MaxIterationsError:
                    msg = "⚠️ Could not find a confident answer within the iteration limit. Try rephrasing."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                except Exception as e:
                    msg = f"⚠️ Something went wrong: {e}"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

    # clear button
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", key="clear"):
            st.session_state.messages = []
            st.rerun()


# ================================================================
# TAB 2 — ABOUT
# ================================================================
with tab_about:
    st.markdown(f"## 🏗️ How It Works &nbsp;&nbsp; [![GitHub](https://img.shields.io/badge/GitHub-View_Source-black?logo=github)]({GITHUB_URL})")
    st.divider()

    # what it is
    st.markdown("### What This Is")
    st.markdown("""
A production-grade **agentic RAG system** for multi-company financial due diligence.
Ask questions across 12 SEC 10-K filings — Tesla, GM, and Ford, years 2022 through 2025 —
and get cited, source-grounded answers with full retrieval transparency.

Built entirely from scratch without LangChain or LlamaIndex.
The goal was to understand every layer of the stack before abstracting it.
""")

    st.divider()

    # corpus
    st.markdown("### Corpus")
    col1, col2, col3 = st.columns(3)
    col1.metric("Companies", "3", "Tesla · GM · Ford")
    col2.metric("Years", "4", "2022 · 2023 · 2024 · 2025")
    col3.metric("Filings", "12", "10-K annual reports")

    st.divider()

    # pipeline
    st.markdown("### Pipeline")
    col_ingest, col_query = st.columns(2)

    with col_ingest:
        st.markdown("**Ingestion (offline, run once)**")
        st.code("""PDFs
→ pdfplumber extraction
→ Hierarchical chunker
  (parent sections + child paragraphs)
→ OpenAI text-embedding-3-large
→ ChromaDB (dense index)
→ BM25 index
  (every chunk tagged: company + year)""", language="text")

    with col_query:
        st.markdown("**Query (agentic, ReAct loop)**")
        st.code("""Question
→ Agent (GPT-4o + function calling)
  → retrieve(query, companies, years)
    → Dense search (ChromaDB)
    + Sparse search (BM25)
    → Reciprocal Rank Fusion
    → Cross-encoder reranking
      (BAAI/bge-reranker-v2-m3)
    → Parent section lookup
  → repeat up to MAX_ITERATIONS = 5
→ Cited answer""", language="text")

    st.divider()

    # key decisions
    st.markdown("### Key Engineering Decisions")
    decisions = [
        ("Built from scratch, no LangChain", "LangChain abstracts away the agent loop. Building it raw forced understanding of how ReAct actually works — message history, tool dispatch, iteration caps, failure modes. The abstractions make more sense after you've written the thing yourself."),
        ("Hierarchical chunking", "Flat chunks split sentences mid-paragraph. We use a two-level structure: small child chunks for retrieval precision, large parent sections for LLM context. The reranker scores child chunks; the LLM reads the full parent."),
        ("Hybrid search (dense + BM25)", "Dense embeddings fail on exact numbers and ticker symbols. BM25 fails on paraphrase and semantic similarity. Running both and merging with Reciprocal Rank Fusion gives better coverage than either alone."),
        ("Cross-encoder reranking", "Bi-encoder embeddings rank candidates fast but roughly. The cross-encoder (BAAI/bge-reranker-v2-m3) re-scores the top 20 candidates with full query-document attention — significantly better precision at the cost of latency."),
        ("No router, no rewriter", "Earlier systems used GPT-4o-mini to route queries to the right year/company and rewrite follow-up questions. The ReAct agent handles both natively — it decides which companies/years to retrieve based on the question and uses conversation history for reference resolution."),
        ("MAX_ITERATIONS = 5", "Hard cap prevents runaway loops and unbounded cost. Five iterations handles the hardest multi-company, multi-year queries. If the agent can't answer in five tool calls, it says so — which is better than hallucinating."),
        ("LLM-as-judge eval", "Instead of brittle regex pattern matching, every eval run sends (question, answer, ground truth, required facts) to GPT-4o-mini in a single call. It scores refusal, fact coverage, and correctness simultaneously — much more robust to phrasing variation."),
    ]
    for title, explanation in decisions:
        with st.expander(f"**{title}**"):
            st.markdown(explanation)

    st.divider()

    # eval results
    st.markdown("### Evaluation Results")
    st.markdown("Validated against a 20-question golden dataset across 8 categories (single-company, multi-company, multi-year, out-of-corpus, ambiguous, numeric, adversarial).")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Routing", "1.00", "Perfect")
    col2.metric("Faithfulness", "1.00", "Perfect")
    col3.metric("Fact Coverage", "0.71")
    col4.metric("Refusal", "1.00", "Perfect")
    col5.metric("Correctness", "0.74")

    st.caption("Routing = agent called retrieve with the right company/year. Faithfulness = no hallucinated citations. Fact coverage = required facts present in answer. Refusal = correctly declines out-of-corpus questions.")

    st.divider()

    # stack
    st.markdown("### Tech Stack")
    stack = {
        "LLM": "GPT-4o (answer generation)",
        "Embeddings": "text-embedding-3-large (OpenAI)",
        "Vector DB": "ChromaDB (local)",
        "Keyword search": "BM25 (rank-bm25)",
        "Reranker": "BAAI/bge-reranker-v2-m3 (local, sentence-transformers)",
        "Eval judge": "GPT-4o-mini (single call — refusal + fact coverage + correctness)",
        "Frontend": "Streamlit",
        "Language": "Python 3.13",
    }
    for k, v in stack.items():
        st.markdown(f"- **{k}:** {v}")

    st.divider()

    st.markdown(f"[📂 View source on GitHub]({GITHUB_URL})")
