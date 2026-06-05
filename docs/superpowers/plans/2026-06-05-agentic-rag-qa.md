# Agentic RAG QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch agentic RAG system over Tesla, GM, and Ford 10-K filings (2022–2025) using a ReAct loop with OpenAI function calling.

**Architecture:** GPT-4o agent loop calls two tools (`retrieve`, `get_document_index`) one at a time, up to MAX_ITERATIONS=5, then writes a final cited answer. Ingestion splits into PDF extraction (`extract.py`) and embedding (`ingest.py`). Retrieval is hybrid dense+BM25 → RRF → cross-encoder rerank → parent lookup, ported from the previous project with the router removed.

**Tech Stack:** Python 3.10+, OpenAI (gpt-4o, text-embedding-3-large), ChromaDB, rank-bm25, sentence-transformers (BAAI/bge-reranker-v2-m3), pdfplumber, pytest

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `config.py` | Create | Constants + DOCUMENTS registry (12 PDFs) |
| `extract.py` | Create | PDF → txt via pdfplumber |
| `chunker.py` | Port | Hierarchical chunking (identical to previous project) |
| `ingest.py` | Port+adapt | txt → chunk → embed → ChromaDB + BM25 |
| `retriever.py` | Port+adapt | Hybrid search → RRF → rerank → parent lookup (router removed) |
| `tool_schemas.py` | Create | OpenAI function call JSON schemas |
| `tools.py` | Create | Tool implementations wrapping retriever |
| `agent.py` | Create | ReAct loop — tool dispatch, MAX_ITERATIONS cap |
| `logger.py` | Port+adapt | JSON trace logging for agent trace format |
| `query.py` | Create | Single-question entry point |
| `chat.py` | Create | Conversational loop with sliding window history |
| `tests/test_chunker.py` | Port | Hierarchy correctness (identical to previous project) |
| `tests/test_retriever.py` | Port | RRF algorithm (identical to previous project) |
| `tests/test_agent.py` | Create | Agent loop behavior with mocked OpenAI |

---

## Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `.env` (from template)
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
openai
chromadb
rank-bm25
sentence-transformers
pdfplumber
python-dotenv
pytest
```

- [ ] **Step 2: Create `.env`**

```
OPENAI_API_KEY=sk-...
```

- [ ] **Step 3: Create `tests/__init__.py`**

Empty file — makes `tests/` a package so pytest finds imports correctly.

```python
```

- [ ] **Step 4: Create virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Expected: `.venv/` directory created, prompt shows `(.venv)`.

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "feat: project setup — requirements and test package"
```

---

## Task 2: `config.py`

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write `config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-4o"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

CHROMA_PATH = "chroma_db"
BM25_INDEX_PATH = "bm25_index.pkl"
PARENTS_PATH = "parents.pkl"
LOGS_DIR = "logs"

MAX_ITERATIONS = 5

TOP_K_DENSE = 30
TOP_K_BM25 = 30
TOP_K_RERANK_BASE = 5
RRF_K = 60
MAX_CONTEXT_CHARS = 3000
GENERATION_TEMPERATURE = 0.3

DOCUMENTS = [
    {"company": "tesla", "year": "2022", "path": "sec_docs/tesla_2022.pdf", "txt_path": "sec_docs/tesla_2022.txt"},
    {"company": "tesla", "year": "2023", "path": "sec_docs/tesla_2023.pdf", "txt_path": "sec_docs/tesla_2023.txt"},
    {"company": "tesla", "year": "2024", "path": "sec_docs/tesla_2024.pdf", "txt_path": "sec_docs/tesla_2024.txt"},
    {"company": "tesla", "year": "2025", "path": "sec_docs/tesla_2025.pdf", "txt_path": "sec_docs/tesla_2025.txt"},
    {"company": "gm", "year": "2022", "path": "sec_docs/gm_2022.pdf", "txt_path": "sec_docs/gm_2022.txt"},
    {"company": "gm", "year": "2023", "path": "sec_docs/gm_2023.pdf", "txt_path": "sec_docs/gm_2023.txt"},
    {"company": "gm", "year": "2024", "path": "sec_docs/gm_2024.pdf", "txt_path": "sec_docs/gm_2024.txt"},
    {"company": "gm", "year": "2025", "path": "sec_docs/gm_2025.pdf", "txt_path": "sec_docs/gm_2025.txt"},
    {"company": "ford", "year": "2022", "path": "sec_docs/ford_2022.pdf", "txt_path": "sec_docs/ford_2022.txt"},
    {"company": "ford", "year": "2023", "path": "sec_docs/ford_2023.pdf", "txt_path": "sec_docs/ford_2023.txt"},
    {"company": "ford", "year": "2024", "path": "sec_docs/ford_2024.pdf", "txt_path": "sec_docs/ford_2024.txt"},
    {"company": "ford", "year": "2025", "path": "sec_docs/ford_2025.pdf", "txt_path": "sec_docs/ford_2025.txt"},
]

AVAILABLE_COMPANIES = sorted({d["company"] for d in DOCUMENTS})
AVAILABLE_YEARS = sorted({d["year"] for d in DOCUMENTS})
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python3 -c "from config import DOCUMENTS, AVAILABLE_COMPANIES, AVAILABLE_YEARS; print(AVAILABLE_COMPANIES, AVAILABLE_YEARS)"
```

Expected: `['ford', 'gm', 'tesla'] ['2022', '2023', '2024', '2025']`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: config with 12-document registry"
```

---

## Task 3: `chunker.py` + `tests/test_chunker.py`

**Files:**
- Create: `chunker.py` (ported verbatim from previous project)
- Create: `tests/test_chunker.py` (ported verbatim, one new test added)

- [ ] **Step 1: Write `tests/test_chunker.py`**

```python
from chunker import build_chunks, parse_sections

SAMPLE_TEXT = """ITEM 1. BUSINESS
Overview
We design and manufacture electric vehicles and energy storage systems.

We sell them directly to customers through our website and stores.

ITEM 1A. RISK FACTORS
We may experience delays in launching products.

We face significant competition from other manufacturers."""


def test_parse_sections_finds_two_items():
    sections = parse_sections(SAMPLE_TEXT)
    assert len(sections) == 2


def test_parse_sections_names_start_with_item():
    sections = parse_sections(SAMPLE_TEXT)
    assert sections[0]["name"].startswith("ITEM 1.")
    assert sections[1]["name"].startswith("ITEM 1A.")


def test_each_section_has_text():
    sections = parse_sections(SAMPLE_TEXT)
    for section in sections:
        assert len(section["text"].strip()) > 0


def test_build_chunks_produces_parents_and_children():
    chunks = build_chunks(SAMPLE_TEXT)
    types = {c.chunk_type for c in chunks}
    assert "parent" in types
    assert "child" in types


def test_child_chunk_parent_id_matches_a_parent():
    chunks = build_chunks(SAMPLE_TEXT)
    parent_ids = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
    children = [c for c in chunks if c.chunk_type == "child"]
    for child in children:
        assert child.parent_id in parent_ids


def test_no_empty_chunks():
    chunks = build_chunks(SAMPLE_TEXT)
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_section_name_propagated_to_children():
    chunks = build_chunks(SAMPLE_TEXT)
    children = [c for c in chunks if c.chunk_type == "child"]
    for child in children:
        assert child.section_name != ""


def test_very_short_paragraphs_are_skipped():
    text = """ITEM 2. PROPERTIES
We own the following properties across our global operations.

Note

These properties support manufacturing operations across all regions."""
    chunks = build_chunks(text)
    children = [c for c in chunks if c.chunk_type == "child"]
    child_texts = [c.text for c in children]
    assert not any(t.strip() == "Note" for t in child_texts)


def test_chunk_ids_are_unique():
    chunks = build_chunks(SAMPLE_TEXT)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_company_and_year_tagged_on_every_chunk():
    chunks = build_chunks(SAMPLE_TEXT, company="tesla", year="2024")
    for chunk in chunks:
        assert chunk.company == "tesla"
        assert chunk.year == "2024"


def test_chunk_id_prefixed_with_company_and_year():
    chunks = build_chunks(SAMPLE_TEXT, company="gm", year="2023")
    for chunk in chunks:
        assert chunk.chunk_id.startswith("gm_2023_")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunker'`

- [ ] **Step 3: Write `chunker.py`**

```python
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: str
    section_name: str
    chunk_type: str  # "parent" or "child"
    company: str = ""
    year: str = ""


_MIN_SUBSECTION_CHARS = 150


def parse_sections(text: str) -> list[dict]:
    pattern = re.compile(r'(ITEM\s+\d+[A-C]?\.\s+[^\n]+)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"name": name, "text": text[start:end].strip()})
    return sections


def _is_subheading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    words = s.split()
    return (
        1 <= len(words) <= 12 and
        len(s) <= 80 and
        s[0].isupper() and
        not s.isupper() and
        not s.endswith('.') and
        not s.endswith(',') and
        not s.endswith(':') and
        '/' not in s and
        '%' not in s and
        not s.lower().startswith('total ') and
        not s[-1].isdigit() and
        not re.search(r'\d{4}.*\d{4}', s)
    )


def _split_subsections(section_name: str, section_text: str) -> list[dict]:
    subsections = []
    current_name = section_name
    current_lines: list[str] = []

    for line in section_text.split('\n'):
        if _is_subheading(line):
            text = '\n'.join(current_lines).strip()
            if len(text) >= _MIN_SUBSECTION_CHARS:
                subsections.append({"name": current_name, "text": text})
                current_name = f"{section_name} — {line.strip()}"
                current_lines = []
            else:
                current_name = f"{section_name} — {line.strip()}"
                current_lines = [l for l in current_lines if l.strip()]
        else:
            current_lines.append(line)

    text = '\n'.join(current_lines).strip()
    if text:
        subsections.append({"name": current_name, "text": text})

    return subsections if len(subsections) > 1 else [{"name": section_name, "text": section_text}]


_MAX_CHILD_CHARS = 6000


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    result = []
    for p in paragraphs:
        p = p.strip()
        if len(p.split()) < 10:
            continue
        if len(p) <= _MAX_CHILD_CHARS:
            result.append(p)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > _MAX_CHILD_CHARS and current:
                    result.append(current.strip())
                    current = sentence
                else:
                    current = current + " " + sentence if current else sentence
            if current.strip():
                result.append(current.strip())
    return result


def build_chunks(text: str, company: str = "", year: str = "") -> list[Chunk]:
    sections = parse_sections(text)
    all_chunks: list[Chunk] = []
    prefix = f"{company}_{year}_" if company and year else ""

    for i, section in enumerate(sections):
        subsections = _split_subsections(section["name"], section["text"])

        for j, subsec in enumerate(subsections):
            parent_id = f"{prefix}parent_{i}_{j}"

            all_chunks.append(Chunk(
                chunk_id=parent_id,
                text=subsec["text"],
                parent_id=parent_id,
                section_name=subsec["name"],
                chunk_type="parent",
                company=company,
                year=year,
            ))

            for k, para in enumerate(_split_paragraphs(subsec["text"])):
                all_chunks.append(Chunk(
                    chunk_id=f"{prefix}child_{i}_{j}_{k}",
                    text=para,
                    parent_id=parent_id,
                    section_name=subsec["name"],
                    chunk_type="child",
                    company=company,
                    year=year,
                ))

    return all_chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chunker.py tests/test_chunker.py
git commit -m "feat: chunker with company/year tagging"
```

---

## Task 4: `extract.py`

**Files:**
- Create: `extract.py`

No unit test — output is verified by inspecting the produced `.txt` files.

- [ ] **Step 1: Write `extract.py`**

```python
import pdfplumber
from config import DOCUMENTS


def extract_pdf(pdf_path: str, txt_path: str) -> None:
    """Extract text from a PDF using pdfplumber and save to a .txt file."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                pages.append(text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pages))
    print(f"  Extracted {len(pages)} pages → {txt_path}")


def extract_all() -> None:
    """Extract all PDFs in the DOCUMENTS registry to .txt files."""
    for doc in DOCUMENTS:
        print(f"Extracting {doc['company']} {doc['year']}...")
        extract_pdf(doc["path"], doc["txt_path"])
    print("\nExtraction complete.")


if __name__ == "__main__":
    extract_all()
```

- [ ] **Step 2: Run extraction**

```bash
python3 extract.py
```

Expected: 12 `.txt` files created in `sec_docs/`. Each line prints `Extracted N pages → sec_docs/<company>_<year>.txt`.

- [ ] **Step 3: Spot-check one file**

```bash
head -50 sec_docs/tesla_2024.txt
```

Expected: readable text with ITEM headings visible (e.g., `ITEM 1. BUSINESS`).

- [ ] **Step 4: Commit**

```bash
git add extract.py
git commit -m "feat: PDF extraction to txt via pdfplumber"
```

---

## Task 5: `retriever.py` + `tests/test_retriever.py`

**Files:**
- Create: `retriever.py` (ported, router removed, signature changed)
- Create: `tests/test_retriever.py` (ported verbatim)

- [ ] **Step 1: Write `tests/test_retriever.py`**

```python
from retriever import rrf_merge


def test_item_in_both_lists_ranks_first():
    dense = ["a", "b", "c"]
    bm25  = ["b", "d", "e"]
    result = rrf_merge(dense, bm25)
    assert result[0] == "b"


def test_includes_all_unique_ids():
    dense = ["a", "b"]
    bm25  = ["c", "d"]
    result = rrf_merge(dense, bm25)
    assert set(result) == {"a", "b", "c", "d"}


def test_higher_rank_beats_lower_rank():
    result = rrf_merge(["first", "second", "third"], [])
    assert result[0] == "first"
    assert result[-1] == "third"


def test_handles_empty_bm25():
    result = rrf_merge(["a", "b", "c"], [])
    assert result == ["a", "b", "c"]


def test_handles_empty_dense():
    result = rrf_merge([], ["x", "y", "z"])
    assert result == ["x", "y", "z"]


def test_handles_both_empty():
    assert rrf_merge([], []) == []


def test_no_duplicates_in_output():
    dense = ["a", "b", "c"]
    bm25  = ["a", "b", "d"]
    result = rrf_merge(dense, bm25)
    assert len(result) == len(set(result))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'retriever'`

- [ ] **Step 3: Write `retriever.py`**

```python
import pickle
import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH,
    BM25_INDEX_PATH, PARENTS_PATH, RERANKER_MODEL,
    TOP_K_DENSE, TOP_K_BM25, RRF_K,
)

client = OpenAI(api_key=OPENAI_API_KEY)
reranker = CrossEncoder(RERANKER_MODEL)


def _build_chroma_filter(companies: list[str], years: list[str]) -> dict | None:
    conditions = []
    if len(companies) == 1:
        conditions.append({"company": {"$eq": companies[0]}})
    elif len(companies) > 1:
        conditions.append({"company": {"$in": companies}})

    if len(years) == 1:
        conditions.append({"year": {"$eq": years[0]}})
    elif len(years) > 1:
        conditions.append({"year": {"$in": years}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _dense_search(question: str, companies: list[str], years: list[str]) -> tuple[list[str], dict, dict, dict]:
    q_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("filings")

    where_filter = _build_chroma_filter(companies, years)
    query_kwargs = dict(
        query_embeddings=[q_vec],
        n_results=TOP_K_DENSE,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    result = collection.query(**query_kwargs)
    ids = result["ids"][0]
    texts = {cid: doc for cid, doc in zip(ids, result["documents"][0])}
    metas = {cid: meta for cid, meta in zip(ids, result["metadatas"][0])}
    scores = {cid: float(dist) for cid, dist in zip(ids, result["distances"][0])}
    return ids, texts, metas, scores


def _bm25_search(question: str, companies: list[str], years: list[str]) -> tuple[list[str], dict, dict, dict]:
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    tokenized_query = question.lower().split()
    raw_scores = bm25_data["bm25"].get_scores(tokenized_query)

    companies_set = set(companies)
    years_set = set(years)

    ranked = sorted(
        [
            (cid, score)
            for cid, score in zip(bm25_data["chunk_ids"], raw_scores)
            if bm25_data["chunk_metas"][cid]["company"] in companies_set
            and bm25_data["chunk_metas"][cid]["year"] in years_set
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K_BM25]

    ids = [cid for cid, _ in ranked]
    texts = {cid: bm25_data["chunk_texts"][cid] for cid, _ in ranked}
    metas = {cid: bm25_data["chunk_metas"][cid] for cid, _ in ranked}
    scores = {cid: float(score) for cid, score in ranked}
    return ids, texts, metas, scores


def rrf_merge(dense_ids: list[str], bm25_ids: list[str], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def retrieve(question: str, companies: list[str], years: list[str], top_k: int = 5) -> dict:
    """Hybrid search filtered by company+year → RRF → rerank → parent lookup.

    companies and years come from the agent — no router involved.
    top_k is passed by the tool layer (scaled by number of years requested).
    """
    dense_ids, dense_texts, dense_metas, dense_scores = _dense_search(question, companies, years)
    bm25_ids, bm25_texts, bm25_metas, bm25_scores = _bm25_search(question, companies, years)

    all_texts = {**bm25_texts, **dense_texts}
    all_metas = {**bm25_metas, **dense_metas}

    merged_ids = rrf_merge(dense_ids, bm25_ids)
    valid_ids = [cid for cid in merged_ids if cid in all_texts]

    rerank_candidates = valid_ids[:20]
    pairs = [[question, all_texts[cid]] for cid in rerank_candidates]
    rerank_scores_list = reranker.predict(pairs)
    reranked = sorted(zip(rerank_candidates, rerank_scores_list), key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in reranked[:top_k]]
    rerank_score_map = {cid: float(score) for cid, score in reranked}

    with open(PARENTS_PATH, "rb") as f:
        parents: dict[str, str] = pickle.load(f)

    parent_texts: list[str] = []
    seen: set[str] = set()
    for cid in top_ids:
        pid = all_metas[cid]["parent_id"]
        if pid not in seen and pid in parents:
            parent_texts.append(parents[pid])
            seen.add(pid)

    return {
        "top_child_chunks": [
            {
                "chunk_id": cid,
                "parent_id": all_metas.get(cid, {}).get("parent_id", ""),
                "text": all_texts.get(cid, ""),
                "section_name": all_metas.get(cid, {}).get("section_name", ""),
                "company": all_metas.get(cid, {}).get("company", ""),
                "year": all_metas.get(cid, {}).get("year", ""),
                "reranker_score": rerank_score_map.get(cid, 0.0),
            }
            for cid in top_ids
        ],
        "parent_texts": parent_texts,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: retriever — hybrid search, RRF, rerank, parent lookup"
```

---

## Task 6: `ingest.py`

**Files:**
- Create: `ingest.py` (ported, reads `.txt` files, collection renamed to "filings")

No unit test — verified by running ingestion and checking ChromaDB count.

- [ ] **Step 1: Write `ingest.py`**

```python
import pickle
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chunker import build_chunks
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH,
    BM25_INDEX_PATH, PARENTS_PATH, DOCUMENTS,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest() -> None:
    """Read .txt files for all documents, chunk → embed → ChromaDB + BM25."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection("filings")
    except Exception:
        pass
    collection = chroma_client.create_collection("filings")

    all_parents: dict[str, str] = {}
    all_bm25_children = []

    for doc in DOCUMENTS:
        company, year, txt_path = doc["company"], doc["year"], doc["txt_path"]
        print(f"\nIngesting {company} {year} from {txt_path}...")

        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        all_chunks = build_chunks(text, company=company, year=year)
        children = [c for c in all_chunks if c.chunk_type == "child"]
        parents = {c.chunk_id: c.text for c in all_chunks if c.chunk_type == "parent"}
        print(f"  Built {len(parents)} parent sections, {len(children)} child paragraphs")

        all_parents.update(parents)

        batch_size = 100
        for i in range(0, len(children), batch_size):
            batch = children[i: i + batch_size]
            embeddings = _embed_batch([c.text for c in batch])
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "parent_id": c.parent_id,
                        "section_name": c.section_name,
                        "company": c.company,
                        "year": c.year,
                    }
                    for c in batch
                ],
            )
            print(f"  Embedded {min(i + batch_size, len(children))}/{len(children)} child chunks")

        all_bm25_children.extend(children)

    with open(PARENTS_PATH, "wb") as f:
        pickle.dump(all_parents, f)
    print(f"\nParent store saved: {len(all_parents)} total parent sections")

    tokenized = [c.text.lower().split() for c in all_bm25_children]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in all_bm25_children],
            "chunk_texts": {c.chunk_id: c.text for c in all_bm25_children},
            "chunk_metas": {
                c.chunk_id: {
                    "parent_id": c.parent_id,
                    "section_name": c.section_name,
                    "company": c.company,
                    "year": c.year,
                }
                for c in all_bm25_children
            },
        }, f)
    print(f"BM25 index saved: {len(all_bm25_children)} total child chunks")
    print("\nIngestion complete.")


if __name__ == "__main__":
    ingest()
```

- [ ] **Step 2: Run ingestion** (requires extraction from Task 4 to be complete)

```bash
python3 ingest.py
```

Expected: progress prints for all 12 documents, ends with `Ingestion complete.`

- [ ] **Step 3: Verify ChromaDB count**

```bash
python3 -c "
import chromadb
c = chromadb.PersistentClient(path='chroma_db')
col = c.get_collection('filings')
print('Total chunks:', col.count())
"
```

Expected: a number in the thousands (typically 3000–8000 across 12 filings).

- [ ] **Step 4: Commit**

```bash
git add ingest.py
git commit -m "feat: ingest pipeline for 12 filings into ChromaDB + BM25"
```

---

## Task 7: `tool_schemas.py`

**Files:**
- Create: `tool_schemas.py`

- [ ] **Step 1: Write `tool_schemas.py`**

```python
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search the 10-K filings for relevant content. "
                "Call once per focused sub-query. For multi-company or multi-year questions, "
                "call multiple times — once per company/year combination. "
                "Returns labeled source passages for the specified companies and years."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused question or topic to retrieve context for.",
                    },
                    "companies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Companies to search. Valid values: 'tesla', 'gm', 'ford'.",
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filing years to search. Valid values: '2022', '2023', '2024', '2025'.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of source passages to return. Default 5. Use higher values (7-9) for multi-year queries.",
                        "default": 5,
                    },
                },
                "required": ["query", "companies", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_index",
            "description": (
                "Return the list of available companies and years in the corpus. "
                "Call this first when the question is ambiguous about which companies or years to search."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python3 -c "from tool_schemas import TOOL_SCHEMAS; print(len(TOOL_SCHEMAS), 'tools defined')"
```

Expected: `2 tools defined`

- [ ] **Step 3: Commit**

```bash
git add tool_schemas.py
git commit -m "feat: OpenAI function call schemas for retrieve and get_document_index"
```

---

## Task 8: `tools.py` + `tests/test_tools.py`

**Files:**
- Create: `tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write `tests/test_tools.py`**

```python
from unittest.mock import patch, MagicMock
import json
from tools import execute_tool


def _make_tool_call(name: str, args: dict):
    tool_call = MagicMock()
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(args)
    tool_call.id = "call_123"
    return tool_call


def test_get_document_index_returns_companies_and_years():
    tool_call = _make_tool_call("get_document_index", {})
    result = execute_tool(tool_call)
    data = json.loads(result)
    assert "companies" in data
    assert "years" in data
    assert "tesla" in data["companies"]
    assert "gm" in data["companies"]
    assert "ford" in data["companies"]
    assert "2024" in data["years"]


def test_retrieve_returns_formatted_string():
    mock_result = {
        "parent_texts": ["Some financial data here about revenue growth."],
        "top_child_chunks": [
            {
                "section_name": "Risk Factors",
                "company": "tesla",
                "year": "2024",
                "reranker_score": 0.85,
            }
        ],
    }
    with patch("tools._retrieve", return_value=mock_result):
        tool_call = _make_tool_call("retrieve", {
            "query": "What were Tesla risks in 2024?",
            "companies": ["tesla"],
            "years": ["2024"],
        })
        result = execute_tool(tool_call)
    assert "tesla" in result.lower()
    assert "2024" in result
    assert "Some financial data" in result


def test_execute_tool_raises_on_unknown_tool():
    tool_call = _make_tool_call("nonexistent_tool", {})
    try:
        execute_tool(tool_call)
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent_tool" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Write `tools.py`**

```python
import json
from retriever import retrieve as _retrieve
from config import AVAILABLE_COMPANIES, AVAILABLE_YEARS, TOP_K_RERANK_BASE, MAX_CONTEXT_CHARS


def _tool_retrieve(query: str, companies: list[str], years: list[str], top_k: int = None) -> str:
    """Run retrieval and return labeled source passages as a formatted string."""
    if top_k is None:
        top_k = TOP_K_RERANK_BASE + max(0, (len(years) - 1) * 2)

    result = _retrieve(query, companies, years, top_k)

    chunks = result["top_child_chunks"]
    parent_texts = result["parent_texts"]

    if not parent_texts:
        return "No relevant content found for this query."

    blocks = []
    for i, (text, chunk) in enumerate(zip(parent_texts, chunks)):
        company = chunk.get("company", "").capitalize()
        year = chunk.get("year", "")
        section = chunk.get("section_name", "")
        label = f"[Source: {company} {year} 10-K — {section}]"
        blocks.append(f"{label}\n{text[:MAX_CONTEXT_CHARS]}")

    return "\n\n---\n\n".join(blocks)


def _tool_get_document_index() -> str:
    return json.dumps({
        "companies": AVAILABLE_COMPANIES,
        "years": AVAILABLE_YEARS,
    })


_TOOL_FUNCTIONS = {
    "retrieve": lambda args: _tool_retrieve(**args),
    "get_document_index": lambda args: _tool_get_document_index(),
}


def execute_tool(tool_call) -> str:
    """Dispatch a tool call object to the matching function and return the string result."""
    name = tool_call.function.name
    if name not in _TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    args = json.loads(tool_call.function.arguments)
    return _TOOL_FUNCTIONS[name](args)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "feat: tool implementations with execute_tool dispatcher"
```

---

## Task 9: `agent.py` + `tests/test_agent.py`

**Files:**
- Create: `agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write `tests/test_agent.py`**

```python
from unittest.mock import patch, MagicMock
import json
import pytest
from agent import run_agent, MaxIterationsError


def _make_response(content: str = None, tool_calls: list = None):
    """Build a mock OpenAI response object."""
    response = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    response.choices = [MagicMock(message=msg)]
    return response


def _make_tool_call(name: str, args: dict, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def test_agent_returns_answer_when_no_tool_calls():
    final_response = _make_response(content="Tesla revenue was $97B.")
    with patch("agent.client.chat.completions.create", return_value=final_response):
        with patch("agent.execute_tool") as mock_tool:
            result = run_agent("What was Tesla revenue?")
    assert result["answer"] == "Tesla revenue was $97B."
    assert result["total_tool_calls"] == 0
    mock_tool.assert_not_called()


def test_agent_calls_tool_and_appends_result():
    tool_call = _make_tool_call("get_document_index", {})
    tool_response = _make_response(tool_calls=[tool_call])
    final_response = _make_response(content="Tesla, GM, Ford are available.")

    with patch("agent.client.chat.completions.create", side_effect=[tool_response, final_response]):
        with patch("agent.execute_tool", return_value='{"companies": ["tesla"]}') as mock_tool:
            result = run_agent("What companies are available?")

    assert result["total_tool_calls"] == 1
    assert result["agent_trace"][0]["tool"] == "get_document_index"
    mock_tool.assert_called_once()


def test_agent_raises_after_max_iterations():
    tool_call = _make_tool_call("get_document_index", {})
    always_tool = _make_response(tool_calls=[tool_call])

    with patch("agent.client.chat.completions.create", return_value=always_tool):
        with patch("agent.execute_tool", return_value="result"):
            with pytest.raises(MaxIterationsError):
                run_agent("Infinite loop question")


def test_agent_passes_history_to_messages():
    final_response = _make_response(content="GM revenue grew.")
    history = [
        {"role": "user", "content": "Tell me about GM."},
        {"role": "assistant", "content": "GM is a car company."},
    ]
    captured_messages = []

    def capture_call(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return final_response

    with patch("agent.client.chat.completions.create", side_effect=capture_call):
        run_agent("How did GM revenue grow?", history=history)

    roles = [m["role"] for m in captured_messages]
    assert "system" in roles
    assert roles.count("user") >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Write `agent.py`**

```python
import json
import time
from openai import OpenAI
from tool_schemas import TOOL_SCHEMAS
from tools import execute_tool
from config import OPENAI_API_KEY, GENERATION_MODEL, MAX_ITERATIONS

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant with access to 10-K filings for Tesla, GM, and Ford (2022–2025). "
    "Use the retrieve tool to search for relevant content. "
    "For multi-company or multi-year questions, call retrieve separately for each company or year you need. "
    "Call get_document_index if you are unsure which companies or years are available. "
    "Once you have enough context, write a comprehensive answer. "
    "Every sentence that states a fact or number must end with a citation in this exact format: "
    "*(Company Year 10-K — Section Name)*. Use the label from the source block the fact came from. "
    "If retrieved context does not contain enough information to answer, say so clearly."
)


class MaxIterationsError(Exception):
    pass


def run_agent(question: str, history: list[dict] | None = None) -> dict:
    """Run the ReAct agent loop for one question.

    Returns dict with answer, agent_trace, total_tool_calls, timing_ms.
    """
    start = time.time()
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    trace = []

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return {
                "answer": msg.content,
                "agent_trace": trace,
                "total_tool_calls": len(trace),
                "timing_ms": int((time.time() - start) * 1000),
            }

        messages.append(msg)

        for tool_call in msg.tool_calls:
            result = execute_tool(tool_call)
            args = json.loads(tool_call.function.arguments)
            trace.append({
                "tool": tool_call.function.name,
                "args": args,
                "result": result[:500],  # truncate for log readability
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    raise MaxIterationsError(f"Agent did not produce a final answer within {MAX_ITERATIONS} iterations.")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: ReAct agent loop with MAX_ITERATIONS=5 and tool dispatch"
```

---

## Task 10: `logger.py`

**Files:**
- Create: `logger.py`

- [ ] **Step 1: Write `logger.py`**

```python
import json
import os
from datetime import datetime
from config import LOGS_DIR


def log_query(data: dict) -> str:
    """Print agent trace to console and write full trace to a JSON file.

    Returns the path to the JSON log file.
    """
    question = data.get("question", "")
    answer = data.get("answer", "")
    trace = data.get("agent_trace", [])
    total_calls = data.get("total_tool_calls", 0)
    timing = data.get("timing_ms", 0)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    if trace:
        print(f"\nAGENT TRACE ({total_calls} tool calls):")
        for i, step in enumerate(trace, 1):
            print(f"\n  [{i}] {step['tool']}")
            args = step.get("args", {})
            if args:
                print(f"      args: {json.dumps(args)}")
            result_preview = str(step.get("result", ""))[:120]
            print(f"      result: {result_preview}...")

    print(f"\nANSWER:\n{answer}")
    print(f"\nTiming: {timing}ms")

    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Full trace → {log_path}")
    print("=" * 60 + "\n")
    return log_path
```

- [ ] **Step 2: Commit**

```bash
git add logger.py
git commit -m "feat: logger with agent trace console output and JSON file"
```

---

## Task 11: `query.py`

**Files:**
- Create: `query.py`

- [ ] **Step 1: Write `query.py`**

```python
import sys
from agent import run_agent
from logger import log_query


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python3 query.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    result = run_agent(question)

    log_query({
        "question": question,
        "answer": result["answer"],
        "agent_trace": result["agent_trace"],
        "total_tool_calls": result["total_tool_calls"],
        "timing_ms": result["timing_ms"],
    })


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a smoke test**

```bash
python3 query.py "What companies and years are available in this corpus?"
```

Expected: agent calls `get_document_index`, returns an answer listing Tesla/GM/Ford and 2022–2025, prints trace and writes JSON log.

- [ ] **Step 3: Commit**

```bash
git add query.py
git commit -m "feat: single-question entry point"
```

---

## Task 12: `chat.py`

**Files:**
- Create: `chat.py`

- [ ] **Step 1: Write `chat.py`**

```python
from collections import deque
from agent import run_agent
from logger import log_query


def main() -> None:
    # 3 turns × (user + assistant) = 6 messages max in sliding window
    history: deque[dict] = deque(maxlen=6)

    print("Financial RAG — Tesla, GM, Ford 10-K filings (2022–2025)")
    print("Commands: 'exit' to quit | 'clear' to reset history\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "clear":
            history.clear()
            print("History cleared.\n")
            continue

        result = run_agent(question, history=list(history))

        print(f"\nAssistant: {result['answer']}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

        log_query({
            "question": question,
            "answer": result["answer"],
            "agent_trace": result["agent_trace"],
            "total_tool_calls": result["total_tool_calls"],
            "timing_ms": result["timing_ms"],
            "history_length": len(history) // 2,
        })


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a smoke test**

```bash
python3 chat.py
```

Ask two questions:
1. `What were Tesla's main risks in 2024?`
2. `Which of those were new compared to 2023?`

Expected: second question resolved using first answer in history, no explicit year needed in second question.

- [ ] **Step 3: Commit**

```bash
git add chat.py
git commit -m "feat: conversational loop with sliding window history"
```

---

## Run All Tests

After all tasks are complete:

```bash
pytest tests/ -v
```

Expected: all 22 tests PASS (8 chunker + 7 retriever + 3 tools + 4 agent).
