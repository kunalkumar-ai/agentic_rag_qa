import json
from retriever import retrieve as _retrieve
from config import AVAILABLE_COMPANIES, AVAILABLE_YEARS, TOP_K_RERANK_BASE, MAX_CONTEXT_CHARS


def _tool_retrieve(query: str, companies: list[str], years: list[str], top_k: int = None) -> tuple[str, list[dict]]:
    """Run retrieval and return labeled source passages + chunk metadata.

    Returns:
        context: formatted string passed to the agent
        chunks_metadata: list of dicts with section, company, year, reranker_score per chunk
    """
    if top_k is None:
        top_k = TOP_K_RERANK_BASE + max(0, (len(years) - 1) * 2)

    result = _retrieve(query, companies, years, top_k)

    parent_texts = result["parent_texts"]
    chunks = result["top_child_chunks"]

    if not parent_texts:
        return "No relevant content found for this query.", []

    blocks = []
    chunks_metadata = []
    for text, chunk in zip(parent_texts, chunks):
        company = chunk.get("company", "").capitalize()
        year = chunk.get("year", "")
        section = chunk.get("section_name", "")
        label = f"[Source: {company} {year} 10-K — {section}]"
        blocks.append(f"{label}\n{text[:MAX_CONTEXT_CHARS]}")
        chunks_metadata.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "parent_id": chunk.get("parent_id", ""),
            "section_name": section,
            "company": chunk.get("company", ""),
            "year": year,
            "reranker_score": round(chunk.get("reranker_score", 0.0), 4),
        })

    return "\n\n---\n\n".join(blocks), chunks_metadata


def _tool_get_document_index() -> tuple[str, list]:
    return json.dumps({
        "companies": AVAILABLE_COMPANIES,
        "years": AVAILABLE_YEARS,
    }), []


_TOOL_FUNCTIONS = {
    "retrieve": lambda args: _tool_retrieve(**args),
    "get_document_index": lambda args: _tool_get_document_index(),
}


def execute_tool(tool_call) -> tuple[str, list[dict]]:
    """Dispatch a tool call object to the matching function.

    Returns:
        context: string result passed to the agent
        metadata: chunk metadata list (empty for non-retrieve tools)
    """
    name = tool_call.function.name
    if name not in _TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    args = json.loads(tool_call.function.arguments)
    return _TOOL_FUNCTIONS[name](args)
