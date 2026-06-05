import json
from retriever import retrieve as _retrieve
from config import AVAILABLE_COMPANIES, AVAILABLE_YEARS, TOP_K_RERANK_BASE, MAX_CONTEXT_CHARS


def _tool_retrieve(query: str, companies: list[str], years: list[str], top_k: int = None) -> str:
    """Run retrieval and return labeled source passages as a formatted string."""
    if top_k is None:
        top_k = TOP_K_RERANK_BASE + max(0, (len(years) - 1) * 2)

    result = _retrieve(query, companies, years, top_k)

    parent_texts = result["parent_texts"]
    chunks = result["top_child_chunks"]

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
