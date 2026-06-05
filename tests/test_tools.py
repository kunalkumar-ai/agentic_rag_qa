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
    context, metadata = execute_tool(tool_call)
    data = json.loads(context)
    assert "companies" in data
    assert "years" in data
    assert "tesla" in data["companies"]
    assert "gm" in data["companies"]
    assert "ford" in data["companies"]
    assert "2024" in data["years"]
    assert metadata == []


def test_retrieve_returns_formatted_string_and_metadata():
    mock_result = {
        "parent_texts": ["Some financial data here about revenue growth."],
        "top_child_chunks": [
            {
                "chunk_id": "tesla_2024_child_0_0_0",
                "parent_id": "tesla_2024_parent_0_0",
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
        context, metadata = execute_tool(tool_call)
    assert "tesla" in context.lower()
    assert "2024" in context
    assert "Some financial data" in context
    assert len(metadata) == 1
    assert metadata[0]["company"] == "tesla"
    assert metadata[0]["year"] == "2024"
    assert metadata[0]["reranker_score"] == 0.85
    assert metadata[0]["section_name"] == "Risk Factors"


def test_execute_tool_raises_on_unknown_tool():
    tool_call = _make_tool_call("nonexistent_tool", {})
    try:
        execute_tool(tool_call)
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent_tool" in str(e)
