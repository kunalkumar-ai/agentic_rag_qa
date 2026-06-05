from unittest.mock import patch, MagicMock
import json
import pytest
from agent import run_agent, MaxIterationsError


def _make_response(content: str = None, tool_calls: list = None):
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
