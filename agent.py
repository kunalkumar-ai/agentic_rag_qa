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
                "result": result[:500],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    raise MaxIterationsError(f"Agent did not produce a final answer within {MAX_ITERATIONS} iterations.")
