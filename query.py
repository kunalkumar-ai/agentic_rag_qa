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
