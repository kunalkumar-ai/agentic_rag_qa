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
