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
