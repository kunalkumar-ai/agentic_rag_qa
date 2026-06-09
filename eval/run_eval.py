"""Eval harness for the agentic RAG pipeline.

Loads eval/golden_dataset.jsonl, runs each question through run_agent,
captures metrics, and writes a timestamped results JSON to eval/results/.

Metrics (all computable without chunk-id labeling):
  - routing:       agent called retrieve with the expected companies and years
  - fact_coverage: must_mention strings appear in the answer
  - refusal:       for should_refuse=true items, the answer is a refusal
  - faithfulness:  every citation in the answer maps to a chunk that was actually retrieved
  - latency / iterations / errors

Stubbed (TODO):
  - correctness:   LLM-as-judge against ground_truth_answer
  - cost / tokens: needs token instrumentation in agent.py

Usage:
    python3 -m eval.run_eval
    python3 -m eval.run_eval --limit 3
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent import run_agent, MaxIterationsError  # noqa: E402
from eval.llm_judge import judge  # noqa: E402


# ---------- dataset ----------

def load_dataset(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


# ---------- per-metric scorers ----------

def score_routing(trace: list[dict], expected_companies: list[str], expected_years: list[str]) -> dict:
    """Did the agent call retrieve with the expected companies and years (union across calls)?

    For out-of-corpus questions with empty expected_*, score is n/a.
    """
    if not expected_companies and not expected_years:
        return {"score": None, "called_companies": [], "called_years": []}

    called_companies: set[str] = set()
    called_years: set[str] = set()
    for step in trace:
        if step.get("tool") != "retrieve":
            continue
        args = step.get("args") or {}
        for c in args.get("companies") or []:
            called_companies.add(str(c).lower())
        for y in args.get("years") or []:
            called_years.add(str(y))

    exp_c = {c.lower() for c in expected_companies}
    exp_y = {str(y) for y in expected_years}

    company_hit = exp_c.issubset(called_companies) if exp_c else True
    year_hit = exp_y.issubset(called_years) if exp_y else True

    return {
        "score": 1.0 if (company_hit and year_hit) else 0.0,
        "called_companies": sorted(called_companies),
        "called_years": sorted(called_years),
        "missing_companies": sorted(exp_c - called_companies),
        "missing_years": sorted(exp_y - called_years),
    }




_CITATION_RE = re.compile(r"\*\(([^)]+?)\)\*")


def _parse_citation(text: str) -> dict | None:
    """Parse '*(Tesla 2024 10-K — Risk Factors)*' → {company, year, section}."""
    inner = text.strip()
    if "10-K" not in inner:
        return None
    head, _, tail = inner.partition("10-K")
    head = head.strip()
    parts = head.split()
    if len(parts) < 2:
        return None
    company = parts[0].lower()
    year = parts[1]
    section = tail.lstrip(" —–-").strip()
    return {"company": company, "year": year, "section": section}


def score_faithfulness(answer: str | None, trace: list[dict]) -> dict:
    """Every citation in the answer should point to a chunk that was actually retrieved.

    We match on (company, year) — section name matching is too brittle since the
    LLM may paraphrase it. If even (company, year) wasn't retrieved, the citation
    is hallucinated.
    """
    if not answer:
        return {"score": None, "n_citations": 0, "hallucinated": []}

    citations_raw = _CITATION_RE.findall(answer)
    parsed = [c for c in (_parse_citation(c) for c in citations_raw) if c]
    if not parsed:
        return {"score": None, "n_citations": 0, "hallucinated": []}

    retrieved_pairs: set[tuple[str, str]] = set()
    for step in trace:
        for chunk in step.get("chunks_metadata") or []:
            c = str(chunk.get("company", "")).lower()
            y = str(chunk.get("year", ""))
            if c and y:
                retrieved_pairs.add((c, y))

    hallucinated = []
    for cit in parsed:
        if (cit["company"], cit["year"]) not in retrieved_pairs:
            hallucinated.append(cit)

    n = len(parsed)
    return {
        "score": (n - len(hallucinated)) / n,
        "n_citations": n,
        "hallucinated": hallucinated,
    }


# ---------- per-question runner ----------

def evaluate_one(item: dict) -> dict:
    question = item["question"]
    record = {
        "id": item["id"],
        "question": question,
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "expected_companies": item.get("expected_companies", []),
        "expected_years": item.get("expected_years", []),
        "should_refuse": item.get("should_refuse", False),
        "ground_truth_answer": item.get("ground_truth_answer"),
        "must_mention": item.get("must_mention", []),
    }

    start = time.time()
    try:
        result = run_agent(question)
        record["answer"] = result["answer"]
        record["agent_trace"] = result["agent_trace"]
        record["total_tool_calls"] = result["total_tool_calls"]
        record["timing_ms"] = result["timing_ms"]
        record["iterations_used"] = len(result["agent_trace"])
        record["max_iterations_hit"] = False
        record["error"] = None
    except MaxIterationsError as e:
        record.update({
            "answer": None, "agent_trace": [], "total_tool_calls": 0,
            "timing_ms": int((time.time() - start) * 1000),
            "iterations_used": None, "max_iterations_hit": True, "error": str(e),
        })
    except Exception as e:
        record.update({
            "answer": None, "agent_trace": [], "total_tool_calls": 0,
            "timing_ms": int((time.time() - start) * 1000),
            "iterations_used": None, "max_iterations_hit": False,
            "error": f"{type(e).__name__}: {e}",
        })

    record["routing"] = score_routing(record["agent_trace"], record["expected_companies"], record["expected_years"])
    record["faithfulness"] = score_faithfulness(record["answer"], record["agent_trace"])

    llm_scores = judge(
        question=question,
        answer=record["answer"],
        ground_truth=record.get("ground_truth_answer", ""),
        must_mention=record["must_mention"],
        should_refuse=record["should_refuse"],
    )
    record["fact_coverage"] = llm_scores["fact_coverage"]
    record["refusal"] = llm_scores["refusal"] if record["should_refuse"] else {"score": None, "reasoning": None}
    record["correctness"] = llm_scores["correctness"]
    # TODO (phase 3): instrument token usage in agent.py.
    record["cost_usd"] = None
    record["input_tokens"] = None
    record["output_tokens"] = None

    return record


# ---------- aggregation ----------

def _mean(values: list[float]) -> float | None:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


def aggregate(records: list[dict]) -> dict:
    timings = sorted([r["timing_ms"] for r in records if r["timing_ms"]])

    def pct(p: float) -> int | None:
        if not timings:
            return None
        idx = min(len(timings) - 1, int(round(p * (len(timings) - 1))))
        return timings[idx]

    by_category: dict[str, dict] = {}
    for r in records:
        cat = r.get("category") or "uncategorized"
        bucket = by_category.setdefault(cat, {"count": 0, "errors": 0, "routing": [], "fact_coverage": [], "refusal": [], "faithfulness": [], "correctness": []})
        bucket["count"] += 1
        if r["error"]:
            bucket["errors"] += 1
        bucket["routing"].append(r["routing"]["score"])
        bucket["fact_coverage"].append(r["fact_coverage"]["score"])
        bucket["refusal"].append(r["refusal"]["score"])
        bucket["faithfulness"].append(r["faithfulness"]["score"])
        bucket["correctness"].append(r["correctness"]["score"])

    for cat, b in by_category.items():
        b["routing_mean"] = _mean(b.pop("routing"))
        b["fact_coverage_mean"] = _mean(b.pop("fact_coverage"))
        b["refusal_mean"] = _mean(b.pop("refusal"))
        b["faithfulness_mean"] = _mean(b.pop("faithfulness"))
        b["correctness_mean"] = _mean(b.pop("correctness"))

    return {
        "n_questions": len(records),
        "n_errors": sum(1 for r in records if r["error"]),
        "n_max_iterations_hit": sum(1 for r in records if r["max_iterations_hit"]),
        "routing_mean": _mean([r["routing"]["score"] for r in records]),
        "fact_coverage_mean": _mean([r["fact_coverage"]["score"] for r in records]),
        "refusal_mean": _mean([r["refusal"]["score"] for r in records]),
        "faithfulness_mean": _mean([r["faithfulness"]["score"] for r in records]),
        "correctness_mean": _mean([r["correctness"]["score"] for r in records]),
        "latency_ms_p50": pct(0.5),
        "latency_ms_p95": pct(0.95),
        "by_category": by_category,
    }


# ---------- entrypoint ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(REPO_ROOT / "eval" / "golden_dataset.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args()

    items = load_dataset(Path(args.dataset))
    if args.limit:
        items = items[: args.limit]

    print(f"Running eval on {len(items)} questions from {args.dataset}")
    records = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['question'][:80]}")
        records.append(evaluate_one(item))

    summary = aggregate(records)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.results_dir) / f"{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
