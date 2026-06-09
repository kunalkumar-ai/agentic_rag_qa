"""LLM-as-judge for eval metrics.

Single GPT-4o-mini call per question evaluates all three metrics at once:
  - refused:       did the answer refuse / state info is unavailable?
  - fact_coverage: which must_mention facts are present (semantically)?
  - correctness:   how correct is the answer vs ground truth? (1-5)

Use:
    from eval.llm_judge import judge
    result = judge(question, answer, ground_truth, must_mention)
"""

import json
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
JUDGE_MODEL = "gpt-4o-mini"

_SYSTEM = """You are an eval judge for a financial document QA system.
Given a question, the system's answer, a ground truth answer, and a list of required facts, evaluate three things in a single pass:

1. REFUSAL — Did the answer refuse to answer or say the information is unavailable?
   A refusal includes: saying the company/year is not in the corpus, saying it cannot provide real-time data, saying the question is out of scope, or any explicit acknowledgement it cannot answer.
   A partial answer that also flags a limitation counts as a refusal.

2. FACT COVERAGE — Which required facts are present in the answer, even if worded differently?
   Be generous with matching: "19.4 percent" matches "19.4%", "$96 billion" matches "96,773 million", "supply chain disruptions" matches "supply chain", "Ford Blue segment" matches "Ford Blue".

3. CORRECTNESS — How correct is the answer vs the ground truth? Use this rubric:
   5 = Fully correct, all key facts accurate, nothing contradicted
   4 = Mostly correct, minor omissions or imprecise wording
   3 = Partially correct, main point present but significant gaps or one wrong fact
   2 = Mostly wrong or off-topic, only one correct element
   1 = Completely wrong, irrelevant, or refuses when it should answer

Respond with this exact JSON structure:
{
  "refused": true or false,
  "fact_hits": ["fact1", "fact2"],
  "fact_misses": ["fact3"],
  "correctness_score": 1-5,
  "reasoning": "one or two sentences covering all three"
}"""


def judge(
    question: str,
    answer: str | None,
    ground_truth: str,
    must_mention: list[str],
) -> dict:
    """Single LLM call that returns all three eval metrics.

    Returns:
        refusal:      {score, reasoning}
        fact_coverage:{score, hits, misses, reasoning}
        correctness:  {score, raw_score, reasoning}
    """
    if not answer:
        no_answer = "No answer produced."
        n = len(must_mention)
        return {
            "refusal": {"score": 0.0, "reasoning": no_answer},
            "fact_coverage": {"score": 0.0 if n else None, "hits": [], "misses": list(must_mention), "reasoning": no_answer},
            "correctness": {"score": 0.0, "raw_score": 1, "reasoning": no_answer},
        }

    facts_str = "\n".join(f"- {f}" for f in must_mention) if must_mention else "(none)"
    user = (
        f"Question: {question}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"System answer: {answer}\n\n"
        f"Required facts:\n{facts_str}"
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        # refusal
        refusal = {
            "score": 1.0 if result.get("refused") else 0.0,
            "reasoning": result.get("reasoning", ""),
        }

        # fact coverage
        hits = result.get("fact_hits", [])
        misses = result.get("fact_misses", [])
        n = len(must_mention)
        fact_coverage = {
            "score": len(hits) / n if n else None,
            "hits": hits,
            "misses": misses,
            "reasoning": result.get("reasoning", ""),
        }

        # correctness
        raw_score = int(result.get("correctness_score", 1))
        raw_score = max(1, min(5, raw_score))
        correctness = {
            "score": (raw_score - 1) / 4,
            "raw_score": raw_score,
            "reasoning": result.get("reasoning", ""),
        }

        return {
            "refusal": refusal,
            "fact_coverage": fact_coverage,
            "correctness": correctness,
        }

    except Exception as e:
        err = f"Judge error: {e}"
        n = len(must_mention)
        return {
            "refusal": {"score": None, "reasoning": err},
            "fact_coverage": {"score": None, "hits": [], "misses": list(must_mention), "reasoning": err},
            "correctness": {"score": None, "raw_score": None, "reasoning": err},
        }
