import time
from langchain_ollama import OllamaLLM

# =========================
# IMPORT YOUR AGENTIC SYSTEM
# =========================
from app import app   # LangGraph app


# =========================
# EVALUATION DATASET
# =========================
EVAL_QUESTIONS = [
    # In-domain (should answer)
    {"question": "What is agent memory in multi-agent systems?", "expected": "answer"},
    {"question": "Why is memory important for agent coordination?", "expected": "answer"},

    # Out-of-domain (should refuse)
    {"question": "Who is the PM of India?", "expected": "refuse"},
    {"question": "What is the capital of Germany?", "expected": "refuse"},

    # Ambiguous
    {"question": "Explain memory in agents.", "expected": "answer"},
]


# =========================
# BASELINE (RAW LLM)
# =========================
baseline_llm = OllamaLLM(model="mistral")

def run_baseline(question: str):
    start = time.time()
    answer = baseline_llm.invoke(question)
    latency = time.time() - start
    return answer, latency


# =========================
# AGENTIC SYSTEM
# =========================
def run_agentic(question: str):
    start = time.time()
    result = app.invoke({
        "question": question,
        "retries": 0
    })
    latency = time.time() - start
    return result, latency


# =========================
# BEHAVIOR CLASSIFICATION
# =========================
def classify_behavior(answer: str):
    refusal_phrases = [
        "don't have enough information",
        "cannot answer",
        "not enough information",
        "unable to answer"
    ]
    answer_lower = answer.lower()
    return "refuse" if any(p in answer_lower for p in refusal_phrases) else "answer"


# =========================
# EVALUATION LOOP
# =========================
def evaluate():
    baseline_stats = {"correct": 0, "hallucinations": 0, "latencies": []}
    agentic_stats = {"correct": 0, "hallucinations": 0, "latencies": []}

    for item in EVAL_QUESTIONS:
        q = item["question"]
        expected = item["expected"]

        # ----- BASELINE -----
        base_ans, base_lat = run_baseline(q)
        base_behavior = classify_behavior(base_ans)

        if base_behavior == expected:
            baseline_stats["correct"] += 1
        else:
            baseline_stats["hallucinations"] += 1

        baseline_stats["latencies"].append(base_lat)

        # ----- AGENTIC -----
        agent_res, agent_lat = run_agentic(q)
        agent_ans = agent_res["answer"]
        agent_behavior = classify_behavior(agent_ans)

        if agent_behavior == expected:
            agentic_stats["correct"] += 1
        else:
            agentic_stats["hallucinations"] += 1

        agentic_stats["latencies"].append(agent_lat)

    return baseline_stats, agentic_stats


# =========================
# SUMMARY
# =========================
def summarize(stats, total):
    return {
        "accuracy": round(stats["correct"] / total, 2),
        "hallucination_rate": round(stats["hallucinations"] / total, 2),
        "avg_latency_sec": round(sum(stats["latencies"]) / len(stats["latencies"]), 2)
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    baseline, agentic = evaluate()
    total = len(EVAL_QUESTIONS)

    base_summary = summarize(baseline, total)
    agent_summary = summarize(agentic, total)

    print("\n========== EVALUATION RESULTS ==========\n")

    print("Baseline (Raw Mistral):")
    print(base_summary)

    print("\nAgentic RAG System:")
    print(agent_summary)

    print("\n========================================\n")
