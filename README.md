SentinelRAG
A Safety-Aware Agentic Retrieval-Augmented Generation System
SentinelRAG is a fully local, agentic Retrieval-Augmented Generation (RAG) system designed to reduce hallucinations and enforce grounded responses in large language models.

The system introduces explicit routing, evaluation, and supervision agents to ensure safe, explainable, and reliable behavior, even in offline environments.

🔍 Motivation
Large Language Models (LLMs) often hallucinate when asked questions outside their knowledge scope.

SentinelRAG addresses this problem by:

Enforcing retrieval before generation
Explicitly refusing to answer when context is missing
Logging and supervising each decision step
Quantitatively evaluating safety improvements over a baseline LLM
The goal is not to maximize fluency, but to maximize trustworthiness.

🏗️ System Architecture
SentinelRAG is built as a multi-agent system using LangGraph:

Router Agent – Decides whether a query should be answered from internal knowledge or requires external information.
Retriever Agent – Retrieves context from the knowledge source (simulated locally in this version).
Generator Agent – Produces answers strictly conditioned on retrieved context.
Evaluator Agent – Verifies whether the generated answer is grounded in retrieved information.
Supervisor Agent – Controls retries, accepts correct answers, or enforces safe refusal.
Each step is logged for observability and evaluation.

⚙️ Key Features
Fully offline execution using local LLMs (Ollama + Mistral)
Explicit hallucination prevention
Safe refusal for out-of-domain queries
Explainable routing decisions
Modular, production-style agent design
Reproducible evaluation pipeline
📊 Evaluation & Baseline Comparison
SentinelRAG was evaluated against a raw LLM baseline (Mistral without RAG or safety checks) on a mixed question set containing:

In-domain questions (answerable from knowledge)
Out-of-domain questions (should be refused)
Ambiguous queries
📈 Results
System	Accuracy	Hallucination Rate	Avg Latency (s)
Raw Mistral (Baseline)	0.60	0.40	12.9
SentinelRAG	1.00	0.00	5.7
Key Observations
SentinelRAG achieved zero hallucinations on the evaluation set.
Out-of-domain queries were safely refused instead of guessed.
Latency improved due to early refusal and supervised execution.
These results demonstrate that agentic supervision and retrieval-aware refusal significantly improve safety and reliability.

🧪 How to Run
Setup
bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Run Interactive CLI
bash
python app.py
Ask any question directly from the terminal.

Run Evaluation
bash
python -m evaluation.evaluate
🧠 Design Principles
Safety over fluency
Refuse rather than hallucinate
Observable decision-making
Separation of concerns
Evaluation-driven development
🚀 Future Work
Replace simulated knowledge with real document collections
Add confidence scoring to responses
Expose the system via a REST API
Scale evaluation with larger benchmark datasets
📌 Disclaimer
This project focuses on system design and safety behavior, not state-of-the-art language modeling performance.

Evaluation results are reported honestly on a small, controlled dataset.

