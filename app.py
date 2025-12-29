import time
from langchain_ollama import OllamaLLM

from agents.router_agent import RouterAgent
from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.supervisor_agent import SupervisorAgent

from evaluation.metrics import log_run
from graph import build_graph


# =========================
# CONFIG
# =========================
OLLAMA_MODEL = "mistral"


# =========================
# LOAD LOCAL LLM
# =========================
llm = OllamaLLM(model=OLLAMA_MODEL)


# =========================
# DUMMY DOCUMENTS (TEMP)
# =========================
DUMMY_DOCS = [
    {
        "content": (
            "Agent memory in multi-agent systems refers to mechanisms that allow "
            "agents to store, retrieve, and share information across interactions. "
            "Memory can be short-term, long-term, or shared among agents to support "
            "coordination, reasoning, and learning."
        )
    }
]


# =========================
# SIMPLE RETRIEVER FUNCTION
# =========================
class SimpleRetriever:
    def invoke(self, query: str):
        return DUMMY_DOCS


# =========================
# AGENT INITIALIZATION
# =========================
router_agent = RouterAgent(llm)

retriever_agent = RetrieverAgent(
    vectorstore=SimpleRetriever(),
    web_tool=None
)

generator_agent = GeneratorAgent(llm)
evaluator_agent = EvaluatorAgent(llm)
supervisor_agent = SupervisorAgent()


# =========================
# BUILD AGENTIC GRAPH
# =========================
app = build_graph(
    router_agent=router_agent,
    retriever_agent=retriever_agent,
    generator_agent=generator_agent,
    evaluator_agent=evaluator_agent,
    supervisor_agent=supervisor_agent,
    log_fn=log_run,
    max_retries=2
)


# =========================
# RUN QUERY
# =========================
def run_query(question: str):
    start = time.time()

    result = app.invoke({
        "question": question,
        "retries": 0
    })

    latency = round(time.time() - start, 2)

    print("\n================ ANSWER ================\n")
    print(result["answer"])
    print("\n=======================================\n")

    print(f"Latency  : {latency}s")
    print(f"Grounded : {result['grounded']}")
    print(f"Retries  : {result['retries']}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    while True:
        question = input("\nAsk your question (type 'exit' to quit): ").strip()

        if not question:
            continue   # ⛔ ignore empty input

        if question.lower() in ["exit", "quit"]:
            break

        run_query(question)



