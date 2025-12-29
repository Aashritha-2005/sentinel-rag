from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# =========================
# Graph State Definition
# =========================
class GraphState(TypedDict):
    question: str
    route: str
    documents: List
    answer: str
    grounded: bool
    retries: int


# =========================
# Build Agentic RAG Graph
# =========================
def build_graph(
    router_agent,
    retriever_agent,
    generator_agent,
    evaluator_agent,
    supervisor_agent,
    log_fn,
    max_retries: int = 2,
):
    """
    Builds an Agentic RAG system with:
    - Autonomous routing
    - Retrieval
    - Answer generation
    - Grounding evaluation
    - Self-correction loop
    """

    # -------------------------
    # Node: Route Query
    # -------------------------
    def route_query(state: GraphState):
        decision = router_agent.decide(state["question"])
        state["route"] = decision.datasource
        state["route_reason"] = decision.reason


        log_fn({
            "stage": "route",
            "question": state["question"],
            "route": state["route"],
            "reason": state["route_reason"]
        })


        return state

    # -------------------------
    # Node: Retrieve Documents
    # -------------------------
    def retrieve_docs(state: GraphState):
        docs = retriever_agent.retrieve(
            question=state["question"],
            source=state["route"]
        )
        state["documents"] = docs

        log_fn({
            "stage": "retrieve",
            "route": state["route"],
            "documents_count": len(docs)
        })

        return state

    # -------------------------
    # Node: Generate Answer
    # -------------------------
    def generate_answer(state: GraphState):
        answer = generator_agent.generate(
            question=state["question"],
            documents=state["documents"]
        )
        state["answer"] = answer

        log_fn({
            "stage": "generate",
            "answer_preview": answer[:200]
        })

        return state

    # -------------------------
    # Node: Evaluate Answer
    # -------------------------
    def evaluate_answer(state: GraphState):
        grounded = evaluator_agent.is_grounded(
            answer=state["answer"],
            documents=state["documents"]
        )
        state["grounded"] = grounded

        log_fn({
            "stage": "evaluate",
            "grounded": grounded,
            "retries": state["retries"]
        })

        return state

    # -------------------------
    # Node: Supervisor Decision
    # -------------------------
    def supervisor_decision(state: GraphState):
        decision = supervisor_agent.decide(
            grounded=state["grounded"],
            retries=state["retries"],
            max_retries=max_retries
        )

        log_fn({
            "stage": "supervisor",
            "decision": decision,
            "retries": state["retries"]
        })

        if decision == "retry":
            state["retries"] += 1
            return "retry"

        return "accept"

    # =========================
    # LangGraph Construction
    # =========================
    graph = StateGraph(GraphState)

    graph.add_node("route", route_query)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)
    graph.add_node("evaluate", evaluate_answer)

    graph.set_entry_point("route")

    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        supervisor_decision,
        {
            "retry": "retrieve",
            "accept": END,
        },
    )

    return graph.compile()
