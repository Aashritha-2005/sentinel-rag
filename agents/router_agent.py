from pydantic import BaseModel
from typing import Literal

class RouteDecision(BaseModel):
    datasource: Literal["vectorstore", "web_search"]
    reason: str


class RouterAgent:
    def __init__(self, llm):
        self.llm = llm

    def decide(self, question: str) -> RouteDecision:
        prompt = f"""
You are a routing agent.

Decide the best datasource and explain WHY.

Rules:
- Use vectorstore if the question is about AI, ML, agents, memory, or concepts.
- Use web_search if the question is about recent events or real-time facts.

Return JSON with:
- datasource
- reason

Question: {question}
"""

        response = self.llm.invoke(prompt)

        # Very simple parsing (safe for demo)
        text = response.lower()

        if "web" in text:
            return RouteDecision(
                datasource="web_search",
                reason="Query appears to require external or recent information."
            )

        return RouteDecision(
            datasource="vectorstore",
            reason="Query asks about conceptual AI topics present in internal knowledge."
        )
