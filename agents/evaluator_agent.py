class EvaluatorAgent:
    def __init__(self, llm):
        self.llm = llm

    def is_grounded(self, answer, documents):
        context = "\n".join(
            d.page_content if hasattr(d, "page_content") else d["content"]
            for d in documents
        )
        prompt = f"""
Is the answer fully supported by the context?

Context:
{context}

Answer:
{answer}

Reply yes or no.
"""
        return "yes" in self.llm.invoke(prompt).lower()
