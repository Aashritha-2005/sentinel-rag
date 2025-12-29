class GeneratorAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question, documents):
        # 🚨 CRITICAL SAFETY CHECK
        if not documents or len(documents) == 0:
            return (
                "I don't have enough information in my knowledge base "
                "to answer this question."
            )

        context = "\n".join(
            d.get("content", "") if isinstance(d, dict) else str(d)
            for d in documents
        )

        prompt = f"""
Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question:
{question}
"""

        return self.llm.invoke(prompt)
