class RetrieverAgent:
    def __init__(self, vectorstore, web_tool=None):
        self.vectorstore = vectorstore
        self.web_tool = web_tool

    def retrieve(self, question: str, source: str):
        if source == "vectorstore":
            return self.vectorstore.invoke(question)

        # Offline-safe fallback
        if source == "web_search":
            if self.web_tool is None:
                return []   # No documents → evaluator will reject hallucination
            return self.web_tool.invoke({"query": question})

        return []

