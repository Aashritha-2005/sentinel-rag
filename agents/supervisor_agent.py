class SupervisorAgent:
    def decide(self, grounded: bool, retries: int, max_retries: int):
        if grounded:
            return "accept"
        if retries >= max_retries:
            return "force_accept"
        return "retry"
