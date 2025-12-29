import json
from datetime import datetime

LOG_FILE = "evaluation/runs.jsonl"

def log_run(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
