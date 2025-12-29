import json
import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "evaluation/runs.jsonl"

records = []
with open(LOG_FILE) as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# 1. Grounded vs Hallucinated
df["grounded"].value_counts().plot(
    kind="bar",
    title="Grounded vs Hallucinated Answers"
)
plt.show()

# 2. Retry distribution
df["retries"].value_counts().sort_index().plot(
    kind="bar",
    title="Retry Distribution"
)
plt.show()

# 3. Route usage
df["route"].value_counts().plot(
    kind="bar",
    title="Routing Decisions"
)
plt.show()
