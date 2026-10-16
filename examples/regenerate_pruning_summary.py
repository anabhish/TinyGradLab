
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

with open("pruning_trials.csv", newline="") as file:
    trials = list(csv.DictReader(file))

groups = defaultdict(list)
for row in trials:
    key = (row["method"], float(row["target_sparsity"]))
    groups[key].append(row)

expected_methods = [
    "one-shot-magnitude",
    "progressive-magnitude",
    "one-shot-taylor",
    "progressive-taylor",
]
targets = [0.0, 0.50, 0.75, 0.90, 0.95]

summary = []
for target in targets:
    for method in expected_methods:
        rows = groups[(method, target)]
        if len(rows) != 5:
            raise ValueError(
                f"Expected 5 trials for {method} at {target}, "
                f"found {len(rows)}"
            )

        accuracy = np.array([float(r["accuracy"]) for r in rows])
        loss = np.array([float(r["loss"]) for r in rows])
        sparsity = np.array(
            [float(r["actual_sparsity"]) for r in rows]
        )

        summary.append({
            "method": method,
            "target_sparsity": target,
            "mean_accuracy": float(accuracy.mean()),
            "std_accuracy": float(accuracy.std()),
            "mean_loss": float(loss.mean()),
            "mean_sparsity": float(sparsity.mean()),
        })

with open("pruning_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

plt.figure()
for method in expected_methods:
    rows = sorted(
        [r for r in summary if r["method"] == method],
        key=lambda r: r["mean_sparsity"],
    )
    plt.plot(
        [r["mean_sparsity"] * 100 for r in rows],
        [r["mean_accuracy"] * 100 for r in rows],
        marker="o",
        label=method,
    )

plt.xlabel("Actual sparsity (%)")
plt.ylabel("Mean training accuracy (%)")
plt.title("Pruning Pareto Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("pruning_pareto.png", dpi=150)
plt.close()

print(f"Trials verified: {len(trials)}")
print(f"Summary rows written: {len(summary)}")
print("Regenerated pruning_results.csv and pruning_pareto.png")