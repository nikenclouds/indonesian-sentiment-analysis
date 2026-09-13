import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# DATA MODEL
# =========================

models = [
    "Naive Bayes",
    "Logistic Regression",
    "SVM"
]

accuracy = np.array([
    0.8300,
    0.8732,
    0.8718
])

macro_f1 = np.array([
    0.75,
    0.84,
    0.84
])

# =========================
# PENGATURAN
# =========================

x = np.arange(len(models))
width = 0.35

# Burgundy palette
color_accuracy = "#8B1E3F"
color_f1 = "#D98FA3"

Path("Results").mkdir(exist_ok=True)

# =========================
# CREATE CHART
# =========================

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2,
    accuracy,
    width,
    label="Accuracy",
    color=color_accuracy
)

bars2 = ax.bar(
    x + width / 2,
    macro_f1,
    width,
    label="Macro F1-Score",
    color=color_f1
)

# =========================
# TITLE & LABEL
# =========================

ax.set_title(
    "Machine Learning Model Performance Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_ylabel(
    "Performance Score",
    fontsize=11
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0.65, 0.95)

# =========================
# VALUE LABELS
# =========================

for bar in bars1:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.1%}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

for bar in bars2:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.1%}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# =========================
# GRID & LEGEND
# =========================

ax.grid(
    axis="y",
    alpha=0.2
)

ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    loc="upper right"
)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# =========================
# SAVE
# =========================

plt.savefig(
    "Results/model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDiagram berhasil dibuat!")
print("Saved to: Results/model_performance_comparison.png")