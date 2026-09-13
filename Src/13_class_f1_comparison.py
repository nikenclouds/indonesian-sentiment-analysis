import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# DATA F1-SCORE PER CLASS
# =========================

classes = [
    "Negative",
    "Neutral",
    "Positive"
]

naive_bayes = [
    0.76,
    0.60,
    0.89
]

logistic_regression = [
    0.82,
    0.80,
    0.91
]

svm = [
    0.82,
    0.80,
    0.91
]

# =========================
# PENGATURAN
# =========================

x = np.arange(len(classes))
width = 0.25

# Burgundy palette
color_nb = "#D9A5B3"
color_lr = "#8B1E3F"
color_svm = "#4A0E22"

Path("Results").mkdir(exist_ok=True)

# =========================
# CREATE CHART
# =========================

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width,
    naive_bayes,
    width,
    label="Naive Bayes",
    color=color_nb
)

bars2 = ax.bar(
    x,
    logistic_regression,
    width,
    label="Logistic Regression",
    color=color_lr
)

bars3 = ax.bar(
    x + width,
    svm,
    width,
    label="SVM",
    color=color_svm
)

# =========================
# TITLE & LABEL
# =========================

ax.set_title(
    "F1-Score Comparison by Sentiment Class",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_ylabel(
    "F1-Score",
    fontsize=11
)

ax.set_xlabel(
    "Sentiment Class",
    fontsize=11
)

ax.set_xticks(x)
ax.set_xticklabels(classes)

ax.set_ylim(0.50, 1.00)

# =========================
# VALUE LABELS
# =========================

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

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
    loc="upper left"
)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# =========================
# SAVE
# =========================

plt.savefig(
    "Results/class_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDiagram berhasil dibuat!")
print("Saved to: Results/class_f1_comparison.png")