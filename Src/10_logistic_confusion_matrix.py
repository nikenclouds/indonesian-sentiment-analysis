import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# =========================
# 1. MEMBACA DATASET
# =========================

df = pd.read_csv(
    "Data/train_preprocess_ori.tsv",
    sep="\t"
)

# =========================
# 2. CLEANING TEXT
# =========================

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["clean_text"] = df["text"].apply(clean_text)

# =========================
# 3. INPUT DAN TARGET
# =========================

X = df["clean_text"]
y = df["sentiment"]

# =========================
# 4. SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. TF-IDF
# =========================

tfidf = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# 6. LOGISTIC REGRESSION
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

# =========================
# 7. PREDIKSI
# =========================

y_pred = model.predict(X_test_tfidf)

# =========================
# 8. CONFUSION MATRIX
# =========================

labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("=== CONFUSION MATRIX ===")
print(cm)

# =========================
# 9. WARNA BURGUNDY
# =========================

from matplotlib.colors import LinearSegmentedColormap

burgundy = LinearSegmentedColormap.from_list(
    "burgundy",
    ["#F5E6E8", "#8B1E3F", "#4A0E22"]
)

# =========================
# 10. MEMBUAT DIAGRAM
# =========================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=burgundy,
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

# =========================
# 11. SIMPAN DIAGRAM
# =========================

plt.savefig(
    "Results/logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDiagram berhasil disimpan!")
print("Results/logistic_regression_confusion_matrix.png")