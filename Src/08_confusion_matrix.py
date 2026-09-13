import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

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
# 3. SPLIT DATA
# =========================

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. TF-IDF
# =========================

tfidf = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# 5. NAIVE BAYES
# =========================

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

# =========================
# 6. PREDIKSI
# =========================

y_pred = model.predict(X_test_tfidf)

# =========================
# 7. CONFUSION MATRIX
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
# 8. MEMBUAT DIAGRAM
# =========================

plt.figure(figsize=(8, 6))

burgundy = LinearSegmentedColormap.from_list(
    "burgundy",
    ["#F5E6E8", "#8B1E3F", "#4A0E22"]
)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="RdPu",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

# =========================
# 9. SIMPAN HASIL
# =========================

plt.savefig(
    "Results/naive_bayes_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDiagram berhasil disimpan!")
print("Results/naive_bayes_confusion_matrix.png")