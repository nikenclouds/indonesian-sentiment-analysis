import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

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
# 8. EVALUASI
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy)

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_test,
        y_pred
    )
)
# Membuat folder Results
Path("Results").mkdir(exist_ok=True)

# Menyimpan hasil evaluasi
report = classification_report(y_test, y_pred)

with open("Results/logistic_regression_report.txt", "w") as file:
    file.write("LOGISTIC REGRESSION - SENTIMENT ANALYSIS\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write("CLASSIFICATION REPORT\n")
    file.write("=" * 45 + "\n")
    file.write(report)

print("\nHasil berhasil disimpan:")
print("Results/logistic_regression_report.txt")