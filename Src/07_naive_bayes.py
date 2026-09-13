import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Membaca dataset
df = pd.read_csv(
    "Data/train_preprocess_ori.tsv",
    sep="\t"
)

# Fungsi cleaning text
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

# Cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Input dan target
X = df["clean_text"]
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Membuat model Naive Bayes
model = MultinomialNB()

# Training
model.fit(X_train_tfidf, y_train)

# Prediksi
y_pred = model.predict(X_test_tfidf)

# Membuat confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
)

# Membuat diagram
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")

plt.tight_layout()

# Menyimpan diagram ke folder Results
plt.savefig(
    "Results/naive_bayes_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nDiagram berhasil disimpan di Results/")

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)

print("=== NAIVE BAYES ===")
print("Accuracy:", accuracy)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Membuat folder Results jika belum ada
Path("Results").mkdir(exist_ok=True)

# Menyimpan classification report
report = classification_report(y_test, y_pred)

with open("Results/naive_bayes_report.txt", "w") as file:
    file.write("NAIVE BAYES - SENTIMENT ANALYSIS\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"Accuracy: {accuracy:.2f}\n\n")
    file.write("CLASSIFICATION REPORT\n")
    file.write("=" * 40 + "\n")
    file.write(report)

print("\nHasil berhasil disimpan ke:")
print("Results/naive_bayes_report.txt")