import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Membuat TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000
)

# TF-IDF hanya fit pada data training
X_train_tfidf = tfidf.fit_transform(X_train)

# Data testing hanya ditransform
X_test_tfidf = tfidf.transform(X_test)

# Menampilkan hasil
print("=== TF-IDF ===")
print("Jumlah data training :", X_train_tfidf.shape[0])
print("Jumlah data testing  :", X_test_tfidf.shape[0])
print("Jumlah fitur         :", X_train_tfidf.shape[1])

print("\n=== CONTOH FEATURE ===")
print(tfidf.get_feature_names_out()[:20])