import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca dataset
df = pd.read_csv(
    "Data/train_preprocess_ori.tsv",
    sep="\t"
)

# Cleaning text
import re

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

# Input dan target
X = df["clean_text"]
y = df["sentiment"]

# Membagi dataset 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Menampilkan ukuran data
print("=== PEMBAGIAN DATA ===")
print("Jumlah data keseluruhan :", len(df))
print("Data training           :", len(X_train))
print("Data testing            :", len(X_test))

# Distribusi sentiment
print("\n=== DISTRIBUSI TRAINING ===")
print(y_train.value_counts())

print("\n=== DISTRIBUSI TESTING ===")
print(y_test.value_counts())