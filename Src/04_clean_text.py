import pandas as pd
import re
#Membaca Dataset
df = pd.read_csv(
    "Data/train_preprocess_ori.tsv",
    sep="\t"
)
# Fungsi cleaning text
def clean_text(text):
    text = str(text)

    # Mengubah huruf menjadi lowercase
    text = text.lower()

    # Menghapus URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # Menghapus mention (@username)
    text = re.sub(r"@\w+", "", text)

    # Menghapus hashtag (#)
    text = re.sub(r"#", "", text)

    # Menghapus angka
    text = re.sub(r"\d+", "", text)

    # Menghapus tanda baca dan karakter khusus
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Menghapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Membuat kolom teks hasil cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Menampilkan perbandingan sebelum dan sesudah cleaning
print("=== SEBELUM vs SESUDAH CLEANING ===")

for i in range(10):
    print(f"\n{i+1}. SEBELUM : {df['text'].iloc[i]}")
    print(f"   SESUDAH : {df['clean_text'].iloc[i]}")

# Mengecek missing value
print("\n=== MISSING VALUE ===")
print(df["clean_text"].isnull().sum())