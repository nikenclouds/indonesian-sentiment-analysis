import pandas as pd

# Membaca dataset training
df = pd.read_csv(
    "Data/train_preprocess_ori.tsv",
    sep="\t"
)

# Menampilkan contoh teks
print("=== CONTOH TEKS ===")
for i in range(10):
    print(f"{i+1}. {df['text'].iloc[i]}")

# Mengecek tipe data
print("\n=== TIPE DATA ===")
print(df.dtypes)

# Mengecek panjang teks
df["text_length"] = df["text"].astype(str).apply(len)

print("\n=== STATISTIK PANJANG TEKS ===")
print(df["text_length"].describe())