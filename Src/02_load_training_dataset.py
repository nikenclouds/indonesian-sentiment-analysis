import pandas as pd
#Membaca dataset training
df = pd.read_csv(
    "Data/train_preprocess_ori.tsv",
    sep="\t"
)

#Menampilkan 5 data pertama
print(df.head())

#Ukuran dataset
print("\n=== UKURAN DATASET ===")
print(df.shape)

#Nama Kolom
print("\n=== NAMA KOLOM ===")
print(df.shape)

#Missing Value 
print("\n=== Missing Value ===")
print(df.isnull(). sum())

#Duplicate
print("\n=== Duplicate ===")
print(df.duplicated().sum())

#Distribusi Sentiment
print("\n=== DISTRIBUSI SENTIMEN ===")
print(df["sentiment"].value_counts())