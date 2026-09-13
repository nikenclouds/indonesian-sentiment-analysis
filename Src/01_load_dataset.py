import pandas as pd
import matplotlib.pyplot as plt
#1.Membaca dataset
df = pd.read_csv(
    "Data/test_preprocess_masked_label.tsv",
    sep="\t"
)
#2.Menampilkan 5 data pertama
print(df.head())

#3.Melihat ukuran dataset
print("\n=== UKURAN DATASET ===")
print(df.shape)

#4.Melihat nama kolom
print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

#5.Melihat Informasi dataset
print("\n=== INFORMASI DATASET ===")
df.info()

#6.Mengecek Missing Value
print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

#7. Mengecek data duplikat
print("\n=== DATA DUPLIKAT ===")
print(df.duplicated().sum())

# 8. Melihat distribusi sentiment
print("\n=== DISTRIBUSI SENTIMENT ===")
print(df["sentiment"].value_counts())

#9. Visualisasi distribusi sentiment
sentiment_counts = df["sentiment"].value_counts()

sentiment_counts.plot(kind="bar")

plt.title("Distribusi Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Jumlah Data")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("Results/sentiment_distribution.png")
plt.show()


