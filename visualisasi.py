import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("comments_1000_partial_labeled.csv")

# Pastikan Timestamp dalam format datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# 1. Aktivitas Komentar per Hari
df['Tanggal'] = df['Timestamp'].dt.date
aktivitas_harian = df.groupby('Tanggal').size()

plt.figure(figsize=(10, 5))
sns.lineplot(x=aktivitas_harian.index, y=aktivitas_harian.values, marker='o')
plt.title('Aktivitas Komentar per Hari')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Komentar')
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs('static', exist_ok=True)
plt.savefig('static/aktivitas_per_hari.png')
plt.close()

# 2. Distribusi Komentar per Platform
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Platform', order=df['Platform'].value_counts().index, palette='viridis')
plt.title('Distribusi Komentar per Platform')
plt.xlabel('Platform')
plt.ylabel('Jumlah Komentar')
plt.tight_layout()
plt.savefig('static/distribusi_platform.png')
plt.close()

print("✅ Grafik berhasil disimpan di folder static/")
