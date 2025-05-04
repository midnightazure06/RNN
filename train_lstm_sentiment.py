
import pandas as pd
import numpy as np
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# 1. Load dataset
df = pd.read_csv("comments_1000_partial_labeled.csv")
df = df[['Komentar', 'Sentimen']].dropna()

# 2. Preprocessing
factory_stopwords = StopWordRemoverFactory()
stopwords = factory_stopwords.get_stop_words()
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed)

df['preprocessed'] = df['Komentar'].astype(str).apply(preprocess)

# 3. Encode label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentimen'])

# 4. Tokenization + Padding
texts = df['preprocessed'].astype(str)
labels = df['label']

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len)
y = labels

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 7. Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 8. Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=-1)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

df['Sentimen'] = label_encoder.inverse_transform(df['label'])

for label, filename in zip(['negatif', 'netral', 'positif'], 
                           ['wordcloud_neg.png', 'wordcloud_netral.png', 'wordcloud_pos.png']):
    text = ' '.join(df[df['Sentimen'] == label]['preprocessed'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(f"static/{filename}")

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - LSTM Model")
plt.tight_layout()
plt.savefig("static/barplot.png")  # Simpan sebagai file gambar
plt.close()


# 10. Simpan model dan tokenizer
model.save("model_lstm.h5")

import pickle
with open("tokenizer_lstm.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model dan tokenizer berhasil disimpan.")
