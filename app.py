from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os

app = Flask(__name__)

# Load model dan tokenizer
model = load_model("model_lstm.h5")
with open("tokenizer_lstm.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Label mapping
label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}

# Preprocessing tools
stopwords = StopWordRemoverFactory().get_stop_words()
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

@app.route('/', methods=['GET', 'POST'])
def index():
    komentar = None
    sentimen = None

    if request.method == 'POST':
        komentar = request.form['komentar']
        cleaned = preprocess(komentar)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)
        pred = np.argmax(model.predict(padded), axis=-1)[0]
        sentimen = label_mapping[pred]

    # tetap kirim ke template walau belum ada komentar
    return render_template(
        'index.html',
        komentar=komentar,
        sentimen=sentimen
    )

if __name__ == '__main__':
    app.run(debug=True)
