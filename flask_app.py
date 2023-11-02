import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained CNN model
model = load_model("sentimentt.h5")

# Load the CountVectorizer's vocabulary using pickle
vectorizer = CountVectorizer(decode_error="replace")
with open('vectorizer_vocabulary.pkl', 'rb') as vocab_file:
    vectorizer.vocabulary_ = np.load(vocab_file, allow_pickle=True)

# Define sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    user_input = request.form.get('comments')

    # Tokenize and vectorize the input comments
    new_comments_vectorized = vectorizer.transform([user_input]).toarray()

    # Ensure that the input data has the expected shape (None, 145)
    if new_comments_vectorized.shape[1] > 145:
        new_comments_vectorized = new_comments_vectorized[:, :145]
    elif new_comments_vectorized.shape[1] < 145:
        # You can pad with zeros if needed
        padding = 145 - new_comments_vectorized.shape[1]
        new_comments_vectorized = np.pad(new_comments_vectorized, ((0, 0), (0, padding)), 'constant')

    # Make predictions with your CNN model
    predictions = model.predict(new_comments_vectorized)

    # Interpret predictions
    predicted_label = sentiment_labels[np.argmax(predictions)]

    return render_template('result.html', sentiment=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
