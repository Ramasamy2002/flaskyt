import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import numpy as np
from googleapiclient.discovery import build

# Replace with your YouTube Data API key
YOUTUBE_API_KEY = "AIzaSyAskP8XGk-kOuiSWowfEl3EsokF378K-vw"

app = Flask(__name__)

# Load your trained CNN model
model = load_model("sentimentt.h5")

# Load the CountVectorizer's vocabulary using pickle
vectorizer = CountVectorizer(decode_error="replace")
with open('vectorizer_vocabulary.pkl', 'rb') as vocab_file:
    vectorizer.vocabulary_ = np.load(vocab_file, allow_pickle=True)

# Define sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]

# Function to fetch comments from a YouTube video using the YouTube API
def fetch_youtube_comments(video_url, max_comments=5):
    video_id = video_url.split('v=')[-1]

    # Initialize the YouTube API client
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    # Fetch the video comments
    results = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=max_comments
    ).execute()

    comments = []
    for item in results['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    return comments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    user_input = request.form.get('video_url')

    # Fetch comments from the video
    fetched_comments = fetch_youtube_comments(user_input, max_comments=5)
    
    # Make predictions for each comment
    comment_sentiments = []
    for comment in fetched_comments:
        # Tokenize and vectorize the input comment
        comment_vectorized = vectorizer.transform([comment]).toarray()
        if comment_vectorized.shape[1] > 148:
            comment_vectorized = comment_vectorized[:, :148]
        elif comment_vectorized.shape[1] < 148:
            padding = 148 - comment_vectorized.shape[1]
            comment_vectorized = np.pad(comment_vectorized, ((0, 0), (0, padding)), 'constant')
        
        # Make predictions with your CNN model
        predictions = model.predict(comment_vectorized)
        
        # Interpret predictions
        predicted_label = sentiment_labels[np.argmax(predictions)]
        
        comment_sentiments.append((comment, predicted_label))

    # Determine the overall sentiment based on majority sentiment
    sentiment_counts = [sentiment for _, sentiment in comment_sentiments]
    overall_sentiment = max(set(sentiment_counts), key=sentiment_counts.count)

    return render_template('result.html', sentiment=overall_sentiment, comment_sentiments=comment_sentiments)

if __name__ == '__main__':
    app.run(debug=True)
