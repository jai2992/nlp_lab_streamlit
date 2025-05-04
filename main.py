
import streamlit as st

st.set_page_config(page_title="NLP Lab Questions", layout="wide")
st.title("🧠 NLP Laboratory - Streamlit Showcase")
st.sidebar.title("Choose a Lab Task")

tasks = {
    "1. Sentiment Analysis using Naïve Bayes": '''# Sentiment Analysis using Naïve Bayes
import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

nltk.download('movie_reviews')
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)
texts = [" ".join(doc) for doc, _ in docs]
labels = [label for _, label in docs]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

split_idx = int(0.8 * len(X.toarray()))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, predictions))
''',

    "2. SVM on Twitter Sentiment": '''# SVM on Twitter Sentiment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Example dataset: Replace with actual Sentiment140 CSV
data = pd.DataFrame({
    "text": ["I love this!", "I hate this!", "It was okay."],
    "sentiment": ["positive", "negative", "neutral"]
})

X_train, X_test, y_train, y_test = train_test_split(data["text"], data["sentiment"], test_size=0.3)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

st.text(classification_report(y_test, y_pred))
'''
# More tasks can be added similarly
}

task_name = st.sidebar.selectbox("Lab Task", list(tasks.keys()))
st.header(task_name)
st.code(tasks[task_name], language="python")
