
import streamlit as st

st.set_page_config(page_title="NLP Lab Questions", layout="wide")
st.title("🧠 NLP Laboratory - Streamlit Showcase")
st.sidebar.title("Choose a Lab Task")

tasks = {
    "11. Word Embeddings using Word2Vec": """# Word Embeddings using Word2Vec
import gensim
from gensim.models import Word2Vec

sentences = [["this", "is", "the", "first", "sentence"],
             ["this", "is", "another", "sentence"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['sentence']

st.write("Vector for 'sentence':", vector)
""",
    "12. Text Classification using LSTM (example only, no training)": """# Text Classification using LSTM (Model Structure Only)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.summary(print_fn=lambda x: st.text(x))
""",
    "13. Text Summarization using Gensim": """# Text Summarization using Gensim
from gensim.summarization import summarize

text = "Natural language processing is a field of AI that gives machines the ability to read and understand human language. It helps computers communicate with humans in their own language."

summary = summarize(text)

st.write("Original Text:", text)
st.write("Summary:", summary)
""",
    "14. Translation using Transformers (Hugging Face)": """# Translation using Transformers
from transformers import pipeline

translator = pipeline("translation_en_to_fr")
result = translator("NLP is fun and exciting!")

st.write("Translation:", result[0]['translation_text'])
""",
    "15. Text Generation using GPT-2": """# Text Generation using GPT-2
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Natural Language Processing is", max_length=30, num_return_sequences=1)

st.write("Generated Text:", result[0]["generated_text"])
""",
    "16. Speech Recognition using SpeechRecognition": """# Speech Recognition (Offline Audio File Example)
import speech_recognition as sr

recognizer = sr.Recognizer()
st.write("Upload an audio file (WAV format)")
uploaded_audio = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_audio:
    with sr.AudioFile(uploaded_audio) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("Recognized Text:", text)
        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError:
            st.write("Could not request results")
""",
    "17. Speech Synthesis using pyttsx3": """# Speech Synthesis using pyttsx3 (Local only)
import pyttsx3

engine = pyttsx3.init()
text = "Hello! This is a speech synthesis demo."
engine.say(text)
engine.runAndWait()

st.write("Speech Synthesized:", text)
""",
    "18. Extracting MFCC Features": """# Extract MFCC features using librosa
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.write("Upload an audio file (WAV format)")
uploaded_file = st.file_uploader("Choose a file", type=["wav"])

if uploaded_file:
    y, sr = librosa.load(uploaded_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    st.write("MFCC Shape:", mfccs.shape)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)
""",
    "3. Text Preprocessing (Tokenization, Stopwords Removal)": """# Text Preprocessing Example
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

text = "Natural Language Processing is amazing and powerful!"
tokens = word_tokenize(text)
filtered = [word for word in tokens if word.lower() not in stopwords.words('english')]

st.write("Original Text:", text)
st.write("Tokens:", tokens)
st.write("Filtered Tokens:", filtered)
""",
    "4. Lemmatization and Stemming": """# Lemmatization and Stemming
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

text = "running runs runner easily fairly"
tokens = word_tokenize(text)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
stemmed = [stemmer.stem(w) for w in tokens]

st.write("Tokens:", tokens)
st.write("Lemmatized:", lemmatized)
st.write("Stemmed:", stemmed)
""",
    "5. POS Tagging": """# Part-of-Speech Tagging
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

st.write("POS Tags:", pos_tags)
""",
    "6. Named Entity Recognition": """# Named Entity Recognition using spaCy
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple is looking at buying U.K. startup for $1 billion"

doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]

st.write("Named Entities:", entities)
""",
    "7. N-gram Generation": """# Generating N-grams
from nltk import ngrams

text = "Natural Language Processing with Python"
tokens = text.split()
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

st.write("Bigrams:", bigrams)
st.write("Trigrams:", trigrams)
""",
    "8. Bag of Words Model": """# Bag of Words Model using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

texts = ["NLP is fun", "I love NLP", "NLP is powerful"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

st.write("Feature Names:", vectorizer.get_feature_names_out())
st.write("Bag of Words Matrix:")
st.dataframe(X.toarray())
""",
    "9. TF-IDF Vectorization": """# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["NLP is fun", "I love NLP", "NLP is powerful"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

st.write("TF-IDF Matrix:")
st.dataframe(X.toarray())
""",
    "10. Cosine Similarity": """# Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = ["I love NLP", "NLP is amazing", "Deep learning is part of NLP"]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

st.write("Cosine Similarity with first sentence:")
st.write(cos_sim)
""",
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
