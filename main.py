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

split_idx = int(0.8 * len(texts))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, predictions))

# Predict on new data
new_data = ["this is the bad movie"]
X_new = vectorizer.transform(new_data)
prediction = model.predict(X_new)
st.write("Prediction for new data:", prediction)
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
''',

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
    "19. Your NER system is mislabeling entities. Your task is to enhance it using a CRF layer on top of LSTM using Python.\nDataset: CoNLL-2003 or OntoNotes 5": '''# NER with LSTM+CRF
# Dataset: CoNLL-2003 or OntoNotes 5
import torch
import torch.nn as nn
from torchcrf import CRF

class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# Example usage: model = LSTM_CRF(vocab_size, tagset_size)
''',
    "20. You need to report NER performance. Your task is to evaluate the model using entity-level precision, recall, and F1-score in Python.\nDataset: Same as Q19": '''# Evaluate NER Performance
# Compute entity-level precision, recall, F1-score
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

y_true = [['B-PER', 'O', 'B-LOC'], ['B-ORG', 'O']]
y_pred = [['B-PER', 'O', 'B-LOC'], ['O', 'O']]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
''',
    "21. Your LSTM model isn’t accurate enough. Your task is to integrate GloVe embeddings and measure the improvement using Python.\nDataset: CoNLL-2003 + GloVe embeddings": '''# LSTM with GloVe Embeddings
# Dataset: CoNLL-2003 + GloVe
import numpy as np
import torch.nn as nn

def load_glove_embeddings(glove_path, word2idx, embedding_dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
    return embeddings

# Example: embedding_matrix = load_glove_embeddings('glove.6B.100d.txt', word2idx)
# model.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
''',
    "22. You're creating a semantic search tool for legal documents. Your task is to compute TF-IDF vectors and identify similar documents using Python.\nDataset: Legal Case Reports (e.g., from Kaggle)": '''# Semantic Search for Legal Documents
# Compute TF-IDF and find similar docs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "The contract was signed in 2020.",
    "A legal agreement was reached.",
    "The court ruled in favor of the plaintiff."
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs)
sim = cosine_similarity(tfidf)
print(sim)
''',
    "23. You are building a plagiarism checker. Your task is to calculate cosine similarity between student reports using Python.\nDataset: Custom or PAN Plagiarism Dataset": '''# Plagiarism Checker
# Calculate cosine similarity between reports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

reports = [
    "This is the first student report.",
    "This is the second student report.",
    "This is a copied student report."
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(reports)
sim = cosine_similarity(tfidf)
print(sim)
''',
    "24. You’re developing a knowledge-base search engine. Your task is to retrieve relevant documents using TF-IDF and cosine similarity in Python.\nDataset: StackOverflow Questions or Wikipedia dump": '''# Knowledge-base Search Engine
# Retrieve docs using TF-IDF + cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "How to use Python for web development?",
    "What is a lambda function in Python?",
    "Best practices for REST API design."
]
query = ["python lambda function"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs + query)
sim = cosine_similarity(tfidf[-1], tfidf[:-1])
print(sim)
''',
    "25. A publisher wants to group similar news stories. Your task is to cluster the articles using cosine similarity and visualize results in Python.\nDataset: BBC News Dataset": '''# News Clustering and Visualization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

articles = [
    "Elections were held in the city.",
    "The government announced new policies.",
    "A new tech startup launched."
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(articles)
km = KMeans(n_clusters=2, random_state=42).fit(X)
pca = PCA(n_components=2).fit_transform(X.toarray())
plt.scatter(pca[:,0], pca[:,1], c=km.labels_)
plt.title('News Clusters')
plt.show()
''',
    "26. You’re fine-tuning your similarity system. Your task is to evaluate the impact of stopword removal and lemmatization on similarity results using Python.\nDataset: Any news or article dataset": '''# Evaluate Preprocessing on Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split() if w.lower() not in stop_words])

docs = [
    "Cats are running in the garden.",
    "A cat runs in gardens."
]
pre_docs = [preprocess(doc) for doc in docs]
vectorizer = TfidfVectorizer()
sim_before = cosine_similarity(vectorizer.fit_transform(docs))
sim_after = cosine_similarity(vectorizer.fit_transform(pre_docs))
print("Before:", sim_before)
print("After:", sim_after)
''',
    "27. You’ve been asked to build an auto-summarizer for reports. Your task is to extract sentence embeddings using BERT and identify key sentences using Python.\nDataset: CNN/DailyMail or Legal Summarization Dataset": '''# Auto-summarizer with BERT
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentences = [
    "The report discusses the impact of AI.",
    "AI is transforming industries."
]
embeddings = []
for sent in sentences:
    inputs = tokenizer(sent, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    embeddings.append(emb)

sim = np.inner(embeddings, embeddings)
print("Sentence similarity:", sim)
# Select key sentences based on similarity or clustering
''',
    "28. You need to present summaries of long technical reports. Your task is to implement an extractive summarizer using BERT embeddings in Python.\nDataset: ArXiv Papers or CNN/DailyMail": '''# Extractive Summarizer with BERT
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def get_sentence_embedding(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

text = """BERT is a transformer-based model. It is used for many NLP tasks. Extractive summarization selects important sentences."""
sentences = text.split('.')
embeddings = [get_sentence_embedding(s) for s in sentences if s.strip()]
sim_matrix = np.inner(embeddings, embeddings)
# Select sentences with highest centrality
print("Similarity matrix:", sim_matrix)
''',
    "29. Your manager wants to evaluate the summarizer. Your task is to compute ROUGE scores to compare generated summaries against gold-standard ones using Python.\nDataset: CNN/DailyMail or custom summaries": '''# Summarizer Evaluation with ROUGE
from rouge import Rouge
rouge = Rouge()
gold = "The cat sat on the mat."
pred = "A cat was on the mat."
scores = rouge.get_scores(pred, gold)
print(scores)
''',
    "30. You are building an AI tutor. Your task is to implement a BERT-based question answering system in Python.\nDataset: SQuAD v1.1 or v2.0": '''# BERT-based QA System
from transformers import pipeline
qa = pipeline('question-answering')
context = "Python is a programming language created by Guido van Rossum."
question = "Who created Python?"
result = qa(question=question, context=context)
print(result)
''',
    "31. You want to deploy the QA model on a platform. Your task is to create a web interface using Flask or Streamlit with a BERT QA model in Python.\nDataset: SQuAD or custom FAQs": '''# QA Web Interface with Streamlit
import streamlit as st
from transformers import pipeline
qa = pipeline('question-answering')
st.title("BERT QA Demo")
context = st.text_area("Context")
question = st.text_input("Question")
if st.button("Get Answer") and context and question:
    result = qa(question=question, context=context)
    st.write("Answer:", result['answer'])
''',
    "32. A creative writing app wants story suggestions. Your task is to generate short stories using GPT-3 or GPT-Neo via API integration or Hugging Face Transformers in Python.\nDataset: None required (model is pre-trained, prompt-based)": '''# Story Generation with GPT-2 (Hugging Face)
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
prompt = "Once upon a time in a distant land,"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
''',
    "33. You’re deploying a model to mobile hardware. Your task is to apply symmetric quantization to a classification or summarization model using Python and TensorFlow Lite or ONNX.\nDataset: Any small model-compatible dataset (e.g., IMDb, MNIST)": '''# Model Quantization for Mobile (TensorFlow Lite)
import tensorflow as tf
# Assume you have a trained model 'model'
# model = ...
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_model)
print("Quantized model saved as model_quant.tflite")
''',
}

if 'selected_task' not in st.session_state:
    st.session_state.selected_task = list(tasks.keys())[0]

ordered_keys = [k for k in tasks.keys() if k.split('.')[0].isdigit()]
ordered_keys.sort(key=lambda x: int(x.split('.')[0]))

# Sidebar: Add search box and display numbered, properly aligned buttons
search_query = st.sidebar.text_input("Search questions...")

# Filter questions by search query (case-insensitive)
filtered_keys = [k for k in ordered_keys if search_query.lower() in k.lower()]

for idx, task in enumerate(filtered_keys, 1):
    # Extract the number from the question for button label
    num = task.split('.')[0]
    label = f"{num}. {task[len(num)+1:].strip()}"
    if st.sidebar.button(label, key=task):
        st.session_state.selected_task = task

st.header(st.session_state.selected_task)
st.code(tasks[st.session_state.selected_task], language="python")
