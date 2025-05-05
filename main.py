import streamlit as st
# changes done
st.set_page_config(page_title="NLP Lab Questions", layout="wide")
st.title("🧠 NLP Laboratory - Streamlit Showcase")
st.sidebar.title("Choose a Lab Task")

tasks = {
    "1. You are working for a movie review platform that wants to automatically analyze viewer sentiment. Your task is to build a sentiment classifier using the Naïve Bayes algorithm and test it on a dataset of movie reviews using Python.\nDataset: IMDb Movie Review Dataset (nltk.corpus.movie_reviews or IMDb Large Dataset)": '''import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
nltk.download('movie_reviews')
docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
labels = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.2, random_state=42)
vec = CountVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)
preds = clf.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, preds))''',
    "2. A client wants to filter social media posts for customer feedback. Your task is to train an SVM model to classify tweets as positive, negative, or neutral using Python.\nDataset: Sentiment140 or Kaggle Twitter US Airline Sentiment Dataset": '''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# df = pd.read_csv('Tweets.csv') # Load your dataset
# X = df['text']
# y = df['airline_sentiment']
# For demonstration, use dummy data:
X = ["I love this!", "This is bad", "Okay experience"]
y = ["positive", "negative", "neutral"]
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
clf = LinearSVC()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(classification_report(y_test, preds))''',
    "3. A company wants to know which model performs better for sentiment analysis. Your task is to compare Naïve Bayes and SVM using both Bag-of-Words and TF-IDF features in Python.\nDataset: IMDb or any subset of the Twitter dataset": '''from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# X, y = ... # Load your data
X = ["good movie", "bad movie", "average"]
y = ["pos", "neg", "pos"]
vecs = {"BoW": CountVectorizer(), "TF-IDF": TfidfVectorizer()}
models = {"NB": MultinomialNB(), "SVM": LinearSVC()}
for vname, vec in vecs.items():
    X_vec = vec.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    for mname, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"{mname} with {vname}: {accuracy_score(y_test, preds)}")''',
    "4. You have been asked to automate review analysis for an e-commerce website. Your task is to create a complete sentiment analysis pipeline using TF-IDF and SVM with Python.\nDataset: Amazon Product Reviews (Kaggle)": '''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# df = pd.read_csv('amazon_reviews.csv')
# X = df['review']
# y = df['sentiment']
X = ["Great product", "Terrible service", "Okay"]
y = ["positive", "negative", "neutral"]
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
clf = LinearSVC()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(classification_report(y_test, preds))''',
    "5. You’ve built a sentiment classifier, and now your manager wants insights into its performance. Your task is to compute and interpret the confusion matrix, precision, recall, and F1-score using Python.": '''from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
y_true = ["pos", "neg", "pos", "neg"]
y_pred = ["pos", "pos", "pos", "neg"]
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, pos_label="pos"))
print("Recall:", recall_score(y_true, y_pred, pos_label="pos"))
print("F1-score:", f1_score(y_true, y_pred, pos_label="pos"))''',
    "6. Your company is developing a topic-based news aggregator. Your task is to extract TF-IDF vectors from news articles using Python and analyze feature importance.\nDataset: 20 Newsgroups (sklearn.datasets.fetch_20newsgroups)": '''from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
data = fetch_20newsgroups(subset='train')
vec = TfidfVectorizer(max_features=10)
X = vec.fit_transform(data.data)
print("Top features:", vec.get_feature_names_out())''',
    "7. You’re building a content recommendation engine. Your task is to implement a KNN classifier using TF-IDF vectors to categorize new articles using Python.\nDataset: 20 Newsgroups": '''from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'])
vec = TfidfVectorizer()
X = vec.fit_transform(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Accuracy:", knn.score(X_test, y_test))''',
    "8. You notice classification accuracy is varying. Your task is to experiment with different k values in KNN and analyse how they affect performance in Python.\nDataset: 20 Newsgroups or Reuters-21578": '''from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'])
vec = TfidfVectorizer()
X = vec.fit_transform(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"k={k}, Accuracy={knn.score(X_test, y_test):.2f}")''',
    "9. Your lead wants a visual representation of document clusters. Your task is to apply dimensionality reduction techniques (PCA or t-SNE) to TF-IDF features and visualize clusters in Python.\nDataset: 20 Newsgroups": '''from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'])
vec = TfidfVectorizer()
X = vec.fit_transform(data.data).toarray()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1], c=data.target)
plt.title('PCA of TF-IDF Features')
plt.show()''',
    "10. You’re tasked with categorizing customer complaints into multiple topics. Your task is to implement a multi-class text classification system using TF-IDF and KNN in Python.\nDataset: Consumer Complaints Dataset (from Kaggle)": '''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# df = pd.read_csv('complaints.csv')
# X = df['complaint']
# y = df['category']
X = ["Bank charged extra fees", "Loan denied", "Credit card fraud"]
y = ["bank", "loan", "credit card"]
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Accuracy:", knn.score(X_test, y_test))''',
    "11. A finance firm wants to extract key entities from documents. Your task is to prepare a dataset with BIO-tagged sequences for NER in Python.\nDataset: CoNLL-2003 or Kaggle NER Dataset": '''# Example for preparing BIO-tagged data for multiple sentences
sentences = [
    ("John lives in New York", ["B-PER", "O", "O", "B-LOC", "I-LOC"]),
    ("Apple was founded by Steve Jobs", ["B-ORG", "O", "O", "O", "B-PER", "I-PER"])
]

# Output in CoNLL format
for sentence, tags in sentences:
    tokens = sentence.split()
    for token, tag in zip(tokens, tags):
        print(f"{token}\t{tag}")
    print()  # Sentence separator
''',
    "12. You’ve been assigned to build a custom NER engine. Your task is to implement an LSTM-based model for named entity recognition in Python.\nDataset: CoNLL-2003": '''import tensorflow as tf
from tensorflow.keras import layers
# X_train, y_train = ... # Prepare your tokenized and padded data
model = tf.keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.TimeDistributed(layers.Dense(10, activation='softmax'))
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=5)''',
    "13. Your NER system is mislabeling entities. Your task is to enhance it using a CRF layer on top of LSTM using Python.\nDataset: CoNLL-2003 or OntoNotes 5": '''# Example using tensorflow_addons for CRF
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
# X_train, y_train = ...
input = layers.Input(shape=(None,))
emb = layers.Embedding(input_dim=10000, output_dim=64, mask_zero=True)(input)
lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(emb)
dense = layers.TimeDistributed(layers.Dense(10))(lstm)
crf = tfa.layers.CRF(10)
output = crf(dense)
model = tf.keras.Model(input, output)
model.compile(optimizer='adam', loss=crf.loss, metrics=[crf.accuracy])''',
    "14. You need to report NER performance. Your task is to evaluate the model using entity-level precision, recall, and F1-score in Python.\nDataset: Same as Q12": '''from seqeval.metrics import classification_report
y_true = [['B-PER', 'O', 'B-LOC']]
y_pred = [['B-PER', 'O', 'O']]
print(classification_report(y_true, y_pred))''',
    "15. Your LSTM model isn’t accurate enough. Your task is to integrate GloVe embeddings and measure the improvement using Python.\nDataset: CoNLL-2003 + GloVe embeddings": '''import numpy as np
from tensorflow.keras.layers import Embedding
# Load GloVe
embeddings_index = {}
with open('glove.6B.50d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
# Prepare embedding matrix and use in Embedding layer
# embedding_matrix = ...
# model = ...''',
    "16. You're creating a semantic search tool for legal documents. Your task is to compute TF-IDF vectors and identify similar documents using Python.\nDataset: Legal Case Reports (e.g., from Kaggle)": '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["Case about contract law", "Case about criminal law"]
vec = TfidfVectorizer()
X = vec.fit_transform(docs)
sim = cosine_similarity(X)
print(sim)''',
    "17. You are building a plagiarism checker. Your task is to calculate cosine similarity between student reports using Python.\nDataset: Custom or PAN Plagiarism Dataset": '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["Student report 1 text", "Student report 2 text"]
vec = TfidfVectorizer()
X = vec.fit_transform(docs)
sim = cosine_similarity(X)
print(sim)''',
    "18. You’re developing a knowledge-base search engine. Your task is to retrieve relevant documents using TF-IDF and cosine similarity in Python.\nDataset: StackOverflow Questions or Wikipedia dump": '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["How to use Python?", "Python for data science"]
query = ["Python usage"]
vec = TfidfVectorizer()
X = vec.fit_transform(docs + query)
sim = cosine_similarity(X[-1], X[:-1])
print(sim)''',
    "19. A publisher wants to group similar news stories. Your task is to cluster the articles using cosine similarity and visualize results in Python.\nDataset: BBC News Dataset": '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
docs = ["News story 1", "News story 2", "News story 3"]
vec = TfidfVectorizer()
X = vec.fit_transform(docs).toarray()
clustering = AgglomerativeClustering(n_clusters=2).fit(X)
plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
plt.show()''',
    "20. You’re fine-tuning your similarity system. Your task is to evaluate the impact of stopword removal and lemmatization on similarity results using Python.\nDataset: Any news or article dataset": '''from sklearn.feature_extraction.text import TfidfVectorizer
docs = ["Cats are running", "A cat runs"]
vec1 = TfidfVectorizer()
X1 = vec1.fit_transform(docs)
vec2 = TfidfVectorizer(stop_words='english')
X2 = vec2.fit_transform(docs)
print("With stopwords:", X1.toarray())
print("Without stopwords:", X2.toarray())''',
    "21. You’ve been asked to build an auto-summarizer for reports. Your task is to extract sentence embeddings using BERT and identify key sentences using Python.\nDataset: CNN/DailyMail or Legal Summarization Dataset": '''from transformers import BertTokenizer, BertModel
import torch
sentences = ["This is a sentence.", "Another sentence."]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)
print(embeddings)''',
    "22. You need to present summaries of long technical reports. Your task is to implement an extractive summarizer using BERT embeddings in Python.\nDataset: ArXiv Papers or CNN/DailyMail": '''# Use sentence-transformers for extractive summarization
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
sim = cosine_similarity([embeddings[0]], embeddings)[0]
print("Similarity to first sentence:", sim)''',
    "23. Your manager wants to evaluate the summarizer. Your task is to compute ROUGE scores to compare generated summaries against gold-standard ones using Python.\nDataset: CNN/DailyMail or custom summaries": '''from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
summary = "The cat sat on the mat."
gold = "A cat was sitting on the mat."
scores = scorer.score(gold, summary)
print(scores)''',
    "24. You are building an AI tutor. Your task is to implement a BERT-based question answering system in Python.\nDataset: SQuAD v1.1 or v2.0": '''from transformers import pipeline
qa = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
context = "Python is a programming language."
question = "What is Python?"
result = qa(question=question, context=context)
print(result['answer'])''',
    "25. You want to deploy the QA model on a platform. Your task is to create a web interface using Flask or Streamlit with a BERT QA model in Python.\nDataset: SQuAD or custom FAQs": '''import streamlit as st
from transformers import pipeline
st.title("BERT QA Demo")
qa = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
context = st.text_area("Context")
question = st.text_input("Question")
if st.button("Get Answer") and context and question:
    result = qa(question=question, context=context)
    st.write("Answer:", result['answer'])''',
    "26. A creative writing app wants story suggestions. Your task is to generate short stories using GPT-3 or GPT-Neo via API integration or Hugging Face Transformers in Python.\nDataset: None required (model is pre-trained, prompt-based)": '''from transformers import pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
prompt = "Once upon a time"
story = generator(prompt, max_length=50)[0]['generated_text']
print(story)''',
    "27. You’re deploying a model to mobile hardware. Your task is to apply symmetric quantization to a classification or summarization model using Python and TensorFlow Lite or ONNX.\nDataset: Any small model-compatible dataset (e.g., IMDb, MNIST)": '''import tensorflow as tf
# model = ... # Your trained model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_model)''',
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
