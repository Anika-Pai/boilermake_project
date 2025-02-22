import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

real_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')
real_news['label'] = 1
fake_news['label'] = 0
data = pd.concat([real_news, fake_news], ignore_index=True)

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['cleaned_text'] = data['text'].apply(preprocess_text)

X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform (X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()