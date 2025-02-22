import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

real_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')
