# train_model.py
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path

# Download NLTK stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

# Load dataset
data_path = Path("data/spam.csv")
df = pd.read_csv(data_path, encoding='latin-1')

# Some datasets have extra unnamed columns — drop them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

df['clean_message'] = df['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("✅ Model trained successfully!")
print(f"Accuracy: {acc*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model & vectorizer
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model and vectorizer saved in 'model/' folder.")
