# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

MODEL_PATH = Path("model/spam_model.pkl")
VECTORIZER_PATH = Path("model/vectorizer.pkl")

@st.cache_resource
def load_objects(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def predict_messages(model, vectorizer, messages):
    processed = [preprocess_text(m) for m in messages]
    X = vectorizer.transform(processed)
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X).max(axis=1)
    else:
        probs = np.ones(len(preds)) * np.nan
    return preds, probs

def main():
    st.set_page_config(page_title="📨 SMS Spam Detection", page_icon="📱", layout="centered")
    st.title("📨 SMS Spam Detection Web App")
    st.markdown("### Detect whether an SMS is **Spam** or **Ham (Not Spam)** using a Machine Learning model")

    try:
        model, vectorizer = load_objects()
    except Exception as e:
        st.error(f"❌ Error loading model/vectorizer: {e}")
        st.stop()

    # Single text prediction
    st.subheader("🔹 Predict a Single Message")
    sms_input = st.text_area("Enter your SMS message below:", height=120)

    if st.button("Predict"):
        if sms_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            pred, prob = predict_messages(model, vectorizer, [sms_input])
            label = "🚨 SPAM" if pred[0].lower() == "spam" else "✅ HAM (Not Spam)"
            st.success(f"**Prediction:** {label}")
            if not np.isnan(prob[0]):
                st.info(f"Confidence: {prob[0]*100:.2f}%")

    st.markdown("---")

    # Batch prediction via CSV
    st.subheader("📂 Batch Prediction via CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'message' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'message' not in df.columns:
            st.error("The CSV must contain a 'message' column.")
        else:
            preds, probs = predict_messages(model, vectorizer, df['message'].astype(str).tolist())
            df['prediction'] = preds
            df['confidence'] = probs
            st.write(df.head(20))
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit | Project: SMS Spam Detection")

if __name__ == "__main__":
    main()
