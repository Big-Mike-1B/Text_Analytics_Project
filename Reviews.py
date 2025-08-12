import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re
import nltk
from PIL import Image
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt')
nltk.download('stopwords')
import base64
from pathlib import Path
import time


# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Amazon Fine Food Reviews", layout="wide", page_icon="⭐⭐⭐⭐⭐")

# ===================== FILE PATHS =====================
BASE = Path(__file__).parent
IMAGE_FILES = [
    BASE / "Food.jpeg",
    BASE / "Food1.jpeg",
    BASE / "Food2.jpeg"
    ]
DATA_PATH = BASE / "data" / "Reviews.csv"

# ===================== UTILITIES =====================
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Robust CSV loader: infers delimiter and handles BOM."""
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, low_memory=False)
        except Exception:
            continue
    # last attempt with defaults to surface a clear error
    return pd.read_csv(path, low_memory=False)

# ===================== HEADER IMAGE =====================
# Initialize index in session state
if "img_index" not in st.session_state:
    st.session_state.img_index = 0



# Get current image
current_image = IMAGE_FILES[st.session_state.img_index]

# Display current image
if current_image.exists():
    try:
        image_base64 = get_base64_image(current_image)
        st.markdown(
            f"""
            <div style="width:100%; text-align:center;">
                <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Header Image">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.image(str(current_image), use_container_width=True)
else:
    st.warning(f"Image not found: {current_image.name}")

# Move to the next image
st.session_state.img_index = (st.session_state.img_index + 1) % len(IMAGE_FILES)

# ===================== INTRO =====================
st.markdown("""
###  🤗😤 Emotion-based on Review

""")


#Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Reviews.csv")
    df_sample = df.sample(5000, random_state=42).reset_index(drop=True)
    return df_sample


df_reviews = load_data()

# Auto-label emotions
@st.cache_resource
def label_emotions(data):
    classifier = pipeline("text-classification",
                          model="cardiffnlp/twitter-roberta-base-emotion",
                          top_k=None)
    emotions = []
    for text in data['Text']:
        try:
            result = classifier(str(text))[0]
        # Get emotion with highest score
            top_emotion = max(result, key=lambda x: x['score'])['label']
            emotions.append(top_emotion)
        except:
            emotions.append("neutral")
    data['emotion'] = emotions
    return data


df_labeled = label_emotions(df_reviews)


# Clean Text
def clean_text(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'<.*?>', " ", text)  # Remove HTML
        text = re.sub(r'[^\w\s]', "", text)  # Remove punctuation
        text = re.sub(r'\d+', " ", text)  # Remove digits
        text = re.sub(r'\s+', " ", text).strip()
        cleaned.append(text)
    return cleaned


df_labeled['clean_text'] = clean_text(df_labeled['Text'])


# Word Clouds
def plot_wordcloud(emotion):
    text = " ".join(df_labeled[df_labeled['emotion'] == emotion]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# Model Training
def train_models():
    X = df_labeled['clean_text']
    y = df_labeled['emotion']

    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    # SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    # Evaluation
    metrics = {
        'Model': ['Decision Tree', 'SVM'],
        'Precision': [
            precision_score(y_test, dt_pred, average='weighted'),
            precision_score(y_test, svm_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_test, dt_pred, average='weighted'),
            recall_score(y_test, svm_pred, average='weighted')
        ],
        'F1-score': [
            f1_score(y_test, dt_pred, average='weighted'),
            f1_score(y_test, svm_pred, average='weighted')
        ],
        'ROC-AUC': [
            roc_auc_score(pd.get_dummies(y_test), dt.predict_proba(X_test), average='weighted', multi_class='ovr'),
            roc_auc_score(pd.get_dummies(y_test), svm.predict_proba(X_test), average='weighted', multi_class='ovr')
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, dt, svm, tfidf


metrics_df, dt_model, svm_model, tfidf_vectorizer = train_models()

# Custom CSS for spacing tabs
st.markdown("""
    <style>
    div[data-baseweb="tab-list"] {
        gap: 40px; /* space between tabs */
    }
    </style>
    """, unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "😊 Emotion Detection", "☁ Word Clouds", "✅ Conclusion"])

#Streamlit Pages
with tab1:
    st.subheader("📊 Dataset content")

    if st.checkbox("Show Sample Data"):
        st.write(df_labeled.head())
    st.write("Emotion distribution in the sample:")
    st.bar_chart(df_labeled['emotion'].value_counts())

with tab2:

    st.subheader("😊 Emotion Detection")
    st.write("Model performance metrics:")
    st.dataframe(metrics_df)
    st.bar_chart(metrics_df.set_index('Model')[['Precision', 'Recall', 'F1-score', 'ROC-AUC']])

    user_input = st.text_area("Enter text for emotion prediction:")
    if st.button("Predict Emotion"):
        if user_input.strip():
            cleaned_input = clean_text([user_input])
            input_tfidf = tfidf_vectorizer.transform(cleaned_input)
            pred_dt = dt_model.predict(input_tfidf)[0]
            pred_svm = svm_model.predict(input_tfidf)[0]
            st.write(f"*Decision Tree prediction:* {pred_dt}")
            st.write(f"*SVM prediction:* {pred_svm}")

with tab3:
    st.subheader("☁ Word clouds content")
    emotions = df_labeled['emotion'].unique()
    emotion_choice = st.selectbox("Select emotion:", emotions)
    plot_wordcloud(emotion_choice)

with tab4:
    st.write("✅ Conclusion content here")
    st.subheader("Conclusion")
    st.write("""
    - We used TF-IDF with Decision Tree and SVM classifiers for emotion detection.
    - SVM generally performs better in text classification tasks.
    - Word clouds give insights into common words per emotion.
    - Auto-labeling was done using a pre-trained RoBERTa model.
    """)




