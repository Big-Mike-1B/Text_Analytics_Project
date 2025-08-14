import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"  # Allow up to 1GB
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
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import pipeline
import torch


nltk.download('punkt')
nltk.download('stopwords')
import base64
from pathlib import Path


from datetime import datetime




# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Amazon Fine Food Reviews", layout="wide", page_icon="⭐⭐⭐⭐⭐")

# ===================== FILE PATHS =====================
BASE = Path(__file__).parent
IMAGE_FILES = [
    BASE / "Food.jpeg",
    BASE / "Food1.jpeg",
    BASE / "Food2.jpeg"
    ]
# Upload CSV file
DATA_PATH = BASE / "Reviews.csv"

# ===================== UTILITIES =====================
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

@st.cache_resource(show_spinner=False)
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
@st.cache_resource
def load_data():
    df = pd.read_csv("Reviews.csv")
    df_sample = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)
    return df_sample


df_reviews = load_data()

# Auto-label emotions
@st.cache_resource
def label_emotions(data):
    classifier = pipeline("text-classification",
                          model="bhadresh-savani/distilbert-base-uncased-emotion",
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

#Tokenization
words = word_tokenize(' '.join(df_labeled['clean_text']))
tok_text = pd.DataFrame({'Tokens': words})

# Display stop words from the nltk
stop_words_disp = set(stopwords.words('english'))

# Removal of the stopwords
filtered_text = [word for word in words if word.lower() not in stop_words_disp]

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

    # Align classes for ROC-AUC
    lb = LabelBinarizer()
    lb.fit(y_train)  # Fit only on training labels to include all classes
    y_test_bin = lb.transform(y_test)

    # Handle binary case: lb.transform returns 1D array
    if y_test_bin.ndim == 1:
        y_test_bin = np.vstack([1 - y_test_bin, y_test_bin]).T

    # Align predicted probabilities with LabelBinarizer classes
    dt_probs = pd.DataFrame(dt.predict_proba(X_test), columns=dt.classes_)
    dt_probs = dt_probs.reindex(columns=lb.classes_, fill_value=0)

    svm_probs = pd.DataFrame(svm.predict_proba(X_test), columns=svm.classes_)
    svm_probs = svm_probs.reindex(columns=lb.classes_, fill_value=0)

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
            roc_auc_score(y_test_bin, dt_probs, average='weighted', multi_class='ovr'),
            roc_auc_score(y_test_bin, svm_probs, average='weighted', multi_class='ovr')
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, dt, svm, tfidf

# Custom CSS for spacing tabs
st.markdown("""
    <style>
    div[data-baseweb="tab-list"] {
        gap: 40px; /* space between tabs */
    }
    </style>
    """, unsafe_allow_html=True)
tab1, tab2, tab3, tab4,tab5, tab6 = st.tabs(["📊 Dataset", "😊 Emotion Detection", "☁ Word Clouds", "✅ Conclusion","User Interface","Submitted Review"])

#Streamlit Pages
with tab1:
    st.subheader("📊 Dataset content")

    if st.checkbox("Show Sample Data"):
        st.write(df_labeled.head())
    st.write("Emotion distribution in the sample:")
    st.bar_chart(df_labeled['emotion'].value_counts())

with tab2:
    st.subheader("😊 Emotion Detection")

    # Train models and get metrics
    metrics_df, dt_model, svm_model, tfidf_vectorizer = train_models()

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


with tab5:
    # ===================== LOAD EMOTION MODEL =====================
    @st.cache_resource
    def load_emotion_model():
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


    emotion_classifier = load_emotion_model()

    # ===================== PAGE CONFIG =====================
    st.set_page_config(page_title="🍽 Food Review App", page_icon="🍔", layout="centered")
    st.title("🍽 Food Review & Feedback System")
    st.write("Share your experience with our dishes!")

    # ===================== CSV FILE PATH =====================
    CSV_FILE = "food_reviews.csv"

    # ===================== LOAD EXISTING REVIEWS =====================
    if os.path.exists(CSV_FILE):
        df_reviews = pd.read_csv(CSV_FILE)
    else:
        df_reviews = pd.DataFrame(columns=["Customer", "Food Item", "Rating", "Review", "Emotion", "Date"])

    # ===================== REVIEW FORM =====================
    with st.form("review_form", clear_on_submit=True):
        customer_name = st.text_input("👤 Your Name")
        food_item = st.text_input("🍔 Food Item")
        rating = st.slider("⭐ Rating", 1, 5, 5)
        review_text = st.text_area("📝 Write your review:", height=150)
        submitted = st.form_submit_button("💾 Submit Review")

        if submitted:
            if customer_name.strip() and food_item.strip() and review_text.strip():
                emotion_result = emotion_classifier(review_text)[0]["label"]

                new_review = {
                    "Customer": customer_name,
                    "Food Item": food_item,
                    "Rating": rating,
                    "Review": review_text,
                    "Emotion": emotion_result,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Append new review
                df_reviews = pd.concat([df_reviews, pd.DataFrame([new_review])], ignore_index=True)

                # Save to CSV
                df_reviews.to_csv(CSV_FILE, index=False)

                st.success(f"✅ Review saved with detected emotion: **{emotion_result}**")
            else:
                st.warning("⚠ Please fill in all fields.")


with tab6:
    st.subheader("📊 Reviews Summary")

    # CSV file path
    CSV_FILE = "food_reviews.csv"

    if os.path.exists(CSV_FILE):
        df_reviews = pd.read_csv(CSV_FILE)

        # Download button
        csv_data = df_reviews.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Reviews CSV",
            data=csv_data,
            file_name="food_reviews.csv",
            mime="text/csv"
        )

        # Summary stats
        st.write(f"**Total Reviews:** {len(df_reviews)}")
        st.write(f"**Average Rating:** {df_reviews['Rating'].mean():.2f} ⭐")

        # Most ordered food
        if not df_reviews['Food Item'].empty:
            most_food = df_reviews["Food Item"].mode()[0]
            st.write(f"**Most Ordered Dish:** {most_food}")

        # Emotion distribution
        if "Emotion" in df_reviews.columns:
            emotion_counts = df_reviews["Emotion"].value_counts()
            st.markdown("### 📊 Distribution of Emotion")
            st.bar_chart(emotion_counts)

        # Ratings distribution
        st.markdown("### 📊 Distribution of Ratings")
        st.bar_chart(df_reviews["Rating"].value_counts().sort_index())

    else:
        st.info("No reviews available yet.")