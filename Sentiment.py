import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, roc_auc_score, classification_report
from PIL import Image

# Import data
dataset='C:/Users/macheampong.PINNACLELIFE/Text_Analytics/Labelled_stories.txt'
with open(dataset,'r',encoding='utf-8') as file:
    lines=file.readlines()

# Pass lines into 2 columns
parsed_data=[]

for line in lines:
    parts=line.strip().split('\t')
    if len(parts)==2:
        story, sentiments=parts
        parsed_data.append([story,sentiments])
    else:
        parsed_data.append(['Unknown',line.strip()])

df_text=pd.DataFrame(parsed_data,columns=['story','sentiments'])

# Data Cleaning
def clean_text(texts):
    cleaned=[]
    for text in texts:
        text=text.lower() # converts all text into lower cases
        text=re.sub(r'<.*?'," ",text) # removes html tags
        text = re.sub(r'[^\w\s]', " ", text) # Removes punctuation
        text = re.sub(r'\d+', " ", text) # Removes numbers
        text = re.sub(r'\s+', " ", text).strip() # Removes white spaces
        cleaned.append(text)
    return cleaned

# Clean the story
df_clean=df_text.copy()
df_clean['story']= clean_text(df_text['story'])


def page1():
    st.subheader('Dataset')
    #displaying data
    if st.checkbox('Show Unclean Data'):
        st.write(df_text)

    if st.checkbox('Show clean Data'):
        st.write(df_clean)

def page2():
    st.subheader('Sentiment Analysis')
    # Tokenize the entire corpus
    words = word_tokenize(' '.join(df_clean['story']))
    tok_text = pd.DataFrame({'Tokens': words})

    # Display stop words from the NLTK
    stop_words_disp = set(stopwords.words('english'))

    # Removal of the stopwords
    filtered_text = [word for word in words if word.lower not in stop_words_disp]



    #Separate the predictors from the output
    st.write('Prediction')
    x_predictor=df_clean['story']
    y_output=df_clean['sentiments']

    # Holdout (test_train)
    x_train, x_test, y_train,y_test=train_test_split(x_predictor,y_output,test_size=0.1, random_state=40)

    #Feature Selection using TF-IDF
    tfidf=TfidfVectorizer()
    x_train_tfidf=tfidf.fit_transform(x_train)
    x_test_tfidf=tfidf.transform(x_test)


    #Initiate the algorithms multinomial Naive bayes
    nb=MultinomialNB()
    naiveb=nb.fit(x_train_tfidf,y_train)

    #Prediction
    predict_x=naiveb.predict(x_test_tfidf)

    # Evaluate
    st.write(accuracy_score(y_test,predict_x))

    #Create user form
    user_input=st.text_area("Enter Text for Analysis")

    file=st.file_uploader("Upload file",type=['txt','csv'])
    if file is not None:
        file_content=file.read().decode('utf-8')
        st.success('File uploaded successfully')

    # Predict button creation
    if st.button("Predict"):
        if user_input.strip():
            cleaned_input = [user_input]  # Assuming clean_text is already implemented
            input_tfidf = tfidf.transform(cleaned_input)
            prediction = nb.predict(input_tfidf)[0]

            if prediction == 'Positive':
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Sentiment is: 😊 {prediction}")
                with col2:
                    image = Image.open("sabinus2.png")
                    st.image(image)
            else:
                col3, col4 = st.columns(2)
                with col3:
                    st.write(f"Sentiment is: ☹️ {prediction}")
                with col4:
                    image1 = Image.open("sabinus.png")
                    st.image(image1)

        if file is not None:
            try:
                if file.name.endswith('.txt'):
                    file_content = file.read().decode('utf-8').splitlines()
                    file_df = pd.DataFrame({'story': file_content})
                elif file.name.endswith('.csv'):
                    file_df = pd.read_csv(file)

                # Predict sentiments for file data
                file_tfidf = tfidf.transform(file_df['story'])
                file_predictions = nb.predict(file_tfidf)
                file_df['Predicted Sentiment'] = file_predictions

                st.success("File processed successfully!")
                st.dataframe(file_df)

            except Exception as e:
                st.error(f"Error processing file: {e}")
        elif not user_input.strip():
            st.warning("Please enter text or upload a file before clicking Predict.")


def page3():
    st.subheader('Emotion Detection')

def page4():
    st.subheader('Topic Modeling')

def page5():
    st.subheader('Conclusion')

pages={
    'Dataset':page1,
    'Sentiment Analysis':page2,
    'Emotion Detection':page3,
    'Topic Modeling':page4,
    'Conclusion':page5
}
#linking the pages
select_page=st.sidebar.selectbox('Select Page',list(pages.keys()))

#display
pages[select_page]()