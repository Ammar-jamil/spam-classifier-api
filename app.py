import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("spam_model.pkl", "rb"))

def text_transformation(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

st.title("Spam Classifier")
st.write("Detect whether a message is Spam or Not Spam.")

input_text = st.text_area("Enter your message here...", height=150)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        transformed_text = text_transformation(input_text)
        
        vector_input = tfidf.transform([transformed_text])
        
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.error(" Spam Message")
        else:
            st.success(" Not Spam")
