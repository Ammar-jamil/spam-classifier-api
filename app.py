import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data required for tokenization and stopwords
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize stemmer
ps = PorterStemmer()

# Load model and vectorizer 
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


# --- Streamlit UI ---
st.title("Spam Classifier")
st.write("Detect whether a message is Spam or Not Spam.")

# Text area for user input
input_text = st.text_area("Enter your message here...", height=150)

# Predict button
if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # 1. Preprocess
        transformed_text = text_transformation(input_text)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_text])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display result
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")