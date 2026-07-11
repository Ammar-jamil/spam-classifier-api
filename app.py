import pickle
import string
import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("spam_model.pkl", "rb"))
app = FastAPI(title="Spam Classifier API")
class MessageInput(BaseModel):
    text: str
def text_transformation(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.get("/")
def read_root():
    return {"message": "Spam Classifier API is running. Go to /docs to test it."}

@app.post("/predict")
def predict_spam(input_data: MessageInput):
    try:
        transformed = text_transformation(input_data.text)
        vector = tfidf.transform([transformed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            return {
                "input_text": input_data.text,
                "prediction": "Spam Message",
                "is_spam": True
            }
        else:
            return {
                "input_text": input_data.text,
                "prediction": "Not Spam",
                "is_spam": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


