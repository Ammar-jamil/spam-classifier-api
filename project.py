import pandas as pd 

df=pd.read_csv("spam.csv",encoding="latin-1")

# print(df)

# print(df.info())

# print(df.describe())

# print(df.dtypes)

# print(df.head(10))

# print(df.duplicated().sum())

# print(df[df.duplicated()])
# print(df.columns)
# print(df)
df=df.drop_duplicates(keep="first")

df=df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"])

df=df.rename(columns={"v1":"label","v2":"message"})
# print(df)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["label"]=le.fit_transform(df["label"])
# print(df)

# print(df["label"].value_counts())

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,10))
# plt.pie(df["label"].value_counts(),labels=["ham","Spam"],autopct="%0.2f")
# plt.show()

import nltk 

df["num_characters"]=df["message"].apply(len)
df["num_words"]=df["message"].apply(lambda y: len(nltk.word_tokenize(y)))
df["num_sentences"]=df["message"].apply(lambda y:len(nltk.sent_tokenize(y)))

# summary statics of all messages
print(df[["num_characters","num_words","num_sentences"]].describe())

# summary statics of both ham ans spam messages of both
# print("Ham messsages") 
# print(df[df["label"]==0][["num_characters","num_words","num_sentences"]].describe())
# print("spam messges")
# spam messages summary statics
# print(df[df["label"]==1][["num_characters","num_words","num_sentences"]].describe())


import seaborn as sb
import matplotlib.pyplot as plt 

# plt.figure(figsize=(10,8))
# sb.histplot(df[df["label"]==0]["num_characters"],bins=50)
# plt.show()

# plt.figure(figsize=(10,8))
# sb.histplot(df[df["label"]==1]["num_characters"],bins=50)
# plt.show()

# plt.figure(figsize=(10,8))
# sb.histplot(df[df["label"]==0]["num_words"],bins=50)
# plt.show()

# plt.figure(figsize=(10,8))
# sb.histplot(df[df["label"]==1]["num_words"],bins=50)
# plt.show()
# plt.figure(figsize=(10,8))
# sb.histplot(df[df["label"]==0]["num_sentences"],bins=20)
# plt.show()

# plt.figure(figsize=(10,8))
# sb.histplot(df[df["label"]==1]["num_sentences"],bins=30)
# plt.show()
corr_matrix=df.select_dtypes(include="number").corr()

corr_matrix=df[["num_characters","num_words","num_sentences","label"]].corr()

# print(corr_matrix)

# sb.heatmap(corr_matrix,annot=True)
# plt.show()
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_transformation(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i) 

    text=y[:]
    y.clear()
    for i in text: 
        if i  not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

df['transformed_text']=df["message"].apply(text_transformation)


spam_corpus=[]
for msg in df[df["label"]==1]["transformed_text"].tolist():

    for word in msg.split():
        spam_corpus.append(word)

# print("Total words count in spams",len(spam_corpus))

# NOW MOST REPEATED COMMOM HAM MESSAGES 

from collections import Counter

common_40_words=Counter(spam_corpus).most_common(40)

fdata=pd.DataFrame(common_40_words)
# print(fdata)
# ------> now  total HAM messagges 

ham_corpus=[]
for msg in df[df["label"]==0]["transformed_text"].tolist():

    for word in msg.split():
        ham_corpus.append(word)

# print("Total words count in hams",len(ham_corpus))

from collections import Counter

common_45_words=Counter(ham_corpus).most_common(40)

f1data=pd.DataFrame(common_45_words)
# print(fdata)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.model_selection import train_test_split

# cv=CountVectorizer()
gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()
tfidf=TfidfVectorizer()

# X=cv.fit_transform(df["transformed_text"]).toarray()  # this is done using counvectrizer 
X=tfidf.fit_transform(df["transformed_text"]).toarray()

y=df["label"].values
# print(X)
# print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

gnb.fit(X_train,y_train)
ypred=gnb.predict(X_test)

# print(accuracy_score(y_test,ypred))
# print(confusion_matrix(y_test,ypred))
# print(precision_score(y_test,ypred))

mnb.fit(X_train,y_train)
ypred1 = mnb.predict(X_test)

# print(accuracy_score(y_test,ypred1))
# print(confusion_matrix(y_test,ypred1))
# print(precision_score(y_test,ypred1))

bnb.fit(X_train,y_train)
ypred2 = bnb.predict(X_test)

# print(accuracy_score(y_test,ypred2))
# print(confusion_matrix(y_test,ypred2))
# print(precision_score(y_test,ypred2))

import pickle

pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(mnb, open("spam_model.pkl", "wb"))  # save bnb as it has good accuracy and precision 

print("Model Saved Successfully")