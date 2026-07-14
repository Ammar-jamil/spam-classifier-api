1. Spam Classifier

"A machine learning project that classifies SMS messages as Spam or Not Spam."

2. Technologies

* Python
* Pandas
* Scikit-learn
* NLTK
* Streamlit

3.Model

This project uses "TF-IDF Vectorizer" to convert text into numerical features and "Multinomial Naive Bayes (MNB)" for classification. MNB is a supervised machine learning algorithm that performs well on text classification tasks by predicting whether a message belongs to the Spam or Not Spam class based on the words it contains.

4.Performance

* Accuracy: 95.94%
* Precision: 97.00%

5.Files

* app.py
* project.py
* spam.csv
* spam_model.pkl
* vectorizer.pkl
* requirements.txt

6.Installation

pip install -r requirements.txt

7.Run
streamlit run app.py
