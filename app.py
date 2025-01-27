import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    # converting to lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    removedSC = list()
    for i in text:
        if i.isalnum():
            removedSC.append(i)

    text = removedSC[:]

    removedSWPC = list()
    for i in text:

        if i not in stopwords.words('english') and i not in string.punctuation:
            removedSWPC.append(i)

    text = removedSWPC[:]

    ps = PorterStemmer()
    stemmed = list()
    for i in text:
        stemmed.append(ps.stem(i))
    text = stemmed[:]
    return " ".join(text)

tk = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("model.pkl",'rb'))

st.title("SMS Spam Detection Model")
st.write("Made by Haridutt")

input_sms = st.text_input("Enter a SMS : ")
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_input = tk.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")