import streamlit as st
import time
import streamlit.components.v1 as com
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

com.html("""
    

""")
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# html code

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    arr = ['be', 'further', 'does', 's', 'some', 'because', 'aren', "shan't", 'most', 'i', 'had', 'about', 'shan', 'on', 'nor', "won't", "isn't", 'they', "you've", 'why', 're', 'haven', 'while', "weren't", "mustn't", 'or', 'but', 'couldn', 'didn', 'you', 'too', 'myself', 't', 'own', 'were', 'than', 'that', 'by', 'he', 'isn', 'where', "should've", "you'd", 'hers', 'same', "wasn't", 'during', "you'll", "couldn't", 'do', "didn't", 'as', 'which', 'after', 'against', 'doing', "aren't", 'can', 'of', 'have', 'hadn', 'both', 'having', 'any', 'more', 'before', 'ain', 'mustn', 'was', "wouldn't", 'mightn', 'other', 'wasn', 'through', 'if', 'no', 'o', 'hasn', 'has', 'from', 'below', 'into', 'this', 'off', 'very', "needn't", 'won', 'to', 'd', 'the', 'been', 'don', 'then', 'our', 'their', 'under', 'yours', 'whom', 'at', 'between', 'them', 'these', 'an', 'out', "mightn't", 'me', 'itself', 'who', 'down', 'will', 'themselves', 'doesn', 'for', 'weren', 'what', "that'll", 'there', 'herself', 'over', 'wouldn', "hasn't", "you're", 'with', "haven't", 'her', "she's", 'each', 'ma', 'few', 've', 'once', "shouldn't", 'his', 'your', 'until', 'and', 'above', 'him', 'yourselves', 'm', 'here', 'not', 'shouldn', 'needn', 'ourselves', 'she', 'its', 'being', 'my', 'a', 'those', 'theirs', 'am', 'so', 'should', "it's", 'ours', 'up', 'in', 'are', 'again', 'it', 'such', 'yourself', 'how', 'all', 'just', "hadn't", 'y', 'is', "don't", 'we', 'when', 'now', 'only', 'himself', "doesn't", 'll', 'did']
    con=[port_stem.stem(word) for word in con if not word in arr]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title('Financial Fraud Detector app ')
    st.subheader("Write the Message below")
    sentence = st.text_area("",placeholder="Enter your message here",height=200)

    predict_btt = st.button("Predict")
    with st.spinner('Wait for it...'):
        if(predict_btt):
            time.sleep(1)
            
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success("✅ no issues")
            st.balloons()
        if prediction_class == [1]:
            st.error('Spam Message',icon="🚨")
            # st.error()