import streamlit as st
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

st.title("Subjective Answers Evaluation")

# st.header("Question")
# ques = st.text_area("Enter Question:")

col1, col2 = st.columns(2)
with col1:
    st.header("Submitted Answer")
    sub_ans = st.text_area("Submitted Answer:", height=350).lower()
with col2:
    st.header("Correct Answer")
    corr_ans = st.text_area("Correct Answer:", height=350).lower()

sub_tokens = word_tokenize(sub_ans)
cor_tokens = word_tokenize(corr_ans)

stop_words = set(stopwords.words('english'))
sub_words = [word for word in sub_tokens if word.lower() not in stop_words]
cor_words = [word for word in cor_tokens if word.lower() not in stop_words]

sub_words = [word for word in sub_words if word.lower() not in string.punctuation]
cor_words = [word for word in cor_words if word.lower() not in string.punctuation]

total_words = set(sub_words).union(set(cor_words))
l1, l2 = [], []
for word in total_words:
    if word in sub_words:
        l1.append(1)
    else:
        l1.append(0)
    if word in cor_words:
        l2.append(1)
    else:
        l2.append(0)
if(st.button('Submit')):
    c = 0
    for i in range(len(total_words)):
        c += l1[i] * l2[i]
    cosine = c / float(sum(l1)*sum(l2)) ** 0.5
    st.text(["Answer Similarity:", cosine])