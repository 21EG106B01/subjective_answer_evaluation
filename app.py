import streamlit as st
import string
# from torchtext.vocab import GloVe
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer, util

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# glove = GloVe(name='6B', dim=100)

st.title("Subjective Answers Evaluation")

st.header("Question")
ques = st.text_area("Enter Question:")

col1, col2 = st.columns(2)
with col1:
    st.header("Submitted Answer")
    sub_ans = st.text_area("Submitted Answer:", height=350).lower()
with col2:
    st.header("Correct Answer")
    corr_ans = st.text_area("Correct Answer:", height=350).lower()

# Tokenizer
sub_tokens = word_tokenize(sub_ans)
cor_tokens = word_tokenize(corr_ans)

# Stopword Removal
stop_words = set(stopwords.words('english'))
sub_words = [word for word in sub_tokens if word.lower() not in stop_words]
cor_words = [word for word in cor_tokens if word.lower() not in stop_words]

# Punctuation Removal
sub_words = [word for word in sub_words if word.lower() not in string.punctuation]
cor_words = [word for word in cor_words if word.lower() not in string.punctuation]

# Lemmatizer
lemmatizer = WordNetLemmatizer()
sub_words = [lemmatizer.lemmatize(word) for word in sub_words]
cor_words = [lemmatizer.lemmatize(word) for word in cor_words]

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


model_new = SentenceTransformer('paraphrase-minilm-1')
ques_embeed = model_new.encode(sub_ans, show_progress_bar=True, convert_to_tensor=True)
submitted_embeed = model_new.encode(sub_ans,show_progress_bar=True,convert_to_tensor=True)
correct_embeed = model_new.encode(corr_ans,show_progress_bar=True,convert_to_tensor=True)


if(st.button('Check Similarity')):
    st.write(f"Answer Score: {round(util.cos_sim(submitted_embeed,correct_embeed).item()*10,1)} / 10")
    # c = 0
    # for i in range(len(total_words)):
    #     c += l1[i] * l2[i]
    # cosine = c / float(sum(l1)*sum(l2)) ** 0.5
    # st.write("Answer Similarity:", cosine)