import os
from pyexpat import model
import streamlit as st
import torch
import string
import pynput
from transformers import BertTokenizer, BertForMaskedLM
from pynput import keyboard
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
import heapq

import pandas as pd

unique_words = pd.read_pickle(r'saved_dictionary.pkl')
id2word = pd.read_pickle(r'id2word.pkl')
word2id = pd.read_pickle(r'word2id.pkl')

WORD_LENGTH = 5

tmodel = load_model('next_word_model2.h5')

tokenizer = RegexpTokenizer(r'\w+')


#use joblib to fast your function

@st.cache()

def decode(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])




def encode(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx







def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
  return {'bert': bert}





def get_prediction_eos(input_text):
  try:
    input_text += ' <mask>'
    res = get_all_predictions(input_text, top_clean=int(top_k))
    return res
  except Exception as error:
    pass


# Trained -----------------------------------------------

def prepare_input(text):
    words = tokenizer.tokenize(text)
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(words):
        x[0, t, word2id[word]] = 1.
        
    return x

def sample(preds, top_n= 3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_next_word(text, n=3):
    x = prepare_input(text)
    preds = tmodel.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [id2word[idx] for idx in next_indices]

# -----------------------------------------------------------------------------


st.title("Next Word Prediction")
st.sidebar.text("Next Word Prediction")

top_k = st.sidebar.slider("How many words do you need", 1 , 25, 1) #some times it is possible to have less words
print(top_k)
model_name = st.sidebar.selectbox(label='Select Model to Apply',  options=['BERT', 'TrainM'], index=0,  key = "model_name")

if model_name.lower() == "bert":
  try:
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      
      input_text = st.text_area("Enter your text here")
      #click outside box of input text to get result
      res = get_prediction_eos(input_text)

      answer = []
      print(res['bert'].split("\n"))
      for i in res['bert'].split("\n"):
          answer.append(i)
      answer_as_string = "    ".join(answer)
      st.text_area("Predicted List is Here",answer_as_string,key="predicted_list")
  except Exception as e:
    print("SOME PROBLEM OCCURED")


if model_name.lower() == "trainm":
  try:
    input_text = st.text_area("Enter your text here")
    res = predict_next_word(input_text)

    answer = []
    for i in res:
      answer.append(i)
    answer_as_string = "    ".join(answer)
    st.text_area("Predicted List is Here",answer_as_string,key="predicted_list")
  except Exception as e:
    print(e)
