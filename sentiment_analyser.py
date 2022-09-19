# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:13:50 2022

@author: STPI0560
"""

### information ---------------------------------------------------------------
# This script creates an app that takes a sentence as an arguement and outputs
# a positive and negative sentiment score. The scentiment score comes from 
# model weights from the Roberta transformer model.

### import libraries ----------------------------------------------------------
import streamlit as st
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax

### functions -----------------------------------------------------------------
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='tf')
    output = model(**encoded_text)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict, scores[0], scores[1], scores[2]

### streamlit app interface ---------------------------------------------------
st.title('neap Sentiment Analyser App')
st.write('Please enter some text and click submit.')
form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')

### streamlit app output ------------------------------------------------------
if submit:
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    result = polarity_scores_roberta(user_input)
    label_pos = 'positive'
    score = result[3]
    st.success(f'{label_pos} sentiment (score: {score})')
    label_neg = 'negative'
    score = result[1]
    st.error(f'{label_neg} sentiment (score: {score})')