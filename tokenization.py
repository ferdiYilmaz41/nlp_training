# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:05:43 2024

@author: ferdi
"""
import nltk
nltk.download("punkt_tab")

text="Hello world! How are you?"

#kelimeleri tokenlama
word_tokens=nltk.word_tokenize(text)
print(word_tokens)

#cümleleri tokenleme
sentence_token=nltk.sent_tokenize(text)
