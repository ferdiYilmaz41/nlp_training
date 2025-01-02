# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:24:33 2024

@author: ferdi
"""

import nltk

nltk.download("wordnet")
# %% stemming
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()

#örnek kelimeler
words = ["running","runner","runs","ran","better","go","went"]
stems = [stemmer.stem(w) for w in words]
print("sonuçlar: ", stems)

#%% lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas_verb= [lemmatizer.lemmatize(w,pos="v") for w in words]

lemmas_noun= [lemmatizer.lemmatize(w,pos="n") for w in words]
