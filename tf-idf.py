from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup
# Download the stop words list
nltk.download('stopwords')
# Real dataset works
df = pd.read_csv("data.csv")
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove html tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove sequences of more than two 'z' characters
    text = re.sub(r'z{3,}', '', text, flags=re.IGNORECASE)
    # Remove 'br' tags
    text = re.sub(r'\bbr\b', '', text, flags=re.IGNORECASE)
    # Split text into words
    words = text.split()
    # Get the list of stop words
    stop_words = set(stopwords.words('english'))
    # Remove stop words
    text_wo_stopwords = [word for word in words if word.lower() not in stop_words]
    # Join words back into a single string
    return ' '.join(text_wo_stopwords)
cleaned_text= df["text"].apply(clean_text)
tf_idf_vectorizer = TfidfVectorizer()
X= tf_idf_vectorizer.fit_transform(cleaned_text)

feature_names = tf_idf_vectorizer.get_feature_names_out()

tf_idf_score=X.mean(axis=0).A1

tf_idf = pd.DataFrame({"word":feature_names, "score":tf_idf_score})
# Sort the DataFrame by tf-idf score in descending order
tf_idf_sorted = tf_idf.sort_values(by="score", ascending=False)
# Get the 5 most important words
print(Counter(dict(zip(feature_names, tf_idf_score))).most_common(5))
print(tf_idf_sorted)







# # Basic initialization for tf-idf
# documents = [
#     "Kedi çok tatlı bir hayvandır.",
#     "Kedi ve köpekler çok tatlı hayvanlardır.",
#     "Yılanlar çok tatlı değillerdir."
# ]

# tf_idf_vectorizer = TfidfVectorizer()
# X = tf_idf_vectorizer.fit_transform(documents)

# feature_names = tf_idf_vectorizer.get_feature_names_out()
# #print(X.toarray())

# df_tfidf = pd.DataFrame(X.toarray(), columns=feature_names)
# #print(df_tfidf)
