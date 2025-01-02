from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Create a DataFrame
documents = [
    "Kedi çok tatlı bir hayvandır.",
    "Kedi ve köpekler çok tatlı hayvanlardır.",
    "Yılanlar çok tatlı değillerdir."
]

tf_idf_vectorizer = TfidfVectorizer()
X = tf_idf_vectorizer.fit_transform(documents)

feature_names = tf_idf_vectorizer.get_feature_names_out()
#print(X.toarray())

df_tfidf = pd.DataFrame(X.toarray(), columns=feature_names)
print(df_tfidf)