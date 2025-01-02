from sklearn.feature_extraction.text import CountVectorizer

#örnek metin
documents = [
    "Kedi çok tatlı bir hayvandır.",
    "Kedi ve köpekler çok tatlı hayvanlardır.",
    "Yılanlar çok tatlı değillerdir."]

# CountVectorizer nesnesi oluşturma
vectorizer_unigram = CountVectorizer()
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))

# Belgeleri vektörize etme
X_unigram = vectorizer_unigram.fit_transform(documents)
X_bigram = vectorizer_bigram.fit_transform(documents)
X_trigram = vectorizer_trigram.fit_transform(documents)

X_unigram_feature_names = vectorizer_unigram.get_feature_names_out()
X_bigram_feature_names = vectorizer_bigram.get_feature_names_out()
X_trigram_feature_names = vectorizer_trigram.get_feature_names_out()
# Kelime kümesini yazdırma
print("Unigram feature names:", X_unigram_feature_names)
print("Bigram feature names:", X_bigram_feature_names)
print("Trigram feature names:", X_trigram_feature_names)