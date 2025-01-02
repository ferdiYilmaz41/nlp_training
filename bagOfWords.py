from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Download the stop words list
nltk.download('stopwords')

# Read the data
df = pd.read_csv("data.csv")

# Extract text and labels
documents = df["text"]
labels = df["sentiment"]

# Function to clean text
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Split text into words
    words = text.split()
    # Get the list of stop words
    stop_words = set(stopwords.words('english'))
    # Remove stop words
    text_wo_stopwords = [word for word in words if word.lower() not in stop_words]
    # Join words back into a single string
    return ' '.join(text_wo_stopwords)

# Apply the cleaning function to the documents
cleaned_documents = documents.apply(clean_text)
print(cleaned_documents)

# Create a CountVectorizer instance
vectorizer = CountVectorizer()

# Fit and transform the cleaned documents
X = vectorizer.fit_transform(cleaned_documents[:100])

# Get feature names
featureNames = vectorizer.get_feature_names_out()

print("Vektör temsili")
print(X.toarray())

# Create a DataFrame with the bag-of-words representation
df_bow = pd.DataFrame(X.toarray(), columns=featureNames)
print(df_bow)

# Calculate word counts
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(featureNames, word_counts))

# Get the 5 most common words
most_common_5 = Counter(word_freq).most_common(5)
print("En çok kullanılan 5 kelime")
print(most_common_5)

def deneme():
    # Belgeler listesi
    documents = [
        "Kedi evde",
        "Kedi bahçede"
    ]

    # CountVectorizer nesnesi oluşturma
    vectorizer = CountVectorizer()

    # Belgeleri vektörize etme
    X = vectorizer.fit_transform(documents)

    # Kelime kümesini yazdırma
    print("kelime kümesi:", vectorizer.get_feature_names_out())

    # Vektör temsili yazdırma
    print("vektör temsili:")
    print(X.toarray())

    # Vektörlerin sparse matrix temsili
    print(X)


