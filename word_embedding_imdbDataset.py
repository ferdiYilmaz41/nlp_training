import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
# Download the stop words list
nltk.download('stopwords')

# Read the dataset
df = pd.read_csv("data.csv")

# Function to clean text
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove short words (less than 3 characters)
    text = " ".join([word for word in text.split() if len(word) > 2])
    # Split text into words
    words = text.split()
    # Get the list of stop words
    stop_words = set(stopwords.words('english'))
    # Remove stop words
    text_wo_stopwords = [word for word in words if word.lower() not in stop_words]
    # Join words back into a single string
    return ' '.join(text_wo_stopwords)

# Function to tokenize text
def tocenize_text(text):
    return simple_preprocess(text)

# Apply the cleaning function to the documents
cleaned_text = df["text"].apply(clean_text)

# Apply the tokenization function to the cleaned text
tokenized_text = cleaned_text.apply(tocenize_text)

# Print tokenized text (optional)
# print(tokenized_text)

# Train a Word2Vec model
word2vec_model = Word2Vec(tokenized_text, vector_size=50, window=5, min_count=1, sg=0)

# Get word vectors from the model
word_vectors = word2vec_model.wv

# Get the list of words (limit to 500 words for visualization)
words = list(word_vectors.index_to_key[:500])

# Create a list of vectors for the words
vectors = [word_vectors[word] for word in words]

# Clustering with KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(vectors)
clusters = kmeans.labels_

# Dimensionality reduction with PCA
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

# Plotting the clusters
# plt.figure()
# plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters)

# Plotting the clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], c=clusters, cmap='viridis')


# Plotting the cluster centers
centers = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=200, label='Merkez')


# Annotating the words on the plot
for i, word in enumerate(words):
    ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1],reduced_vectors[i, 2], word, fontsize=10)

# Adding labels and title
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('3D PCA of Word Embeddings')

# Show the plot
plt.show()