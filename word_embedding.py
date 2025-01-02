import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#örnek veri
sentences = [
    "Kedi çok tatlı bir hayvandır.",
    "Hayvanlar biraz salaktır",
    "Köpekler çok zekidir.",
    "Kedi ve köpekler çok iyi arkadaştır",
    "Hayat gerçekten çok zor"
]

tocenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
#print(tocenized_sentences)

# Word2vec modeli
word2vec_model = Word2Vec(tocenized_sentences, vector_size=50, window=5, min_count=1, sg=0)
#print(word2vec_model.wv['kedi'])
# FastText modeli
fasttext_model = FastText(tocenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

def plot_embedded_words(model, title):
    word_vectors = model.wv
    word=list(word_vectors.index_to_key[:1000])
    vectors=[word_vectors[word] for word in word]
    pca = PCA(n_components=2)
    #PCA
    pca= PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)

    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111, projection='3d')

    #vectorleri çizme
    ax.scatter(reduced_vectors[:,0], reduced_vectors[:,1], reduced_vectors[:,2])

    #kelimeleri etiketleme
    for i, word in enumerate(word):
        ax.text(reduced_vectors[i,0], reduced_vectors[i,1], reduced_vectors[i,2], word, fontsize=10)

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

plot_embedded_words(word2vec_model, "Word2Vec")
plot_embedded_words(fasttext_model, "FastText")