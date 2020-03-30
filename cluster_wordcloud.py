import random
from sklearn.metrics import silhouette_samples, silhouette_score
import xlrd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
import unidecode
import pandas as pd
import re
from textblob import TextBlob
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering


nltk.download('stopwords')
loc = ("dadosExcel.xls")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

dados = []
for i in range(1, sheet.nrows):
    linha = []
    for j in range(6, 36):
        value = sheet.cell_value(i, j)
        linha.append(value)
    dados.append(linha)



le = preprocessing.LabelEncoder()
ok = []
for j in range(2):
    x = []
    for i in range(len(dados)):
        x.append(dados[i][j])
    le.fit(x)
    ok = le.classes_
    x = le.transform(x)
    for i in range(len(dados)):
        dados[i][j] = x[i]

texts = []
for i in range(len(dados)):
    texts.append(dados[i][-1])
    dados[i][-1] = 0


# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

porter_stemmer = PorterStemmer()
rslpstemmer = nltk.RSLPStemmer()


def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9]", " ", str_input).lower().split()

    # words = [rslpstemmer.stem(word) for word in words]
    n_words = []
    for word in words:
	    if (len(word) > 4):                         
            n_words.append(word)
    return n_words


vec = TfidfVectorizer(
    stop_words=stopwords.words('portuguese'),
                      tokenizer=stemming_tokenizer,
                      analyzer='word',
                      ngram_range=(1, 1), lowercase=True, use_idf=True)

# Say hey vectorizer, please read our stuff
matrix = vec.fit_transform(texts)
r = matrix.toarray()

# And make a dataframe out of it
results = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

old_dados = []
for i in range(len(dados)):
    temp = []
    for j in range(1):
        temp.append(dados[i][j])
    old_dados.append(temp)
for i in range(len(r)):
    dados[i][-1] = 0
    for j in range(len(r[0])):
        try:
            dados[i].append(r[i][j])
        except:
            continue



def getIndiceSortido(grupo):
    tamanho = len(grupo)
    indices = [x for x in range(tamanho)]
    for i in range(tamanho):
        for j in range(i + 1, tamanho):
            if (grupo[j] > grupo[i]):
                temp = grupo[j]
                grupo[j] = grupo[i]
                grupo[i] = temp

                temp = indices[j]
                indices[j] = indices[i]
                indices[i] = temp
    return indices


df = r

stopwords = set(STOPWORDS)

#Adicionando a lista stopwords em português
new_words = []
#Esse arquivo pode ser encontrado em https://gist.github.com/alopes/5358189
with open("stopwords.txt", 'r') as f:
    [new_words.append(word) for line in f for word in line.split()]

new_stopwords = stopwords.union(new_words)

def show_wordcloud(freq, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=new_stopwords,
        max_words=200,
        max_font_size=80,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    # ).generate(str(data))
    ).generate_from_frequencies(freq)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)

    plt.show()
    #fig.savefig(title + '-' + str(random.randint(0, 100000)) + ".png")



clusterer = AgglomerativeClustering(n_clusters=5)
# clusterer = KMeans(n_clusters=5, random_state=1224)
# clusterer = DBSCAN()
# method = " DBSCAN "
method = " AgglomerativeClustering "
preds = clusterer.fit_predict(df)
labels = clusterer.labels_

for num_k_labels in range(min(labels), max(labels)+1):
    new_df = []
    for i in range(len(labels)):
        if(labels[i]==num_k_labels):#cluster 1
            new_df.append(df[i])
    frequences = {}
    names = vec.get_feature_names()
    for j in range(len(new_df[0])):
        coluna = []
        for i in range(len(new_df)):
            coluna.append(new_df[i][j])
        if(names[j] not in new_stopwords):
            frequences[names[j]] = int(100*sum(coluna)) +1

    show_wordcloud(frequences, title="IFG - Cluster"+ str(num_k_labels)+method)

