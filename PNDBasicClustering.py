import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from nltk.cluster import GAAClusterer
from sklearn.metrics.cluster import adjusted_rand_score
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

# WordNet only cares about 5 parts of speech.
# The other parts of speech will be tagged as nouns.

part = {
    'N' : 'n',
    'V' : 'v',
    'J' : 'a',
    'S' : 's',
    'R' : 'r'
}

wnl = WordNetLemmatizer()

def convert_tag(penn_tag):
    '''
    convert_tag() accepts the first letter of a Penn part-of-speech tag,
    then uses a dict lookup to convert it to the appropriate WordNet tag.
    '''
    if penn_tag in part.keys():
        return part[penn_tag]
    else:
        # other parts of speech will be tagged as nouns
        return 'n'


def tag_and_lem(element):
    '''
    tag_and_lem() accepts a string, tokenizes, tags, converts tags,
    lemmatizes, and returns a string
    '''
    # list of tuples [('token', 'tag'), ('token2', 'tag2')...]
    sent = pos_tag(word_tokenize(element)) # must tag in context
    return ' '.join([wnl.lemmatize(sent[k][0], convert_tag(sent[k][1][0]))
                    for k in range(len(sent))])

#cache de las stopwrods para mejorar rendimiento
stops = set(stopwords.words('english'))

def read_file(file):
    myfile = open(file,"r")
    data = ""
    lines = myfile.readlines()
    for line in lines:
        data = data + line
    myfile.close
    return data

def cluster_texts(texts, clustersNumber, distance):

    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(tfidf(f,unique_terms, collection)) for f in texts]

    print("Vectors created.")
    # initialize the clusterer
    clusterer = GAAClusterer(clustersNumber)
    clusters = clusterer.cluster(vectors, True)
    #clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
    #                                  linkage="average", affinity=distanceFunction)
    #clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf


#Mejora 1 - cambio del TF por TF-IDF    puntc : -0.14   a  0.16

def tfidf(document, unique_terms, collection):
	word_tfidf = []
	for word in unique_terms:
            if re.match('\w+', word):
    		        word_tfidf.append(collection.tf_idf(word, document))
	return word_tfidf

if __name__ == "__main__":
    folder = "Thomas_Baker"
    # Empty list to hold text documents.
    texts = []

    listing = os.listdir(folder)
    for file in listing:
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read().lower()
            f.close()
            #tokens = nltk.word_tokenize(raw)
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(raw.lower())

            text = nltk.Text(tokens)


            #Mejora 2 :Eliminar stopwords: punct: 0.16 a 0.54
            text = [i for i in text if i not in stops]


            #Mejora 4: Lemmatiser   empeora a 0.33
            #text = [tag_and_lem(i) for i in text if len(i)>2]
                        #Mejora 3: Stemmer  de 0.54 a 0.55
            #stemmer = SnowballStemmer("english")
            #text = [stemmer.stem(i) for i in text]

            print("___sinonimoooos")
            #mejora 5 : sinonimos
            #for word in text:
            #	wordNetSynset = wn.synsets(word)
            #	for synSet in wordNetSynset:
            #		for synWords in synSet.lemma_names():
            #			text.append(synWords[0])
            texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction ="cosine"
    #distanceFunction = "euclidean"  #0.744623655914

    test = cluster_texts(texts,4,distanceFunction)
    print("test: ", test)
    # Gold Standard
    reference =[0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 3, 0, 1, 2, 0, 1]
    print("reference: ", reference)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference,test))
