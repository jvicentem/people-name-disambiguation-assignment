import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from nltk.cluster import GAAClusterer
from sklearn.metrics.cluster import adjusted_rand_score
import string # NUEVO
from nltk.corpus import stopwords, wordnet # NUEVO
from nltk.stem import WordNetLemmatizer # NUEVO
from nltk.stem.porter import PorterStemmer # NUEVO
from nltk.stem import SnowballStemmer # NUEVO
from nltk.tokenize import wordpunct_tokenize #NUEVO
from nltk import ngrams #NUEVO
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger #NUEVO
from nltk.internals import find_jars_within_path, config_java #NUEVO
import re #NUEVO
# http://stackoverflow.com/a/28327086

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
    unique_terms = list(set(collection)) #NUEVO. Para lemmatización comentar esta línea y descomentar la de abajo
    #unique_terms = collection # NUEVO. Sólo descomentar para lematización

    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF_IDF(f,unique_terms, collection)) for f in texts] # NUEVO
    print("Vectors created.")

    # initialize the clusterer
    #clusterer = GAAClusterer(clustersNumber)
    #clusters = clusterer.cluster(vectors, True) 

    clusterer = AgglomerativeClustering(n_clusters=clustersNumber, linkage="average", affinity=distanceFunction) 
    clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf

def TF_IDF(document, unique_terms, collection): # NUEVO
    word_idf = []
    for word in unique_terms:
        word_idf.append(collection.tf_idf(word, document))

    return word_idf # /NUEVO

def IDF(document, unique_terms, collection): # NUEVO
    word_idf = []
    for word in unique_terms:
        word_idf.append(collection.idf(word))

    return word_idf # /NUEVO

if __name__ == "__main__":
    folder = "Thomas_Baker"
    # Empty list to hold text documents.
    texts = []

    listing = os.listdir(folder)
    listing.sort()

    stanfordner_dir = '/home/jose/Descargas/stanford-ner-2016-10-31/' #NUEVO
    ner_jarfile = stanfordner_dir + 'stanford-ner.jar'
    ner_modelfile = stanfordner_dir + 'classifiers/english.muc.7class.distsim.crf.ser.gz' #probar con otros
    st = StanfordNERTagger(ner_modelfile,ner_jarfile)
    
    # stanfordpost_dir = '/home/jose/Descargas/stanford-postagger-full-2016-10-31/' #NUEVO
    # post_jarfile = stanfordpost_dir + 'stanford-postagger.jar'
    # post_modelfile = stanfordpost_dir + 'models/english-bidirectional-distsim.tagger' #probar con otros
    # st = StanfordPOSTagger(post_modelfile,post_jarfile, java_options='-mx13000m')

    for file in listing:
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()

            tokens = nltk.word_tokenize(raw) 
            #tokens = nltk.word_tokenize(raw.lower()) # NUEVO no se puede usar con EN NER
            #tokens = wordpunct_tokenize(raw.lower()) #NUEVO no se puede usar con EN NER

            # stop = set(stopwords.words('english')) # NUEVO
            # filter_tokens = []
            # for token in tokens:
            #      if token not in string.punctuation and token not in stop:
            #          filter_tokens.append(token) 
            # tokens = filter_tokens 

            filter_tokens = []
            for token in tokens:
                if re.search('[a-zA-Z0-9]', token):
                    filter_tokens.append(token)
            tokens = filter_tokens
            # /NUEVO

            # # #Seleccionamos el lematizador. # NUEVO
            # wordnet_lemmatizer = WordNetLemmatizer()
            # lemmatizeds = []
            # nlemmas = []
            # for token in tokens: 
            #     lemmatized = wordnet_lemmatizer.lemmatize(token)
            #     lemmatizeds.append(lemmatized)
            #     # Obtenemos los lemmas consultando la base de datos de WordNet.
            #     list = wordnet.synsets(token)
            #     # Si encontramos alguna palabra relacionada obtenemos sus lemas y nos quedamos con el primero.
            #     if len(list) >= 1:
            #         lemma = list[0].lemma_names('eng')
            #         if len(lemma) > 1:
            #             nlemmas.append(lemma[0])
            #         else:
            #             nlemmas.append((token))
            #     # En caso contrario simplemente introducimos en la solución la palabra actual.
            #     else:
            #         nlemmas.append(token) 
            # tokens = nlemmas # /NUEVO

            # # Seleccionamos el steamer que deseados utilizar. # NUEVO
            # stemmer = SnowballStemmer('english') #No se puede usar con EN
            # #stemmer = PorterStemmer()
            # stemmeds = []
            # # Para cada token del texto obtenemos su raíz.
            # for token in tokens:
            #     stemmed = stemmer.stem(token)
            #     stemmeds.append(stemmed)
            # tokens = stemmeds # /NUEVO

            named_entities = []
            tagged_words = st.tag(tokens) 
            
            for tagged_word in tagged_words:
                if tagged_word[1] != 'O':
                    named_entities.append(tagged_word[0])

            n = 3
            ngramss = list(set(ngrams(named_entities, n)))
                         
            text = nltk.Text(ngramss) #/NUEVO           
            #text = nltk.Text(tokens) 
            texts.append(text)



    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction = "cosine" #<-
    #distanceFunction = "euclidean"
    #distanceFunction =  "l1"
    #distanceFunction =  "l2"
    #distanceFunction =  "manhattan"
    test = cluster_texts(texts,4,distanceFunction)

    print("test: ", test)
    # Gold Standard
    reference =[0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 3, 0, 1, 2, 0, 1]
    print("reference: ", reference)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference,test))

