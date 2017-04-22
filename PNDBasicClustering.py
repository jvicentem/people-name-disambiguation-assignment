import re, os, numpy
import nltk
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from nltk import ngrams
from nltk.tag.stanford import StanfordNERTagger
from nltk.internals import find_jars_within_path, config_java


#http://stackoverflow.com/a/28327086 -> para solucionar error "list object is not callable", se ha de modificar el archivo nltk/cluster/util.py

# Read all documents contained into Thomas Baker directory
def read_file(file):
    with open(file, 'r') as myfile:
        data = ""
        lines = myfile.readlines()
        for line in lines:
            data = data + line
    return data



# Function used to separate into words of the diferent documents
def cluster_texts(texts, clustersNumber, distance):
    
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    # Get a list of unique terms
    unique_terms = list(set(collection)) #NUEVO. Para lemmatizaci�n comentar esta l�nea y descomentar la de abajo
    #unique_terms = collection # NUEVO. S�lo descomentar para lematizaci�n

    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF_IDF(f,unique_terms, collection)) for f in texts] # NUEVO
    print("Vectors created.")

    # Initialize the clusterer -> classify the words into groups
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

def IDF(document, unique_terms, collection):
    word_idf = []
    for word in unique_terms:
        word_idf.append(collection.idf(word))
    return word_idf

# We use that document in the analysis, it presents the best score in all the test performed
def TF_IDF(document, unique_terms, collection):
    word_tfidf = []
    for word in unique_terms:
        word_tfidf.append(collection.tf_idf(word, document))
    return word_tfidf


if __name__ == "__main__":
    folder = "Thomas_Baker" # Call to the folder which contains the different texts
    # Empty list to hold text documents.
    texts = []

    listing = os.listdir(folder) # create a dictionay with all the .txt
    listing.sort() # to fix indexes, we use the function sort -> we receive always the same result, not random


    # We use the stanford named entity relation package to text mining, which contains classifiers to analyze and compare the words

    #stanfordner_dir = '/home/jose/Descargas/stanford-ner-2016-10-31/' #NUEVO
    stanfordner_dir = './stanford-ner-2016-10-31/'
    ner_jarfile = stanfordner_dir + 'stanford-ner.jar'
    ner_modelfile = stanfordner_dir + 'classifiers/english.muc.7class.distsim.crf.ser.gz' #probar con otros
    st = StanfordNERTagger(ner_modelfile,ner_jarfile)


    # Read all files contained into listing, and tokenize the words. suddenly, filter the tokens which contains the condition
    for file in listing:
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()

            tokens = nltk.word_tokenize(raw)

            filter_tokens = []
            for token in tokens:
                if re.search('[a-zA-Z0-9]', token):
                    filter_tokens.append(token)
            tokens = filter_tokens

            named_entities = []
            tagged_words = st.tag(tokens)

            # Set a tag for all tokens to name the different entities
            for tagged_word in tagged_words:
                if tagged_word[1] != 'O':
                    named_entities.append(tagged_word[0])

            # Classify the different named entities into 3 ngrams (groups)
            n = 3
            ngramss = list(set(ngrams(named_entities, n)))
                         
            text = nltk.Text(ngramss) #/NUEVO
            #text = nltk.Text(tokens)
            texts.append(text)



    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    # A distance function is used to locate and compare the ambiguation (similarity) of the tokens
    distanceFunction  = "braycurtis"   #  BEST RAND SCORE: 0.893854748603352
    #distanceFunction = "cosine"       <- BEST RAND SCORE: 0.893854748603352
    #distanceFunction = "dice"         <- 0.8038540949759119
    #distanceFunction = "manhattan"    <- 0.2261521972132905
    #distanceFunction = "euclidean"    <- 0.0836012861736335
    #distanceFunction = "jaccard"      <- 0.06323687031082535
    #distanceFunction = "correlation"  <- 0.011312217194570151
    #distanceFunction = "hamming"      <- -0.058949624866023516


    # This array contains the results of our analysis given a distance function -> identify which thomas baker is named in each documents
    test = cluster_texts(texts,4,distanceFunction)

    print("test: ", test)
    # Gold Standard -> create an array with the default data (reference)
    reference =[0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 3, 0, 1, 2, 0, 1]
    print("reference: ", reference)

    # Evaluation -> print the final result of the disambiguation
    print("rand_score: ", adjusted_rand_score(reference,test))
