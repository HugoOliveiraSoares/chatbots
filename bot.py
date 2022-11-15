# things we need for NLP
import nltk
stemmer = nltk.stem.RSLPStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import pandas as pd
import string
import spacy
from spacy.lang.pt import stop_words

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

sp = spacy.load("pt_core_news_md")

# restore all of our data structures
import pickle
data = pickle.load( open( "Mytraining_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


df = pd.read_csv('breast_cancer_wiki.csv')

documents = {}
for i, line in df.iterrows():
    if line['classes'] not in documents:
        documents.update({line['classes'] : [line['sentencas']]})
    else:
        documents[line['classes']].append(line['sentencas'])  


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='Mytflearn_logs')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words =[token.text for token in sp(sentence) ] 
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# load our saved model
model.load('./Mymodel.tflearn')

# create a data structure to hold user context
ERROR_THRESHOLD = 0.025
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    # results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    resultados = []
    for i, r in enumerate(results):
        if r > ERROR_THRESHOLD:
            resultados.append([i,r])
    # sort by strength of probability
    resultados.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in resultados:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

# Preprocessing
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return [token.lemma_.lower().translate(remove_punct_dict) for token in sp(text)]

def response(sentence):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for classe, sentencas in documents.items():
                # find a tag matching the first result
                if classe == results[0][0]:
                    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stop_words.STOP_WORDS)
                    sentencas.append(sentence)
                    tfidf = TfidfVec.fit_transform(sentencas)
                    vals = cosine_similarity(tfidf[-1], tfidf)
                    idx = vals.argsort()[0][-2]
                    flat = vals.flatten()
                    flat.sort()
                    req_tfidf = flat[-2]
                    if(req_tfidf != 0):
                        return print(sentencas[idx])
            results.pop(0)

user_response = 'Cancer de mama' 
print(user_response)
response(user_response)

user_response = 'sinais e sintomas' 
print(user_response)
response(user_response)

user_response = 'o que é cancer de mama' 
print(user_response)
response(user_response)

user_response = 'sinais' 
print(user_response)
response(user_response)

