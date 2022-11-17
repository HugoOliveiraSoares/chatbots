import nltk
stemmer = nltk.stem.RSLPStemmer()

import numpy as np
import tflearn
import pandas as pd
import string
import spacy
from spacy.lang.pt import stop_words
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Carrega o modulo em portugues do spacy
sp = spacy.load("pt_core_news_md")

# restaura as estruturas de dados
data = pickle.load( open( "Mytraining_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
documents = data['documents']

# constroi a rede neural
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define o modelo e configura o tensorboard
model = tflearn.DNN(net, tensorboard_dir='Mytflearn_logs')

# Carrega o modelo 
model.load('./Mymodel.tflearn')

# Preprocessamento

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return [token.lemma_.lower().translate(remove_punct_dict) for token in sp(text)]

def clean_up_sentence(sentence):
    sentence_words = [word for word in sentence if word not in stop_words.STOP_WORDS]
    # tokenize the pattern
    sentence_words =[token for token in sp(sentence).text ] 
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
                break
    return(np.array(bag))

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

def response(sentence):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for classe, sentencas in documents.items():
                s = []
                # find a tag matching the first result
                if classe == results[0][0]:
                    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stop_words.STOP_WORDS)
                    s.extend(sentencas)
                    s.append(sentence)
                    tfidf = TfidfVec.fit_transform(s)
                    vals = cosine_similarity(tfidf[-1], tfidf)
                    idx = vals.argsort()[0][-2]
                    flat = vals.flatten()
                    flat.sort()
                    req_tfidf = flat[-2]
                    if(req_tfidf != 0):
                        return print(s[idx])
            results.pop(0)
        return print("Desculpe não consegui entender. Poderia repetir?")

flag = True

print('')
print("Bot: Olá, meu nome é Bot.")
print("Bot: Pode me perguntar o que quiser sobre câncer de mama.")
print("Bot: Caso queira encerrar digite 'sair'!")

while flag:
    user_response = input("> ")
    if(user_response.lower() != 'tchau'):
        print("Bot: ",end="")
        response(user_response)
    else:
        flag=False
        print("Bot: Obrigado, volte sempre!")
