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
    # remove stop words
    sentence_words = [word for word in sentence if word not in stop_words.STOP_WORDS]
    # tokeniza a sentença
    sentence_words =[token for token in sp(sentence).text ] 
    # stemiza cada palavra
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# retorna uma bag of words onde: 0 ou 1 para cada palavra na bag que exista na sentença 
def bow(sentence, words, show_details=False):
    # tokenize sentença e remove stop_words
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("encontrado na bag: %s" % w)
                break
    return(np.array(bag))

ERROR_THRESHOLD = 0.025
def classify(sentence):
    # gera as probabilidades do modelo
    results = model.predict([bow(sentence, words)])[0]
    # filtra as predicões a baixo de a cordo com o threshold
    resultados = []
    for i, r in enumerate(results):
        if r > ERROR_THRESHOLD:
            resultados.append([i,r])
    # ordena por força de probabilidade
    resultados.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in resultados:
        return_list.append((classes[r[0]], r[1]))
    # returna uma tupla de classes e probabilidade
    return return_list

def response(sentence):
    results = classify(sentence)
    # se tem a classificação então procuramos a resposta adequada
    if results:
        # loop enquanto tem resultados pra processar
        while results:
            for classe, sentencas in documents.items():
                s = []
                # encontra a classe correpondente 
                if classe == results[0][0]:
                    # Calcula o TF-IDF para encontar a frase mais adequada para retornar
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
        # Se percorrer por todos os resultados e não encontrar uma resposta, não pude entender a entrada do usuario 
        return print("Desculpe não consegui entender. Poderia repetir?") 

flag = True

print('')
print("Bot: Olá, meu nome é Bot.")
print("Bot: Pode me perguntar o que quiser sobre câncer de mama.")
print("Bot: Caso queira encerrar digite 'sair' ou 'tchau'!")

while flag:
    user_response = input("> ")
    if(user_response.lower() != 'sair' and user_response.lower() != 'tchau'):
        print("Bot: ",end="")
        response(user_response)
    else:
        flag=False
        print("Bot: Obrigado, volte sempre!")
