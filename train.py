import nltk
from nltk import tokenize 
stemmer = nltk.stem.RSLPStemmer()
import pandas as pd
import numpy as np
import tflearn
import tensorflow as tf
import random

import wikipedia as wiki
import re
import spacy
from spacy.lang.pt import stop_words
import json

#Carrega o modulo em portugues do spacy
sp = spacy.load("pt_core_news_md")

# Pontuações a serem ignoradas
ignore_punct = ["!", "@", "#", "$", "%", "*", "?", "=", "(", ")", " "]

classes = []
words = []
documents = {}

# Importa o arquivo intents.json
intents = json.loads(open('intents.json').read())

# adiciona as tags em uma lista de classes
classes = [i['tag'] for i in intents['intents']]

# PERCORREMOS O ARRAY DE OBJETOS DO JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenizaçao dos patterns 
        # e adicionamos na lista de palavras
        word = [token for token in sp(pattern).text]
        words.extend(word)
        
        # adiciona aos documentos para identificarmos a tag para a mesma
        if intent['tag'] not in documents:
            documents.update({intent['tag'] : [pattern]})
        else:
            documents[intent['tag']].append(pattern)

#DOWNLOAD DO TEXTO BRUTO DA WIKIPEDIA
wiki.set_lang("pt")
breast_cancer_content = wiki.page("Breast cancer").content

#CRIANDO UM DATAFRAME E CATEGORIZANDO CADA SENTENÇA DO TEXTO BRUTO

fim_classe_intents = len(classes) # marca o fim das classes do arquivo intents
# separa todos os titulos e adiciona na lista de classes
classes.append('Cancer de mama')
classes.extend(re.findall('=\s(.*)\s=',breast_cancer_content))

# separa os paragrafos do titulos
raw = re.split(r'=+\s(.*)\s=+',breast_cancer_content)  
blocos = []
for r in range(0, len(raw)-1, 2):
    blocos.append(raw[r])

#monta o dataframe com cada sentença e sua classe
d = {'classes': [], 'sentencas': []}
df = pd.DataFrame(d)

for b in range(0, len(blocos)-1):
    sentences = [str(sent) for sent in sp(blocos[b]).sents] 
    for s in sentences:
        s = re.sub(r'\n', '', s)
        if s == '' or '\n' in s:
            continue
        #Tokeniza cada palavra da sentença
        w = [token for token in sp(s).text]
        # adiciona na lista de words
        words.extend(w)
        df = pd.concat([df, pd.Series({'classes': classes[fim_classe_intents + b], 'sentencas': s}).to_frame().T], ignore_index=True)
        if classes[fim_classe_intents + b] not in documents:
            documents.update({classes[fim_classe_intents + b] : [s]})
        else:
            documents[classes[fim_classe_intents + b]].append(s)

# stemização, diminui cada letra, remove duplicados e as pontuações 
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_punct]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Salva o dataset em um csv
df.to_csv('breast_cancer_wiki.csv')

# PREPARANDO O TREINAMENTO ==========================================================================================
training = []
output = []
output_empty = [0] * len(classes)

for classe in documents.keys():
    for sentenca in documents[classe]:
        bag = [] # inicializa a bag of words
    
        #lista de tokens da sentença
        pattern_words = [token for token in sp(sentenca).text]
        # stem cada palavra
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
    
        output_row = list(output_empty)
        output_empty[classes.index(classe)] = 1
    
        training.append([output_row, bag])

# embaralha o conjunto de treinamento 
random.shuffle(training)
training = np.array(training)

# cria duas listas de treino sendo x as classes e y as sentenças
train_x = list(training[:,1])
train_y = list(training[:,0])

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='Mytflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('Mymodel.tflearn')

def clean_up_sentence(sentence):
    sentence_words = [word for word in sentence if word not in stop_words.STOP_WORDS]
    # tokenize the pattern
    sentence_words =[token for token in sp(sentence).text ] 
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# retorna a bag of words array: 0 ou 1 para cada palavra na bag que existe na sentença 
def bow(sentence, words, show_details=False):
    # tokenização a sentença
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

p = bow("cancer de mama", words, True)
print ("cancer de mama", p)
p = bow("Ola", words, True)
print ("Ola", p)


print(model.predict([p]))

# salva as estruturas de dados
import pickle
pickle.dump( {'words':words, 'classes' :classes, 'train_x':train_x, 'train_y': train_y, 'documents' : documents}, open( "Mytraining_data", "wb" ) )