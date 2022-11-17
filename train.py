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

sp = spacy.load("pt_core_news_md")

ignore_words = ["!", "@", "#", "$", "%", "*", "?", "=", "(", ")", " "]

#DOWNLOAD DO TEXTO
wiki.set_lang("pt")
breast_cancer_content = wiki.page("Breast cancer").content

#CRIANDO UM DATAFRAME CATEGORIZANDO CADA SENTENÇA ====================================================================
# separa todos os titulos
classes = re.findall('=\s(.*)\s=',breast_cancer_content) 
classes.insert(0, 'Cancer de mama')

# separa os paragrafos do titulos
raw = re.split(r'=+\s(.*)\s=+',breast_cancer_content)  
blocos = []
for r in range(0, len(raw)-1, 2):
    blocos.append(raw[r])

#monta o dataframe com cada sentença e sua classe
d = {'classes': [], 'sentencas': []}
df = pd.DataFrame(d)

words = []
for b in range(0, len(blocos)-1):
    sentences = [str(sent) for sent in sp(blocos[b]).sents] 
    for s in sentences:
        s = re.sub(r'\n', '', s)
        if s == '' or '\n' in s:
            continue
        #Tokeniza cada palavra da sentença
        w = sp(s).text
        # adiciona na lista de words
        words.extend(w)
        df = pd.concat([df, pd.Series({'classes': classes[b], 'sentencas': s}).to_frame().T], ignore_index=True) 

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Salva o dataset em um csv
df.to_csv('breast_cancer_wiki.csv')

# PREPARANDO O TREINAMENTO ==========================================================================================
training = []
output = []
output_empty = [0] * len(classes)

for i, line in df.iterrows():
    bag = [] # inicializa a bag of words

    #lista de tokens da sentença
    pattern_words = sp(line['sentencas']).text
    # stem cada palavra
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_empty[classes.index(line['classes'])] = 1

    training.append([bag, output_row])


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

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

p = bow("cancer de mama", words, True)
print ("cancer de mama", p)

print(model.predict([p]))

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes' :classes, 'train_x':train_x, 'train_y': train_y}, open( "Mytraining_data", "wb" ) )