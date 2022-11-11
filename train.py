import nltk
from nltk import tokenize  
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pandas as pd
import numpy as np
import tflearn
import tensorflow as tf
import random

import wikipedia as wiki
import re

ignore_words = ["!", "@", "#", "$", "%", "*", "?", "'"]

nltk.download("stopwords")

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
    sentences = tokenize.sent_tokenize(blocos[b], language='portuguese')
    for s in sentences:
        s = re.sub(r'\n', '', s)
        #Tokeniza cada palavra da sentença
        w = tokenize.word_tokenize(s)
        # adiciona na lista de words
        words.extend(w)
        df = pd.concat([df, pd.Series({'classes': classes[b], 'sentencas': w}).to_frame().T], ignore_index=True) 


# PREPARANDO O TREINAMENTO ==========================================================================================
training = []
output = []
output_empty = [0] * len(classes)

for i, line in df.iterrows():
    bag = [] # inicializa a bag of words

    #lista de tokens da sentença
    pattern_words = line['sentencas']
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
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('Mymodel.tflearn')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
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


p = bow("sintomas", words, True)
print (p)

print(model.predict([p]))

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes' :classes, 'train_x':train_x, 'train_y': train_y}, open( "Mytraining_data", "wb" ) )