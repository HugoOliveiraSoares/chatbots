# things we need for NLP
import nltk
stemmer = nltk.stem.RSLPStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pandas as pd

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
    sentence_words = nltk.word_tokenize(sentence, language='portuguese')
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

p = bow("Cancer de mama", words, show_details=True)
print (p)
# print (classes)


# load our saved model
model.load('./Mymodel.tflearn')

# create a data structure to hold user context
context = {}

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
                # find a tag matching the first result
                if classe == results[0][0]:
                    return print(random.choice(sentencas))

            results.pop(0)


print('sintomas')
response('Sinais e sintomas')