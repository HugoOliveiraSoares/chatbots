import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

import wikipedia as wiki
import warnings
warnings.filterwarnings('ignore')

ignore_words = ["!", "@", "#", "$", "%", "*", "?", "'"]

wiki.set_lang("pt")
dataset = wiki.page("Breast_cancer").content

# Tokenização
words = nltk.word_tokenize(dataset)
sentences = nltk.sent_tokenize(dataset)


# Stem e lower cada palavra e remove duplicados

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

training = []
output = []

# training set, bag of words for each sentence
for doc in sentences:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append([bag])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print(train_x)
print()
print(train_y)


net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')