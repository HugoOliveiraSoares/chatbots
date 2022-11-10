import nltk
from nltk import tokenize  
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

import wikipedia as wiki
import re

ignore_words = ["!", "@", "#", "$", "%", "*", "?", "'"]

nltk.download("stopwords")

wiki.set_lang("pt")
breast_cancer_content = wiki.page("Breast_cancer").content

classes = re.findall('=\s(.*)\s=',breast_cancer_content)
sentences = re.sub(r'=(.*)=', '',breast_cancer_content)
sentences = re.split(r'\n', sentences)
while '' in sentences:
    sentences.remove('')

# Tokenização
words = []
documents = []
for i in range(1,len(sentences)-1):
    w = tokenize.word_tokenize(sentences[1], language='portuguese')
    words.extend(w)
    documents.append((w, classes[i]))

# Stem e lower cada palavra e remove duplicados
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print(len(documents), "documents", documents)
print(len(classes), "classes", classes)
print(len(words), "words", words)

training = []
output = []
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for sent in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = sent
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

print(training)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(training[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(training, training, n_epoch=1000, batch_size=8, show_metric=True)
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


p = bow("fatores de risco", words, True)
print (p)

print(model.predict([p]))

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'train_x':training, 'train_y':training}, open( "Mytraining_data", "wb" ) )