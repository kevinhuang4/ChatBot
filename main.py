import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

import nltk
nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

# print(data["intents"])
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(word) for word in words]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # one-hot encoded

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc if w != "?"]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for sentence_word in sentence_words:
        for i, word in enumerate(words):
            if word == sentence_word:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking to your bot! (type exit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "exit":
            break
        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for intent in data["intents"]:
                if intent["tag"] == tag:
                    responses = intent["responses"]
            print(random.choice(responses))
        else:
            print("I don't really know. Ask another question?")
chat()