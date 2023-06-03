import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
import pickle
import random
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPool1D, Flatten
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Load the intents file
intents_file = "intents.json"
with open(intents_file) as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["input"]:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add documents with corresponding class labels
        documents.append((word_list, intent["tag"]))
        # Add to the list of classes
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Fit the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("model.h5")

# Save the necessary variables as a pickle file
data = [words, classes, train_x, train_y]
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)
