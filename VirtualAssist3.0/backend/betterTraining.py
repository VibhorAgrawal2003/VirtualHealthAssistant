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

with open('intents.json') as content:
    data1 = json.load(content)

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', ':', ';']

for intent in data1['intents']:
    for pattern in intent['input']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])          


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
word = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Removing punctuations
import string
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation))

# Tokenize the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

# Manually pad the sequences
max_len = max(len(seq) for seq in train)
x_train = np.zeros((len(train), max_len))
for i, seq in enumerate(train):
    x_train[i, :len(seq)] = seq

# Encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print("Input shape:", input_shape)

# Define vocabulary
vocabulary = len(tokenizer.word_index)
print("Number of unique words:", vocabulary)

output_length = len(le.classes_)
print("Output length:", output_length)

# Creating the model
i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Training the model
train = model.fit(x_train, y_train, epochs=200)

model.save('chatbotmodel.h5', train)
print("Training done.")


# training = []
# output_empty = [0] * len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])

# random.shuffle(training)
# training = np.array(training)

# train_x = list(training[:, 0])
# train_y = list(training[:, 1])

# model = Sequential()
# model.add(Dense(256, input_shape = (len(train_x[0]),), activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation = "relu"))
# model.add(Dropout(0.5))
# # model.add(Dense(32, activation = "relu"))
# # model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation = 'softmax'))

# sgd = SGD(lr = 0.01, decay=1e-6, momentum = 0.9, nesterov = True)
# model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# hist = model.fit(np.array(train_x), np.array(train_y), epochs=50, batch_size=5, verbose = 1)

# model.save('chatbotmodel.h5', hist)
# print("Done")