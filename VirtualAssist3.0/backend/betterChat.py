import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import random
import nltk

lemmatizer = WordNetLemmatizer()

# Load the intents file
intents_file = "intents.json"
with open(intents_file) as file:
    intents = json.load(file)

# Load the preprocessed data
data_file = "data.pkl"
with open(data_file, "rb") as file:
    words, classes, train_x, train_y = pickle.load(file)

# Load the trained model
model_file = "model.h5"
model = keras.models.load_model(model_file)

# Set the threshold for classifying intents
ERROR_THRESHOLD = 0.25


def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag-of-words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {word}")
    return np.array(bag)


def predict_class(sentence):
    # Generate probabilities from the model
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # Filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


print("The Chat Assistant is Live!")
print("ChatBot: Hello! How can I assist you today?")

while True:
    message = input("User: ")
    if message.lower() == "quit":
        break

    ints = predict_class(message)
    response = get_response(ints, intents)
    print("ChatBot:", response)
