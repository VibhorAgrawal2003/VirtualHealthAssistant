import io
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import requests

def get_batch_data():
    url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
    response = requests.get(url)

    try:
        data = response.json()  # Convert the response to JSON format
    except json.JSONDecodeError:
        print("Error: The API response is not in JSON format")
        return None

    # Preprocess the data and convert it to the desired format
    texts = []  # List to store the text data

    # Extract the relevant information from the API response and preprocess it
    for entry in data:
        text = entry["text"]
        # Preprocess the text as per your requirements
        # For example, you can tokenize, remove stopwords, etc.
        # Append the preprocessed text to the texts list
        texts.append(preprocess_text(text))

    # Convert the preprocessed texts to TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices(texts)

    # Define the padded shapes and create batches
    padded_shapes = ([None], ())
    batches = dataset.shuffle(1000).padded_batch(10, padded_shapes=padded_shapes)

    return batches

def preprocess_text(text):
    # Implement your preprocessing steps for the text data
    # For example, tokenization, removing stopwords, etc.
    return text

def get_model(embedding_dim=16):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5, 1))
    plt.show()

batches = get_batch_data()
if batches is not None:
    vocab_size = 10000  # Update with the appropriate vocabulary size
    model = get_model()
    history = model.fit(batches, epochs=10, validation_data=batches, validation_steps=20)
    plot_history(history)
