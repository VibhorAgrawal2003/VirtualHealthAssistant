import io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
from tensorboard.plugins import projector
import requests

API_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"

def fetch_data_from_api(api_url):
    response = requests.get(api_url)
    text = response.text  # Get the response text instead of JSON
    return text


def preprocess_text(text):
    # Perform text preprocessing steps here
    processed_text = text.lower()  # Example: converting to lowercase
    return processed_text

def get_dataset_from_text(text, encoder):
    encoded_text = encoder.encode(preprocess_text(text))
    dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
    return dataset

def get_batch_data(dataset):
    # Calculate the maximum sequence length
    max_seq_length = tf.reduce_max(tf.shape(dataset))

    # Set the padded shape to the maximum sequence length
    padded_shapes = ([max_seq_length],)

    # Create the padded batches
    batches = dataset.padded_batch(10, padded_shapes=padded_shapes)

    return batches





def get_model(encoder, embedding_dim=16):
    model = keras.Sequential([
        layers.Embedding(encoder.vocab_size, embedding_dim),
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

def retrieve_embeddings(encoder, model):
    out_vectors = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_metadata = io.open('meta.tsv', 'w', encoding='utf-8')

    weights = model.layers[0].get_weights()[0]
    for num, word in enumerate(encoder.subwords):
        vec = weights[num + 1]
        out_metadata.write(word + '\n')
        out_vectors.write('\t'.join([str(x) for x in vec]) + '\n')
    out_vectors.close()
    out_metadata.close()

# Fetch data from the API
text = fetch_data_from_api(API_URL)

# Preprocess the text
processed_text = preprocess_text(text)

# Build the encoder from corpus
encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus([processed_text], target_vocab_size=2**13)

# Save the encoder to a file
encoder.save_to_file('encoder_file')

# Load the encoder from the file
encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file('encoder_file')

# Create a dataset from the text data
dataset = get_dataset_from_text(processed_text, encoder)

# Get batches from the dataset
batches = get_batch_data(dataset)

# Get the model
model = get_model(encoder)

# Train the model on the batches
history = model.fit(batches, epochs=10)

# Plot the training history
plot_history(history)

# Retrieve embeddings and save them
retrieve_embeddings(encoder, model)

# Set up a logs directory, ...
