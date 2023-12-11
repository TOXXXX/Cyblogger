import time
import os
import gensim
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset_builder

# Load the "blog_authorship_corpus" dataset
builder = load_dataset_builder('blog_authorship_corpus')
builder.download_and_prepare()

ds_train = builder.as_dataset(split='train')
ds_val = builder.as_dataset(split='validation')

# Extract features (text) and labels from the dataset ## moved to train word2vec
# texts = ds_train['text']
# labels_train = ds_train['gender']  # Assuming "gender" is one of the labels

#Convert string label into numeric labels
label_mapping = {'female': 0, 'male': 1}  # Add more labels if necessary
labels_train = [label_mapping[label] for label in ds_train['gender']]   # change label


# Train Word2Vec on the text data
start_time = time.time()
tokenized_texts = [text.split() for text in ds_train['text']]
word2vec_model = gensim.models.Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

# Tokenize the text and convert to sequences of word indices
max_sequence_length = 100
sequences_train =[[word2vec_model.wv.key_to_index[word]+1 if word in word2vec_model.wv else 0 for word in text] for text in tokenized_texts]

data_train = pad_sequences(sequences_train, maxlen=max_sequence_length)

end_time = time.time()
tokenization_time = end_time - start_time
print(f"Tokenization Time: {tokenization_time} seconds")
# Pad sequences to have consistent length
data_train = pad_sequences(sequences_train, maxlen=max_sequence_length, dtype="long", value=0, truncating="post", padding="post")

# # Convert Word2Vec embeddings to a matrix for Keras Embedding layer
# embedding_matrix = np.zeros((len(word2vec_model.wv) + 1, word2vec_model.vector_size))
# for word, i in word2vec_model.wv():
#     embedding_matrix[i] = word2vec_model.wv[word]

# # Build the classification model with LSTM on top of Word2Vec embeddings
# model = Sequential([
#     Embedding(input_dim=len(word2vec_model.wv) + 1, output_dim=word2vec_model.vector_size, input_length=max_length, weights=[embedding_matrix], trainable=False),
#     LSTM(64, return_sequences=True),
#     LSTM(64),
#     Dense(128, activation='relu'),
#     Dense(2, activation='softmax')  # Assuming binary classification, adjust units accordingly
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# start_time = time.time()

# model.fit(input_data, labels, epochs=5, batch_size=32, validation_split=0.2)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Training Time: {elapsed_time} seconds")