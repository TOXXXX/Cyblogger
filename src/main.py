#######################################
##############Preprocess###############
#######################################

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

# Get rid of unclassified data, for label 'job'
ds_train = ds_train.filter(lambda example: example['job'] != 'indUnk')
ds_val = ds_val.filter(lambda example: example['job'] != 'indUnk')

# age_mapping = {
#     "10s": ["13", "14", "15", "16", "17"],
#     "20s": ["23", "24", "25", "26", "27"],
#     "30s": ["33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47"],
# }
# Assuming 'ages' is a list of age labels

# Extract features (text) and labels from the dataset ## moved to train word2vec
# texts = ds_train['text']
# labels_train = ds_train['gender']  # Assuming "gender" is one of the labels

#Convert string label into numeric labels
# label_mapping = {'male': 0, 'female': 1}  # Add more labels if necessary
# labels_train = [label_mapping[label] for label in ds_train['gender']]
# label_mapping = {'Aries': 0, 'Taurus': 1, 'Gemini': 2,  'Cancer': 3, 'Leo' : 4, 'Virgo': 5, 'Libra': 6, 'Scorpio' : 7, 'Sagittarius' : 8, 'Capricorn' :  9, 'Aquarius' : 10, 'Pisces' : 11}  # Add more labels if necessary
# labels_train = [label_mapping[label] for label in ds_train['horoscope']]   # change label
label_mapping = {'Student': 0, 'Arts': 1, 'Engineering': 1, 'Religion': 1, 'Architecture': 1, 'Technology': 1, 'RealEstate': 1, 'Publishing': 1, 'Communications-Media': 1, 'Education': 1, 'Banking': 1, 'Biotech': 1, 'Non-Profit': 1, 'Telecommunications': 1, 'Internet': 1, 'Government': 1, 'Manufacturing': 1, 'LawEnforcement-Security': 1, 'Marketing': 1, 'Sports-Recreation': 1, 'Law': 1, 'Military': 1, 'Museums-Libraries': 1, 'Science': 1, 'Advertising': 1, 'Consulting': 1, 'Accounting': 1, 'HumanResources': 1, 'BusinessServices': 1, 'Transportation': 1, 'Fashion': 1, 'Tourism': 1, 'Automotive': 1, 'Agriculture': 1, 'Chemicals': 1, 'Environment': 1, 'Construction': 1, 'InvestmentBanking': 1, 'Maritime': 1}
# label_mapping = {'Student': 0, 'Arts': 1, 'Engineering': 2, 'Religion': 3, 'Architecture': 4, 'Technology': 5, 'RealEstate': 6, 'Publishing': 7, 'Communications-Media': 8, 'Education': 9, 'Banking': 10, 'Biotech': 11, 'Non-Profit': 12, 'Telecommunications': 13, 'Internet': 14, 'Government': 15, 'Manufacturing': 16, 'LawEnforcement-Security': 17, 'Marketing': 18, 'Sports-Recreation': 19, 'Law': 20, 'Military': 21, 'Museums-Libraries': 22, 'Science': 23, 'Advertising': 24, 'Consulting': 25, 'Accounting': 26, 'HumanResources': 27, 'BusinessServices': 28, 'Transportation': 29, 'Fashion': 30, 'Tourism': 31, 'Automotive': 32, 'Agriculture': 33, 'Chemicals': 34, 'Environment': 35, 'Construction': 36, 'InvestmentBanking': 37, 'Maritime': 38}
# labels_train = [label_mapping[label] for label in ds_train['job']]
# label_mapping = {
#     13: 0, 14: 0, 15: 0, 16: 0, 17: 0,
#     23: 1, 24: 1, 25: 1, 26: 1, 27: 1,
#     33: 2, 34: 2, 35: 2, 36: 2, 37: 2,
#     38: 2, 39: 2, 40: 2, 41: 2, 42: 2,
#     43: 2, 44: 2, 45: 2, 46: 2, 47: 2,
#     48: 2
# }
labels_train = [label_mapping[label] for label in ds_train['job']]  #change the class name

# Train Word2Vec on the text data
start_time = time.time()
tokenized_texts = [text.split() for text in ds_train['text']]
word2vec_model = gensim.models.Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Tokenize the text and convert to sequences of word indices
max_sequence_length = 100
sequences_train =[[word2vec_model.wv.key_to_index[word]+1 if word in word2vec_model.wv else 0 for word in text] for text in tokenized_texts]

data_train = pad_sequences(sequences_train, maxlen=max_sequence_length)

end_time = time.time()
tokenization_time = end_time - start_time
print(f"Tokenization Time: {tokenization_time} seconds")
# Pad sequences to have consistent length
data_train = pad_sequences(sequences_train, maxlen=max_sequence_length, dtype="long", value=0, truncating="post", padding="post")

#######################################
##############Training#################
#######################################

# Convert Word2Vec embeddings to a matrix for Keras Embedding layer
embedding_matrix = np.zeros((len(word2vec_model.wv) + 1, word2vec_model.vector_size))
for word, i in word2vec_model.wv.key_to_index.items():
    embedding_matrix[i] = word2vec_model.wv[word]

# Convert data_train and labels to NumPy arrays
data_train = np.array(data_train)
labels_train = np.array(labels_train)

# Build the classification model with LSTM on top of Word2Vec embeddings
model = Sequential([
    Embedding(input_dim=len(word2vec_model.wv) + 1, output_dim=word2vec_model.vector_size, input_length=max_sequence_length, weights=[embedding_matrix], trainable=False),
    LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Adjust units and activation function according to number of labels
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # adam -> sgd categorical_crossentropy:one-hot-encoding(no need for mutual excluded classes)

# Train the model
start_time = time.time()
# input_data = np.array(input_data)
model.fit(data_train, labels_train, epochs=12, batch_size=32, validation_split=0.2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time} seconds ")



#######################################
##############Evaluation###############
#######################################
print(model.summary())

import numpy as np
# Evaluate on validation set
tokenized_texts_val = [text.split() for text in ds_val['text']]
#labels_val = [label_mapping[label] for label in ds_val['gender']]
# labels_val = [label_mapping[label] for label in ds_val['job']]
labels_val = [label_mapping[label] for label in ds_val['job']]
# sequences_val = [list(word2vec_model.wv.key_to_index.items()) for word in tokenized_texts_val]
# sequences_val = [
#     [word2vec_model.wv.index_to_key.index(word) + 1 for word in text if word in word2vec_model.wv]  # +1 because 0 is reserved for padding
#     for text in tokenized_texts_val
# ]
sequences_val =[[word2vec_model.wv.key_to_index[word]+1 if word in word2vec_model.wv else 0 for word in text] for text in tokenized_texts_val]

data_val = pad_sequences(sequences_val, maxlen=max_sequence_length)

# print(labels_val)
labels_val = np.array(labels_val)
_, accuracy = model.evaluate(data_val, labels_val)    # caution: takes only numpy array as input
print(f'Validation Accuracy: {accuracy * 100:.2f}%')