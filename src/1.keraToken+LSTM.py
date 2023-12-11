import time
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from datasets import load_dataset_builder

# Load the "blog_authorship_corpus" dataset
builder = load_dataset_builder('blog_authorship_corpus')
builder.download_and_prepare()
ds = builder.as_dataset(split='train')

# Extract features (text) and labels from the dataset
texts = ds['text']
labels = ds['gender']  # Assuming "gender" is one of the labels

# Convert string labels to numerical format
label_mapping = {'female': 0, 'male': 1}  # Add more labels if necessary
labels = [label_mapping[label] for label in labels]

# Tokenize the text
max_words = 10000  # Adjust as needed
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to have consistent length
max_sequence_length = 100  # Adjust as needed
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels)

# Build the model
embedding_dim = 50  # Adjust as needed
lstm_unit = 256


model = Sequential()

start_time = time.time()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=lstm_unit))
model.add(Dense(units=2, activation='softmax'))  # Assuming binary classification, adjust units accordingly
end_time = time.time()
elapsed_time = end_time - start_time
print(f"RNN-LSTM Embedding Time: {elapsed_time} seconds")
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
model.fit(data, labels, epochs=5, batch_size=32, validation_split=0.2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time} seconds")