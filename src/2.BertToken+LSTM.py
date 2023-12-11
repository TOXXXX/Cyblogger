# BERT embedding + LSTM netwrok
import time
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GlobalAveragePooling1D

from datasets import load_dataset_builder

# Load the "blog_authorship_corpus" dataset
builder = load_dataset_builder('blog_authorship_corpus')
builder.download_and_prepare()
ds = builder.as_dataset(split='train')

# Extract features (text) and labels from the dataset
texts = ds['text']
labels = ds['gender']  # Assuming "gender" is one of the labels

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize the text
max_length = 100  # Adjust as needed
input_ids = [tokenizer.encode(text, max_length=max_length, truncation=True) for text in texts]

# Pad sequences to have consistent length
input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", value=0, truncating="post", padding="post")

# Get BERT embeddings
start_time = time.time()

bert_embeddings = bert_model(input_ids)[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"BERT Embedding Time: {elapsed_time} seconds")

# Build the classification model with LSTM on top of BERT embeddings
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(max_length, bert_embeddings.shape[-1])),
    LSTM(64),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Assuming binary classification, adjust units accordingly
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()

model.fit(bert_embeddings, labels, epochs=5, batch_size=32, validation_split=0.2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time} seconds")