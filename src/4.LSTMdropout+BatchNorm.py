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
    LSTM(128, dropout=0.2, recurrent_dropout-0.2, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    LSTM(128),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification, adjust units accordingly
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
# input_data = np.array(input_data)
model.fit(data_train, labels_train, epochs=12, batch_size=32, validation_split=0.2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time} seconds ")

# Evaluate on validation set
tokenized_texts_val = [text.split() for text in ds_val['text']]
labels_val = [label_mapping[label] for label in ds_val['gender']]
sequences_val = [word2vec_model.wv.index_to_key.index(word) for word in tokenized_texts_val]
data_val = pad_sequences(sequences_val, maxlen=max_sequence_length)

_, accuracy = model.evaluate(data_val, labels_val)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')