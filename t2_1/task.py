# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Assume you have a dataset in the following format
# [(query, product_title, product_description, relevance), ...]
# where relevance is a binary label indicating whether the product is relevant to the query.

# Dummy data for illustration purposes
training_data = [
    ("laptop", "Best Laptop Ever", "High-performance laptop", 1),
    ("phone", "Latest Smartphone", "Powerful smartphone with great features", 1),
    ("book", "Science Fiction Novel", "Exciting science fiction story", 0),
    # Add more data...
]

# Preprocess the data
queries, titles, descriptions, relevances = zip(*training_data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(queries + titles + descriptions)

query_seq = tokenizer.texts_to_sequences(queries)
title_seq = tokenizer.texts_to_sequences(titles)
desc_seq = tokenizer.texts_to_sequences(descriptions)

max_len = max(max(map(len, query_seq)), max(map(len, title_seq)), max(map(len, desc_seq)))

query_seq = pad_sequences(query_seq, maxlen=max_len)
title_seq = pad_sequences(title_seq, maxlen=max_len)
desc_seq = pad_sequences(desc_seq, maxlen=max_len)

# Model definition
embedding_dim = 50
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([query_seq, title_seq, desc_seq], relevances, epochs=5, batch_size=16, validation_split=0.2)

# Save the model
model.save("semantic_search_model.h5")

# Assume you have a database with product information

# Dummy product database
product_database = [
    {"title": "Best Laptop Ever", "description": "High-performance laptop", "id": 1},
    {"title": "Latest Smartphone", "description": "Powerful smartphone with great features", "id": 2},
    {"title": "Science Fiction Novel", "description": "Exciting science fiction story", "id": 3},
    # Add more products...
]

# Assume you have a web framework like Flask for the web application
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
loaded_model = tf.keras.models.load_model("semantic_search_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Preprocess the query
    query_seq = pad_sequences(tokenizer.texts_to_sequences([query]), maxlen=max_len)
    
    # Make a prediction
    prediction = loaded_model.predict(query_seq)[0][0]
    
    # Rank products based on the prediction (dummy ranking for illustration)
    ranked_products = sorted(product_database, key=lambda x: abs(prediction - 0.5))
    
    return render_template('results.html', query=query, ranked_products=ranked_products)

if __name__ == '__main__':
    app.run(debug=True)
