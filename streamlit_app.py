import streamlit as st                              # Used for interface
from langchain.embeddings import OllamaEmbeddings   # Used for MiniLM embedding
from math import dist                               # Used for euclidian distance metric
from numpy import dot                               # Used for dot plot metric
from numpy import shape                             # Used for verbose embedding output
from numpy.linalg import norm                       # Used for cosine similarity metric
from json import dumps                              # Used for making data objects

# Function to make embeddings
def make_embedding(text, verbose=False):
    ollama_emb = OllamaEmbeddings(model='llama2')
    embedding = ollama_emb.embed_query(text)
    if verbose==True:
        print('Creating vectors for text "{}"'.format(text))
        print('Vector shape is {}'.format(shape(embedding)))
    return embedding

# Function to calculate and return metrics as JSON
def get_mini_LM_metrics(embedding1, embedding2):
    # Calculate metrics
    euclidean_distance = dist(embedding1, embedding2)
    dot_product = dot(embedding1, embedding2)
    cosine_similarity = dot_product / (norm(embedding1) * norm(embedding2))

    # Create a dictionary to store the metrics
    metrics_dict = {
        "model": "MiniLM",
        "euclidean_distance": euclidean_distance,
        "dot_product": dot_product,
        "cosine_similarity": cosine_similarity
    }

    return metrics_dict

# Streamlit app
st.title("Text Embeddings Metrics Comparison")

# Input text boxes
text1 = st.text_input("Enter the first text:")
text2 = st.text_input("Enter the second text:")

if st.button("Calculate Metrics"):
    if text1 and text2:
        emb1 = make_embedding(text1, verbose=False)
        emb2 = make_embedding(text2, verbose=False)
        metrics = get_mini_LM_metrics(emb1, emb2)

        # Display the metrics as JSON
        st.json(metrics)
    else:
        st.warning("Please enter both texts to calculate metrics.")

