import streamlit as st                              # Used for interface
from langchain.embeddings import OllamaEmbeddings   # Used for MiniLM embedding
from langchain.embeddings import SpacyEmbeddings    # Used for Spacy embedding
from math import dist                               # Used for euclidian distance metric
from numpy import dot                               # Used for dot plot metric
from numpy import shape                             # Used for verbose embedding output
from numpy.linalg import norm                       # Used for cosine similarity metric
from json import dumps                              # Used for making data objects
import pandas as pd

#Function to make Spacy embeddings
def spacy_embedding(text, verbose=False):
    embedder = SpacyEmbeddings()
    embedding = embedder.embed_query(text)
    if verbose==True:
        print('Creating vectors for text "{}"'.format(text))
        print('Vector shape is {}'.format(shape(embedding)))
    return embedding

# Function to make MiniLM embeddings
def miniLM_embedding(text, verbose=False):
    embedder = OllamaEmbeddings(model='llama2')
    embedding = embedder.embed_query(text)
    if verbose==True:
        print('Creating vectors for text "{}"'.format(text))
        print('Vector shape is {}'.format(shape(embedding)))
    return embedding

# Function to calculate and return MiniLM metrics as JSON
def calculate_metrics(text1, text2, embedding_model):
    match embedding_model:
        case "MiniLM":
            embedding1 = miniLM_embedding(text1)
            embedding2 = miniLM_embedding(text2)
        case "SpaCy":
            embedding1 = spacy_embedding(text1)
            embedding2 = spacy_embedding(text2)
        case _:
            print('Unknown model {}'.format(embedding_model))
    # Calculate metrics
    euclidean_distance = dist(embedding1, embedding2)
    dot_product = dot(embedding1, embedding2)
    cosine_similarity = dot_product / (norm(embedding1) * norm(embedding2))

    # Create a dictionary to store the metrics
    metrics_dict = {
        "model": embedding_model,
        "size": shape(embedding1)[0],
        "euclidean_distance": euclidean_distance,
        "dot_product": dot_product,
        "cosine_similarity": cosine_similarity
    }

    return metrics_dict

def get_all_metrics(text1, text2):
    metrics_list = []
    metrics_list.append(calculate_metrics(text1, text2, 'MiniLM'))
    metrics_list.append(calculate_metrics(text1, text2, 'SpaCy'))
    return metrics_list


# Streamlit app
st.title("Text Embeddings Metrics Comparison")

# Input text boxes
text1 = st.text_input("Enter the first text:")
text2 = st.text_input("Enter the second text:")

if st.button("Calculate Metrics"):
    if text1 and text2:
        metrics = get_all_metrics(text1, text2)

        # Display the metrics as JSON
        #st.json(metrics)
        st.dataframe(pd.DataFrame(calculate_metrics))
    else:
        st.warning("Please enter both texts to calculate metrics.")

