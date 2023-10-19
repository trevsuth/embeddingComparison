from langchain.embeddings import OllamaEmbeddings   #import MiniLM embeddings
from math import dist                               #needed for euclidian calc
from numpy import dot                               #needed for dot product
from numpy.linalg import norm #needed for cosine calculation
from json import dumps #formatting for metrics return

def make_embedding(text):
    ollama_emb = OllamaEmbeddings(model='llama2',)
    embedding = ollama_emb.embed_query(text1)
    return embedding

def get_metrics(embedding1, embedding2):
    #calculate metrics
    euclidian_distance = dist(embedding1, embedding2)
    dot_product = dot(embedding1, embedding2)
    cosine_simalarity = dot_product /(norm(embedding1)*norm(embedding2))
    
    #make JSON
    metrics_dict = {
        "euclidian_distance": euclidian_distance,
        "dot_product": dot_product,
        "cosine_simalarity": cosine_simalarity
    }

    json_string = dumps(metrics_dict, indent=4)
    return json_string

#main block
text1 = 'Hello there.'
text2 = 'How are you?'

emb1 = make_embedding(text1)
emb2 = make_embedding(text2)

metrics = get_metrics(emb1, emb2)

print(metrics)

