from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
host = os.getenv("PINECONE_HOST")

pc = Pinecone(api_key=api_key)
index = pc.Index(host=host)
model = SentenceTransformer('all-MiniLM-L6-v2')

query = 'Data Scientist'
xq = model.encode(query).tolist()
xc = index.query(vector=xq, top_k=5, include_metadata=True)
print(xc)


transformed_documents = [
    {
        'id': match['id'],
        'reranking_field': '; '.join([f"{key}: {value}" for key, value in match['metadata'].items()])
    }
    for match in xc['matches']
]

print(transformed_documents)
# Perform reranking based on the query and specified field
reranked_results_field = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=transformed_documents,
    rank_fields=["reranking_field"],
    top_n=2,
    return_documents=True,
)

print(reranked_results_field)