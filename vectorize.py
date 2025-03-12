from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index('interview-query')
data = pd.read_csv('./processed_data/dataset_iq_recommendations - entity_dataset.csv')
# print(data.head())
model = SentenceTransformer('all-MiniLM-L6-v2')
data = data.fillna('')

def encodeUpsert(data, start, iterations):
    chunk_size = 1
    start_idx = start
    num_iterations = iterations

    for iteration in range(num_iterations):
        end_idx = start_idx + chunk_size
        vectors_to_upsert = []

        for i in tqdm(range(start_idx, end_idx)):
            vectors_to_upsert.append({
                'id': str(i),
                'values': model.encode(data.loc[i, 'title']).tolist(),
                'metadata': {'title': data.loc[i, 'title'],
                             'description': data.loc[i, 'description'],
                              'roles': data.loc[i, 'roles'].split(', '),
                                'url': data.loc[i, 'url'],
                                  'category': data.loc[i, 'category']}
            })
            print(i)
        index.upsert(vectors=vectors_to_upsert)
        start_idx = end_idx

encodeUpsert(data, 501, 510)