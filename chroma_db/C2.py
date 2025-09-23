import chromadb
import random
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import time
import statistics

client = chromadb.PersistentClient(path="/home/aina/Documents/fib/cbde-lab1/chroma_db")
collection = client.get_collection(name="bookcorpus_collection")

data = collection.get()
texts = data['documents']
doc_ids = data['ids']
embeddings = np.array(data['embeddings'])

# Seleccionem les 10 frases per comparar
query_ids = random.sample(doc_ids, 10)

times = []

for qid in query_ids:
    idx_query = doc_ids.index(qid)
    query_emb = embeddings[idx_query].reshape(1, -1)

    start = time.time()

    # Cosine similarity
    sims_cos = cosine_similarity(query_emb, embeddings)[0]
    sims_cos[idx_query] = -np.inf
    top2_cos_idx = np.argsort(-sims_cos)[:2]

    # Euclidean distance
    dists_euc = euclidean_distances(query_emb, embeddings)[0]
    dists_euc[idx_query] = np.inf
    top2_euc_idx = np.argsort(dists_euc)[:2]

    end = time.time()
    times.append(end - start)

    print(f"\nFrase base ({qid}): {texts[idx_query]}")

    print("  -> Top-2 més similars (Cosine Similarity):")
    for i in top2_cos_idx:
        print(f"      {texts[i]} (ID: {doc_ids[i]})")

    print("  -> Top-2 més similars (Euclidean Distance):")
    for i in top2_euc_idx:
        print(f"      {texts[i]} (ID: {doc_ids[i]})")

    
print(f"Temps mínim: {min(times):.6f} s")
print(f"Temps màxim: {max(times):.6f} s")
print(f"Temps mitjà: {statistics.mean(times):.6f} s")
print(f"Desviació estàndard: {statistics.stdev(times):.6f} s")
