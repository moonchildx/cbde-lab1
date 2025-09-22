import psycopg2
import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import time
import statistics

def connect_postgres():
    return psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='postgres',
        host='localhost'
    )

# Obtenim tots els embeddings en un diccionari {sentence_id: (chunk_id, embedding)}
def get_embeddings(conn):
    query = "SELECT sentence_id, chunk_id, embedding FROM embedding_table;"
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return {sid: (cid, np.array(embed)) for sid, cid, embed in rows}

# Obtenim les frases i les guarda en un diccionari {sentence_id: text}
def get_sentences(conn):
    query = "SELECT id, sentence FROM chunks_db;"
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return {sid: text for sid, text in rows}

# Retorna els top-k embeddings mes similars a un embedding donat
def top_k_similar(query_id, embeddings, k=2, metric="cosine"):
    query_emb = embeddings[query_id][1].reshape(1, -1)
    other_ids = [sid for sid in embeddings if sid != query_id]
    other_embs = np.vstack([embeddings[sid][1] for sid in other_ids])

    if metric == "cosine":
        sims = cosine_similarity(query_emb, other_embs)[0]
        top_indices = np.argsort(-sims)[:k]
    elif metric == "euclidean":
        dists = euclidean_distances(query_emb, other_embs)[0]
        top_indices = np.argsort(dists)[:k]
    else:
        raise ValueError("Metrica no suportada")
    
    return [other_ids[i] for i in top_indices]

def main():
    conn = connect_postgres()
    embeddings = get_embeddings(conn)
    sentences = get_sentences(conn)

    # Seleccionem 10 frases
    query_ids = random.sample(list(sentences.keys()), 10)

    times = []

    for qid in query_ids:
        print(f"\nFrase base ({qid}): {sentences[qid]}")
        start = time.time()

        top_cosine = top_k_similar(qid, embeddings, k=2, metric="cosine")
        print("  -> Top-2 similars (Cosine Similarity):")
        for tid in top_cosine:
            print(f"      [{tid}] {sentences[tid]}")

        top_euclidean = top_k_similar(qid, embeddings, k=2, metric="euclidean")
        print("  -> Top-2 similars (Euclidean Distance):")
        for tid in top_euclidean:
            print(f"      [{tid}] {sentences[tid]}")

        end = time.time()
        times.append(end - start)
    
    if times:
        print(f"Temps minim: {min(times):.5f} segons")
        print(f"Temps maxim: {max(times):.5f} segons")
        print(f"Temps mitja: {statistics.mean(times):.5f} segons")
        print(f"Desviacio estandard: {statistics.stdev(times):.5f} segons")

    conn.close()

if __name__ == "__main__":
    main()
