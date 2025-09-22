import chromadb
from sentence_transformers import SentenceTransformer

import time
import statistics

# Fitxer amb les 10.000 frases
txt_file_path = "sentences.txt"

# Carreguem les frases des del fitxer
with open(txt_file_path, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file.readlines()]

client = chromadb.PersistentClient(path="/home/aina/Documents/fib/cbde-lab1/chroma_db")
collection = client.get_or_create_collection(name="bookcorpus_collection")

# Inicialitzem model embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

times = []

# Afeguim frases amb embeddings
for idx, sentence in enumerate(sentences):
    start = time.time()

    embedding_vector = model.encode(sentence).tolist()
    collection.add(
        ids=[f"id_{idx}"],
        documents=[sentence],
        embeddings=[embedding_vector],
        metadatas=[{"text": sentence}]
    )
    end = time.time()
    times.append(end - start)

print(f"Temps mínim: {min(times):.6f} s")
print(f"Temps màxim: {max(times):.6f} s")
print(f"Temps mitjà: {statistics.mean(times):.6f} s")
print(f"Desviació estàndard: {statistics.stdev(times):.6f} s")