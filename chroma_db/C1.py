import chromadb
from sentence_transformers import SentenceTransformer

import time
import statistics

client = chromadb.PersistentClient(path="/home/aina/Documents/fib/cbde-lab1/chroma_db")
collection = client.get_collection(name="bookcorpus_collection")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

records = collection.get()
documents = records['documents']
doc_ids = records['ids']
embeddings = model.encode(documents, convert_to_tensor=False)

def update_doc(doc_id, doc_text, embedding_vector):
    start = time.perf_counter()
    collection.update(ids=[doc_id], documents=[doc_text], embeddings=[embedding_vector])
    end = time.perf_counter()
    return end - start

times = [update_doc(doc_id, doc, emb) for doc_id, doc, emb in zip(doc_ids, documents, embeddings)]

print(f"Temps mínim: {min(times):.6f} s")
print(f"Temps màxim: {max(times):.6f} s")
print(f"Temps mitjà: {statistics.mean(times):.6f} s")
print(f"Desviació estàndard: {statistics.stdev(times):.6f} s")
