from datasets import load_dataset

import psycopg2

import time
import statistics

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Descarregem dataset Bookcorpus
def fetch_dataset(max_texts=50000):
    dataset = load_dataset("rojagtap/bookcorpus", split="train", trust_remote_code=True)
    return dataset['text'][:max_texts]

def split_sentences(text):
    return nltk.sent_tokenize(text)

def chunk_sentences(texts, chunk_size=10000):
    sentences = []
    chunks = []

    for text in texts:
        sentences.extend(split_sentences(text))
        if len(sentences) >= chunk_size:
            chunks.append(sentences[:chunk_size])
            sentences = sentences[chunk_size:]

    if sentences:
        chunks.append(sentences)

    return chunks
 
def create_table(conn):
    query = """
        CREATE TABLE chunks_db (
            id SERIAL PRIMARY KEY,
            chunk_id INT,
            sentence TEXT
        );
        """
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()

def insert_chunk(conn, chunk_id, sentences):
    query = "INSERT INTO chunks_db (chunk_id, sentence) VALUES (%s, %s)"
    
    start = time.time()
    with conn.cursor() as cur:
        for sentence in sentences:
            cur.execute(query, (chunk_id, sentence))
    
    conn.commit()
    end = time.time()

    return end - start

def connect_postgres():
    return psycopg2.connect(
        dbname='suppliers',
        user="postgres",
        password="postgres",
        host="localhost"
    )

def main():
    conn = connect_postgres()
    create_table(conn)

    texts = fetch_dataset()
    chunks = chunk_sentences(texts)

    times = []
    for i, chunk in enumerate(chunks):
        print(f"Insertant chunk {i+1}/{len(chunks)}")
        t = insert_chunk(conn, i, chunk)
        times.append(t)

    if times:
        print(f"Temps minim: {min(times):.5f} segons")
        print(f"Temps maxim: {max(times):.5f} segons")
        print(f"Temps mitja: {statistics.mean(times):.5f} segons")
        print(f"Desviacio estandard: {statistics.stdev(times):.5f} segons")

if __name__ == "__main__":
    main()