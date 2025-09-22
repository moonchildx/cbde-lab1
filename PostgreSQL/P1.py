import psycopg2
from psycopg2 import sql

from sentence_transformers import SentenceTransformer

import time
import statistics

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def connect_postgres():
    return psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='postgres',
        host='localhost'
    )

def create_embedding_table(conn):
    query = """
        CREATE TABLE IF NOT EXISTS embedding_table (
            sentence_id INT PRIMARY KEY,
            chunk_id INT,
            embedding FLOAT8[],
            CONSTRAINT fk_sentence FOREIGN KEY(sentence_id)
                REFERENCES chunk_db(id)
                ON DELETE CASCADE
        );
    """
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()

# Retorna totes les frases
def get_all_sentences(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, chunk_id, sentence FROM chunks_db;")
        return cur.fetchall()
    
def insert_embedding(conn, sentence_id, chunk_id, embedding):
    query = sql.SQL("INSERT INTO embeddings_table (sentence_id, chunk_id, embedding) VALUES (%s, %s, %s)")
    with conn.cursor() as cur:
        cur.execute(insert_query, (sentence_id, chunk_id, embedding))
    conn.commit()

# Genera i insereix embeddings per totes les frases
def generate_all_embeddings(conn):
    sentences = get_all_sentences(conn)
    
    insertion_times = []
    start_global = time.time()

    for sentence_id, chunk_id, sentence in sentences:
        start = time.time()

        embedding_vector = model.encode(sentence).tolist()
        insert_embedding(conn, sentence_id, chunk_id, embedding_vector)
        
        end = time.time()
        elapsed_time = end - start
        insertion_times.append(elapsed_time)

    end_global = time.time()
    print(f"Generació de embeddings completat en {end_global - start_global:.5f} segons")

    if insertion_times:
        print(f"Temps minim: {min(insertion_times):.5f} segons")
        print(f"Temps maxim: {max(insertion_times):.5f} segons")
        print(f"Temps mitja: {statistics.mean(insertion_times):.5f} segons")
        print(f"Desviacio estandard: {statistics.stdev(insertion_times):.5f} segons")

def main():
    conn = connect_postgres()
    create_embedding_table(conn)
    generate_all_embeddings(conn)
    conn.close()

if __name__ == "__main__":
    main()
