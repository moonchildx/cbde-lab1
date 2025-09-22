import chromadb
from sentence_transformers import SentenceTransformer

import time
import statistics

# Fitxer amb les 10.000 frases
txt_file_path = "sentences.txt"

# Carreguem les frases des del fitxer
with open(txt_file_path, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in f.readlines()]

