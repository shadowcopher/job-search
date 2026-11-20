import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_sql("SELECT id, title, company_name FROM jobs ORDER BY id",
                 sqlite3.connect("data/jobs.db"))
emb = np.load("data/emb.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

q = model.encode(["python backend developer, entry level"], normalize_embeddings=True)
scores = emb @ q[0]
for i in scores.argsort()[::-1][:5]:
    print(round(float(scores[i]), 3), df.title[i], "-", df.company_name[i])