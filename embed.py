import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

con = sqlite3.connect("data/jobs.db")
df = pd.read_sql("SELECT id, text FROM jobs ORDER BY id", con)

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(df["text"].tolist(), batch_size=64,
                   normalize_embeddings=True, show_progress_bar=True)
np.save("data/emb.npy", emb)
print(emb.shape)  # expect (4000, 384)  