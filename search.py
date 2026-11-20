import re
import sqlite3
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

df = pd.read_sql("SELECT * FROM jobs ORDER BY id", sqlite3.connect("data/jobs.db"))
df = df.fillna("")  # empty cells would break JSON output
emb = np.load("data/emb.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize(s):
    return re.findall(r"[a-z0-9+#]+", s.lower())


bm25 = BM25Okapi([tokenize(t) for t in df["text"]])


def bm25_rank(q, n):
    scores = bm25.get_scores(tokenize(q))
    return scores.argsort()[::-1][:n]


def vector_rank(q, n):
    v = model.encode([q], normalize_embeddings=True)[0]
    scores = emb @ v
    return scores.argsort()[::-1][:n]


def hybrid_rank(q, n, rrf_k=60, pool=100):
    # Reciprocal Rank Fusion: a job ranked high in either list gets a high score
    fused = {}
    for ranking in (bm25_rank(q, pool), vector_rank(q, pool)):
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0) + 1 / (rrf_k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)[:n]


def search(q, mode="hybrid", k=10):
    fn = {"bm25": bm25_rank, "vector": vector_rank, "hybrid": hybrid_rank}[mode]
    results = []
    for rank, i in enumerate(fn(q, k), start=1):
        row = df.iloc[int(i)]
        results.append({
            "rank": rank,
            "id": int(row["id"]),
            "title": row["title"],
            "company": row["company_name"],
            "location": row["location"],
        })
    return results  