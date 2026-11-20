import json
from search import search

labels = json.load(open("labels.json"))
labels = {q: set(v) for q, v in labels.items() if v}  # skip queries with no relevant jobs

print(f"{len(labels)} queries\n")
print(f"{'mode':8} {'MRR':>6} {'Recall@10':>10} {'Precision@10':>13}")
for mode in ("bm25", "vector", "hybrid"):
    rr, rec, prec = [], [], []
    for q, rel in labels.items():
        ids = [r["id"] for r in search(q, mode, 10)]
        hits = [i for i, x in enumerate(ids) if x in rel]
        rr.append(1 / (hits[0] + 1) if hits else 0)
        rec.append(len(rel & set(ids)) / len(rel))
        prec.append(len(rel & set(ids)) / 10)
    n = len(rr)
    print(f"{mode:8} {sum(rr)/n:6.3f} {sum(rec)/n:10.3f} {sum(prec)/n:13.3f}")