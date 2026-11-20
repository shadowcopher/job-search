import json
from search import search

labels = {q: set(v) for q, v in json.load(open("labels.json")).items() if v}
print(f"{'query':50} {'bm25':>5} {'vec':>5} {'hyb':>5}")
for q, rel in labels.items():
    r = []
    for mode in ("bm25", "vector", "hybrid"):
        ids = {x["id"] for x in search(q, mode, 10)}
        r.append(len(rel & ids) / len(rel))
    flag = "  <--" if max(r) - min(r) > 0.15 else ""
    print(f"{q[:50]:50} {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f}{flag}")