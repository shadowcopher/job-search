import json, os
from search import search

queries = [l.strip() for l in open("queries.txt", encoding="utf-8") if l.strip()]
path = "labels.json"
labels = json.load(open(path)) if os.path.exists(path) else {}

for q in queries:
    if q in labels:
        continue
    # pool the top 10 from every mode; sorting by title (not by mode) groups similar jobs
    # together and hides which mode found what
    pool = {}
    for mode in ("bm25", "vector", "hybrid"):
        for r in search(q, mode, 10):
            pool[r["id"]] = r
    items = sorted(pool.values(), key=lambda r: r["title"].lower())

    print(f"\nQUERY: {q}  ({len(items)} candidates)")
    for n, r in enumerate(items, start=1):
        print(f"  {n:2}. {r['title']} - {r['company']} | {r['location']}")

    a = input("Relevant numbers (e.g. 1 3 5) | a = all | a 2 7 = all except 2 and 7 | Enter = none: ")
    nums = {int(x) for x in a.replace(",", " ").split() if x.isdigit()}
    if a.strip().lower().startswith("a"):
        chosen = [r for n, r in enumerate(items, start=1) if n not in nums]
    else:
        chosen = [r for n, r in enumerate(items, start=1) if n in nums]
    labels[q] = [r["id"] for r in chosen]
    json.dump(labels, open(path, "w"), indent=2)  # progress saved after every query