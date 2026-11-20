import sys
from search import search

q = " ".join(sys.argv[1:]) or "entry level python backend developer remote"
for mode in ("bm25", "vector", "hybrid"):
    print(f"\n=== {mode.upper()} ===")
    for r in search(q, mode, 5):
        print(r["rank"], r["title"], "-", r["company"])