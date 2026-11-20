from fastapi import FastAPI, Query
from search import search

app = FastAPI(title="Job Search")


@app.get("/search")
def api_search(
    q: str,
    mode: str = Query("hybrid", pattern="^(bm25|vector|hybrid)$"),
    k: int = Query(10, ge=1, le=50),
):
    return {"query": q, "mode": mode, "results": search(q, mode, k)}