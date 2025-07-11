from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import rag_search

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

@app.post("/api/search")
async def search(req: SearchRequest):
    try:
        results = rag_search(req.query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
