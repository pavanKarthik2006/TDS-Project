import os
import json
import numpy as np
import aiohttp
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncpg

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS RAG API", docs_url="/api-docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
SIMILARITY_THRESHOLD = 0.67
MAX_RESULTS = 10

# Database connection pool
pg_pool = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

@app.on_event("startup")
async def startup():
    global pg_pool
    try:
        pg_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool established")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown():
    if pg_pool:
        await pg_pool.close()
        logger.info("PostgreSQL connection pool closed")

@app.get("/health")
async def health_check():
    try:
        async with pg_pool.acquire() as conn:
            return {
                "status": "healthy",
                "discourse_chunks": await conn.fetchval("SELECT COUNT(*) FROM discourse_chunks"),
                "markdown_chunks": await conn.fetchval("SELECT COUNT(*) FROM markdown_chunks")
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

def cosine_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def get_embedding(text: str) -> list:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://aipipe.org/openai/v1/embeddings",
            headers={"Authorization": API_KEY},
            json={"input": text, "model": "text-embedding-3-small"}
        ) as response:
            if response.status != 200:
                error = await response.text()
                logger.error(f"Embedding API error: {error}")
                raise HTTPException(status_code=502, detail="Embedding service unavailable")
            
            return (await response.json())["data"][0]["embedding"]

async def retrieve_context(query_embedding: list) -> List[LinkInfo]:
    results = []
    async with pg_pool.acquire() as conn:
        for table in ["discourse_chunks", "markdown_chunks"]:
            records = await conn.fetch(
                f"""SELECT content, url, embedding 
                    FROM {table} 
                    WHERE embedding IS NOT NULL"""
            )
            
            for record in records:
                try:
                    similarity = cosine_similarity(
                        query_embedding,
                        json.loads(record["embedding"])
                    )
                    if similarity >= SIMILARITY_THRESHOLD:
                        results.append({
                            "url": record["url"] or f"https://tds-docs.iitm.ac.in/{table}/{record['id']}",
                            "text": record["content"][:300] + "..."  # Truncate for response
                        })
                except json.JSONDecodeError:
                    continue

    return sorted(results, key=lambda x: x["text"], reverse=True)[:MAX_RESULTS]

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        embedding = await get_embedding(request.question)
        context = await retrieve_context(embedding)
        return QueryResponse(
            answer="Here are the most relevant resources:",
            links=[LinkInfo(**item) for item in context]
        )
    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# For local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000)  # Only accessible from local machine
