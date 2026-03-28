from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List

from app.services.rag_pipeline import AgenticRAG
from app.core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Instantiate the heavy model into memory when API boots
logger.info("Spinning up internal RAG Subprocess handler from FastAPI Router.")
try:
    rag_agent = AgenticRAG()
except Exception as e:
    logger.critical(f"RAG Engine failed to initialize during mount. API will be unresponsive. {e}")
    rag_agent = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceContext(BaseModel):
    source: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceContext]

@router.post("/query", response_model=QueryResponse)
async def query_rag_engine(req: QueryRequest):
    """Answers user queries directly using local AI RAG Engine."""
    if not req.query.strip():
        logger.warning(f"Discarding empty query from generic user.")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="RAG Model Service Offline. Contact Administrator.")
        
    try:
        # LLM Invocation - Technically blocks, but FastAPI scales OK until vLLM refactor
        raw_answer, raw_docs = rag_agent.query(req.query, k=req.top_k)
        
        # Serialize Langchain Objects into Pydantic Safes
        sources = [
            SourceContext(
                source=d.metadata.get("source", "unknown metadata"), 
                snippet=d.page_content[:150]
            ) 
            for d in raw_docs
        ]
        
        return QueryResponse(answer=raw_answer, sources=sources)
        
    except Exception as e:
        logger.error(f"Inference Engine crashed during pipeline execution: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error within RAG engine.")
