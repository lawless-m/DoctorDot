from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from contextlib import asynccontextmanager

from backend.config import settings
from backend.models import (
    QueryRequest, QueryResponse, Citation,
    HealthResponse
)
from backend.embeddings import get_embedding_service
from backend.retrieval import create_vector_store
from backend.claude_client import ClaudeClient
from guardrails.hr_policies import create_guardrail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "embedding_service": None,
    "vector_store": None,
    "claude_client": None,
    "guardrail": None,
    "current_collection": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # Startup
    logger.info("Starting up application...")

    try:
        # Initialize embedding service (loads model on GPU)
        logger.info("Loading embedding model on GPU...")
        app_state["embedding_service"] = get_embedding_service()

        # Initialize Claude client
        logger.info("Initializing Claude client...")
        app_state["claude_client"] = ClaudeClient()

        # Load default collection and guardrail
        default_collection = "hr_policies"
        default_guardrail = "hr_policies"

        logger.info(f"Loading collection: {default_collection}")
        app_state["vector_store"] = create_vector_store(default_collection)
        app_state["current_collection"] = default_collection

        logger.info(f"Loading guardrail: {default_guardrail}")
        app_state["guardrail"] = create_guardrail(default_guardrail)

        logger.info("Application startup complete!")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")
    if app_state["vector_store"]:
        app_state["vector_store"].close()


app = FastAPI(
    title="HR Policy Chatbot API",
    description="Query company policies using semantic search and Claude",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    import torch

    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        embedding_model_loaded=app_state["embedding_service"] is not None,
        active_collection=app_state["current_collection"],
        duckdb_status="connected" if app_state["vector_store"] else "disconnected"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the chatbot with a question.

    Process:
    1. Check guardrails
    2. Embed question
    3. Search vector store
    4. Call Claude API
    5. Return response with citations
    """
    start_time = time.time()

    try:
        # Get components
        embedding_service = app_state["embedding_service"]
        vector_store = app_state["vector_store"]
        claude_client = app_state["claude_client"]
        guardrail = app_state["guardrail"]

        # Check if query should be rejected
        should_reject, rejection_reason = guardrail.should_reject_query(request.question)

        if should_reject:
            logger.info(f"Query rejected: {rejection_reason}")
            return QueryResponse(
                answer=guardrail.get_rejection_message(rejection_reason),
                citations=[],
                guardrail_triggered=True,
                rejection_reason=rejection_reason,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )

        # Embed the question
        logger.info(f"Embedding question: {request.question[:100]}...")
        query_embedding = embedding_service.embed_text(request.question)

        # Search vector store
        logger.info("Searching vector store...")
        retrieval_config = guardrail.get_retrieval_config()
        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=retrieval_config['top_k'],
            similarity_threshold=retrieval_config['similarity_threshold']
        )

        if not search_results:
            logger.warning("No relevant documents found")
            return QueryResponse(
                answer="I couldn't find any relevant information in our HR policies to answer your question. Please try rephrasing, or contact the HR team directly.",
                citations=[],
                guardrail_triggered=False,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )

        logger.info(f"Found {len(search_results)} relevant chunks")

        # Format context for Claude
        context = guardrail.format_context(search_results)

        # Get system prompt
        system_prompt = guardrail.get_system_prompt()

        # Call Claude API
        logger.info("Calling Claude API...")
        answer, claude_metadata = claude_client.generate_response(
            system_prompt=system_prompt,
            context=context,
            user_question=request.question,
            conversation_history=request.conversation_history
        )

        # Build citations
        citations = [
            Citation(
                document_name=chunk['document_name'],
                page_number=chunk.get('page_number'),
                chunk_text=chunk['chunk_text'][:200] + "...",  # Truncate for display
                relevance_score=chunk['similarity'],
                chunk_id=chunk['chunk_id']
            )
            for chunk in search_results
        ]

        processing_time = round((time.time() - start_time) * 1000, 2)

        logger.info(f"Query processed successfully in {processing_time}ms")

        return QueryResponse(
            answer=answer,
            citations=citations if request.include_citations else [],
            guardrail_triggered=False,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the current collection.
    """
    try:
        vector_store = app_state["vector_store"]
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        stats = vector_store.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """
    List available document collections.
    """
    collections_path = settings.COLLECTIONS_PATH
    collections = [
        d.name for d in collections_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    return {"collections": collections}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
