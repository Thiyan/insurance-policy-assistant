import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config.config import APP_CONFIG
from app.exception.policy_application_exception import PolicyApplicationException
from app.model.api_model import QueryResponse, QueryRequest, MatchedChunk
from app.observation.logging_setup import setup_logging
from app.persistence.db import load_collection
from app.service.ingestion.ingestion_orchestration import execute_ingestion_pipeline
from app.service.retrieval.retrieve import rag_query

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if APP_CONFIG.RUN_INGESTION:
        logger.info("RUN_INGESTION=true — running ingestion pipeline")
        execute_ingestion_pipeline(
            APP_CONFIG.PDF_PATH,
            APP_CONFIG.DB_PATH,
            APP_CONFIG.COLLECTION_NAME,
        )
    else:
        logger.info("RUN_INGESTION=false — skipping ingestion, loading existing collection")

    app.state.collection = load_collection(APP_CONFIG.DB_PATH, APP_CONFIG.COLLECTION_NAME)
    logger.info("Collection loaded and ready for queries")

    yield

    logger.info("Application shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Insurance Policy Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(PolicyApplicationException)
async def rag_exception_handler(request: Request, exc: PolicyApplicationException):
    logger.error(
        "Application error | path=%s | code=%s | message=%s",
        request.url.path, exc.error_code, exc.error_message,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"error_code": exc.error_code, "detail": exc.error_message},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(l) for l in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    logger.warning("Validation error | path=%s | errors=%s", request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={"error_code": "VALIDATION_ERROR", "detail": errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error | path=%s | error=%s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error_code": "INTERNAL_ERROR", "detail": "An unexpected error occurred"},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    """Run a RAG query and return the answer with source context.

    Returns:
        question:      The original question.
        answer:        LLM-generated answer grounded in retrieved chunks.
        page_numbers:  Deduplicated page numbers of matched chunks.
        matched_chunks: The retrieved text chunks with their source metadata.
    """
    logger.info("Query received: %r", request.question)

    response = rag_query(
        collection=http_request.app.state.collection,
        question=request.question,
    )

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        page_numbers=sorted({m.page_number for m in response.retrieval.matches}),
        matched_chunks=[
            MatchedChunk(
                text=m.text,
                page_number=m.page_number,
                source=m.source,
            )
            for m in response.retrieval.matches
        ],
    )