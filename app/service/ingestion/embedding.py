import logging

from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from app.config.config import APP_CONFIG
from app.exception.sub_exception.ingestion_exception import EmbeddingException
from app.model.chunk import Chunk
from app.model.embedded_chunk import EmbeddedChunk

logger = logging.getLogger(__name__)


def embed_chunks(
    chunks: list[Chunk],
    model: str = APP_CONFIG.EMBEDDING_MODEL,
    batch_size: int = APP_CONFIG.EMBED_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Embed chunks using OpenAI's text embedding model.

    Args:
        chunks:     Chunks to embed.
        model:      OpenAI embedding model name.
        batch_size: Number of chunks per API request.

    Returns:
        List of EmbeddedChunk pairing each Chunk with its embedding vector.

    Raises:
        EmbeddingException: API call fails, returns wrong count, or any
                            embedding vector is empty.
    """
    if not chunks:
        return []

    logger.info("Embedding %d chunks with model '%s'", len(chunks), model)

    embeddings_model = OpenAIEmbeddings(model=model)
    texts = [chunk.text for chunk in chunks]
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(texts))
        batch_embeddings = _embed_batch(batch, model, embeddings_model, batch_end, len(texts))
        all_embeddings.extend(batch_embeddings)

    if len(all_embeddings) != len(chunks):
        raise EmbeddingException(
            model=model,
            reason=f"Embedding count mismatch: expected {len(chunks)}, got {len(all_embeddings)}",
        )

    logger.info("Successfully embedded %d chunks", len(chunks))

    return [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, all_embeddings)
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embed_batch(
    batch: list[str],
    model: str,
    embeddings_model: OpenAIEmbeddings,
    batch_end: int,
    total: int,
) -> list[list[float]]:
    """Call the OpenAI embeddings API for a single batch.

    Raises:
        EmbeddingException: On any API error or empty vector in the response.
    """
    try:
        embeddings = embeddings_model.embed_documents(batch)
    except AuthenticationError as exc:
        raise EmbeddingException(model=model, reason=f"Invalid API key: {exc}") from exc
    except RateLimitError as exc:
        raise EmbeddingException(model=model, reason=f"Rate limit exceeded: {exc}") from exc
    except APITimeoutError as exc:
        raise EmbeddingException(model=model, reason=f"Request timed out: {exc}") from exc
    except APIConnectionError as exc:
        raise EmbeddingException(model=model, reason=f"Connection error: {exc}") from exc
    except BadRequestError as exc:
        raise EmbeddingException(model=model, reason=f"Bad request — check model name or input: {exc}") from exc
    except Exception as exc:
        raise EmbeddingException(model=model, reason=f"Unexpected error during embedding: {exc}") from exc

    for i, vector in enumerate(embeddings):
        if not vector:
            raise EmbeddingException(
                model=model,
                reason=f"Empty embedding vector returned for chunk at batch index {i}",
            )

    logger.debug("Embedded %d / %d chunks", batch_end, total)
    return embeddings