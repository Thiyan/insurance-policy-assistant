import logging

from langchain_openai import OpenAIEmbeddings

from app.config.config import EMBEDDING_MODEL, EMBED_BATCH_SIZE
from app.model.chunk import Chunk
from app.model.embedded_chunk import EmbeddedChunk

logger = logging.getLogger(__name__)

def embed_chunks(
    chunks: list[Chunk],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Embed chunks using OpenAI's text embedding model.

    Args:
        chunks: Chunks to embed.
        model: OpenAI embedding model name.
        batch_size: Number of chunks per API request.

    Returns:
        List of EmbeddedChunk pairing each Chunk with its embedding vector.

    Raises:
        ValueError: If any returned embedding has an unexpected dimension.
    """
    if not chunks:
        return []

    embeddings_model = OpenAIEmbeddings(model=model)
    texts = [chunk.text for chunk in chunks]
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        all_embeddings.extend(embeddings_model.embed_documents(batch))
        logger.debug("Embedded %d / %d chunks", min(batch_start + batch_size, len(texts)), len(texts))

    return [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, all_embeddings)
    ]