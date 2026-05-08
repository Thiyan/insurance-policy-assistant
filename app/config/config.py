PDF_PATH = "/Users/thiyan/Documents/Projects/insurance-policy-assistant/Insurance_Handbook_20103.pdf"
DB_PATH = "/Users/thiyan/Documents/Projects/insurance-policy-assistant/data/chroma_db"  # folder where ChromaDB is persisted
COLLECTION_NAME = "rag_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
N_RESULTS = 3
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
EMBED_BATCH_SIZE = 100
HNSW_SPACE = "cosine"
TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the provided context. If the answer is not in the context, say "
    "'I don't have enough information to answer that.' "
    "Always mention the page number(s) your answer is based on."
)
