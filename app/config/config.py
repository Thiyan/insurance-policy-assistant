import os

PDF_PATH = "../../Insurance_Handbook_20103.pdf"
DB_PATH = "../../data/chroma_db"  # folder where ChromaDB is persisted
COLLECTION_NAME = "policy_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
N_RESULTS = 3
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
