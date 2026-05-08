PDF_PATH = "/Users/thiyan/Documents/Projects/insurance-policy-assistant/Insurance_Handbook_20103.pdf"
DB_PATH = "/Users/thiyan/Documents/Projects/insurance-policy-assistant/data/chroma_db"  # folder where ChromaDB is persisted
COLLECTION_NAME = "rag_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
N_RESULTS = 3
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
