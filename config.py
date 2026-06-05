import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-4o"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

CHROMA_PATH = "chroma_db"
BM25_INDEX_PATH = "bm25_index.pkl"
PARENTS_PATH = "parents.pkl"
LOGS_DIR = "logs"

MAX_ITERATIONS = 5

TOP_K_DENSE = 30
TOP_K_BM25 = 30
TOP_K_RERANK_BASE = 5
RRF_K = 60
MAX_CONTEXT_CHARS = 3000
GENERATION_TEMPERATURE = 0.3

DOCUMENTS = [
    {"company": "tesla", "year": "2022", "path": "sec_docs/tesla_2022.pdf", "txt_path": "extracted_texts/tesla_2022.txt"},
    {"company": "tesla", "year": "2023", "path": "sec_docs/tesla_2023.pdf", "txt_path": "extracted_texts/tesla_2023.txt"},
    {"company": "tesla", "year": "2024", "path": "sec_docs/tesla_2024.pdf", "txt_path": "extracted_texts/tesla_2024.txt"},
    {"company": "tesla", "year": "2025", "path": "sec_docs/tesla_2025.pdf", "txt_path": "extracted_texts/tesla_2025.txt"},
    {"company": "gm", "year": "2022", "path": "sec_docs/gm_2022.pdf", "txt_path": "extracted_texts/gm_2022.txt"},
    {"company": "gm", "year": "2023", "path": "sec_docs/gm_2023.pdf", "txt_path": "extracted_texts/gm_2023.txt"},
    {"company": "gm", "year": "2024", "path": "sec_docs/gm_2024.pdf", "txt_path": "extracted_texts/gm_2024.txt"},
    {"company": "gm", "year": "2025", "path": "sec_docs/gm_2025.pdf", "txt_path": "extracted_texts/gm_2025.txt"},
    {"company": "ford", "year": "2022", "path": "sec_docs/ford_2022.pdf", "txt_path": "extracted_texts/ford_2022.txt"},
    {"company": "ford", "year": "2023", "path": "sec_docs/ford_2023.pdf", "txt_path": "extracted_texts/ford_2023.txt"},
    {"company": "ford", "year": "2024", "path": "sec_docs/ford_2024.pdf", "txt_path": "extracted_texts/ford_2024.txt"},
    {"company": "ford", "year": "2025", "path": "sec_docs/ford_2025.pdf", "txt_path": "extracted_texts/ford_2025.txt"},
]

AVAILABLE_COMPANIES = sorted({d["company"] for d in DOCUMENTS})
AVAILABLE_YEARS = sorted({d["year"] for d in DOCUMENTS})
