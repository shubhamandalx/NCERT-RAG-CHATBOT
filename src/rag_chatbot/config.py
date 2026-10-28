from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / '.env')

PDF_DIRECTORY = PROJECT_ROOT / 'data' / 'pdfs'
CHROMA_DIRECTORY = PROJECT_ROOT / 'chroma_db'
CHROMA_COLLECTION_NAME = 'ncert_chapters'

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '900'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", '4'))

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq').strip().lower()
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")