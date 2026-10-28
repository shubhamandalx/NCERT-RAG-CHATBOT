from langchain_huggingface import HuggingFaceEmbeddings
from rag_chatbot.config import EMBEDDING_MODEL_NAME

def create_embedding_model():
      print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
      
      return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,model_kwargs={'device':'cpu'},encode_kwargs={'normalize_embeddings': True})