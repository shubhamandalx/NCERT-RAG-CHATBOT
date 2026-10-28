import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rag_chatbot.config import(CHROMA_COLLECTION_NAME,CHROMA_DIRECTORY)

def open_vector_store(embedding_model: HuggingFaceEmbeddings,persist_directory: Path = CHROMA_DIRECTORY):
      persist_directory.mkdir(parents=True, exist_ok=True)
      
      return Chroma(collection_name=CHROMA_COLLECTION_NAME, embedding_function=embedding_model,persist_directory=str(persist_directory),)

def build_or_load_vector_store(
      chunks: list[Document],
      embedding_model: HuggingFaceEmbeddings,
      rebuild: bool = False,
):
      if rebuild and CHROMA_DIRECTORY.exists():
            print('Removing the existing directory')
            shutil.rmtree(CHROMA_DIRECTORY)
            
      vector_store = open_vector_store(embedding_model)
      
      existing_count = vector_store._collection.count()
      
      if existing_count > 0:
            print(f'Reusing existing chroma index with {existing_count}')
            
            return vector_store
      
      if not chunks:
            raise ValueError('Cannot build a vector database from zero chunks')
      
      print(f'Embedding and Saving {len(chunks)} chunks to ChromaDB')
      vector_store.add_documents(chunks)
      print(f'Index created at: {CHROMA_DIRECTORY}')
      return vector_store
      