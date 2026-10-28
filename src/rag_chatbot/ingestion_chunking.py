from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chatbot.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIRECTORY

def load_pdf_pages(pdf_directory: Path = PDF_DIRECTORY):
      pdf_directory.mkdir(parents=True, exist_ok=True)
      pdf_paths = sorted(pdf_directory.glob('*.pdf'))
      
      if not pdf_paths:
            raise FileNotFoundError(f"No PDFs found in {pdf_directory}\n")
      
      all_pages: list[Document] = []
      
      for pdf_path in pdf_paths:
            print(f'Loading PDF: {pdf_path.name}')
            
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            for page in pages:
                  page.metadata['source'] = pdf_path.name
                  if page.page_content and page.page_content.strip():
                        all_pages.append(page)
                        
      if not all_pages:
            raise ValueError('PDFs were found but contained no extractable content')
      
      print(f'Loaded {len(all_pages)} pages with extractable text')
      
      return all_pages

def split_pages_into_chunks(pages: list[Document]):
      if CHUNK_SIZE <= 0:
            raise ValueError('Chunk size must be greater than zero')
      
      if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
            raise ValueError('Chunk overlap must be >= 0 and smaller than chunk size')
      
      splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '.', ' '],
            add_start_index=True,
      )
      
      chunks = splitter.split_documents(pages)
      
      print(f'Split pages into {len(chunks)} text chunks')
      
      return chunks

def load_and_split_pdfs():
      pages = load_pdf_pages()  # Fixed: calls load_pdf_pages() instead of itself
      
      return split_pages_into_chunks(pages)