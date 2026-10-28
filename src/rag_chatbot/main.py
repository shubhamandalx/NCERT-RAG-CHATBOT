import argparse
from rag_chatbot.embedding import create_embedding_model
from rag_chatbot.ingestion_chunking import load_and_split_pdfs
from rag_chatbot.llm import create_chat_model
from rag_chatbot.rag_graph import ask_question, create_rag_graph
from rag_chatbot.vector_store import build_or_load_vector_store

def print_sources(documents):
      if not documents:
            print('\n Sources: no textbook chunks were retrieved')
            return 
      unique_sources = []
      seen = set()
      
      for document in documents:
            source = document.metadata.get('source','Unknown Source')
            page_index = document.metadata.get('page')
            page_number = int(page_index) + 1 if isinstance(page_index, int) else None
            
            source_key = (source, page_number)
            if source_key not in seen:
                  seen.add(source_key)
                  unique_sources.append(source_key)
                  
      print('\n Sources Retrieved')
      
      for source, page_number in unique_sources:
            if page_number is None:
                  print(f'- {source} (page unavailable)')
                  
            else:
                  print(f' - {source}, PDF page {page_number}')
                  
                  
                  
def main():
      
      parser = argparse.ArgumentParser(description='Chat with your NCERT PDF chapters using RAG')
      
      parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Delete and rebuild the local Chroma vector index'
      )
      
      args = parser.parse_args()
      
      print('\n === NCERT RAG CHATBOT ===\n')
      
      embedding_model = create_embedding_model()
      
      if args.rebuild:
            chunks  = load_and_split_pdfs()
            
      else:
            from rag_chatbot.config import(
                  CHROMA_COLLECTION_NAME,
                  CHROMA_DIRECTORY
            )
            
            from langchain_chroma import Chroma
            
            existing_store = Chroma(
                  collection_name=CHROMA_COLLECTION_NAME,
                  embedding_function=embedding_model,
                  persist_directory=str(CHROMA_DIRECTORY)
            )
            
            index_exists = existing_store._collection.count() > 0
            chunks = [] if index_exists else load_and_split_pdfs()
            
      vector_store = build_or_load_vector_store(
            chunks=chunks,
            embedding_model=embedding_model,
            rebuild=args.rebuild,
      )
      
      chat_model = create_chat_model()
      graph = create_rag_graph(vector_store, chat_model)
      
      print(f'Please. Ask questions about your NCERT PDFs')
      print(f"Commands: type 'exit' or 'quit' to stop. \n")
      
      while True:
            try: 
                  question = input('You: ').strip()
            except (EOFError, KeyboardInterrupt):
                  print('\n GoodBye!')
                  break
            
            if question.lower() in {'exit', 'quit'}:
                  print('Goodbye!')
                  break
            
            if not question:
                  continue
            
            try:
                  result = ask_question(graph, question)
                  print(f"\nAssistant: {result.get('answer', 'No answer was returned.')}")
                  
                  print_sources(result.get('documents', []))

                  print()
                  
            except Exception as error:
                  print(f"\n Something went wrong: {error}")
                  print("Check your API key, internet connection, and model name\n")
                  
                  
if __name__ == "__main__":
      main()

                  
      
      
      