from langchain_core.language_models.chat_models import BaseChatModel
from rag_chatbot.config import (
      GROQ_API_KEY,
      GROQ_MODEL,
      LLM_PROVIDER
)

def create_chat_model() -> BaseChatModel:
      if LLM_PROVIDER == 'groq':
            if not GROQ_API_KEY:
                  raise ValueError('Groq API key is not found.')
            
            from langchain_groq import ChatGroq
            
            return ChatGroq(
                  model=GROQ_MODEL,
                  api_key=GROQ_API_KEY,
                  temperature=0.2,
            )
            
      raise ValueError(f'Unsupported LLM Provider: {LLM_PROVIDER}')