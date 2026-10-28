from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from rag_chatbot.config import RETRIEVAL_K

class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str
      
def create_rag_graph(vector_store, chat_model: BaseChatModel):
    # Fixed typo: as_retriever
    retriever = vector_store.as_retriever(search_kwargs={'k': RETRIEVAL_K})
      
    # Fixed typo: from_messages (plural)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system', 
                "You are an NCERT study assistant. Answer the user's question using only the supplied textbook context. "
                "If the context does not contain enough information, say clearly that you could not find the answer in the provided NCERT documents. "
                "Do not invent facts, page numbers, quotations, or textbook content. Explain concepts in clear, student-friendly language. "
                "You may simplify the wording, but do not change the scientific meaning. TEXTBOOK CONTEXT: {context}"
            ),
            ("human", "{question}")
        ]
    )
      
    def retrieve_node(state: RAGState):
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents}
      
    def generate_node(state: RAGState):
        documents_list = state.get('documents', [])
        context_parts = []
        
        # Fixed: Renamed iteration item to 'doc' to avoid shadowing and undefined variables
        for index, doc in enumerate(documents_list, start=1):
            source = doc.metadata.get('source', 'Unknown Source')
            page_index = doc.metadata.get('page')
            page_label = str(int(page_index) + 1) if isinstance(page_index, (int, float)) else 'unknown'
            
            context_parts.append(
                f"[Source {index}: {source}, PDF page {page_label}]\n{doc.page_content}"
            )
            
        # Fixed: Moved context compilation and invocation OUTSIDE the for loop
        context = "\n\n --- \n\n".join(context_parts)
        
        formatted_prompt = prompt.invoke({
            "question": state["question"],
            "context": context or "(No relevant textbook passages retrieved)"
        })
        
        response = chat_model.invoke(formatted_prompt)
        return {'answer': str(response.content)}
            
    # Fixed: Corrected indentation scope for the graph builder
    graph_builder = StateGraph(RAGState)
    graph_builder.add_node('retrieve', retrieve_node)
    graph_builder.add_node('generate', generate_node)
    
    graph_builder.add_edge(START, 'retrieve')
    graph_builder.add_edge("retrieve", 'generate')
    graph_builder.add_edge("generate", END)
    
    return graph_builder.compile()
      
def ask_question(graph, question: str):
    return graph.invoke({'question': question})
