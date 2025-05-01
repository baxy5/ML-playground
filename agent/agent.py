from agent.config import initialize_components
from agent.utils import process_document


llm, embeddings, vector_store, tavily_tool = initialize_components()

process_document(file_path="src/agent/test.pdf", vector_store=vector_store)
