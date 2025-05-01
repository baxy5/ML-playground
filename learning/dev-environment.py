import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from utils.utils import load_document, count_characters_in_pdf
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from utils.logger import logger

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

PROMPT_SYSTEM_MESSAGE = os.getenv("PROMPT_SYSTEM_MESSAGE")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
tavily_tool = TavilySearch(
    max_results=5, topic="general", search_depth="advanced", chunks_per_source=250
)
dummy_data_path = "dummy-data/test-article.pdf"

if not LANGCHAIN_API_KEY:
    raise EnvironmentError("LangChain api key is not set.")
if not OPENAI_API_KEY:
    raise EnvironmentError("OpenAI api key is not set.")
if not TAVILY_API_KEY:
    raise EnvironmentError("Tavily api key is not set.")

def process_document(file_path: str):
    """Process a document by loading, splitting, and adding it to the vector store."""
    try:
        logger.info("Processing document...")
        docs = load_document(file_path)
        characters_in_pdf = count_characters_in_pdf(file_path)
        chunk_size = characters_in_pdf // 16
        chunk_overlap = 20
        logger.info(f"Words in PDF: {characters_in_pdf}")
        logger.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=all_splits)
        logger.info("Processing document finished.")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise

def run_similarity_search(query: str):
    """Run a similarity search on the vector store."""
    try:
        logger.info(f"Running similarity search. Query: {query}")
        retrieved_context = vector_store.similarity_search(query, k=1)
        all_context = " ".join([doc.page_content for doc in retrieved_context])
        logger.info("Similarity search finished.")
        return all_context
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise

def generate_response_with_llm(prompt: List):
    """Generate a response using the LLM with streaming."""
    try:
        logger.info("Processing context with LLM.")
        response_text = ""
        for chunk in llm.stream(prompt):
            response_text += chunk.text()
        logger.info("LLM processing finished.")
        return response_text
    except Exception as e:
        logger.error(f"Error during LLM processing: {e}")
        raise

def run_tavily_tool(query: str, context: str):
    """Run the Tavily tool with the given query and context."""
    try:
        logger.info("Running Tavily tool.")
        tavily_prompt = f"Question: {query} Context: {context}"
        response = tavily_tool.invoke({"query": tavily_prompt})
        results = response["results"]
        best_result = max(results, key=lambda x: x["score"])
        logger.info("Tavily tool finished.")
        return best_result["content"]
    except Exception as e:
        logger.error(f"Error during Tavily tool execution: {e}")
        raise

# Main execution
if __name__ == "__main__":
    process_document(dummy_data_path)

    user_q = "Mi az a VFS?"
    all_context = run_similarity_search(user_q)

    context_message = f"Kérdés: {user_q} Szöveg: {all_context}"
    refactor_prompt = [
        SystemMessage(
            content="A feladatod, hogy kielégítsd a felhasználó igényeit. Jártas vagy szövegek írásában."
        ),
        HumanMessage(
            content=f"Add meg a szöveg tartalmának a témáját és rövid leírását, maximum 250 karakterben. A kérdést nem kell újra leirnod a válaszodban. {context_message}"
        ),
    ]
    response_text = generate_response_with_llm(refactor_prompt)

    final_context = run_tavily_tool(user_q, response_text)

    final_context_message = f"Kérdés: {user_q} Kontextus: {response_text} További kontextus: {final_context}"
    final_prompt = [
        SystemMessage(
            content="Egy kedves, jókifejező készséggel rendelkező tanár vagy akinek a feladata, hogy tanítványait megtanítsa éppen az adott témára. Képes vagy felmérni a tanítványod tudását és értelmét, hogy minél pontosabban tudj megfogalmazni. Ha nem tudod a választ valamilyen kérdésre akkor csak annyit mondasz 'Nem tudom a választ.'. Semmi személyes kérdést nem teszel fel és nem válaszolsz rájuk."
        ),
        HumanMessage(
            content=f"Válaszolj a kérdésre, magyarázd el minél pontosabb tudás szerint. Ha kell, mondj példát vagy adj hozzá további kontextust. {final_context_message}"
        ),
        HumanMessage(
            content="Write the following text using proper Markdown formatting. Use headings (#), subheadings (##), bullet points (- or *), numbered lists (1.), bold (**bold**), italic (*italic*), code blocks (`code` or triple backticks), blockquotes (>) where appropriate, and ensure all links are in [text](url) format."
        ),
    ]
    final_response = generate_response_with_llm(final_prompt)
    print(final_response)
