#TODO: testing AI Agent algorithms

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from utils.utils import load_document, count_words_in_pdf
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START
from langchain_core.documents import Document

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_SYSTEM_MESSAGE = os.getenv("PROMPT_SYSTEM_MESSAGE")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# State for the app
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Input Document Loader
def document_indexer(input_document_path: str):
    """Indexing input document into the vector store."""
    try:
        docs = load_document(input_document_path)

        words_in_pdf = count_words_in_pdf(input_document_path)
        chunk_size = words_in_pdf // 4
        chunk_overlap = chunk_size // 2
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        
        all_splits = text_splitter.split_documents(docs)
        # Index chunks into vectorDB (store)
        #TODO: Switch to an actual vectorDB not memory
        vector_store.add_documents(documents=all_splits)
    except Exception as e:
        print(f"Error: {e}")
        raise

def report(input: str):
    """Generate summary and key concept report."""
    try: 
        document_indexer(input)
        retrieved_docs = vector_store.similarity_search("", k=100)
        all_text = " ".join([doc.page_content for doc in retrieved_docs])
        
        summary_messages = [
            SystemMessage(content=PROMPT_SYSTEM_MESSAGE),
            HumanMessage(content=f"Készíts egy összefoglalót a következő dokumentumból. {all_text}"),
            HumanMessage(content="Use headers (`#`), bullet points (`-`), and bold text (`**`) where necessary.")
        ]
        summary = llm(summary_messages)
        
        key_concepts_message = [
            SystemMessage(content=PROMPT_SYSTEM_MESSAGE),
            HumanMessage(content=f"Szedd ki a szövegből az összes olyan kulcs fogalmat ami nehezen megérthető. Írd ki a nevét a kulcsfogalomnak és leírását/magyarázatát. Csak a kulcsfogalmakat és leírásukat/magyarázatukat add vissza, nem kell semmiféle további szövegkörnyezet. {all_text}"),
            HumanMessage(content="Use headers (`#`), bullet points (`-`), and bold text (`**`) where necessary.")
        ]
        key_concepts = llm(key_concepts_message)
        
        return summary.content, key_concepts.content
    except Exception as e:
        print(f"Error: {e}")
        raise

def retrieve_context(state: State):
    """Retrieve context from input document."""
    try:
        res = vector_store.similarity_search(state["question"])
        return {"context": res}
    except Exception as e:
        print(f"Error: {e}")
        raise

def generate_answer(state: State):
    """Generate answer to the question."""
    try:
        docs_content = " ".join(doc.page_content for doc in state["context"])
        messages = [
            SystemMessage(content=PROMPT_SYSTEM_MESSAGE),
            HumanMessage(content=f'''Válaszolj a kérdésre maximális pontossággal. Ha valamiben nem vagy teljesen magabiztos mond azt hogy nem tudod.
                        
                        Kérdés: {state["question"]}
                        
                        Contextus: {state["context"]}
                        
                        Válasz:
                        ''')
        ]
        res = llm.invoke(messages)
        return {"answer": res.content}
    except Exception as e:
        print(f"Error: {e}")
        raise

def qa_retrieve_and_answer(query: str):
    """Retrieve context and generate answer to the question."""
    try:
        graph_builder = StateGraph(State).add_sequence([retrieve_context, generate_answer])
        graph_builder.add_edge(START, "retrieve_context")
        graph = graph_builder.compile()
        response = graph.invoke({"question": query})
        return response["answer"]
    except Exception as e:
        print(f"Error: {e}")
        raise