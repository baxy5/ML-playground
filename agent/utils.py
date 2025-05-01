from fastapi import HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger
from utils.utils import count_characters_in_pdf, load_document


def process_document(file_path: str, vector_store):
    """Process a document by loading, splitting, and adding it to the vector store."""
    try:
        logger.info("Processing document...")
        docs = load_document(file_path)

        characters_in_pdf = count_characters_in_pdf(file_path)
        chunk_size = characters_in_pdf // 16
        chunk_overlap = 20

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        all_splits = text_splitter.split_documents(docs)

        vector_store.add_documents(documents=all_splits)

        logger.info("Processing document finished.")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {e}",
        )


def run_similarity_search(file: UploadFile, vector_store):
    """Run a similarity search on the vector store."""
    try:
        logger.info(f"Running similarity search.")
        all_context = " ".join([doc.page_content for doc in retrieved_context])
        logger.info("Similarity search finished.")
        return all_context
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise


def run_similarity_search_with_query(query: str, vector_store):
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
