import tempfile
from dotenv import load_dotenv
import os

from fastapi import HTTPException, UploadFile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_tavily import TavilySearch

from utils.logger import logger

load_dotenv()


def get_env_variable(var_name):
    """Get an environment variable or raise an error if not set."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"The environment variable {var_name} is not set.")
    return value


def initialize_components():
    """Initialize the components needed for the application."""
    try:
        logger.info("Initializing components...")
        llm = ChatOpenAI(model="gpt-4o-mini")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = InMemoryVectorStore(embeddings)
        tavily_tool = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="advanced",
            chunks_per_source=250,
        )
        return llm, embeddings, vector_store, tavily_tool
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing components: {e}",
        )


async def process_pdf_file(file_path: UploadFile):
    """Save PDF in a temporary directory and return its content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Temporary directory created: {temp_dir}")
        temp_file_path = os.path.join(temp_dir, "temp_file.pdf")

        with open(temp_file_path, "wb") as temp_file:
            logger.info(f"Saving file to {temp_file_path}")
            temp_file.write(await file_path.read())

        # Extract content from the PDF
        pdf_content = ""
        try:
            with open(temp_file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                pdf_content = " ".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
                logger.info("PDF content extracted successfully.")
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting PDF content: {e}",
            )

    return temp_file_path, pdf_content
