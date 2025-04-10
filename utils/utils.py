from langchain_community.document_loaders import PyPDFLoader
import fitz

def load_document(path: str):
    """ Loads the document using PyPDFLoader from LangChain, then remove white spaces. """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.page_content = " ".join(d.page_content.split())

        return docs
    except Exception as e:
        print(f"Error: {e}")
        raise
    
def count_characters_in_pdf(pdf_path):
    """Counts the number of characters in a given PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text("text")
        
        return len(text)

    except Exception as e:
        print(f"Error: {e}")
        raise
    