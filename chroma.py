import os
import logging
import traceback
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini embeddings
from dotenv import load_dotenv

load_dotenv()

# Configuration
PDF_DIR = "documents/"
CHROMA_PERSIST_DIR = "chroma_db"
# Set your maximum batch size (adjust based on your Chroma version/configuration)
BATCH_SIZE = 5461

# Configure logging to include detailed error context
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory, adding file metadata for traceability."""
    pdf_documents = []
    directory = Path(directory)

    for pdf_path in directory.glob('*.pdf'):
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            documents = loader.load()
            # Add metadata to each document to retain source details
            for doc in documents:
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = pdf_path.name
            pdf_documents.extend(documents)
            logging.info(f"✅ Loaded: {pdf_path.name}")
        except Exception as e:
            logging.exception(f"Error loading {pdf_path.name}: {e}")
    return pdf_documents


def process_documents(documents):
    """Split documents into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    try:
        chunks = text_splitter.split_documents(documents)
        logging.info(f"✅ Processed documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.exception("Error during document splitting")
        raise


def index_to_chroma(chunks):
    """Index document chunks to ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma(
        collection_name="documents",  # Set a collection name (optional)
        persist_directory=CHROMA_PERSIST_DIR,  # Ensure persistence
        embedding_function=embeddings
    )

    vectordb.add_documents(chunks)  # Index the chunks

    print("✅ Successfully indexed documents!")
    return vectordb  # No need for .persist()



def search_chroma(vectordb, query):
    """Perform a similarity search with error handling."""
    try:
        results = vectordb.similarity_search(query)
        logging.info(f"✅ Found {len(results)} result(s) for query: '{query}'")
        return results
    except Exception as e:
        logging.exception(f"Error during similarity search for query '{query}': {e}")
        return None


def main():
    # Create directories if they don't exist
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    logging.info("🔍 Loading PDFs...")
    raw_docs = load_pdfs_from_directory(PDF_DIR)
    if not raw_docs:
        logging.warning("⚠️ No PDF documents found!")
        return

    logging.info(f"📄 Found {len(raw_docs)} document(s). Processing...")
    chunks = process_documents(raw_docs)

    logging.info("📥 Indexing document chunks to ChromaDB...")
    vectordb = index_to_chroma(chunks)
    logging.info(f"✅ Successfully indexed {len(chunks)} chunk(s) to ChromaDB")

    # Example search query (replace with your actual search)
    query = "Your detailed search query here"
    results = search_chroma(vectordb, query)
    if results is not None:
        for idx, res in enumerate(results, start=1):
            logging.info(f"Result {idx}: {res}")
    else:
        logging.error("No results found or an error occurred during the search.")


if __name__ == "__main__":
    main()
