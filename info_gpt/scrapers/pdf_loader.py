import logging
import os

from chromadb.config import Settings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_document_pages(file_path):
    loader = UnstructuredPDFLoader(file_path)
    pages = loader.load()

    num_pages = len(pages)
    logger.info(f"{num_pages} document(s) loaded from {file_path}")

    if num_pages > 0:
        num_chars = len(pages[0].page_content)
        logger.info(f"The first page contains {num_chars} characters")

    return pages


def split_document_into_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.split_documents(pages)

    num_texts = len(texts)
    print(f"The documents have been split into {num_texts} chunks")

    return texts


def ingest_chunks_into_db(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="kotak",
        client_settings=Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=os.environ.get("DB_DIRECTORY", ".db"),
            anonymized_telemetry=False,
        ),
    )

    logger.info("Inserting documents into DB...")
    vector_store.add_documents(texts)


for file in os.listdir("/Users/devang/hackathon/peak-genie/pdf"):
    file_path = os.path.join("pdf", file)
    logger.info(f"Loading {file_path}")
    pages = load_document_pages(file_path)
    texts = split_document_into_chunks(pages)
    ingest_chunks_into_db(texts)
