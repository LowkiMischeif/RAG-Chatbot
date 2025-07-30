import os
import glob
import logging
import textwrap
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("course_guide")

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PERSIST_DIR = "chroma_db"
CSV_GLOB   = "data/*.csv"

# ------------------------------------------------------------------------------
# 1. Pure loading & splitting helper
# ------------------------------------------------------------------------------
def docs_preprocessing_helper(file_paths: List[str]):
    """Load & split multiple CSVs—no persistence here."""
    all_docs = []
    splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    for path in file_paths:
        logger.info("Loading CSV: %s", path)
        loader = CSVLoader(path, encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        logger.info(" -> %d chunks from %s", len(chunks), path)
        all_docs.extend(chunks)

    logger.info("Total chunks across all CSVs: %d", len(all_docs))
    return all_docs

# ------------------------------------------------------------------------------
# 2. Embeddings + persistent Chroma index (module‐level)
# ------------------------------------------------------------------------------
logger.info("Initializing embedding function")
embedding_function = OllamaEmbeddings(model="nomic-embed-text:latest")

if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    logger.info("Loading existing Chroma DB from '%s'", PERSIST_DIR)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function
    )
else:
    logger.info("No Chroma DB found—building from CSVs")
    csv_files = glob.glob(CSV_GLOB)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {CSV_GLOB}")
    docs = docs_preprocessing_helper(csv_files)

    db = Chroma.from_documents(
        docs,
        embedding_function,
        persist_directory=PERSIST_DIR
    )
    logger.info("Persisted new Chroma index to '%s'", PERSIST_DIR)

# ------------------------------------------------------------------------------
# 3. Build RetrievalQA chain once
# ------------------------------------------------------------------------------
logger.info("Loading LLM (OllamaLLM)")
model = OllamaLLM(model="gemma3n:e4b", max_tokens=1024)

prompt = PromptTemplate(
    template=textwrap.dedent("""
      You are a helpful High School Guidance Counselor chatbot. List all requirements found in the provided documents as concisely as possible, using bullet points or a short list. Do not add extra description or explanation. Combine information from multiple documents if needed. If the answer is not in the documents, or you are unsure, reply exactly with: I don't know. Do not make up any information. Do not include greetings, introductions, or repeated statements. Never start your answer with a question; always answer factually. Use only the provided documents.
      
      Provided documents:
      {context}
      
      Question:
      {question}
    """),
    input_variables=["context", "question"]
)

logger.info("Setting up RetrievalQA chain")
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 10}),
    chain_type_kwargs={"prompt": prompt},
)

# ------------------------------------------------------------------------------
# 4. run() just invokes the chain
# ------------------------------------------------------------------------------
def run(question: str, history: list = None) -> str:
    logger.info("Received query: %r", question)
    try:
        result = chain.invoke({"query": question})
        # Extract answer string from result dict
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or str(result)
        else:
            answer = str(result)
        logger.info("Answer generated: %.80r", answer)
        if "I don't know" in answer or answer.strip() == "":
            return "I don't know."
        return answer
    except Exception:
        logger.exception("Error in RetrievalQA chain")
        return "Sorry, something went wrong. Please try again later."
