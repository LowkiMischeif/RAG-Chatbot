import os
import glob
import logging
from typing import List, Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("course_guide")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
PERSIST_DIR = "chroma_db"          # folder to persist the vectorstore
CSV_GLOB = "data/*.csv"            # where your CSVs live
OLLAMA_MODEL = "gemma3n:e4b"       # change if you don’t have this locally
EMBED_MODEL = "nomic-embed-text:latest"        # embeddings model name for Ollama

# ------------------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful high-school guidance counselor. Answer briefly and clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (concise and accurate):"
    ),
)

# ------------------------------------------------------------------------------
# Embeddings (module-level singletons)
# ------------------------------------------------------------------------------
embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _load_csv_docs(csv_glob: str) -> List:
    """Load and split CSVs into chunks."""
    file_paths = sorted(glob.glob(csv_glob))
    if not file_paths:
        raise FileNotFoundError(
            f"No CSV files found matching pattern: {csv_glob}. "
            "Create a 'data' folder with at least one CSV."
        )

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_docs = []

    for path in file_paths:
        logger.info("Loading CSV: %s", path)
        # Some versions of CSVLoader don’t accept encoding directly; try both styles.
        try:
            loader = CSVLoader(path, encoding="utf-8")
        except TypeError:
            loader = CSVLoader(path, csv_args={"encoding": "utf-8"})

        raw_docs = loader.load()
        chunks = splitter.split_documents(raw_docs)
        logger.info(" -> %d chunks from %s", len(chunks), path)
        all_docs.extend(chunks)

    logger.info("Total chunks loaded: %d", len(all_docs))
    return all_docs


def _build_retriever(docs: List):
    """Create (or refresh) a Chroma index and return a retriever."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    db = Chroma.from_documents(
        docs,
        embedding=embedding_function,          # use keyword to avoid API drift
        persist_directory=PERSIST_DIR,
    )
    return db.as_retriever(search_kwargs={"k": 10})


def _build_chain(retriever):
    llm = OllamaLLM(model=OLLAMA_MODEL)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain

# ------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------
def run(question: str, history: Optional[list] = None) -> str:
    """
    Answer a user question using CSV-backed retrieval.
    `history` is optional; if provided, the last few turns are prepended to the question.
    """
    if not isinstance(question, str) or not question.strip():
        return "Please provide a valid question."

    # Light-touch history conditioning (optional)
    if history:
        try:
            pairs = [f"{m.get('role','user')}: {m.get('content','')}" for m in history[-8:]]
            hist_text = "\n".join(pairs)
            question = f"Chat History:\n{hist_text}\n\nUser question: {question}"
        except Exception:
            logger.exception("Failed to incorporate chat history; continuing without it.")

    try:
        docs = _load_csv_docs(CSV_GLOB)
    except Exception as e:
        logger.exception("Failed to load CSVs.")
        return f"Could not load data: {e}"

    retriever = _build_retriever(docs)
    chain = _build_chain(retriever)

    logger.info("Received query: %r", question)
    try:
        result = chain.invoke({"query": question})
        # Extract answer string safely
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or str(result)
        else:
            answer = str(result)

        logger.info("Answer generated: %.80r", answer)
        if "I don't know" in answer or not answer.strip():
            return "I don't know."
        return answer
    except Exception:
        logger.exception("Error while running RetrievalQA.")
        return "Sorry, something went wrong. Please try again later."


if __name__ == "__main__":
    # Quick manual test
    print(run("Which AP science classes are offered?"))
