# backker.py
import os
import glob
import json
import shutil
import atexit
import logging
from typing import List, Optional, Dict

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("course_guide")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PERSIST_DIR = "chroma_db"
CSV_GLOB = "data/*.csv"
MANIFEST_PATH = os.path.join(PERSIST_DIR, "_manifest.json")

OLLAMA_MODEL = "gemma3n:e4b"
EMBED_MODEL = "nomic-embed-text:latest"

# Optional: force a rebuild by env var
FORCE_REBUILD = os.getenv("REBUILD_INDEX", "0") == "1"

# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful high-school guidance counselor. Answer briefly and clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (concise and accurate):"
    ),
)

# ---------------------------------------------------------------------
# Singletons (module-level)
# ---------------------------------------------------------------------
embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

_STATE: Dict[str, object] = {
    "db": None,
    "retriever": None,
    "chain": None,
    "initialized": False,
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _current_manifest() -> Dict[str, Dict[str, float]]:
    files = sorted(glob.glob(CSV_GLOB))
    return {
        f: {"mtime": os.path.getmtime(f), "size": os.path.getsize(f)}
        for f in files
    }

def _manifests_match(a: Dict, b: Dict) -> bool:
    return a == b

def _read_manifest_on_disk() -> Optional[Dict]:
    if not os.path.exists(MANIFEST_PATH):
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.exception("Failed to read manifest.")
        return None

def _write_manifest(manifest: Dict):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

def _load_csv_docs(csv_glob: str) -> List:
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

def _build_new_index(docs: List) -> Chroma:
    # Start fresh to avoid duplicate inserts when rebuilding
    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    logger.info("Building new Chroma index...")
    db = Chroma.from_documents(
        docs,
        embedding=embedding_function,               # correct arg name for from_documents
        persist_directory=PERSIST_DIR,
        # collection_name="course_guide",           # optional, if you want a stable name
    )
    logger.info("Chroma index built & persisted.")
    return db

def _load_existing_index() -> Chroma:
    logger.info("Loading existing Chroma index from %s ...", PERSIST_DIR)
    # For the constructor, use embedding_function kw
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
        # collection_name="course_guide",           # optional, if you set it above
    )
    return db

def _build_chain(db: Chroma):
    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = OllamaLLM(model=OLLAMA_MODEL)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return retriever, chain

# ---------------------------------------------------------------------
# Public lifecycle
# ---------------------------------------------------------------------
def initialize(force_rebuild: bool = False):
    """Initialize embeddings/vectorstore/retriever/chain once."""
    if _STATE["initialized"] and not force_rebuild:
        return

    current = _current_manifest()
    on_disk = _read_manifest_on_disk()

    must_rebuild = force_rebuild or FORCE_REBUILD

    # Rebuild if no DB yet or manifest changed
    if not os.path.isdir(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        must_rebuild = True
    elif not _manifests_match(current, on_disk or {}):
        logger.info("CSV files changed; will rebuild index.")
        must_rebuild = True

    if must_rebuild:
        docs = _load_csv_docs(CSV_GLOB)
        db = _build_new_index(docs)
        _write_manifest(current)
    else:
        db = _load_existing_index()

    retriever, chain = _build_chain(db)

    _STATE["db"] = db
    _STATE["retriever"] = retriever
    _STATE["chain"] = chain
    _STATE["initialized"] = True
    logger.info("CourseGuide initialized (rebuild=%s).", must_rebuild)

def shutdown():
    """Persist and release references; called automatically at interpreter exit."""
    try:
        db = _STATE.get("db")
        if db is not None:
            logger.info("Persisting Chroma DB...")
    except Exception:
        logger.exception("Error while persisting DB on shutdown.")
    finally:
        _STATE.update({"db": None, "retriever": None, "chain": None, "initialized": False})
        logger.info("CourseGuide shutdown complete.")

atexit.register(shutdown)

# ---------------------------------------------------------------------
# Query entry point
# ---------------------------------------------------------------------
def run(question: str, history: Optional[list] = None) -> str:
    """
    Answer a user question using CSV-backed retrieval.
    Initializes on first call; subsequent calls reuse the same DB/retriever/chain.
    """
    if not isinstance(question, str) or not question.strip():
        return "Please provide a valid question."

    if not _STATE["initialized"]:
        initialize()

    # Light-touch history conditioning (optional)
    if history:
        try:
            pairs = [f"{m.get('role','user')}: {m.get('content','')}" for m in history[-8:]]
            hist_text = "\n".join(pairs)
            question = f"Chat History:\n{hist_text}\n\nUser question: {question}"
        except Exception:
            logger.exception("Failed to incorporate chat history; continuing without it.")

    chain = _STATE["chain"]
    logger.info("Received query: %r", question)

    try:
        result = chain.invoke({"query": question})
        answer = result.get("result") if isinstance(result, dict) else str(result)
        logger.info("Answer generated: %.80r", answer)
        if not answer.strip() or "I don't know" in answer:
            return "I don't know."
        return answer
    except Exception:
        logger.exception("Error while running RetrievalQA.")
        return "Sorry, something went wrong. Please try again later."

if __name__ == "__main__":
    initialize()  # build or load once at process start
    print(run("Which AP science classes are offered?"))
    # shutdown()  # optional; atexit will handle this
