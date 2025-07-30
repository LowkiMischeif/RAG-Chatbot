import logging
import hashlib
from pathlib import Path

from cachetools import TTLCache
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -----------------------------
# Logging & Paths
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "db"
COLLECTION_NAME = "course_list"

logger.info("Looking for CSV files in %s", DATA_DIR)

# -----------------------------
# In‑memory TTL Cache
# -----------------------------
response_cache = TTLCache(maxsize=100, ttl=300)  # 5 min TTL

def get_cache_key(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()

# -----------------------------
# Initialize LLM & Embeddings
# -----------------------------
logger.info("Initializing LLM and embedding function…")
model = ChatOllama(model="gemma3n:e4b")
embedding_function = OllamaEmbeddings(model="nomic-embed-text:latest")

# -----------------------------
# Prompt Templates
# -----------------------------
main_prompt = PromptTemplate(
    template="""
You are a guidance counselor at a high school with grades 9–12. A student asks:
Question: {question}
Context: {context}
Answer (short and concise):
""",
    input_variables=["context", "question"],
)

refine_prompt = PromptTemplate(
    template="""
We have an existing answer: {existing_answer}
New context: {context}
Question: {question}

Refine it. If no new data, keep original. If no data, say:
"I’m sorry, but I couldn’t find the information in the provided data."

Refined Answer:
""",
    input_variables=["existing_answer", "context", "question"],
)

chain_type_kwargs = {
    "question_prompt": main_prompt,
    "refine_prompt": refine_prompt,
    "document_variable_name": "context",
}

# -----------------------------
# Build or Load Vector Store
# -----------------------------
if PERSIST_DIR.exists():
    logger.info("Loading existing Chroma vector store from %s", PERSIST_DIR)
    db = Chroma(
        embedding_function=embedding_function,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
else:
    logger.info("Building Chroma vector store for the first time…")
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(DATA_DIR.glob("*.csv"))
    logger.info("Found CSV files: %s", [p.name for p in csv_files])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )

    all_docs = []
    for csv_path in csv_files:
        loader = CSVLoader(str(csv_path), encoding="utf-8")
        docs = loader.load()
        split_docs = splitter.split_documents(docs)
        all_docs.extend(split_docs)

    logger.info("Total documents after splitting: %d", len(all_docs))

    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_function,
        persist_directory=str(PERSIST_DIR)
    )

# -----------------------------
# Create the RetrievalQA Chain
# -----------------------------
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="refine",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs=chain_type_kwargs,
)

# -----------------------------
# Public API for Flask
# -----------------------------
def run(query: str) -> str:
    """
    Answer `query`, using a short‑lived in‑memory cache to skip repeats.
    If no context is found, replies with a default apology.
    """
    key = get_cache_key(query)
    if key in response_cache:
        logger.info("[CACHE HIT] %s", query)
        return response_cache[key]

    docs = chain.retriever.get_relevant_documents(query)
    if not docs:
        logger.warning("[NO DOCS] for query: %s", query)
        answer = "I’m sorry, but I couldn’t find the information in the provided data."
        response_cache[key] = answer
        return answer

    try:
        logger.info("[CACHE MISS] Processing query: %s", query)
        result = chain.invoke(query)
        if isinstance(result, dict):
            answer = result.get("result", "").strip()
        else:
            answer = str(result).strip()
    except Exception:
        logger.exception("Error running RetrievalQA chain")
        answer = "Oops—something went wrong trying to answer that."

    response_cache[key] = answer
    return answer
