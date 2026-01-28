import os
import json
import hashlib
from typing import List, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# -----------------------------
# CONFIG
# -----------------------------
PDF_FOLDER = "documents"
DB_PATH = "chroma_db"
MANIFEST_PATH = os.path.join(DB_PATH, "_manifest.json")

LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOP_K = 3  # retrieve more; then we apply stricter filtering
SNIPPET_CHARS = 200  # how much snippet to show per source chunk

# Evidence thresholds (simple but effective)
MIN_SOURCES_TO_ANSWER = 1        # require at least N retrieved chunks
MAX_SOURCES_TO_DISPLAY = 6       # display at most N sources/snippets
REFUSE_IF_NO_CONTEXT = True      # if retrieval returns nothing, refuse


# -----------------------------
# Helpers
# -----------------------------
def ensure_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_pdfs() -> List[str]:
    if not os.path.exists(PDF_FOLDER):
        return []
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    return [os.path.join(PDF_FOLDER, f) for f in sorted(pdfs)]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(pdf_paths: List[str]) -> dict:
    return {
        "pdfs": [
            {"file": os.path.basename(p), "sha256": file_sha256(p)}
            for p in pdf_paths
        ]
    }


def manifest_matches_current(pdf_paths: List[str]) -> bool:
    if not os.path.exists(MANIFEST_PATH):
        return False
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return False

    current = build_manifest(pdf_paths)
    return saved == current


def save_manifest(pdf_paths: List[str]) -> None:
    ensure_folder(DB_PATH)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(build_manifest(pdf_paths), f, indent=2)


def pretty_source(doc) -> str:
    src = doc.metadata.get("source_file") or doc.metadata.get("source") or "unknown"
    page = doc.metadata.get("page")
    if page is not None:
        return f"{src} (page {page + 1})"
    return f"{src}"


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "…"


def dedupe_docs_keep_order(docs) -> List:
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source_file") or d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# -----------------------------
# Vector DB
# -----------------------------
def build_or_load_vector_db(force_rebuild: bool = False) -> Chroma:
    embedding_fn = OllamaEmbeddings(model=EMBED_MODEL)
    ensure_folder(PDF_FOLDER)

    pdf_paths = list_pdfs()
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF found in '{PDF_FOLDER}'. Put at least one PDF in '{PDF_FOLDER}' and run again."
        )

    db_exists = os.path.exists(DB_PATH) and os.listdir(DB_PATH)

    # If DB exists and manifest matches, load it
    if db_exists and (not force_rebuild) and manifest_matches_current(pdf_paths):
        print("[INFO] Loading existing vector database (PDFs unchanged).")
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_fn)

    # Otherwise rebuild
    if force_rebuild and os.path.exists(DB_PATH):
        print("[INFO] Force rebuild requested.")
    elif db_exists:
        print("[INFO] PDFs changed or manifest missing. Rebuilding DB...")
    else:
        print("[INFO] No existing DB found. Creating a new one from PDFs...")

    # Load PDFs
    all_docs = []
    for path in pdf_paths:
        print(f"[INFO] Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(path)
        all_docs.extend(docs)

    print(f"[INFO] Loaded {len(all_docs)} pages total from {len(pdf_paths)} PDF(s).")

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    # Build DB
    ensure_folder(DB_PATH)
    print("[INFO] Creating embeddings + saving to local Chroma DB (first time may take a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=DB_PATH
    )

    save_manifest(pdf_paths)
    print(f"[DONE] Vector DB ready at: {DB_PATH}")
    return vectorstore


# -----------------------------
# RAG Chain
# -----------------------------
def make_rag_chain(vectorstore: Chroma):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1,num_ctx=2048)  # low temp = less guessing
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    system_prompt = (
         "You are a strict offline assistant.\n"
    "Use ONLY the provided context.\n"
    "Allowed tasks:\n"
    "1) Summarize the provided context (e.g., resume summary, bullet points).\n"
    "2) Extract information that is explicitly present in the context.\n"
    "Rules:\n"
    "- If the user asks for something NOT supported by the context, say exactly:\n"
    "  'Not found in the provided documents.'\n"
    "- Do NOT say you are unable to summarize if context is provided.\n"
    "- Keep answers concise and directly based on the context.\n\n"
    "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain


def evidence_ok(context_docs: List) -> bool:
    if not context_docs:
        return not REFUSE_IF_NO_CONTEXT
    return len(context_docs) >= MIN_SOURCES_TO_ANSWER


def print_sources_with_snippets(context_docs: List) -> None:
    if not context_docs:
        print("Sources: (none)")
        return

    docs = dedupe_docs_keep_order(context_docs)[:MAX_SOURCES_TO_DISPLAY]

    print("Sources (with snippets):")
    for i, d in enumerate(docs, 1):
        src = pretty_source(d)
        snip = make_snippet(d.page_content)
        print(f" {i}. {src}")
        print(f"    “{snip}”")


# -----------------------------
# Chat Loop + Commands
# -----------------------------
def chat_loop():
    vectorstore = build_or_load_vector_db(force_rebuild=False)
    rag_chain = make_rag_chain(vectorstore)

    last_context = []

    print("\n--- Offline AI Ready ---")
    print("Commands: /sources  /rebuild  /stats  /exit\n")

    while True:
        query = input("User: ").strip()

        if not query:
            continue

        if query.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Goodbye!")
            break

        if query.lower() == "/sources":
            print_sources_with_snippets(last_context)
            continue

        if query.lower() == "/stats":
            # basic stats
            pdfs = list_pdfs()
            db_exists = os.path.exists(DB_PATH) and os.listdir(DB_PATH)
            print(f"[STATS] PDFs: {len(pdfs)} | DB exists: {bool(db_exists)} | DB path: {DB_PATH}")
            for p in pdfs:
                print(f" - {os.path.basename(p)}")
            continue

        if query.lower() == "/rebuild":
            # rebuild vector db and recreate chain
            vectorstore = build_or_load_vector_db(force_rebuild=True)
            rag_chain = make_rag_chain(vectorstore)
            print("[DONE] Rebuilt DB. Ask your question now.")
            continue

        print("Thinking...")
        try:
            result = rag_chain.invoke({"input": query})
            answer = (result.get("answer") or "").strip()
            context_docs = result.get("context") or []
            last_context = context_docs

            if not evidence_ok(context_docs):
                print("\nAI: Not found in the provided documents.\n")
                continue

            print(f"\nAI: {answer}\n")
            print_sources_with_snippets(context_docs)

        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    chat_loop()
