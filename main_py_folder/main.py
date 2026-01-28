import os
from typing import List

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

LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4  # how many chunks to retrieve per question


def ensure_documents_folder() -> None:
    """Create documents folder if not exists."""
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER, exist_ok=True)
        print(f"[INFO] Created folder '{PDF_FOLDER}'. Put your PDF(s) inside it and run again.")


def list_pdfs() -> List[str]:
    """Return full paths of PDFs in documents folder."""
    if not os.path.exists(PDF_FOLDER):
        return []
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    return [os.path.join(PDF_FOLDER, f) for f in pdfs]


def build_or_load_vector_db() -> Chroma:
    """
    If DB exists -> load it
    else -> ingest PDFs and create DB
    """
    embedding_fn = OllamaEmbeddings(model=EMBED_MODEL)

    # If DB already exists and is non-empty, just load it
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("[INFO] Loading existing vector database...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_fn)

    # Otherwise create DB from PDFs
    ensure_documents_folder()
    pdf_paths = list_pdfs()

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF found in '{PDF_FOLDER}'. Add at least one PDF file to '{PDF_FOLDER}' and run again."
        )

    print("[INFO] No existing DB found. Creating a new one from PDFs...")
    all_docs = []
    for path in pdf_paths:
        print(f"[INFO] Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()

        # Ensure source filename is present (helps citations)
        for d in docs:
            d.metadata["source_file"] = os.path.basename(path)
        all_docs.extend(docs)

    print(f"[INFO] Loaded {len(all_docs)} pages total from {len(pdf_paths)} PDF(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    print("[INFO] Creating embeddings + saving to local Chroma DB (first time may take a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=DB_PATH
    )
    print(f"[DONE] Vector DB created at: {DB_PATH}")
    return vectorstore


def make_rag_chain(vectorstore: Chroma):
    """
    Build retrieval + LLM chain with strict prompting to reduce hallucinations.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    system_prompt = (
        "You are a strict offline study assistant.\n"
        "Answer the user's question ONLY using the provided context.\n"
        "If the answer is not clearly present in the context, say exactly: 'Not found in the provided documents.'\n"
        "Keep the answer concise.\n\n"
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


def print_sources(context_docs) -> None:
    """Print source citations (file + page) for transparency."""
    if not context_docs:
        print("Sources: (none)")
        return

    seen = set()
    citations = []
    for d in context_docs:
        src = d.metadata.get("source_file") or d.metadata.get("source") or "unknown"
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            citations.append(f"{src} (page {page + 1})")  # page usually 0-indexed
        else:
            citations.append(f"{src}")

    print("Sources:")
    for c in citations[:10]:
        print(f" - {c}")


def chat_loop():
    vectorstore = build_or_load_vector_db()
    rag_chain = make_rag_chain(vectorstore)

    print("\n--- Offline AI Ready (type 'exit' to quit) ---")
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        print("Thinking...")
        try:
            result = rag_chain.invoke({"input": query})
            answer = result.get("answer", "").strip()
            context_docs = result.get("context", [])

            print(f"\nAI: {answer}\n")
            print_sources(context_docs)

        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    chat_loop()
