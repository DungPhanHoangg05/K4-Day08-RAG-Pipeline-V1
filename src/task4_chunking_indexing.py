"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb python-dotenv
    # (chỉ cần thêm google-generativeai / openai nếu dùng provider tương ứng)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# --- Chunking strategy: RecursiveCharacterTextSplitter ---
# Lý do chọn "recursive":
#   - Corpus gồm nhiều loại văn bản (legal + news), độ dài đoạn văn không đồng đều,
#     không phải lúc nào cũng có heading rõ ràng để dùng MarkdownHeaderTextSplitter.
#   - RecursiveCharacterTextSplitter là lựa chọn an toàn, phổ biến, không cần model
#     phụ (khác với SemanticChunker cần gọi embedding cho từng lần cắt -> chậm & tốn
#     API call nếu dùng provider trả phí).
#   - Dùng list separators ưu tiên cắt theo đoạn ("\n\n") trước, xuống dòng, câu, rồi
#     mới đến ký tự, để giữ ngữ nghĩa chunk trọn vẹn nhất có thể.
CHUNK_SIZE = 500        # 500 ký tự: đủ nhỏ để retrieval chính xác (không lẫn nhiều ý),
                        # đủ lớn để giữ ngữ cảnh cho câu trả lời của RAG.
CHUNK_OVERLAP = 100      # 50 ký tự (10% chunk_size): tránh cắt đứt ý ở ranh giới chunk,
                        # không quá lớn để tránh trùng lặp dữ liệu index.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# --- Embedding model: BAAI/bge-m3 ---
# Lý do:
#   - Multilingual, hỗ trợ tốt cả tiếng Việt lẫn tiếng Anh (corpus có thể lẫn cả 2).
#   - Chạy local (sentence-transformers), không cần API key -> phù hợp dev/test,
#     không phát sinh chi phí gọi API khi reindex nhiều lần.
#   - Đánh đổi: cài đặt nặng hơn (~1-2GB do kéo theo torch) và chậm hơn so với gọi
#     API, nhưng chấp nhận được cho quy mô đồ án.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# EMBEDDING_PROVIDER đọc từ .env để cả nhóm đổi provider mà không cần sửa code.
# Lưu ý: đổi provider phải xoá chroma_db/ cũ và reindex vì dimension khác nhau
# (1024/768/1536) không tương thích ngược trong cùng 1 collection.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

# --- Vector store: ChromaDB ---
# Lý do: đơn giản, chạy local persistent trên đĩa, không cần Docker/Cloud như
# Weaviate. FAISS bị loại vì chỉ hỗ trợ dense search, không lưu metadata tiện lợi
# bằng Chroma cho việc filter theo "type" (legal/news) sau này.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"

_st_model_cache = None  # cache SentenceTransformer model để không load lại nhiều lần


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục {STANDARDIZED_DIR}. "
            "Hãy chạy các task chuẩn hoá dữ liệu trước."
        )

    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file).lower() else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (recursive character splitting).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Dispatch embedding theo EMBEDDING_PROVIDER (.env), dùng chung cho Task 4 & 5
    để không lặp lại logic embed ở 2 nơi.

    Providers hỗ trợ:
        - "sentence_transformers" (mặc định, local, BAAI/bge-m3)
        - "google"  (cần GEMINI_API_KEY, models/text-embedding-004, 768 dim)
        - "openai"  (cần OPENAI_API_KEY, text-embedding-3-small, 1536 dim)
    """
    global _st_model_cache

    if EMBEDDING_PROVIDER == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        if _st_model_cache is None:
            _st_model_cache = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = _st_model_cache.encode(texts, show_progress_bar=True)
        return [emb.tolist() for emb in embeddings]

    elif EMBEDDING_PROVIDER == "google":
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
        )
        return result["embedding"]

    elif EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in response.data]

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn (qua embed_texts dispatch).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (persistent, local).
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM}, provider={EMBEDDING_PROVIDER})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()