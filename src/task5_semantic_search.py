"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import chromadb
from pathlib import Path
from src.task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    embed_texts,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    if not CHROMA_DIR.exists():
        return []

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    # 1. Embed query
    query_embeddings = embed_texts([query])
    if not query_embeddings:
        return []

    # 2. Query ChromaDB
    total_docs = collection.count()
    if total_docs == 0:
        return []

    n_results = min(top_k, total_docs)
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    output = []
    docs = results["documents"][0]
    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
    distances = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

    for doc, meta, dist in zip(docs, metas, distances):
        score = max(0.0, 1.0 - float(dist))  # Cosine distance -> Cosine similarity
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta or {}
        })

    # Sort descending by score
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("quy định tham gia concert âm nhạc", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['metadata'].get('source', '')} - {r['content'][:80]}...")

