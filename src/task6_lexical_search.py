"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
import numpy as np
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from src.task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        load_documents,
        chunk_documents,
    )
except ImportError:
    try:
        from task4_chunking_indexing import (
            CHROMA_DIR,
            COLLECTION_NAME,
            load_documents,
            chunk_documents,
        )
    except ImportError:
        CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
        COLLECTION_NAME = "ecommerce_support_docs"
        load_documents = None
        chunk_documents = None

# Cache for corpus and BM25 index
_CORPUS_CACHE: list[dict] = []
_BM25_INDEX = None


def tokenize(text: str) -> list[str]:
    """
    Tokenize text thành danh sách các từ/tokens.
    Chuyển về chữ thường và loại bỏ các ký tự đặc biệt.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def load_corpus() -> list[dict]:
    """
    Load corpus từ ChromaDB (ưu tiên) hoặc từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    global _CORPUS_CACHE
    if _CORPUS_CACHE:
        return _CORPUS_CACHE

    corpus = []

    # 1. Thử load từ ChromaDB vector store
    if chromadb is not None and CHROMA_DIR.exists():
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(name=COLLECTION_NAME)
            results = collection.get(include=["documents", "metadatas"])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            if docs:
                for doc, meta in zip(docs, metas):
                    corpus.append({
                        "content": doc,
                        "metadata": meta or {}
                    })
        except Exception:
            corpus = []

    # 2. Nếu chưa lấy được từ ChromaDB, thử load và chunk từ data/standardized/
    if not corpus and load_documents is not None and chunk_documents is not None:
        try:
            raw_docs = load_documents()
            corpus = chunk_documents(raw_docs)
        except Exception:
            corpus = []

    _CORPUS_CACHE = corpus
    return _CORPUS_CACHE


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]

    if BM25Okapi is not None:
        return BM25Okapi(tokenized_corpus)
    else:
        # Simple fallback BM25 implementation if rank_bm25 is not installed
        class FallbackBM25:
            def __init__(self, corpus_tokens, k1=1.5, b=0.75):
                self.k1 = k1
                self.b = b
                self.corpus_size = len(corpus_tokens)
                self.doc_len = [len(doc) for doc in corpus_tokens]
                self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1.0
                self.doc_freqs = []
                self.idf = {}

                df = {}
                for doc in corpus_tokens:
                    frequencies = {}
                    for word in doc:
                        frequencies[word] = frequencies.get(word, 0) + 1
                    self.doc_freqs.append(frequencies)
                    for word in frequencies:
                        df[word] = df.get(word, 0) + 1

                for word, freq in df.items():
                    import math
                    self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

            def get_scores(self, query_tokens):
                scores = np.zeros(self.corpus_size)
                for q in query_tokens:
                    if q not in self.idf:
                        continue
                    q_idf = self.idf[q]
                    for idx, doc_freq in enumerate(self.doc_freqs):
                        if q in doc_freq:
                            freq = doc_freq[q]
                            denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                            scores[idx] += q_idf * (freq * (self.k1 + 1)) / denom
                return scores

        return FallbackBM25(tokenized_corpus)


def get_bm25_index():
    """
    Lấy hoặc khởi tạo singleton BM25 index.
    """
    global _BM25_INDEX
    if _BM25_INDEX is None:
        corpus = load_corpus()
        _BM25_INDEX = build_bm25_index(corpus)
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    corpus = load_corpus()
    if not corpus:
        return []

    bm25 = get_bm25_index()
    if bm25 is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score_val = float(scores[idx])
        if score_val > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": round(score_val, 4),
                "metadata": corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

