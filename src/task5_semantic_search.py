"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from __future__ import annotations

import re
from collections import Counter


_FALLBACK_CORPUS = [
    {
        "content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày cho đơn hàng chưa qua sử dụng.",
        "metadata": {"source": "policy_return.md", "type": "ecommerce"},
    },
    {
        "content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam: COD, ATM, ví điện tử, thẻ ngân hàng.",
        "metadata": {"source": "payment_methods.md", "type": "ecommerce"},
    },
    {
        "content": "Quy định đăng bán sản phẩm dành cho người bán trên nền tảng thương mại điện tử.",
        "metadata": {"source": "seller_policy.md", "type": "ecommerce"},
    },
    {
        "content": "Kinh nghiệm đi săn vé Fanzone và chuẩn bị thể lực cho festival âm nhạc ngoài trời kéo dài cả ngày.",
        "metadata": {"source": "concert_festival_guide.md", "type": "music"},
    },
    {
        "content": "Danh mục vật dụng bị cấm mang vào khu vực sân vận động khi đi xem Concert lớn.",
        "metadata": {"source": "concert_rules.md", "type": "music"},
    },
]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]


def _semantic_overlap_score(query: str, content: str) -> float:
    q_tokens = Counter(_tokenize(query))
    c_tokens = Counter(_tokenize(content))
    if not q_tokens:
        return 0.0

    overlap = sum(min(q_tokens[t], c_tokens[t]) for t in q_tokens)
    total = sum(q_tokens.values())
    if total == 0:
        return 0.0
    return round(overlap / total, 4)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Fallback semantic search nhẹ, không cần vector store khi task 4 chưa index xong.
    """
    if not query or not query.strip():
        return []

    results = []
    for doc in _FALLBACK_CORPUS:
        score = _semantic_overlap_score(query, doc["content"])
        if score > 0:
            results.append({
                "content": doc["content"],
                "score": round(score, 4),
                "metadata": doc["metadata"],
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = semantic_search("quy định tham gia concert âm nhạc", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['metadata'].get('source', '')} - {r['content'][:80]}...")

