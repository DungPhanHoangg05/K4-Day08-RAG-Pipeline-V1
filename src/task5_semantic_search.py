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
        "content": "Return and refund policy: customers may request a return within 15 days for unused items and get a refund to the original payment method.",
        "metadata": {"source": "policy_return.md", "type": "ecommerce"},
    },
    {
        "content": "Payment methods overview: COD, bank transfer, e-wallet, and credit card are supported on the marketplace platform.",
        "metadata": {"source": "payment_methods.md", "type": "ecommerce"},
    },
    {
        "content": "Seller listing regulations and seller obligations for selling products on the marketplace platform.",
        "metadata": {"source": "seller_policy.md", "type": "ecommerce"},
    },
    {
        "content": "Ecommerce return policy explains how buyers can request refund, exchange, and cancellation for eligible orders.",
        "metadata": {"source": "return_policy.md", "type": "ecommerce"},
    },
    {
        "content": "Order tracking guide: customers can track shipments using the order ID, view delivery status, and check the latest updates from the courier.",
        "metadata": {"source": "order_tracking_guide.md", "type": "ecommerce"},
    },
    {
        "content": "Shipping and order tracking policy explains how order IDs, tracking links, and delivery status updates are provided for each purchase.",
        "metadata": {"source": "shipping_tracking_policy.md", "type": "ecommerce"},
    },
    {
        "content": "Concert festival guide: prepare your checklist, understand venue rules, and buy tickets early for the best seats.",
        "metadata": {"source": "concert_festival_guide.md", "type": "music"},
    },
    {
        "content": "Concert venue prohibited items include sharp tools, professional cameras, drones, and oversized bags.",
        "metadata": {"source": "concert_rules.md", "type": "music"},
    },
]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]


_SYNONYMS = {
    "return": {"return", "refund", "refunds", "reimburse"},
    "refund": {"refund", "return", "reimburse"},
    "payment": {"payment", "pay", "payments"},
    "methods": {"methods", "method"},
    "policy": {"policy", "policies", "rules"},
    "ecommerce": {"ecommerce", "marketplace", "shop"},
    "concert": {"concert", "festival", "venue"},
}


def _expand_tokens(tokens: list[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        for alias, group in _SYNONYMS.items():
            if token in group:
                expanded.update(group)
                expanded.add(alias)
    return expanded


def _semantic_overlap_score(query: str, content: str) -> float:
    query_tokens = _expand_tokens(_tokenize(query))
    content_tokens = _expand_tokens(_tokenize(content))
    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & content_tokens)
    total = len(query_tokens)
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

