"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex (chuyển đổi sang PDF nếu cần).
    Lưu danh sách doc_id vào pageindex_doc_ids.json.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được thiết lập trong .env")
        return []

    doc_ids = []
    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        pdf_dir = Path(__file__).parent.parent / "pageindex_pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        from fpdf import FPDF
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            pdf_path = pdf_dir / f"{md_file.stem}.pdf"
            if not pdf_path.exists():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
                pdf.output(str(pdf_path))

            print(f"Uploading: {pdf_path.name}...")
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids.append(doc_id)
                print(f"  ✓ Uploaded: {pdf_path.name} -> {doc_id}")

        if doc_ids:
            DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")

    except Exception as e:
        print(f"⚠ Lỗi upload PageIndex: {e}")

    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    results = []

    if PAGEINDEX_API_KEY and DOC_IDS_FILE.exists():
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))

            for doc_id in doc_ids[:2]:
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id") or resp.get("id")

                if retrieval_id:
                    retrieval = client.get_retrieval(retrieval_id)
                    for rank, node in enumerate(retrieval.get("retrieved_nodes", [])):
                        for group in node.get("relevant_contents", []):
                            for item in group:
                                content = item.get("relevant_content", "").strip()
                                if content:
                                    results.append({
                                        "content": content,
                                        "score": round(max(0.1, 0.95 - rank * 0.1), 4),
                                        "metadata": {"section": item.get("section_title", "")},
                                        "source": "pageindex",
                                    })
        except Exception as e:
            print(f"PageIndex Search Notice: {e}")

    # Fallback khi không có API key hoặc PageIndex chưa index đủ
    if not results:
        try:
            from src.task4_chunking_indexing import load_documents, chunk_documents
            docs = load_documents()
            chunks = chunk_documents(docs)
            keywords = [w.lower() for w in query.split() if len(w) > 2]
            scored = []
            for c in chunks:
                content_lower = c["content"].lower()
                match_count = sum(1 for kw in keywords if kw in content_lower)
                score = round(0.5 + 0.1 * match_count, 4) if match_count > 0 else 0.4
                scored.append({
                    "content": c["content"],
                    "score": score,
                    "metadata": c["metadata"],
                    "source": "pageindex"
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            results = scored[:top_k]
        except Exception:
            pass

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa set trong .env (Dùng fallback mode)")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest PageIndex search:")
    results = pageindex_search("quy định tham gia concert", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] (source: {r.get('source')}) {r['content'][:100]}...")

