# RAG Evaluation Results

## Framework sử dụng

> **RAGAS** (Retrieval-Augmented Generation Assessment) — đánh giá trên 15 câu hỏi từ `golden_dataset.json`, sử dụng 4 metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision.
> Model đánh giá: `google/gemma-4-31b-it` qua OpenRouter API.
> Corpus: 5 file tài liệu sự kiện Concert Anh Trai Say Hi 2025.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|:---:|:---:|:---:|
| Faithfulness | **0.91** | 0.78 | +0.13 |
| Answer Relevancy | **0.87** | 0.81 | +0.06 |
| Context Recall | **0.89** | 0.74 | +0.15 |
| Context Precision | **0.84** | 0.69 | +0.15 |
| **Average** | **0.8775** | **0.755** | **+0.1225** |

---

## A/B Comparison Analysis

**Config A: Hybrid Search + Cross-encoder Reranking**
> Kết hợp BM25 (lexical, α = 0.5) và dense vector search (ChromaDB + embedding model), sau đó áp dụng cross-encoder reranker để sắp xếp lại top-10 chunks trước khi đưa vào LLM. Top-K = 5, với reordering "lost-in-the-middle" (chunks quan trọng nhất ở đầu và cuối prompt).

**Config B: Dense-only (không reranking)**
> Chỉ dùng dense vector search (ChromaDB, cosine similarity), không có BM25 và không có reranking. Top-K = 5, thứ tự chunks giữ nguyên theo cosine score.

**Kết luận:**
> Config A (Hybrid + Rerank) vượt trội rõ rệt ở tất cả 4 metrics, với mức cải thiện trung bình +0.12 điểm. Sự kết hợp giữa BM25 (bắt từ khóa chính xác như "Ticketbox", "14 tuổi", "25cm × 35cm") và dense search (hiểu ngữ nghĩa) giúp retrieval toàn diện hơn. Reranker giảm nhiễu context, giúp Faithfulness tăng mạnh từ 0.78 lên 0.91. **Config A là lựa chọn tốt hơn cho production.**

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevancy | Recall | Failure Stage | Root Cause |
|---|----------|:---:|:---:|:---:|---|---|
| 1 | Khán giả cần những yếu tố gì để được phép ra/vào lại khu vực biểu diễn trong khi sự kiện diễn ra? | 0.62 | 0.71 | 0.58 | Retrieval | Câu hỏi về "ra/vào lại" bị chunk chia nhỏ, context thiếu đoạn nói về "Dấu Mộc di chuyển"; BM25 không khớp từ khóa chính xác |
| 2 | Hành vi livestream hoặc quay phim tiết mục biểu diễn bằng thiết bị chuyên nghiệp bị xử lý như thế nào? | 0.69 | 0.74 | 0.65 | Generation | LLM suy luận thêm hình thức phạt cụ thể không có trong context; Faithfulness thấp do hallucination nhỏ |
| 3 | Thời hạn cuối cùng để khán giả gửi các khiếu nại về vé và quyền lợi kèm vé là khi nào? | 0.71 | 0.76 | 0.67 | Retrieval + Generation | Thông tin nằm trong đoạn cuối của file quy định, reranker đẩy xuống dưới; LLM trả lời mơ hồ về "thời điểm đóng cửa" |

---

## Recommendations

### Cải tiến 1: Tăng chunk overlap để giảm context bị cắt đứt
**Action:** Tăng `chunk_overlap` từ 50 lên 100–150 tokens, đặc biệt cho các điều khoản có cấu trúc liệt kê (ví dụ: danh sách điều kiện check-in, điều kiện ra-vào). Áp dụng sentence-aware splitting thay vì hard-cut theo ký tự.
**Expected impact:** Context Recall tăng ~0.05–0.08; giải quyết trực tiếp Worst Performer #1 khi "Dấu Mộc di chuyển" bị cắt khỏi chunk retrieval.

### Cải tiến 2: Fine-tune reranker threshold và bổ sung metadata filtering
**Action:** Áp dụng hard-filter theo `source` file trước khi rerank (ví dụ: câu hỏi về check-in → ưu tiên `File 5_Quy_dinh_check_in.md`). Phân loại ý định câu hỏi (intent classification) để routing đúng file.
**Expected impact:** Context Precision tăng ~0.06–0.10; giảm nhiễu từ các chunk không liên quan; Faithfulness cải thiện vì LLM nhận context "sạch" hơn.

### Cải tiến 3: Thêm self-consistency check trong generation
**Action:** Sau khi LLM sinh câu trả lời, chạy thêm bước verification: yêu cầu LLM tự kiểm tra từng mệnh đề có được hỗ trợ bởi context không. Nếu không → xóa hoặc đánh dấu `[không xác minh được]`.
**Expected impact:** Faithfulness tăng ~0.04–0.07; giảm hallucination nhỏ như Worst Performer #2; phù hợp với domain pháp lý/chính sách nơi độ chính xác quan trọng hơn độ phong phú.
