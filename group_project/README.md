# Bài Tập Nhóm — Concert Event Support RAG Chatbot

## Mục Tiêu

Nhóm xây dựng **2 sản phẩm** sau khi hoàn thành bài cá nhân:
1. **RAG Chatbot** hỏi đáp về chính sách sự kiện Concert Anh Trai Say Hi 2025
2. **Evaluation Pipeline** đánh giá chất lượng hệ thống bằng RAGAS

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot ✅

Chatbot trả lời câu hỏi về chính sách mua vé, quy định tham gia, check-in, đổi trả vé và độ tuổi Concert Anh Trai Say Hi 2025.

**Đã thực hiện:**
- Giao diện chat bằng **Streamlit** (`app.py`)
- Trả lời có citation theo nguồn tài liệu (Task 10)
- Hỗ trợ follow-up questions (conversation memory qua `st.session_state`)
- Hiển thị source documents kèm điểm score và đoạn trích nội dung
- Sidebar với câu hỏi gợi ý và điều chỉnh `top_k`

**Stack thực tế:**
```
Streamlit UI
    └→ Task 9: Retrieval Pipeline (Hybrid: BM25 + Semantic + RRF Rerank + PageIndex Fallback)
        └→ Task 10: Generation có Citation (google/gemma-4-31b-it qua OpenRouter)
            └→ Streamlit Display (answer + source expander)
```

---

## Yêu cầu 2: RAG Evaluation Pipeline ✅

Sử dụng **RAGAS** để đánh giá pipeline RAG với 4 metrics trên 15 câu hỏi golden dataset.

### Framework đã chọn

| Framework | Cài đặt | Lý do chọn |
|-----------|---------|------------|
| **RAGAS** ✅ | `pip install ragas` | Chuẩn industry cho RAG eval, tích hợp trực tiếp với HuggingFace `datasets`, hỗ trợ đủ 4 metrics yêu cầu |

### Kết quả Evaluation

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|:---:|:---:|:---:|
| Faithfulness | **0.91** | 0.78 | +0.13 |
| Answer Relevancy | **0.87** | 0.81 | +0.06 |
| Context Recall | **0.89** | 0.74 | +0.15 |
| Context Precision | **0.84** | 0.69 | +0.15 |
| **Average** | **0.8775** | **0.755** | **+0.1225** |

→ Xem chi tiết tại [`evaluation/results.md`](evaluation/results.md)

### Deliverables Evaluation

- [x] `group_project/evaluation/golden_dataset.json` — 15 cặp Q&A về sự kiện Concert
- [x] `group_project/evaluation/eval_pipeline.py` — script chạy evaluation (RAGAS)
- [x] `group_project/evaluation/results.md` — bảng điểm + phân tích worst performers + đề xuất
- [x] So sánh A/B: Config A (Hybrid + Rerank) vs Config B (Dense-only)

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  data/standardized/  ──→  Task 4: Chunking & Indexing           │
│  (5 markdown files)        └→ ChromaDB (chroma_db/)             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     RETRIEVAL LAYER                             │
│                                                                 │
│   Query ──┬──→ Task 5: Semantic Search (ChromaDB + Embeddings)  │
│           │         (sentence-transformers / Google / OpenAI)   │
│           ├──→ Task 6: Lexical Search (BM25 via rank-bm25)      │
│           │                                                     │
│           └──→ Task 7: Reranking (RRF Fusion)                   │
│                    │                                            │
│                    ├── score ≥ 0.3 → Hybrid Results             │
│                    └── score < 0.3 → Task 8: PageIndex Fallback │
└─────────────────────┬───────────────────────────────────────────┘
                      │  Task 9: retrieve()  (top-K chunks)
┌─────────────────────▼───────────────────────────────────────────┐
│                    GENERATION LAYER                             │
│   Task 10: generate_with_citation()                             │
│   ├─ Reorder chunks (lost-in-the-middle mitigation)             │
│   ├─ Format context với source labels                           │
│   └─ LLM: google/gemma-4-31b-it (OpenRouter)                   │
│          temperature=0.3, top_p=0.9, top_k=5                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                       UI LAYER                                  │
│   Streamlit (app.py)                                            │
│   ├─ Chat history (session_state)                               │
│   ├─ Answer với citation inline                                 │
│   ├─ Source expander (chunk content + score)                    │
│   └─ Sidebar: suggestions, top_k slider, kiến trúc mô tả       │
└─────────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  EVALUATION LAYER                               │
│   RAGAS (group_project/evaluation/)                             │
│   ├─ golden_dataset.json — 15 Q&A pairs                         │
│   ├─ eval_pipeline.py — script chạy A/B evaluation             │
│   └─ results.md — báo cáo kết quả đầy đủ                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|:---:|
| Phan Hoàng Dũng | 2A202601348 | Role 1: **Team Leader & RAG Architect** | Done |
| Tòng Văn Tiến | 2A202601996 | Role 2: **Data & Dense Search Dev** | Done |
| Ngô Nguyễn Khải Hưng | 2A202601216 | Role 3: **Sparse & Rerank Dev** | Done |
| Mai Tiến Mạnh | 2A202601922 | Role 4: **Frontend & Chatbot Dev** | Done |
| Phạm Duy Hoàn | 2A202601378 | Role 5: **Evaluation & QA Engineer** | Done |

---

## Hướng Dẫn Cài Đặt & Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API key

Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Điền các key cần thiết:

```env
# LLM Generation (chọn 1)
OPENROUTER_API_KEY=sk-or-...   # Dùng google/gemma-4-31b-it free
OPENAI_API_KEY=sk-...           # Hoặc OpenAI trực tiếp

# Embedding (mặc định: sentence-transformers, không cần key)
EMBEDDING_PROVIDER=sentence_transformers
```

### 3. Index tài liệu (lần đầu)

```bash
python -m src.task4_chunking_indexing
```

### 4. Chạy Chatbot

```bash
streamlit run app.py
```

Truy cập tại: http://localhost:8501

### 5. Chạy Evaluation (tùy chọn)

```bash
python group_project/evaluation/eval_pipeline.py
```

> ⚠️ **Lưu ý rate limit:** RAGAS gọi LLM RẤT NHIỀU LẦN (nhiều lần/metric/câu hỏi). Với model `:free` của OpenRouter (giới hạn 50 req/ngày), chỉ nên chạy subset 5 câu để tránh bị block.

---

## Cấu Trúc Thư Mục

```
K4-Day08-RAG-Pipeline-V1/
├── app.py                          # Streamlit chatbot UI
├── requirements.txt
├── .env / .env.example
├── data/
│   └── standardized/               # Tài liệu sự kiện (5 file .md)
├── chroma_db/                      # ChromaDB vector store (auto-generated)
├── src/
│   ├── task4_chunking_indexing.py  # Chunking + ChromaDB indexing
│   ├── task5_semantic_search.py    # Dense search (embeddings)
│   ├── task6_lexical_search.py     # BM25 sparse search
│   ├── task7_reranking.py          # RRF / Cross-encoder / MMR rerank
│   ├── task8_pageindex_vectorless.py # PageIndex fallback
│   ├── task9_retrieval_pipeline.py # Pipeline tích hợp (entry point)
│   └── task10_generation.py        # LLM generation có citation
└── group_project/
    ├── README.md                   # File này
    └── evaluation/
        ├── golden_dataset.json     # 15 cặp Q&A ground truth
        ├── eval_pipeline.py        # RAGAS evaluation script
        └── results.md              # Kết quả A/B + phân tích
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
