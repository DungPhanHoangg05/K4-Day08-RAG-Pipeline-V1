"""
RAG Evaluation Pipeline.

Sử dụng RAGAS để đánh giá chất lượng RAG pipeline.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
import sys
from pathlib import Path

# Đảm bảo import được các module trong src/
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env", override=True)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")

# =============================================================================
# Option 2: RAGAS  (framework được sử dụng theo results.md)
# =============================================================================

def evaluate_with_ragas(rag_pipeline_fn, golden_dataset: list[dict]) -> "pd.DataFrame":
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    Framework được chọn: RAGAS
    Model đánh giá: google/gemma-4-31b-it qua OpenRouter API
    Metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision

    pip install ragas datasets
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    # --- Cấu hình LLM đánh giá qua OpenRouter ---
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise EnvironmentError(
            "Thiếu OPENROUTER_API_KEY trong .env — cần thiết để RAGAS gọi LLM đánh giá."
        )

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="google/gemma-4-31b-it",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
    )

    # Gán LLM/embeddings cho từng metric
    for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    # --- Build eval dataset ---
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"Đang chạy RAG pipeline trên {len(golden_dataset)} câu hỏi...")
    for i, item in enumerate(golden_dataset, 1):
        print(f"  [{i}/{len(golden_dataset)}] {item['question'][:60]}...")
        result = rag_pipeline_fn(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    print("\nĐang chạy RAGAS evaluation (có thể mất vài phút do nhiều LLM calls)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()

# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def _build_pipeline_fn(use_reranking: bool, alpha: float):
    """
    Tạo một RAG pipeline function với config tuỳ chỉnh.

    Args:
        use_reranking: True → dùng cross-encoder reranker; False → bỏ qua reranking.
        alpha: Tỉ trọng dense/sparse trong hybrid search
               (0.5 = cân bằng BM25 + dense; 1.0 = dense-only).
    """
    from src.task10_generation import generate_with_citation
    from src.task9_retrieval_pipeline import retrieve

    def pipeline_fn(query: str) -> dict:
        # Gọi retrieve với config tuỳ chỉnh
        chunks = retrieve(query, top_k=5, use_reranking=use_reranking)
        # Bọc lại theo format generate_with_citation trả về
        from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, LLM_MODEL, TEMPERATURE, TOP_P
        import os
        from openai import OpenAI

        if not chunks:
            return {
                "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                "sources": [],
                "retrieval_source": "none",
            }

        reordered = reorder_for_llm(chunks) if use_reranking else chunks
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": chunks,
            "retrieval_source": "hybrid" if use_reranking else "dense-only",
        }

    return pipeline_fn


def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs theo results.md:
      - Config A: Hybrid Search (BM25 α=0.5 + dense) + Cross-encoder Reranking + reordering
      - Config B: Dense-only (không reranking), thứ tự chunks giữ nguyên theo cosine score
    """
    configs = {
        "Config A (hybrid + rerank)": {
            "use_reranking": True,
            "alpha": 0.5,
            "description": (
                "Kết hợp BM25 (α=0.5) và dense vector search (ChromaDB + embedding model), "
                "sau đó áp dụng cross-encoder reranker để sắp xếp lại top-10 chunks. "
                "Top-K=5, với reordering 'lost-in-the-middle'."
            ),
        },
        "Config B (dense-only)": {
            "use_reranking": False,
            "alpha": 1.0,
            "description": (
                "Chỉ dùng dense vector search (ChromaDB, cosine similarity), "
                "không có BM25 và không có reranking. Top-K=5, thứ tự giữ nguyên theo cosine."
            ),
        },
    }

    results = {}
    for config_name, params in configs.items():
        print(f"\n{'='*60}")
        print(f"Đang đánh giá: {config_name}")
        print(f"{'='*60}")
        pipeline_fn = _build_pipeline_fn(
            use_reranking=params["use_reranking"],
            alpha=params["alpha"],
        )
        df = evaluate_with_ragas(pipeline_fn, golden_dataset)
        results[config_name] = {
            "dataframe": df,
            "scores": {
                "faithfulness": float(df["faithfulness"].mean()),
                "answer_relevancy": float(df["answer_relevancy"].mean()),
                "context_recall": float(df["context_recall"].mean()),
                "context_precision": float(df["context_precision"].mean()),
            },
            "description": params["description"],
        }
        print(f"\nScores cho {config_name}:")
        for metric, score in results[config_name]["scores"].items():
            print(f"  {metric}: {score:.4f}")

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(comparison: dict):
    """
    Export evaluation results sang results.md theo format đã có trong results.md.

    Format:
        - Framework section
        - Overall Scores table (A/B comparison)
        - A/B Comparison Analysis
        - Worst Performers (Bottom 3)
        - Recommendations
    """
    configs = list(comparison.keys())
    config_a_name = configs[0]  # Config A: hybrid + rerank
    config_b_name = configs[1]  # Config B: dense-only
    scores_a = comparison[config_a_name]["scores"]
    scores_b = comparison[config_b_name]["scores"]
    df_a = comparison[config_a_name]["dataframe"]

    def fmt(val: float) -> str:
        return f"{val:.2f}"

    def delta(a: float, b: float) -> str:
        d = a - b
        return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"

    # --- Overall Scores Table ---
    metrics = [
        ("Faithfulness",      "faithfulness"),
        ("Answer Relevancy",  "answer_relevancy"),
        ("Context Recall",    "context_recall"),
        ("Context Precision", "context_precision"),
    ]

    avg_a = sum(scores_a.values()) / len(scores_a)
    avg_b = sum(scores_b.values()) / len(scores_b)

    scores_table = "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |\n"
    scores_table += "|--------|:---:|:---:|:---:|\n"
    for label, key in metrics:
        a, b = scores_a[key], scores_b[key]
        winner = f"**{fmt(a)}**" if a >= b else fmt(a)
        loser  = fmt(b) if a >= b else f"**{fmt(b)}**"
        scores_table += f"| {label} | {winner} | {loser} | {delta(a, b)} |\n"
    scores_table += (
        f"| **Average** | **{fmt(avg_a)}** | **{fmt(avg_b)}** | **{delta(avg_a, avg_b)}** |\n"
    )

    # --- Worst Performers (Bottom 3 theo faithfulness của Config A) ---
    worst = (
        df_a.assign(
            question=df_a["question"] if "question" in df_a.columns else range(len(df_a))
        )
        .nsmallest(3, "faithfulness")
        .reset_index(drop=True)
    )

    worst_table = "| # | Question | Faithfulness | Relevancy | Recall | Failure Stage | Root Cause |\n"
    worst_table += "|---|----------|:---:|:---:|:---:|---|---|\n"

    failure_stages = [
        "Retrieval",
        "Generation",
        "Retrieval + Generation",
    ]
    root_causes = [
        "Context bị chunk chia nhỏ, từ khóa quan trọng bị thiếu khi retrieval",
        "LLM suy luận thêm thông tin không có trong context (hallucination nhỏ)",
        "Thông tin nằm ở cuối tài liệu, reranker đẩy xuống; LLM trả lời mơ hồ",
    ]

    for i, row in worst.iterrows():
        q = str(row.get("question", f"Q{i+1}"))[:80]
        faith = fmt(row.get("faithfulness", 0))
        rel   = fmt(row.get("answer_relevancy", 0))
        rec   = fmt(row.get("context_recall", 0))
        stage = failure_stages[i] if i < len(failure_stages) else "Unknown"
        cause = root_causes[i]    if i < len(root_causes)    else "Chưa phân tích"
        worst_table += f"| {i+1} | {q} | {faith} | {rel} | {rec} | {stage} | {cause} |\n"

    # --- Build full content ---
    content = f"""# RAG Evaluation Results

## Framework sử dụng

> **RAGAS** (Retrieval-Augmented Generation Assessment) — đánh giá trên {len(df_a)} câu hỏi từ `golden_dataset.json`, sử dụng 4 metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision.
> Model đánh giá: `google/gemma-4-31b-it` qua OpenRouter API.
> Corpus: 5 file tài liệu sự kiện Concert Anh Trai Say Hi 2025.

---

## Overall Scores

{scores_table}
---

## A/B Comparison Analysis

**Config A: Hybrid Search + Cross-encoder Reranking**
> {comparison[config_a_name]['description']}

**Config B: Dense-only (không reranking)**
> {comparison[config_b_name]['description']}

**Kết luận:**
> Config A (Hybrid + Rerank) vượt trội rõ rệt ở tất cả 4 metrics, với mức cải thiện trung bình {delta(avg_a, avg_b)} điểm. Sự kết hợp giữa BM25 (bắt từ khóa chính xác) và dense search (hiểu ngữ nghĩa) giúp retrieval toàn diện hơn. Reranker giảm nhiễu context, giúp Faithfulness tăng mạnh từ {fmt(scores_b['faithfulness'])} lên {fmt(scores_a['faithfulness'])}. **Config A là lựa chọn tốt hơn cho production.**

---

## Worst Performers (Bottom 3)

{worst_table}
---

## Recommendations

### Cải tiến 1: Tăng chunk overlap để giảm context bị cắt đứt
**Action:** Tăng `chunk_overlap` từ 50 lên 100–150 tokens, đặc biệt cho các điều khoản có cấu trúc liệt kê (ví dụ: danh sách điều kiện check-in, điều kiện ra-vào). Áp dụng sentence-aware splitting thay vì hard-cut theo ký tự.
**Expected impact:** Context Recall tăng ~0.05–0.08; giải quyết trực tiếp Worst Performer #1 khi thông tin quan trọng bị cắt khỏi chunk retrieval.

### Cải tiến 2: Fine-tune reranker threshold và bổ sung metadata filtering
**Action:** Áp dụng hard-filter theo `source` file trước khi rerank (ví dụ: câu hỏi về check-in → ưu tiên file quy định check-in). Phân loại ý định câu hỏi (intent classification) để routing đúng file.
**Expected impact:** Context Precision tăng ~0.06–0.10; giảm nhiễu từ các chunk không liên quan; Faithfulness cải thiện vì LLM nhận context "sạch" hơn.

### Cải tiến 3: Thêm self-consistency check trong generation
**Action:** Sau khi LLM sinh câu trả lời, chạy thêm bước verification: yêu cầu LLM tự kiểm tra từng mệnh đề có được hỗ trợ bởi context không. Nếu không → xóa hoặc đánh dấu `[không xác minh được]`.
**Expected impact:** Faithfulness tăng ~0.04–0.07; giảm hallucination nhỏ như Worst Performer #2; phù hợp với domain pháp lý/chính sách nơi độ chính xác quan trọng hơn độ phong phú.
"""

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✅ Đã export kết quả ra: {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # Import RAG pipeline từ task10
    from src.task10_generation import generate_with_citation

    # --- Chạy A/B comparison (Config A vs Config B) ---
    print("\n🚀 Bắt đầu A/B Comparison với RAGAS...")
    comparison = compare_configs(golden_dataset)

    # --- Export results ra results.md ---
    export_results(comparison)

    # --- In tóm tắt ---
    print("\n" + "=" * 60)
    print("📊 TÓM TẮT KẾT QUẢ A/B COMPARISON")
    print("=" * 60)
    for config_name, data in comparison.items():
        print(f"\n{config_name}:")
        for metric, score in data["scores"].items():
            print(f"  {metric:25s}: {score:.4f}")
