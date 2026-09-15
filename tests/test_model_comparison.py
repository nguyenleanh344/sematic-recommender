import time

from app.data.products_with_metadata import PRODUCTS_WITH_METADATA
from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService

from tests.test_embedding_evaluation import (
    GROUND_TRUTH,
    evaluate_model,
    summarize_evaluations,
)


MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "E5": "intfloat/multilingual-e5-small",
}


def build_vector_search(model_name):
    embedding_service = EmbeddingService(
        model_name=model_name
    )

    vector_search = VectorSearchService(
        dimension=embedding_service.dimension
    )

    document_embedding_start = time.perf_counter()

    for product in PRODUCTS_WITH_METADATA:
        text = (
            f"{product['name']}. "
            f"{product['description']}"
        )

        embedding = embedding_service.embed_document(text)

        metadata = {
            "category": product["category"],
            "subcategory": product["subcategory"],
        }

        vector_search.add_product(
            product_id=product["id"],
            name=product["name"],
            embedding=embedding,
            metadata=metadata,
        )

    document_embedding_time = (
        time.perf_counter()
        - document_embedding_start
    )

    return (
        embedding_service,
        vector_search,
        document_embedding_time,
    )


def measure_query_latency(
    embedding_service,
    queries,
    warmup_runs=1,
    benchmark_runs=10,
):
    """
    Measure average query embedding latency.
    """

    # Warm up the model before measuring.
    for query in queries[:warmup_runs]:
        embedding_service.embed_query(query)

    total_time = 0.0
    total_runs = 0

    for _ in range(benchmark_runs):
        for query in queries:
            start = time.perf_counter()

            embedding_service.embed_query(query)

            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_runs += 1

    return total_time / total_runs


def evaluate_model_name(model_name):
    (
        embedding_service,
        vector_search,
        document_embedding_time,
    ) = build_vector_search(model_name)

    evaluations = evaluate_model(
        embedding_service=embedding_service,
        vector_search=vector_search,
        ground_truth=GROUND_TRUTH,
        k_values=(5, 10),
    )

    summary = summarize_evaluations(evaluations)

    queries = list(GROUND_TRUTH.keys())

    query_latency = measure_query_latency(
        embedding_service=embedding_service,
        queries=queries,
    )

    return {
        "summary": summary,
        "dimension": embedding_service.dimension,
        "document_embedding_time": document_embedding_time,
        "query_latency": query_latency,
    }


def main():
    results = {}

    for name, model_name in MODELS.items():
        print()
        print("=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        results[name] = evaluate_model_name(
            model_name
        )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, result in results.items():
        summary = result["summary"]

        print()
        print(name)

        print(
            f"Dimension: "
            f"{result['dimension']}"
        )

        print(
            f"Document embedding time: "
            f"{result['document_embedding_time']:.4f}s"
        )

        print(
            f"Average query latency: "
            f"{result['query_latency'] * 1000:.2f} ms"
        )

        for k, metrics in summary.items():
            print(
                f"@{k} | "
                f"Precision={metrics['precision']:.2f} | "
                f"Recall={metrics['recall']:.2f} | "
                f"MRR={metrics['mrr']:.2f} | "
                f"NDCG={metrics['ndcg']:.2f}"
            )


if __name__ == "__main__":
    main()