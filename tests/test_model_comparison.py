from app.data.products_with_metadata import PRODUCTS_WITH_METADATA
from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService

from .test_embedding_evaluation import (
    GROUND_TRUTH,
    evaluate_query,
    evaluate_model,
    summarize_evaluations,
    print_evaluation,
)


MODELS = [
    "all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-small",
]


def build_vector_search(
    embedding_service: EmbeddingService,
) -> VectorSearchService:
    vector_search = VectorSearchService(
        dimension=embedding_service.dimension
    )

    for product in PRODUCTS_WITH_METADATA:
        text = f"{product['name']}. {product['description']}"

        embedding = embedding_service.embed_document(text)

        vector_search.add_product(
            product_id=product["id"],
            name=product["name"],
            embedding=embedding,
            metadata=product,
        )

    return vector_search


def test_model(model_name: str):
    """
    Test a single embedding model
    and return summary metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"Testing Model: {model_name}")
    print(f"{'=' * 60}")

    embedding_service = EmbeddingService(
        model_name=model_name
    )

    vector_search = build_vector_search(
        embedding_service
    )

    # Run evaluation across all queries
    evaluations = evaluate_model(
        embedding_service=embedding_service,
        vector_search=vector_search,
        ground_truth=GROUND_TRUTH,
        k=5,
    )

    # Print detailed results
    print_evaluation(evaluations, k=5)

    # Calculate summary metrics
    summary = summarize_evaluations(evaluations)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY - {model_name}")
    print(f"{'=' * 60}")
    print(f"Dimension:   {embedding_service.dimension}")
    print(f"Precision@5: {summary['precision']:.4f}")
    print(f"Recall@5:    {summary['recall']:.4f}")
    print(f"MRR@5:       {summary['mrr']:.4f}")
    print(f"NDCG@5:      {summary['ndcg']:.4f}")

    return summary


def main():
    """
    Compare embedding models systematically.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = test_model(model_name)

    # Print final comparison table
    print(f"\n{'=' * 80}")
    print("MODELS COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Model':<35} | {'Precision':>10} | {'Recall':>10} | {'MRR':>10} | {'NDCG':>10}")
    print("-" * 80)

    for model_name, summary in results.items():
        print(
            f"{model_name:<35} | "
            f"{summary['precision']:>10.4f} | "
            f"{summary['recall']:>10.4f} | "
            f"{summary['mrr']:>10.4f} | "
            f"{summary['ndcg']:>10.4f}"
        )

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()