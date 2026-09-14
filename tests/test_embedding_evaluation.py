import math

from app.data.products_with_metadata import PRODUCTS_WITH_METADATA
from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService


GROUND_TRUTH = {
    "I want something comfortable and loose to wear every day": {
        "Black Oversized T-Shirt",
        "White Cotton T-Shirt",
        "Gray Hoodie",
    },

    "I need something for running and working out": {
        "Running Shoes",
        "Running Shorts",
        "Dumbbell Set",
        "Yoga Mat",
    },

    "I need equipment for programming and working at my desk": {
        "Mechanical Keyboard",
        "Wireless Mouse",
        "4K Monitor",
        "Office Chair",
    },

    "I am going on an outdoor trip and need useful gear": {
        "Hiking Boots",
        "Camping Tent",
        "Travel Backpack",
        "Stainless Steel Water Bottle",
    },

    "I want something that makes cooking and daily household tasks easier": {
        "Air Fryer",
        "Electric Kettle",
        "Coffee Maker",
        "Robot Vacuum",
    },
}


def precision_at_k(results, relevant, k):
    """
    Calculate Precision@K.

    Precision@K measures how many of the top K
    results are relevant.
    """
    top_k = results[:k]

    relevant_count = sum(
        1
        for product in top_k
        if product["name"] in relevant
    )

    return relevant_count / k


def recall_at_k(results, relevant, k):
    """
    Calculate Recall@K.

    Recall@K measures how many of all relevant
    products were retrieved within the top K.
    """
    top_k = results[:k]

    relevant_count = sum(
        1
        for product in top_k
        if product["name"] in relevant
    )

    return relevant_count / len(relevant)


def reciprocal_rank(results, relevant):
    """
    Calculate Reciprocal Rank.

    Returns the reciprocal of the rank of the
    first relevant result.
    """
    for rank, product in enumerate(results, start=1):
        if product["name"] in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(results, relevant, k):
    """
    Calculate NDCG@K.

    NDCG evaluates the quality of the ranking,
    giving more weight to relevant results
    appearing near the top.
    """
    top_k = results[:k]

    # Calculate DCG
    dcg = 0.0

    for rank, product in enumerate(top_k, start=1):
        relevance = (
            1
            if product["name"] in relevant
            else 0
        )

        dcg += (
            (2**relevance - 1)
            / math.log2(rank + 1)
        )

    # Calculate ideal DCG
    ideal_relevances = [
        1
    ] * min(len(relevant), k)

    idcg = 0.0

    for rank, relevance in enumerate(
        ideal_relevances,
        start=1,
    ):
        idcg += (
            (2**relevance - 1)
            / math.log2(rank + 1)
        )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_query(
    query,
    embedding_service,
    vector_search,
    relevant,
    k=5,
):
    """
    Run semantic search for one query
    and calculate all evaluation metrics.
    """
    query_embedding = embedding_service.embed_query(
        query
    )

    results = vector_search.search(
        query_embedding=query_embedding,
        top_k=k,
    )

    precision = precision_at_k(
        results,
        relevant,
        k,
    )

    recall = recall_at_k(
        results,
        relevant,
        k,
    )

    rr = reciprocal_rank(
        results,
        relevant,
    )

    ndcg = ndcg_at_k(
        results,
        relevant,
        k,
    )

    return {
        "query": query,
        "results": results,
        "precision": precision,
        "recall": recall,
        "rr": rr,
        "ndcg": ndcg,
    }


def evaluate_model(
    embedding_service,
    vector_search,
    ground_truth,
    k=5,
):
    """
    Evaluate an embedding/search model
    across all queries in the ground truth.
    """
    evaluations = []

    for query, relevant in ground_truth.items():
        evaluation = evaluate_query(
            query=query,
            embedding_service=embedding_service,
            vector_search=vector_search,
            relevant=relevant,
            k=k,
        )

        evaluations.append(evaluation)

    return evaluations


def summarize_evaluations(evaluations):
    """
    Calculate average evaluation metrics
    across all queries.
    """
    average_precision = sum(
        evaluation["precision"]
        for evaluation in evaluations
    ) / len(evaluations)

    average_recall = sum(
        evaluation["recall"]
        for evaluation in evaluations
    ) / len(evaluations)

    mean_reciprocal_rank = sum(
        evaluation["rr"]
        for evaluation in evaluations
    ) / len(evaluations)

    average_ndcg = sum(
        evaluation["ndcg"]
        for evaluation in evaluations
    ) / len(evaluations)

    return {
        "precision": average_precision,
        "recall": average_recall,
        "mrr": mean_reciprocal_rank,
        "ndcg": average_ndcg,
    }


def print_evaluation(evaluations, k):
    """
    Print detailed evaluation results
    for every query.
    """
    for index, evaluation in enumerate(
        evaluations,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(
            f"QUERY {index}: "
            f"{evaluation['query']}"
        )
        print("=" * 60)

        for result in evaluation["results"]:
            print(
                f"{result['similarity']:.4f}"
                f" | {result['name']}"
            )

        print(
            f"\nPrecision@{k}: "
            f"{evaluation['precision']:.2f}"
        )

        print(
            f"Recall@{k}:    "
            f"{evaluation['recall']:.2f}"
        )

        print(
            f"Reciprocal Rank: "
            f"{evaluation['rr']:.2f}"
        )

        print(
            f"NDCG@{k}:        "
            f"{evaluation['ndcg']:.2f}"
        )


def print_summary(summary, k):
    """
    Print average evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Average Precision@{k}: "
        f"{summary['precision']:.2f}"
    )

    print(
        f"Average Recall@{k}:    "
        f"{summary['recall']:.2f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{summary['mrr']:.2f}"
    )

    print(
        f"Average NDCG@{k}:      "
        f"{summary['ndcg']:.2f}"
    )


def build_vector_search():
    """
    Create the embedding service and populate
    the vector search index with product embeddings.
    """
    embedding_service = EmbeddingService()

    vector_search = VectorSearchService(
        dimension=embedding_service.dimension
    )

    for product in PRODUCTS_WITH_METADATA:
        text = (
            f"{product['name']}. "
            f"{product['description']}"
        )

        embedding = embedding_service.embed_document(
            text
        )

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

    return embedding_service, vector_search


def main():
    k = 5

    embedding_service, vector_search = (
        build_vector_search()
    )

    evaluations = evaluate_model(
        embedding_service=embedding_service,
        vector_search=vector_search,
        ground_truth=GROUND_TRUTH,
        k=k,
    )

    print_evaluation(
        evaluations,
        k,
    )

    summary = summarize_evaluations(
        evaluations
    )

    print_summary(
        summary,
        k,
    )


if __name__ == "__main__":
    main()