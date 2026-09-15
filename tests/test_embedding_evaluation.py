import math

from app.data.products_with_metadata import PRODUCTS_WITH_METADATA
from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService

GROUND_TRUTH = {
    "I want something comfortable and loose to wear every day": {
        "Black Oversized T-Shirt": 3,
        "White Cotton T-Shirt": 3,
        "Gray Hoodie": 2,
        "Blue Denim Jeans": 2,
        "Black Jogger Pants": 2,
        "Oversized Gray T-Shirt": 3,
        "Cotton Polo Shirt": 1,
        "Leather Jacket": 1,
    },

    "I need something for running and working out": {
        "Running Shoes": 3,
        "Running Shorts": 3,
        "Dumbbell Set": 3,
        "Yoga Mat": 3,
        "Training Shoes": 3,
        "Trail Running Shoes": 2,
        "Resistance Bands": 2,
        "Kettlebell": 2,
        "Exercise Bench": 2,
        "Running Socks": 1,
        "Gym Bag": 1,
    },

    "I need equipment for programming and working at my desk": {
        "Mechanical Keyboard": 3,
        "Wireless Mouse": 3,
        "4K Monitor": 3,
        "Office Chair": 3,
        "Laptop Stand": 2,
        "27-inch QHD Monitor": 3,
        "Laptop Docking Station": 2,
        "Monitor Arm": 2,
        "Wireless Keyboard": 2,
        "USB-C Hub": 2,
        "Desk Lamp": 2,
        "Desk Mat": 1,
        "Webcam": 1,
        "Desk Organizer": 1,
        "Standing Desk": 1,
    },

    "I am going on an outdoor trip and need useful gear": {
        "Camping Tent": 3,
        "Hiking Boots": 3,
        "Travel Backpack": 3,
        "Stainless Steel Water Bottle": 3,
        "Trekking Backpack": 3,
        "Sleeping Bag": 3,
        "Camping Stove": 2,
        "Camping Lantern": 2,
        "Hiking Poles": 2,
        "Waterproof Dry Bag": 2,
        "Portable Camping Chair": 1,
        "Travel Pillow": 1,
        "Portable Water Filter": 2,
        "Travel Organizer": 1,
        "Travel Adapter": 1,
    },

    "I want something that makes cooking and daily household tasks easier": {
        "Air Fryer": 3,
        "Electric Kettle": 3,
        "Coffee Maker": 3,
        "Robot Vacuum": 3,
        "Toaster": 2,
        "Blender": 2,
        "Rice Cooker": 2,
        "Food Processor": 2,
        "Electric Grill": 2,
        "Dishwasher": 2,
        "Cordless Vacuum": 2,
        "Robot Mop": 2,
        "Kitchen Scale": 1,
        "Storage Container Set": 1,
    },
}


def precision_at_k(results, relevance, k):
    top_k = results[:k]

    relevant_count = sum(
        relevance.get(item["name"], 0) > 0
        for item in top_k
    )

    return relevant_count / k


def recall_at_k(results, relevance, k):
    top_k = results[:k]

    relevant_count = sum(
        relevance.get(item["name"], 0) > 0
        for item in top_k
    )

    total_relevant = sum(
        grade > 0
        for grade in relevance.values()
    )

    if total_relevant == 0:
        return 0.0

    return relevant_count / total_relevant


def reciprocal_rank(results, relevance, k):
    for rank, item in enumerate(results[:k], start=1):
        if relevance.get(item["name"], 0) > 0:
            return 1.0 / rank

    return 0.0


def dcg_at_k(results, relevance, k):
    score = 0.0

    for rank, item in enumerate(results[:k], start=1):
        grade = relevance.get(item["name"], 0)

        score += (2**grade - 1) / math.log2(rank + 1)

    return score


def ndcg_at_k(results, relevance, k):
    actual_dcg = dcg_at_k(results, relevance, k)

    ideal_results = sorted(
        relevance.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    ideal_dcg = 0.0

    for rank, (_, grade) in enumerate(ideal_results[:k], start=1):
        ideal_dcg += (2**grade - 1) / math.log2(rank + 1)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(
    vector_search,
    embedding_service,
    query,
    relevance,
    k_values=(5, 10),
):
    query_embedding = embedding_service.embed_query(query)

    results = vector_search.search(
        query_embedding=query_embedding,
        top_k=max(k_values),
    )

    evaluation = {
        "query": query,
        "results": results,
        "metrics": {},
    }

    for k in k_values:
        evaluation["metrics"][k] = {
            "precision": precision_at_k(
                results,
                relevance,
                k,
            ),
            "recall": recall_at_k(
                results,
                relevance,
                k,
            ),
            "mrr": reciprocal_rank(
                results,
                relevance,
                k,
            ),
            "ndcg": ndcg_at_k(
                results,
                relevance,
                k,
            ),
        }

    return evaluation


def evaluate_model(
    embedding_service,
    vector_search,
    ground_truth,
    k_values=(5, 10),
):
    """
    Evaluate an embedding/search model
    across all queries in the ground truth.
    """
    evaluations = []

    for query, relevance in ground_truth.items():
        evaluation = evaluate_query(
            query=query,
            embedding_service=embedding_service,
            vector_search=vector_search,
            relevance=relevance,
            k_values=k_values,
        )

        evaluations.append(evaluation)

    return evaluations


def summarize_evaluations(evaluations):
    k_values = evaluations[0]["metrics"].keys()

    summary = {}

    for k in k_values:
        summary[k] = {
            "precision": sum(
                evaluation["metrics"][k]["precision"]
                for evaluation in evaluations
            ) / len(evaluations),

            "recall": sum(
                evaluation["metrics"][k]["recall"]
                for evaluation in evaluations
            ) / len(evaluations),

            "mrr": sum(
                evaluation["metrics"][k]["mrr"]
                for evaluation in evaluations
            ) / len(evaluations),

            "ndcg": sum(
                evaluation["metrics"][k]["ndcg"]
                for evaluation in evaluations
            ) / len(evaluations),
        }

    return summary


def print_evaluation(evaluation):
    print("=" * 60)
    print(f"QUERY: {evaluation['query']}")
    print("=" * 60)

    for rank, item in enumerate(evaluation["results"], start=1):
        print(
            f"{rank:2}. "
            f"{item['similarity']:.4f} | "
            f"{item['name']}"
        )

    for k, metrics in evaluation["metrics"].items():
        print()
        print(f"Metrics @ {k}")
        print(f"Precision@{k}: {metrics['precision']:.2f}")
        print(f"Recall@{k}:    {metrics['recall']:.2f}")
        print(f"MRR@{k}:       {metrics['mrr']:.2f}")
        print(f"NDCG@{k}:      {metrics['ndcg']:.2f}")


def print_summary(summary):
    """
    Print average evaluation metrics for each K.
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for k, metrics in summary.items():
        print()
        print(f"@{k}")

        print(
            f"Average Precision@{k}: "
            f"{metrics['precision']:.2f}"
        )

        print(
            f"Average Recall@{k}:    "
            f"{metrics['recall']:.2f}"
        )

        print(
            f"Mean Reciprocal Rank@{k}: "
            f"{metrics['mrr']:.2f}"
        )

        print(
            f"Average NDCG@{k}:      "
            f"{metrics['ndcg']:.2f}"
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
    k_values = (5, 10)

    embedding_service, vector_search = build_vector_search()

    evaluations = evaluate_model(
        embedding_service=embedding_service,
        vector_search=vector_search,
        ground_truth=GROUND_TRUTH,
        k_values=k_values,
    )

    for evaluation in evaluations:
        print_evaluation(evaluation)

    summary = summarize_evaluations(evaluations)

    print_summary(summary)  


if __name__ == "__main__":
    main()