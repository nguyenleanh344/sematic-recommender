from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService
from app.data.products_with_metadata import PRODUCTS_WITH_METADATA


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
    Calculate Precision@k metric.
    
    Args:
        results: List of result dictionaries (each has 'name' key)
        relevant: Set of relevant product names
        k: The k value for Precision@k
        
    Returns:
        Float between 0 and 1
    """
    top_k = results[:k]
    
    relevant_count = sum(
        1 for product in top_k
        if product["name"] in relevant
    )
    
    return relevant_count / k


# ============================================================================
# Setup: Initialize services and product embeddings (same as test_recommendation_flow.py)
# ============================================================================

embedding_service = EmbeddingService()
vector_search = VectorSearchService(
    dimension=embedding_service.dimension
)

# ============================================================================
# Setup: Initialize services and product embeddings
# ============================================================================

embedding_service = EmbeddingService()
vector_search = VectorSearchService(
    dimension=embedding_service.dimension
)

# Embed all products with metadata and add to vector search
for product in PRODUCTS_WITH_METADATA:
    text = (
        f"{product['name']}. "
        f"{product['description']}"
    )
    
    embedding = embedding_service.embed(text)
    
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


# ============================================================================
# Evaluation: Run recommendation flow and calculate Precision@5
# ============================================================================

queries_to_evaluate = [
    "I want something comfortable and loose to wear every day",
    "I need something for running and working out",
    "I need equipment for programming and working at my desk",
    "I am going on an outdoor trip and need useful gear",
    "I want something that makes cooking and daily household tasks easier",
]

precision_scores = []

for query_index, query in enumerate(queries_to_evaluate, 1):
    print("\n" + "=" * 60)
    print(f"QUERY {query_index}: {query}")
    print("=" * 60)
    
    # Get query embedding
    query_embedding = embedding_service.embed(query)
    
    # Search for top 5 products
    results = vector_search.search(
        query_embedding=query_embedding,
        top_k=5,
    )
    
    # Print results with similarity scores
    for result in results:
        print(
            f"{result['similarity']:.4f}"
            f" | {result['name']}"
        )
    
    # Calculate Precision@5
    relevant = GROUND_TRUTH[query]
    score = precision_at_k(results, relevant, 5)
    precision_scores.append(score)
    
    print(f"\nPrecision@5: {score:.2f}")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for query_index, score in enumerate(precision_scores, 1):
    print(f"Query {query_index}: {score:.2f}")

average_precision = sum(precision_scores) / len(precision_scores)
print(f"\nAverage Precision@5: {average_precision:.2f}")