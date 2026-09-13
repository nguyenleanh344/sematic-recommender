"""
Hybrid filtering strategy that uses category as a soft constraint,
not a hard filter. This allows high-quality matches from other categories
while still prioritizing category-relevant results.
"""

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

# Query to category hints (soft, not hard filters)
QUERY_CATEGORY_HINTS = {
    "I want something comfortable and loose to wear every day": ["Clothing"],
    "I need something for running and working out": ["Sports", "Footwear"],
    "I need equipment for programming and working at my desk": ["Electronics", "Home"],
    "I am going on an outdoor trip and need useful gear": ["Outdoor", "Sports"],
    "I want something that makes cooking and daily household tasks easier": ["Home"],
}


def precision_at_k(results, relevant, k):
    """Calculate Precision@k metric."""
    top_k = results[:k]
    
    relevant_count = sum(
        1 for product in top_k
        if product["name"] in relevant
    )
    
    return relevant_count / k


def boost_category_relevant(
    results: list,
    preferred_categories: list,
    boost_score: float = 0.05,
) -> list:
    """
    Boost relevance score for products in preferred categories.
    This creates a soft preference without hard filtering.
    
    Args:
        results: Search results with similarity scores
        preferred_categories: List of preferred category names
        boost_score: How much to boost the score (0.05 = 5% boost)
        
    Returns:
        Re-ranked results based on boosted scores
    """
    boosted_results = []
    
    for result in results:
        boosted_result = result.copy()
        
        # If product is in preferred category, boost its similarity score
        if result.get("category") in preferred_categories:
            boosted_result["similarity"] *= (1 + boost_score)
            boosted_result["boosted"] = True
        else:
            boosted_result["boosted"] = False
        
        boosted_results.append(boosted_result)
    
    # Re-sort by boosted similarity
    boosted_results.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )
    
    return boosted_results


def evaluate_hybrid_approach(
    queries: list,
) -> dict:
    """Evaluate hybrid filtering with soft category boosting."""
    print(f"\n{'='*70}")
    print(f"HYBRID APPROACH: Soft Category Boosting (5% boost)")
    print(f"{'='*70}\n")
    
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_search = VectorSearchService(
        dimension=embedding_service.dimension
    )
    
    # Add all products
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
    
    precision_scores = []
    
    for query_idx, query in enumerate(queries, 1):
        print(f"Query {query_idx}: {query}")
        
        query_embedding = embedding_service.embed(query)
        
        # Get base results (without filtering)
        results = vector_search.search(
            query_embedding=query_embedding,
            top_k=5,
        )
        
        # Apply soft category boosting
        preferred_cats = QUERY_CATEGORY_HINTS.get(query, [])
        if preferred_cats:
            results = boost_category_relevant(results, preferred_cats, boost_score=0.05)
        
        # Print results
        for result in results:
            marker = "✓ BOOSTED" if result.get("boosted") else ""
            print(
                f"  {result['similarity']:.4f}"
                f" | {result['name']}"
                f" ({result['category']}) {marker}"
            )
        
        # Calculate Precision@5
        relevant = GROUND_TRUTH[query]
        score = precision_at_k(results, relevant, 5)
        precision_scores.append(score)
        
        print(f"  Precision@5: {score:.2f}\n")
    
    average_precision = sum(precision_scores) / len(precision_scores)
    
    return {
        "name": "Hybrid (Soft Boosting 5%)",
        "scores": precision_scores,
        "average": average_precision,
    }


def evaluate_aggressive_boosting(queries: list) -> dict:
    """Test with more aggressive boosting (10%)."""
    print(f"\n{'='*70}")
    print(f"HYBRID APPROACH: Aggressive Category Boosting (10% boost)")
    print(f"{'='*70}\n")
    
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_search = VectorSearchService(
        dimension=embedding_service.dimension
    )
    
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
    
    precision_scores = []
    
    for query_idx, query in enumerate(queries, 1):
        print(f"Query {query_idx}: {query}")
        
        query_embedding = embedding_service.embed(query)
        
        results = vector_search.search(
            query_embedding=query_embedding,
            top_k=5,
        )
        
        # Apply more aggressive boosting
        preferred_cats = QUERY_CATEGORY_HINTS.get(query, [])
        if preferred_cats:
            results = boost_category_relevant(results, preferred_cats, boost_score=0.10)
        
        for result in results:
            marker = "✓ BOOSTED" if result.get("boosted") else ""
            print(
                f"  {result['similarity']:.4f}"
                f" | {result['name']}"
                f" ({result['category']}) {marker}"
            )
        
        relevant = GROUND_TRUTH[query]
        score = precision_at_k(results, relevant, 5)
        precision_scores.append(score)
        
        print(f"  Precision@5: {score:.2f}\n")
    
    average_precision = sum(precision_scores) / len(precision_scores)
    
    return {
        "name": "Hybrid (Aggressive 10%)",
        "scores": precision_scores,
        "average": average_precision,
    }


if __name__ == "__main__":
    queries_to_evaluate = [
        "I want something comfortable and loose to wear every day",
        "I need something for running and working out",
        "I need equipment for programming and working at my desk",
        "I am going on an outdoor trip and need useful gear",
        "I want something that makes cooking and daily household tasks easier",
    ]
    
    all_results = []
    
    # Baseline (from previous test)
    baseline = {
        "name": "Baseline (all-MiniLM, no filter)",
        "average": 0.64,
        "scores": [0.20, 0.80, 0.80, 0.60, 0.80],
    }
    all_results.append(baseline)
    
    # Test soft boosting
    result1 = evaluate_hybrid_approach(queries_to_evaluate)
    all_results.append(result1)
    
    # Test aggressive boosting
    result2 = evaluate_aggressive_boosting(queries_to_evaluate)
    all_results.append(result2)
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70 + "\n")
    
    print(f"{'Strategy':<40} {'Avg P@5':<10}")
    print("-"*70)
    
    for result in all_results:
        print(f"{result['name']:<40} {result['average']:.4f}")
    
    print("\n")
    
    # Detailed comparison
    print("\nDetailed Scores:")
    print("-"*70)
    for result in all_results:
        print(f"\n{result['name']}")
        for idx, score in enumerate(result['scores'], 1):
            print(f"  Query {idx}: {score:.2f}")
        print(f"  Average: {result['average']:.4f}")
