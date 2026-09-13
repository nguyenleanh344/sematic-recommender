"""
Multi-model embedding evaluation and comparison.
Tests different embedding models and filtering strategies to improve recommendation accuracy.
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

# Query to category mapping for filtering
QUERY_CATEGORY_HINT = {
    "I want something comfortable and loose to wear every day": "Clothing",
    "I need something for running and working out": ["Sports", "Footwear"],
    "I need equipment for programming and working at my desk": "Electronics",
    "I am going on an outdoor trip and need useful gear": ["Outdoor", "Sports"],
    "I want something that makes cooking and daily household tasks easier": "Home",
}


def precision_at_k(results, relevant, k):
    """Calculate Precision@k metric."""
    top_k = results[:k]
    
    relevant_count = sum(
        1 for product in top_k
        if product["name"] in relevant
    )
    
    return relevant_count / k


def evaluate_model(
    model_name: str,
    queries: list,
    use_category_filter: bool = False,
) -> dict:
    """
    Evaluate a specific embedding model.
    
    Args:
        model_name: Embedding model name
        queries: List of query strings to evaluate
        use_category_filter: Whether to apply category filtering
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"Category Filter: {'ON' if use_category_filter else 'OFF'}")
    print(f"{'='*70}\n")
    
    embedding_service = EmbeddingService(model_name=model_name)
    vector_search = VectorSearchService(
        dimension=embedding_service.dimension
    )
    
    # Add all products with metadata
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
    results_per_query = []
    
    for query_idx, query in enumerate(queries, 1):
        print(f"Query {query_idx}: {query}")
        
        query_embedding = embedding_service.embed(query)
        
        # Determine category filter
        category_filter = None
        if use_category_filter and query in QUERY_CATEGORY_HINT:
            hint = QUERY_CATEGORY_HINT[query]
            # If hint is a list, use first one; otherwise use as-is
            category_filter = hint[0] if isinstance(hint, list) else hint
        
        results = vector_search.search(
            query_embedding=query_embedding,
            top_k=5,
            category_filter=category_filter,
        )
        
        # Print results
        for result in results:
            print(
                f"  {result['similarity']:.4f}"
                f" | {result['name']}"
                f" ({result['category']})"
            )
        
        # Calculate Precision@5
        relevant = GROUND_TRUTH[query]
        score = precision_at_k(results, relevant, 5)
        precision_scores.append(score)
        
        results_per_query.append({
            "query": query,
            "results": results,
            "precision": score,
        })
        
        print(f"  Precision@5: {score:.2f}\n")
    
    average_precision = sum(precision_scores) / len(precision_scores)
    
    return {
        "model_name": model_name,
        "use_filter": use_category_filter,
        "scores": precision_scores,
        "average": average_precision,
        "details": results_per_query,
    }


def print_comparison_table(results: list):
    """Print comparison table of all evaluation results."""
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70 + "\n")
    
    print(f"{'Model':<35} {'Filter':<8} {'Avg P@5':<10}")
    print("-"*70)
    
    for result in results:
        model_display = result["model_name"][:30]
        filter_display = "ON" if result["use_filter"] else "OFF"
        avg = result["average"]
        
        print(f"{model_display:<35} {filter_display:<8} {avg:.4f}")
    
    print("\n")


def print_detailed_scores(results: list):
    """Print detailed per-query scores for all models."""
    print("\n" + "="*70)
    print("DETAILED SCORES (Per Query)")
    print("="*70 + "\n")
    
    for result in results:
        print(f"\n{result['model_name']} (Filter: {result['use_filter']})")
        print("-"*50)
        
        for idx, score in enumerate(result['scores'], 1):
            print(f"  Query {idx}: {score:.2f}")
        
        print(f"  Average: {result['average']:.4f}")


# Main evaluation
if __name__ == "__main__":
    queries_to_evaluate = [
        "I want something comfortable and loose to wear every day",
        "I need something for running and working out",
        "I need equipment for programming and working at my desk",
        "I am going on an outdoor trip and need useful gear",
        "I want something that makes cooking and daily household tasks easier",
    ]
    
    all_results = []
    
    # Test 1: Baseline (all-MiniLM-L6-v2 without filter)
    result1 = evaluate_model(
        model_name="all-MiniLM-L6-v2",
        queries=queries_to_evaluate,
        use_category_filter=False,
    )
    all_results.append(result1)
    
    # Test 2: all-MiniLM-L6-v2 with category filtering
    result2 = evaluate_model(
        model_name="all-MiniLM-L6-v2",
        queries=queries_to_evaluate,
        use_category_filter=True,
    )
    all_results.append(result2)
    
    # Test 3: Better model without filter (if you want to test more models)
    # Uncomment to test additional models
    # Models to try:
    # - "paraphrase-MiniLM-L6-v2" - optimized for semantic search
    # - "all-mpnet-base-v2" - larger, more accurate
    # - "sentence-transformers/all-roberta-large-v1" - even more accurate
    
    # For now, we'll test paraphrase-MiniLM which is optimized for similarity
    try:
        result3 = evaluate_model(
            model_name="paraphrase-MiniLM-L6-v2",
            queries=queries_to_evaluate,
            use_category_filter=False,
        )
        all_results.append(result3)
        
        result4 = evaluate_model(
            model_name="paraphrase-MiniLM-L6-v2",
            queries=queries_to_evaluate,
            use_category_filter=True,
        )
        all_results.append(result4)
    except Exception as e:
        print(f"\nWarning: Could not load paraphrase-MiniLM-L6-v2: {e}")
        print("Skipping this model...\n")
    
    # Print summaries
    print_comparison_table(all_results)
    print_detailed_scores(all_results)
    
    # Find best configuration
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70 + "\n")
    
    best_result = max(all_results, key=lambda x: x["average"])
    print(f"Model: {best_result['model_name']}")
    print(f"Category Filter: {'ON' if best_result['use_filter'] else 'OFF'}")
    print(f"Average Precision@5: {best_result['average']:.4f}")
    print(f"Improvement: +{(best_result['average'] - all_results[0]['average'])*100:.1f}%")
