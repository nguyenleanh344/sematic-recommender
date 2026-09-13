# Recommendation System Improvement Analysis

## Executive Summary

Tested 3 strategies to improve the precision of the recommendation system:

1. **Baseline**: all-MiniLM-L6-v2 without any filtering → **Avg P@5: 0.64** ✅
2. **Hard Category Filtering**: Strict category match → **Avg P@5: 0.60** ❌ (worse)
3. **Soft Boosting** (5% & 10%): Category preference without strict filtering → **Avg P@5: 0.64** (same as baseline)

## Key Findings

### 1. Baseline is Optimal ✅
The original `all-MiniLM-L6-v2` model without any filtering **achieves 0.64 Precision@5**, which is the best result.

```
Per-Query Scores:
- Query 1 (comfortable clothing): 0.20 (problematic)
- Query 2 (running/workout):     0.80 ✅
- Query 3 (desk equipment):       0.80 ✅
- Query 4 (outdoor trip):         0.60
- Query 5 (cooking/household):    0.80 ✅
```

### 2. Hard Filtering Makes Things Worse ❌
Strict category filtering **reduces accuracy to 0.60** because:

```
Example - Query 2: "I need something for running and working out"
- Baseline results:
  1. Running Shoes (Footwear) ✅ relevant
  2. Running Shorts (Sports) ✅ relevant
  3. Dumbbell Set (Sports) ✅ relevant
  4. Yoga Mat (Sports) ✅ relevant
  5. White Sneakers (Footwear) ❌ not relevant
  → Precision@5 = 0.80

- With Sports category filter:
  1. Running Shorts (Sports) ✅
  2. Dumbbell Set (Sports) ✅
  3. Yoga Mat (Sports) ✅
  [Only 3 items in Sports category, need 5]
  → Precision@5 = 0.60 (forced to fill with non-relevant items)
```

**Lesson**: Semantic similarity ≠ category match. Forcing category constraints loses good matches.

### 3. Soft Boosting Doesn't Improve ✅
Category boosting (5% and 10%) maintains the same 0.64 score because the baseline ranking is already optimal.
- The top-5 results don't change with soft boosting
- The model already ranks category-relevant items near the top naturally

## The Real Problem: Query 1

```
Query: "I want something comfortable and loose to wear every day"
Expected: Black Oversized T-Shirt, White Cotton T-Shirt, Gray Hoodie
Got:      Black Oversized T-Shirt ✅, Blue Denim Jeans, Formal Black Shirt, ...
Precision: 0.20 ❌
```

Why this fails:
- Model understands "comfortable" and "daily" but misses "loose"
- Formal Black Shirt: comfortable ✓ but NOT loose ✗
- Model sees semantic similarity (clothing, comfort) but misses the important constraint

## What Would Actually Help

### Option 1: Better Models 📚
Try larger, more powerful models:
```python
"paraphrase-MiniLM-L6-v2"     # ❌ Tested, no improvement
"all-mpnet-base-v2"           # Larger (109M params)
"all-roberta-large-v1"        # Even larger (355M params)
"bge-large-en-v1.5"          # Specialized for retrieval
```

### Option 2: Re-ranking with Business Rules 🔄
```
1. Search with embedding model      → [all products ranked by similarity]
2. Apply business rules in order:   → Filter/rank by constraints
   - Must have category match
   - Must have key attribute (loose, lightweight, etc.)
   - Boost by popularity/rating
3. Return top-5                     → [re-ranked results]
```

### Option 3: Hybrid Search 🔄
```
Combine:
- Semantic search (embeddings)
- Keyword search (exact word match for important terms)
- Metadata filtering (category, attributes)

Then fuse results with score aggregation
```

### Option 4: Fine-tune Model 🎓
Fine-tune the embedding model on your own query-product pairs with relevance labels:
```
Query: "comfortable and loose daily wear"
Relevant: [Black Oversized T-Shirt, White Cotton T-Shirt, Gray Hoodie]
Not relevant: [Formal Black Shirt, Blue Denim Jeans]
```

## Recommendations

### Short-term (Quick Win):
1. ✅ **Use baseline** (all-MiniLM-L6-v2, no filtering) - maintains 0.64 P@5
2. Add metadata (category) for display and future re-ranking

### Medium-term (Better Results):
1. Test larger embedding models (all-mpnet-base-v2)
2. Implement re-ranking with business rules
3. Add keyword matching for critical terms

### Long-term (Best Results):
1. Collect user feedback (click-through, conversions)
2. Fine-tune model on your domain data
3. Build hybrid search combining embeddings + keywords + metadata

## Implementation Status

✅ **Completed**:
- [x] Added metadata (category, subcategory) to products
- [x] Extended VectorSearchService with metadata support
- [x] Tested baseline: 0.64 P@5
- [x] Tested hard filtering: 0.60 P@5 (worse)
- [x] Tested soft boosting: 0.64 P@5 (same)
- [x] Tested alternative model (paraphrase-MiniLM): 0.60 P@5 (worse)

🚀 **Next Steps**:
- [ ] Test larger models (all-mpnet-base-v2, etc.)
- [ ] Implement re-ranking layer with business rules
- [ ] Add re-ranking to evaluation script
- [ ] Create API endpoint with best configuration

## Code Usage

```python
# Using the best configuration
from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService

embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
vector_search = VectorSearchService(dimension=embedding_service.dimension)

# Add products with metadata
for product in products:
    embedding = embedding_service.embed(product['description'])
    vector_search.add_product(
        product_id=product['id'],
        name=product['name'],
        embedding=embedding,
        metadata={"category": product["category"]},
    )

# Search (no filtering needed - baseline is best)
query_embedding = embedding_service.embed(query)
results = vector_search.search(query_embedding=query_embedding, top_k=5)
```

## Files Created

1. `app/data/products_with_metadata.py` - Products with category metadata
2. `app/services/vector_search_service.py` - Updated with metadata support
3. `tests/test_model_comparison.py` - Compare models and filtering strategies
4. `tests/test_hybrid_filtering.py` - Soft boosting approach
5. `tests/test_embedding_evaluation.py` - Updated with metadata

## Conclusion

**The baseline configuration (all-MiniLM-L6-v2 without filtering) is already optimal for this dataset.**

The main challenge is Query 1 (Precision: 0.20), which requires either:
- Larger, more powerful models
- Re-ranking with business rules
- Hybrid search combining embeddings + keywords
- Fine-tuning on domain data

Avoid hard category filtering as it removes good semantic matches.
