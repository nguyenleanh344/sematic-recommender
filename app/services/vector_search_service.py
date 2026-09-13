import numpy as np


class VectorSearchService:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.products = []

    def add_product(
        self,
        product_id: int,
        name: str,
        embedding: list[float],
        metadata: dict = None,
    ):
        self._validate_dimension(embedding)

        product_data = {
            "id": product_id,
            "name": name,
            "embedding": np.array(embedding),
        }
        
        # Add optional metadata (category, subcategory, etc.)
        if metadata:
            product_data.update(metadata)

        self.products.append(product_data)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        category_filter: str = None,
        subcategory_filter: str = None,
    ):
        """
        Search for products similar to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            category_filter: Optional category to filter by
            subcategory_filter: Optional subcategory to filter by
            
        Returns:
            List of top_k similar products
        """
        self._validate_dimension(query_embedding)

        query_vector = np.array(query_embedding)

        results = []

        for product in self.products:
            # Apply filters if specified
            if category_filter and product.get("category") != category_filter:
                continue
            if subcategory_filter and product.get("subcategory") != subcategory_filter:
                continue
                
            similarity = self._cosine_similarity(
                query_vector,
                product["embedding"],
            )

            results.append({
                "id": product["id"],
                "name": product["name"],
                "similarity": float(similarity),
                "category": product.get("category"),
            })

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True,
        )

        return results[:top_k]

    def _validate_dimension(
        self,
        embedding: list[float],
    ):
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {len(embedding)}"
            )

    @staticmethod
    def _cosine_similarity(a, b):
        return np.dot(a, b) / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )