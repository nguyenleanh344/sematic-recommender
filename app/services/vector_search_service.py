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
    ):
        self._validate_dimension(embedding)

        self.products.append({
            "id": product_id,
            "name": name,
            "embedding": np.array(embedding),
        })

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ):
        self._validate_dimension(query_embedding)

        query_vector = np.array(query_embedding)

        results = []

        for product in self.products:
            similarity = self._cosine_similarity(
                query_vector,
                product["embedding"],
            )

            results.append({
                "id": product["id"],
                "name": product["name"],
                "similarity": float(similarity),
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