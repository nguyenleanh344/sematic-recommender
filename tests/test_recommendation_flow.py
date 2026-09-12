from app.services.embedding_service import EmbeddingService
from app.services.vector_search_service import VectorSearchService


embedding_service = EmbeddingService()
vector_search = VectorSearchService(
    dimension=embedding_service.dimension
)

products = [
    {
        "id": 1,
        "name": "Black Oversized T-Shirt",
        "description": "Black cotton oversized shirt",
    },
    {
        "id": 2,
        "name": "Gray Hoodie",
        "description": "Loose gray hoodie for casual wear",
    },
    {
        "id": 3,
        "name": "Running Shoes",
        "description": "Lightweight running shoes",
    },
    {
        "id": 4,
        "name": "Gaming Laptop",
        "description": "High performance gaming laptop",
    },
    {
        "id": 5,
        "name": "Loose T-shirt",
        "description": "Loose T-Shirt for annual wear",
    },
]


for product in products:
    text = (
        f"{product['name']}. "
        f"{product['description']}"
    )

    embedding = embedding_service.embed(text)

    vector_search.add_product(
        product_id=product["id"],
        name=product["name"],
        embedding=embedding,
    )


query = "I want a loose shirt for casual wear"

query_embedding = embedding_service.embed(query)

results = vector_search.search(
    query_embedding=query_embedding,
    top_k=2,
)


for result in results:
    print(result)