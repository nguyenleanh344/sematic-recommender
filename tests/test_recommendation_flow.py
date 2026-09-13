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
        "description": "Loose fit black cotton t-shirt for casual everyday wear",
    },
    {
        "id": 2,
        "name": "White Cotton T-Shirt",
        "description": "Classic white cotton t-shirt with a comfortable regular fit",
    },
    {
        "id": 3,
        "name": "Blue Denim Jeans",
        "description": "Classic blue denim jeans suitable for casual everyday outfits",
    },
    {
        "id": 4,
        "name": "Gray Hoodie",
        "description": "Warm gray hoodie with a loose fit for casual and outdoor wear",
    },
    {
        "id": 5,
        "name": "Formal Black Shirt",
        "description": "Slim fit black dress shirt for business meetings and formal occasions",
    },
    {
        "id": 6,
        "name": "Linen Summer Shirt",
        "description": "Lightweight breathable linen shirt designed for hot summer weather",
    },
    {
        "id": 7,
        "name": "Leather Jacket",
        "description": "Classic black leather jacket for stylish casual outfits and cool weather",
    },
    {
        "id": 8,
        "name": "Running Shoes",
        "description": "Lightweight breathable running shoes designed for jogging and daily exercise",
    },
    {
        "id": 9,
        "name": "White Sneakers",
        "description": "Minimalist white sneakers for casual everyday outfits and walking",
    },
    {
        "id": 10,
        "name": "Hiking Boots",
        "description": "Durable waterproof hiking boots designed for mountains and outdoor trails",
    },

    {
        "id": 11,
        "name": "Gaming Laptop",
        "description": "High performance laptop with dedicated graphics for gaming and demanding applications",
    },
    {
        "id": 12,
        "name": "Ultrabook Laptop",
        "description": "Thin lightweight laptop designed for office work, studying, and travel",
    },
    {
        "id": 13,
        "name": "Mechanical Keyboard",
        "description": "Mechanical keyboard with tactile switches designed for gaming and programming",
    },
    {
        "id": 14,
        "name": "Wireless Mouse",
        "description": "Ergonomic wireless mouse suitable for office work, productivity, and gaming",
    },
    {
        "id": 15,
        "name": "Noise Cancelling Headphones",
        "description": "Wireless over-ear headphones with active noise cancellation for music and travel",
    },
    {
        "id": 16,
        "name": "Bluetooth Earbuds",
        "description": "Compact wireless earbuds with a charging case for music and calls",
    },
    {
        "id": 17,
        "name": "4K Monitor",
        "description": "High resolution 4K monitor suitable for programming, office work, and creative tasks",
    },
    {
        "id": 18,
        "name": "Portable SSD",
        "description": "Fast compact external SSD for storing and transferring large files",
    },

    {
        "id": 19,
        "name": "Stainless Steel Water Bottle",
        "description": "Reusable insulated water bottle that keeps drinks cold or hot for hours",
    },
    {
        "id": 20,
        "name": "Travel Backpack",
        "description": "Spacious lightweight backpack designed for travel, work, and carrying a laptop",
    },
    {
        "id": 21,
        "name": "Running Shorts",
        "description": "Lightweight breathable athletic shorts designed for running and exercise",
    },
    {
        "id": 22,
        "name": "Yoga Mat",
        "description": "Non-slip exercise mat designed for yoga, stretching, and home workouts",
    },
    {
        "id": 23,
        "name": "Dumbbell Set",
        "description": "Adjustable dumbbell set for strength training and home gym workouts",
    },
    {
        "id": 24,
        "name": "Camping Tent",
        "description": "Lightweight waterproof tent designed for camping and outdoor adventures",
    },

    {
        "id": 25,
        "name": "Coffee Maker",
        "description": "Automatic coffee machine for brewing fresh coffee at home",
    },
    {
        "id": 26,
        "name": "Electric Kettle",
        "description": "Fast boiling electric kettle for preparing tea, coffee, and hot water",
    },
    {
        "id": 27,
        "name": "Air Fryer",
        "description": "Compact kitchen appliance for cooking crispy food with little or no oil",
    },
    {
        "id": 28,
        "name": "Robot Vacuum",
        "description": "Smart robotic vacuum cleaner that automatically cleans floors around the home",
    },
    {
        "id": 29,
        "name": "Desk Lamp",
        "description": "Adjustable LED desk lamp for studying, reading, and working at night",
    },
    {
        "id": 30,
        "name": "Office Chair",
        "description": "Ergonomic adjustable chair designed for comfortable long hours of desk work",
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


queries = [
    "I want something comfortable and loose to wear every day",
    "I need something for running and working out",
    "I need equipment for programming and working at my desk",
    "I am going on an outdoor trip and need useful gear",
    "I want something that makes cooking and daily household tasks easier",
    "I don't want anything"
]


for query in queries:
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    query_embedding = embedding_service.embed(query)

    results = vector_search.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    for result in results:
        print(
            f"{result['similarity']:.4f}"
            f" | {result['name']}"
        )