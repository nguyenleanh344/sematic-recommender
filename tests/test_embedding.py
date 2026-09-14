from app.services.embedding_service import EmbeddingService


def test_minilm_query_and_document_dimension():
    service = EmbeddingService(
        model_name="all-MiniLM-L6-v2"
    )

    query_embedding = service.embed_query(
        "I need something for running"
    )

    document_embedding = service.embed_document(
        "Running Shoes. Lightweight shoes for running."
    )

    assert len(query_embedding) == service.dimension
    assert len(document_embedding) == service.dimension
    
def test_e5_query_and_document_dimension():
    service = EmbeddingService(
        model_name="intfloat/multilingual-e5-small"
    )

    query_embedding = service.embed_query(
        "I need something for running"
    )

    document_embedding = service.embed_document(
        "Running Shoes. Lightweight shoes for running."
    )

    assert len(query_embedding) == service.dimension
    assert len(document_embedding) == service.dimension