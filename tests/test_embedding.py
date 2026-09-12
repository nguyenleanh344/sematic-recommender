import numpy as np

from app.services.embedding_service import EmbeddingService


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


def test_semantic_similarity():
    service = EmbeddingService()

    vector_a = service.embed(
        "black oversized shirt"
    )

    vector_b = service.embed(
        "loose black t-shirt"
    )

    vector_c = service.embed(
        "gaming laptop"
    )

    similarity_ab = cosine_similarity(
        vector_a,
        vector_b,
    )

    similarity_ac = cosine_similarity(
        vector_a,
        vector_c,
    )

    print("shirt vs shirt:", similarity_ab)
    print("shirt vs laptop:", similarity_ac)

    assert similarity_ab > similarity_ac