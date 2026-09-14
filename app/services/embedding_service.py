from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """
        Backward-compatible alias.

        Treat the input as a document.
        """
        return self.embed_document(text)

    def embed_query(self, text: str) -> list[float]:
        """
        Convert a user query into an embedding.
        """
        text = self._prepare_query(text)

        vector = self.model.encode(text)

        return vector.tolist()

    def embed_document(self, text: str) -> list[float]:
        """
        Convert a document/product into an embedding.
        """
        text = self._prepare_document(text)

        vector = self.model.encode(text)

        return vector.tolist()

    def _prepare_query(self, text: str) -> str:
        """
        Prepare query text before embedding.

        MiniLM:
            return text unchanged

        E5:
            should use "query: ..." prefix
        """
        if "e5" in self.model_name.lower():
            return f"query: {text}"

        return text

    def _prepare_document(self, text: str) -> str:
        """
        Prepare document text before embedding.

        MiniLM:
            return text unchanged

        E5:
            should use "passage: ..." prefix
        """
        if "e5" in self.model_name.lower():
            return f"passage: {text}"

        return text

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()