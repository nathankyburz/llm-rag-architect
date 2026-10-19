"""fastembed adapter for the Embedder port."""

from collections.abc import Sequence

from fastembed import TextEmbedding

from rag_architect.config import EmbeddingSettings


class FastEmbedEmbedder:
    """
    Embeds text with fastembed, which runs the model through ONNX rather than
    PyTorch: inference only, and an image measured in hundreds of megabytes
    instead of gigabytes.

    Documents and queries go through different methods because BGE models are
    trained asymmetrically: the query side takes an instruction prefix. fastembed
    does not apply it -- its query_embed and passage_embed both fall through to
    embed() for this model class -- so the prefix is applied here.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._model = TextEmbedding(
            model_name=settings.model_name,
            cache_dir=str(settings.cache_dir) if settings.cache_dir else None,
        )
        self._batch_size = settings.batch_size
        self._query_prefix = settings.query_prefix
        self._dimension = int(self._model.embedding_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.query_embed(self._query_prefix + text)))
        return list(vector.tolist())
