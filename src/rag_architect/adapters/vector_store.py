"""Qdrant adapter for the VectorStore port."""

from collections.abc import Sequence
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http.models import models as qm

from rag_architect.config import VectorStoreSettings
from rag_architect.domain.documents import Chunk, DocumentType
from rag_architect.domain.retrieval import SearchResult

# Payload keys given an index so they can be filtered on cheaply. Without an
# index Qdrant still filters, by scanning every point.
INDEXED_FIELDS: dict[str, qm.PayloadSchemaType] = {
    "metadata.document_type": qm.PayloadSchemaType.KEYWORD,
    "metadata.document_id": qm.PayloadSchemaType.KEYWORD,
    "metadata.club": qm.PayloadSchemaType.KEYWORD,
    "metadata.fiscal_year": qm.PayloadSchemaType.INTEGER,
    "content_hash": qm.PayloadSchemaType.KEYWORD,
}


class QdrantVectorStore:
    """
    Stores chunk vectors in Qdrant and searches them by cosine similarity.

    The payload is the chunk minus its id, so a search result reconstructs a
    full domain `Chunk` without a second lookup, and gains any field the model
    gains later.
    """

    def __init__(self, settings: VectorStoreSettings, client: QdrantClient | None = None) -> None:
        self._collection = settings.collection_name
        self._client = client or QdrantClient(
            url=settings.url,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            timeout=int(settings.timeout_s),
        )

    def ensure_collection(self, dimension: int) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
            )
        for field, schema in INDEXED_FIELDS.items():
            self._client.create_payload_index(
                collection_name=self._collection, field_name=field, field_schema=schema
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            msg = f"got {len(chunks)} chunks for {len(vectors)} vectors"
            raise ValueError(msg)
        if not chunks:
            return 0

        points = [
            qm.PointStruct(
                id=str(chunk.id),
                vector=list(vector),
                payload=chunk.model_dump(mode="json", exclude={"id"}),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        document_type: DocumentType | None = None,
    ) -> list[SearchResult]:
        query_filter = None
        if document_type is not None:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="metadata.document_type",
                        match=qm.MatchValue(value=document_type.value),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk=Chunk(id=UUID(str(point.id)), **(point.payload or {})),
                score=point.score,
            )
            for point in response.points
        ]

    def count(self) -> int:
        """Points currently stored. Used by the ingestion CLI to report progress."""
        return int(self._client.count(collection_name=self._collection, exact=True).count)
