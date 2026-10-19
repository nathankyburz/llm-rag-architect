"""
The adapter is a thin wrapper, so these tests cover the part we wrote -- the
query prefix fastembed does not apply -- with a recording double, rather than
downloading a 130 MB model to re-test fastembed.
"""

import numpy as np
import pytest

from rag_architect.adapters import embedding as embedding_module
from rag_architect.adapters.embedding import FastEmbedEmbedder
from rag_architect.config import EmbeddingSettings


class RecordingModel:
    embedding_size = 384

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.queries: list[str] = []
        self.passages: list[list[str]] = []

    def query_embed(self, query, **kwargs):
        self.queries.append(query)
        return iter([np.zeros(384, dtype=np.float32)])

    def passage_embed(self, texts, **kwargs):
        self.passages.append(list(texts))
        return iter([np.zeros(384, dtype=np.float32) for _ in texts])


@pytest.fixture
def model(monkeypatch) -> RecordingModel:
    created: list[RecordingModel] = []

    def factory(**kwargs):
        created.append(RecordingModel(**kwargs))
        return created[-1]

    monkeypatch.setattr(embedding_module, "TextEmbedding", factory)
    yield lambda: created[-1]


def test_dimension_is_read_from_the_model(model):
    """Never configured: a dimension that can contradict the model is a bug."""
    assert FastEmbedEmbedder(EmbeddingSettings()).dimension == 384


def test_query_is_prefixed(model):
    embedder = FastEmbedEmbedder(EmbeddingSettings())

    embedder.embed_query("What was Brighton's turnover in 2024?")

    assert model().queries == [
        "Represent this sentence for searching relevant passages: "
        "What was Brighton's turnover in 2024?"
    ]


def test_passages_are_not_prefixed(model):
    """BGE is asymmetric: the instruction belongs on the query side only."""
    embedder = FastEmbedEmbedder(EmbeddingSettings())

    embedder.embed_documents(["Turnover 221,293 203,574"])

    assert model().passages == [["Turnover 221,293 203,574"]]


def test_prefix_is_configurable(model):
    embedder = FastEmbedEmbedder(EmbeddingSettings(query_prefix="query: "))

    embedder.embed_query("turnover")

    assert model().queries == ["query: turnover"]


def test_vectors_come_back_as_plain_lists(model):
    """The port returns lists, not numpy arrays: the domain stays dependency-free."""
    embedder = FastEmbedEmbedder(EmbeddingSettings())

    vectors = embedder.embed_documents(["a", "b"])
    query = embedder.embed_query("a")

    assert isinstance(vectors, list) and isinstance(vectors[0], list)
    assert isinstance(query, list) and len(query) == 384
