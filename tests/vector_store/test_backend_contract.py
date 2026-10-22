"""Each vector backend's method set against the canonical contract.

The facade probes backends for whichever method name they happen to expose.
These tests record which backends implement the canonical methods directly.
Known gaps are strict xfails, so fixing one makes the test XPASS, which fails
the suite until the marker is removed.
"""

import importlib

import pytest

from semantica.vector_store.backend import CONTRACT_METHODS, VectorBackend

BACKENDS = {
    "pgvector": ("semantica.vector_store.pgvector_store", "PgVectorStore"),
    "sqlite": ("semantica.vector_store.sqlite_vec_store", "SQLiteVecStore"),
    "faiss": ("semantica.vector_store.faiss_store", "FAISSStore"),
    "qdrant": ("semantica.vector_store.qdrant_store", "QdrantStore"),
    "pinecone": ("semantica.vector_store.pinecone_store", "PineconeStore"),
    "weaviate": ("semantica.vector_store.weaviate_store", "WeaviateStore"),
    "milvus": ("semantica.vector_store.milvus_store", "MilvusStore"),
}

# Methods each backend still lacks. Its native method (add_vectors, search_similar,
# delete_vectors, ...) is what the facade calls today through name probing.
_MISSING = {
    "faiss": {"add", "search", "update", "delete"},
    "qdrant": {"add", "search", "update", "delete", "scan_vectors", "count"},
    "pinecone": {"add", "search", "update", "delete", "scan_vectors", "count"},
    "weaviate": {"add", "search", "update", "delete", "scan_vectors", "count"},
    "milvus": {"add", "search", "update", "delete", "scan_vectors", "count"},
}


def _cls(backend):
    module_name, class_name = BACKENDS[backend]
    return getattr(importlib.import_module(module_name), class_name)


def _param(backend, method):
    marks = []
    if method in _MISSING.get(backend, set()):
        marks.append(pytest.mark.xfail(
            strict=True,
            reason=f"{backend} has no canonical {method}(); the facade reaches its native method by name",
        ))
    return pytest.param(backend, method, marks=marks, id=f"{backend}.{method}")


@pytest.mark.parametrize("backend, method", [_param(b, m) for b in BACKENDS for m in CONTRACT_METHODS])
def test_backend_implements_contract_method(backend, method):
    assert hasattr(_cls(backend), method), f"{backend} lacks {method}()"


@pytest.mark.parametrize("backend", ["pgvector", "sqlite"])
def test_conformant_backends_subclass_the_contract(backend):
    cls = _cls(backend)

    assert issubclass(cls, VectorBackend)
    assert not cls.__abstractmethods__, f"{backend} leaves {sorted(cls.__abstractmethods__)} abstract"


def test_contract_base_has_no_concrete_required_methods():
    """The required methods stay abstract; only the optional ones have defaults."""
    required = {"add", "search", "update", "delete", "get_vector", "get_metadata"}
    assert required == set(VectorBackend.__abstractmethods__)
