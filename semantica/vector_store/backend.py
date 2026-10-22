"""The contract a vector store backend implements.

The facade in :mod:`semantica.vector_store.vector_store` talks to every backend
through these method names and signatures. Backends that predate the contract
expose different names (``add_vectors``, ``insert_vectors``, ``search_similar``
...), which the facade probes for at call time. New backends should subclass
:class:`VectorBackend` so a missing method fails at construction, not on the
first write.

Methods a backend cannot support default to raising ``NotImplementedError`` with
the backend name in the message, so callers see an explicit gap instead of an
attribute error.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np


class VectorBackend(ABC):
    """Canonical method set for a vector store backend."""

    @abstractmethod
    def add(
        self,
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Store vectors and return their ids (generated when ``ids`` is None)."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` results, each with ``id``, ``score`` and ``metadata``."""

    @abstractmethod
    def update(
        self,
        ids: List[str],
        vectors: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Replace the vector and/or metadata of existing ids; return the updated ids."""

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Remove the given ids. Return True when the backend accepted the delete."""

    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Return the stored vector for ``vector_id``, or None if it is absent."""

    @abstractmethod
    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored metadata for ``vector_id``, or None if it is absent."""

    def scan_vectors(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Page through stored vectors. Optional: not every backend can enumerate."""
        raise NotImplementedError(f"{type(self).__name__} does not support scan_vectors")

    def count(self) -> int:
        """Return the number of stored vectors. Optional."""
        raise NotImplementedError(f"{type(self).__name__} does not support count")

    def filter_by_metadata(self, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Return records whose metadata matches ``filters``. Optional."""
        raise NotImplementedError(f"{type(self).__name__} does not support filter_by_metadata")


# The canonical method names, in the order the contract lists them.
CONTRACT_METHODS = (
    "add",
    "search",
    "update",
    "delete",
    "get_vector",
    "get_metadata",
    "scan_vectors",
    "count",
    "filter_by_metadata",
)
