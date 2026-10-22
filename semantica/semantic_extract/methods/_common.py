"""semantic_extract_methods: common (split from methods.py)."""

"""
Extraction Methods Module

This module provides all extraction methods as simple, reusable functions for
entities, relations, and triplets. It supports multiple extraction approaches
ranging from simple pattern matching to advanced LLM-based extraction.

Supported Methods:

Entity Extraction:
    - "pattern": Pattern-based entity extraction using regex
    - "regex": Advanced regex-based entity extraction with custom patterns
    - "rules": Rule-based entity extraction using linguistic rules
    - "ml": ML-based entity extraction using spaCy
    - "huggingface": HuggingFace NER model extraction
    - "llm": LLM-based entity extraction

Relation Extraction:
    - "pattern": Pattern-based relation extraction
    - "regex": Advanced regex-based relation extraction
    - "cooccurrence": Co-occurrence based relation detection
    - "similarity": Similarity-based relation extraction
    - "dependency": Dependency parsing-based relation extraction
    - "huggingface": HuggingFace relation extraction models
    - "llm": LLM-based relation extraction

Triplet Extraction:
    - "pattern": Pattern-based triplet extraction
    - "rules": Rule-based triplet extraction
    - "huggingface": HuggingFace triplet extraction models
    - "llm": LLM-based triplet extraction

Algorithms Used:

Entity Extraction:
    - Regular Expression Matching: Finite automata-based pattern matching
    - Rule-based NLP: Linguistic rule application and pattern matching
    - Neural NER: CNN/Transformer-based named entity recognition (spaCy)
    - Transformer Token Classification: BERT/RoBERTa for token-level classification
    - LLM Generation: Transformer-based language models for entity extraction

Relation Extraction:
    - Pattern Matching: Regex and string pattern matching algorithms
    - Co-occurrence Analysis: Proximity-based entity relationship detection
    - Dependency Parsing: Transition-based and graph-based parsing algorithms
    - Sequence Classification: Transformer-based relation classification
    - LLM Generation: Language model-based relation extraction

Triplet Extraction:
    - Pattern Matching: Subject-predicate-object pattern extraction
    - Rule-based Extraction: Linguistic rule application
    - Seq2Seq Models: Encoder-decoder transformer models for triplet generation
    - LLM Generation: Structured output generation from language models

Key Features:
    - Multiple extraction methods for entities:
        * Pattern-based: Simple regex pattern matching
        * Regex-based: Advanced regex with custom patterns
        * Rules-based: Linguistic rule-based extraction
        * ML-based: spaCy-based machine learning extraction
        * HuggingFace: Custom HuggingFace NER models
        * LLM-based: Large language model extraction
    - Multiple extraction methods for relations:
        * Pattern-based: Pattern matching for common relations
        * Regex-based: Advanced regex relation extraction
        * Co-occurrence: Proximity-based relation detection
        * Dependency: Dependency parsing-based extraction
        * HuggingFace: Custom HuggingFace relation models
        * LLM-based: LLM-powered relation extraction
    - Multiple extraction methods for triplets:
        * Pattern-based: Pattern matching for triplet extraction
        * Rules-based: Rule-based triplet extraction
        * HuggingFace: Custom HuggingFace triplet models
        * LLM-based: LLM-powered triplet extraction
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
    - extract_entities_pattern: Pattern-based entity extraction
    - extract_entities_regex: Regex-based entity extraction
    - extract_entities_rules: Rule-based entity extraction
    - extract_entities_ml: ML-based (spaCy) entity extraction
    - extract_entities_huggingface: HuggingFace model entity extraction
    - extract_entities_llm: LLM-based entity extraction
    - extract_relations_pattern: Pattern-based relation extraction
    - extract_relations_regex: Regex-based relation extraction
    - extract_relations_cooccurrence: Co-occurrence relation extraction
    - extract_relations_dependency: Dependency parsing relation extraction
    - extract_relations_huggingface: HuggingFace relation extraction
    - extract_relations_llm: LLM-based relation extraction
    - extract_triplets_pattern: Pattern-based triplet extraction
    - extract_triplets_ml: ML-based (spaCy dependency) triplet extraction
    - extract_triplets_rules: Rule-based triplet extraction
    - extract_triplets_huggingface: HuggingFace triplet extraction
    - extract_triplets_llm: LLM-based triplet extraction
    - get_entity_method: Get entity extraction method by name
    - get_relation_method: Get relation extraction method by name
    - get_triplet_method: Get triplet extraction method by name

Example Usage:
    >>> from semantica.semantic_extract.methods import get_entity_method
    >>> extract_fn = get_entity_method("llm")
    >>> entities = extract_fn("Apple Inc. was founded in 1976.", provider="openai")

Author: Semantica Contributors
License: MIT
"""

import hashlib
import os
import re
import difflib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from ...utils.exceptions import ProcessingError
from ...utils.logging import get_logger
from ..providers import HuggingFaceModelLoader, create_provider
from ..registry import method_registry
from ..cache import ExtractionCache, InMemoryBackend
from ..config import config
from ..types import (
    CONFIDENCE_SOURCE_KEY,
    CONFIDENCE_SOURCE_MODEL,
    CONFIDENCE_SOURCE_UNAVAILABLE,
    Entity,
    Relation,
    Triplet,
)

# Try to import spaCy
from ...utils.helpers import safe_import

import logging

from . import triplets


try:
    from ..schemas import (
        EntitiesResponse,
        RelationsResponse,
        RelationsWithTemporalResponse,
        TripletsResponse,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


logger = get_logger("methods")


# Initialize global result cache
def _default_cache_path() -> str:
    """A per-user, private default location for the persistent cache.

    Uses ``$XDG_CACHE_HOME`` (or ``~/.cache``) so the file is not a predictable,
    world-writable temp path — important because the sqlite backend can
    deserialize its rows with pickle. The directory is created ``0o700``; if that
    is not possible, a private ``tempfile`` directory is used as a fallback.
    """
    import tempfile

    base = triplets.os.environ.get("XDG_CACHE_HOME") or triplets.os.path.join(
        triplets.os.path.expanduser("~"), ".cache"
    )
    directory = triplets.os.path.join(base, "semantica")
    try:
        triplets.os.makedirs(directory, mode=0o700, exist_ok=True)
        if hasattr(triplets.os, "chmod"):
            triplets.os.chmod(directory, 0o700)
    except OSError:
        directory = tempfile.mkdtemp(prefix="semantica-cache-")  # 0o700, unique
    return triplets.os.path.join(directory, "extract_cache.sqlite3")


def _build_cache_backend():
    """Select a cache backend from config.

    Returns ``None`` (ExtractionCache's default in-memory backend) unless a
    persistent backend is requested. Any failure to construct the persistent
    backend degrades gracefully to the in-memory default so extraction never
    breaks because of a cache misconfiguration.
    """
    backend_kind = str(config.get("cache_backend", "memory") or "memory").lower()
    if backend_kind == "sqlite":
        from ..cache import SqliteCacheBackend

        path = config.get("cache_path") or _default_cache_path()
        try:
            return SqliteCacheBackend(path, max_size=config.get("cache_size", 1000))
        except Exception as exc:  # defensive fallback (e.g. unsafe/owned path)
            logger.warning(
                f"Failed to init sqlite cache backend at {path!r} "
                f"({exc}); falling back to in-memory cache."
            )
    return None


def _make_result_cache() -> ExtractionCache:
    cache = ExtractionCache(
        max_size=config.get("cache_size", 1000),
        ttl=config.get("cache_ttl", 3600),
        backend=_build_cache_backend(),
    )
    if not config.get("cache_enabled", True):
        cache.enabled = False
    return cache


_result_cache = _make_result_cache()


def configure_cache(
    backend: Optional[str] = None,
    path: Optional[str] = None,
    ttl: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> ExtractionCache:
    """Reconfigure the global extraction cache at runtime and rebuild it.

    The module-level cache is created at import time, so this is the supported
    way to switch to persistent (sqlite) storage programmatically after import
    (the ``SEMANTICA_CACHE_*`` environment variables work at import time too).

    Args:
        backend: ``"memory"`` or ``"sqlite"``.
        path: sqlite database path (defaults to a per-user private location).
        ttl: time-to-live in seconds.
        enabled: enable/disable caching.

    Returns:
        The rebuilt global :class:`ExtractionCache`.
    """
    updates: Dict[str, Any] = {}
    if backend is not None:
        updates["cache_backend"] = backend
    if path is not None:
        updates["cache_path"] = path
    if ttl is not None:
        updates["cache_ttl"] = ttl
    if enabled is not None:
        updates["enable_cache"] = enabled
    if updates:
        config.set_optimization(**updates)

    # Mutate the existing cache in place (rather than rebinding the module
    # global) so any code holding a reference to it stays valid.
    new_max = config.get("cache_size", 1000)
    new_backend = _build_cache_backend()
    if hasattr(_result_cache._backend, "close"):
        _result_cache._backend.close()
    _result_cache.max_size = new_max
    _result_cache.ttl = config.get("cache_ttl", 3600)
    _result_cache._backend = (
        new_backend if new_backend is not None else InMemoryBackend(max_size=new_max)
    )
    _result_cache.enabled = bool(config.get("cache_enabled", True))
    return _result_cache


# Generation kwargs that affect provider output and must therefore be part of
# the cache key. This is the union of every generation-affecting parameter
# read across providers.py, including params picked up outside _add_if_set
# (e.g. AnthropicProvider's manual pass-through loop). Sensitive values
# (api_key, token, etc.) are already filtered out by
# ExtractionCache._generate_key, so they need not be excluded here.
_GENERATION_CACHE_KEYS = frozenset({
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "stop_sequences",  # Anthropic/Gemini spelling of "stop"
    "logit_bias",
    "user",
    "system",  # Anthropic system prompt
    "metadata",  # Anthropic request metadata
    "candidate_count",  # Gemini
    "repeat_penalty",  # Ollama
    "num_ctx",  # Ollama
    "context_window",  # Ollama alias for num_ctx
})


def _generation_cache_params(kwargs: dict) -> dict:
    """Return the subset of *kwargs* that affects generation output.

    Only keys listed in ``_GENERATION_CACHE_KEYS`` are included so that
    irrelevant or sensitive caller kwargs do not pollute the cache key.
    Values that are ``None`` are omitted; a caller passing
    ``temperature=None`` is equivalent to not passing it at all.
    """
    return {
        k: v for k, v in kwargs.items()
        if k in _GENERATION_CACHE_KEYS and v is not None
    }


def _stable_fingerprint(items) -> str:
    """Deterministic SHA-256 fingerprint over a collection of strings.

    Used to fold entity/relation inputs into a cache key. Unlike the built-in
    ``hash()``, this is stable across interpreter processes (``hash()`` is
    randomized per-process via ``PYTHONHASHSEED``), which the persistent cache
    backend relies on so identical inputs map to the same key after a restart.
    """
    joined = "\x00".join(sorted(items))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


_logger = logging.getLogger(__name__)


spacy, SPACY_AVAILABLE = safe_import("spacy")
