"""semantic_extract_methods: scoring (split from methods.py)."""

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

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

from ..types import (
    Entity,
)

# Try to import spaCy


from . import _common
from . import triplets


# ============================================================================
# Scoring Helper Functions
# ============================================================================

# Sentinel stored in _nlp_cache / _embedder_cache after a failed load attempt.
# Waiting threads check for this value and immediately return None rather than
# retrying the expensive (and already-failed) construction on every call.
# Using a dedicated object instead of a string avoids any accidental truthiness
# collision with a real model object.
_LOAD_FAILED = object()


# Global cache for spacy model and text embedder to avoid reloading.
# Values are either None (not yet attempted), _LOAD_FAILED (attempted and
# failed — callers get None without retrying), or the live model object.
_nlp_cache = None


_embedder_cache = None


# Guards for _nlp_cache and _embedder_cache lazy init and pipeline use.
# _nlp_cache_lock  — serializes the check + spacy.load() sequence so only one
#                    thread ever constructs the model.
# _nlp_call_lock   — serializes nlp(text) calls on the shared Language object;
#                    spaCy Language instances are not safe to call concurrently.
# _embedder_cache_lock — serializes the check + TextEmbedder() construction so
#                        only one instance is ever built.
_nlp_cache_lock = threading.Lock()


_nlp_call_lock = threading.Lock()


_embedder_cache_lock = threading.Lock()


# Cache for models loaded by name, so extraction functions do not pay
# spacy.load() on every call. Entries record the spacy module they were loaded
# from: tests patch `methods.spacy` with a mock, and an entry produced by a
# different module object must not be handed back to a later caller.
# Bounded: each spaCy Language costs hundreds of MB, and a long-running
# service that varies the `model` option would otherwise pin every model it
# ever touched for the life of the process. 4 covers lg/md/sm + one custom.
MAX_SPACY_MODELS_CACHED = 4


_spacy_model_cache: "OrderedDict[str, Tuple[Any, Any, threading.Lock]]" = OrderedDict()


_spacy_model_cache_lock = threading.Lock()


def _load_spacy_entry(name: str) -> Tuple[Any, Any, threading.Lock]:
    """Return the cache entry for ``name``: ``(spacy_module, nlp, call_lock)``.

    The entire lookup-plus-update is performed under ``_spacy_model_cache_lock``
    so that the LRU ``move_to_end`` and eviction are always consistent.
    ``spacy.load`` is called inside the lock; that is acceptable here because
    model loads are rare (at most ``MAX_SPACY_MODELS_CACHED`` per process) and
    the simpler design eliminates the TOCTOU window that a lockless fast-path
    would introduce.

    Raises whatever ``spacy.load`` raises (``OSError`` for a missing model).
    """
    if _common.spacy is None:
        raise ImportError(
            "spaCy is not installed. Install with: pip install 'semantica[nlp-spacy]'"
        )
    with _spacy_model_cache_lock:
        cached = _spacy_model_cache.get(name)
        if cached is not None and cached[0] is _common.spacy:
            _spacy_model_cache.move_to_end(name)
            return cached
        nlp = _common.spacy.load(name)
        entry: _common.Tuple[_common.Any, _common.Any, _common.threading.Lock] = (_common.spacy, nlp, _common.threading.Lock())
        _spacy_model_cache[name] = entry
        while len(_spacy_model_cache) > MAX_SPACY_MODELS_CACHED:
            _spacy_model_cache.popitem(last=False)
        return entry


def load_spacy_model(name: str):
    """Load a spaCy model once per process, keyed by model name.

    Raises whatever ``spacy.load`` raises (``OSError`` for a missing model), so
    callers keep their existing fallback behavior.

    .. warning::
        The returned ``Language`` object is shared process-wide.  Calling it
        from multiple threads concurrently is not safe.  Use
        ``spacy_pipeline_guard`` (or ``run_spacy_text``) to serialize calls.
    """
    _spacy_module, nlp, _call_lock = _load_spacy_entry(name)
    return nlp


@contextmanager
def spacy_pipeline_guard(name: str):
    """Yield the cached ``Language`` for ``name`` under its per-model call lock.

    Concurrent calls on the *same* model serialize; calls on *different* models
    run in parallel.  Raises whatever ``spacy.load`` raises (``OSError`` for a
    missing model).
    """
    _spacy_module, nlp, call_lock = _load_spacy_entry(name)
    with call_lock:
        yield nlp


def run_spacy_text(
    name: str,
    text: str,
    *,
    fallback: Optional[str] = None,
    log_label: str = "spaCy",
):
    """Call ``nlp(text)`` under the per-model guard; try ``fallback`` on ``OSError``.

    Returns the ``Doc``, or ``None`` when no usable model is available — the
    caller's cue to take its own pattern fallback.  All ``nlp(text)`` calls go
    through ``spacy_pipeline_guard``, serializing concurrent thread access for
    the same model while letting different models run in parallel.

    Fallback is only attempted when the *primary* model is missing (``OSError``).
    A pipeline error (non-OSError) returns ``None`` immediately without trying
    the fallback, because the fallback would likely hit the same error.
    """
    try:
        with spacy_pipeline_guard(name) as nlp:
            return nlp(text)
    except OSError:
        if not fallback:
            _common.logger.warning("%s model '%s' not found", log_label, name)
            return None
        _common.logger.warning(
            "%s model '%s' not found, trying fallback '%s'",
            log_label, name, fallback,
        )
    except Exception:
        _common.logger.warning(
            "%s model '%s' raised an unexpected error; skipping.",
            log_label, name, exc_info=True,
        )
        return None

    # Reached only when primary raised OSError and fallback is set.
    try:
        with spacy_pipeline_guard(fallback) as nlp:
            return nlp(text)
    except OSError:
        _common.logger.warning(
            "%s fallback model '%s' not found either",
            log_label, fallback,
        )
    except Exception:
        _common.logger.warning(
            "%s fallback model '%s' raised an unexpected error; skipping.",
            log_label, fallback, exc_info=True,
        )
    return None


def clear_spacy_model_cache() -> None:
    """Drop every cached spaCy model. Intended for tests."""
    with _spacy_model_cache_lock:
        _spacy_model_cache.clear()


def get_text_embedder():
    """
    Get or load the TextEmbedder model for high-accuracy semantic similarity.

    Thread-safe: ``_embedder_cache_lock`` serializes the check + construction
    so that concurrent threads never build more than one instance.

    A failed construction stores ``_LOAD_FAILED`` so subsequent callers return
    ``None`` immediately instead of each retrying the expensive construction.
    """
    global _embedder_cache
    with _embedder_cache_lock:
        if _embedder_cache is _LOAD_FAILED:
            return None
        if _embedder_cache:
            return _embedder_cache

        try:
            from ...embeddings.text_embedder import TextEmbedder
            # Use a lightweight but effective model for speed/accuracy balance
            # BAAI/bge-small-en-v1.5 is excellent for semantic similarity
            # Enable caching within the embedder if supported, or use our own
            _embedder_cache = TextEmbedder(model_name="BAAI/bge-small-en-v1.5", normalize=True)
            _common.logger.info("Loaded TextEmbedder for high-accuracy similarity")
            return _embedder_cache
        except Exception as e:
            _common.logger.warning(f"Failed to load TextEmbedder: {e}")
            _embedder_cache = _LOAD_FAILED
            return None


def get_nlp_model():
    """
    Get or load a spaCy model for similarity calculations.
    Prioritizes larger models for better vectors.

    Thread-safe: ``_nlp_cache_lock`` serializes the check + spacy.load() so
    that concurrent threads never load the model more than once.

    A failed load stores ``_LOAD_FAILED`` so subsequent callers return ``None``
    immediately instead of each retrying the expensive load attempt.
    """
    global _nlp_cache
    with _nlp_cache_lock:
        if _nlp_cache is _LOAD_FAILED:
            return None
        if _nlp_cache:
            return _nlp_cache

        if not _common.SPACY_AVAILABLE:
            return None

        try:
            # Prefer larger models for vectors
            # Note: 'en_core_web_lg' has true vectors. 'sm' only has context tensors.
            for model_name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
                if _common.spacy.util.is_package(model_name):
                    try:
                        # Disable parser/ner for speed if we only need vectors
                        _nlp_cache = _common.spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
                        _common.logger.info(f"Loaded spaCy model for similarity: {model_name}")
                        return _nlp_cache
                    except Exception:
                        continue

            # Try loading generic if specific ones fail
            try:
                _nlp_cache = _common.spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
                return _nlp_cache
            except Exception as ignored_exc:
                _common._logger.debug("get_nlp_model() ignored an exception: %s", ignored_exc)

        except Exception as e:
            _common.logger.warning(f"Failed to load spaCy model for similarity: {e}")
        _nlp_cache = _LOAD_FAILED
        return None


# Common synonyms for entity matching optimization
_ENTITY_SYNONYMS = {
    # Entity Types
    "person": ["people", "human", "name", "individual", "artist", "actor", "author", "politician"],
    "org": ["company", "organization", "business", "institution", "agency", "brand", "corporation"],
    "organization": ["company", "business", "institution", "agency", "brand", "corporation"],
    "gpe": ["location", "place", "city", "country", "state", "nation", "region"],
    "loc": ["location", "place", "region", "area"],
    "date": ["time", "year", "day", "month", "period", "duration"],
    "money": ["cost", "price", "value", "currency", "amount"],
    "product": ["item", "object", "commodity", "goods", "device", "tool", "vehicle", "software", "app"],
    "event": ["incident", "occasion", "activity", "happening", "ceremony"],
    "drug": ["medication", "medicine", "pharmaceutical", "chemical", "treatment", "therapy"],
    "chemical": ["drug", "substance", "compound", "element"],
    "disease": ["condition", "illness", "sickness", "disorder", "syndrome", "ailment"],
    
    # Relation Types
    "founded_by": ["founder", "creator", "established_by", "started_by", "originator"],
    "acquired": ["bought", "purchased", "acquisition", "takeover", "ownership", "merged_with"],
    "subsidiary_of": ["owned_by", "parent_company", "part_of", "division_of", "unit_of"],
    "works_for": ["employee_of", "employed_by", "staff_of", "team_member", "employs", "hired_by"],
    "located_in": ["based_in", "headquartered_in", "situated_in", "found_in", "operates_in"],
    "ceo_of": ["leader_of", "head_of", "director_of", "president_of", "chief_executive", "managed_by"],
    "invested_in": ["funded", "financed", "backed", "shareholder_of", "venture_capital"],
    "partner_with": ["collaborate_with", "joint_venture", "alliance", "deal_with", "partnership"],
    "competitor_of": ["rival", "competes_with", "opponent", "nemesis"],
    "manufacturer_of": ["producer_of", "maker_of", "creator_of", "builder_of"],
    "treats": ["cures", "heals", "remedy_for", "used_for", "prescribed_for"],
    "causes": ["leads_to", "results_in", "triggers", "produces", "creates"],
    "diagnosed_with": ["suffers_from", "has_condition", "patient_of", "victim_of"],
}


def find_best_match_index(text: str, candidates: List[str]) -> Tuple[int, float]:
    """
    Find the best matching candidate index and score.
    Uses hybrid similarity approach: Exact -> Synonym -> Substring -> Embeddings -> Vector -> Fuzzy.
    Optimized for batch processing to avoid redundant embedding calculations.
    Embedding/vector similarity stages are best-effort; if they fail, matching falls back
    to remaining strategies instead of raising.
    
    Returns:
        Tuple[int, float]: (best_candidate_index, best_score). Index is -1 if no candidates.
    """
    if not candidates:
        return -1, 0.0
    
    if not text:
        return -1, 0.0
        
    text_lower = text.lower().strip()
    if not text_lower:
        return -1, 0.0
        
    candidates_lower = [c.lower().strip() for c in candidates]
    best_idx = -1
    best_score = 0.0
    
    # 1. Exact Match (Fastest)
    try:
        idx = candidates_lower.index(text_lower)
        return idx, 1.0
    except ValueError:
        pass

    # 1b. Common Synonyms (Fast Heuristic)
    # Use global _ENTITY_SYNONYMS dictionary
    synonyms = _ENTITY_SYNONYMS
    
    # Check text synonyms
    if text_lower in synonyms:
        for syn in synonyms[text_lower]:
            if syn in candidates_lower:
                return candidates_lower.index(syn), 0.95
    
    # Check if any candidate is a synonym of text
    for i, cand in enumerate(candidates_lower):
        if cand in synonyms:
            if text_lower in synonyms[cand]:
                if 0.95 > best_score:
                    best_score = 0.95
                    best_idx = i
    
    # 2. Substring Match (Fast)
    word_pat = None
    try:
        word_pat = triplets.re.compile(rf"\b{triplets.re.escape(text_lower)}\b")
    except Exception:
        word_pat = None

    for i, cand in enumerate(candidates_lower):
        if not cand: continue
        score = 0.0
        if text_lower in cand or cand in text_lower:
            # Calculate length ratio
            ratio = min(len(text_lower), len(cand)) / max(len(text_lower), len(cand))
            score = 0.9 * ratio + 0.1

            if word_pat and word_pat.search(cand):
                score = max(score, 0.88)
        
        if score > best_score:
            best_score = score
            best_idx = i

    # 3. Text Embeddings (High Accuracy Semantic) - Batch Optimized
    if best_score >= 0.85:
        return best_idx, float(best_score)

    embedder = get_text_embedder()
    embedding_idx = -1
    embedding_score = 0.0
    
    if embedder:
        try:
            # Batch embedding: [text, cand1, cand2, ...]
            # We filter out empty candidates to save compute, but need to map back to original indices
            valid_cands_with_idx = [(c, i) for i, c in enumerate(candidates) if c and c.strip()]
            
            if valid_cands_with_idx:
                texts_to_embed = [text] + [c for c, i in valid_cands_with_idx]
                embeddings = list(embedder.embed_batch(texts_to_embed))
                
                if embeddings and len(embeddings) > 1:
                    text_emb = embeddings[0]
                    cand_embs = embeddings[1:]
                    
                    import numpy as np
                    text_norm = np.linalg.norm(text_emb)
                    
                    if text_norm > 0:
                        # Vectorized cosine similarity
                        cand_matrix = np.array(cand_embs)
                        cand_norms = np.linalg.norm(cand_matrix, axis=1)
                        
                        # Avoid division by zero
                        cand_norms[cand_norms == 0] = 1e-10
                        
                        dot_products = np.dot(cand_matrix, text_emb)
                        sims = dot_products / (cand_norms * text_norm)
                        
                        max_sim_idx = np.argmax(sims)
                        max_sim = float(sims[max_sim_idx])
                        
                        if max_sim > embedding_score:
                            embedding_score = max_sim
                            # Map back to original index
                            embedding_idx = valid_cands_with_idx[max_sim_idx][1]
        except Exception as e:
            _common.logger.debug(f"Embedding calculation failed: {e}")
            pass
            
    if embedding_score > best_score:
        best_score = embedding_score
        best_idx = embedding_idx

    # 4. Vector Similarity (Legacy/Fallback)
    if best_score < 0.9:
        nlp = get_nlp_model()
        vector_score = 0.0
        vector_idx = -1
        
        if nlp and nlp.vocab.vectors.shape[0] > 0:
            failing_candidate_idx = -1
            try:
                # _nlp_call_lock serializes all nlp(text) calls on the shared
                # Language object — spaCy Language is not thread-safe to invoke
                # concurrently.  The lock is acquired once for the entire
                # doc + candidates batch so the pipeline state stays consistent.
                with _nlp_call_lock:
                    doc = nlp(text)
                    if doc.vector_norm:
                        for i, candidate in enumerate(candidates):
                            failing_candidate_idx = i
                            if not candidate: continue
                            cand_doc = nlp(candidate)
                            if cand_doc.vector_norm:
                                score = doc.similarity(cand_doc)
                                if score > vector_score:
                                    vector_score = score
                                    vector_idx = i
            except Exception:
                _common.logger.debug(
                    "Vector similarity calculation failed at candidate index %s; continuing with fallback scoring.",
                    failing_candidate_idx if failing_candidate_idx >= 0 else "N/A",
                    exc_info=True,
                )
                vector_score = 0.0
                vector_idx = -1
        
        if vector_score > best_score:
            best_score = vector_score
            best_idx = vector_idx

    # 5. Fuzzy Match (Fallback)
    if best_score < 0.9:
        for i, cand in enumerate(candidates_lower):
            if not cand: continue
            score = _common.difflib.SequenceMatcher(None, text_lower, cand).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
                
    return best_idx, float(best_score)


def calculate_similarity(text: str, candidates: List[str]) -> float:
    """
    Calculate the maximum similarity between text and a list of candidates.
    Wrapper around find_best_match_index.
    """
    _, score = find_best_match_index(text, candidates)
    return score


def match_entity(text: str, entities: List[Entity], threshold: float = 0.8) -> Optional[Entity]:
    """
    Find the best matching entity for the given text.
    Uses optimized batch similarity matching.
    """
    if not text or not entities:
        return None
        
    # Optimization: Check exact match first (case-insensitive)
    text_lower = text.lower().strip()
    for entity in entities:
        if entity.text.lower().strip() == text_lower:
            return entity

    # Use batch matcher
    candidates = [e.text for e in entities]
    best_idx, best_score = find_best_match_index(text, candidates)
    
    if best_idx >= 0 and best_score >= threshold:
        return entities[best_idx]
        
    return None


def filter_entities_for_text(
    text: str,
    entities: List[Entity],
    max_keep: int = 80,
) -> List[Entity]:
    if not text or not entities:
        return []

    if max_keep < 1:
        return []

    if len(entities) <= max_keep:
        return entities

    text_lower = text.lower()
    stop_tokens = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "llc",
        "plc",
        "group",
        "holdings",
        "limited",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "for",
        "to",
        "a",
        "an",
    }

    seen = set()
    matched: _common.List[_common.Entity] = []
    for entity in entities:
        ent_text = getattr(entity, "text", "")
        if not ent_text:
            continue

        key = ent_text.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)

        if key in text_lower:
            matched.append(entity)
            continue

        tokens = triplets.re.findall(r"[a-z0-9]+", key)
        keep = False
        for tok in tokens:
            if len(tok) < 2:
                continue
            if tok in stop_tokens:
                continue
            if tok in text_lower:
                keep = True
                break
        if keep:
            matched.append(entity)

    if matched:
        if len(matched) > max_keep:
            matched.sort(key=lambda e: len(getattr(e, "text", "")), reverse=True)
            return matched[:max_keep]
        return matched

    entities_sorted = sorted(entities, key=lambda e: len(getattr(e, "text", "")), reverse=True)
    return entities_sorted[:max_keep]


def calculate_weighted_confidence(
    item_type: str,
    original_confidence: Optional[float],
    valid_types: Optional[List[str]] = None,
    item_text: Optional[str] = None,
    weight_method: float = 0.5,
    weight_similarity: float = 0.5
) -> Optional[float]:
    """
    Calculate weighted confidence score using both Label and Content similarity.
    Final Score = (weight_method * original_confidence) + (weight_similarity * max(label_sim, content_sim))

    Args:
        item_type: The extracted type/label/predicate (e.g., "PERSON", "founded_by")
        original_confidence: The confidence score from the extraction method
            (0.0-1.0), or None if the method provided no score
        valid_types: List of valid/preferred types provided by user
        item_text: The actual text content extracted (e.g., "Steve Jobs", "acquired")
        weight_method: Weight for the original method confidence (default 0.5)
        weight_similarity: Weight for the similarity score (default 0.5)

    Returns:
        float or None: Weighted confidence score (0.0-1.0). When
        original_confidence is None, returns the similarity score alone —
        or None (still unknown) if valid_types is empty or the similarity
        weight is disabled (<= 0).
    """
    if not valid_types:
        return original_confidence
        
    # Similarity 1: Label vs Valid Types (e.g., "PERSON" vs "Artist")
    label_similarity = calculate_similarity(item_type, valid_types)
    
    # Similarity 2: Content vs Valid Types (e.g., "Picasso" vs "Artist")
    content_similarity = 0.0
    if item_text:
        content_similarity = calculate_similarity(item_text, valid_types)
        
    # Take the best similarity match
    best_similarity = max(label_similarity, content_similarity)
    
    # Normalize weights
    total_weight = weight_method + weight_similarity
    if total_weight <= 0:
        return original_confidence
        
    if original_confidence is None:
        if weight_similarity <= 0:
            # Similarity disabled and no measured confidence: still unknown
            return None
        # No measured confidence to blend; rely on similarity alone
        return max(0.0, min(1.0, best_similarity))

    w_m = weight_method / total_weight
    w_s = weight_similarity / total_weight

    final_score = (w_m * original_confidence) + (w_s * best_similarity)

    return max(0.0, min(1.0, final_score))
