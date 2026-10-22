"""semantic_extract_methods: relations (split from methods.py)."""

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

from typing import Any, Dict, List, Optional

from ..types import (
    Entity,
    Relation,
)

# Try to import spaCy


from . import _common
from . import scoring
from . import triplets


# ============================================================================
# Relation Extraction Methods
# ============================================================================


def extract_relations_pattern(
    text: str, entities: List[Entity], **kwargs
) -> List[Relation]:
    """Pattern-based relation extraction."""
    relations = []

    if not entities:
        return []

    # Create entity pattern from provided entities
    # Sort by length descending to match longest entities first (e.g. "Apple Inc." before "Apple")
    sorted_entities = sorted(entities, key=lambda e: len(e.text), reverse=True)
    
    # Escape entity texts and join with OR
    # We use a non-capturing group for the alternatives
    entity_texts = [triplets.re.escape(e.text) for e in sorted_entities]
    # Remove duplicates
    entity_texts = list(dict.fromkeys(entity_texts))
    
    if not entity_texts:
        return []
        
    ent_pat = f"(?:{'|'.join(entity_texts)})"
    
    # Use entity pattern for subject as well, since we require the subject to be a known entity
    # This prevents matching long strings of text that happen to end with a relation keyword
    subject_pat = ent_pat

    relation_patterns = {
        "founded_by": [
            fr"(?P<subject>{subject_pat})(?:[.,])?\s+(?:was\s+)?founded\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})(?:[.,])?\s+founded\s+(?P<subject>{ent_pat})",
            fr"(?P<subject>{subject_pat})(?:[.,])?\s+(?:was\s+)?established\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})(?:[.,])?\s+established\s+(?P<subject>{ent_pat})",
            fr"(?P<subject>{subject_pat})(?:[.,])?\s+(?:was\s+)?created\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})(?:[.,])?\s+created\s+(?P<subject>{ent_pat})",
            fr"(?P<subject>{subject_pat})(?:[.,])?\s+(?:was\s+)?started\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})(?:[.,])?\s+started\s+(?P<subject>{ent_pat})",
            fr"(?P<subject>{subject_pat})(?:[.,])?\s+(?:was\s+)?co-founded\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})(?:[.,])?\s+co-founded\s+(?P<subject>{ent_pat})",
            fr"(?P<object>{ent_pat})(?:[.,])?\s+is\s+(?:the\s+)?founder\s+of\s+(?P<subject>{ent_pat})",
        ],
        "located_in": [
            fr"(?P<subject>{subject_pat})\s+is\s+located\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+(?:is\s+)?headquartered\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+(?:is\s+)?based\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+has\s+(?:its\s+)?headquarters\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+operates\s+(?:out\s+of|from)\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+has\s+offices\s+in\s+(?P<object>{ent_pat})",
        ],
        "works_for": [
            fr"(?P<subject>{subject_pat})\s+works?\s+for\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+works?\s+at\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+is\s+an?\s+employee\s+of\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+is\s+(?:the\s+)?(?:CEO|CFO|CTO|COO|director|manager|president|founder)\s+of\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+joined\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+was\s+hired\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+serves\s+at\s+(?P<object>{ent_pat})",
        ],
        "born_in": [
            fr"(?P<subject>{subject_pat})\s+was\s+born\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+born\s+in\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+is\s+a\s+native\s+of\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+hails\s+from\s+(?P<object>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+is\s+originally\s+from\s+(?P<object>{ent_pat})",
        ],
        "acquired_by": [
            fr"(?P<subject>{subject_pat})\s+(?:was\s+)?acquired\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})\s+acquired\s+(?P<subject>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+(?:was\s+)?bought\s+by\s+(?P<object>{ent_pat})",
            fr"(?P<object>{ent_pat})\s+bought\s+(?P<subject>{ent_pat})",
            fr"(?P<subject>{subject_pat})\s+is\s+a\s+subsidiary\s+of\s+(?P<object>{ent_pat})",
        ],
    }

    entity_map = {e.text.lower(): e for e in entities}

    for relation_type, patterns in relation_patterns.items():
        for pattern in patterns:
            for match in triplets.re.finditer(pattern, text, triplets.re.IGNORECASE):
                subject_text = match.group("subject").strip()
                object_text = match.group("object").strip()
                
                subject_entity = entity_map.get(subject_text.lower())
                object_entity = entity_map.get(object_text.lower())
                
                if subject_entity and object_entity:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    relations.append(
                        _common.Relation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.7,
                            context=context,
                            metadata={
                                "extraction_method": "pattern",
                                "pattern": pattern,
                            },
                        )
                    )

    return relations


def extract_relations_regex(
    text: str,
    entities: List[Entity],
    patterns: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> List[Relation]:
    """Advanced regex-based relation extraction."""
    if patterns is None:
        patterns = {
            "founded_by": [
                r"(?P<subject>\w+)\s+(?:was\s+)?founded\s+by\s+(?P<object>\w+(?:\s+\w+)*)"
            ],
            "located_in": [r"(?P<subject>\w+)\s+is\s+located\s+in\s+(?P<object>\w+)"],
        }

    relations = []
    entity_map = {e.text.lower(): e for e in entities}

    for relation_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in triplets.re.finditer(pattern, text, triplets.re.IGNORECASE):
                subject_text = match.group("subject")
                object_text = match.group("object")

                subject_entity = entity_map.get(subject_text.lower())
                object_entity = entity_map.get(object_text.lower())

                if subject_entity and object_entity:
                    relations.append(
                        _common.Relation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.75,
                            context=text[
                                max(0, match.start() - 30) : min(
                                    len(text), match.end() + 30
                                )
                            ],
                            metadata={"extraction_method": "regex"},
                        )
                    )

    return relations


def extract_relations_cooccurrence(
    text: str, entities: List[Entity], **kwargs
) -> List[Relation]:
    """Co-occurrence based relation extraction."""
    relations = []

    for i, entity1 in enumerate(entities):
        for entity2 in entities[i + 1 :]:
            distance = abs(entity1.end_char - entity2.start_char)
            if distance < 100:  # Within 100 characters
                start = min(entity1.start_char, entity2.start_char)
                end = max(entity1.end_char, entity2.end_char)
                context = text[max(0, start - 30) : min(len(text), end + 30)]

                relations.append(
                    _common.Relation(
                        subject=entity1,
                        predicate="related_to",
                        object=entity2,
                        confidence=0.6,  # Meets default threshold
                        context=context,
                        metadata={
                            "extraction_method": "co_occurrence",
                            "distance": distance,
                        },
                    )
                )

    return relations


def extract_relations_similarity(
    text: str, entities: List[Entity], relation_types: Optional[List[str]] = None, **kwargs
) -> List[Relation]:
    """
    Similarity-based relation extraction.
    Uses semantic similarity to match the context between entities to provided relation types.
    """
    if not entities:
        return []

    # If no relation types provided, we can't do similarity matching against types
    if not relation_types:
        # Fallback to co-occurrence if no types to match against
        _common.logger.warning("No relation types provided for similarity matching. Falling back to co-occurrence.")
        return extract_relations_cooccurrence(text, entities, **kwargs)

    relations = []

    # Resolve which model to use for vectors (prefer larger models), then
    # invoke it only under its guard: spaCy Languages are shared process-wide
    # and not safe to call from concurrent threads.
    chosen_model = None
    if _common.SPACY_AVAILABLE:
        for model_name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
            if _common.spacy.util.is_package(model_name):
                chosen_model = model_name
                break

    # Pre-compute relation type vectors if possible
    relation_vectors = {}
    has_vectors = False
    if chosen_model:
        try:
            with scoring.spacy_pipeline_guard(chosen_model) as guarded_nlp:
                # Check if model has vectors
                if guarded_nlp.vocab.vectors.shape[0] > 0:
                    has_vectors = True
                    for rt in relation_types:
                        relation_vectors[rt] = guarded_nlp(rt)
        except Exception as ignored_exc:
            _common._logger.debug("extract_relations_similarity() ignored an exception: %s", ignored_exc)
    
    for entity1 in entities:
        for entity2 in entities:
            if entity1 == entity2:
                continue
                
            # Check distance
            distance = abs(entity1.end_char - entity2.start_char)
            # Only consider entities reasonably close (e.g., within same sentence or clause)
            if distance > 100: 
                continue

            # Ensure correct order for extracting text between
            if entity1.end_char < entity2.start_char:
                start_pos = entity1.end_char
                end_pos = entity2.start_char
            else:
                start_pos = entity2.end_char
                end_pos = entity1.start_char
                
            between_text = text[start_pos:end_pos].strip()
            
            if not between_text:
                continue

            # Calculate similarity
            best_type = None
            best_score = 0.0

            if has_vectors and relation_vectors and chosen_model:
                # Vector similarity — each call re-enters the model's guard
                doc = scoring.run_spacy_text(
                    chosen_model, between_text, log_label="spaCy similarity"
                )
                if doc is not None and doc.vector_norm:
                    for rt, vec in relation_vectors.items():
                        if vec.vector_norm:
                            sim = doc.similarity(vec)
                            if sim > best_score:
                                best_score = sim
                                best_type = rt
            else:
                # String similarity / Keyword matching
                from difflib import SequenceMatcher
                for rt in relation_types:
                    # Check for direct keyword presence (strong signal)
                    if rt.lower() in between_text.lower():
                        score = 1.0
                    else:
                        # Fuzzy match
                        score = SequenceMatcher(None, rt.lower(), between_text.lower()).ratio()
                    
                    if score > best_score:
                        best_score = score
                        best_type = rt

            # Threshold
            threshold = kwargs.get("similarity_threshold", 0.4 if has_vectors else 0.6)
            
            if best_type and best_score >= threshold:
                relations.append(
                    _common.Relation(
                        subject=entity1,
                        predicate=best_type,
                        object=entity2,
                        confidence=float(best_score),
                        context=text[max(0, min(entity1.start_char, entity2.start_char) - 20) : min(len(text), max(entity1.end_char, entity2.end_char) + 20)],
                        metadata={
                            "extraction_method": "similarity",
                            "similarity_score": float(best_score),
                            "between_text": between_text
                        },
                    )
                )

    return relations


def extract_relations_dependency(
    text: str, entities: List[Entity], model: str = "en_core_web_sm", **kwargs
) -> List[Relation]:
    """Dependency parsing based relation extraction."""
    if not _common.SPACY_AVAILABLE:
        _common.logger.warning("spaCy not available, falling back to pattern extraction")
        return extract_relations_pattern(text, entities, **kwargs)

    doc = scoring.run_spacy_text(model, text, log_label="spaCy dependency")
    if doc is None:
        return extract_relations_pattern(text, entities, **kwargs)
    relations = []
    
    # Map tokens to entities
    token_to_entity = {}
    for token in doc:
        for entity in entities:
            # Check if token is within entity span
            if token.idx >= entity.start_char and (token.idx + len(token)) <= entity.end_char:
                token_to_entity[token] = entity
                break
    
    # DEBUG: Print token to entity mapping
    # print(f"DEBUG: Token to entity mapping keys: {[t.text for t in token_to_entity.keys()]}")

    # DEBUG: Dump full dependency tree
    # print("DEBUG: Dependency Tree:")
    # for token in doc:
    #     print(f"  {token.i}: {token.text} ({token.dep_}) -> {token.head.text} ({token.head.i})")

    # Helper to find entity for a token (or its head chain)
    def find_subject_entity(token):
        # 1. If acl, check head
        if token.dep_ == "acl":
            head = token.head
            if head in token_to_entity:
                return token_to_entity[head]
            
            # 2. If head is attr/acomp, check its head's nsubj
            if head.dep_ in ["attr", "acomp", "dobj"] and head.head.pos_ in ["VERB", "AUX"]:
                copula = head.head
                for child in copula.children:
                    if child.dep_ in ["nsubj", "nsubjpass"] and child in token_to_entity:
                        return token_to_entity[child]
        return None

    # Helper to expand objects via conjunctions
    def expand_conjunctions(token):
        results = [token]
        for child in token.children:
            if child.dep_ == "conj":
                results.extend(expand_conjunctions(child))
        return results

    for token in doc:
        # Check if token is a potential predicate (Verb)
        # We process verbs that have nsubj OR are acl OR are ROOT/VERB
        
        subject_entity = None
        
        # Case 1: Token has nsubj/nsubjpass
        nsubj = next((c for c in token.children if c.dep_ in ["nsubj", "nsubjpass"]), None)
        if nsubj:
            subject_entity = token_to_entity.get(nsubj)
        
        # Case 2: Token is acl or other modifier, try to infer subject from context
        if not subject_entity:
             subject_entity = find_subject_entity(token)
             
        if not subject_entity:
            continue

        verb = token
        # DEBUG: Print found subject
        # print(f"DEBUG: Found subject {subject_entity.text} for verb {verb.text}")

        # Find objects
        potential_objects = []
        for child in verb.children:
            # Direct objects
            if child.dep_ in ["dobj", "attr", "acomp"]:
                potential_objects.extend(expand_conjunctions(child))
                # Check for "attr of object" pattern (e.g. "CEO of Apple")
                for grandchild in child.children:
                    if grandchild.dep_ == "prep":
                        for greatgrandchild in grandchild.children:
                            if greatgrandchild.dep_ == "pobj":
                                potential_objects.extend(expand_conjunctions(greatgrandchild))
                                
            # Prepositional objects (including agent)
            elif child.dep_ in ["prep", "agent"]:
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        potential_objects.extend(expand_conjunctions(grandchild))
        
        # DEBUG: Print potential objects
        # print(f"DEBUG: Potential objects for verb {verb.text}: {[t.text for t in potential_objects]}")

        for obj_token in potential_objects:
            object_entity = token_to_entity.get(obj_token)
            
            # DEBUG: Print object checking
            # print(f"DEBUG: Checking object token: {obj_token.text}, Entity: {object_entity}")

            if object_entity and subject_entity != object_entity:
                relations.append(
                    _common.Relation(
                        subject=subject_entity,
                        predicate=verb.lemma_,
                        object=object_entity,
                        confidence=0.8,
                        context=text[
                            max(0, token.idx - 30) : min(
                                len(text), obj_token.idx + len(obj_token.text) + 30
                            )
                        ],
                        metadata={
                            "extraction_method": "dependency",
                            "dependency_path": f"{token.dep_} -> ... -> {obj_token.dep_}",
                        },
                    )
                )

    return relations


def extract_relations_huggingface(
    text: str,
    entities: List[Entity],
    model: str,
    device: Optional[str] = None,
    **kwargs,
) -> List[Relation]:
    """HuggingFace relation extraction."""
    loader = _common.HuggingFaceModelLoader(device=device)
    model_obj = loader.load_relation_model(model, **kwargs)

    # Pass kwargs (e.g. threshold)
    results = loader.extract_relations(model_obj, text, entities, **kwargs)

    relations = []
    for result in results:
        relations.append(
            _common.Relation(
                subject=result["subject"],
                predicate=result["relation"],
                object=result["object"],
                confidence=result.get("score", 1.0),
                context=text,
                metadata={
                    "model": model,
                    "extraction_method": "huggingface"
                }
            )
        )
    return relations


def extract_relations_llm(
    text: str,
    entities: List[Entity],
    provider: str = "openai",
    model: Optional[str] = None,
    silent_fail: bool = False,
    max_text_length: Optional[int] = None,
    structured_output_mode: str = "typed",
    max_retries: int = 3,
    extract_temporal_bounds: bool = False,
    **kwargs,
) -> List[Relation]:
    """
    LLM-based relation extraction.

    Args:
        text: Input text
        entities: Pre-extracted entities
        provider: LLM provider
        model: LLM model
        silent_fail: If True, return empty list on error. If False (default), raise exception.
        max_text_length: Maximum text length before auto-chunking. None = provider default.
        max_retries: Maximum number of retries for LLM calls (default: 3)
        extract_temporal_bounds: If True, extend the prompt to extract temporal validity
            per relation. Each relation's metadata gains: valid_from, valid_until,
            temporal_confidence (0.0–1.0), and temporal_source_text. Low confidence (<0.5)
            produces a warning log but is not suppressed. Default False.
        **kwargs: Additional options
    """
    # Support llm_model parameter to disambiguate from ML model
    if "llm_model" in kwargs:
        model = kwargs.pop("llm_model")
    
    # Check cache
    cache_params = {
        "provider": provider,
        "model": model,
        "max_text_length": max_text_length,
        "structured_output_mode": structured_output_mode,
        "max_retries": max_retries,
        "relation_types": kwargs.get("relation_types"),
        "extract_temporal_bounds": extract_temporal_bounds,
        # Deterministic fingerprint over the entity inputs (text + label) that
        # affect the prompt. Stable across processes so the persistent cache
        # rehits after a restart (unlike the process-randomized built-in hash()).
        "entities_hash": _common._stable_fingerprint(
            [f"{e.text}\x01{e.label}" for e in entities]
        ) if entities else "",
        **_common._generation_cache_params(kwargs),
    }
    cached_result = _common._result_cache.get("relations", text, **cache_params)
    if cached_result is not None:
        _common.logger.debug(f"Cache hit for relation extraction ({len(cached_result)} relations)")
        return cached_result
    
    # 1. PRE-EXTRACTION VALIDATION
    if not text or not text.strip():
        error_msg = "Text is empty or whitespace only"
        _common.logger.error(error_msg)
        if not silent_fail:
            raise _common.ProcessingError(error_msg)
        return []

    if not entities:
        error_msg = "No entities provided for relation extraction. Relations require existing entities."
        _common.logger.error(error_msg)
        if not silent_fail:
            raise _common.ProcessingError(error_msg)
        return []

    # Pass api_key if provided in kwargs
    provider_kwargs = kwargs.copy()
    
    # Check if api_key is provided but empty, or not provided at all
    if "api_key" not in provider_kwargs or not provider_kwargs["api_key"]:
        import os
        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        if api_key:
            provider_kwargs["api_key"] = api_key
            
    # Remove None/empty API key if still present to avoid provider errors
    if "api_key" in provider_kwargs and not provider_kwargs["api_key"]:
        del provider_kwargs["api_key"]

    # 2. PROVIDER VALIDATION
    try:
        llm = _common.create_provider(provider, model=model, **provider_kwargs)
        if not llm.is_available():
            error_msg = f"{provider} provider not available for relation extraction (key missing?)."
            _common.logger.error(error_msg)
            if not silent_fail:
                raise _common.ProcessingError(error_msg)
            return []
    except Exception as e:
        error_msg = f"Failed to create {provider} provider for relations: {e}"
        _common.logger.error(error_msg)
        if not silent_fail:
            raise _common.ProcessingError(error_msg) from e
        return []

    # 3. TEXT LENGTH CHECK AND CHUNKING
    if max_text_length is None:
        # Default limits for chunking only - NOT for LLM generation
        max_text_length = {
            "groq": 64000,
            "openai": 64000,
            "gemini": 64000,
            "anthropic": 64000,
            "deepseek": 64000,
        }.get(provider.lower(), 32000)
    
    if len(text) > max_text_length:
        _common.logger.info(f"Text length ({len(text)}) exceeds limit for relations. Chunking...")
        return _extract_relations_chunked(
            text, entities, provider=provider, model=model,
            silent_fail=silent_fail, max_text_length=max_text_length,
            max_retries=max_retries,
            extract_temporal_bounds=extract_temporal_bounds,
            **kwargs
        )

    original_entities = entities
    # Use a fixed internal default for prompt entity cap (do not accept overrides from kwargs)
    max_entities_prompt = 80
    prompt_entities = original_entities
    if max_entities_prompt > 0 and len(original_entities) > max_entities_prompt:
        prompt_entities = scoring.filter_entities_for_text(
            text,
            original_entities,
            max_keep=max_entities_prompt,
        )

    entities_str = ", ".join([f"{e.text} ({e.label})" for e in prompt_entities])

    # Use custom relation types if provided
    relation_types = kwargs.get("relation_types")
    if relation_types:
        relation_types_str = ", ".join(relation_types)
        relation_types_instruction = f"""
Preferred relation types: {relation_types_str}.
You may also use related or similar relation types if they better capture the relationship (e.g., variations, synonyms, or domain-specific relations).
If a relation doesn't fit any of the preferred types, use the most appropriate type from the preferred list or a closely related type that accurately describes the relationship."""
    else:
        relation_types_instruction = """
Extract meaningful relationships between entities. Use appropriate relation types that accurately describe how entities are connected.
Common relation types include: related_to, part_of, located_in, created_by, uses, depends_on, interacts_with, and similar variations."""

    verbose_mode = kwargs.get("verbose", False)
    if verbose_mode:
        _common.logger.debug(
            "[methods.extract_relations_llm] Constructing prompt for %d entities...",
            len(prompt_entities),
        )

    if not _common.SCHEMAS_AVAILABLE:
        raise ImportError("Pydantic schemas not available. Install pydantic/instructor to use LLM extraction.")

    # ── Base prompt (always included) ───────────────────────────────────────
    if not extract_temporal_bounds:
        prompt = f"""Extract relations between entities from the provided text.
Return the result as a JSON object with a "relations" key containing the list of relations.
Each relation must have 'subject', 'predicate', and 'object' fields.

Example output (JSON format only):
{{
  "relations": [
    {{"subject": "Entity A", "predicate": "related_to", "object": "Entity B", "confidence": 0.95}},
    {{"subject": "Subject Entity", "predicate": "action_verb", "object": "Object Entity", "confidence": 0.90}}
  ]
}}

Instructions:
1. Extract relations ONLY from the text provided below.
2. Do not include any relations from the example above.
3. Use the provided entities list as a reference for subjects and objects.
4. {relation_types_instruction}

Text to extract from:
{text}
Entities found in text: {entities_str}"""
    else:
        # ── Temporal-extended prompt ─────────────────────────────────────────
        prompt = f"""Extract relations between entities from the provided text, along with temporal validity information for each relation.
Return the result as a JSON object with a "relations" key. Each relation must have:
'subject', 'predicate', 'object', 'confidence', 'valid_from', 'valid_until', 'temporal_confidence', 'temporal_source_text'.

TEMPORAL EXTRACTION RULES:
- valid_from: ISO 8601 date or exact phrase from the text for when this relation became valid. Set to null if no temporal signal is present.
- valid_until: ISO 8601 date or exact phrase for when this relation ceased. Set to null if open-ended or absent.
- temporal_confidence (float 0.0–1.0) — calibrated as follows:
    1.00 = full ISO date ("2022-03-15", "March 15, 2022")
    0.90 = explicit year + month ("March 2022", "2022-03")
    0.85 = explicit year only ("in 2022", "since 2021", "from 2019")
    0.75 = quarter ("Q3 2023", "Q2 2021")
    0.65 = named season or approximate range ("summer 2022", "early 2020s", "mid-2022")
    0.50 = vague relative with computable anchor ("last year", "three months ago")
    0.35 = highly vague relative ("recently", "years ago", "in the past")
    0.00 = no temporal signal present for this relation
- temporal_source_text: the EXACT verbatim substring from the source text that contains the temporal signal. Set to null when temporal_confidence is 0.0.

IMPORTANT: Do NOT invent or guess dates. If the text contains no temporal signal for a relation, set valid_from and valid_until to null and temporal_confidence to 0.0.

Few-shot examples (do NOT include these in your output):
  Text: "Apple acquired Beats in May 2014."
  → valid_from: "2014-05-01", valid_until: null, temporal_confidence: 0.90, temporal_source_text: "May 2014"

  Text: "The CEO has led the company since Q3 2020."
  → valid_from: "Q3 2020", valid_until: null, temporal_confidence: 0.75, temporal_source_text: "since Q3 2020"

  Text: "Last year, Google partnered with Samsung."
  → valid_from: "last year", valid_until: null, temporal_confidence: 0.50, temporal_source_text: "Last year"

  Text: "The firm was under enhanced supervision between Q2 and Q4 2021."
  → valid_from: "Q2 2021", valid_until: "Q4 2021", temporal_confidence: 0.75, temporal_source_text: "between Q2 and Q4 2021"

  Text: "Microsoft develops Windows."
  → valid_from: null, valid_until: null, temporal_confidence: 0.00, temporal_source_text: null

Example JSON output format:
{{
  "relations": [
    {{
      "subject": "Apple", "predicate": "acquired", "object": "Beats",
      "confidence": 0.97,
      "valid_from": "2014-05-01", "valid_until": null,
      "temporal_confidence": 0.90, "temporal_source_text": "May 2014"
    }}
  ]
}}

Instructions:
1. Extract relations ONLY from the text provided below.
2. Do not include any relations from the examples above.
3. Use the provided entities list as a reference for subjects and objects.
4. {relation_types_instruction}

Text to extract from:
{text}
Entities found in text: {entities_str}"""

    try:
        # Use typed generation with Pydantic schema
        if verbose_mode:
            _common.logger.debug(
                "[methods.extract_relations_llm] Calling llm.generate_typed (%s/%s)...",
                provider, model,
            )
        # Forward all caller-supplied generation kwargs so they reach
        # generate_typed and the underlying provider API. max_retries is
        # always set from the explicit parameter.
        call_kwargs = kwargs.copy()
        call_kwargs["max_retries"] = max_retries

        # Select schema based on whether temporal extraction is requested
        active_schema = _common.RelationsWithTemporalResponse if extract_temporal_bounds else _common.RelationsResponse
        result_obj = llm.generate_typed(prompt, schema=active_schema, **call_kwargs)
        if verbose_mode:
            _common.logger.debug(
                "[methods.extract_relations_llm] Received response from %s.", provider
            )

        # Convert back to internal Relation format (robust across providers)
        # Normalize typed result to a plain dict compatible with _parse_relation_result
        try:
            if hasattr(result_obj, "model_dump"):
                parsed = result_obj.model_dump()
            elif isinstance(result_obj, dict):
                parsed = result_obj
            elif hasattr(result_obj, "relations"):
                # Instructor may return objects for each relation; convert where possible
                rel_items = []
                for r in getattr(result_obj, "relations", []):
                    if hasattr(r, "model_dump"):
                        rel_items.append(r.model_dump())
                    elif isinstance(r, dict):
                        rel_items.append(r)
                    else:
                        # Best-effort attribute access — include temporal fields when present
                        rel_items.append({
                            "subject": getattr(r, "subject", ""),
                            "object": getattr(r, "object", ""),
                            "predicate": getattr(r, "predicate", "related_to"),
                            "confidence": getattr(r, "confidence", 0.9),
                            "valid_from": getattr(r, "valid_from", None),
                            "valid_until": getattr(r, "valid_until", None),
                            "temporal_confidence": getattr(r, "temporal_confidence", 0.0),
                            "temporal_source_text": getattr(r, "temporal_source_text", None),
                        })
                parsed = {"relations": rel_items}
            else:
                parsed = result_obj
        except Exception:
            parsed = result_obj

        # Use common parser to build internal Relation objects
        relations = _parse_relation_result(
            parsed, original_entities, text, provider, model,
            extraction_method="llm_typed",
            extract_temporal_bounds=extract_temporal_bounds,
        )

        # If typed path returned no relations, attempt a structured JSON fallback
        if not relations:
            try:
                if verbose_mode:
                    _common.logger.debug(
                        "[methods.extract_relations_llm] Typed result empty, attempting structured JSON fallback..."
                    )
                raw_json = llm.generate_structured(prompt, **call_kwargs)
                relations = _parse_relation_result(
                    raw_json, original_entities, text, provider, model,
                    extraction_method="llm_typed",
                    extract_temporal_bounds=extract_temporal_bounds,
                )
            except Exception as _e:
                # Keep relations as empty if fallback fails
                _common._logger.debug("fallback relation parse ignored an exception: %s", _e)

        _common.logger.info(f"Successfully extracted {len(relations)} relations using {provider}/{model} (typed)")
        _common._result_cache.set("relations", text, relations, **cache_params)
        return relations

    except Exception as e:
        # Check for length/token limit errors
        error_msg_str = str(e).lower()
        if "length" in error_msg_str or "max_tokens" in error_msg_str:
            _common.logger.warning(f"LLM output truncated due to length limit. Reducing chunk size and retrying... ({e})")

            # Determine new chunk size (halve it)
            current_max = max_text_length or len(text)
            new_max = current_max // 2

            if new_max > 100:  # Minimum viable chunk size check
                return _extract_relations_chunked(
                    text, entities, provider=provider, model=model,
                    silent_fail=silent_fail, max_text_length=new_max,
                    structured_output_mode=structured_output_mode,
                    extract_temporal_bounds=extract_temporal_bounds,
                    **kwargs
                )

        error_msg = f"LLM relation extraction failed ({provider}/{model}): {e}"
        _common.logger.error(error_msg, exc_info=True)
        if not silent_fail:
            if isinstance(e, _common.ProcessingError):
                raise
            raise _common.ProcessingError(error_msg) from e
        return []


def _parse_relation_result(
    result: Any,
    entities: List[Entity],
    text: str,
    provider: str,
    model: Optional[str],
    extraction_method: str = "llm",
    extract_temporal_bounds: bool = False,
) -> List[Relation]:
    """Helper to parse raw LLM result into Relation objects."""
    relations = []
    items = []

    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        for key in ["relations", "data", "results"]:
            if key in result and isinstance(result[key], list):
                items = result[key]
                break
        if not items and "subject" in result:
            items = [result]

    for item in items:
        if not isinstance(item, dict):
            continue

        subject_text = item.get("subject", "")
        object_text = item.get("object", "")

        if not subject_text or not object_text:
            continue

        # Ensure they are strings
        subject_text = str(subject_text)
        object_text = str(object_text)

        # Find matching entities using hybrid similarity; fall back to a
        # synthetic entity so multi-value results are never silently dropped.
        subject_entity = scoring.match_entity(subject_text, entities)
        object_entity = scoring.match_entity(object_text, entities)

        if not subject_entity:
            subject_entity = _common.Entity(
                text=subject_text, label="UNKNOWN",
                start_char=0, end_char=len(subject_text),
                confidence=0.8, metadata={"synthetic": True},
            )
        if not object_entity:
            object_entity = _common.Entity(
                text=object_text, label="UNKNOWN",
                start_char=0, end_char=len(object_text),
                confidence=0.8, metadata={"synthetic": True},
            )

        metadata: dict = {
            "provider": provider,
            "model": model,
            "extraction_method": extraction_method,
        }

        if extract_temporal_bounds:
            temporal_confidence = float(item.get("temporal_confidence") or 0.0)
            valid_from = item.get("valid_from")
            valid_until = item.get("valid_until")
            temporal_source_text = item.get("temporal_source_text")

            metadata["valid_from"] = valid_from
            metadata["valid_until"] = valid_until
            metadata["temporal_confidence"] = temporal_confidence
            metadata["temporal_source_text"] = temporal_source_text

            if temporal_confidence < 0.5 and (valid_from is not None or valid_until is not None):
                _common.logger.warning(
                    "Low temporal confidence (%.2f) for '%s' (%s → %s). Source: %r",
                    temporal_confidence,
                    item.get("predicate", ""),
                    item.get("subject", ""),
                    item.get("object", ""),
                    temporal_source_text,
                )

        relations.append(
            _common.Relation(
                subject=subject_entity,
                predicate=item.get("predicate", "related_to"),
                object=object_entity,
                confidence=item.get("confidence", 0.9),
                context=text,
                metadata=metadata,
            )
        )
    return relations


def _extract_relations_chunked(
    text: str,
    entities: List[Entity],
    provider: str,
    model: Optional[str],
    silent_fail: bool,
    max_text_length: int,
    structured_output_mode: str = "typed",
    max_retries: int = 3,
    extract_temporal_bounds: bool = False,
    **kwargs
) -> List[Relation]:
    """Internal helper to extract relations from long text by chunking."""
    from ...split import TextSplitter
    
    splitter = TextSplitter(
        method="recursive",
        chunk_size=max_text_length,
        chunk_overlap=int(max_text_length * 0.1)
    )
    chunks = splitter.split(text)
    
    all_relations = []
    
    from ..config import resolve_max_workers
    max_workers = resolve_max_workers(explicit=kwargs.get("max_workers"))
    
    with _common.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {}
        for i, chunk in enumerate(chunks):
            # Only include entities that appear in this chunk (or close to it)
            chunk_entities = [
                e for e in entities 
                if e.start_char >= chunk.start_index - 100 and e.end_char <= chunk.end_index + 100
            ]
            
            if not chunk_entities:
                continue
                
            _common.logger.debug(f"Scheduling relation extraction for chunk {i+1}/{len(chunks)} with {len(chunk_entities)} entities")
            
            # Only pass minimal kwargs downstream
            limited_kwargs = {k: kwargs[k] for k in ("relation_types", "temperature", "verbose") if k in kwargs}
            future = executor.submit(
                extract_relations_llm,
                chunk.text,
                entities=chunk_entities,
                provider=provider,
                model=model,
                silent_fail=False,
                max_text_length=len(chunk.text) + 1,
                structured_output_mode=structured_output_mode,
                max_retries=max_retries,
                extract_temporal_bounds=extract_temporal_bounds,
                **limited_kwargs
            )
            future_to_chunk[future] = i
            
        for future in _common.as_completed(future_to_chunk):
            i = future_to_chunk[future]
            try:
                chunk_rels = future.result()
                all_relations.extend(chunk_rels)
            except Exception as e:
                if not silent_fail:
                    _common.logger.error(f"Chunk {i+1} failed: {e}")
                    raise
                _common.logger.warning(f"Chunk {i+1} failed (silent): {e}")
        
    return all_relations
