"""semantic_extract_methods: triplets (split from methods.py)."""

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

import re
from typing import List, Optional

from ..types import (
    Entity,
    Relation,
    Triplet,
)

# Try to import spaCy


from . import _common
from . import scoring


# ============================================================================
# Triplet Extraction Methods
# ============================================================================


def extract_triplets_pattern(
    text: str,
    entities: Optional[List[Entity]] = None,
    relations: Optional[List[Relation]] = None,
    **kwargs,
) -> List[Triplet]:
    """Pattern-based triplet extraction."""
    triplets = []

    if relations:
        # Convert relations to triplets
        for relation in relations:
            triplets.append(
                _common.Triplet(
                    subject=relation.subject.text,
                    predicate=relation.predicate,
                    object=relation.object.text,
                    confidence=relation.confidence,
                    metadata={"extraction_method": "pattern", **relation.metadata},
                )
            )
    elif entities:
        # Simple triplet extraction from entities
        # Look for subject-verb-object patterns
        pattern = r"(?P<subject>\w+)\s+(?P<predicate>\w+)\s+(?P<object>\w+)"
        for match in re.finditer(pattern, text):
            subject_text = match.group("subject")
            predicate_text = match.group("predicate")
            object_text = match.group("object")

            subject_entity = scoring.match_entity(subject_text, entities)
            object_entity = scoring.match_entity(object_text, entities)

            if subject_entity and object_entity:
                triplets.append(
                    _common.Triplet(
                        subject=subject_entity.text,
                        predicate=predicate_text,
                        object=object_entity.text,
                        confidence=0.7,
                        metadata={"extraction_method": "pattern"},
                    )
                )

    return triplets


def extract_triplets_ml(
    text: str,
    entities: Optional[List[Entity]] = None,
    relations: Optional[List[Relation]] = None,
    **kwargs,
) -> List[Triplet]:
    """ML-based triplet extraction.

    Mirrors what ``ml`` means for the entity and relation stages, where it is
    the spaCy-backed backend: relations are resolved with dependency parsing
    and every subject-predicate-object relation becomes a triplet. Relations
    are generated here when the caller has none, so the method can also be
    used on its own.
    """
    if relations is None and entities:
        relations = relations.extract_relations_dependency(text, entities, **kwargs)

    triplets = []
    for relation in relations or []:
        triplets.append(
            _common.Triplet(
                subject=relation.subject.text,
                predicate=relation.predicate,
                object=relation.object.text,
                confidence=relation.confidence,
                # Keep the relation's provenance (context, dependency path, …)
                # but record the method the caller asked for last, so the
                # output shows which method actually produced the triplet.
                metadata={**relation.metadata, "extraction_method": "ml"},
            )
        )

    return triplets


def extract_triplets_rules(
    text: str, entities: Optional[List[Entity]] = None, **kwargs
) -> List[Triplet]:
    """Rule-based triplet extraction."""
    triplets = []

    if not entities:
        return triplets

    # Rule: Look for verb patterns between entities
    sentences = re.split(r"[.!?]+", text)
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if word.lower() in ["is", "was", "has", "founded", "located"]:
                # Look for entities before and after
                if i > 0 and i < len(words) - 1:
                    before = " ".join(words[:i])
                    after = " ".join(words[i + 1 :])

                    subject_entity = next(
                        (e for e in entities if e.text.lower() in before.lower()), None
                    )
                    object_entity = next(
                        (e for e in entities if e.text.lower() in after.lower()), None
                    )

                    if subject_entity and object_entity:
                        triplets.append(
                            _common.Triplet(
                                subject=subject_entity.text,
                                predicate=word,
                                object=object_entity.text,
                                confidence=0.7,
                                metadata={"extraction_method": "rules"},
                            )
                        )

    return triplets


def extract_triplets_huggingface(
    text: str, model: str, device: Optional[str] = None, entities: Optional[List[Entity]] = None, **kwargs
) -> List[Triplet]:
    """HuggingFace triplet extraction."""
    loader = _common.HuggingFaceModelLoader(device=device)
    model_obj = loader.load_triplet_model(model, **kwargs)
    
    # REBEL needs special tokens to be preserved
    if "skip_special_tokens" not in kwargs:
        kwargs["skip_special_tokens"] = False
        
    results = loader.extract_triplets(model_obj, text, **kwargs)

    triplets = []
    for result in results:
        if "triplet" in result:
            decoded_text = result["triplet"]
            
            # Clean up common special tokens that might interfere or are noise
            decoded_text = decoded_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
            
            # Parse REBEL format: <triplet> subject <subj> predicate <obj> object
            # We use a non-greedy match and lookahead for next triplet or end of string
            import re
            pattern = r"<triplet>(?P<head>.*?)<subj>(?P<relation>.*?)<obj>(?P<tail>.*?)(?=<triplet>|$)"
            
            matches = re.finditer(pattern, decoded_text)
            for match in matches:
                head = match.group("head").strip()
                relation = match.group("relation").strip()
                tail = match.group("tail").strip()
                
                if head and relation and tail:
                    # Head/tail are raw decoded strings from the model. Tag any
                    # that match no known entity so the GraphBuilder promotes
                    # them instead of leaving a dangling edge (#1463).
                    # TripletExtractor dispatches with entities=..., so this is
                    # the real NER list here, not a dead comparison.
                    hf_entities = entities or []
                    synthetic_endpoints = [
                        endpoint_text
                        for endpoint_text in (head, tail)
                        if not scoring.match_entity(endpoint_text, hf_entities)
                    ]
                    hf_metadata = {
                        "model": model,
                        "extraction_method": "huggingface_rebel",
                    }
                    if synthetic_endpoints:
                        hf_metadata["synthetic_endpoints"] = synthetic_endpoints
                    triplets.append(
                        _common.Triplet(
                            subject=head,
                            predicate=relation,
                            object=tail,
                            confidence=0.9, # Model generation doesn't provide per-triplet confidence
                            metadata=hf_metadata
                        )
                    )

    return triplets


def extract_triplets_llm(
    text: str,
    entities: Optional[List[Entity]] = None,
    relations: Optional[List[Relation]] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    silent_fail: bool = False,
    max_text_length: Optional[int] = None,
    structured_output_mode: str = "typed",
    max_retries: int = 3,
    **kwargs,
) -> List[Triplet]:
    """
    LLM-based triplet extraction.
    
    Args:
        text: Input text
        entities: Pre-extracted entities (optional)
        relations: Pre-extracted relations (optional)
        provider: LLM provider
        model: LLM model
        silent_fail: If True, return empty list on error. If False (default), raise exception.
        max_text_length: Maximum text length before auto-chunking. None = provider default.
        max_retries: Maximum number of retries for LLM calls (default: 3)
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
        "triplet_types": kwargs.get("triplet_types"),
        # Deterministic fingerprints over the entity/relation inputs that affect
        # the prompt. Stable across processes so the persistent cache rehits
        # after a restart (unlike the process-randomized built-in hash()).
        "entities_hash": _common._stable_fingerprint(
            [f"{e.text}\x01{e.label}" for e in entities]
        ) if entities else "",
        "relations_hash": _common._stable_fingerprint(
            [str(r) for r in relations]
        ) if relations else "",
        **_common._generation_cache_params(kwargs),
    }
    cached_result = _common._result_cache.get("triplets", text, **cache_params)
    if cached_result is not None:
        _common.logger.debug(f"Cache hit for triplet extraction ({len(cached_result)} triplets)")
        return cached_result
    
    # 1. PRE-EXTRACTION VALIDATION
    if not text or not text.strip():
        error_msg = "Text is empty or whitespace only"
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
            error_msg = f"{provider} provider not available for triplet extraction."
            _common.logger.error(error_msg)
            if not silent_fail:
                raise _common.ProcessingError(error_msg)
            return []
    except Exception as e:
        error_msg = f"Failed to create {provider} provider for triplets: {e}"
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
        _common.logger.info(f"Text length ({len(text)}) exceeds limit for triplets. Chunking...")
        return _extract_triplets_chunked(
            text, provider=provider, model=model, 
            silent_fail=silent_fail, max_text_length=max_text_length, 
            max_retries=max_retries,
            **kwargs
        )
    
    # Use custom triplet types if provided
    triplet_types = kwargs.get("triplet_types")
    if triplet_types:
        triplet_types_str = ", ".join(triplet_types)
        triplet_types_instruction = f"""
Preferred triplet predicates: {triplet_types_str}.
You may also use related or similar predicates if they better capture the relationship (e.g., variations, synonyms, or domain-specific predicates).
If a predicate doesn't fit any of the preferred types, use the most appropriate type from the preferred list or a closely related type that accurately describes the relationship."""
    else:
        triplet_types_instruction = """
Extract meaningful triplets (subject-predicate-object). Use appropriate predicates that accurately describe the relationship.
Common predicates include: is_a, part_of, has_property, related_to, caused_by, etc."""

    if not _common.SCHEMAS_AVAILABLE:
        raise ImportError("Pydantic schemas not available. Install pydantic/instructor to use LLM extraction.")

    prompt = f"""Extract RDF triplets (subject-predicate-object) from the provided text.
Return the result as a JSON object with a "triplets" key containing the list of triplets.
Each triplet must have 'subject', 'predicate', and 'object' fields.

Example output (JSON format only):
{{
  "triplets": [
    {{"subject": "Subject", "predicate": "predicate_relation", "object": "Object", "confidence": 0.99}},
    {{"subject": "Concept A", "predicate": "is_a", "object": "Concept B", "confidence": 0.95}}
  ]
}}

Instructions:
1. Extract triplets ONLY from the text provided below.
2. Do not include any triplets from the example above.
3. Ensure subjects and objects are substrings from the text.
4. {triplet_types_instruction}

Text to extract from:
{text}"""

    try:
        # Use typed generation with Pydantic schema
        call_kwargs = kwargs.copy()
        call_kwargs["max_retries"] = max_retries
        result_obj = llm.generate_typed(prompt, schema=_common.TripletsResponse, **call_kwargs)
        
        # Convert back to internal Triplet format
        triplets = []
        for t_out in result_obj.triplets:
            # An LLM triple may reference an endpoint that does not match any
            # entity extracted by NER. Record those endpoints so the GraphBuilder
            # can promote them as synthetic entities instead of leaving a
            # dangling edge (see issue #1463).
            synthetic_endpoints = []
            for endpoint_text in (t_out.subject, t_out.object):
                if not scoring.match_entity(endpoint_text, entities or []):
                    synthetic_endpoints.append(endpoint_text)
            metadata = {
                "provider": provider,
                "model": model,
                "extraction_method": "llm_typed",
            }
            if synthetic_endpoints:
                metadata["synthetic_endpoints"] = synthetic_endpoints
            triplets.append(_common.Triplet(
                subject=t_out.subject,
                predicate=t_out.predicate,
                object=t_out.object,
                confidence=t_out.confidence,
                metadata=metadata,
            ))
        
        _common.logger.info(f"Successfully extracted {len(triplets)} triplets using {provider}/{model} (typed)")
        _common._result_cache.set("triplets", text, triplets, **cache_params)
        return triplets
        
    except Exception as e:
        # Check for length/token limit errors
        error_msg_str = str(e).lower()
        if "length" in error_msg_str or "max_tokens" in error_msg_str:
            _common.logger.warning(f"LLM output truncated due to length limit. Reducing chunk size and retrying... ({e})")
            
            # Determine new chunk size (halve it)
            current_max = max_text_length or len(text)
            new_max = current_max // 2
            
            if new_max > 100: # Minimum viable chunk size check
                return _extract_triplets_chunked(
                    text, provider=provider, model=model, 
                    silent_fail=silent_fail, max_text_length=new_max, 
                    **kwargs
                )

        error_msg = f"LLM triplet extraction failed ({provider}/{model}): {e}"
        _common.logger.error(error_msg, exc_info=True)
        if not silent_fail:
            if isinstance(e, _common.ProcessingError):
                raise
            raise _common.ProcessingError(error_msg) from e
        return []


def _extract_triplets_chunked(
    text: str,
    provider: str,
    model: Optional[str],
    silent_fail: bool,
    max_text_length: int,
    structured_output_mode: str = "typed",
    **kwargs
) -> List[Triplet]:
    """Internal helper to extract triplets from long text by chunking."""
    from ...split import TextSplitter
    
    splitter = TextSplitter(
        method="recursive",
        chunk_size=max_text_length,
        chunk_overlap=int(max_text_length * 0.1)
    )
    chunks = splitter.split(text)
    
    all_triplets = []
    
    from ..config import resolve_max_workers
    max_workers = resolve_max_workers(explicit=kwargs.get("max_workers"))
    
    with _common.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {}
        for i, chunk in enumerate(chunks):
            _common.logger.debug(f"Scheduling triplet extraction for chunk {i+1}/{len(chunks)}")
            future = executor.submit(
                extract_triplets_llm,
                chunk.text,
                provider=provider,
                model=model,
                silent_fail=False,
                max_text_length=len(chunk.text) + 1,
                structured_output_mode=structured_output_mode,
                **kwargs
            )
            future_to_chunk[future] = i
        
        for future in _common.as_completed(future_to_chunk):
            i = future_to_chunk[future]
            try:
                chunk_triplets = future.result()
                all_triplets.extend(chunk_triplets)
            except Exception as e:
                if not silent_fail:
                    _common.logger.error(f"Chunk {i+1} failed: {e}")
                    raise
                _common.logger.warning(f"Chunk {i+1} failed (silent): {e}")
                
    return all_triplets
