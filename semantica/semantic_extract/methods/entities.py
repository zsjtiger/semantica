"""semantic_extract_methods: entities (split from methods.py)."""

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

from typing import Dict, List, Optional

from ..types import (
    Entity,
)

# Try to import spaCy


from . import _common
from . import scoring
from . import triplets


# ============================================================================
# Entity Extraction Methods
# ============================================================================


def extract_entities_pattern(text: str, **kwargs) -> List[Entity]:
    """Pattern-based entity extraction using regex."""
    entities = []

    patterns = {
        "ORG": r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company)(?:\.|\b))",
        "PERSON": r"\b([A-Z][a-z]+(?:\s+(?!Inc|Corp|LLC|Ltd|Company)[A-Z][a-z]+)+)\b",
        "GPE": r"\b([A-Z][a-z]+\s*(?:City|State|Country|Nation))\b",
        "DATE": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b",
    }

    for label, pattern in patterns.items():
        for match in triplets.re.finditer(pattern, text):
            entities.append(
                _common.Entity(
                    text=match.group(1) if match.groups() else match.group(0),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.7,
                    metadata={"extraction_method": "pattern"},
                )
            )

    return entities


def extract_entities_regex(
    text: str, patterns: Optional[Dict[str, str]] = None, **kwargs
) -> List[Entity]:
    """Advanced regex-based entity extraction with custom patterns."""
    entities = []

    if patterns is None:
        patterns = {
            "PERSON": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
            "ORG": r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation))\b",
            "GPE": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            "DATE": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b",
            "MONEY": r"\b(\$[\d,]+(?:\.\d{2})?)\b",
            "PERCENT": r"\b(\d+(?:\.\d+)?%)\b",
        }

    for label, pattern in patterns.items():
        for match in triplets.re.finditer(pattern, text, triplets.re.IGNORECASE):
            entities.append(
                _common.Entity(
                    text=match.group(1) if match.groups() else match.group(0),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.75,
                    metadata={"extraction_method": "regex", "pattern": pattern},
                )
            )

    return entities


def extract_entities_rules(text: str, **kwargs) -> List[Entity]:
    """Rule-based entity extraction using linguistic rules."""
    entities = []
    words = text.split()

    # Rule: Capitalized words at sentence start are likely entities
    sentences = triplets.re.split(r"[.!?]+", text)
    char_offset = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            char_offset += 1
            continue

        words_in_sent = sentence.split()
        if words_in_sent:
            first_word = words_in_sent[0]
            if first_word and first_word[0].isupper() and len(first_word) > 2:
                start = text.find(first_word, char_offset)
                if start >= 0:
                    entities.append(
                        _common.Entity(
                            text=first_word,
                            label="PERSON",  # Default assumption
                            start_char=start,
                            end_char=start + len(first_word),
                            confidence=0.6,
                            metadata={
                                "extraction_method": "rules",
                                "rule": "sentence_start",
                            },
                        )
                    )

        char_offset += len(sentence) + 1

    return entities


def extract_entities_ml(
    text: str, model: str = "en_core_web_sm", **kwargs
) -> List[Entity]:
    """ML-based entity extraction using spaCy."""
    if not _common.SPACY_AVAILABLE:
        _common.logger.warning("spaCy not available, falling back to pattern extraction")
        return extract_entities_pattern(text, **kwargs)

    doc = scoring.run_spacy_text(
        model, text, fallback="en_core_web_sm", log_label="spaCy NER"
    )
    if doc is None:
        return extract_entities_pattern(text, **kwargs)
    entities = []

    for ent in doc.ents:
        # Standard spaCy spans expose no per-entity probability. Report the
        # absence explicitly (None) instead of fabricating certainty (1.0).
        confidence = None
        if hasattr(ent, "confidence"):
            confidence = ent.confidence
        elif hasattr(ent, "score"):
            confidence = ent.score

        entities.append(
            _common.Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=confidence,
                metadata={
                    "extraction_method": "ml",
                    "model": model,
                    "lemma": ent.lemma_ if hasattr(ent, "lemma_") else ent.text,
                    _common.CONFIDENCE_SOURCE_KEY: (
                        _common.CONFIDENCE_SOURCE_MODEL
                        if confidence is not None
                        else _common.CONFIDENCE_SOURCE_UNAVAILABLE
                    ),
                },
            )
        )

    return entities


def extract_entities_huggingface(
    text: str,
    model: str = "dslim/bert-base-NER",
    device: Optional[str] = None,
    **kwargs,
) -> List[Entity]:
    """
    Extract entities using HuggingFace transformers.
    
    Args:
        text: Input text
        model: Model name or path
        device: Device to use (cpu, cuda, mps)
        **kwargs: Additional arguments passed to the pipeline (e.g., aggregation_strategy)
    """
    loader = _common.HuggingFaceModelLoader(device=device)
    # Pass kwargs (like aggregation_strategy) to load_ner_model
    loader_kwargs = {
        key: value for key, value in kwargs.items() if key != "huggingface_model"
    }
    model_obj = loader.load_ner_model(model, **loader_kwargs)
    results = loader.extract_entities(model_obj, text)
    entities = []
    
    # Check if manual aggregation is needed (raw IOB tags detected)
    needs_manual_aggregation = False
    if results and isinstance(results[0], dict):
        first_label = results[0].get("label", "")
        # If we see B- tags and no entity_group (which implies aggregation wasn't done), we aggregate manually
        if (first_label.startswith("B-") or first_label.startswith("I-")) and "entity_group" not in results[0]:
            needs_manual_aggregation = True

    if needs_manual_aggregation:
        current_entity = None
        for result in results:
            label = result.get("label", "")
            word = result.get("word", result.get("entity", ""))
            score = result.get("score", 1.0)
            start = result.get("start", 0)
            end = result.get("end", 0)
            
            # Clean word (handle BERT ## and RoBERTa Ġ)
            clean_word = word.replace("##", "").replace("Ġ", "")
            if not clean_word:
                continue

            # Determine tag type and entity type
            tag_prefix = label[:2] if len(label) > 2 else ""
            entity_type = label[2:] if len(label) > 2 else label

            if tag_prefix == "B-":
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                current_entity = _common.Entity(
                    text=clean_word,
                    label=entity_type,
                    start_char=start,
                    end_char=end,
                    confidence=score,
                    metadata={
                        "model": model, 
                        "extraction_method": "huggingface",
                        "source": "huggingface",
                        "raw_iob": True
                    },
                )
            
            elif tag_prefix == "I-" and current_entity:
                # Check if type matches (loose check allows for some noise, strict check enforces type)
                # We'll be lenient and allow continuation if it makes sense contextually, 
                # but ideally types should match.
                if current_entity.label == entity_type:
                    # Append text
                    # Use offsets to determine spacing
                    if start > current_entity.end_char:
                        # If there's a gap, add space (unless it was a subword that got split but has gap? Unlikely)
                        # Usually gap means space.
                        # However, for ## subwords, start usually equals end.
                        # For Ġ, it implies space.
                        current_entity.text += " " + clean_word
                    else:
                        current_entity.text += clean_word
                    
                    current_entity.end_char = end
                    # Update confidence (average)
                    current_entity.confidence = (current_entity.confidence + score) / 2
                else:
                    # Type mismatch - treat as new entity or ignore?
                    # Treating as new B- is safer to avoid losing data
                    if current_entity:
                        entities.append(current_entity)
                    
                    current_entity = _common.Entity(
                        text=clean_word,
                        label=entity_type,
                        start_char=start,
                        end_char=end,
                        confidence=score,
                        metadata={
                            "model": model, 
                            "extraction_method": "huggingface",
                            "source": "huggingface",
                            "raw_iob": True
                        },
                    )
            
            else:
                # O tag or I- without B or other cases
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Append last entity
        if current_entity:
            entities.append(current_entity)

    else:
        # Standard processing for aggregated results or simple output
        for result in results:
            if isinstance(result, dict):
                # Handle different output formats based on aggregation strategy
                label = result.get("entity_group", result.get("label", "UNKNOWN"))
                text_content = result.get("word", result.get("entity", ""))
                
                # Clean up text content (remove ## for subwords if raw)
                if "##" in text_content and "aggregation_strategy" not in kwargs:
                     text_content = text_content.replace("##", "")
                if "Ġ" in text_content: # RoBERTa
                     text_content = text_content.replace("Ġ", " ").strip()
                     
                entities.append(
                    _common.Entity(
                        text=text_content,
                        label=label,
                        start_char=result.get("start", 0),
                        end_char=result.get("end", 0),
                        confidence=result.get("score", 1.0),
                        metadata={
                            "model": model, 
                            "extraction_method": "huggingface",
                            "source": "huggingface"
                        },
                    )
                )
            elif isinstance(result, list):
                 # Handle list of lists (sometimes returned by pipeline)
                 pass

    return entities


def extract_entities_llm(
    text: str,
    provider: str = "openai",
    model: Optional[str] = None,
    silent_fail: bool = False,
    max_text_length: Optional[int] = None,
    structured_output_mode: str = "typed",
    **kwargs,
) -> List[Entity]:
    """
    LLM-based entity extraction.
    
    Args:
        text: Input text
        provider: LLM provider
        model: LLM model
        silent_fail: If True, return empty list on error. If False (default), raise exception.
        max_text_length: Maximum text length before auto-chunking. None = provider default.
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
        "entity_types": kwargs.get("entity_types"),
        **_common._generation_cache_params(kwargs),
    }
    cached_result = _common._result_cache.get("entities", text, **cache_params)
    if cached_result is not None:
        _common.logger.debug(f"Cache hit for entity extraction ({len(cached_result)} entities)")
        return cached_result
    
    # 1. PRE-EXTRACTION VALIDATION
    if not text or not text.strip():
        error_msg = "Text is empty or whitespace only"
        _common.logger.error(error_msg)
        if not silent_fail:
            raise _common.ProcessingError(error_msg)
        return []

    # Pass api_key if provided in kwargs (needed for all providers)
    provider_kwargs = kwargs.copy()
    if "api_key" not in provider_kwargs or not provider_kwargs["api_key"]:
        # Try to get from environment as fallback for all providers
        import os
        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        if api_key:
            provider_kwargs["api_key"] = api_key

    # 2. PROVIDER VALIDATION
    try:
        llm = _common.create_provider(provider, model=model, **provider_kwargs)
        if not llm.is_available():
            error_msg = f"{provider} provider not available. Check API key and dependencies."
            _common.logger.error(error_msg)
            if not silent_fail:
                raise _common.ProcessingError(error_msg)
            return []
    except Exception as e:
        error_msg = f"Failed to create {provider} provider: {e}"
        _common.logger.error(error_msg)
        if not silent_fail:
            raise _common.ProcessingError(error_msg) from e
        return []

    # 3. TEXT LENGTH CHECK AND CHUNKING
    if max_text_length is None:
        # Provider-specific defaults
        max_text_length = {
            "groq": 64000,
            "openai": 64000,
            "gemini": 64000,
            "anthropic": 64000,
            "deepseek": 64000,
        }.get(provider.lower(), 32000)
    
    if len(text) > max_text_length:
        _common.logger.info(f"Text length ({len(text)}) exceeds limit ({max_text_length}). Chunking...")
        return _extract_entities_chunked(
            text, 
            provider=provider, 
            model=model, 
            silent_fail=silent_fail,
            max_text_length=max_text_length,
            **kwargs
        )

    # Use custom entity types if provided, otherwise use defaults
    entity_types = kwargs.get("entity_types")
    if entity_types:
        entity_types_str = ", ".join(entity_types)
        entity_types_instruction = f"""Preferred entity types: {entity_types_str}.
You may also use related or similar entity types if they better match the context (e.g., variations, synonyms, or domain-specific types).
If an entity doesn't fit any of the preferred types, use the most appropriate type from the preferred list or a closely related type."""
    else:
        entity_types_instruction = """Entity types should be one of: 
- PERSON (People, names, roles)
- ORG (Companies, organizations, institutions, brands)
- GPE (Countries, cities, states, locations)
- DATE (Dates, years, time periods)
- EVENT (Named events, conferences)
- PRODUCT (Software, hardware, vehicles)
- CONCEPT (Abstract ideas, technologies)

Use the most appropriate type for each entity.
Examples:
- 'Microsoft' is an ORG
- 'Satya Nadella' is a PERSON
- Job titles/roles like 'CEO', 'CTO', 'President', 'Engineer' are CONCEPT unless part of a person's name
- 'Python' is a PRODUCT or CONCEPT depending on context."""

    if not _common.SCHEMAS_AVAILABLE:
        raise ImportError("Pydantic schemas not available. Install pydantic/instructor to use LLM extraction.")

    try:
        prompt = f"""Extract named entities from the provided text.
Return the result as a JSON object with an "entities" key containing the list of entities.
Each entity should have 'text', 'label', and 'confidence' fields.

IMPORTANT: 
- Return a FLAT LIST of entities. 
- DO NOT group entities by type.
- The output structure must exactly match: {{ "entities": [ {{ "text": "...", "label": "...", "confidence": ... }}, ... ] }}

Example output (JSON format only):
{{
  "entities": [
    {{"text": "Entity Name", "label": "CATEGORY", "confidence": 0.95}},
    {{"text": "Another Entity", "label": "OTHER_CATEGORY", "confidence": 0.90}}
  ]
}}

Instructions:
1. Extract entities ONLY from the text provided below.
2. Do not include any entities from the example above.
3. {entity_types_instruction}

Text to extract from:
{text}"""
        
        # Use typed generation with Pydantic schema
        result_obj = llm.generate_typed(prompt, schema=_common.EntitiesResponse, **kwargs)
        
        # Convert back to internal Entity format
        entities = []
        for e_out in result_obj.entities:
            entities.append(_common.Entity(
                text=e_out.text,
                label=e_out.label,
                start_char=e_out.start if hasattr(e_out, "start") else 0, # Schema might not force these
                end_char=e_out.end if hasattr(e_out, "end") else 0,
                confidence=e_out.confidence,
                metadata={
                    "provider": provider, 
                    "model": model, 
                    "extraction_method": "llm_typed",
                }
            ))
        
        _common.logger.info(f"Successfully extracted {len(entities)} entities using {provider}/{model} (typed)")
        _common._result_cache.set("entities", text, entities, **cache_params)
        return entities
        
    except Exception as e:
        # Check for length/token limit errors
        error_msg_str = str(e).lower()
        if "length" in error_msg_str or "max_tokens" in error_msg_str:
            _common.logger.warning(f"LLM output truncated due to length limit. Reducing chunk size and retrying... ({e})")
            
            # Determine new chunk size (halve it)
            current_max = max_text_length or len(text)
            new_max = current_max // 2
            
            if new_max > 100: # Minimum viable chunk size check
                return _extract_entities_chunked(
                    text, 
                    provider=provider, 
                    model=model, 
                    silent_fail=silent_fail, 
                    max_text_length=new_max,
                    **kwargs
                )

        error_msg = f"LLM entity extraction failed ({provider}/{model}): {e}"
        _common.logger.error(error_msg, exc_info=True)
        if not silent_fail:
            if isinstance(e, _common.ProcessingError):
                raise
            raise _common.ProcessingError(error_msg) from e
        return []


def _extract_entities_chunked(
    text: str,
    provider: str,
    model: Optional[str],
    silent_fail: bool,
    max_text_length: int,
    structured_output_mode: str = "typed",
    **kwargs
) -> List[Entity]:
    """Internal helper to extract entities from long text by chunking."""
    from ...split import TextSplitter
    
    splitter = TextSplitter(
        method="recursive",
        chunk_size=max_text_length,
        chunk_overlap=int(max_text_length * 0.1) # 10% overlap
    )
    chunks = splitter.split(text)
    
    all_entities = []
    
    from ..config import resolve_max_workers
    max_workers = resolve_max_workers(explicit=kwargs.get("max_workers"))
    
    with _common.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {}
        for i, chunk in enumerate(chunks):
            _common.logger.debug(f"Scheduling entity extraction for chunk {i+1}/{len(chunks)}")
            # We recursively call extract_entities_llm with the chunk
            # but ensure we don't trigger re-chunking by setting max_text_length large
            future = executor.submit(
                extract_entities_llm,
                chunk.text,
                provider=provider,
                model=model,
                silent_fail=False, # We want to know if a chunk fails
                max_text_length=len(chunk.text) + 1,
                structured_output_mode=structured_output_mode,
                **kwargs
            )
            future_to_chunk[future] = (i, chunk)
            
        for future in _common.as_completed(future_to_chunk):
            i, chunk = future_to_chunk[future]
            try:
                chunk_entities = future.result()
                
                # Adjust entity positions to account for chunk offset
                for entity in chunk_entities:
                    entity.start_char += chunk.start_index
                    entity.end_char += chunk.start_index
                    all_entities.append(entity)
                    
            except Exception as e:
                if not silent_fail:
                    _common.logger.error(f"Chunk {i+1} failed: {e}")
                    raise
                _common.logger.warning(f"Chunk {i+1} failed (silent): {e}")

    return all_entities
