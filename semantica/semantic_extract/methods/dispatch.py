"""semantic_extract_methods: dispatch (split from methods.py)."""

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



# Try to import spaCy


from . import _common
from . import entities
from . import relations
from . import triplets


# ============================================================================
# Method Dispatchers
# ============================================================================


def get_entity_method(method_name: str):
    """Get entity extraction method - checks registry for custom methods."""
    # Check registry first
    custom_method = _common.method_registry.get("entity", method_name)
    if custom_method:
        return custom_method

    # Built-in methods
    builtin = {
        "pattern": entities.extract_entities_pattern,
        "regex": entities.extract_entities_regex,
        "rules": entities.extract_entities_rules,
        "ml": entities.extract_entities_ml,
        "spacy": entities.extract_entities_ml,  # Alias for ml
        "huggingface": entities.extract_entities_huggingface,
        "llm": entities.extract_entities_llm,
    }

    method_func = builtin.get(method_name)
    if not method_func:
        raise ValueError(
            f"Unknown method: {method_name}. Register custom method or use built-in: {list(builtin.keys())}"
        )

    return method_func


def get_relation_method(method_name: str):
    """Get relation extraction method - checks registry for custom methods."""
    # Check registry first
    custom_method = _common.method_registry.get("relation", method_name)
    if custom_method:
        return custom_method

    # Built-in methods
    builtin = {
        "pattern": relations.extract_relations_pattern,
        "regex": relations.extract_relations_regex,
        "cooccurrence": relations.extract_relations_cooccurrence,
        "similarity": relations.extract_relations_similarity,
        "dependency": relations.extract_relations_dependency,
        "ml": relations.extract_relations_dependency,  # Alias for dependency
        "spacy": relations.extract_relations_dependency,  # Alias for dependency
        "huggingface": relations.extract_relations_huggingface,
        "llm": relations.extract_relations_llm,
    }

    method_func = builtin.get(method_name)
    if not method_func:
        raise ValueError(
            f"Unknown method: {method_name}. Register custom method or use built-in: {list(builtin.keys())}"
        )

    return method_func


def get_triplet_method(method_name: str):
    """Get triplet extraction method - checks registry for custom methods."""
    # Check registry first
    custom_method = _common.method_registry.get("triplet", method_name)
    if custom_method:
        return custom_method

    # Built-in methods
    builtin = {
        "pattern": triplets.extract_triplets_pattern,
        "ml": triplets.extract_triplets_ml,  # ML (spaCy dependency) relations
        "rules": triplets.extract_triplets_rules,
        "huggingface": triplets.extract_triplets_huggingface,
        "llm": triplets.extract_triplets_llm,
    }

    method_func = builtin.get(method_name)
    if not method_func:
        raise ValueError(
            f"Unknown method: {method_name}. Register custom method or use built-in: {list(builtin.keys())}"
        )

    return method_func
