"""Shared schemas, helpers and constants for the ontology routes."""


import os
import asyncio
import ipaddress
import logging
import socket
import uuid
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from typing_extensions import Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from ...session import GraphSession
from ...utils.rdf_parser import _safe_parse_rdf

from . import registry

"""
Ontology Hub routes: registry, URL/file loading, preview, creation, entity search, and SKOS.
"""


try:
    from rdflib.namespace import DCT, DC
except ImportError:
    # Fallback if DCT/DC not available
    DCT = None
    DC = None


logger = logging.getLogger(__name__)


_MAX_FETCH_BYTES = 20 * 1024 * 1024  # 20 MB


_MAX_ANALYSIS_NODES = 5_000   # cap for health/suggest/shacl node scans to avoid OOM


_MAX_ENTITIES_PER_SIDE = 500  # per-ontology cap for the O(n²) pairwise suggestion loop


_GRAPH_TOO_LARGE_DETAIL = (
    "Ontology editor graph exceeds the maximum size "
    f"({_MAX_ANALYSIS_NODES} nodes or edges)."
)


class GraphTruncationError(Exception):
    """Raised when a graph exceeds _MAX_ANALYSIS_NODES so analysis would be truncated."""
    pass


_CLASS_TYPES = frozenset({
    "owl:Class", "rdfs:Class",
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2000/01/rdf-schema#Class",
})


_PROPERTY_TYPES = frozenset({
    "owl:ObjectProperty", "owl:DatatypeProperty", "owl:AnnotationProperty",
    "rdfs:Property",
    "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#AnnotationProperty",
})


_INDIVIDUAL_TYPES = frozenset({
    "owl:NamedIndividual",
    "http://www.w3.org/2002/07/owl#NamedIndividual",
})


_CONCEPT_TYPES = frozenset({
    "skos:Concept",
    "http://www.w3.org/2004/02/skos/core#Concept",
})


_SCHEME_TYPES = frozenset({
    "skos:ConceptScheme",
    "http://www.w3.org/2004/02/skos/core#ConceptScheme",
})


_ONTOLOGY_TYPES = frozenset({
    "owl:Ontology",
    "http://www.w3.org/2002/07/owl#Ontology",
}) | _SCHEME_TYPES


_SEARCHABLE_TYPES = _CLASS_TYPES | _PROPERTY_TYPES | _INDIVIDUAL_TYPES | _CONCEPT_TYPES | _SCHEME_TYPES


_SCHEMA_NODE_TYPES = _CLASS_TYPES | _PROPERTY_TYPES | _CONCEPT_TYPES | _ONTOLOGY_TYPES


_STRUCTURE_EDGE_TYPES = frozenset({
    "rdf:type",
    "rdfs:subClassOf",
    "rdfs:domain",
    "rdfs:range",
    "owl:disjointWith",
    "owl:equivalentClass",
    "owl:equivalentProperty",
    "owl:inverseOf",
    "skos:broader",
    "skos:narrower",
    "skos:related",
})


_URI_PREFIX_MAP = {
    "http://www.w3.org/2002/07/owl#": "owl:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2004/02/skos/core#": "skos:",
    "http://purl.org/dc/terms/": "dcterms:",
    "http://purl.org/dc/elements/1.1/": "dc:",
    "http://schema.org/": "schema:",
    "http://www.w3.org/ns/shacl#": "sh:",
}


_FORMAT_ALIASES: Dict[str, str] = {
    "ttl": "turtle",
    "rdf": "xml",
    "owl": "xml",
    "jsonld": "json-ld",
    "json": "json-ld",
}


_ALIGNMENT_RELATIONS: Dict[str, str] = {
    "owl:equivalentClass": "http://www.w3.org/2002/07/owl#equivalentClass",
    "owl:equivalentProperty": "http://www.w3.org/2002/07/owl#equivalentProperty",
    "skos:exactMatch": "http://www.w3.org/2004/02/skos/core#exactMatch",
    "skos:closeMatch": "http://www.w3.org/2004/02/skos/core#closeMatch",
    "skos:broadMatch": "http://www.w3.org/2004/02/skos/core#broadMatch",
    "skos:narrowMatch": "http://www.w3.org/2004/02/skos/core#narrowMatch",
    "skos:relatedMatch": "http://www.w3.org/2004/02/skos/core#relatedMatch",
}


_INGEST_FORMAT_SUFFIXES: Dict[str, str] = {
    "turtle": ".ttl",
    "xml": ".rdf",
    "json-ld": ".jsonld",
    "nt": ".nt",
    "n3": ".n3",
    "trig": ".trig",
    "nquads": ".nq",
}


# ---------------------------------------------------------------------------
# SHACL Validation Resource Guardrails
# ---------------------------------------------------------------------------
_MAX_SHACL_TURTLE_BYTES: int = int(
    os.environ.get("SEMANTICA_MAX_SHACL_TURTLE_BYTES", "262144")
)  # 256 KB


_MAX_SHACL_TRIPLES: int = int(
    os.environ.get("SEMANTICA_MAX_SHACL_TRIPLES", "1000")
)  # 1,000 triples


_MAX_SHACL_TIMEOUT_SECONDS: float = float(
    os.environ.get("SEMANTICA_MAX_SHACL_TIMEOUT", "15.0")
)


_MAX_SHACL_CONCURRENCY: int = int(
    os.environ.get("SEMANTICA_MAX_SHACL_CONCURRENCY", "4")
)


_shacl_validation_semaphore: Optional[
    Tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]
] = None


def _get_shacl_semaphore() -> asyncio.Semaphore:
    global _shacl_validation_semaphore
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    if (
        _shacl_validation_semaphore is None
        or _shacl_validation_semaphore[0] != current_loop
    ):
        _shacl_validation_semaphore = (
            current_loop,
            asyncio.Semaphore(_MAX_SHACL_CONCURRENCY),
        )
    return _shacl_validation_semaphore[1]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OntologyEntry(BaseModel):
    uri: str
    name: str
    description: Optional[str] = None
    format: str = "unknown"
    status: Literal["published", "draft", "external"] = "external"
    source_url: Optional[str] = None
    version: Optional[str] = None
    class_count: int = 0
    concept_count: int = 0
    property_count: int = 0
    loaded_at: str = ""
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)


class OntologyPreview(BaseModel):
    uri: str
    name: str
    description: Optional[str] = None
    namespace: Optional[str] = None
    version: Optional[str] = None
    license: Optional[str] = None
    format: str
    estimated_triples: int = 0
    source_url: Optional[str] = None


class LoadOntologyRequest(BaseModel):
    url: Optional[str] = None
    content: Optional[str] = None
    format: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class PreviewOntologyRequest(BaseModel):
    url: Optional[str] = None
    content: Optional[str] = None
    format: Optional[str] = None


class CreateOntologyRequest(BaseModel):
    mode: Literal["scratch", "data", "text"] = "scratch"
    namespace: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    sample_data: Optional[str] = None
    schema_text: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class OntologySearchResult(BaseModel):
    uri: str
    label: str
    type: str
    entity_type: str
    definition: Optional[str] = None
    source_ontology: Optional[str] = None
    namespace_prefix: Optional[str] = None


class EntityDetailResponse(BaseModel):
    uri: str
    label: str
    type: str
    entity_type: str
    definition: Optional[str] = None
    source_ontology: Optional[str] = None
    owning_ontology: Optional[str] = None
    superclasses: List[str] = Field(default_factory=list)
    subclasses: List[str] = Field(default_factory=list)
    domain: List[str] = Field(default_factory=list)
    range: List[str] = Field(default_factory=list)
    instance_count: int = 0
    properties: Dict[str, Any] = Field(default_factory=dict)


class OntologyGraphResponse(BaseModel):
    uri: str
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class SKOSScheme(BaseModel):
    uri: str
    title: str
    description: Optional[str] = None
    concept_count: int = 0


class SKOSConceptDetail(BaseModel):
    uri: str
    pref_label: str
    alt_labels: List[str] = Field(default_factory=list)
    hidden_labels: List[str] = Field(default_factory=list)
    definition: Optional[str] = None
    scope_note: Optional[str] = None
    editorial_note: Optional[str] = None
    broader: List[str] = Field(default_factory=list)
    narrower: List[str] = Field(default_factory=list)
    related: List[str] = Field(default_factory=list)
    exact_match: List[str] = Field(default_factory=list)
    close_match: List[str] = Field(default_factory=list)
    broad_match: List[str] = Field(default_factory=list)
    narrow_match: List[str] = Field(default_factory=list)
    scheme_uri: Optional[str] = None


class SKOSConceptSearchRequest(BaseModel):
    query: str
    scheme_uri: Optional[str] = None


class LoadOntologyResponse(BaseModel):
    status: str = "success"
    uri: str
    name: str
    nodes_added: int = 0
    edges_added: int = 0
    format: str = "unknown"


class ToggleResponse(BaseModel):
    uri: str
    enabled: bool


class RefreshResponse(BaseModel):
    status: str = "success"
    uri: str
    nodes_added: int = 0
    edges_added: int = 0


AlignmentRelation = Literal[
    "owl:equivalentClass",
    "owl:equivalentProperty",
    "skos:exactMatch",
    "skos:closeMatch",
    "skos:broadMatch",
    "skos:narrowMatch",
    "skos:relatedMatch",
]


class OntologyAlignment(BaseModel):
    id: str
    source_uri: str
    source_label: str = ""
    target_uri: str
    target_label: str = ""
    relation: AlignmentRelation
    predicate_uri: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Optional[str] = None
    source: Optional[str] = None
    reviewer: Optional[str] = None
    created_at: str
    updated_at: str


class OntologyAlignmentRequest(BaseModel):
    source_uri: str
    source_label: Optional[str] = None  # override for external/unloaded URIs
    target_uri: str
    target_label: Optional[str] = None  # override for external/unloaded URIs
    relation: AlignmentRelation
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Optional[str] = None
    source: Optional[str] = None
    reviewer: Optional[str] = None


class AlignmentSuggestionRequest(BaseModel):
    source_ontology_uri: Optional[str] = None
    target_ontology_uri: Optional[str] = None
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    limit: int = Field(default=25, ge=1, le=100)


class AlignmentSuggestion(BaseModel):
    source_uri: str
    source_label: str
    target_uri: str
    target_label: str
    relation: AlignmentRelation
    score: float
    label_similarity: float
    embedding_similarity: Optional[float] = None
    reason: str


class HealthDimension(BaseModel):
    key: str
    label: str
    score: float = Field(ge=0.0, le=100.0)
    status: Literal["ok", "warning", "critical", "unavailable"] = "ok"
    detail: str


class HealthIssue(BaseModel):
    id: str
    severity: Literal["info", "warning", "critical"] = "warning"
    category: str
    entity_uri: Optional[str] = None
    entity_label: Optional[str] = None
    message: str
    action: Optional[str] = None


class OntologyHealthResponse(BaseModel):
    uri: str
    name: str
    total_score: float = Field(ge=0.0, le=100.0)
    dimensions: List[HealthDimension]
    issues: List[HealthIssue] = Field(default_factory=list)
    generated_at: str


class ShaclGenerateRequest(BaseModel):
    uri: str
    quality_tier: Literal["standard", "strict"] = "strict"


class ShaclValidateRequest(BaseModel):
    uri: Optional[str] = None
    shacl_turtle: str


class ShaclViolation(BaseModel):
    node: Optional[str] = None
    path: Optional[str] = None
    severity: str = "Violation"
    message: str
    focus_node: Optional[str] = None
    source_shape: Optional[str] = None


class ShaclShapeSummary(BaseModel):
    id: str
    target_class: Optional[str] = None
    constraint_count: int = 0
    constraints: List[str] = Field(default_factory=list)
    violation_count: int = 0


class ShaclGenerateResponse(BaseModel):
    uri: str
    shacl_turtle: str
    shape_count: int
    generated_at: str


class ShaclShapesResponse(BaseModel):
    uri: str
    shapes: List[ShaclShapeSummary]
    shacl_turtle: str
    generated_at: str


class ShaclValidationResponse(BaseModel):
    uri: Optional[str] = None
    conforms: bool
    status: Literal["success", "unavailable", "error"] = "success"
    message: str
    violations: List[ShaclViolation] = Field(default_factory=list)
    report_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Draft, Proposal, and Version Schemas
# ---------------------------------------------------------------------------

class DraftDiff(BaseModel):
    added_classes: List[str] = Field(default_factory=list)
    removed_classes: List[str] = Field(default_factory=list)
    modified_classes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    added_properties: List[str] = Field(default_factory=list)
    removed_properties: List[str] = Field(default_factory=list)
    modified_properties: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    added_restrictions: List[Dict[str, Any]] = Field(default_factory=list)
    removed_restrictions: List[Dict[str, Any]] = Field(default_factory=list)
    added_axioms: List[Dict[str, Any]] = Field(default_factory=list)
    removed_axioms: List[Dict[str, Any]] = Field(default_factory=list)
    annotation_changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class DraftRequest(BaseModel):
    ontology_uri: str
    diff: DraftDiff
    author: str
    summary: Optional[str] = None
    timestamp: Optional[str] = None


class DraftResponse(BaseModel):
    draft_id: str
    ontology_uri: str
    diff: DraftDiff
    author: str
    summary: Optional[str] = None
    created_at: str
    updated_at: str


class ProposalRequest(BaseModel):
    draft_id: str
    ontology_uri: str
    summary: str
    reviewer: Optional[str] = None


class ProposalResponse(BaseModel):
    proposal_id: str
    draft_id: str
    ontology_uri: str
    summary: str
    author: str
    reviewer: Optional[str] = None
    state: Literal["draft", "proposed", "approved", "published", "rejected"]
    impact_analysis: Dict[str, Any] = Field(default_factory=dict)
    shacl_validation: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    comments: List[Dict[str, Any]] = Field(default_factory=list)


class CommentRequest(BaseModel):
    element_uri: str
    text: str
    author: str


class VersionEntry(BaseModel):
    version_id: str
    ontology_uri: str
    state: Literal["draft", "published"]
    author: str
    date: str
    diff_summary: Dict[str, Any] = Field(default_factory=dict)


class VersionCompareRequest(BaseModel):
    version1: str
    version2: str


class VersionCompareResponse(BaseModel):
    version1: str
    version2: str
    metadata_changes: Dict[str, Any] = Field(default_factory=dict)
    class_changes: Dict[str, Any] = Field(default_factory=dict)
    property_changes: Dict[str, Any] = Field(default_factory=dict)
    restriction_changes: Dict[str, Any] = Field(default_factory=dict)
    axiom_changes: Dict[str, Any] = Field(default_factory=dict)


class AlignmentRequest(BaseModel):
    source_uri: str
    target_uri: str
    predicate: str


class AlignmentResponse(BaseModel):
    source: str
    predicate: str
    target: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_registry(request: Request) -> Dict[str, OntologyEntry]:
    if not hasattr(request.app.state, "ontology_registry"):
        request.app.state.ontology_registry = {}
    return request.app.state.ontology_registry


def _get_alignment_store(request: Request) -> Dict[str, OntologyAlignment]:
    if not hasattr(request.app.state, "ontology_alignments"):
        request.app.state.ontology_alignments = {}
    return request.app.state.ontology_alignments


def _get_drafts(request: Request) -> Dict[str, DraftResponse]:
    if not hasattr(request.app.state, "ontology_drafts"):
        request.app.state.ontology_drafts = {}
    return request.app.state.ontology_drafts


def _get_proposals(request: Request) -> Dict[str, ProposalResponse]:
    if not hasattr(request.app.state, "ontology_proposals"):
        request.app.state.ontology_proposals = {}
    return request.app.state.ontology_proposals


def _get_versions(request: Request) -> Dict[str, List[VersionEntry]]:
    if not hasattr(request.app.state, "ontology_versions"):
        request.app.state.ontology_versions = {}
    return request.app.state.ontology_versions


def _alignment_key(source: str, predicate: str, target: str) -> str:
    return f"{source}::{predicate}::{target}"


def _coerce_alignment(raw: Any) -> Optional[AlignmentResponse]:
    if isinstance(raw, AlignmentResponse):
        return raw
    if isinstance(raw, dict):
        source = raw.get("source") or raw.get("source_uri")
        target = raw.get("target") or raw.get("target_uri")
        predicate = raw.get("predicate") or raw.get("relation")
        if source and target and predicate:
            return AlignmentResponse(source=source, predicate=predicate, target=target)
    return None


def _version_field(version: Any, field: str, default: Any = None) -> Any:
    if isinstance(version, VersionEntry):
        return getattr(version, field)
    if isinstance(version, dict):
        return version.get(field, default)
    return default


def _uri_to_prefix(uri: str) -> str:
    for base, prefix in _URI_PREFIX_MAP.items():
        if uri.startswith(base):
            return prefix + uri[len(base):]
    return uri


def _classify_node_type(node_type: str) -> str:
    if node_type in _CLASS_TYPES:
        return "class"
    if node_type in _PROPERTY_TYPES:
        return "property"
    if node_type in _INDIVIDUAL_TYPES:
        return "individual"
    if node_type in _CONCEPT_TYPES:
        return "concept"
    if node_type in _SCHEME_TYPES:
        return "scheme"
    if node_type in _ONTOLOGY_TYPES:
        return "ontology"
    return "unknown"


def _node_label(node: Dict[str, Any]) -> str:
    props = node.get("properties", {})
    return (
        props.get("pref_label")
        or props.get("rdfs:label")
        or props.get("skos:prefLabel")
        or props.get("label")
        or props.get("content")
        or node.get("content", "")
        or node.get("id", "")
    )


def _as_uri_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if item is None or item == "":
                continue
            if isinstance(item, dict):
                uri = item.get("uri") or item.get("id") or item.get("@id")
                if uri:
                    result.append(str(uri))
            else:
                result.append(str(item))
        return result
    if isinstance(value, dict):
        uri = value.get("uri") or value.get("id") or value.get("@id")
        return [str(uri)] if uri else []
    return [str(value)]


def _convert_ontology_to_graph(ontology_dict: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert OntologyIngestor ontology dict to graph nodes and edges."""
    nodes = []
    edges = []
    
    # Add ontology node
    ontology_uri = ontology_dict.get("uri", f"temp:{uuid.uuid4().hex[:12]}")
    nodes.append({
        "id": ontology_uri,
        "type": "owl:Ontology",
        "content": ontology_dict.get("name", "Ontology"),
        "properties": {
            "rdfs:label": ontology_dict.get("name", "Ontology"),
            "rdfs:comment": ontology_dict.get("description", ""),
            "uri": ontology_uri,
        },
    })
    
    # Add class nodes
    for cls in ontology_dict.get("classes", []):
        cls_uri = cls.get("uri", f"temp:class:{uuid.uuid4().hex[:12]}")
        node = {
            "id": cls_uri,
            "type": "owl:Class",
            "content": cls.get("name", cls.get("label", "")),
            "properties": {
                "rdfs:label": cls.get("label", cls.get("name", "")),
                "rdfs:comment": cls.get("description", ""),
                "uri": cls_uri,
                "scheme_uri": ontology_uri,
            },
        }
        nodes.append(node)
        
        # Add subclass edges
        for parent in cls.get("parents", []):
            edges.append({
                "source": cls_uri,
                "target": parent,
                "type": "rdfs:subClassOf",
                "weight": 1.0,
            })
    
    # Add property nodes and edges
    for prop in ontology_dict.get("properties", []):
        prop_uri = prop.get("uri", f"temp:prop:{uuid.uuid4().hex[:12]}")
        property_type = {
            "object": "owl:ObjectProperty",
            "data": "owl:DatatypeProperty",
            "datatype": "owl:DatatypeProperty",
            "annotation": "owl:AnnotationProperty",
        }.get(str(prop.get("type", "object")).lower(), "owl:ObjectProperty")
        node = {
            "id": prop_uri,
            "type": property_type,
            "content": prop.get("name", prop.get("label", "")),
            "properties": {
                "rdfs:label": prop.get("label", prop.get("name", "")),
                "rdfs:comment": prop.get("description", ""),
                "uri": prop_uri,
                "scheme_uri": ontology_uri,
            },
        }
        nodes.append(node)
        
        # Add domain and range edges
        for domain_uri in _as_uri_list(prop.get("domain")):
            edges.append({
                "source": prop_uri,
                "target": domain_uri,
                "type": "rdfs:domain",
                "weight": 1.0,
            })
        for range_uri in _as_uri_list(prop.get("range")):
            edges.append({
                "source": prop_uri,
                "target": range_uri,
                "type": "rdfs:range",
                "weight": 1.0,
            })
    
    return nodes, edges


def _extract_namespace(uri: str) -> Optional[str]:
    if "#" in uri:
        return uri.rsplit("#", 1)[0] + "#"
    if "/" in uri:
        return uri.rsplit("/", 1)[0] + "/"
    return None


def _ontology_namespace(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[0] + "#"
    if uri.endswith("/"):
        return uri
    return uri.rstrip("#/") + "#"


def _alignment_id(source_uri: str, relation: str, target_uri: str) -> str:
    key = f"{source_uri}|{relation}|{target_uri}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, key))


def _label_from_uri(uri: str) -> str:
    """Derive a readable label from a URI when no graph node is present (e.g. external vocabularies)."""
    fragment = uri.rsplit("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
    return fragment or uri


def _node_source_ontology(node: Dict[str, Any]) -> Optional[str]:
    props = node.get("properties", {})
    return (
        props.get("scheme_uri")
        or props.get("source_ontology")
        or props.get("ontology_uri")
        or props.get("ontology")
    )


def _node_belongs_to_ontology(
    node: Dict[str, Any],
    ontology_uri: str,
    known_ontology_uris: Optional[set[str]] = None,
) -> bool:
    nid = node.get("id", "")
    if nid == ontology_uri:
        return True
    owner = _node_source_ontology(node)
    if owner:
        return owner == ontology_uri
    if known_ontology_uris:
        namespace_owners = [
            candidate
            for candidate in known_ontology_uris
            if nid == candidate
            or nid.startswith(
                (candidate.rstrip("#/") + "#", candidate.rstrip("#/") + "/")
            )
        ]
        if namespace_owners and max(namespace_owners, key=len) != ontology_uri:
            return False
    stem = ontology_uri.rstrip("#/")
    if not nid.startswith((stem + "#", stem + "/")):
        return False
    # Prefix ownership only extends to names minted directly in the
    # ontology's namespace (<stem>#Term or <stem>/Term). Any further
    # delimiter marks a nested vocabulary (<stem>/child#Term,
    # <stem>/child/Term), which must not be absorbed into the parent
    # until it is registered or carries an explicit owner.
    local_name = nid[len(stem) + 1 :]
    return "#" not in local_name and "/" not in local_name


def _resolve_owning_ontology(
    node: Dict[str, Any],
    known_ontology_uris: set[str],
) -> Optional[str]:
    """Return the one known ontology that owns this node, or None if none does.

    The most specific (longest) match wins, so a nested vocabulary claims its
    own terms instead of the parent absorbing them.

    Kept agreeing with _node_belongs_to_ontology by construction — the same
    three rules in the same order — but in one pass over the candidates rather
    than one pass per candidate, each of which rescanned the whole set to find
    the longest namespace. That made resolution quadratic in the number of
    registered ontologies.
    """
    nid = str(node.get("id", ""))
    if not nid:
        return None

    # An ontology node owns itself, ahead of any scheme_uri it may carry.
    if nid in known_ontology_uris:
        return nid

    # An explicit owner is authoritative even when it is not registered:
    # naming a different ontology by namespace guess would be worse than
    # reporting the one the node itself points at.
    explicit_owner = _node_source_ontology(node)
    if explicit_owner:
        return explicit_owner

    longest_namespace: Optional[str] = None
    for candidate in known_ontology_uris:
        stem = candidate.rstrip("#/")
        if not nid.startswith((stem + "#", stem + "/")):
            continue
        if longest_namespace is None or len(candidate) > len(longest_namespace):
            longest_namespace = candidate
    if longest_namespace is None:
        return None

    # Prefix ownership only extends to names minted directly in the namespace.
    # A further delimiter marks a nested vocabulary, which stays unowned until
    # it is registered or carries an explicit owner.
    local_name = nid[len(longest_namespace.rstrip("#/")) + 1 :]
    if "#" in local_name or "/" in local_name:
        return None
    return longest_namespace


def _is_ontology_entity(node: Dict[str, Any]) -> bool:
    return _classify_node_type(node.get("type", "")) in {"class", "property", "concept", "scheme"}


def _entity_description(node: Dict[str, Any]) -> Optional[str]:
    props = node.get("properties", {})
    return (
        props.get("rdfs:comment")
        or props.get("skos:definition")
        or props.get("definition")
        or props.get("description")
    )


def _entity_definition(node: Dict[str, Any]) -> Optional[str]:
    props = node.get("properties", {})
    return props.get("skos:definition") or props.get("definition")


def _label_similarity(left: str, right: str) -> float:
    left_norm = " ".join(left.lower().replace("_", " ").replace("-", " ").split())
    right_norm = " ".join(right.lower().replace("_", " ").replace("-", " ").split())
    if not left_norm or not right_norm:
        return 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    token_score = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return round(max(sequence, token_score), 4)


def _token_set(label: str) -> frozenset:
    return frozenset(label.lower().replace("_", " ").replace("-", " ").split())


def _tfidf_embedding_vectors(labels: List[str]) -> Optional[Dict[str, List[float]]]:
    """Build character-ngram TF-IDF vectors from entity labels (sklearn-based, no gensim needed)."""
    if len(labels) < 2:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        normed = [" ".join(l.lower().replace("_", " ").replace("-", " ").split()) for l in labels]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        matrix = vec.fit_transform(normed)
        return {label: matrix[i].toarray().flatten().tolist() for i, label in enumerate(labels)}
    except Exception:
        return None


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    try:
        from ....kg import SimilarityCalculator
        return SimilarityCalculator().cosine_similarity(v1, v2)
    except Exception:
        import math
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        return dot / (n1 * n2) if n1 and n2 else 0.0


def _candidate_relation(source: Dict[str, Any], target: Dict[str, Any]) -> AlignmentRelation:
    source_type = _classify_node_type(source.get("type", ""))
    target_type = _classify_node_type(target.get("type", ""))
    if source_type == "class" and target_type == "class":
        return "owl:equivalentClass"
    if source_type == "property" and target_type == "property":
        return "owl:equivalentProperty"
    if source_type == "concept" and target_type == "concept":
        return "skos:exactMatch"
    return "skos:closeMatch"


def _ontology_entities(nodes: List[Dict[str, Any]], ontology_uri: Optional[str] = None) -> List[Dict[str, Any]]:
    result = []
    for node in nodes:
        if not _is_ontology_entity(node):
            continue
        if ontology_uri and not _node_belongs_to_ontology(node, ontology_uri):
            continue
        result.append(node)
    return result


def _data_graph_entities(nodes: List[Dict[str, Any]], ontology_uri: Optional[str] = None) -> List[Dict[str, Any]]:
    result = []
    for node in nodes:
        if _classify_node_type(node.get("type", "")) not in {"class", "property", "individual", "concept", "scheme"}:
            continue
        if ontology_uri and not _node_belongs_to_ontology(node, ontology_uri):
            continue
        result.append(node)
    return result


def _ontology_dict_from_nodes(uri: str, name: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    classes = []
    properties = []
    for node in nodes:
        entity_type = _classify_node_type(node.get("type", ""))
        label = _node_label(node) or node.get("id", "")
        item = {
            "name": label,
            "uri": node.get("id", ""),
            "label": label,
            "description": _entity_description(node) or "",
        }
        if entity_type == "class":
            classes.append(item)
        elif entity_type == "property":
            domain = [
                edge.get("target", "")
                for edge in edges
                if edge.get("source") == node.get("id") and edge.get("type") == "rdfs:domain"
            ]
            range_ = [
                edge.get("target", "")
                for edge in edges
                if edge.get("source") == node.get("id") and edge.get("type") == "rdfs:range"
            ]
            item.update({
                "type": "object" if "ObjectProperty" in node.get("type", "") else "datatype",
                "domain": domain,
                "range": range_,
                "required": False,
            })
            properties.append(item)

    return {
        "name": name,
        "namespace": _ontology_namespace(uri),
        "classes": classes,
        "properties": properties,
    }


def _basic_shacl_turtle(uri: str, name: str, nodes: List[Dict[str, Any]]) -> str:
    namespace = _ontology_namespace(uri)
    lines = [
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        f"@prefix onto: <{namespace}> .",
        "",
    ]
    class_nodes = [n for n in nodes if _classify_node_type(n.get("type", "")) == "class"]
    for index, node in enumerate(class_nodes or nodes[:1], start=1):
        label = (_node_label(node) or f"{name}Shape").replace(" ", "")
        target = node.get("id") or uri
        lines.extend([
            f"onto:{label}Shape a sh:NodeShape ;",
            f"  sh:targetClass <{target}> ;",
            "  sh:severity sh:Violation .",
            "",
        ])
    return "\n".join(lines)


def _summarize_shapes(shacl_turtle: str, violations: Optional[List[ShaclViolation]] = None) -> List[ShaclShapeSummary]:
    violations = violations or []
    normalised = shacl_turtle.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in normalised.split(".\n") if "sh:NodeShape" in block]
    summaries: List[ShaclShapeSummary] = []
    for index, block in enumerate(blocks, start=1):
        first = block.splitlines()[0].strip()
        shape_id = first.split()[0] if first else f"shape-{index}"
        target_class = None
        constraints: List[str] = []
        for token in ("sh:targetClass", "sh:path", "sh:minCount", "sh:maxCount", "sh:datatype", "sh:class"):
            if token in block:
                constraints.append(token.replace("sh:", ""))
        if "sh:targetClass" in block:
            after = block.split("sh:targetClass", 1)[1].strip()
            target_class = after.split()[0].strip(" ;")
        violation_count = sum(
            1
            for violation in violations
            if violation.source_shape == shape_id or (target_class and violation.node == target_class)
        )
        summaries.append(ShaclShapeSummary(
            id=shape_id,
            target_class=target_class,
            constraint_count=max(0, len(constraints) - 1),
            constraints=constraints,
            violation_count=violation_count,
        ))
    return summaries


async def _registry_entries(request: Request, session: GraphSession) -> List[OntologyEntry]:
    return await registry.list_registry(request=request, q=None, status=None, format=None, session=session)


def _detect_format(content: str) -> str:
    stripped = content.strip()[:500]
    if stripped.startswith("{") or stripped.startswith("["):
        return "json-ld"
    if stripped.startswith("<"):
        return "xml"
    if "@prefix" in stripped or "@base" in stripped:
        return "turtle"
    # N-Triples blank-node subject: "_:word <predicate-uri> ..."
    # URI-subject N-Triples ("<uri> <uri>") are already caught by the XML
    # branch above, so only the blank-node form needs to be checked here.
    # Plain string ops avoid the polynomial regex that CodeQL flags (py/polynomial-redos).
    if stripped.startswith("_:") and " <" in stripped:
        return "nt"
    return "turtle"


def _normalize_format(fmt: Optional[str]) -> str:
    if not fmt:
        return "turtle"
    lower = fmt.strip().lower()
    return _FORMAT_ALIASES.get(lower, lower)


def _validate_fetch_url(url: str) -> List[str]:
    """Reject non-HTTP(S) schemes and private/loopback/link-local targets.

    Returns every resolved, validated IP address (deduplicated, in
    resolution order) so the caller can pin the actual connection to them
    (see _make_pinned_session) with fallback across all of them — not just
    the first — since a hostname can have multiple A/AAAA records and the
    first one isn't guaranteed reachable. Resolving the hostname again at
    connect time would open a DNS check-then-use window (a low-TTL or
    rebinding DNS answer could differ between this check and the client's
    own lookup), which is what pinning to these specific addresses avoids.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=422, detail="Only http and https URLs are allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=422, detail="Invalid URL: missing hostname.")
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise HTTPException(status_code=422, detail=f"Cannot resolve hostname '{hostname}': {exc}") from exc
    validated_ips: List[str] = []
    for _family, _type, _proto, _canonname, sockaddr in addrinfos:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(
                status_code=422,
                detail="Fetching from private, loopback, or reserved network addresses is not allowed.",
            )
        if sockaddr[0] not in validated_ips:
            validated_ips.append(sockaddr[0])
    if not validated_ips:
        raise HTTPException(status_code=422, detail=f"Cannot resolve hostname '{hostname}' to a usable address.")
    return validated_ips


def _make_pinned_session(pinned_ips: List[str], url: str):
    """Build a requests.Session whose connection is pinned to pinned_ips
    (tried in order, falling back on connection failure), regardless of
    what url's hostname resolves to at connect time.

    _validate_fetch_url() resolves and validates the hostname once; letting
    the HTTP client resolve it again independently at connect time reopens
    the exact gap that validation exists to close — a low-TTL or rebinding
    DNS answer can differ between the two lookups. This pins the pool's
    connect target to the already-validated addresses directly (bypassing
    DNS resolution for the connection entirely), while keeping the original
    hostname as the outgoing HTTP Host header and, for HTTPS, the TLS SNI
    server_hostname / assert_hostname — otherwise the connection would
    reach the right IP but present the wrong identity, breaking name-based
    virtual hosting and (for HTTPS) certificate hostname verification.

    Falls back across every validated address (not just the first) so a
    hostname with multiple A/AAAA records doesn't fail outright just
    because the first-returned address happens to be unreachable.

    Note: urllib3's Connection.host is a property that reads/writes the
    same underlying value as `_dns_host` in this version — it is NOT the
    separate "presented identity" field it is in some older releases, so
    overriding just `_dns_host` post-construction (as an earlier version of
    this fix did) actually changes the Host header too. Pinning the pool's
    `host` directly and restoring the real hostname via an explicit Host
    header (+ SNI params for HTTPS) is the correct mechanism here.
    """
    import requests as _req
    import urllib3.util.connection as _u3_connection
    from urllib3.exceptions import NewConnectionError

    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port
    default_port = 443 if parsed.scheme == "https" else 80
    host_header = hostname if port in (None, default_port) else f"{hostname}:{port}"

    class _MultiIPConnectionMixin:
        """Overrides _new_conn to fall back across every pinned IP in
        order, instead of urllib3's default single-host connect."""

        def _new_conn(self):
            last_exc: Optional[BaseException] = None
            for ip in pinned_ips:
                try:
                    return _u3_connection.create_connection(
                        (ip, self.port),
                        self.timeout,
                        source_address=self.source_address,
                        socket_options=self.socket_options,
                    )
                except OSError as exc:
                    last_exc = exc
                    continue
            raise NewConnectionError(
                self, f"Failed to establish a connection to any of {pinned_ips}: {last_exc}"
            )

    class _PinnedIPHTTPAdapter(_req.adapters.HTTPAdapter):
        def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
            # A proxy would perform its own DNS resolution of the target
            # host on this process's behalf — a resolution outside this
            # process's visibility or control, so there is no client-side
            # pin that closes that race. Proxies are disabled outright for
            # this SSRF-sensitive fetcher (session.trust_env=False below),
            # so this should be unreachable via environment proxies; fail
            # closed rather than silently skip pinning if a proxy is
            # somehow still configured (e.g. passed explicitly in the
            # future). _validate_fetch_url's destination classification is
            # a separate, always-enforced check — this only guards the
            # secondary DNS-pinning hardening.
            if _req.utils.select_proxy(request.url, proxies):
                raise HTTPException(
                    status_code=502,
                    detail="Proxied requests are not supported for ontology URL fetching.",
                )
            host_params, pool_kwargs = self.build_connection_pool_key_attributes(request, verify, cert)
            if host_params.get("scheme") == "https":
                pool_kwargs.setdefault("assert_hostname", hostname)
                pool_kwargs.setdefault("server_hostname", hostname)
            host_params["host"] = pinned_ips[0]
            pool = self.poolmanager.connection_from_host(**host_params, pool_kwargs=pool_kwargs)
            base_connection_cls = pool.ConnectionCls
            if not issubclass(base_connection_cls, _MultiIPConnectionMixin):
                pool.ConnectionCls = type(
                    "_PinnedConnection", (_MultiIPConnectionMixin, base_connection_cls), {}
                )
            return pool

    session = _req.Session()
    # Never honor HTTP_PROXY/HTTPS_PROXY/NO_PROXY env vars for this
    # SSRF-sensitive fetcher: a configured proxy would perform its own DNS
    # resolution of the target host outside this process's control,
    # silently reopening the DNS check-then-use race pinning exists to
    # close. See _PinnedIPHTTPAdapter.get_connection_with_tls_context for
    # the fail-closed backstop if a proxy is somehow still configured.
    session.trust_env = False
    session.headers["Host"] = host_header
    adapter = _PinnedIPHTTPAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _fetch_url_sync(url: str) -> bytes:
    pinned_ips = _validate_fetch_url(url)
    _MAX_REDIRECTS = 5
    current_url = url
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            session = _make_pinned_session(pinned_ips, current_url)
            try:
                resp = session.get(
                    current_url,
                    headers={"Accept": "text/turtle, application/rdf+xml, application/ld+json, */*;q=0.1"},
                    timeout=30,
                    stream=True,
                    allow_redirects=False,  # SECURITY: follow redirects manually
                )
                if resp.is_redirect or resp.is_permanent_redirect:
                    redirect_url = resp.headers.get("Location")
                    resp.close()  # Release the streamed connection before following the redirect
                    if not redirect_url:
                        raise HTTPException(status_code=502, detail="Redirect without Location header.")
                    # Resolve relative redirects (e.g. /ontology.ttl) against the current URL
                    redirect_url = urljoin(current_url, redirect_url)
                    # Re-validate the redirect target to prevent SSRF via
                    # open-redirect to internal/cloud-metadata endpoints, and
                    # get fresh pins for the new host.
                    pinned_ips = _validate_fetch_url(redirect_url)
                    current_url = redirect_url
                    continue
                try:
                    resp.raise_for_status()
                    chunks: List[bytes] = []
                    total = 0
                    for chunk in resp.iter_content(65536):
                        total += len(chunk)
                        if total > _MAX_FETCH_BYTES:
                            raise HTTPException(status_code=413, detail="Remote resource exceeds 20 MB limit.")
                        chunks.append(chunk)
                    return b"".join(chunks)
                finally:
                    resp.close()  # Release the streamed connection once fully read (or on error)
            finally:
                session.close()
        raise HTTPException(status_code=502, detail=f"Too many redirects (max {_MAX_REDIRECTS}).")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch {url}: {exc}") from exc


def _parse_rdf_sync(content: bytes, fmt: str) -> tuple:
    """Return (nodes, edges, metadata). Raises HTTPException on failure."""
    try:
        import rdflib
    except ImportError:
        raise HTTPException(status_code=501, detail="rdflib is not installed.")

    fmt_map = {
        "turtle": "turtle", "xml": "xml", "nt": "nt",
        "json-ld": "json-ld", "n3": "n3",
    }
    parse_fmt = fmt_map.get(fmt, "turtle")

    g = rdflib.Graph()
    try:
        _safe_parse_rdf(g, content, parse_fmt)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"RDF parse error: {exc}") from exc

    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    RDF = rdflib.RDF
    RDFS = rdflib.RDFS
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    DCT = rdflib.Namespace("http://purl.org/dc/terms/")
    DC = rdflib.Namespace("http://purl.org/dc/elements/1.1/")

    metadata: Dict[str, Any] = {}

    for subj in g.subjects(RDF.type, OWL.Ontology):
        metadata["uri"] = str(subj)
        for pred, obj in g.predicate_objects(subj):
            p = str(pred)
            if p in {str(RDFS.label), str(DCT.title), str(DC.title)}:
                metadata.setdefault("name", str(obj))
            elif p in {str(RDFS.comment), str(DCT.description), str(DC.description)}:
                metadata.setdefault("description", str(obj))
            elif p == str(OWL.versionInfo):
                metadata.setdefault("version", str(obj))
            elif p in {str(DCT.license), str(DC.rights)}:
                metadata.setdefault("license", str(obj))
        break

    if "uri" not in metadata:
        for subj in g.subjects(RDF.type, SKOS.ConceptScheme):
            metadata["uri"] = str(subj)
            for pred, obj in g.predicate_objects(subj):
                p = str(pred)
                if p in {str(SKOS.prefLabel), str(DCT.title), str(DC.title)}:
                    metadata.setdefault("name", str(obj))
                elif p in {str(SKOS.definition), str(DCT.description)}:
                    metadata.setdefault("description", str(obj))
            break

    synthetic_uri = "uri" not in metadata
    if synthetic_uri:
        metadata["uri"] = f"urn:semantica:onto:{uuid.uuid4().hex[:8]}"
    metadata.setdefault("name", metadata["uri"].rsplit("/", 1)[-1].rsplit("#", 1)[-1] or "Unnamed")
    metadata["triple_count"] = len(g)

    # Collect literal properties per subject
    literal_props: Dict[str, Dict[str, str]] = {}
    for subj, pred, obj in g:
        if isinstance(subj, rdflib.BNode) or not isinstance(obj, rdflib.Literal):
            continue
        sid = str(subj)
        pk = _uri_to_prefix(str(pred))
        literal_props.setdefault(sid, {})[pk] = str(obj)

    # Build nodes from rdf:type statements
    seen_ids: set = set()
    nodes: List[Dict[str, Any]] = []
    for subj, _, type_obj in g.triples((None, RDF.type, None)):
        if isinstance(subj, rdflib.BNode):
            continue
        sid = str(subj)
        ntype = _uri_to_prefix(str(type_obj))
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        props = dict(literal_props.get(sid, {}))
        props["uri"] = sid
        label = (
            props.get("rdfs:label")
            or props.get("skos:prefLabel")
            or props.get("dcterms:title")
            or sid.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        )
        nodes.append({"id": sid, "type": ntype, "content": label, "properties": props})

    # Build edges from non-literal object statements
    edges: List[Dict[str, Any]] = []
    for subj, pred, obj in g:
        if isinstance(subj, rdflib.BNode) or isinstance(obj, (rdflib.Literal, rdflib.BNode)):
            continue
        edges.append({
            "source": str(subj),
            "target": str(obj),
            "type": _uri_to_prefix(str(pred)),
            "weight": 1.0,
        })

    if synthetic_uri:
        # No owl:Ontology / skos:ConceptScheme declaration exists, so the
        # synthetic registry URI shares no namespace with any node. Ownership
        # must be recorded explicitly, and the editor needs a matching graph
        # node, or the registered ontology resolves to an empty core and 404s.
        for node in nodes:
            node["properties"].setdefault("scheme_uri", metadata["uri"])
        nodes.append({
            "id": metadata["uri"],
            "type": "owl:Ontology",
            "content": metadata["name"],
            "properties": {"rdfs:label": metadata["name"], "uri": metadata["uri"]},
        })

    return nodes, edges, metadata


def _known_ontology_uris(
    session: GraphSession, registry: Dict[str, OntologyEntry]
) -> set[str]:
    known = set(registry)
    for node_type in _ONTOLOGY_TYPES:
        for node in session.iter_nodes(node_type=node_type):
            node_id = str(node.get("id", ""))
            if node_id:
                known.add(node_id)
    return known


def _collect_core_nodes(
    session: GraphSession, uri: str, known_ontology_uris: set[str]
) -> Dict[str, Dict[str, Any]]:
    """Stream schema nodes, keeping only the ones this ontology owns.

    Filtering as each node arrives makes _MAX_ANALYSIS_NODES bound the work and
    not merely the response: foreign nodes are discarded instead of materialized,
    and the scan stops once the owned ones pass the cap. The ownership filter has
    to stay ahead of that check — thousands of *other* ontologies' nodes must
    never make this one too large to open. Requesting pages instead would bound
    nothing: paginate_nodes normalizes the whole matching set on every call.
    """
    core_nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for node_type in _SCHEMA_NODE_TYPES:
        for node in session.iter_nodes(node_type=node_type):
            node_id = str(node.get("id", ""))
            if not node_id or not _node_belongs_to_ontology(
                node, uri, known_ontology_uris
            ):
                continue
            core_nodes_by_id[node_id] = node
            if len(core_nodes_by_id) > _MAX_ANALYSIS_NODES:
                raise GraphTruncationError(_GRAPH_TOO_LARGE_DETAIL)
    return core_nodes_by_id


def _select_structure_edges(
    session: GraphSession, core_node_ids: set[str]
) -> List[Dict[str, Any]]:
    """Stream structural edges, keeping only those leaving a core node.

    The requested ontology may reference outward (e.g. rdfs:range to an external
    vocabulary), but an unrelated ontology's property pointing at a core class
    must not leak inward.
    """
    selected_edges: List[Dict[str, Any]] = []
    for edge_type in _STRUCTURE_EDGE_TYPES:
        for edge in session.iter_edges(edge_type=edge_type):
            if str(edge.get("source", "")) not in core_node_ids:
                continue
            selected_edges.append(edge)
            if len(selected_edges) > _MAX_ANALYSIS_NODES:
                raise GraphTruncationError(_GRAPH_TOO_LARGE_DETAIL)
    return selected_edges


async def _fetch_analysis_graph(
    session: GraphSession,
    uri: str,
    log_tag: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch nodes/edges for SHACL analysis, raising GraphTruncationError if oversized.

    Shared by `_generated_shacl_for_uri` and `_data_graph_turtle_for_uri` so callers that
    need both (e.g. `/health`) can fetch once and pass the results to both via `nodes=`/`edges=`.
    """
    nodes, total_nodes = await asyncio.to_thread(session.get_nodes, skip=0, limit=_MAX_ANALYSIS_NODES)
    edges, total_edges = await asyncio.to_thread(session.get_edges, skip=0, limit=_MAX_ANALYSIS_NODES)
    if total_nodes > _MAX_ANALYSIS_NODES or total_edges > _MAX_ANALYSIS_NODES:
        logger.warning(
            "%s: graph for %s is too large to analyse (nodes=%d, edges=%d, limit=%d); skipping.",
            log_tag, uri, total_nodes, total_edges, _MAX_ANALYSIS_NODES,
        )
        raise GraphTruncationError(
            f"Graph size (nodes={total_nodes}, edges={total_edges}) exceeds maximum analysis limit ({_MAX_ANALYSIS_NODES}). "
            "SHACL validation is skipped because the graph is too large to validate fully under current limits."
        )
    return nodes, edges


async def _generated_shacl_for_uri(
    request: Request,
    session: GraphSession,
    uri: str,
    quality_tier: str = "strict",
    *,
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, List[ShaclShapeSummary]]:
    registry = {entry.uri: entry for entry in await _registry_entries(request, session)}
    entry = registry.get(uri)
    if entry is None:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")

    if nodes is None or edges is None:
        nodes, edges = await _fetch_analysis_graph(session, uri, "shacl-generate")
    entities = _ontology_entities(nodes, uri)
    ontology_dict = _ontology_dict_from_nodes(uri, entry.name, entities, edges)

    try:
        from ....ontology import OntologyEngine
        engine = OntologyEngine()
        shacl_turtle = await asyncio.to_thread(
            engine.to_shacl,
            ontology_dict,
            format="turtle",
            quality_tier=quality_tier,
            validate_output=False,
        )
    except Exception as exc:
        logger.debug("OntologyEngine.to_shacl unavailable; using basic generator: %s", exc)
        shacl_turtle = _basic_shacl_turtle(uri, entry.name, entities)

    return shacl_turtle, _summarize_shapes(shacl_turtle)


async def _data_graph_turtle_for_uri(
    request: Request,
    session: GraphSession,
    uri: str,
    *,
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
) -> str:
    try:
        import rdflib
    except ImportError as exc:
        raise ImportError("rdflib is not installed.") from exc

    registry = {entry.uri: entry for entry in await _registry_entries(request, session)}
    if uri not in registry:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")

    if nodes is None or edges is None:
        nodes, edges = await _fetch_analysis_graph(session, uri, "shacl-data-graph")
    entities = _data_graph_entities(nodes, uri)

    g = rdflib.Graph()
    base_namespace = _ontology_namespace(uri)

    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    RDF = rdflib.RDF
    RDFS = rdflib.RDFS
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    DCT = rdflib.Namespace("http://purl.org/dc/terms/")
    DC = rdflib.Namespace("http://purl.org/dc/elements/1.1/")
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    XSD = rdflib.XSD
    ONTO = rdflib.Namespace(base_namespace)

    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("dct", DCT)
    g.bind("dc", DC)
    g.bind("sh", SH)
    g.bind("xsd", XSD)
    g.bind("onto", ONTO)

    def _resolve_uri(val: str, base_ns: str) -> rdflib.URIRef:
        val_str = str(val).strip()
        if val_str == "a":
            return rdflib.RDF.type
        if val_str.startswith(("http://", "https://", "urn:", "ftp://")):
            return rdflib.URIRef(val_str)
        prefix_map = {
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dct": "http://purl.org/dc/terms/",
            "dc": "http://purl.org/dc/elements/1.1/",
            "sh": "http://www.w3.org/ns/shacl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "onto": base_ns,
        }
        if ":" in val_str and not val_str.startswith("/"):
            prefix, _, rest = val_str.partition(":")
            if prefix in prefix_map:
                return rdflib.URIRef(prefix_map[prefix] + rest)
            return rdflib.URIRef(val_str)
        if val_str.startswith("#"):
            val_str = val_str.lstrip("#")
        if base_ns.endswith(("/", "#")):
            return rdflib.URIRef(base_ns + val_str.lstrip("/"))
        return rdflib.URIRef(base_ns + "#" + val_str.lstrip("/"))

    _shorthand_types = {
        "class": "owl:Class",
        "object_property": "owl:ObjectProperty",
        "datatype_property": "owl:DatatypeProperty",
        "annotation_property": "owl:AnnotationProperty",
        "property": "rdf:Property",
        "individual": "owl:NamedIndividual",
        "concept": "skos:Concept",
        "scheme": "skos:ConceptScheme",
        "ontology": "owl:Ontology",
    }
    _uri_predicates = {
        "rdf:type",
        "a",
        "rdfs:subClassOf",
        "rdfs:domain",
        "rdfs:range",
        "owl:equivalentClass",
        "owl:equivalentProperty",
        "owl:sameAs",
        "skos:exactMatch",
        "skos:closeMatch",
        "skos:broadMatch",
        "skos:narrowMatch",
        "skos:relatedMatch",
        "sh:targetClass",
        "sh:targetNode",
    }
    _literal_predicates = {
        "rdfs:label",
        "rdfs:comment",
        "skos:definition",
        "skos:prefLabel",
        "skos:altLabel",
        "dct:description",
        "dct:title",
        "dc:title",
        "dc:description",
        "version",
    }
    _skip_keys = {
        "id",
        "type",
        "content",
        "valid_from",
        "valid_until",
        "scheme_uri",
        "ontology_uri",
        "ontology",
    }

    entity_ids = set()
    for node in entities:
        nid_str = str(node.get("id", "")).strip()
        if not nid_str:
            continue
        entity_ids.add(nid_str)
        subj = _resolve_uri(nid_str, base_namespace)

        node_type = node.get("type", "")
        if node_type:
            for t in _as_uri_list(node_type):
                t_mapped = _shorthand_types.get(str(t).lower(), str(t))
                g.add((subj, rdflib.RDF.type, _resolve_uri(t_mapped, base_namespace)))

        content = str(node.get("content", "")).strip()
        props = node.get("properties", {}) or {}
        if content and "rdfs:label" not in props and "pref_label" not in props and "label" not in props:
            g.add((subj, RDFS.label, rdflib.Literal(content)))

        for key, val in props.items():
            if key in _skip_keys or val is None or val == "":
                continue
            pred = _resolve_uri(str(key), base_namespace)
            for item in _as_uri_list(val) if isinstance(val, (list, dict)) else [val]:
                if item is None or item == "":
                    continue
                if isinstance(item, dict):
                    item = item.get("uri") or item.get("id") or item.get("@id")
                    if not item:
                        continue
                if str(key) in _uri_predicates or (
                    str(key) not in _literal_predicates
                    and str(item).strip().startswith(("http://", "https://", "urn:"))
                ):
                    g.add((subj, pred, _resolve_uri(str(item), base_namespace)))
                else:
                    g.add((subj, pred, rdflib.Literal(str(item))))

    for edge in edges:
        source = str(edge.get("source", edge.get("source_id", ""))).strip()
        target = str(edge.get("target", edge.get("target_id", ""))).strip()
        pred_str = str(edge.get("type", "related_to")).strip()
        if not source or not target:
            continue
        if source in entity_ids or target in entity_ids:
            g.add((
                _resolve_uri(source, base_namespace),
                _resolve_uri(pred_str, base_namespace),
                _resolve_uri(target, base_namespace),
            ))

    return await asyncio.to_thread(g.serialize, format="turtle")
