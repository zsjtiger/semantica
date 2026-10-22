"""
Context Graph Implementation

In-memory GraphStore implementation for building and querying context graphs
from conversations and entities with advanced analytics integration.

Core Features:
    - In-memory GraphStore implementation
    - Entity and relationship extraction from conversations
    - BFS-based neighbor discovery
    - Type-based indexing
    - Export to dictionary format
    - Decision tracking integration

Comprehensive Decision Management:
    - Decision Recording: Store decisions with full context and metadata
    - Precedent Search: Find similar decisions using hybrid search algorithms
    - Influence Analysis: Analyze decision impact and relationships
    - Causal Analysis: Trace decision causality chains
    - Policy Enforcement: Built-in policy compliance checking
    - Advanced Analytics: Comprehensive decision insights

KG Algorithm Integration:
    - Centrality Analysis: Degree, betweenness, closeness, eigenvector centrality
    - Community Detection: Modularity-based community identification
    - Node Embeddings: Node2Vec embeddings for similarity analysis
    - Path Finding: Shortest path and advanced path algorithms
    - Link Prediction: Relationship prediction between entities
    - Similarity Calculation: Multi-type similarity measures

Vector Store Integration:
    - Hybrid Search: Semantic + structural similarity
    - Custom Similarity Weights: Configurable scoring
    - Advanced Precedent Search: KG-enhanced similarity
    - Multi-Embedding Support: Multiple embedding types

Advanced Graph Analytics:
    - Node Centrality Analysis: Multiple centrality measures
    - Community Detection: Identify clusters and communities
    - Node Similarity: Content and structural similarity
    - Graph Structure Analysis: Comprehensive metrics
    - Path Analysis: Find paths and connectivity
    - Embedding Generation: Node embeddings for ML

Decision Tracking Integration:
    - Decision Storage: Store decisions with full context
    - Precedent Search: Find similar decisions using graph traversal
    - Causal Analysis: Trace decision influence
    - Decision Analytics: Analyze decision patterns
    - Influence Analysis: Decision influence scoring and analysis
    - Policy Engine: Policy enforcement and compliance checking
    - Relationship Mapping: Map decision dependencies

Enhanced Methods:
    - analyze_graph_with_kg(): Comprehensive graph analysis
    - get_node_centrality(): Get centrality measures for nodes
    - find_similar_nodes(): Find similar nodes with advanced similarity
    - record_decision(): Add decisions with context integration
    - find_precedents(): Find decision precedents
    - analyze_decision_influence(): Analyze decision influence
    - get_decision_insights(): Get comprehensive decision analytics
    - trace_decision_causality(): Trace decision causality
    - enforce_decision_policy(): Enforce decision policies
    - get_graph_metrics(): Get comprehensive statistics
    - export_graph(): Export graph in various formats

Example Usage:
    >>> from semantica.context import ContextGraph
    >>> graph = ContextGraph(advanced_analytics=True,
    ...                    centrality_analysis=True,
    ...                    community_detection=True,
    ...                    node_embeddings=True)
    >>> 
    >>> # Basic graph operations
    >>> graph.add_node("Python", "language", popularity="high")
    >>> graph.add_node("Programming", "concept")
    >>> graph.add_edge("Python", "Programming", "related_to")
    >>> centrality = graph.get_node_centrality("Python")
    >>> similar = graph.find_similar_nodes("Python", similarity_type="content")
    >>> analysis = graph.analyze_graph_with_kg()
    >>> 
    >>> # Decision management
    >>> decision_id = graph.record_decision(
    ...     category="loan_approval",
    ...     scenario="First-time homebuyer",
    ...     reasoning="Good credit score",
    ...     outcome="approved",
    ...     confidence=0.95,
    ...     entities=["customer_123", "property_456"]
    ... )
    >>> precedents = graph.find_precedents(decision_id, limit=5)
    >>> influence = graph.analyze_decision_influence(decision_id)
    >>> insights = graph.get_decision_insights()
    >>> causality = graph.trace_decision_causality(decision_id)

Production Use Cases:
    - Knowledge Management: Build and analyze knowledge graphs
    - Decision Support: Context graphs for decision making
    - Recommendation Systems: Graph-based recommendations
    - Social Networks: Analyze connections and influence
    - Research Networks: Map collaborations and citations
    - Financial Services: Loan approvals, fraud detection, risk assessment
    - Healthcare: Treatment decisions, policy compliance, clinical pathways
    - Legal: Case precedent analysis, decision consistency
    - Business: Workflow decisions, policy compliance, audit trails
"""

import copy
import errno
import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml
from ..utils.helpers import classify_path_distance
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from ..utils.skos import is_skos_hierarchy_edge, validate_skos_hierarchy
from ._markdown_filesystem import find_filesystem_link
from .entity_linker import EntityLinker
from .markdown import (
    MarkdownIdentityError,
    MarkdownResourceNotFoundError,
    MarkdownRevisionConflictError,
    markdown_document_revision,
)

from .context_graph_base import (  # noqa: F401  (re-exported for existing importers)
    CentralityCalculator,
    CommunityDetector,
    ConnectivityAnalyzer,
    ContextEdge,
    ContextNode,
    GraphAnalyzer,
    GraphBuilder,
    KG_AVAILABLE,
    LinkPredictor,
    NodeEmbedder,
    PathFinder,
    SimilarityCalculator,
    _ATTRS_MISSING,
    _CAUSAL_EDGE_ALIASES,
    _CAUSAL_EDGE_TYPES,
    _CAUSAL_TRAVERSAL_TYPES,
    _CausalChain,
    _UniqueKeySafeLoader,
    _closing_valid_until,
    _coerce_edge_endpoint,
    _coerce_float,
    _coerce_metadata_map,
    _coerce_node_id,
    _construct_unique_mapping,
    _default_edge_id,
    _normalize_temporal_input,
    _parse_iso_dt,
    _pick_first,
    _resolve_edge_identity,
    normalize_temporal_input,
)
from .context_graph_helpers import _InternalHelpersMixin
from .context_graph_builders import _BuilderMixin
from .context_graph_decisions import _DecisionMixin
from .context_graph_convenience import _ConvenienceMixin


class ContextGraph(_InternalHelpersMixin, _BuilderMixin, _DecisionMixin, _ConvenienceMixin):
    """
    Easy-to-Use Context Graph with All Advanced Features.
    
    This class provides simple methods for:
    - Building knowledge graphs
    - Recording and analyzing decisions
    - Finding precedents and patterns
    - Causal analysis and policy enforcement
    - Advanced graph analytics
    
    Perfect for building intelligent AI agents that can learn from decisions!
    """

    _MARKDOWN_FORMAT = "semantica-context-graph"

    _MARKDOWN_VERSION = 1

    _MARKDOWN_MANIFEST = "graph.md"

    _MARKDOWN_NODES_DIRECTORY = "nodes"

    _MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize context graph with optional advanced features.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - extract_entities: Extract entities from content (default: True)
                - extract_relationships: Extract relationships (default: True)
                - entity_linker: Entity linker instance
                - advanced_analytics: Enable KG algorithms (default: True)
                - centrality_analysis: Enable centrality measures (default: True)
                - community_detection: Enable community detection (default: True)
                - node_embeddings: Enable Node2Vec embeddings (default: True)
        """
        self.logger = get_logger("context_graph")
        self.config = config or {}
        self.config.update(kwargs)

        self.extract_entities = self.config.get("extract_entities", True)
        self.extract_relationships = self.config.get("extract_relationships", True)

        self.entity_linker = self.config.get("entity_linker") or EntityLinker()

        self._lock = threading.RLock()

        self.graph_id: str = str(uuid.uuid4())

        self.nodes: Dict[str, ContextNode] = {}
        self.edges: List[ContextEdge] = []
        self._edge_index: Dict[str, ContextEdge] = {}

        self._adjacency: Dict[str, List[ContextEdge]] = defaultdict(list)

        self.node_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.edge_type_index: Dict[str, List[ContextEdge]] = defaultdict(list)

        self._linked_graphs: Dict[str, Tuple["ContextGraph", str, str]] = {}

        self._unresolved_links: Dict[str, Dict[str, str]] = {}

        # Retraction closes an entity's validity window but keeps it in the
        # graph; a tombstone records that an entity was purged outright,
        # without retaining the purged content. Keyed by
        # ``(entity_kind, entity_id)`` -- node ids are caller-supplied strings
        # and edge ids are UUID strings, so a single id keyspace would let a
        # node record mask an edge of the same id, and vice versa.
        self._retractions: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._tombstones: Dict[Tuple[str, str], Dict[str, Any]] = {}

  
        self.progress_tracker = get_progress_tracker()
  
        if not self.progress_tracker.enabled:
            self.progress_tracker.enabled = True
        
   
        self.kg_components = {}
        self._analytics_cache = {}
        
        self.mutation_callback = self.config.get("mutation_callback", None)
        self._suspend_mutation_callback = False
        
        enable_advanced = self.config.get("advanced_analytics", True)
        
        if KG_AVAILABLE and enable_advanced:
            try:
                if self.config.get("centrality_analysis", True):
                    self.kg_components["centrality_calculator"] = CentralityCalculator()
                if self.config.get("community_detection", True):
                    self.kg_components["community_detector"] = CommunityDetector()
                if self.config.get("node_embeddings", True):
                    try:
                        self.kg_components["node_embedder"] = NodeEmbedder()
                    except ImportError as e:
                        self.logger.warning(f"Node2Vec embeddings disabled: {e}")
                self.kg_components["path_finder"] = PathFinder()
                self.kg_components["similarity_calculator"] = SimilarityCalculator()
                self.kg_components["connectivity_analyzer"] = ConnectivityAnalyzer()
                
                self.logger.info("Advanced KG components initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize KG components: {e}")
                self.kg_components = {}

    def add_nodes(self, nodes: List[Dict[str, Any]]) -> int:
        """
        Add nodes to graph.

        Args:
            nodes: List of nodes to add (dicts with id, type, properties)

        Returns:
            Number of nodes added
        """
        count = 0
        with self._lock:
            for raw_node in nodes:
                if not isinstance(raw_node, dict):
                    continue

                node_id = _coerce_node_id(raw_node)
                if node_id is None:
                    self.logger.warning("Skipping node without a usable id: %r", raw_node)
                    continue

                node_props = _coerce_metadata_map(
                    raw_node.get("metadata"),
                    raw_node.get("properties"),
                )
                node_type = _pick_first(
                    raw_node.get("type"),
                    raw_node.get("node_type"),
                    raw_node.get("category"),
                    raw_node.get(":LABEL"),
                    node_props.get("type"),
                    "entity",
                )
                content = _pick_first(
                    raw_node.get("content"),
                    raw_node.get("text"),
                    raw_node.get("label"),
                    raw_node.get("name"),
                    raw_node.get("title"),
                    raw_node.get("pref_label"),
                    node_props.get("content"),
                    node_props.get("text"),
                    node_props.get("label"),
                    node_props.get("name"),
                    node_props.get("title"),
                    node_props.get("pref_label"),
                    node_id,
                )

                valid_from = _pick_first(
                    raw_node.get("valid_from"),
                    node_props.get("valid_from"),
                )
                valid_until = _pick_first(
                    raw_node.get("valid_until"),
                    node_props.get("valid_until"),
                )
                metadata = {
                    k: v
                    for k, v in node_props.items()
                    if k not in ("content", "text", "valid_from", "valid_until")
                }

                internal_node = ContextNode(
                    node_id=node_id,
                    node_type=str(node_type or "entity"),
                    content=str(content or node_id),
                    metadata=metadata,
                    properties=node_props,
                    valid_from=valid_from,
                    valid_until=valid_until,
                )

                if self._add_internal_node(internal_node):
                    count += 1
        return count

    def add_edges(self, edges: List[Dict[str, Any]]) -> int:
        """
        Add edges to graph.

        Args:
            edges: List of edges to add (dicts with source_id, target_id, type,
                weight, properties)

        Returns:
            Number of edges added
        """
        count = 0
        with self._lock:
            # Keep the SKOS hierarchy invariant at the lowest common write
            # layer so direct graph users cannot bypass API/session checks.
            hierarchy_edges = [edge for edge in edges if is_skos_hierarchy_edge(edge)]
            if hierarchy_edges:
                existing_edges = [
                    edge for edge in self.find_edges() if is_skos_hierarchy_edge(edge)
                ]
                validate_skos_hierarchy(hierarchy_edges, existing_edges)
            for raw_edge in edges:
                if not isinstance(raw_edge, dict):
                    continue

                source_id = _coerce_edge_endpoint(raw_edge, "source")
                target_id = _coerce_edge_endpoint(raw_edge, "target")
                if source_id is None or target_id is None:
                    self.logger.warning("Skipping edge without usable endpoints: %r", raw_edge)
                    continue

                edge_props = _coerce_metadata_map(
                    raw_edge.get("metadata"),
                    raw_edge.get("properties"),
                )
                edge_type = _pick_first(
                    raw_edge.get("type"),
                    raw_edge.get("edge_type"),
                    raw_edge.get("relationship"),
                    raw_edge.get("predicate"),
                    raw_edge.get("relation"),
                    raw_edge.get(":TYPE"),
                    edge_props.get("type"),
                    "related_to",
                )

                valid_from = _pick_first(raw_edge.get("valid_from"), edge_props.get("valid_from"))
                valid_until = _pick_first(raw_edge.get("valid_until"), edge_props.get("valid_until"))
                weight = _coerce_float(_pick_first(raw_edge.get("weight"), edge_props.get("weight")), default=1.0)
                explicit_edge_id = _pick_first(
                    raw_edge.get("id"),
                    raw_edge.get("edge_id"),
                    edge_props.pop("id", None),
                    edge_props.pop("edge_id", None),
                )
                explicit_family_id = _pick_first(
                    raw_edge.get("familyId"),
                    raw_edge.get("family_id"),
                    edge_props.pop("familyId", None),
                    edge_props.pop("family_id", None),
                )
                edge_id, family_id = _resolve_edge_identity(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=str(edge_type or "related_to"),
                    weight=weight,
                    metadata=edge_props,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    edge_id=explicit_edge_id,
                    family_id=explicit_family_id,
                )
                internal_edge = ContextEdge(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=str(edge_type or "related_to"),
                    family_id=str(family_id) if family_id is not None else edge_id,
                    weight=weight,
                    metadata=edge_props,
                    valid_from=valid_from,
                    valid_until=valid_until,
                )

                if self._add_internal_edge(internal_edge):
                    count += 1
        return count

    def __contains__(self, node_id: object) -> bool:
        if not isinstance(node_id, str):
            return False
        with self._lock:
            return node_id in self.nodes

    def has_node(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self.nodes

    def neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        return self.get_neighbors(node_id, hops=1)

    def get_neighbor_ids(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
    ) -> List[str]:
        with self._lock:
            if node_id not in self.nodes:
                return []

            rel_filter = set(relationship_types) if relationship_types else None
            neighbor_ids: List[str] = []
            for edge in self._adjacency.get(node_id, []):
                if rel_filter is None or edge.edge_type in rel_filter:
                    neighbor_ids.append(edge.target_id)
            return neighbor_ids

    def get_nodes_by_label(self, label: str) -> List[Dict[str, Any]]:
        result = []
        with self._lock:
            for nid in self.node_type_index.get(label, set()):
                node = self.nodes.get(nid)
                if node:
                    result.append({
                        "id": node.node_id,
                        "content": node.content,
                        "type": node.node_type,
                        "metadata": (node.properties or {}).copy(),
                    })
        return result

    def get_node_property(
        self,
        node_id: str,
        property_name: str,
        default: Any = None,
    ) -> Any:
        """Return the value of *property_name* on *node_id*.

        Returns *default* when the node does not exist or when the property is
        not set on the node.  Both failure modes return the same *default*, so
        a sentinel can identify *any not-found result* as distinct from a
        property whose value is legitimately ``None``::

            _MISSING = object()
            val = graph.get_node_property(node_id, "score", default=_MISSING)
            if val is _MISSING:
                ...  # node absent or property not set

        To distinguish a missing node from a missing property specifically,
        call ``find_node()`` first to check node existence.

        Args:
            node_id: ID of the node to look up.
            property_name: Name of the property to retrieve.
            default: Value returned when the node or property is absent.
                Defaults to ``None`` (backward-compatible).

        Returns:
            The property value, or *default* if not found.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                return default
            return node.properties.get(property_name, default)

    def get_node_attributes(
        self,
        node_id: str,
        default: Any = _ATTRS_MISSING,
    ) -> Any:
        """Return a copy of all properties on *node_id*.

        Returns *default* when the node does not exist.  The historical
        default is ``{}`` (an empty dict), preserved for backward
        compatibility.  Pass a private sentinel as *default* to detect a
        missing node unambiguously::

            _MISSING = object()
            attrs = graph.get_node_attributes(node_id, default=_MISSING)
            if attrs is _MISSING:
                ...  # node does not exist

        Args:
            node_id: ID of the node to look up.
            default: Value returned when the node is absent.
                Defaults to ``{}`` (backward-compatible).

        Returns:
            A shallow copy of the node's properties dict, or *default*.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                return {} if default is _ATTRS_MISSING else default
            return node.properties.copy()

    def add_node_attribute(self, node_id: str, attributes: Dict[str, Any]) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                return
            node.properties.update(attributes)
            node.metadata.update(attributes)
            # Keep derived decision indexes consistent when a decision node is
            # mutated so that category / entity / temporal lookups reflect the
            # new property values without requiring a full graph reload.
            if (getattr(node, "node_type", None) or "").lower() == "decision":
                self._sync_decision_from_node(node_id)

        if getattr(self, "mutation_callback", None) and not getattr(
            self, "_suspend_mutation_callback", False
        ):
            try:
                self.mutation_callback("UPDATE_NODE", node_id, node.to_dict())
            except Exception as e:
                self.logger.warning(f"Audit trail callback failed for node {node_id}: {e}")

    def get_edge_data(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Return metadata for the edge between *source_id* and *target_id*.

        Returns an empty dict ``{}`` when no edge exists between the two nodes
        or when either node is absent.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            A dict containing edge metadata (``id``, ``familyId``, ``type``,
            ``weight``, plus any custom metadata), or ``{}`` if not found.
        """
        with self._lock:
            for edge in self._adjacency.get(source_id, []):
                if edge.target_id == target_id:
                    data = edge.metadata.copy()
                    data["id"] = edge.edge_id
                    data["familyId"] = edge.family_id or edge.edge_id
                    data["type"] = edge.edge_type
                    data["weight"] = edge.weight
                    return data
        return {}

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        relationship_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
        skip: int = 0,
        limit: Optional[int] = None,
        include_distance_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node.

        Args:
            node_id: Starting node ID.
            hops: Maximum number of hops to traverse (BFS depth).
            relationship_types: Optional whitelist of edge types to follow.
            min_weight: Minimum edge weight required to traverse an edge (default 0.0
                means all edges pass). Use e.g. ``min_weight=0.5`` to follow only
                strong/high-confidence relationships.
            skip: Number of items to skip for pagination.
            limit: Maximum items to return.

        Returns:
            List of dicts with neighbor info (id, type, content, relationship, weight, hop).
        """
        with self._lock:
            if node_id not in self.nodes:
                return []

            neighbors: List[Dict[str, Any]] = []
            visited = {node_id}
            queue = deque([(node_id, 0, [node_id], 1.0)])
            rel_filter = set(relationship_types) if relationship_types else None

            while queue:
                current_id, current_hop, path_so_far, decay_so_far = queue.popleft()
                if current_hop >= hops:
                    continue

                outgoing_edges = self._adjacency.get(current_id, [])
                for edge in outgoing_edges:
                    if rel_filter is not None and edge.edge_type not in rel_filter:
                        continue
                    if edge.weight < min_weight:
                        continue
                    neighbor_id = edge.target_id
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    next_hop = current_hop + 1
                    next_decay = decay_so_far * edge.weight
                    next_path = path_so_far + [neighbor_id]
                    queue.append((neighbor_id, next_hop, next_path, next_decay))

                    node = self.nodes.get(neighbor_id)
                    if not node:
                        continue
                    entry: Dict[str, Any] = {
                        "id": node.node_id,
                        "type": node.node_type,
                        "content": node.content,
                        "relationship": edge.edge_type,
                        "weight": edge.weight,
                        "hop": next_hop,
                    }
                    # Expose the merged node metadata so consumers such as
                    # support-aware retrieval can validate annotations on
                    # neighbor attachments instead of dropping them blindly.
                    entry_metadata: Dict[str, Any] = dict(node.properties or {})
                    entry_metadata.update(node.metadata or {})
                    entry["metadata"] = entry_metadata
                    if include_distance_metadata:
                        entry["distance_band"] = classify_path_distance(next_hop)
                        entry["confidence_decay"] = next_decay
                        entry["path_to_anchor"] = next_path
                    neighbors.append(entry)

            if limit is not None:
                return neighbors[skip: skip + limit]
            return neighbors[skip:]

    def get_neighbor_distances(
        self,
        node_id: str,
        hops: int = 3,
        relationship_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return neighbors with distance metadata, filtered by confidence decay.

        Results are ordered by nearest hop first, then by strongest path confidence.
        """
        neighbors = self.get_neighbors(
            node_id,
            hops=hops,
            relationship_types=relationship_types,
            include_distance_metadata=True,
        )
        filtered = [
            item for item in neighbors
            if item.get("confidence_decay", 0.0) >= min_confidence
        ]
        return sorted(
            filtered,
            key=lambda item: (item.get("hop", 0), -item.get("confidence_decay", 0.0)),
        )

    def query(
        self, query: str, skip: int = 0, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a simple keyword search query on the graph nodes.

        Args:
            query: Keyword query string
            skip: Items to skip
            limit: Max items to return

        Returns:
            List of matching node dicts
        """
        results = []
        query_lower = query.lower().split()

        with self._lock:
            for node in self.nodes.values():
                content_lower = node.content.lower()
                if any(word in content_lower for word in query_lower):
                    # Calculate simple score
                    overlap = sum(1 for word in query_lower if word in content_lower)
                    score = overlap / len(query_lower) if query_lower else 0.0

                    results.append(
                        {
                            "node": node.to_dict(),
                            "score": score,
                            "content": node.content,
                        }
                    )

        sorted_res = sorted(results, key=lambda x: x["score"], reverse=True)
        if limit is not None:
            return sorted_res[skip: skip + limit]
        return sorted_res[skip:]

    def add_node(
        self,
        node_id: str,
        node_type: str,
        content: Optional[str] = None,
        **properties,
    ) -> bool:
        """
        Add a single node to the graph.

        Args:
            node_id: Unique identifier
            node_type: Node type (e.g., 'entity', 'concept')
            content: Node content/label
            **properties: Additional properties. Use `valid_from` and `valid_until`
                (ISO datetime strings) to define a temporal validity window.
        """
        content = content or node_id
        valid_from = properties.pop("valid_from", None)
        valid_until = properties.pop("valid_until", None)
        with self._lock:
            return self._add_internal_node(
                ContextNode(
                    node_id=node_id,
                    node_type=node_type,
                    content=content,
                    metadata=properties,
                    properties=properties,
                    valid_from=valid_from,
                    valid_until=valid_until,
                )
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related_to",
        weight: float = 1.0,
        **properties,
    ) -> bool:
        """
        Add a single edge to the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            weight: Edge weight
            **properties: Additional properties. Use `valid_from` and `valid_until`
                (ISO datetime strings) to define a temporal validity window.
        """
        if isinstance(weight, dict):
            properties = {**weight, **properties}
            weight = properties.pop("weight", 1.0)
        valid_from = properties.pop("valid_from", None)
        valid_until = properties.pop("valid_until", None)
        explicit_edge_id = properties.pop("id", properties.pop("edge_id", None))
        explicit_family_id = properties.pop("familyId", properties.pop("family_id", None))
        edge_id, family_id = _resolve_edge_identity(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=properties,
            valid_from=valid_from,
            valid_until=valid_until,
            edge_id=explicit_edge_id,
            family_id=explicit_family_id,
        )
        with self._lock:
            candidate = {"source": source_id, "target": target_id, "type": edge_type}
            if is_skos_hierarchy_edge(candidate):
                existing_edges = [
                    edge for edge in self.find_edges() if is_skos_hierarchy_edge(edge)
                ]
                validate_skos_hierarchy([candidate], existing_edges)
            return self._add_internal_edge(
                ContextEdge(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    family_id=str(family_id) if family_id is not None else edge_id,
                    weight=weight,
                    metadata=properties,
                    valid_from=valid_from,
                    valid_until=valid_until,
                )
            )

    def export_node_markdown(self, node_id: str) -> str:
        """Return one existing node as canonical Markdown.

        Args:
            node_id: Stable identifier of the node to export.

        Returns:
            Canonical Markdown containing the node frontmatter and body.

        Raises:
            MarkdownResourceNotFoundError: If ``node_id`` does not exist.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise MarkdownResourceNotFoundError(
                    f"ContextGraph node {node_id!r} was not found."
                )
            return self._node_markdown_source(node)

    def _node_markdown_source(self, node: ContextNode) -> str:
        frontmatter = {
            "id": node.node_id,
            "type": node.node_type,
            "properties": copy.deepcopy(node.properties),
            "metadata": copy.deepcopy(node.metadata),
            "valid_from": node.valid_from,
            "valid_until": node.valid_until,
        }
        return self._render_markdown_document(
            frontmatter, node.content, f"node {node.node_id!r}"
        )

    def apply_node_markdown(
        self,
        node_id: str,
        document: str,
        *,
        expected_revision: Optional[str] = None,
    ) -> bool:
        """Validate and atomically replace one existing node.

        Args:
            node_id: Stable identifier of the node to update.
            document: Canonical Markdown containing the replacement node.
            expected_revision: Optional revision returned by
                :meth:`export_node_markdown`. A mismatch rejects stale edits.

        Returns:
            ``True`` when the node changed, otherwise ``False``.

        Raises:
            ValueError: If the Markdown or frontmatter is invalid.
            MarkdownIdentityError: If the frontmatter changes the node ID.
            MarkdownResourceNotFoundError: If ``node_id`` does not exist.
            MarkdownRevisionConflictError: If ``expected_revision`` is stale.
        """
        source = f"node {node_id!r}"
        frontmatter, body = self._parse_markdown_document(document, source)
        candidate = self._parse_markdown_node(frontmatter, body, source)
        if candidate.node_id != node_id:
            raise MarkdownIdentityError(
                f"Frontmatter id {candidate.node_id!r} does not match resource id "
                f"{node_id!r}."
            )

        with self._lock:
            existing = self.nodes.get(node_id)
            if existing is None:
                raise MarkdownResourceNotFoundError(
                    f"ContextGraph node {node_id!r} was not found."
                )
            if expected_revision is not None:
                current_revision = markdown_document_revision(
                    self._node_markdown_source(existing)
                )
                if current_revision != expected_revision:
                    raise MarkdownRevisionConflictError(current_revision)
            if existing == candidate:
                return False

            # Decision index rebuilding can still reject YAML-valid property
            # shapes, so retain every affected structure until commit succeeds.
            decision_state_before = {}
            if (
                existing.node_type.lower() == "decision"
                or candidate.node_type.lower() == "decision"
            ):
                for attribute in (
                    "_decisions",
                    "_decision_index",
                    "_entity_index",
                    "_temporal_index",
                ):
                    if not hasattr(self, attribute):
                        decision_state_before[attribute] = None
                    elif attribute in {"_decision_index", "_entity_index"}:
                        decision_state_before[attribute] = {
                            key: set(values)
                            for key, values in getattr(self, attribute).items()
                        }
                    elif attribute == "_temporal_index":
                        decision_state_before[attribute] = list(
                            getattr(self, attribute)
                        )
                    else:
                        decision_state_before[attribute] = dict(
                            getattr(self, attribute)
                        )

            old_type = existing.node_type
            try:
                old_bucket = self.node_type_index.get(old_type)
                if old_bucket is not None:
                    old_bucket.discard(node_id)
                    if not old_bucket:
                        del self.node_type_index[old_type]

                self.nodes[node_id] = candidate
                self.node_type_index[candidate.node_type].add(node_id)
                if (
                    old_type.lower() == "decision"
                    or candidate.node_type.lower() == "decision"
                ):
                    self._sync_decision_from_node(node_id)
                payload = candidate.to_dict()
                self._analytics_cache.clear()
            except Exception:
                self.nodes[node_id] = existing
                candidate_bucket = self.node_type_index.get(candidate.node_type)
                if candidate_bucket is not None:
                    candidate_bucket.discard(node_id)
                    if not candidate_bucket:
                        del self.node_type_index[candidate.node_type]
                self.node_type_index[old_type].add(node_id)
                for attribute, state in decision_state_before.items():
                    if state is None:
                        if hasattr(self, attribute):
                            delattr(self, attribute)
                    else:
                        restored = (
                            defaultdict(set, state)
                            if attribute in {"_decision_index", "_entity_index"}
                            else state
                        )
                        setattr(self, attribute, restored)
                raise

        self._emit_mutation("UPDATE_NODE", node_id, payload)
        return True

    def save_to_file(
        self, path: Union[str, Path], format: str = "json"
    ) -> None:
        """
        Save the context graph in JSON or Markdown format.

        Args:
            path: JSON file path or Markdown export directory
            format: Persistence format (``json`` or ``markdown``)
        """
        normalized_format = self._normalize_persistence_format(format)
        if normalized_format == "markdown":
            self._save_markdown_directory(Path(path))
            self.logger.info(f"Saved context graph Markdown to {path}")
            return

        with self._lock:
    
            links_data = []
            for link_id, (other_graph, source_node_id, target_node_id) in self._linked_graphs.items():
                links_data.append(
                    {
                        "link_id": link_id,
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "other_graph_id": other_graph.graph_id,
                    }
                )

            data = {
                "graph_id": self.graph_id,
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges],
                "links": links_data,
            }

        # Write atomically: serialize to a sibling temp file then replace the
        # destination in one OS-level rename.  This guarantees the destination
        # is either the old contents or the new contents — never a partial write
        # — so a crash or disk-full error during json.dump cannot corrupt the
        # sole persisted copy of the graph.
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=dest.parent, prefix=".kg_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dest)
        except Exception:
            # Clean up the temp file on any failure so we don't litter the
            # directory with partial writes.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self.logger.info(f"Saved context graph to {path}")

    def load_from_file(
        self, path: Union[str, Path], format: str = "json"
    ) -> None:
        """
        Load the context graph from JSON or Markdown.

        Args:
            path: JSON file path or Markdown export directory
            format: Persistence format (``json`` or ``markdown``)
        """
        normalized_format = self._normalize_persistence_format(format)
        if normalized_format == "markdown":
            markdown_path = Path(path)
            linked_component = find_filesystem_link(markdown_path)
            if linked_component is not None:
                raise ValueError(
                    "Refusing to import Markdown symbolic link or junction: "
                    f"{linked_component}"
                )
            if not markdown_path.exists():
                self.logger.warning(f"File not found: {path}")
                return
            self._load_markdown_directory(markdown_path)
            self.logger.info(f"Loaded context graph Markdown from {path}")
            return

        if not os.path.exists(path):
            self.logger.warning(f"File not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            if data and isinstance(data[0], dict) and any(
                key in data[0]
                for key in ("source", "source_id", "target", "target_id", "START_ID", ":START_ID")
            ):
                data = {"edges": data, "nodes": []}
            else:
                data = {"nodes": data, "edges": []}
        elif not isinstance(data, dict):
            raise ValueError("Graph file must contain a JSON object or array payload")

        with self._lock:
            # Clear existing
            self.nodes.clear()
            self.edges.clear()
            self._edge_index.clear()
            self._adjacency.clear()
            self.node_type_index.clear()
            self.edge_type_index.clear()
            self._linked_graphs.clear()
            self._unresolved_links.clear()
            self._analytics_cache.clear()
            # Deletion metadata belongs to the graph being replaced; keeping it
            # would make entities in the loaded graph read as already retracted.
            self._retractions.clear()
            self._tombstones.clear()

            if "graph_id" in data:
                self.graph_id = data["graph_id"]

    
            nodes = data.get("nodes")
            if nodes is None:
                nodes = data.get("entities")
            if nodes is None:
                nodes = data.get("vertices")
            if nodes is None:
                nodes = []
            self.add_nodes(nodes)

            edges = data.get("edges")
            if edges is None:
                edges = data.get("relationships")
            if edges is None:
                edges = data.get("links")
            if edges is None:
                edges = []
            self.add_edges(edges)

        
            for link_meta in data.get("links", []):
                link_id = link_meta.get("link_id")
                if link_id:
                    self._unresolved_links[link_id] = link_meta

            # Rebuild all derived decision indexes from the freshly-loaded
            # nodes so that find_precedents_by_scenario, find_similar_decisions,
            # and all decision analytics work correctly after a reload.
            # _rebuild_decision_indexes() unconditionally clears the old indexes
            # first, so repeated load_from_file calls never accumulate stale
            # entries from a previous file.
            self._rebuild_decision_indexes()

        self.logger.info(f"Loaded context graph from {path}")

    @staticmethod
    def _normalize_persistence_format(format: str) -> str:
        if not isinstance(format, str) or not format.strip():
            raise ValueError("Context graph persistence format must be a string.")
        normalized = format.strip().lower()
        if normalized not in {"json", "markdown"}:
            raise ValueError(
                f"Unsupported context graph persistence format: {format!r}. "
                "Expected 'json' or 'markdown'."
            )
        return normalized

    def _save_markdown_directory(self, destination: Path) -> None:
        if not destination.name:
            raise ValueError("Markdown export destination cannot be a filesystem root.")
        linked_component = find_filesystem_link(destination)
        if linked_component is not None:
            raise ValueError(
                "Refusing to replace Markdown symbolic link or junction: "
                f"{linked_component}"
            )
        if destination.exists() and not destination.is_dir():
            raise ValueError(
                f"Markdown export destination is not a directory: {destination}"
            )
        if destination.exists() and any(destination.iterdir()):
            try:
                self._parse_markdown_directory(
                    destination, require_canonical_layout=True
                )
            except (FileNotFoundError, ValueError):
                is_managed = False
            else:
                is_managed = True
            if not is_managed:
                raise ValueError(
                    "Refusing to replace a non-empty directory that is not a "
                    f"managed ContextGraph export: {destination}"
                )

        manifest_document, node_documents = self._markdown_documents()
        destination.parent.mkdir(parents=True, exist_ok=True)
        linked_component = find_filesystem_link(destination.parent)
        if linked_component is not None:
            raise ValueError(
                "Refusing to export through Markdown symbolic link or junction: "
                f"{linked_component}"
            )
        staging_path = Path(
            tempfile.mkdtemp(
                dir=str(destination.parent), prefix=f".{destination.name}.staging-"
            )
        )
        backup_path: Optional[Path] = None
        try:
            nodes_path = staging_path / self._MARKDOWN_NODES_DIRECTORY
            nodes_path.mkdir()
            self._write_staged_markdown(
                staging_path / self._MARKDOWN_MANIFEST, manifest_document
            )
            for filename, document in node_documents:
                self._write_staged_markdown(nodes_path / filename, document)

            if destination.exists():
                backup_path = destination.parent / (
                    f".{destination.name}.backup-{uuid.uuid4().hex}"
                )
                os.replace(destination, backup_path)
            try:
                os.replace(staging_path, destination)
                staging_path = None
            except BaseException as publish_error:
                if backup_path is not None and not destination.exists():
                    try:
                        os.replace(backup_path, destination)
                    except BaseException as restore_error:
                        self.logger.error(
                            "Failed to restore previous ContextGraph Markdown "
                            "export from %s after publish failure; preserving "
                            "the original publish error",
                            backup_path,
                            exc_info=(
                                type(restore_error),
                                restore_error,
                                restore_error.__traceback__,
                            ),
                        )
                        add_note = getattr(publish_error, "add_note", None)
                        if add_note is not None:
                            add_note(
                                "Restoring the previous ContextGraph Markdown "
                                f"export also failed: {restore_error}"
                            )
                    else:
                        backup_path = None
                raise

            if backup_path is not None:
                shutil.rmtree(backup_path)
                backup_path = None
        finally:
            if staging_path is not None:
                shutil.rmtree(staging_path, ignore_errors=True)
            if backup_path is not None and backup_path.exists():
                self.logger.warning(
                    "ContextGraph export left backup directory %s", backup_path
                )

    @staticmethod
    def _write_staged_markdown(path: Path, document: str) -> None:
        with path.open("x", encoding="utf-8") as output:
            output.write(document)
            output.flush()
            os.fsync(output.fileno())

    def _markdown_documents(self) -> Tuple[str, List[Tuple[str, str]]]:
        with self._lock:
            nodes = [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "properties": copy.deepcopy(node.properties),
                    "metadata": copy.deepcopy(node.metadata),
                    "valid_from": node.valid_from,
                    "valid_until": node.valid_until,
                    "content": node.content,
                }
                for node in self.nodes.values()
            ]
            edges = [
                {
                    "id": edge.edge_id,
                    "family_id": edge.family_id or edge.edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                    "metadata": copy.deepcopy(edge.metadata),
                    "valid_from": edge.valid_from,
                    "valid_until": edge.valid_until,
                }
                for edge in self.edges
            ]
            graph_id = self.graph_id
            links_by_id = {
                link_id: copy.deepcopy(link)
                for link_id, link in self._unresolved_links.items()
            }
            for link_id, (
                other_graph,
                source_node_id,
                target_node_id,
            ) in self._linked_graphs.items():
                links_by_id[link_id] = {
                    "link_id": link_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "other_graph_id": other_graph.graph_id,
                }

        edges.sort(
            key=lambda edge: (
                str(edge["id"]),
                str(edge["source"]),
                str(edge["target"]),
                str(edge["type"]),
            )
        )
        seen_edge_ids: Set[str] = set()
        duplicate_edge_ids: Set[str] = set()
        for edge in edges:
            edge_id = edge["id"]
            if not isinstance(edge_id, str) or not edge_id.strip():
                raise ValueError(
                    "Cannot export ContextGraph: every edge must have a string ID."
                )
            if edge_id in seen_edge_ids:
                duplicate_edge_ids.add(edge_id)
            seen_edge_ids.add(edge_id)
        if duplicate_edge_ids:
            duplicates = ", ".join(
                repr(edge_id) for edge_id in sorted(duplicate_edge_ids)
            )
            raise ValueError(
                f"Cannot export ContextGraph: duplicate edge ID(s): {duplicates}."
            )
        links = sorted(
            links_by_id.values(), key=lambda link: str(link.get("link_id", ""))
        )
        manifest = {
            "format": self._MARKDOWN_FORMAT,
            "version": self._MARKDOWN_VERSION,
            "graph_id": graph_id,
            "edges": edges,
            "links": links,
        }
        manifest_body = (
            "# Context Graph\n\n"
            "Graph relationships are stored in frontmatter. Node content is in "
            "the `nodes` directory.\n"
        )
        manifest_document = self._render_markdown_document(
            manifest, manifest_body, "graph manifest"
        )

        node_documents = []
        filenames = set()
        for node in sorted(nodes, key=lambda item: str(item["id"])):
            filename = self._node_markdown_filename(node["id"])
            normalized_filename = filename.casefold()
            if normalized_filename in filenames:
                raise ValueError(
                    f"Cannot export ContextGraph: duplicate filename {filename!r}."
                )
            filenames.add(normalized_filename)
            content = node.pop("content")
            node_documents.append(
                (
                    filename,
                    self._render_markdown_document(
                        node, content, f"node {node['id']!r}"
                    ),
                )
            )
        return manifest_document, node_documents

    @staticmethod
    def _node_markdown_filename(node_id: Any) -> str:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("Cannot export a ContextGraph node without a string ID.")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", node_id)
        slug = re.sub(r"-+", "-", slug).strip("._-")[:80].rstrip("._-")
        slug = slug or "node"
        digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:12]
        return f"{slug}--{digest}.md"

    @classmethod
    def _render_markdown_document(
        cls, frontmatter: Dict[str, Any], body: str, source: str
    ) -> str:
        if not isinstance(body, str):
            raise ValueError(f"Cannot export {source}: Markdown body must be a string.")
        canonical = cls._canonical_markdown_value(frontmatter, source)
        try:
            yaml_text = yaml.safe_dump(
                canonical,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Cannot export {source}: metadata is not YAML serializable."
            ) from exc
        return f"---\n{yaml_text}---\n\n{body}"

    @classmethod
    def _canonical_markdown_value(
        cls, value: Any, source: str, ancestors: Optional[Set[int]] = None
    ) -> Any:
        ancestors = set() if ancestors is None else ancestors
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError(
                    f"Invalid Markdown metadata in {source}: "
                    "mapping keys must be strings."
                )
            identity = id(value)
            if identity in ancestors:
                raise ValueError(
                    f"Invalid Markdown metadata in {source}: "
                    "values cannot contain cycles."
                )
            ancestors.add(identity)
            try:
                return {
                    key: cls._canonical_markdown_value(
                        value[key], source, ancestors
                    )
                    for key in sorted(value)
                }
            finally:
                ancestors.remove(identity)
        if isinstance(value, list):
            identity = id(value)
            if identity in ancestors:
                raise ValueError(
                    f"Invalid Markdown metadata in {source}: "
                    "values cannot contain cycles."
                )
            ancestors.add(identity)
            try:
                return [
                    cls._canonical_markdown_value(item, source, ancestors)
                    for item in value
                ]
            finally:
                ancestors.remove(identity)
        if isinstance(value, tuple) or isinstance(value, set):
            raise ValueError(
                f"Invalid Markdown metadata in {source}: "
                "tuples and sets are not supported."
            )
        return value

    def _load_markdown_directory(self, source: Path) -> None:
        parsed_state = self._parse_markdown_directory(source)
        graph_id, nodes_by_id, edges, unresolved_links = parsed_state

        adjacency: Dict[str, List[ContextEdge]] = defaultdict(list)
        edge_index: Dict[str, ContextEdge] = {}
        node_type_index: Dict[str, Set[str]] = defaultdict(set)
        edge_type_index: Dict[str, List[ContextEdge]] = defaultdict(list)
        for node in nodes_by_id.values():
            node_type_index[node.node_type].add(node.node_id)
        for edge in edges:
            edge_index[edge.edge_id] = edge
            adjacency[edge.source_id].append(edge)
            edge_type_index[edge.edge_type].append(edge)

        with self._lock:
            self.graph_id = graph_id
            self.nodes.clear()
            self.nodes.update(nodes_by_id)
            self.edges.clear()
            self.edges.extend(edges)
            self._edge_index.clear()
            self._edge_index.update(edge_index)
            self._adjacency.clear()
            self._adjacency.update(adjacency)
            self.node_type_index.clear()
            self.node_type_index.update(node_type_index)
            self.edge_type_index.clear()
            self.edge_type_index.update(edge_type_index)
            self._linked_graphs.clear()
            self._unresolved_links.clear()
            self._unresolved_links.update(unresolved_links)
            self._analytics_cache.clear()
            self._retractions.clear()
            self._tombstones.clear()
            # Rebuild derived decision indexes from the freshly-loaded nodes.
            self._rebuild_decision_indexes()

        if self.mutation_callback and not self._suspend_mutation_callback:
            mutation_events = [
                ("ADD_NODE", node.node_id, node.to_dict())
                for node in nodes_by_id.values()
            ]
            mutation_events.extend(
                ("ADD_EDGE", edge.edge_id, edge.to_dict()) for edge in edges
            )
            for operation, entity_id, payload in mutation_events:
                try:
                    self.mutation_callback(operation, entity_id, payload)
                except Exception as exc:
                    self.logger.warning(
                        "Audit trail callback failed for Markdown graph load "
                        "%s %s: %s",
                        operation,
                        entity_id,
                        exc,
                    )

    def _parse_markdown_directory(
        self, source: Path, require_canonical_layout: bool = False
    ) -> Tuple[
        str,
        Dict[str, ContextNode],
        List[ContextEdge],
        Dict[str, Dict[str, str]],
    ]:
        manifest_document, node_documents = self._read_markdown_directory(
            source, require_canonical_layout=require_canonical_layout
        )
        manifest, _ = self._parse_markdown_document(
            manifest_document, str(source / self._MARKDOWN_MANIFEST)
        )
        graph_id, edges, links = self._parse_markdown_manifest(manifest, source)

        nodes_by_id: Dict[str, ContextNode] = {}
        for node_source, document in node_documents:
            frontmatter, body = self._parse_markdown_document(document, node_source)
            node = self._parse_markdown_node(frontmatter, body, node_source)
            if node.node_id in nodes_by_id:
                raise ValueError(
                    f"Duplicate Markdown node ID {node.node_id!r} in {node_source}."
                )
            node_filename = Path(node_source).name
            if (
                require_canonical_layout
                and node_filename != self._node_markdown_filename(node.node_id)
            ):
                raise ValueError(
                    "Invalid managed ContextGraph export: node file "
                    f"{node_filename!r} is not the canonical filename "
                    f"for node {node.node_id!r}."
                )
            nodes_by_id[node.node_id] = node

        missing_endpoints = sorted(
            {
                endpoint
                for edge in edges
                for endpoint in (edge.source_id, edge.target_id)
                if endpoint not in nodes_by_id
            }
        )
        if missing_endpoints and require_canonical_layout:
            missing = ", ".join(repr(endpoint) for endpoint in missing_endpoints)
            raise ValueError(
                "Invalid managed ContextGraph export: edge endpoint(s) "
                f"{missing} do not have node files."
            )
        for endpoint in missing_endpoints:
            nodes_by_id[endpoint] = ContextNode(endpoint, "entity", endpoint)

        hierarchy_edges = [
            {
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type,
            }
            for edge in edges
            if is_skos_hierarchy_edge(edge.to_dict())
        ]
        if hierarchy_edges:
            validate_skos_hierarchy(hierarchy_edges, [])

        unresolved_links = {}
        for link in links:
            link_id = link["link_id"]
            if link_id in unresolved_links:
                raise ValueError(
                    f"Duplicate cross-graph link ID {link_id!r} in graph manifest."
                )
            if link["source_node_id"] not in nodes_by_id:
                raise ValueError(
                    f"Cross-graph link {link_id!r} references missing source node "
                    f"{link['source_node_id']!r}."
                )
            unresolved_links[link_id] = link
        return graph_id, nodes_by_id, edges, unresolved_links

    def _read_markdown_directory(
        self, source: Path, require_canonical_layout: bool = False
    ) -> Tuple[str, List[Tuple[str, str]]]:
        linked_component = find_filesystem_link(source)
        if linked_component is not None:
            raise ValueError(
                "Refusing to import Markdown symbolic link or junction: "
                f"{linked_component}"
            )
        if not source.exists():
            raise FileNotFoundError(
                f"ContextGraph Markdown import path does not exist: {source}"
            )
        if not source.is_dir():
            raise ValueError(
                f"ContextGraph Markdown import path is not a directory: {source}"
            )

        if require_canonical_layout:
            expected_entries = {
                self._MARKDOWN_MANIFEST,
                self._MARKDOWN_NODES_DIRECTORY,
            }
            actual_entries = {path.name for path in source.iterdir()}
            if actual_entries != expected_entries:
                unexpected = sorted(actual_entries - expected_entries)
                missing = sorted(expected_entries - actual_entries)
                details = []
                if unexpected:
                    details.append(f"unexpected entries: {unexpected!r}")
                if missing:
                    details.append(f"missing entries: {missing!r}")
                raise ValueError(
                    "Invalid managed ContextGraph export layout ("
                    + "; ".join(details)
                    + ")."
                )

        manifest_path = source / self._MARKDOWN_MANIFEST
        manifest_document = self._read_markdown_file(manifest_path)
        nodes_path = source / self._MARKDOWN_NODES_DIRECTORY
        linked_component = find_filesystem_link(nodes_path)
        if linked_component is not None:
            raise ValueError(
                "Refusing to import Markdown symbolic link or junction: "
                f"{linked_component}"
            )
        if not nodes_path.is_dir():
            raise ValueError(
                f"ContextGraph Markdown nodes directory is missing: {nodes_path}"
            )

        node_paths = []
        for path in nodes_path.iterdir():
            linked_component = find_filesystem_link(path)
            if linked_component is not None:
                raise ValueError(
                    "Refusing to import Markdown symbolic link or junction: "
                    f"{linked_component}"
                )
            if path.suffix.lower() not in self._MARKDOWN_EXTENSIONS:
                if require_canonical_layout:
                    raise ValueError(
                        "Invalid managed ContextGraph export: unexpected node "
                        f"entry {path.name!r}."
                    )
                continue
            if not path.is_file():
                raise ValueError(f"Markdown node path is not a regular file: {path}")
            node_paths.append(path)
        linked_component = find_filesystem_link(nodes_path)
        if linked_component is not None:
            raise ValueError(
                "Refusing to import Markdown symbolic link or junction: "
                f"{linked_component}"
            )
        node_paths.sort(key=lambda path: (path.name.casefold(), path.name))
        return manifest_document, [
            (str(path), self._read_markdown_file(path)) for path in node_paths
        ]

    @staticmethod
    def _read_markdown_file(path: Path) -> str:
        linked_component = find_filesystem_link(path)
        if linked_component is not None:
            raise ValueError(
                "Refusing to import Markdown symbolic link or junction: "
                f"{linked_component}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            linked_component = find_filesystem_link(path)
            if exc.errno == errno.ELOOP or linked_component is not None:
                raise ValueError(
                    "Refusing to import Markdown symbolic link or junction: "
                    f"{linked_component or path}"
                ) from exc
            if exc.errno == errno.ENOENT:
                raise FileNotFoundError(f"Markdown file is missing: {path}") from exc
            raise OSError(
                exc.errno,
                f"Failed to read Markdown file {path}: {exc.strerror or str(exc)}",
                exc.filename or str(path),
            ) from exc

        try:
            linked_component = find_filesystem_link(path)
            if linked_component is not None:
                raise ValueError(
                    "Refusing to import Markdown symbolic link or junction: "
                    f"{linked_component}"
                )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"Markdown path is not a regular file: {path}")
            with os.fdopen(descriptor, "r", encoding="utf-8") as input_file:
                descriptor = None
                return input_file.read()
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _parse_markdown_document(
        document: str, source: str
    ) -> Tuple[Dict[str, Any], str]:
        lines = document.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != "---":
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: "
                "document must start with '---'."
            )
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.rstrip("\r\n") == "---"
            ),
            None,
        )
        if closing_index is None:
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: missing closing '---'."
            )
        try:
            loaded = yaml.load(
                "".join(lines[1:closing_index]), Loader=_UniqueKeySafeLoader
            )
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: {exc}"
            ) from exc
        frontmatter = {} if loaded is None else loaded
        if not isinstance(frontmatter, dict):
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: expected a YAML mapping."
            )
        if any(not isinstance(key, str) for key in frontmatter):
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: "
                "field names must be strings."
            )

        body = "".join(lines[closing_index + 1 :])
        if body.startswith("\r\n"):
            body = body[2:]
        elif body.startswith("\n"):
            body = body[1:]
        return frontmatter, body

    def _parse_markdown_manifest(
        self, manifest: Dict[str, Any], source: Path
    ) -> Tuple[str, List[ContextEdge], List[Dict[str, str]]]:
        manifest_source = str(source / self._MARKDOWN_MANIFEST)
        if manifest.get("format") != self._MARKDOWN_FORMAT:
            raise ValueError(
                f"Invalid ContextGraph Markdown manifest in {manifest_source}: "
                f"'format' must be {self._MARKDOWN_FORMAT!r}."
            )
        version = manifest.get("version")
        if isinstance(version, bool) or version != self._MARKDOWN_VERSION:
            raise ValueError(
                f"Unsupported ContextGraph Markdown version {version!r} in "
                f"{manifest_source}; expected {self._MARKDOWN_VERSION}."
            )
        graph_id = self._required_markdown_string(
            manifest.get("graph_id"), "graph_id", manifest_source
        )

        raw_edges = manifest.get("edges", [])
        if not isinstance(raw_edges, list):
            raise ValueError(
                f"Invalid ContextGraph Markdown manifest in {manifest_source}: "
                "'edges' must be a list."
            )
        edges = []
        edge_ids = set()
        for index, raw_edge in enumerate(raw_edges):
            edge_source = f"{manifest_source} edge[{index}]"
            edge = self._parse_markdown_edge(raw_edge, edge_source)
            if edge.edge_id in edge_ids:
                raise ValueError(
                    f"Duplicate Markdown edge ID {edge.edge_id!r} in {edge_source}."
                )
            edge_ids.add(edge.edge_id)
            edges.append(edge)

        raw_links = manifest.get("links", [])
        if not isinstance(raw_links, list):
            raise ValueError(
                f"Invalid ContextGraph Markdown manifest in {manifest_source}: "
                "'links' must be a list."
            )
        links = [
            self._parse_markdown_link(link, f"{manifest_source} link[{index}]")
            for index, link in enumerate(raw_links)
        ]
        return graph_id, edges, links

    def _parse_markdown_node(
        self, frontmatter: Dict[str, Any], body: str, source: str
    ) -> ContextNode:
        node_id = self._required_markdown_string(frontmatter.get("id"), "id", source)
        node_type = self._required_markdown_string(
            frontmatter.get("type"), "type", source
        )
        properties = self._markdown_mapping(
            frontmatter.get("properties", {}), "properties", source
        )
        metadata = self._markdown_mapping(
            frontmatter.get("metadata", {}), "metadata", source
        )
        return ContextNode(
            node_id=node_id,
            node_type=node_type,
            content=body,
            properties=properties,
            metadata=metadata,
            valid_from=self._markdown_temporal_value(
                frontmatter.get("valid_from"), "valid_from", source
            ),
            valid_until=self._markdown_temporal_value(
                frontmatter.get("valid_until"), "valid_until", source
            ),
        )

    def _parse_markdown_edge(self, raw_edge: Any, source: str) -> ContextEdge:
        if not isinstance(raw_edge, dict):
            raise ValueError(f"Invalid Markdown edge in {source}: expected a mapping.")
        edge_id = self._required_markdown_string(raw_edge.get("id"), "id", source)
        family_value = raw_edge.get("family_id", raw_edge.get("familyId", edge_id))
        family_id = self._required_markdown_string(family_value, "family_id", source)
        source_id = self._required_markdown_string(
            raw_edge.get("source"), "source", source
        )
        target_id = self._required_markdown_string(
            raw_edge.get("target"), "target", source
        )
        edge_type = self._required_markdown_string(raw_edge.get("type"), "type", source)
        weight = raw_edge.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(
                f"Invalid Markdown edge in {source}: 'weight' must be a number."
            )
        metadata = self._markdown_mapping(
            raw_edge.get("metadata", {}), "metadata", source
        )
        return ContextEdge(
            edge_id=edge_id,
            family_id=family_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=float(weight),
            metadata=metadata,
            valid_from=self._markdown_temporal_value(
                raw_edge.get("valid_from"), "valid_from", source
            ),
            valid_until=self._markdown_temporal_value(
                raw_edge.get("valid_until"), "valid_until", source
            ),
        )

    def _parse_markdown_link(self, raw_link: Any, source: str) -> Dict[str, str]:
        if not isinstance(raw_link, dict):
            raise ValueError(
                f"Invalid cross-graph link in {source}: expected a mapping."
            )
        return {
            field_name: self._required_markdown_string(
                raw_link.get(field_name), field_name, source
            )
            for field_name in (
                "link_id",
                "source_node_id",
                "target_node_id",
                "other_graph_id",
            )
        }

    @staticmethod
    def _required_markdown_string(value: Any, field_name: str, source: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: "
                f"'{field_name}' must be a non-empty string."
            )
        return value

    @classmethod
    def _markdown_mapping(
        cls, value: Any, field_name: str, source: str
    ) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid Markdown frontmatter in {source}: "
                f"'{field_name}' must be a mapping."
            )
        canonical = cls._canonical_markdown_value(value, source)
        return dict(canonical)

    @staticmethod
    def _markdown_temporal_value(
        value: Any, field_name: str, source: str
    ) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return _normalize_temporal_input(value)
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str) and value.strip():
            try:
                _normalize_temporal_input(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Markdown frontmatter in {source}: "
                    f"'{field_name}' must be a valid ISO-8601 string."
                ) from exc
            return value
        raise ValueError(
            f"Invalid Markdown frontmatter in {source}: "
            f"'{field_name}' must be a valid ISO-8601 string."
        )

    def find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return a dict representation of the node identified by *node_id*.

        Returns ``None`` when the node does not exist.

        Args:
            node_id: ID of the node to look up.

        Returns:
            A dict with keys ``id``, ``type``, ``content``, and ``metadata``,
            or ``None`` if the node is not found.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node:
                merged_metadata = {}
                merged_metadata.update(getattr(node, "metadata", {}) or {})
                merged_metadata.update(getattr(node, "properties", {}) or {})
                return {
                    "id": node.node_id,
                    "type": node.node_type,
                    "content": node.content,
                    "metadata": merged_metadata,
                }
            return None

    def find_nodes(
        self, node_type: Optional[str] = None, skip: int = 0, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find nodes lazily"""
        with self._lock:
            if node_type:
                # Sets are unordered, sort IDs for deterministic pagination
                raw_ids = sorted(
                    (node_id for node_id in self.node_type_index.get(node_type, set()) if node_id is not None),
                    key=lambda value: str(value),
                )
                source = (self.nodes[nid] for nid in raw_ids if nid in self.nodes)
            else:
                source = self.nodes.values()

            gen = (
                {
                    "id": n.node_id,
                    "type": n.node_type or "entity",
                    "content": n.content or "",
                    "metadata": {**(getattr(n, "metadata", {}) or {}), **(getattr(n, "properties", {}) or {})},
                }
                for n in source if n.node_id
            )
            stop = skip + limit if limit is not None else None
    
            return list(itertools.islice(gen, skip, stop))

    def find_active_nodes(
        self,
        node_type: Optional[str] = None,
        at_time: Optional[datetime] = None,
        skip: int = 0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find active nodes lazily."""
        now = at_time or datetime.utcnow()
        with self._lock:
            if node_type:
                raw_ids = sorted(
                    (node_id for node_id in self.node_type_index.get(node_type, set()) if node_id is not None),
                    key=lambda value: str(value),
                )
                source = (self.nodes[nid] for nid in raw_ids if nid in self.nodes)
            else:
                source = self.nodes.values()

            def _active(nodes_iter):
                for n in nodes_iter:
                    if n.node_id and n.is_active(now):
                        yield {
                            "id": n.node_id,
                            "type": n.node_type or "entity",
                            "content": n.content or "",
                            "metadata": {
                                **(getattr(n, "metadata", {}) or {}),
                                **(getattr(n, "properties", {}) or {}),
                            },
                        }

            stop = skip + limit if limit is not None else None
            return list(itertools.islice(_active(source), skip, stop))

    def link_graph(
        self,
        other_graph: "ContextGraph",
        source_node_id: str,
        target_node_id: str,
        link_type: str = "CROSS_GRAPH",
    ) -> str:
        """
        Create a navigable link from a node in this graph to a node in another graph.

        This enables cross-graph navigation: separate ContextGraph instances can be
        linked hierarchically, allowing agents to traverse from one problem space into
        a related one without merging the graphs (like "a dream within a dream").

        Args:
            other_graph: The target ContextGraph instance.
            source_node_id: Node ID in *this* graph that serves as the exit point.
            target_node_id: Node ID in *other_graph* that serves as the entry point.
            link_type: Edge type label for the cross-graph bridge (default "CROSS_GRAPH").

        Returns:
            A unique link ID that can be passed to :meth:`navigate_to`.

        Raises:
            KeyError: If source_node_id is not in this graph or target_node_id is not
                in other_graph.
        """
        with self._lock:
            if source_node_id not in self.nodes:
                raise KeyError(f"Source node '{source_node_id}' not found in this graph")
            if target_node_id not in other_graph.nodes:
                raise KeyError(f"Target node '{target_node_id}' not found in other_graph")

            link_id = str(uuid.uuid4())
            self._linked_graphs[link_id] = (other_graph, source_node_id, target_node_id)

            marker_node_id = f"__cross_graph_{link_id}"
            self._add_internal_node(
                ContextNode(
                    node_id=marker_node_id,
                    node_type="cross_graph_link",
                    content=f"Cross-graph link → {target_node_id}",
                    metadata={"cross_graph": True, "link_id": link_id, "target_node_id": target_node_id},
                    properties={},
                )
            )

            self._add_internal_edge(
                ContextEdge(
                    source_id=source_node_id,
                    target_id=marker_node_id,
                    edge_type=link_type,
                    weight=1.0,
                    metadata={"cross_graph": True, "link_id": link_id},
                )
            )
            return link_id

    def navigate_to(self, link_id: str) -> Tuple["ContextGraph", str]:
        """
        Navigate to the target graph and entry node for a cross-graph link.

        Args:
            link_id: Link ID returned by :meth:`link_graph`.

        Returns:
            Tuple of ``(other_graph, target_node_id)``.

        Raises:
            KeyError: If link_id is not registered on this graph.
        """
        if link_id not in self._linked_graphs:
            if link_id in self._unresolved_links:
                meta = self._unresolved_links[link_id]
                raise KeyError(
                    f"Cross-graph link '{link_id}' exists but its target graph "
                    f"(graph_id={meta.get('other_graph_id')!r}) has not been reconnected. "
                    "Call resolve_links({graph_id: graph_instance, ...}) to restore navigation."
                )
            raise KeyError(
                f"No cross-graph link '{link_id}' found. "
                "Call link_graph() first to create the link."
            )
        other_graph, _, target_node_id = self._linked_graphs[link_id]
        return other_graph, target_node_id

    def cross_graph_path(
        self,
        source_node_id: str,
        target_graph: "ContextGraph",
        target_node_id: str,
        max_hops: int = 10,
    ) -> Dict[str, Any]:
        """
        Find the shortest path across linked ContextGraph instances.
        """
        start = (self.graph_id, source_node_id)
        goal = (target_graph.graph_id, target_node_id)
        if source_node_id not in self.nodes or target_node_id not in target_graph.nodes:
            return {
                "path": [],
                "hop_count": 0,
                "cross_graph_links_used": 0,
                "confidence_decay": 0.0,
                "distance_band": classify_path_distance(max_hops + 1),
                "reachable": False,
            }

        queue = deque([(self, source_node_id, [start], 0, 1.0, 0)])
        visited = {start}

        while queue:
            graph, current_id, path, hop_count, decay, links_used = queue.popleft()
            current_key = (graph.graph_id, current_id)
            if current_key == goal:
                return {
                    "path": path,
                    "hop_count": hop_count,
                    "cross_graph_links_used": links_used,
                    "confidence_decay": decay,
                    "distance_band": classify_path_distance(hop_count),
                    "reachable": True,
                }
            if hop_count >= max_hops:
                continue

            with graph._lock:
                outgoing_edges = list(graph._adjacency.get(current_id, []))

            for edge in outgoing_edges:
                marker = graph.nodes.get(edge.target_id)
                link_id = None
                if marker and marker.node_type == "cross_graph_link":
                    link_id = marker.metadata.get("link_id")

                if link_id:
                    try:
                        next_graph, next_node_id = graph.navigate_to(link_id)
                    except KeyError:
                        continue
                    next_key = (next_graph.graph_id, next_node_id)
                    next_links_used = links_used + 1
                else:
                    next_graph, next_node_id = graph, edge.target_id
                    next_key = (graph.graph_id, edge.target_id)
                    next_links_used = links_used

                if next_key in visited:
                    continue
                visited.add(next_key)
                queue.append(
                    (
                        next_graph,
                        next_node_id,
                        path + [next_key],
                        hop_count + 1,
                        decay * edge.weight,
                        next_links_used,
                    )
                )

        return {
            "path": [],
            "hop_count": 0,
            "cross_graph_links_used": 0,
            "confidence_decay": 0.0,
            "distance_band": classify_path_distance(max_hops + 1),
            "reachable": False,
        }

    def resolve_links(self, graphs: Dict[str, "ContextGraph"]) -> int:
        """
        Reconnect cross-graph links after a :meth:`load_from_file` call.

        Since ``other_graph`` object references cannot be serialised, links are stored
        as metadata only (``other_graph_id``).  Call this method with a mapping of
        ``{graph_id: graph_instance}`` to restore live navigation.

        Args:
            graphs: Mapping of graph_id strings to ContextGraph instances.

        Returns:
            Number of links successfully resolved.

        Example::

            g1.save_to_file("g1.json")
            g2.save_to_file("g2.json")

            g1b, g2b = ContextGraph(), ContextGraph()
            g1b.load_from_file("g1.json")
            g2b.load_from_file("g2.json")
            resolved = g1b.resolve_links({g2b.graph_id: g2b})
        """
        resolved = 0
        with self._lock:
            for link_id, meta in list(self._unresolved_links.items()):
                other_graph_id = meta.get("other_graph_id")
                if other_graph_id in graphs:
                    other_graph = graphs[other_graph_id]
                    source_node_id = meta["source_node_id"]
                    target_node_id = meta["target_node_id"]
                    # Validate target node still exists in the restored graph
                    if target_node_id in other_graph.nodes:
                        self._linked_graphs[link_id] = (other_graph, source_node_id, target_node_id)
                        del self._unresolved_links[link_id]
                        resolved += 1
                    else:
                        self.logger.warning(
                            f"resolve_links: target node '{target_node_id}' not found in "
                            f"graph '{other_graph_id}' for link '{link_id}'"
                        )
        return resolved

    def find_edges(
        self, edge_type: Optional[str] = None, skip: int = 0, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find edges lazily."""
        with self._lock:
            source = self.edge_type_index.get(edge_type, []) if edge_type else self.edges
            
            gen = (
                {
                    "id": e.edge_id,
                    "familyId": e.family_id or e.edge_id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type,
                    "weight": e.weight,
                    "metadata": e.metadata,
                    "valid_from": e.valid_from,
                    "valid_until": e.valid_until,
                }
                for e in source if e.source_id and e.target_id
            )
            stop = skip + limit if limit is not None else None
            return list(itertools.islice(gen, skip, stop))

    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            # Count only items that find_nodes/find_edges can return, so pagination
            # totals reported to callers match what the methods actually yield.
            node_count = sum(1 for n in self.nodes.values() if n.node_id)
            edge_count = sum(1 for e in self.edges if e.source_id and e.target_id)
            node_types = {
                k: sum(
                    1 for nid in v
                    if isinstance(nid, str) and nid in self.nodes and self.nodes[nid].node_id
                )
                for k, v in self.node_type_index.items()
            }
            edge_types = {
                k: sum(1 for e in v if e.source_id and e.target_id)
                for k, v in self.edge_type_index.items()
            }
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "node_types": node_types,
                "edge_types": edge_types,
                "density": self.density(),
            }

    def density(self) -> float:
        """Calculate graph density."""
        with self._lock:
            n = len(self.nodes)
            if n < 2:
                return 0.0
            max_edges = n * (n - 1) 
            return len(self.edges) / max_edges

    def retract_node(
        self,
        node_id: str,
        reason: Optional[str] = None,
        at: Optional[Union[str, datetime]] = None,
        cascade: bool = True,
    ) -> bool:
        """Retract a node: no longer active, but still visible in history.

        Closes the node's validity window rather than deleting it, so
        :meth:`state_at` before ``at`` still returns the node and any decision
        recorded against it remains explainable. Use :meth:`purge_node` when
        the data itself has to be gone.

        Args:
            node_id: Node to retract.
            reason: Why it was retracted, stored on the retraction record.
            at: When the retraction takes effect (ISO string or datetime).
                Defaults to now, UTC.
            cascade: Also retract every edge touching the node. Leaving edges
                active around an inactive node means :meth:`find_active_nodes`
                drops the node while its relationships still read as current,
                so the default keeps the active view self-consistent.

        Retraction is expressed through the temporal window, so it is visible
        to the activity-aware views -- :meth:`find_active_nodes`,
        :meth:`state_at`, ``ContextNode.is_active`` -- and not to membership
        checks like :meth:`has_node` or :meth:`stats`, which continue to count
        the retained record. That matches how ``valid_until`` already behaved
        before retraction existed.

        A node whose ``valid_until`` is already earlier than ``at`` keeps that
        earlier bound: retraction only ever closes a validity window, never
        widens one.

        Returns:
            True if the node was retracted; False if it does not exist or was
            already retracted.

        Note:
            Emits ``UPDATE_NODE`` to the audit-trail callback, since retraction
            changes the validity window rather than removing the record.
        """
        at_iso = _normalize_temporal_input(at) or datetime.now(timezone.utc).isoformat()
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                self.logger.warning("Cannot retract unknown node: %r", node_id)
                return False
            if ("node", node_id) in self._retractions:
                return False

            node.valid_until = _closing_valid_until(node.valid_until, at_iso)
            record = {
                "entity_id": node_id,
                "entity_kind": "node",
                "retracted_at": at_iso,
                "reason": reason,
            }
            self._retractions[("node", node_id)] = record
            node_payload = {**node.to_dict(), "retraction": dict(record)}

            cascaded: List[Tuple[str, Dict[str, Any]]] = []
            if cascade:
                # Snapshotted once, before the loop: edge_id is content-derived
                # and not guaranteed unique (#922), so two distinct edge objects
                # can share one id. Checking the live _retractions dict inside
                # the loop would let the first duplicate's record block the
                # second from ever being closed, leaving it active indefinitely
                # while its retraction record claimed otherwise.
                already_retracted_edge_ids = {
                    key[1] for key in self._retractions if key[0] == "edge"
                }
                for edge in self._incident_edges(node_id):
                    if edge.edge_id in already_retracted_edge_ids:
                        continue
                    edge.valid_until = _closing_valid_until(edge.valid_until, at_iso)
                    edge_record = {
                        "entity_id": edge.edge_id,
                        "entity_kind": "edge",
                        "retracted_at": at_iso,
                        "reason": reason,
                        "cascaded_from": node_id,
                    }
                    self._retractions[("edge", edge.edge_id)] = edge_record
                    # Payloads are snapshotted here, not read back after the
                    # lock is released: a concurrent clear() would otherwise
                    # wipe the record out from under the emission below.
                    cascaded.append(
                        (
                            edge.edge_id,
                            {**edge.to_dict(), "retraction": dict(edge_record)},
                        )
                    )

        self._emit_mutation("UPDATE_NODE", node_id, node_payload)
        for edge_id, edge_payload in cascaded:
            self._emit_mutation("UPDATE_EDGE", edge_id, edge_payload)
        self.logger.info(
            "Retracted node %r at %s (cascaded %d edge(s))",
            node_id,
            at_iso,
            len(cascaded),
        )
        return True

    def retract_edge(
        self,
        edge_id: str,
        reason: Optional[str] = None,
        at: Optional[Union[str, datetime]] = None,
    ) -> bool:
        """Retract a single edge, leaving its endpoints untouched.

        An edge whose ``valid_until`` is already earlier than ``at`` keeps that
        earlier bound; retraction never widens a validity window.

        Args:
            edge_id: Edge to retract.
            reason: Why it was retracted.
            at: When the retraction takes effect. Defaults to now, UTC.

        Returns:
            True if the edge was retracted; False if it does not exist or was
            already retracted.

        Note:
            ``edge_id`` is content-derived and not guaranteed unique (#922):
            two distinct edge objects can share one id. Every edge matching
            ``edge_id`` is closed under a single retraction record, so a
            duplicate can never be left silently active while the record
            claims it was retracted.
        """
        at_iso = _normalize_temporal_input(at) or datetime.now(timezone.utc).isoformat()
        with self._lock:
            edges = [e for e in self.edges if e.edge_id == edge_id]
            if not edges:
                self.logger.warning("Cannot retract unknown edge: %r", edge_id)
                return False
            if ("edge", edge_id) in self._retractions:
                return False

            record = {
                "entity_id": edge_id,
                "entity_kind": "edge",
                "retracted_at": at_iso,
                "reason": reason,
            }
            self._retractions[("edge", edge_id)] = record
            for edge in edges:
                edge.valid_until = _closing_valid_until(edge.valid_until, at_iso)
            payload = {**edges[0].to_dict(), "retraction": dict(record)}

        self._emit_mutation("UPDATE_EDGE", edge_id, payload)
        self.logger.info(
            "Retracted edge %r at %s (%d underlying record(s))",
            edge_id,
            at_iso,
            len(edges),
        )
        return True

    def purge_node(
        self,
        node_id: str,
        reason: Optional[str] = None,
        at: Optional[Union[str, datetime]] = None,
        cascade: bool = True,
    ) -> bool:
        """Permanently remove a node; history no longer contains it.

        Unlike :meth:`retract_node` this is destructive: the node disappears
        from :meth:`state_at` as well as from the active view. Only a tombstone
        remains, recording that a purge happened and why -- deliberately
        without the purged content, since retaining it would defeat the point.

        Scope is this graph only. Copies held elsewhere (``AgentMemory``, a
        bound vector store, an exported file) are not reached, so this is one
        step of an erasure workflow, not the whole of it. Callers who need the
        whole workflow -- and a receipt recording which stores it actually
        reached -- should drive this through
        :class:`~semantica.context.erasure.ErasureCoordinator` rather than
        treating a ``True`` here as proof the content is gone.

        Args:
            node_id: Node to purge.
            reason: Why it was purged, e.g. an erasure-request reference.
            at: When the purge takes effect, recorded as the tombstone's
                ``purged_at`` (ISO string or datetime). Defaults to now, UTC.
            cascade: Also purge every edge touching the node, and the marker
                node of any cross-graph link it exits through. Defaults to True
                because leaving edges pointing at a removed node produces
                dangling endpoints.

        Cross-graph links registered by :meth:`link_graph` out of this node are
        deregistered either way -- a link whose source no longer exists would
        still resolve through :meth:`navigate_to` and still be serialized by
        :meth:`save_to_file`.

        Returns:
            True if the node was purged; False if it does not exist.

        Note:
            Emits ``REMOVE_NODE``/``REMOVE_EDGE`` to the audit-trail callback.
        """
        purged_at = (
            _normalize_temporal_input(at) or datetime.now(timezone.utc).isoformat()
        )
        with self._lock:
            if node_id not in self.nodes:
                self.logger.warning("Cannot purge unknown node: %r", node_id)
                return False

            # The link marker node is scaffolding reachable only from the node
            # being purged, so it goes with the cascade rather than surviving as
            # an orphan. Resolve the markers before deregistering the links they
            # are derived from.
            targets = [node_id]
            if cascade:
                targets.extend(self._cross_graph_marker_nodes(node_id))
            for link_id in self._cross_graph_links_for(node_id):
                self._linked_graphs.pop(link_id, None)
                self._unresolved_links.pop(link_id, None)

            # Tombstones are snapshotted into locals before the lock is
            # released; reading them back afterwards would race a clear().
            purged_edges: List[Tuple[str, Dict[str, Any]]] = []
            purged_nodes: List[Tuple[str, Dict[str, Any]]] = []
            for target in targets:
                cascaded_from = None if target == node_id else node_id
                if cascade:
                    for edge in self._incident_edges(target):
                        self._drop_edge_from_indexes(edge)
                        edge_record = {
                            "entity_id": edge.edge_id,
                            "entity_kind": "edge",
                            "purged_at": purged_at,
                            "reason": reason,
                            "cascaded_from": node_id,
                        }
                        self._tombstones[("edge", edge.edge_id)] = edge_record
                        self._retractions.pop(("edge", edge.edge_id), None)
                        purged_edges.append((edge.edge_id, dict(edge_record)))

                self._drop_node_from_indexes(target)
                node_record = {
                    "entity_id": target,
                    "entity_kind": "node",
                    "purged_at": purged_at,
                    "reason": reason,
                }
                if cascaded_from is not None:
                    node_record["cascaded_from"] = cascaded_from
                self._tombstones[("node", target)] = node_record
                self._retractions.pop(("node", target), None)
                purged_nodes.append((target, dict(node_record)))

        for edge_id, payload in purged_edges:
            self._emit_mutation("REMOVE_EDGE", edge_id, payload)
        for purged_id, payload in purged_nodes:
            self._emit_mutation("REMOVE_NODE", purged_id, payload)
        self.logger.info(
            "Purged node %r (cascaded %d edge(s), %d node(s))",
            node_id,
            len(purged_edges),
            len(purged_nodes) - 1,
        )
        return True

    def purge_edge(
        self,
        edge_id: str,
        reason: Optional[str] = None,
        at: Optional[Union[str, datetime]] = None,
    ) -> bool:
        """Permanently remove a single edge, leaving its endpoints in place.

        If the edge is the bridge of a cross-graph link, the link is also
        deregistered -- :meth:`navigate_to` should not keep resolving a link
        whose bridge is gone. The marker node itself is an endpoint and is left
        in place; purge it directly, or purge the link's source node, to remove
        it too.

        Args:
            edge_id: Edge to purge.
            reason: Why it was purged.
            at: When the purge takes effect, recorded as the tombstone's
                ``purged_at``. Defaults to now, UTC.

        Returns:
            True if the edge was purged; False if it does not exist.

        Note:
            ``edge_id`` is content-derived and not guaranteed unique (#922):
            two distinct edge objects can share one id. Every edge matching
            ``edge_id`` is dropped under a single tombstone, so a duplicate
            can never be left live in the graph while the tombstone claims
            the edge is gone.
        """
        purged_at = (
            _normalize_temporal_input(at) or datetime.now(timezone.utc).isoformat()
        )
        with self._lock:
            edges = [e for e in self.edges if e.edge_id == edge_id]
            if not edges:
                self.logger.warning("Cannot purge unknown edge: %r", edge_id)
                return False
            for edge in edges:
                self._drop_edge_from_indexes(edge)
                link_id = (edge.metadata or {}).get("link_id")
                if (edge.metadata or {}).get("cross_graph") and link_id:
                    self._linked_graphs.pop(link_id, None)
                    self._unresolved_links.pop(link_id, None)
            self._retractions.pop(("edge", edge_id), None)
            record = {
                "entity_id": edge_id,
                "entity_kind": "edge",
                "purged_at": purged_at,
                "reason": reason,
            }
            self._tombstones[("edge", edge_id)] = record
            payload = dict(record)

        self._emit_mutation("REMOVE_EDGE", edge_id, payload)
        self.logger.info(
            "Purged edge %r (%d underlying record(s))", edge_id, len(edges)
        )
        return True

    def get_retraction(
        self, entity_id: str, entity_kind: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the retraction record for a node or edge, or None.

        Args:
            entity_id: Node id or edge id.
            entity_kind: ``"node"`` or ``"edge"``. Records are keyed by kind as
                well as id, so pass this when a node id and an edge id could
                collide; without it a node record is preferred over an edge one.
        """
        with self._lock:
            return self._find_removal_record(self._retractions, entity_id, entity_kind)

    def get_tombstone(
        self, entity_id: str, entity_kind: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the purge tombstone for a node or edge, or None.

        The tombstone records that a purge happened, when, and why. It never
        contains the purged content.

        Args:
            entity_id: Node id or edge id.
            entity_kind: ``"node"`` or ``"edge"``; disambiguates a node id that
                collides with an edge id, as for :meth:`get_retraction`.
        """
        with self._lock:
            return self._find_removal_record(self._tombstones, entity_id, entity_kind)

    def list_retractions(self) -> List[Dict[str, Any]]:
        """Return every retraction record."""
        with self._lock:
            return [dict(record) for record in self._retractions.values()]

    def list_tombstones(self) -> List[Dict[str, Any]]:
        """Return every purge tombstone."""
        with self._lock:
            return [dict(record) for record in self._tombstones.values()]

    def clear(self) -> None:
        """Fully reset the graph state and indexes."""
        with self._lock:
            self.nodes.clear()
            self.edges.clear()
            self._edge_index.clear()
            self._adjacency.clear()
            self.node_type_index.clear()
            self.edge_type_index.clear()
            self._linked_graphs.clear()
            self._unresolved_links.clear()
            self._retractions.clear()
            self._tombstones.clear()
            # Reset derived decision indexes so that decision queries against
            # a cleared graph return empty results rather than stale data.
            self._decisions = {}
            self._decision_index = defaultdict(set)
            self._entity_index = defaultdict(set)
            self._temporal_index = []
        self.logger.debug("Graph state fully cleared.")


ContextGraphBuilder = ContextGraph
