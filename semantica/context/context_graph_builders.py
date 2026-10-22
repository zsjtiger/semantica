"""Builder and utility methods for ContextGraph."""

import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .context_graph_base import (
    ContextEdge,
    ContextNode,
    _CAUSAL_EDGE_ALIASES,
    _CAUSAL_EDGE_TYPES,
    _CAUSAL_TRAVERSAL_TYPES,
    _resolve_edge_identity,
)


class _BuilderMixin:
    """Mixin composed into ContextGraph; methods use ``self``."""


    # --- Builder Methods (Legacy/Utility) ---

    def build_from_conversations(
        self,
        conversations: List[Union[str, Dict[str, Any]]],
        link_entities: bool = True,
        extract_intents: bool = False,
        extract_sentiments: bool = False,
        **options,
    ) -> Dict[str, Any]:
        """
        Build context graph from conversations and return dict representation.

        Args:
            conversations: List of conversation files or dictionaries
            ...

        Returns:
            Graph dictionary (nodes, edges)
        """
        tracking_id = self.progress_tracker.start_tracking(
            file=None,
            module="context",
            submodule="ContextGraph",
            message=f"Building graph from {len(conversations)} conversations",
        )

        try:
            for conv in conversations:
                conv_data = (
                    conv if isinstance(conv, dict) else self._load_conversation(conv)
                )
                self._process_conversation(
                    conv_data,
                    extract_intents=extract_intents,
                    extract_sentiments=extract_sentiments,
                )

            if link_entities:
                self._link_entities()

            self.progress_tracker.stop_tracking(tracking_id, status="completed")
            return self.to_dict()

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def build_from_entities_and_relationships(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build graph from entities and relationships.

        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            **kwargs: Additional options

        Returns:
            Graph dictionary (nodes, edges)
        """
        tracking_id = self.progress_tracker.start_tracking(
            file=None,
            module="context",
            submodule="ContextGraph",
            message=(
                f"Building graph from {len(entities)} entities and "
                f"{len(relationships)} relationships"
            ),
        )

        try:
            # Add entities
            for entity in entities:
                entity_id = entity.get("id") or entity.get("entity_id")
                if entity_id:
                    self._add_internal_node(
                        ContextNode(
                            node_id=entity_id,
                            node_type=entity.get("type", "entity"),
                            content=entity.get("text")
                            or entity.get("label")
                            or entity_id,
                            metadata=entity,
                            properties=entity,
                        )
                    )

            # Add relationships
            for rel in relationships:
                source = rel.get("source_id")
                target = rel.get("target_id")
                if source and target:
                    self._add_internal_edge(
                        ContextEdge(
                            source_id=source,
                            target_id=target,
                            edge_type=rel.get("type", "related_to"),
                            weight=rel.get("confidence", 1.0),
                            metadata=rel,
                        )
                    )

            self.progress_tracker.stop_tracking(tracking_id, status="completed")
            return self.to_dict()

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def _process_conversation(self, conv_data: Dict[str, Any], **kwargs) -> None:
        """Process a single conversation."""
        conv_id = conv_data.get("id") or f"conv_{hash(str(conv_data)) % 10000}"

        # Add conversation node
        self._add_internal_node(
            ContextNode(
                node_id=conv_id,
                node_type="conversation",
                content=conv_data.get("content", "") or conv_data.get("summary", ""),
                metadata={"timestamp": conv_data.get("timestamp")},
            )
        )

        # Track name to ID mapping for relationship resolution
        name_to_id = {}

        # Extract entities
        if self.extract_entities:
            for entity in conv_data.get("entities", []):
                entity_id = entity.get("id") or entity.get("entity_id")
                entity_text = (
                    entity.get("text")
                    or entity.get("label")
                    or entity.get("name")
                    or entity_id
                )
                entity_type = entity.get("type", "entity")

                # Generate ID if missing
                if not entity_id and entity_text and self.entity_linker:
                    # Use EntityLinker to generate ID
                    if hasattr(self.entity_linker, "_generate_entity_id"):
                        entity_id = self.entity_linker._generate_entity_id(
                            entity_text, entity_type
                        )
                    else:
                        # Fallback ID generation
                        import hashlib

                        entity_hash = hashlib.md5(  # nosec B324 - deterministic entity ID, not security-sensitive
                            f"{entity_text}_{entity_type}".encode()
                        ).hexdigest()[:12]
                        entity_id = f"{entity_type.lower()}_{entity_hash}"

                if entity_id:
                    if entity_text:
                        name_to_id[entity_text] = entity_id

                    self._add_internal_node(
                        ContextNode(
                            node_id=entity_id,
                            node_type="entity",
                            content=entity_text,
                            metadata={"type": entity_type, **entity},
                        )
                    )
                    self._add_internal_edge(
                        ContextEdge(
                            source_id=conv_id,
                            target_id=entity_id,
                            edge_type="mentions",
                        )
                    )

        # Extract relationships
        if self.extract_relationships:
            for rel in conv_data.get("relationships", []):
                source = rel.get("source_id")
                target = rel.get("target_id")

                # Resolve IDs from names if missing
                if not source and rel.get("source") and rel.get("source") in name_to_id:
                    source = name_to_id[rel.get("source")]

                if not target and rel.get("target") and rel.get("target") in name_to_id:
                    target = name_to_id[rel.get("target")]

                if source and target:
                    self._add_internal_edge(
                        ContextEdge(
                            source_id=source,
                            target_id=target,
                            edge_type=rel.get("type", "related_to"),
                            weight=rel.get("confidence", 1.0),
                        )
                    )

    def _link_entities(self) -> None:
        """Link similar entities using EntityLinker."""
        if not self.entity_linker:
            return

        entity_nodes = [n for n in self.nodes.values() if n.node_type == "entity"]
        for i, node1 in enumerate(entity_nodes):
            for node2 in entity_nodes[i + 1 :]:
                similarity = self.entity_linker._calculate_text_similarity(
                    node1.content.lower(), node2.content.lower()
                )
                if similarity >= self.entity_linker.similarity_threshold:
                    self._add_internal_edge(
                        ContextEdge(
                            source_id=node1.node_id,
                            target_id=node2.node_id,
                            edge_type="similar_to",
                            weight=similarity,
                        )
                    )

    def _load_conversation(self, file_path: str) -> Dict[str, Any]:
        """Load conversation from file."""

        from ..utils.helpers import read_json_file

        return read_json_file(Path(file_path))

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        with self._lock:
            nodes_out = []
            for n in self.nodes.values():
                entry: Dict[str, Any] = {
                    "id": n.node_id,
                    "type": n.node_type,
                    "content": n.content,
                    "properties": n.properties,
                    "metadata": n.metadata,
                }
                if n.valid_from is not None:
                    entry["valid_from"] = n.valid_from
                if n.valid_until is not None:
                    entry["valid_until"] = n.valid_until
                nodes_out.append(entry)

            edges_out = []
            for e in self.edges:
                entry = {
                    "id": e.edge_id,
                    "familyId": e.family_id or e.edge_id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type,
                    "weight": e.weight,
                }
                if e.metadata:
                    entry["metadata"] = e.metadata
                if e.valid_from is not None:
                    entry["valid_from"] = e.valid_from
                if e.valid_until is not None:
                    entry["valid_until"] = e.valid_until
                edges_out.append(entry)

            return {
                "nodes": nodes_out,
                "edges": edges_out,
                "statistics": {
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges),
                },
            }

    def to_kg_dict(self, entities_only: bool = False) -> Dict[str, Any]:
        """Export graph in the canonical knowledge-graph shape.

        This is the official adapter that converts the ContextGraph's internal
        ``{"nodes", "edges"}`` / ``source`` representation into the
        ``{"entities", "relationships"}`` / ``source_id`` shape expected by
        downstream consumers such as
        :class:`~semantica.export.rdf_exporter.RDFExporter` and
        :meth:`~semantica.kg.temporal_query.TemporalGraphQuery.query_time_range`.

        Users no longer need to hand-map field names between APIs.

        Args:
            entities_only: If True, only nodes whose ``node_type`` is
                ``"entity"`` are exported as entities. When False (default),
                every node is exported. Relationships whose endpoints are not
                in the exported entity set are dropped to avoid dangling
                references in downstream consumers.

        Returns:
            dict: A knowledge-graph dictionary with:
                - ``entities``: list of ``{"id", "text", "type", "properties",
                  "metadata"}`` (plus ``valid_from`` / ``valid_until`` when set)
                - ``relationships``: list of ``{"source_id", "target_id",
                  "type", "weight", "id", "familyId"}`` (plus ``metadata`` and
                  ``valid_from`` / ``valid_until`` when set)
                - ``statistics``: ``{"entity_count", "relationship_count"}``
        """
        with self._lock:
            entities_out = []
            for n in self.nodes.values():
                if entities_only and n.node_type != "entity":
                    continue
                # Normalize the entity id to ``str`` so it matches ContextEdge,
                # which coerces its endpoints to ``str`` in ``__post_init__``.
                # Without this, non-string node ids (e.g. numeric ids loaded via
                # ``from_dict``) would fail the ``valid_ids`` membership check
                # below and silently drop otherwise-valid relationships.
                entity_id = str(n.node_id)
                entity: Dict[str, Any] = {
                    "id": entity_id,
                    "text": n.content,
                    "type": n.node_type,
                    # ``properties`` / ``metadata`` may be ``None`` when a node
                    # was loaded from JSON containing an explicit ``null``;
                    # guard with ``or {}`` so ``dict(...)`` never raises.
                    "properties": dict(n.properties or {}),
                    "metadata": dict(n.metadata or {}),
                }
                if n.valid_from is not None:
                    entity["valid_from"] = n.valid_from
                if n.valid_until is not None:
                    entity["valid_until"] = n.valid_until
                entities_out.append(entity)

            # When only entity nodes are exported, drop relationships whose
            # endpoints were filtered out so downstream consumers never see a
            # source_id/target_id that is absent from ``entities``.
            valid_ids = {e["id"] for e in entities_out} if entities_only else None

            relationships_out = []
            for e in self.edges:
                if valid_ids is not None and (
                    e.source_id not in valid_ids or e.target_id not in valid_ids
                ):
                    continue
                rel: Dict[str, Any] = {
                    "id": e.edge_id,
                    "familyId": e.family_id or e.edge_id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "type": e.edge_type,
                    "weight": e.weight,
                }
                if e.metadata:
                    rel["metadata"] = dict(e.metadata)
                if e.valid_from is not None:
                    rel["valid_from"] = e.valid_from
                if e.valid_until is not None:
                    rel["valid_until"] = e.valid_until
                relationships_out.append(rel)

            return {
                "entities": entities_out,
                "relationships": relationships_out,
                "statistics": {
                    "entity_count": len(entities_out),
                    "relationship_count": len(relationships_out),
                },
            }

    def from_dict(self, graph_dict: Dict[str, Any]) -> None:
        """Load graph from dictionary format."""
        # Clear existing graph
        self.clear()

        # Add nodes — restore validity windows if present
        for node_data in graph_dict.get("nodes", []):
            node_props = node_data.get("properties", {})
            node = ContextNode(
                node_id=node_data["id"],
                node_type=node_data["type"],
                content=node_data.get("content", ""),
                properties=node_props,
                metadata=node_data.get("metadata", {}),
                valid_from=node_data.get("valid_from") or node_props.get("valid_from"),
                valid_until=node_data.get("valid_until") or node_props.get("valid_until"),
            )
            self._add_internal_node(node)

        # Add edges — restore validity windows if present
        for edge_data in graph_dict.get("edges", []):
            edge_metadata = edge_data.get("metadata", edge_data.get("properties", {})) or {}
            edge_weight = edge_data.get("weight", 1.0)
            edge_id, family_id = _resolve_edge_identity(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=edge_data["type"],
                weight=edge_weight,
                metadata=edge_metadata,
                valid_from=edge_data.get("valid_from"),
                valid_until=edge_data.get("valid_until"),
                edge_id=edge_data.get("id", edge_data.get("edge_id")),
                family_id=edge_data.get("familyId", edge_data.get("family_id")),
            )
            edge = ContextEdge(
                edge_id=edge_id,
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=edge_data["type"],
                weight=edge_weight,
                family_id=family_id,
                metadata=edge_metadata,
                valid_from=edge_data.get("valid_from"),
                valid_until=edge_data.get("valid_until"),
            )
            self._add_internal_edge(edge)

        # Rebuild derived decision indexes from the now-populated node store.
        self._rebuild_decision_indexes()

    def state_at(self, timestamp: Union[str, int, float, datetime]) -> Dict[str, Any]:
        """Return a serializable snapshot of graph state valid at the given time."""
        at_time = self._normalize_timestamp(timestamp)
        with self._lock:
            active_nodes = [node for node in self.nodes.values() if node.is_active(at_time)]
            active_node_ids = {node.node_id for node in active_nodes}
            active_edges = [
                edge for edge in self.edges
                if edge.is_active(at_time)
                and edge.source_id in active_node_ids
                and edge.target_id in active_node_ids
            ]

        nodes_payload = [node.to_dict() for node in active_nodes]
        edges_payload = [edge.to_dict() for edge in active_edges]
        decisions_payload = [
            {
                "id": node.node_id,
                "category": node.properties.get("category", ""),
                "scenario": node.properties.get("scenario", node.content),
                "reasoning": node.properties.get("reasoning", ""),
                "outcome": node.properties.get("outcome", ""),
                "confidence": node.properties.get("confidence", 0.0),
                "timestamp": node.properties.get("timestamp"),
                "decision_maker": node.properties.get("decision_maker"),
                "entities": node.properties.get("entities", []),
                "valid_from": node.valid_from,
                "valid_until": node.valid_until,
                "metadata": dict(node.properties.get("metadata", {}) or {}),
            }
            for node in active_nodes
            if isinstance(node.node_type, str) and node.node_type.lower() == "decision"
        ]

        return {
            "timestamp": at_time.isoformat(),
            "nodes": nodes_payload,
            "edges": edges_payload,
            "entities": nodes_payload,
            "relationships": edges_payload,
            "decisions": decisions_payload,
        }

    # Decision Support Methods
    def add_decision(
        self,
        decision: "Decision" = None,
        *,
        category: str = None,
        scenario: str = None,
        reasoning: str = None,
        outcome: str = None,
        confidence: float = 0.5,
        entities: Optional[List[str]] = None,
        decision_maker: Optional[str] = "system",
        valid_from=None,
        valid_until=None,
        **kwargs,
    ) -> str:
        """
        Add decision node to graph.

        Accepts either a Decision object or keyword arguments:

            # From a Decision object
            graph.add_decision(Decision(category="x", scenario="y", ...))

            # From keyword arguments (convenience form)
            graph.add_decision(category="x", scenario="y", reasoning="z",
                               outcome="o", confidence=0.9)

        Args:
            decision: Decision object to add (mutually exclusive with kwargs)
            category: Decision category
            scenario: Decision scenario description
            reasoning: Reasoning behind the decision
            outcome: Decision outcome
            confidence: Confidence score (0.0–1.0)
            entities: Related entity labels
            decision_maker: Who made the decision
            valid_from: Start of validity window (ISO string or datetime)
            valid_until: End of validity window (ISO string or datetime)
            **kwargs: Extra metadata stored on the decision node

        Returns:
            Decision ID
        """

        if decision is not None and (
            any(v is not None for v in (
                category, scenario, reasoning, outcome, entities, valid_from, valid_until,
            )) or kwargs
        ):
            raise ValueError(
                "Pass either a Decision object or keyword arguments, not both."
            )

        if decision is None:
            # Build from kwargs — delegate to record_decision which handles ID gen
            return self.record_decision(
                category=category,
                scenario=scenario,
                reasoning=reasoning,
                outcome=outcome,
                confidence=confidence,
                entities=entities,
                decision_maker=decision_maker,
                valid_from=valid_from,
                valid_until=valid_until,
                metadata=kwargs,
            )

        # Handle empty decision ID by generating UUID for both None and empty string
        # This ensures consistent behavior with Decision model's __post_init__ method
        node_id = decision.decision_id if decision.decision_id else str(uuid.uuid4())

        # Handle None metadata
        metadata = decision.metadata or {}

        # Validate the values the decision indexes need before touching graph
        # state, so a bad value can't leave a stored but unindexed node.
        try:
            hash(decision.category)
            float(decision.confidence)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Decision {node_id!r} has an invalid category or confidence: {e}"
            ) from e
        meta_entities = metadata.get("entities")
        if meta_entities is not None:
            if not isinstance(meta_entities, list):
                raise ValueError("Decision metadata 'entities' must be a list")
            try:
                for entity in meta_entities:
                    hash(entity)
            except TypeError as e:
                raise ValueError(
                    "Decision metadata 'entities' must contain hashable values"
                ) from e

        # Normalize timestamp to ensure consistent storage format
        raw_timestamp = decision.timestamp
        if isinstance(raw_timestamp, str) and raw_timestamp.strip().endswith(("Z", "z")):
            raw_timestamp = raw_timestamp.strip()[:-1] + "+00:00"
        normalized_timestamp = self._normalize_timestamp(raw_timestamp)

        node = ContextNode(
            node_id=node_id,
            node_type="Decision",
            content=decision.scenario,
            properties={
                # Metadata first so it can't override the Decision's own fields.
                **metadata,
                "category": decision.category,
                "scenario": decision.scenario,
                "reasoning": decision.reasoning,
                "outcome": decision.outcome,
                "confidence": decision.confidence,
                "timestamp": normalized_timestamp.isoformat(),
                "decision_maker": decision.decision_maker,
                "reasoning_embedding": decision.reasoning_embedding,
                "node2vec_embedding": decision.node2vec_embedding,
            },
            valid_from=decision.valid_from,
            valid_until=decision.valid_until,
        )
        with self._lock:
            if self._add_internal_node(node):
                # Register in the decision indexes so precedent search,
                # influence, causality and insights can see this decision.
                self._sync_decision_from_node(node_id)
        return node_id

    def add_causal_relationship(
        self,
        source_decision_id: str,
        target_decision_id: str,
        relationship_type: str
    ) -> bool:
        """
        Add causal relationship between decisions.

        Args:
            source_decision_id: Source decision ID
            target_decision_id: Target decision ID
            relationship_type: Type of relationship (CAUSED, INFLUENCED, PRECEDENT_FOR)

        Returns:
            True when a new causal edge was inserted.
            False when the operation was skipped without inserting an edge:
              - source or target decision ID is not present in the graph;
              - source or target node exists but is not a decision node;
              - an equivalent causal relationship (same source, target, and
                normalized relationship type) already exists.
            Skipped operations are logged at WARNING level. Invalid
            relationship types raise ValueError instead of returning False.
        """
        # Normalize so callers may use either vocabulary's spelling
        # ("causes" from CausalChainAnalyzer, or "CAUSED" from this module's
        # canonical constant); the stored form is always canonical. Invalid
        # inputs keep raising ValueError rather than AttributeError.
        if not isinstance(relationship_type, str):
            raise ValueError(f"Relationship type must be one of: {_CAUSAL_EDGE_TYPES}")
        relationship_type = _CAUSAL_EDGE_ALIASES.get(relationship_type.strip().upper())
        if relationship_type is None:
            raise ValueError(f"Relationship type must be one of: {_CAUSAL_EDGE_TYPES}")
        
        # Check if decisions exist - if not, skip adding relationship
        if source_decision_id not in self.nodes or target_decision_id not in self.nodes:
            self.logger.warning(
                "add_causal_relationship skipped: unknown decision id(s) "
                f"'{source_decision_id}' -> '{target_decision_id}'"
            )
            return False

        # Check if nodes are decision nodes - if not, skip adding relationship
        source_node = self.nodes[source_decision_id]
        target_node = self.nodes[target_decision_id]
        if (not hasattr(source_node, 'node_type') or not isinstance(source_node.node_type, str) or
            not hasattr(target_node, 'node_type') or not isinstance(target_node.node_type, str) or
            source_node.node_type.lower() != "decision" or
            target_node.node_type.lower() != "decision"):
            self.logger.warning(
                "add_causal_relationship skipped: nodes are not decision nodes "
                f"('{source_decision_id}' -> '{target_decision_id}')"
            )
            return False

        edge = ContextEdge(
            source_id=source_decision_id,
            target_id=target_decision_id,
            edge_type=relationship_type,
            weight=1.0,
            metadata={"recorded_at": datetime.utcnow().isoformat()},
        )
        # edge_id includes recorded_at, so the same causal link recorded twice
        # gets a new id; compare (target, type) instead.
        if not self._add_internal_edge(edge, unique_type=True):
            self.logger.warning(
                "Causal relationship %s -[%s]-> %s already exists; skipping",
                source_decision_id,
                relationship_type,
                target_decision_id,
            )
            return False
        return True

    def get_causal_chain(
        self,
        decision_id: str,
        direction: str = "upstream",
        max_depth: int = 10
    ) -> List["Decision"]:
        """
        Get causal chain from graph.
        
        Args:
            decision_id: Starting decision ID
            direction: "upstream" or "downstream"
            max_depth: Maximum traversal depth
            
        Returns:
            List of decisions in causal chain
        """
        from .decision_models import Decision
        
        if direction not in ["upstream", "downstream"]:
            raise ValueError("Direction must be 'upstream' or 'downstream'")
        
        # BFS traversal
        visited = set()
        queue = deque([(decision_id, 0)])
        decisions = []
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Skip the starting decision - only add connected decisions
            if current_id != decision_id:
                # Get decision node
                if current_id in self.nodes:
                    node = self.nodes[current_id]
                    if (hasattr(node, 'node_type') and isinstance(node.node_type, str) and
                        node.node_type.lower() == "decision"):
                        decision_data = node.properties
                        timestamp = self._normalize_timestamp(decision_data.get("timestamp"))
                        decision = Decision(
                            decision_id=current_id,
                            category=decision_data.get("category", ""),
                            scenario=decision_data.get("scenario", node.content),
                            reasoning=decision_data.get("reasoning", ""),
                            outcome=decision_data.get("outcome", ""),
                            confidence=decision_data.get("confidence", 0.0),
                            timestamp=timestamp,
                            decision_maker=decision_data.get("decision_maker", ""),
                            reasoning_embedding=decision_data.get("reasoning_embedding"),
                            node2vec_embedding=decision_data.get("node2vec_embedding"),
                            valid_from=node.valid_from,
                            valid_until=node.valid_until,
                            metadata={k: v for k, v in decision_data.items() if k not in [
                                "category", "scenario", "reasoning", "outcome", "confidence", 
                                "timestamp", "decision_maker", "reasoning_embedding", "node2vec_embedding"
                            ]}
                        )
                        decision.metadata["causal_distance"] = depth
                        decisions.append(decision)
            
            # Find connected decisions
            for edge in self.edges:
                if direction == "upstream":
                    if edge.target_id == current_id and edge.edge_type.upper() in _CAUSAL_TRAVERSAL_TYPES:
                        if edge.source_id not in visited and depth < max_depth:
                            queue.append((edge.source_id, depth + 1))
                else:  # downstream
                    if edge.source_id == current_id and edge.edge_type.upper() in _CAUSAL_TRAVERSAL_TYPES:
                        if edge.target_id not in visited and depth < max_depth:
                            queue.append((edge.target_id, depth + 1))
        
        # Sort by depth for upstream (most distant first) and downstream (closest first)
        if direction == "upstream":
            decisions.sort(key=lambda d: d.metadata.get("causal_distance", 0), reverse=True)
        else:
            decisions.sort(key=lambda d: d.metadata.get("causal_distance", 0))
        
        return decisions

    def find_precedents(self, decision_id: str, limit: int = 10) -> List["Decision"]:
        """
        Find precedent decisions.
        
        Args:
            decision_id: Decision ID to find precedents for
            limit: Maximum number of results
            
        Returns:
            List of precedent decisions
        """
        # Find decisions connected via PRECEDENT_FOR relationships, accepting
        # the analyzer vocabulary's "precedes" spelling as well (issue #1184).
        precedent_ids = []
        for edge in self.edges:
            if edge.target_id == decision_id and edge.edge_type.upper() in {
                "PRECEDENT_FOR", "PRECEDES",
            }:
                precedent_ids.append(edge.source_id)
        
        # Convert to Decision objects
        decisions = []
        for pid in precedent_ids[:limit]:
            if pid in self.nodes:
                node = self.nodes[pid]
                if (hasattr(node, 'node_type') and isinstance(node.node_type, str) and
                    node.node_type.lower() == "decision"):
                    decision_data = node.properties
                    from .decision_models import Decision
                    timestamp = self._normalize_timestamp(decision_data.get("timestamp"))
                    decision = Decision(
                        decision_id=pid,
                        category=decision_data.get("category", ""),
                        scenario=decision_data.get("scenario", node.content),
                        reasoning=decision_data.get("reasoning", ""),
                        outcome=decision_data.get("outcome", ""),
                        confidence=decision_data.get("confidence", 0.0),
                        timestamp=timestamp,
                        decision_maker=decision_data.get("decision_maker", ""),
                        reasoning_embedding=decision_data.get("reasoning_embedding"),
                        node2vec_embedding=decision_data.get("node2vec_embedding"),
                        valid_from=node.valid_from,
                        valid_until=node.valid_until,
                        metadata={k: v for k, v in decision_data.items() if k not in [
                            "category", "scenario", "reasoning", "outcome", "confidence", 
                            "timestamp", "decision_maker", "reasoning_embedding", "node2vec_embedding"
                        ]}
                    )
                    decisions.append(decision)
        
        return decisions

    
    # Enhanced methods for comprehensive context graphs
    def analyze_graph_with_kg(self) -> Dict[str, Any]:
        """
        Analyze the context graph using advanced KG algorithms.
        
        Returns:
            Comprehensive graph analysis results
        """
        if not self.kg_components:
            self.logger.warning("KG components not available")
            return {"error": "Advanced features not available"}
        
        try:
            analysis = {
                "graph_metrics": {},
                "centrality_analysis": {},
                "community_analysis": {},
                "connectivity_analysis": {},
                "node_embeddings": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert to KG-compatible format
            kg_graph = self._to_kg_format()
            
            # Basic graph metrics
            analysis["graph_metrics"] = {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": self._get_node_type_distribution(),
                "edge_types": self._get_edge_type_distribution()
            }
            
            # Centrality analysis
            if "centrality_calculator" in self.kg_components:
                centrality = self.kg_components["centrality_calculator"].calculate_all_centrality(kg_graph)
                analysis["centrality_analysis"] = centrality
            
            # Community detection
            if "community_detector" in self.kg_components:
                community_detector = self.kg_components["community_detector"]
                community_result = community_detector.detect_communities(kg_graph)
                communities = community_result["communities"]
                analysis["community_analysis"] = {
                    "communities": communities,
                    "num_communities": len(communities),
                    "modularity": community_result.get("modularity", 0.0)
                }
            
            # Connectivity analysis
            if "connectivity_analyzer" in self.kg_components:
                connectivity = self.kg_components["connectivity_analyzer"].analyze_connectivity(kg_graph)
                analysis["connectivity_analysis"] = connectivity
            
            # Node embeddings
            if "node_embedder" in self.kg_components:
                node_labels = list(self.node_type_index.keys())
                relationship_types = list(self.edge_type_index.keys())
                if node_labels:
                    embeddings = self.kg_components["node_embedder"].compute_embeddings(
                        graph_store=self,
                        node_labels=node_labels,
                        relationship_types=relationship_types,
                    )
                    analysis["node_embeddings"] = embeddings
            
            self.logger.info("Completed comprehensive graph analysis")
            return analysis
            
        except AttributeError as e:
            # A broken internal method call (e.g. calling a method that doesn't
            # exist on one of the kg_components) is a programming error, not a
            # legitimate empty-analysis result. Log it distinctly and re-raise
            # rather than masking it under the generic message below.
            self.logger.error(f"Graph analysis failed due to a broken internal method call: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to analyze graph with KG: {e}")
            return {"error": "Graph analysis failed due to an internal error"}

    
    def get_node_centrality(self, node_id: str) -> Dict[str, float]:
        """
        Get centrality measures for a specific node.
        
        Args:
            node_id: Node ID to analyze
            
        Returns:
            Dictionary of centrality measures
        """
        if "centrality_calculator" not in self.kg_components:
            return {"error": "Centrality calculator not available"}
        
        if node_id not in self.nodes:
            return {"error": "Node not found"}
        
        # Check cache first
        cache_key = f"centrality_{node_id}"
        if cache_key in self._analytics_cache:
            return self._analytics_cache[cache_key]
        
        try:
            # Get subgraph around the node
            subgraph = self._get_node_subgraph(node_id, max_depth=2)
            
            # Calculate centrality
            centrality = self.kg_components["centrality_calculator"].calculate_all_centrality(subgraph)
            
            # Cache result
            self._analytics_cache[cache_key] = centrality.get(node_id, {})
            
            return centrality.get(node_id, {})
            
        except Exception as e:
            self.logger.error(f"Failed to get node centrality: {e}")
            return {"error": "Node centrality calculation failed due to an internal error"}

    
    def find_similar_nodes(
        self, node_id: str, similarity_type: str = "content", top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar nodes using various similarity measures.
        
        Args:
            node_id: Reference node ID
            similarity_type: Type of similarity ("embedding", "structural", "content")
            top_k: Number of similar nodes to return
            
        Returns:
            List of dicts with node ID, type, content, and similarity score
        """
        if node_id not in self.nodes:
            return []
        
        similar_nodes = []
        reference_node = self.nodes[node_id]
        
        try:
            for other_id, other_node in self.nodes.items():
                if other_id != node_id:
                    if similarity_type == "content":
                        similarity = self._calculate_content_similarity(reference_node, other_node)
                    elif similarity_type == "structural":
                        similarity = self._calculate_structural_similarity(reference_node, other_node)
                    else:
                        similarity = self._calculate_content_similarity(reference_node, other_node)
                    
                    similar_nodes.append({
                        "id": other_id,
                        "content": other_node.content,
                        "type": other_node.node_type,
                        "score": similarity,
                    })

            # Sort by similarity and return top_k
            similar_nodes.sort(key=lambda x: x["score"], reverse=True)
            return similar_nodes[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar nodes: {e}")
            return []

    
    # Helper methods for KG integration
    def _to_kg_format(self) -> Dict[str, Any]:
        """Convert context graph to KG-compatible format."""
        nodes = []
        edges = []
        relationships = []
        
        # Convert nodes
        for node_id, node in self.nodes.items():
            nodes.append({
                "id": node_id,
                "type": node.node_type,
                "properties": node.properties,
                "content": node.content
            })
        
        # Convert edges
        for edge in self.edges:
            edge_data = {
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type,
                "weight": edge.weight,
                "properties": edge.metadata
            }
            edges.append(edge_data)
            relationships.append(edge_data)
        
        return {
            "nodes": nodes, 
            "edges": edges,
            "relationships": relationships  # KG algorithms expect this key
        }

    
    def _get_node_type_distribution(self) -> Dict[str, int]:
        """Get distribution of node types."""
        distribution = defaultdict(int)
        for node in self.nodes.values():
            distribution[node.node_type] += 1
        return dict(distribution)

    
    def _get_edge_type_distribution(self) -> Dict[str, int]:
        """Get distribution of edge types."""
        distribution = defaultdict(int)
        for edge in self.edges:
            distribution[edge.edge_type] += 1
        return dict(distribution)

    
    def _get_node_subgraph(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get subgraph around a node."""
        neighbors = self.get_neighbors(node_id, hops=max_depth)
        
        subgraph_nodes = {node_id}
        subgraph_edges = []
        
        for neighbor in neighbors:
            neighbor_id = neighbor["id"]
            subgraph_nodes.add(neighbor_id)
        
        # Add edges between nodes in subgraph
        for edge in self.edges:
            if edge.source_id in subgraph_nodes and edge.target_id in subgraph_nodes:
                subgraph_edges.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "weight": edge.weight
                })
        
        return {
            "nodes": [{"id": nid} for nid in subgraph_nodes],
            "edges": subgraph_edges
        }

    
    def _calculate_structural_similarity(self, node1: ContextNode, node2: ContextNode) -> float:
        """Calculate structural similarity between two nodes."""
        # Simple structural similarity based on node types and connections
        if node1.node_type != node2.node_type:
            return 0.0
        
        # Count connections
        connections1 = len(self._adjacency.get(node1.node_id, []))
        connections2 = len(self._adjacency.get(node2.node_id, []))
        
        # Similarity based on connection count similarity
        max_connections = max(connections1, connections2, 1)
        return 1.0 - abs(connections1 - connections2) / max_connections

    
    def _calculate_content_similarity(self, node1: ContextNode, node2: ContextNode) -> float:
        """Calculate content similarity between two nodes."""
        words1 = set(node1.content.lower().split())
        words2 = set(node2.content.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
