"""Decision recording, querying and governance methods for ContextGraph."""

import uuid
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from ..utils.helpers import classify_path_distance

from .context_graph_base import (
    _CAUSAL_TRAVERSAL_TYPES,
    _CausalChain,
    _normalize_temporal_input,
    _parse_iso_dt,
)


class _DecisionMixin:
    """Mixin composed into ContextGraph; methods use ``self``."""

    
    # --- Comprehensive Decision Management Features ---
    
    def record_decision(
        self,
        category: str,
        scenario: str,
        reasoning: str,
        outcome: str,
        confidence: float,
        entities: Optional[List[str]] = None,
        decision_maker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_from: Optional[Union[str, int, float, datetime]] = None,
        valid_until: Optional[Union[str, int, float, datetime]] = None,
        **kwargs
    ) -> str:
        """
        Record a decision with full context and analytics.
        
        Args:
            category: Decision category (e.g., "loan_approval")
            scenario: Decision scenario description
            reasoning: Decision reasoning explanation
            outcome: Decision outcome
            confidence: Confidence score (0.0 to 1.0)
            entities: Related entities
            decision_maker: Who made the decision
            metadata: Additional metadata
            **kwargs: Additional decision data
            
        Returns:
            Decision ID for reference
        """
        from datetime import datetime

        # Input validation
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Category must be a non-empty string")
        if len(category.strip()) > 100:
            raise ValueError("Category must be 100 characters or less")
        
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError("Scenario must be a non-empty string")
        if len(scenario.strip()) > 5000:
            raise ValueError("Scenario must be 5000 characters or less")
        
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Reasoning must be a non-empty string")
        if len(reasoning.strip()) > 10000:
            raise ValueError("Reasoning must be 10000 characters or less")
        
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("Outcome must be a non-empty string")
        if len(outcome.strip()) > 1000:
            raise ValueError("Outcome must be 1000 characters or less")
        
        if not isinstance(confidence, (int, float)):
            raise ValueError("Confidence must be a number")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if entities is not None:
            if not isinstance(entities, list):
                raise ValueError("Entities must be a list of strings")
            for entity in entities:
                if not isinstance(entity, str) or not entity.strip():
                    raise ValueError("Each entity must be a non-empty string")
                if len(entity.strip()) > 200:
                    raise ValueError("Each entity must be 200 characters or less")
        
        if decision_maker is not None:
            if not isinstance(decision_maker, str) or not decision_maker.strip():
                raise ValueError("Decision maker must be a non-empty string")
            if len(decision_maker.strip()) > 200:
                raise ValueError("Decision maker must be 200 characters or less")
        
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            for key, value in metadata.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Metadata keys must be non-empty strings")
                if len(key.strip()) > 100:
                    raise ValueError("Metadata keys must be 100 characters or less")
                if len(str(value)) > 1000:
                    raise ValueError("Metadata values must be 1000 characters or less")
        
        # Validate kwargs
        for key, value in kwargs.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Additional field names must be non-empty strings")
            if key == "id":
                # kwargs are merged last into the decision mapping, so an "id"
                # would replace the generated one and the returned id would no
                # longer name the stored node.
                raise ValueError("'id' is reserved; the decision id is generated")
            if len(key.strip()) > 100:
                raise ValueError("Additional field names must be 100 characters or less")
            if len(str(value)) > 1000:
                raise ValueError("Additional field values must be 1000 characters or less")
        
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        
        # Sanitize inputs
        category = category.strip()
        scenario = scenario.strip()
        reasoning = reasoning.strip()
        outcome = outcome.strip()
        confidence = float(confidence)
        entities = [entity.strip() for entity in (entities or []) if entity.strip()]
        decision_maker = decision_maker.strip() if decision_maker else None
        normalized_valid_from = _normalize_temporal_input(valid_from)
        normalized_valid_until = _normalize_temporal_input(valid_until)
        
        # Create decision record
        decision = {
            "id": decision_id,
            "category": category,
            "scenario": scenario,
            "reasoning": reasoning,
            "outcome": outcome,
            "confidence": confidence,
            "entities": entities,
            "decision_maker": decision_maker,
            "timestamp": timestamp,
            "recorded_at": datetime.utcnow().isoformat(),
            "valid_from": normalized_valid_from,
            "valid_until": normalized_valid_until,
            "metadata": metadata or {},
            **kwargs
        }
        
        # Store decision in graph
        self._add_decision_to_graph(decision)
        
        # Store in internal decision storage
        if not hasattr(self, '_decisions'):
            self._decisions = {}
            self._decision_index = defaultdict(set)
            self._entity_index = defaultdict(set)
            self._temporal_index = []
        
        self._decisions[decision_id] = decision
        self._decision_index[category].add(decision_id)
        
        for entity in entities or []:
            self._entity_index[entity].add(decision_id)
        
        self._temporal_index.append((decision_id, timestamp))
        self._temporal_index.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Recorded decision {decision_id} in category {category}")
        return decision_id

    
    def find_precedents_by_scenario(
        self,
        scenario: str,
        category: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        use_semantic_search: bool = True,
        include_superseded: bool = False,
        as_of: Optional[Union[str, int, float, datetime]] = None,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Find similar decisions (precedents) using hybrid search.
        
        Args:
            scenario: Scenario to find precedents for
            category: Filter by decision category
            limit: Maximum number of precedents
            similarity_threshold: Minimum similarity score
            use_semantic_search: Use vector embeddings for search
            **filters: Additional filters
            
        Returns:
            List of similar decisions with similarity scores
        """
        if not hasattr(self, '_decisions') or not self._decisions:
            return []
        as_of_time = self._normalize_timestamp(as_of) if as_of is not None else None
        
        candidates = set()
        
        # Get candidates by category
        if category:
            candidates.update(self._decision_index.get(category, set()))
        else:
            candidates.update(self._decisions.keys())
        
        # Filter by entities if provided
        if "entities" in filters:
            entity_candidates = set()
            for entity in filters["entities"]:
                entity_candidates.update(self._entity_index.get(entity, set()))
            candidates = candidates.intersection(entity_candidates)
        
        # Reuse a query-local snapshot of node types and outgoing degrees. The
        # structural score is the maximum over all other nodes, so scanning and
        # sorting the entire graph once per decision is unnecessary.
        structural_scores = (
            self._precedent_structural_scores(candidates)
            if self.config.get("advanced_analytics") else {}
        )

        # Calculate similarities
        precedents = []
        for decision_id in candidates:
            decision = self._decisions[decision_id]
            if not self._decision_matches_temporal_filters(
                decision,
                include_superseded=include_superseded,
                as_of=as_of_time,
            ):
                continue
            
            # Content similarity
            content_sim = self._calculate_decision_content_similarity(scenario, decision)
            
            # Structural similarity (graph-based)
            structural_sim = structural_scores.get(decision_id, 0.0)
            
            # Combined similarity
            combined_sim = 0.7 * content_sim + 0.3 * structural_sim
            
            if combined_sim >= similarity_threshold:
                precedents.append({
                    "decision": decision,
                    "similarity": combined_sim,
                    "content_similarity": content_sim,
                    "structural_similarity": structural_sim
                })
        
        # Sort by similarity and limit
        precedents.sort(key=lambda x: x["similarity"], reverse=True)
        return precedents[:limit]

    
    def analyze_decision_influence(
        self,
        decision_id: str,
        max_depth: int = 3,
        include_indirect: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze decision influence and impact.
        
        Args:
            decision_id: Decision to analyze
            max_depth: Maximum depth for influence analysis
            include_indirect: Include indirect influences
            
        Returns:
            Influence analysis results
        """
        if not hasattr(self, '_decisions') or decision_id not in self._decisions:
            raise ValueError(f"Decision {decision_id} not found")
        
        decision = self._decisions[decision_id]
        
        # Direct influence (same entities, category)
        direct_influence = set()
        for entity in decision["entities"]:
            direct_influence.update(self._entity_index.get(entity, set()))
        direct_influence.discard(decision_id)
        direct_influence.update(self._decision_index.get(decision["category"], set()))
        direct_influence.discard(decision_id)

        # Explicit causal relationships recorded via add_causal_relationship() are
        # ground truth and always count as direct influence, in either direction.
        # The index is keyed by the raw edge_type string ("causes" and "CAUSED"
        # are separate keys), so filter by normalized type instead of iterating
        # a fixed spelling list.
        for edge_type, edges in self.edge_type_index.items():
            if edge_type.upper() not in _CAUSAL_TRAVERSAL_TYPES:
                continue
            for edge in edges:
                if edge.source_id == decision_id and edge.target_id in self._decisions:
                    direct_influence.add(edge.target_id)
                elif edge.target_id == decision_id and edge.source_id in self._decisions:
                    direct_influence.add(edge.source_id)
        
        # Indirect influence (through graph relationships)
        indirect_influence = set()
        if include_indirect and self.config.get("advanced_analytics"):
            indirect_influence = self._find_indirect_decision_influence(decision_id, max_depth) - direct_influence
        
        # Calculate influence scores
        influence_scores = {}
        for influenced_id in direct_influence | indirect_influence:
            influence_scores[influenced_id] = self._calculate_decision_influence_score(
                decision_id, influenced_id
            )
        
        # Sort by influence score
        sorted_influence = sorted(
            influence_scores.items(),
            key=lambda x: x[1].get("score", 0.0),
            reverse=True
        )
        
        def _enrich(did: str) -> Dict[str, Any]:
            dec = self._decisions.get(did, {})
            return {
                "decision_id": did,
                "scenario": dec.get("scenario", ""),
                "outcome": dec.get("outcome", ""),
                "category": dec.get("category", ""),
            }

        return {
            "decision_id": decision_id,
            "direct_influence": [_enrich(did) for did in direct_influence],
            "indirect_influence": [_enrich(did) for did in indirect_influence],
            "influence_scores": [
                {
                    **_enrich(did),
                    "score": details.get("score", 0.0),
                    "score_breakdown": {
                        "entity_overlap": details.get("entity_score", 0.0),
                        "category_match": details.get("category_score", 0.0),
                        "temporal_proximity": details.get("time_score", 0.0),
                    },
                    "is_direct": did in direct_influence,
                }
                for did, details in sorted_influence
            ],
            "total_influenced": len(influence_scores),
            "max_influence_score": max(
                details.get("score", 0.0) for details in influence_scores.values()
            ) if influence_scores else 0.0
        }

    
    def get_decision_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive insights about all decisions.
        
        Returns:
            Comprehensive analytics and insights
        """
        if not hasattr(self, '_decisions') or not self._decisions:
            return {"message": "No decisions recorded yet"}
        
        # Basic statistics
        total_decisions = len(self._decisions)
        categories = {}
        outcomes = {}
        confidence_scores = []
        
        for decision in self._decisions.values():
            # Category distribution
            categories[decision["category"]] = categories.get(decision["category"], 0) + 1
            
            # Outcome distribution
            outcomes[decision["outcome"]] = outcomes.get(decision["outcome"], 0) + 1
            
            # Confidence scores
            confidence_scores.append(decision["confidence"])
        
        # Advanced analytics (if available)
        advanced_insights = {}
        if self.config.get("advanced_analytics"):
            advanced_insights = self.analyze_graph_with_kg()
        
        # Temporal analysis
        temporal_insights = self._get_decision_temporal_analysis()
        
        # Entity analysis
        entity_insights = self._get_decision_entity_analysis()
        
        return {
            "total_decisions": total_decisions,
            "categories": categories,
            "outcomes": outcomes,
            "confidence_stats": {
                "mean": sum(confidence_scores) / len(confidence_scores),
                "min": min(confidence_scores),
                "max": max(confidence_scores),
                "median": sorted(confidence_scores)[len(confidence_scores) // 2]
            },
            "advanced_analytics": advanced_insights,
            "temporal_analysis": temporal_insights,
            "entity_analysis": entity_insights,
            "graph_metrics": self.get_graph_metrics() if hasattr(self, 'get_graph_metrics') else {}
        }

    
    def trace_decision_causality(
        self,
        decision_id: str,
        max_depth: int = 5,
        max_chains: Optional[int] = 10000
    ) -> List[Dict[str, Any]]:
        """
        Trace causal chain for a decision.

        Args:
            decision_id: Decision to trace
            max_depth: Maximum depth for causal analysis
            max_chains: Maximum number of chains to return. Densely connected
                graphs can contain a combinatorial number of distinct causal
                paths, so the traversal stops once this many chains have been
                collected and appends a ``{"truncated": True, ...}`` marker so
                callers can tell the trace is incomplete. Pass None for no limit.

        Returns:
            Causal chain as list of decision relationships
        """
        if not hasattr(self, '_decisions') or decision_id not in self._decisions:
            raise ValueError(f"Decision {decision_id} not found")

        try:
            # Use graph traversal to find causal relationships
            causal_chain = []
            chain_limit = float("inf") if max_chains is None else max_chains
            truncated = False

            # Reverse index of explicit causal edges, built once per call so the
            # traversal does not rescan the edge list at every visited node.
            # Edges may reference decision nodes that were never recorded through
            # record_decision() (e.g. a graph restored via from_dict), so only
            # causes with a known decision record are kept.
            incoming_causal_edges = defaultdict(list)
            # The index is keyed by the raw edge_type string ("causes" and
            # "CAUSED" are separate keys), so filter by normalized type.
            for edge_type, edges in self.edge_type_index.items():
                if edge_type.upper() not in _CAUSAL_TRAVERSAL_TYPES:
                    continue
                for edge in edges:
                    if edge.source_id in self._decisions:
                        incoming_causal_edges[edge.target_id].append(edge)

            def record_chain(cause_path):
                """Record one chain. Returns False once the cap is reached."""
                nonlocal truncated
                if len(causal_chain) >= chain_limit:
                    truncated = True
                    return False
                causal_chain.append(
                    self._build_causal_chain_report(list(reversed(cause_path)))
                )
                return True

            def trace_recursive(current_id, depth, path, path_ids):
                # Cycle detection is per-path rather than global: a decision reached
                # through one branch must stay traversable through another, otherwise
                # branching graphs silently lose valid chains. max_depth bounds the
                # traversal.
                if truncated or depth >= max_depth or current_id in path_ids:
                    return

                path_ids = path_ids | {current_id}
                current_decision = self._decisions[current_id]

                # Explicit causal relationships recorded via add_causal_relationship()
                # take precedence - they are the ground truth the caller recorded.
                # Every edge is traced, so parallel relationships between the same
                # pair of decisions are all reported rather than overwriting.
                explicit_causes = incoming_causal_edges.get(current_id, [])
                explicit_cause_ids = {edge.source_id for edge in explicit_causes}

                for edge in explicit_causes:
                    cause_id = edge.source_id
                    cause_dec = self._decisions[cause_id]
                    weight = getattr(edge, "weight", None)
                    # A stored weight of 0.0 is meaningful and must not be coerced
                    # to the 1.0 default.
                    edge_weight = 1.0 if weight is None else float(weight)
                    hop = {
                        "from": cause_id,
                        "from_scenario": cause_dec.get("scenario", ""),
                        "to": current_id,
                        "to_scenario": current_decision.get("scenario", ""),
                        "type": edge.edge_type,
                        "edge_weight": edge_weight,
                    }
                    cause_path = path + [hop]
                    if not record_chain(cause_path):
                        return
                    trace_recursive(cause_id, depth + 1, cause_path, path_ids)
                    if truncated:
                        return

                # Find potential causes (decisions that influenced this one) via
                # shared entities/timestamps - additive heuristic, skipping anything
                # already covered by an explicit relationship above. Deduplicate by
                # decision id (dict preserves insertion order): a decision sharing
                # several entities with the current one is one potential cause,
                # not one per shared entity, otherwise the trace reports the same
                # "influences" chain once per overlapping entity.
                potential_causes = {}
                for entity in current_decision["entities"]:
                    for other_decision_id in self._entity_index.get(entity, set()):
                        if other_decision_id != current_id and other_decision_id not in explicit_cause_ids:
                            other_decision = self._decisions[other_decision_id]
                            if self._decision_sort_ts(other_decision["timestamp"]) < self._decision_sort_ts(current_decision["timestamp"]):
                                potential_causes[other_decision_id] = None

                for cause_id in potential_causes:
                    cause_dec = self._decisions.get(cause_id, {})
                    edge_weight = float(cause_dec.get("confidence", 1.0))
                    hop = {
                        "from": cause_id,
                        "from_scenario": cause_dec.get("scenario", ""),
                        "to": current_id,
                        "to_scenario": current_decision.get("scenario", ""),
                        "type": "influences",
                        "edge_weight": edge_weight,
                    }
                    cause_path = path + [hop]
                    if not record_chain(cause_path):
                        return
                    trace_recursive(cause_id, depth + 1, cause_path, path_ids)
                    if truncated:
                        return

            trace_recursive(decision_id, 0, [], frozenset())

            if truncated:
                # Never drop chains silently: the caller is told the trace is partial.
                self.logger.warning(
                    "Causal trace for %s truncated at %s chains; "
                    "raise max_chains or lower max_depth for a complete trace.",
                    decision_id,
                    max_chains,
                )
                causal_chain.append({
                    "truncated": True,
                    "max_chains": max_chains,
                    "message": (
                        f"Causal trace truncated at {max_chains} chains. "
                        "The result is incomplete; raise max_chains or lower "
                        "max_depth for a complete trace."
                    ),
                })

            return causal_chain
            
        except Exception as e:
            self.logger.error(f"Causal analysis failed: {e}")
            return [{"error": "Causal analysis failed due to an internal error"}]

    
    def enforce_decision_policy(
        self,
        decision_data: Dict[str, Any],
        policy_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enforce policies on decision data.
        
        Args:
            decision_data: Decision data to check
            policy_rules: Policy rules to enforce
            
        Returns:
            Policy enforcement results
        """
        # Simple policy enforcement implementation
        violations = []
        warnings = []
        
        # Default policy rules
        default_rules = {
            "min_confidence": 0.7,
            "required_outcomes": ["approved", "rejected", "flagged"],
            "required_metadata": ["decision_maker"],
            "max_reasoning_length": 10000
        }
        
        rules = policy_rules or default_rules
        
        # Check confidence
        if decision_data.get("confidence", 0) < rules.get("min_confidence", 0.7):
            violations.append(f"Confidence too low: {decision_data.get('confidence', 0)}")
        
        # Check outcome
        if decision_data.get("outcome") not in rules.get("required_outcomes", []):
            violations.append(f"Invalid outcome: {decision_data.get('outcome')}")
        
        # Check required metadata
        for required_field in rules.get("required_metadata", []):
            if not decision_data.get(required_field):
                violations.append(f"Missing required field: {required_field}")
        
        # Check reasoning length
        reasoning = decision_data.get("reasoning", "")
        if len(reasoning.strip()) > rules.get("max_reasoning_length", 10000):
            warnings.append(f"Reasoning too long: {len(reasoning.strip())} characters")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "policy_rules": rules
        }

    
    # --- Private helper methods for decision management ---
    
    def _add_decision_to_graph(self, decision: Dict[str, Any]) -> None:
        """Add decision to context graph.

        All-or-nothing: on any failure the nodes and edges created by this call
        are removed again and the exception is re-raised, so a decision is
        never left half-stored and ``record_decision`` never reports success
        for one that is not in the graph.
        """
        # Every name add_node() receives other than through **properties, plus
        # the keywords passed explicitly below. A metadata key or **kwargs
        # entry under one of these names would arrive twice and raise
        # TypeError, so both mappings are filtered by the same set. "self" is
        # included because a bound method rejects it as a keyword too.
        node_properties = {
            "self",
            "node_id",
            "node_type",
            "content",
            "category",
            "scenario",
            "reasoning",
            "outcome",
            "confidence",
            "timestamp",
            "decision_maker",
            "recorded_at",
            "valid_from",
            "valid_until",
        }
        # Keys of the decision mapping handled structurally, not as node
        # properties.
        structural_keys = {"id", "entities", "metadata"}
        safe_metadata = {
            key: value
            for key, value in (decision.get("metadata") or {}).items()
            if key not in node_properties
        }
        extra_properties = {
            key: value
            for key, value in decision.items()
            if key not in node_properties and key not in structural_keys
        }

        with self._lock:
            nodes_before = set(self.nodes)
            edges_before = len(self.edges)
            try:
                # Add decision node
                if not self.add_node(
                    decision["id"],
                    "decision",
                    content=decision["scenario"],
                    valid_from=decision.get("valid_from"),
                    valid_until=decision.get("valid_until"),
                    category=decision["category"],
                    outcome=decision["outcome"],
                    confidence=decision["confidence"],
                    timestamp=decision["timestamp"],
                    scenario=decision["scenario"],
                    decision_maker=decision.get("decision_maker", ""),
                    reasoning=decision["reasoning"],
                    recorded_at=decision.get("recorded_at", ""),
                    **safe_metadata,
                    **extra_properties,
                ):
                    raise ValueError(
                        f"Decision node {decision['id']!r} was rejected by the graph"
                    )

                # Add entity nodes and relationships
                for entity in decision["entities"]:
                    # Add entity node if not exists
                    if not self.find_node(entity):
                        self.add_node(
                            entity,
                            "entity",
                            name=entity
                        )

                    # Add relationship
                    self.add_edge(
                        decision["id"],
                        entity,
                        "involves",
                        confidence=decision["confidence"]
                    )

                # Add category node and relationship
                category_id = f"category_{decision['category']}"
                if not self.find_node(category_id):
                    self.add_node(
                        category_id,
                        "category",
                        name=decision["category"]
                    )

                self.add_edge(
                    decision["id"],
                    category_id,
                    "belongs_to"
                )

                # Add decision maker node if provided
                if decision.get("decision_maker"):
                    maker_id = f"maker_{decision['decision_maker']}"
                    if not self.find_node(maker_id):
                        self.add_node(
                            maker_id,
                            "decision_maker",
                            name=decision["decision_maker"]
                        )

                    self.add_edge(
                        decision["id"],
                        maker_id,
                        "made_by"
                    )

            except Exception:
                # Do not swallow this. record_decision() has already generated
                # the id it is about to return, so a silent failure here hands
                # the caller a plausible identifier for a decision that is not
                # in the graph. Undo the partial write first so a retry does
                # not stack a second decision on top of orphaned nodes/edges.
                self.logger.exception("Failed to add decision to graph")
                for edge in list(self.edges[edges_before:]):
                    self._drop_edge_from_indexes(edge)
                    self._edge_index.pop(edge.edge_id, None)
                for node_id in set(self.nodes) - nodes_before:
                    self._drop_node_from_indexes(node_id)
                raise

    def _decision_matches_temporal_filters(
        self,
        decision: Dict[str, Any],
        include_superseded: bool = False,
        as_of: Optional[datetime] = None,
    ) -> bool:
        """Return True when a decision matches temporal filter rules."""
        valid_from = _parse_iso_dt(decision.get("valid_from")) if decision.get("valid_from") else None
        valid_until = _parse_iso_dt(decision.get("valid_until")) if decision.get("valid_until") else None
        reference_time = as_of or datetime.utcnow()

        if as_of is not None:
            if valid_from is not None and reference_time < valid_from:
                return False
            if valid_until is not None and reference_time > valid_until:
                return False
            return True

        if include_superseded:
            return True

        if valid_until is not None and reference_time > valid_until:
            return False
        return True

    
    # ── decision-index helpers ────────────────────────────────────────────────

    # Protected set of node properties whose values are *core* decision fields
    # so that we can distinguish them from user-supplied metadata when
    # rebuilding the in-memory indexes from a persisted node.
    _DECISION_CORE_FIELDS: frozenset = frozenset({
        "id", "category", "scenario", "reasoning", "outcome", "confidence",
        "entities", "decision_maker", "timestamp", "recorded_at",
        "valid_from", "valid_until", "content",
    })

    @staticmethod
    def _decision_sort_ts(raw_ts: Any) -> float:
        """Epoch seconds for ``_temporal_index`` ordering.

        Accepts epoch numbers (``record_decision``) and ISO strings
        (``add_decision`` with a ``Decision`` object); anything unparseable
        sorts as ``0.0``. A trailing ``Z`` is read as UTC; other naive values
        use local time, matching ``record_decision``'s ``datetime.now()``.
        """
        if isinstance(raw_ts, datetime):
            return raw_ts.timestamp()
        try:
            return float(raw_ts)
        except (TypeError, ValueError):
            pass
        if isinstance(raw_ts, str):
            try:
                text = raw_ts.strip()
                if text.endswith(("Z", "z")):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text).timestamp()
            except ValueError:
                pass
        return 0.0

    def _rebuild_decision_indexes(self) -> None:
        """Rebuild all derived decision indexes from the current node store.

        This method is the single authoritative rebuild path.  It must be
        called (under the graph lock) after any operation that wholesale
        replaces ``self.nodes`` — namely ``load_from_file`` (JSON and Markdown
        paths) and ``from_dict``.

        Contract:
        - Unconditionally clears ``_decisions``, ``_decision_index``,
          ``_entity_index``, and ``_temporal_index`` before rebuilding so that
          repeated calls never accumulate stale entries.
        - Derives ``_decisions[node_id]["metadata"]`` from the full set of
          node properties, excluding the protected core fields, so that
          user-supplied metadata survives the round-trip.
        - Runs under ``self._lock`` when called from load paths; callers that
          already hold the lock must invoke ``_rebuild_decision_indexes``
          inside the lock block.
        """
        # Always start fresh so repeated loads don't accumulate stale entries.
        self._decisions: Dict[str, Any] = {}
        self._decision_index: Dict[str, set] = defaultdict(set)
        self._entity_index: Dict[str, set] = defaultdict(set)
        self._temporal_index: List[Tuple[str, float]] = []

        for node in self.nodes.values():
            if (getattr(node, "node_type", None) or "").lower() != "decision":
                continue

            # Merge metadata and properties; properties win on collision.
            meta: Dict[str, Any] = {}
            meta.update(getattr(node, "metadata", {}) or {})
            meta.update(getattr(node, "properties", {}) or {})

            # Timestamp: keep whatever was stored (float epoch or ISO string).
            # The temporal index uses it for sorting; downstream code handles
            # both types via _normalize_timestamp.
            raw_ts = meta.get("timestamp", 0.0)
            sort_ts = self._decision_sort_ts(raw_ts)

            # Entities may be stored as a list in meta or inferred from
            # outgoing "involves" edges if the list field is absent/empty.
            # _add_decision_to_graph creates entity nodes connected via
            # "involves" edges; it does NOT store the list as a node property.
            entities = meta.get("entities") or []
            if not isinstance(entities, list):
                entities = []
            if not entities:
                # Recover entity list from "involves" edges on this decision node
                for edge in self._adjacency.get(node.node_id, []):
                    if edge.edge_type == "involves":
                        entities.append(edge.target_id)

            # Everything that isn't a core field is user-supplied metadata.
            extra_meta = {
                k: v
                for k, v in meta.items()
                if k not in self._DECISION_CORE_FIELDS
            }

            decision: Dict[str, Any] = {
                "id": node.node_id,
                "category": meta.get("category", ""),
                "scenario": meta.get("scenario", getattr(node, "content", "") or ""),
                "reasoning": meta.get("reasoning", ""),
                "outcome": meta.get("outcome", ""),
                "confidence": float(meta.get("confidence", 0.0) or 0.0),
                "entities": entities,
                "decision_maker": meta.get("decision_maker"),
                "timestamp": raw_ts,
                "recorded_at": meta.get("recorded_at", ""),
                "valid_from": getattr(node, "valid_from", None),
                "valid_until": getattr(node, "valid_until", None),
                # Preserve all non-core node properties as decision metadata so
                # that user-supplied fields survive a save → load round-trip.
                "metadata": extra_meta,
            }

            self._decisions[node.node_id] = decision

            category = decision["category"]
            if category:
                self._decision_index[category].add(node.node_id)

            for entity in entities:
                self._entity_index[entity].add(node.node_id)

            self._temporal_index.append((node.node_id, sort_ts))

        self._temporal_index.sort(key=lambda x: x[1], reverse=True)

    def _sync_decision_from_node(self, node_id: str) -> None:
        """Synchronise a single decision index entry from the node store.

        Called after ``add_node_attribute`` mutates a decision node and after
        ``apply_node_markdown`` replaces a node whose old or new type is
        ``"decision"``, so that ``_decisions`` and the derived indexes stay
        consistent without requiring a full rebuild of all decisions.

        Temporal index cleanup (``_temporal_index``) runs unconditionally
        before the node-type guard so that stale entries are removed even when
        transitioning a decision node to a non-decision type.  Callers are
        expected to ensure this is only invoked when at least one of the
        current or previous node types is ``"decision"``; callers that bypass
        that invariant will have ``node_id`` silently removed from
        ``_temporal_index`` even if it was never a decision node.
        """
        node = self.nodes.get(node_id)
        if not hasattr(self, "_decisions"):
            # Indexes don't exist yet — a full rebuild is safer.
            self._rebuild_decision_indexes()
            return

        # Remove stale index entries before deciding whether the current node
        # still belongs in the decision indexes.
        old = self._decisions.pop(node_id, None)
        if old:
            old_cat = old.get("category", "")
            if old_cat:
                self._decision_index[old_cat].discard(node_id)
            for ent in old.get("entities", []):
                self._entity_index[ent].discard(node_id)
        self._temporal_index = [
            (nid, ts) for nid, ts in self._temporal_index if nid != node_id
        ]

        if node is None:
            return
        if (getattr(node, "node_type", None) or "").lower() != "decision":
            return

        # Rebuild the entry for this node and re-insert index entries.
        meta: Dict[str, Any] = {}
        meta.update(getattr(node, "metadata", {}) or {})
        meta.update(getattr(node, "properties", {}) or {})

        raw_ts = meta.get("timestamp", 0.0)
        sort_ts = self._decision_sort_ts(raw_ts)

        entities = meta.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        if not entities:
            # Recover entity list from "involves" edges
            for edge in self._adjacency.get(node_id, []):
                if edge.edge_type == "involves":
                    entities.append(edge.target_id)

        extra_meta = {
            k: v for k, v in meta.items() if k not in self._DECISION_CORE_FIELDS
        }

        decision: Dict[str, Any] = {
            "id": node_id,
            "category": meta.get("category", ""),
            "scenario": meta.get("scenario", getattr(node, "content", "") or ""),
            "reasoning": meta.get("reasoning", ""),
            "outcome": meta.get("outcome", ""),
            "confidence": float(meta.get("confidence", 0.0) or 0.0),
            "entities": entities,
            "decision_maker": meta.get("decision_maker"),
            "timestamp": raw_ts,
            "recorded_at": meta.get("recorded_at", ""),
            "valid_from": getattr(node, "valid_from", None),
            "valid_until": getattr(node, "valid_until", None),
            "metadata": extra_meta,
        }

        self._decisions[node_id] = decision
        if decision["category"]:
            self._decision_index[decision["category"]].add(node_id)
        for ent in entities:
            self._entity_index[ent].add(node_id)
        self._temporal_index.append((node_id, sort_ts))
        self._temporal_index.sort(key=lambda x: x[1], reverse=True)

    @staticmethod
    def _char_bigrams(text: str) -> set:
        """Character bigrams over whitespace-stripped text (CJK fallback).

        Strips whitespace so CJK characters without word-separating spaces are
        treated as a contiguous character sequence rather than a single token.
        """
        chars = "".join(text.lower().split())
        return {chars[i:i + 2] for i in range(len(chars) - 1)}

    @staticmethod
    def _looks_cjk(text: str) -> bool:
        """True if text contains CJK/Japanese/Korean script characters.

        Used to gate the character-bigram similarity fallback so it only
        activates for scripts where whitespace tokenisation doesn't work.
        """
        for ch in text:
            code = ord(ch)
            if (
                0x4E00 <= code <= 0x9FFF     # CJK Unified Ideographs
                or 0x3400 <= code <= 0x4DBF  # CJK Extension A
                or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
                or 0xAC00 <= code <= 0xD7A3  # Hangul Syllables
                or 0x1100 <= code <= 0x11FF  # Hangul Jamo
            ):
                return True
        return False

    def _calculate_decision_content_similarity(self, scenario: str, decision: Dict[str, Any]) -> float:
        """Calculate content similarity between scenario and decision.

        Signal hierarchy (applies to both the word and bigram channels):

          PRIMARY   — Jaccard(query, decision.scenario)
            The scenario field is the authoritative description of what the
            decision was about.  Scoring against it first means a query that
            is identical or very close to a stored scenario always scores high,
            regardless of how long the reasoning or entity list is.

          SECONDARY — 0.8 × Jaccard(query, scenario + reasoning + entities)
            The full decision text is a useful fallback when the query overlaps
            with reasoning or entity context rather than the scenario title.
            The 0.8× discount is intentional: decisions whose scenario is
            unrelated to the query but whose reasoning happens to restate the
            query words are ranked lower than genuine scenario-level matches.
            This also prevents verbose reasoning from diluting exact-scenario
            queries below any reasonable threshold (#1140).

            Practical consequence: a reasoning-only match scores at most
            0.8 × full_text_jaccard.  At the find_similar_decisions default
            threshold of 0.3 (combined_sim = 0.7 × content_sim), the
            full_text_jaccard must be ≥ 0.54 to survive, compared to ≥ 0.43
            without the discount.  Callers that depend on reasoning-heavy
            matching should lower their threshold accordingly.

          FINAL     — max(primary, secondary)
            The higher of the two signals wins so that neither channel can
            suppress a strong match from the other.

        Uses word-level Jaccard for space-separated languages.  For text where
        whitespace tokenisation is unreliable (CJK/Japanese/Korean scripts, or
        a query with no whitespace at all) a character-bigram Jaccard is
        computed over the *stripped* character sequences instead.

        The bigram fallback only activates when whitespace tokenisation would
        not help — i.e. the query is CJK-like or has at most one whitespace
        token — so it never contributes for ordinary multi-word English
        queries, where incidental bigram overlap between unrelated sentences
        would otherwise inflate scores.

        The bigram side uses *Jaccard* (|A∩B|/|A∪B|), not the overlap
        coefficient, so a 2-character query whose single bigram happens to
        appear anywhere in a long document does not silently receive a score of
        1.0.  A minimum bigram set size of 3 is required before the bigram
        signal contributes; this prevents 1- and 2-character English queries
        from polluting results while still allowing 3-character CJK phrases (2
        bigrams) to match.
        """
        try:
            decision_text = (
                f"{decision['scenario']} {decision['reasoning']} "
                f"{' '.join(decision['entities'])}"
            )
            decision_scenario_text = str(decision.get("scenario", ""))

            def _word_jaccard(text: str) -> float:
                words = set(text.lower().split())
                union = scenario_words | words
                return len(scenario_words & words) / len(union) if union else 0.0

            # --- word-level Jaccard (primary metric for Latin/space-delimited) ---
            scenario_words = set(scenario.lower().split())
            # Jaccard against the full decision text is diluted by the reasoning
            # and entity words: a query naming a decision's exact scenario could
            # score below every reasonable threshold and return no precedent at
            # all (#1140). Score the decision's own scenario first and keep the
            # full-text score as a discounted secondary signal.
            word_sim = _word_jaccard(decision_scenario_text)
            full_text_word_sim = _word_jaccard(decision_text)
            word_sim = max(word_sim, 0.8 * full_text_word_sim)

            # --- character-bigram Jaccard (CJK / very-short-query fallback) ---
            # Only used when whitespace tokenisation can't do the job: CJK-like
            # scripts, or a query that is a single whitespace token (no spaces
            # to split on).  Ordinary multi-word English queries rely on
            # word_sim alone, so incidental bigram overlap between unrelated
            # sentences can never inflate their score.
            bigram_sim = 0.0
            needs_bigram_fallback = (
                self._looks_cjk(scenario) or len(scenario.split()) <= 1
            )
            if needs_bigram_fallback:
                scenario_bigrams = self._char_bigrams(scenario)

                def _bigram_jaccard(text: str) -> float:
                    bigrams = self._char_bigrams(text)
                    if not bigrams:
                        return 0.0
                    union = scenario_bigrams | bigrams
                    return len(scenario_bigrams & bigrams) / len(union) if union else 0.0

                # Require at least 3 bigrams in the query before the bigram
                # signal is used.  A 2-char query produces only 1 bigram; that
                # single bigram is far too likely to appear as a substring of
                # any English word and would produce a spuriously high overlap
                # coefficient.  3 bigrams correspond to a 4-char stripped query
                # (e.g. two CJK characters produce 1 bigram each → need ≥3
                # chars stripped).
                if len(scenario_bigrams) >= 3:
                    # Same scenario-vs-full-text treatment as the word channel
                    # above: the decision's own scenario must not be diluted
                    # below threshold by its reasoning text (#1140).
                    bigram_sim = max(
                        _bigram_jaccard(decision_scenario_text),
                        0.8 * _bigram_jaccard(decision_text),
                    )

            return max(word_sim, bigram_sim)

        except Exception:
            self.logger.exception("Content similarity calculation failed")
            return 0.0

    
    def _precedent_structural_scores(self, decision_ids: Set[str]) -> Dict[str, float]:
        """Return the existing structural score without per-decision graph scans.

        ``find_similar_nodes(..., similarity_type="structural")`` compares node
        type and outgoing degree. Its highest score is therefore determined by
        the nearest *other* degree of the same type. Keep this index local to the
        call so graph mutations and reloads cannot leave stale scores behind.
        """
        degree_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        decision_degrees: Dict[str, Tuple[str, int]] = {}
        with self._lock:
            for node_id, node in self.nodes.items():
                degree_counts[node.node_type][len(self._adjacency.get(node_id, ()))] += 1
            for decision_id in decision_ids:
                node = self.nodes.get(decision_id)
                if node is not None:
                    decision_degrees[decision_id] = (
                        node.node_type,
                        len(self._adjacency.get(decision_id, ())),
                    )

        sorted_degrees = {
            node_type: sorted(counts)
            for node_type, counts in degree_counts.items()
        }
        scores: Dict[str, float] = {}
        for decision_id, (node_type, degree) in decision_degrees.items():
            counts = degree_counts[node_type]
            if counts[degree] > 1:
                scores[decision_id] = 1.0
                continue
            degrees = sorted_degrees[node_type]
            position = bisect_left(degrees, degree)
            lower = degrees[position - 1] if position > 0 else None
            upper = degrees[position + 1] if position + 1 < len(degrees) else None
            scores[decision_id] = max(
                (1.0 - abs(degree - other) / max(degree, other, 1)
                 for other in (lower, upper) if other is not None),
                default=0.0,
            )
        return scores

    def _calculate_structural_similarity_for_decision(self, decision_id: str, scenario: str) -> float:
        """Calculate structural similarity using graph algorithms."""
        try:
            if not self.config.get("advanced_analytics"):
                return 0.0
            
            # Use graph similarity algorithms
            similar_nodes = self.find_similar_nodes(
                decision_id,
                similarity_type="structural",
                top_k=5
            )
            
            if similar_nodes:
                return max(
                    item.get("score", 0.0)
                    for item in similar_nodes
                    if isinstance(item, dict)
                )
            
        except Exception as e:
            self.logger.exception("Structural similarity calculation failed")
        
        return 0.0

    
    def _find_indirect_decision_influence(self, decision_id: str, max_depth: int) -> Set[str]:
        """Find indirect influences using graph traversal."""
        try:
            influenced = set()
            
            # Get neighbors in graph
            neighbors = self.get_neighbors(decision_id, hops=max_depth)
            
            for neighbor in neighbors:
                if neighbor.get("type") == "decision":
                    influenced.add(neighbor["id"])
            
            return influenced
            
        except Exception as e:
            self.logger.warning(f"Indirect influence analysis failed: {e}")
            return set()

    
    def _build_causal_chain_report(self, hops: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build an auditable causal-chain response from hop records."""
        hop_count = len(hops)
        confidence_decay = 1.0
        weakest_link = None
        for hop in hops:
            edge_weight = float(hop.get("edge_weight", 1.0))
            confidence_decay *= edge_weight
            if weakest_link is None or edge_weight < float(weakest_link.get("edge_weight", 1.0)):
                weakest_link = hop

        if hop_count <= 1:
            interpretation = f"Direct influence with confidence {confidence_decay:.2f}."
        elif confidence_decay > 0.7:
            interpretation = (
                f"Mediated through {hop_count - 1} step(s) with high confidence "
                f"({confidence_decay:.2f})."
            )
        elif confidence_decay > 0.4:
            interpretation = (
                f"Mediated through {hop_count - 1} step(s) - confidence decays "
                f"to {confidence_decay:.2f}."
            )
        else:
            interpretation = (
                f"Distal influence across {hop_count} causal steps; confidence "
                f"{confidence_decay:.2f} is weak evidence."
            )

        return _CausalChain({
            "hops": hops,
            "hop_count": hop_count,
            "confidence_decay": confidence_decay,
            "weakest_link": weakest_link,
            "distance_band": classify_path_distance(hop_count),
            "interpretation": interpretation,
        })

    def _calculate_decision_influence_score(self, source_id: str, target_id: str) -> Dict[str, float]:
        """Calculate influence score between two decisions."""
        try:
            if not hasattr(self, '_decisions'):
                return {"score": 0.0, "entity_score": 0.0, "category_score": 0.0, "time_score": 0.0}
                
            source_decision = self._decisions[source_id]
            target_decision = self._decisions[target_id]
            
            # Base score from shared entities
            shared_entities = set(source_decision["entities"]) & set(target_decision["entities"])
            entity_score = len(shared_entities) / max(len(source_decision["entities"]), 1)
            
            # Category similarity
            category_score = 1.0 if source_decision["category"] == target_decision["category"] else 0.0
            
            # Temporal proximity (more recent decisions have higher influence)
            # Timestamps may be epoch floats (record_decision) or ISO strings
            # (add_decision with a Decision object); compare as epoch seconds.
            time_diff = abs(
                self._decision_sort_ts(source_decision["timestamp"])
                - self._decision_sort_ts(target_decision["timestamp"])
            )
            time_score = max(0.0, 1.0 - time_diff / (30 * 24 * 3600))  # 30 days window
            
            # Combined score
            combined_score = 0.5 * entity_score + 0.3 * category_score + 0.2 * time_score
            
            return {
                "score": combined_score,
                "entity_score": entity_score,
                "category_score": category_score,
                "time_score": time_score,
            }
            
        except Exception as e:
            self.logger.warning(f"Influence score calculation failed: {e}")
            return {"score": 0.0, "entity_score": 0.0, "category_score": 0.0, "time_score": 0.0}

    
    def _get_decision_temporal_analysis(self) -> Dict[str, Any]:
        """Get temporal analysis of decisions."""
        try:
            if not hasattr(self, '_temporal_index') or not self._temporal_index:
                return {}
            
            # Group decisions by time periods
            recent_decisions = [did for did, ts in self._temporal_index[:10]]
            
            return {
                "recent_decisions": len(recent_decisions),
                "oldest_decision": min(ts for _, ts in self._temporal_index),
                "newest_decision": max(ts for _, ts in self._temporal_index),
                "time_span": max(ts for _, ts in self._temporal_index) - min(ts for _, ts in self._temporal_index)
            }
            
        except Exception as e:
            self.logger.warning(f"Temporal analysis failed: {e}")
            return {}

    
    def _get_decision_entity_analysis(self) -> Dict[str, Any]:
        """Get entity analysis from decisions."""
        try:
            if not hasattr(self, '_decisions'):
                return {}
                
            entity_counts = {}
            for decision in self._decisions.values():
                for entity in decision["entities"]:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            # Get top entities
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_entities": len(entity_counts),
                "top_entities": top_entities,
                "avg_entities_per_decision": sum(len(d["entities"]) for d in self._decisions.values()) / len(self._decisions)
            }
            
        except Exception as e:
            self.logger.warning(f"Entity analysis failed: {e}")
            return {}
