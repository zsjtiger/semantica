"""Convenience wrappers for ContextGraph."""

from typing import Any, Dict, List, Optional



class _ConvenienceMixin:
    """Mixin composed into ContextGraph; methods use ``self``."""

    
    # --- Easy-to-Use Convenience Methods ---
    
    def add_decision_simple(
        self,
        category: str,
        scenario: str,
        reasoning: str,
        outcome: str,
        confidence: float = 0.5,
        entities: Optional[List[str]] = None,
        decision_maker: Optional[str] = "system",
        **kwargs
    ) -> str:
        """
        Easy way to record a decision.
        
        Args:
            category: Decision category (e.g., "loan_approval")
            scenario: What was the situation
            reasoning: Why was this decision made
            outcome: What was decided
            confidence: How confident (0.0 to 1.0)
            entities: Related entities (people, items, etc.)
            decision_maker: Who made the decision
            **kwargs: Additional information
            
        Returns:
            Decision ID for reference
        """
        return self.record_decision(
            category=category,
            scenario=scenario,
            reasoning=reasoning,
            outcome=outcome,
            confidence=confidence,
            entities=entities,
            decision_maker=decision_maker,
            metadata=kwargs
        )

    
    def find_similar_decisions(
        self,
        scenario: str,
        category: Optional[str] = None,
        max_results: int = 10,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Easy way to find similar past decisions.
        
        Args:
            scenario: What situation are you looking for
            category: Filter by decision type
            max_results: Maximum results to return
            min_similarity: Minimum similarity score
            
        Returns:
            List of similar decisions with similarity scores
        """
        return self.find_precedents_by_scenario(
            scenario=scenario,
            category=category,
            limit=max_results,
            similarity_threshold=min_similarity
        )

    
    def analyze_decision_impact(
        self,
        decision_id: str,
        include_indirect: bool = True
    ) -> Dict[str, Any]:
        """
        Easy way to analyze how a decision impacts others.
        
        Args:
            decision_id: Decision to analyze
            include_indirect: Include indirect impacts
            
        Returns:
            Impact analysis results
        """
        return self.analyze_decision_influence(
            decision_id=decision_id,
            max_depth=3,
            include_indirect=include_indirect
        )

    
    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Easy way to get a summary of all decisions.
        
        Returns:
            Summary statistics and insights
        """
        return self.get_decision_insights()

    
    def trace_decision_chain(
        self,
        decision_id: str,
        max_steps: int = 5,
        max_chains: Optional[int] = 10000
    ) -> List[Dict[str, Any]]:
        """
        Easy way to trace how decisions are connected.

        Args:
            decision_id: Starting decision
            max_steps: Maximum steps to trace
            max_chains: Maximum number of chains to return; see
                trace_decision_causality(). Pass None for no limit.

        Returns:
            Decision chain connections
        """
        return self.trace_decision_causality(
            decision_id=decision_id,
            max_depth=max_steps,
            max_chains=max_chains
        )

    
    def check_decision_rules(
        self,
        decision_data: Dict[str, Any],
        rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Easy way to check if a decision follows the rules.
        
        Args:
            decision_data: Decision to check
            rules: Custom rules (uses default if None)
            
        Returns:
            Compliance check results
        """
        return self.enforce_decision_policy(
            decision_data=decision_data,
            policy_rules=rules
        )

    
    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Easy way to get graph statistics.
        
        Returns:
            Graph summary information
        """
        if hasattr(self, 'get_graph_metrics'):
            return self.get_graph_metrics()
        else:
            return {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "node_types": self._get_node_type_distribution(),
                "edge_types": self._get_edge_type_distribution()
            }

    
    def find_related_nodes(
        self,
        node_id: str,
        how_many: int = 10,
        similarity_type: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Easy way to find nodes similar to a given node.
        
        Args:
            node_id: Reference node
            how_many: How many similar nodes to find
            similarity_type: Type of similarity ("content", "structural")
            
        Returns:
            List of dicts with node ID, type, content, and similarity score
        """
        return self.find_similar_nodes(
            node_id=node_id,
            similarity_type=similarity_type,
            top_k=how_many
        )

    
    def get_node_importance(
        self,
        node_id: str
    ) -> Dict[str, float]:
        """
        Easy way to get how important a node is in the graph.
        
        Args:
            node_id: Node to analyze
            
        Returns:
            Centrality measures (importance scores)
        """
        return self.get_node_centrality(node_id)

    
    def analyze_connections(self) -> Dict[str, Any]:
        """
        Easy way to analyze the entire graph structure.
        
        Returns:
            Graph analysis results
        """
        return self.analyze_graph_with_kg()
