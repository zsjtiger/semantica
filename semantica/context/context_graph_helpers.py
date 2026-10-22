"""Internal helper methods for ContextGraph."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .context_graph_base import (
    ContextEdge,
    ContextNode,
)


class _InternalHelpersMixin:
    """Mixin composed into ContextGraph; methods use ``self``."""


    # --- Internal Helpers ---

    def _normalize_timestamp(self, timestamp_value) -> datetime:
        """
        Normalize timestamp value to datetime object.
        
        Handles various timestamp formats:
        - datetime: returns as-is
        - int/float: converts from epoch seconds
        - str: parses ISO format (with optional Z)
        - None/invalid: returns current datetime
        
        Args:
            timestamp_value: Timestamp value in various formats
            
        Returns:
            datetime: Normalized datetime object
        """
        from datetime import datetime
        
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        elif isinstance(timestamp_value, (int, float)):
            return datetime.fromtimestamp(timestamp_value)
        elif isinstance(timestamp_value, str):
            # Handle ISO format with optional Z suffix
            timestamp_str = timestamp_value.rstrip('Z')  #
            try:
                return datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Fallback to current datetime if parsing fails
                return datetime.now()
        else:
            # Fallback for None or other types
            return datetime.now()

    def _add_internal_node(self, node: ContextNode) -> bool:
        """Internal method to add a node."""
        if node.node_id is None or (isinstance(node.node_id, str) and not node.node_id.strip()):
            self.logger.warning("Skipping internal node with invalid id: %r", node)
            return False
        with self._lock:
            self.nodes[node.node_id] = node
            # Handle edge case where node_type might be None or not a string
            if hasattr(node, 'node_type') and isinstance(node.node_type, str):
                self.node_type_index[node.node_type].add(node.node_id)
            else:
                # Use 'unknown' as fallback for invalid node_type
                self.node_type_index['unknown'].add(node.node_id)

        if getattr(self, "mutation_callback", None) and not getattr(
            self, "_suspend_mutation_callback", False
        ):
            try:
                self.mutation_callback("ADD_NODE", node.node_id, node.to_dict())
            except Exception as e:
                self.logger.warning(f"Audit trail callback failed for node {node.node_id}: {e}")
        return True

    
    def _add_internal_edge(self, edge: ContextEdge, unique_type: bool = False) -> bool:
        """Internal method to add an edge.

        With ``unique_type=True`` the edge is skipped when the source already has
        an edge of the same type to the same target.
        """
        if edge.source_id is None or edge.target_id is None:
            self.logger.warning("Skipping internal edge with invalid endpoints: %r", edge)
            return False
        with self._lock:
            # Edge identity is content-derived, so an existing edge_id means this
            # exact edge is already stored; re-adding it is a no-op (issue #922).
            if edge.edge_id in self._edge_index:
                return False
            if unique_type and any(
                e.target_id == edge.target_id and e.edge_type == edge.edge_type
                for e in self._adjacency.get(edge.source_id, [])
            ):
                return False

            # Ensure nodes exist
            if edge.source_id not in self.nodes:
                self._add_internal_node(
                    ContextNode(edge.source_id, "entity", edge.source_id)
                )
            if edge.target_id not in self.nodes:
                self._add_internal_node(
                    ContextNode(edge.target_id, "entity", edge.target_id)
                )

            self._edge_index[edge.edge_id] = edge
            self.edges.append(edge)
            self.edge_type_index[edge.edge_type].append(edge)
            self._adjacency[edge.source_id].append(edge)

        if getattr(self, "mutation_callback", None) and not getattr(
            self, "_suspend_mutation_callback", False
        ):
            try:
                self.mutation_callback("ADD_EDGE", edge.edge_id, edge.to_dict())
            except Exception as e:
                self.logger.warning(
                    f"Audit trail callback failed for edge {edge.edge_id}: {e}"
                )
        return True

    def _emit_mutation(
        self, operation: str, entity_id: str, payload: Dict[str, Any]
    ) -> None:
        """Fire the audit-trail callback, mirroring the add paths.

        Kept in one place so retraction and purge record themselves the same
        way ``_add_internal_node``/``_add_internal_edge`` already do, including
        the ``_suspend_mutation_callback`` guard used during restores.
        """
        if not getattr(self, "mutation_callback", None):
            return
        if getattr(self, "_suspend_mutation_callback", False):
            return
        try:
            self.mutation_callback(operation, entity_id, payload)
        except Exception as e:
            self.logger.warning(
                f"Audit trail callback failed for {operation} {entity_id}: {e}"
            )

    def _incident_edges(self, node_id: str) -> List[ContextEdge]:
        """Every edge touching ``node_id``, in either direction.

        ``_adjacency`` is keyed by source only, so incoming edges have to come
        from a scan of ``self.edges``; relying on ``_adjacency`` alone would
        silently leave inbound edges pointing at a removed node.
        """
        return [
            edge
            for edge in self.edges
            if edge.source_id == node_id or edge.target_id == node_id
        ]

    @staticmethod
    def _find_removal_record(
        store: Dict[Tuple[str, str], Dict[str, Any]],
        entity_id: str,
        entity_kind: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Look a retraction/tombstone up by id, optionally narrowed by kind.

        The caller must hold ``self._lock``. Records are keyed by
        ``(entity_kind, entity_id)``; with no kind given, both keyspaces are
        tried so callers that know an id is unambiguous can pass it alone.
        """
        if entity_kind is not None:
            if entity_kind not in ("node", "edge"):
                raise ValueError(
                    f"entity_kind must be 'node', 'edge' or None, got {entity_kind!r}"
                )
            kinds: Tuple[str, ...] = (entity_kind,)
        else:
            kinds = ("node", "edge")
        for kind in kinds:
            record = store.get((kind, entity_id))
            if record is not None:
                return dict(record)
        return None

    def _cross_graph_links_for(self, node_id: str) -> List[str]:
        """Link ids that ``node_id`` participates in, as exit point or marker.

        The caller must hold ``self._lock``. :meth:`link_graph` registers a link
        in three places -- ``_linked_graphs``, a marker node and the bridge edge
        -- so removing only the node would leave :meth:`navigate_to` resolving a
        link whose source is gone.
        """
        link_ids = [
            link_id
            for link_id, (_, source_node_id, _) in self._linked_graphs.items()
            if source_node_id == node_id
        ]
        link_ids.extend(
            link_id
            for link_id, meta in self._unresolved_links.items()
            if meta.get("source_node_id") == node_id
        )
        node = self.nodes.get(node_id)
        metadata = getattr(node, "metadata", None) or {}
        if metadata.get("cross_graph") and metadata.get("link_id"):
            link_ids.append(metadata["link_id"])
        return list(dict.fromkeys(link_ids))

    def _cross_graph_marker_nodes(self, node_id: str) -> List[str]:
        """Marker nodes of the cross-graph links ``node_id`` exits through.

        The caller must hold ``self._lock``.
        """
        return [
            marker_id
            for marker_id in (
                f"__cross_graph_{link_id}"
                for link_id in self._cross_graph_links_for(node_id)
            )
            if marker_id != node_id and marker_id in self.nodes
        ]

    def _drop_node_from_indexes(self, node_id: str) -> None:
        """Remove one node from ``nodes``, ``node_type_index`` and ``_adjacency``.

        The caller must hold ``self._lock``. Incident edges are not touched --
        see :meth:`_drop_edge_from_indexes`.
        """
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        bucket = self.node_type_index.get(node.node_type)
        if bucket is not None:
            bucket.discard(node_id)
            if not bucket:
                del self.node_type_index[node.node_type]
        self._adjacency.pop(node_id, None)

    def _drop_edge_from_indexes(self, edge: ContextEdge) -> None:
        """Remove one edge from every structure that references it.

        The caller must hold ``self._lock``. ``edges``, ``edge_type_index`` and
        ``_adjacency`` must be updated together or the indexes drift out of
        step with the edge list.
        """
        try:
            self.edges.remove(edge)
        except ValueError:
            pass
        bucket = self.edge_type_index.get(edge.edge_type)
        if bucket is not None:
            try:
                bucket.remove(edge)
            except ValueError:
                pass
            if not bucket:
                del self.edge_type_index[edge.edge_type]
        adjacent = self._adjacency.get(edge.source_id)
        if adjacent is not None:
            try:
                adjacent.remove(edge)
            except ValueError:
                pass
            if not adjacent:
                del self._adjacency[edge.source_id]
