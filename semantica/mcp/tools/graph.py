"""
Graph tools — add entities/relationships, search, analytics, summary.
"""

from __future__ import annotations

import logging
import os

from ..schemas import (
    ADD_ENTITY,
    ADD_RELATIONSHIP,
    DELETE_NODE,
    EMPTY,
    GET_ANALYTICS,
    QUERY_GRAPH,
    SEARCH_GRAPH,
    UPDATE_NODE,
)
from ..session import get_graph, is_persistence_safe

log = logging.getLogger("semantica.mcp.tools.graph")


def handle_add_entity(args: dict) -> dict:
    """Add a node/entity to the Semantica knowledge graph."""
    node_id = args.get("id", "").strip()
    if not node_id:
        return {"error": "id is required"}
    try:
        graph = get_graph()
        graph.add_node(
            node_id=node_id,
            label=args.get("label", node_id),
            node_type=args.get("type", "Entity"),
            metadata=args.get("metadata", {}),
        )
        # Persist back to disk so the entity survives server restarts.
        # Skip when the initial load failed to avoid overwriting original data.
        kg_path = os.environ.get("SEMANTICA_KG_PATH", "").strip()
        if kg_path:
            if not is_persistence_safe():
                # Roll back: remove the node we just added.
                try:
                    with graph._lock:
                        graph._drop_node_from_indexes(node_id)
                except Exception:
                    pass
                return {
                    "error": (
                        "Persistence blocked: the configured SEMANTICA_KG_PATH "
                        "could not be loaded at startup. Restart the server with "
                        "a readable graph file to re-enable persistence."
                    )
                }
            try:
                graph.save_to_file(kg_path)
            except Exception as save_exc:
                # Roll back: remove the node so in-memory and persisted state agree.
                try:
                    with graph._lock:
                        graph._drop_node_from_indexes(node_id)
                except Exception:
                    pass
                log.exception("save_to_file failed after add_entity; mutation rolled back")
                return {"error": f"Mutation rolled back: could not persist graph: {save_exc}"}
        return {"status": "added", "id": node_id, "type": args.get("type", "Entity")}
    except Exception as exc:
        log.exception("add_entity failed")
        return {"error": str(exc)}


def handle_add_relationship(args: dict) -> dict:
    """Add a directed relationship (edge) between two entities."""
    source = args.get("source", "").strip()
    target = args.get("target", "").strip()
    if not source or not target:
        return {"error": "source and target are required"}
    rel_type = args.get("type", "RELATED_TO")
    try:
        graph = get_graph()
        graph.add_edge(
            source_id=source,
            target_id=target,
            edge_type=rel_type,
            metadata=args.get("metadata", {}),
        )
        # Persist back to disk so the relationship survives server restarts.
        # Skip when the initial load failed to avoid overwriting original data.
        kg_path = os.environ.get("SEMANTICA_KG_PATH", "").strip()
        if kg_path:
            if not is_persistence_safe():
                # Roll back: remove the edge we just added (last matching edge).
                try:
                    with graph._lock:
                        for edge in reversed(list(graph.edges)):
                            if (edge.source_id == source
                                    and edge.target_id == target
                                    and edge.edge_type == rel_type):
                                graph._drop_edge_from_indexes(edge)
                                break
                except Exception:
                    pass
                return {
                    "error": (
                        "Persistence blocked: the configured SEMANTICA_KG_PATH "
                        "could not be loaded at startup. Restart the server with "
                        "a readable graph file to re-enable persistence."
                    )
                }
            try:
                graph.save_to_file(kg_path)
            except Exception as save_exc:
                # Roll back: remove the edge so in-memory and persisted state agree.
                try:
                    with graph._lock:
                        for edge in reversed(list(graph.edges)):
                            if (edge.source_id == source
                                    and edge.target_id == target
                                    and edge.edge_type == rel_type):
                                graph._drop_edge_from_indexes(edge)
                                break
                except Exception:
                    pass
                log.exception("save_to_file failed after add_relationship; mutation rolled back")
                return {"error": f"Mutation rolled back: could not persist graph: {save_exc}"}
        return {"status": "added", "source": source, "target": target, "type": rel_type}
    except Exception as exc:
        log.exception("add_relationship failed")
        return {"error": str(exc)}


def handle_search_graph(args: dict) -> dict:
    """Search nodes in the knowledge graph by label or metadata."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "query is required", "results": []}
    node_type = args.get("node_type", "").strip() or None
    limit = int(args.get("limit", 20))
    try:
        graph = get_graph()
        if node_type:
            nodes = list(graph.find_nodes(node_type=node_type))
        else:
            nodes = list(graph.find_nodes())
        q = query.lower()
        matched = [
            n for n in nodes
            if q in str(n.get("label", "")).lower()
            or q in str(n.get("id", "")).lower()
        ][:limit]
        return {"results": matched, "count": len(matched), "query": query}
    except Exception as exc:
        log.exception("search_graph failed")
        return {"error": str(exc), "results": []}


def handle_get_graph_summary(args: dict) -> dict:  # noqa: ARG001
    """Return a high-level summary of the current knowledge graph."""
    try:
        graph = get_graph()
        all_nodes = list(graph.find_nodes())
        decisions = [n for n in all_nodes if n.get("type") in ("decision", "Decision")]
        node_types: dict[str, int] = {}
        for n in all_nodes:
            t = str(n.get("type", "Unknown"))
            node_types[t] = node_types.get(t, 0) + 1
        edge_count = 0
        if hasattr(graph, "edge_count"):
            try:
                edge_count = graph.edge_count()
            except Exception:
                log.exception("graph.edge_count failed; defaulting edge_count to 0")
        elif hasattr(graph, "stats"):
            edge_count = graph.stats().get("edge_count", 0)
        return {
            "node_count": len(all_nodes),
            "edge_count": edge_count,
            "decision_count": len(decisions),
            "node_types": node_types,
            "graph_ready": True,
        }
    except Exception as exc:
        log.exception("get_graph_summary failed")
        return {"error": str(exc), "graph_ready": False}


def _top_rankings(payload, top_n: int) -> list:
    """Return the first ``top_n`` entries of a centrality result.

    ``CentralityCalculator`` returns ``{"centrality": {node: score},
    "rankings": [...]}`` and the rankings list is already sorted, so the
    handler only has to trim it.  PageRank emits ``(node, score)`` pairs
    while the other measures emit ``{"node": ..., "score": ...}`` dicts;
    both are normalised here so the tool returns one shape.
    """
    if not isinstance(payload, dict):
        return []
    rankings = payload.get("rankings")
    if not isinstance(rankings, list):
        centrality = payload.get("centrality")
        if not isinstance(centrality, dict):
            return []
        rankings = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
    trimmed = []
    for entry in rankings[:top_n]:
        if isinstance(entry, dict):
            trimmed.append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            trimmed.append({"node": entry[0], "score": entry[1]})
    return trimmed


def _community_groups(payload) -> list:
    """Return the community groups of a ``detect_communities`` result.

    The detector returns ``{"communities": [set, ...], ...}``; the groups
    are sets, which the JSON transport cannot serialise, so each one is
    returned as a sorted list.
    """
    if isinstance(payload, dict):
        groups = payload.get("communities")
    elif isinstance(payload, list):
        groups = payload
    else:
        groups = None
    if not isinstance(groups, list):
        return []
    return [sorted(group) if isinstance(group, (set, frozenset)) else group for group in groups]


def _edge_count(graph) -> int:
    if hasattr(graph, "edge_count"):
        return graph.edge_count()
    if hasattr(graph, "stats"):
        return graph.stats().get("edge_count", 0)
    return 0


def handle_get_graph_analytics(args: dict) -> dict:
    """Compute centrality, community detection, and other graph metrics."""
    requested = args.get("metrics", ["all"])
    top_n = int(args.get("top_n", 10))
    compute_all = "all" in requested
    result: dict = {}
    try:
        graph = get_graph()
        from semantica.kg import CentralityCalculator, CommunityDetector

        result["node_count"] = len(list(graph.find_nodes()))
        result["edge_count"] = _edge_count(graph)

        if compute_all or "pagerank" in requested:
            try:
                pr = CentralityCalculator().calculate_pagerank(graph)
                result["pagerank"] = _top_rankings(pr, top_n)
                # (node, score) pairs, top 10: the shape earlier clients read.
                result["top_nodes_by_pagerank"] = [
                    (entry["node"], entry["score"]) if isinstance(entry, dict) else tuple(entry)
                    for entry in _top_rankings(pr, 10)
                ]
            except Exception as exc:
                result["pagerank_error"] = str(exc)
                # PageRank is the headline metric, so its failure also fails the response.
                result["error"] = str(exc)

        if compute_all or "betweenness" in requested:
            try:
                bc = CentralityCalculator().calculate_betweenness_centrality(graph)
                result["betweenness"] = _top_rankings(bc, top_n)
            except Exception as exc:
                result["betweenness_error"] = str(exc)

        if compute_all or "communities" in requested:
            try:
                comms = CommunityDetector().detect_communities(graph)
                groups = _community_groups(comms)
                result["community_count"] = len(groups)
                result["communities"] = groups
            except Exception as exc:
                result["communities_error"] = str(exc)

        if compute_all or "degree" in requested:
            try:
                deg = CentralityCalculator().calculate_degree_centrality(graph)
                result["degree"] = _top_rankings(deg, top_n)
            except Exception as exc:
                result["degree_error"] = str(exc)

        return result
    except Exception as exc:
        log.exception("get_graph_analytics failed")
        return {"error": str(exc)}


_PERSISTENCE_BLOCKED = (
    "Persistence blocked: the configured SEMANTICA_KG_PATH could not be loaded "
    "at startup. Restart the server with a readable graph file to re-enable "
    "persistence."
)


def _snapshot_node(graph, node_id: str) -> tuple[dict, dict]:
    """Copy a node's properties and metadata so a failed mutation can be undone."""
    with graph._lock:
        node = graph.nodes[node_id]
        return dict(node.properties), dict(node.metadata)


def _restore_node(graph, node_id: str, snapshot: tuple[dict, dict]) -> None:
    properties, metadata = snapshot
    with graph._lock:
        node = graph.nodes[node_id]
        node.properties = properties
        node.metadata = metadata
        if (getattr(node, "node_type", None) or "").lower() == "decision":
            graph._sync_decision_from_node(node_id)


def _persist_mutation(graph, restore, label: str) -> tuple[bool, dict | None]:
    """Save the graph back to SEMANTICA_KG_PATH after a node mutation.

    Returns ``(persisted, error)``. When persistence is blocked or the save
    fails, ``restore`` undoes the in-memory change first so memory and disk
    still agree.
    """
    kg_path = os.environ.get("SEMANTICA_KG_PATH", "").strip()
    if not kg_path:
        return False, None
    if not is_persistence_safe():
        restore()
        return False, {"error": _PERSISTENCE_BLOCKED}
    try:
        graph.save_to_file(kg_path)
    except Exception as save_exc:
        restore()
        log.exception("save_to_file failed after %s; mutation rolled back", label)
        return False, {"error": f"Mutation rolled back: could not persist graph: {save_exc}"}
    return True, None


def handle_query_graph(args: dict) -> dict:
    """Query the live graph: one node, its neighbours, or a keyword search.

    mode:
      - "node"      : get one node by id (needs node_id)
      - "neighbors" : traverse up to ``depth`` hops from node_id (default 1)
      - "search"    : keyword search over node id and content (needs query)
    """
    mode = args.get("mode", "neighbors")
    try:
        graph = get_graph()
        if mode == "node":
            node_id = args.get("node_id", "")
            if not node_id:
                return {"error": "node_id is required"}
            return {"node": graph.find_node(node_id)}

        if mode == "neighbors":
            node_id = args.get("node_id", "")
            if not node_id:
                return {"error": "node_id is required"}
            depth = min(max(int(args.get("depth", 1)), 1), 5)
            rel_types = args.get("relationship_types")
            if isinstance(rel_types, str):
                rel_types = [rel_types]
            rel_set = set(rel_types) if rel_types else None
            limit = args.get("limit")
            limit = int(limit) if limit is not None else None

            nb = graph.get_neighbors(
                node_id, hops=depth, relationship_types=rel_types, limit=limit,
            )
            outbound = [
                {"id": n.get("id"), "type": n.get("type"),
                 "content": n.get("content"),
                 "relationship": n.get("relationship"),
                 "direction": "out", "hop": n.get("hop", 1)}
                for n in (nb or [])
            ]
            # One hop inbound: edges whose target is node_id, one entry per source.
            inbound = []
            seen_sources = set()
            for edge in graph.find_edges():
                if edge.get("target") != node_id:
                    continue
                if rel_set is not None and edge.get("type") not in rel_set:
                    continue
                src_id = edge.get("source")
                if src_id in seen_sources:
                    continue
                seen_sources.add(src_id)
                src = graph.find_node(src_id) or {}
                inbound.append({"id": src_id, "type": src.get("type"),
                                "content": src.get("content"),
                                "relationship": edge.get("type"),
                                "direction": "in", "hop": 1})
                if limit is not None and len(inbound) >= limit:
                    break
            neighbors = outbound + inbound
            # ``is not None`` so limit=0 returns an empty list rather than everything.
            if limit is not None:
                neighbors = neighbors[:limit]
            return {"node_id": node_id, "depth": depth, "neighbors": neighbors}

        if mode == "search":
            query = (args.get("query") or "").lower()
            if not query:
                return {"error": "query is required"}
            node_type = args.get("node_type")
            limit = int(args.get("limit", 50))
            nodes = graph.find_nodes(node_type=node_type) if node_type else graph.find_nodes()
            hits = []
            for n in nodes:
                # Check the cap before appending so limit=0 returns nothing.
                if len(hits) >= limit:
                    break
                blob = f"{n.get('id', '')} {n.get('content', '')}".lower()
                if query in blob:
                    hits.append({"id": n.get("id"), "type": n.get("type"),
                                 "content": n.get("content")})
            return {"query": query, "results": hits, "total": len(hits)}

        return {"error": f"unknown mode '{mode}': use node|neighbors|search"}
    except Exception as exc:
        log.exception("query_graph failed")
        return {"error": str(exc)}


def handle_update_node(args: dict) -> dict:
    """Merge properties onto an existing node and persist the change.

    Common use: mark an action node's status (todo/doing/done) with a note.
    """
    node_id = args.get("node_id", "")
    if not node_id:
        return {"error": "node_id is required"}
    properties = args.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return {"error": "properties (non-empty object) is required"}
    try:
        graph = get_graph()
        if not graph.find_node(node_id):
            return {"error": f"node '{node_id}' not found"}
        snapshot = _snapshot_node(graph, node_id)
        graph.add_node_attribute(node_id, properties)
        persisted, error = _persist_mutation(
            graph, lambda: _restore_node(graph, node_id, snapshot), "update_node",
        )
        if error:
            return error
        updated = graph.find_node(node_id) or {}
        metadata = updated.get("metadata") or {}
        return {
            "status": "updated",
            "node_id": node_id,
            "properties": {k: metadata.get(k) for k in properties},
            "persisted": persisted,
        }
    except Exception as exc:
        log.exception("update_node failed")
        return {"error": str(exc)}


def handle_delete_node(args: dict) -> dict:
    """Archive a node (soft delete): it stays in the graph with status='archived'."""
    node_id = args.get("node_id", "")
    if not node_id:
        return {"error": "node_id is required"}
    try:
        graph = get_graph()
        if not graph.find_node(node_id):
            return {"error": f"node '{node_id}' not found"}
        snapshot = _snapshot_node(graph, node_id)
        graph.add_node_attribute(node_id, {"status": "archived"})
        persisted, error = _persist_mutation(
            graph, lambda: _restore_node(graph, node_id, snapshot), "delete_node",
        )
        if error:
            return error
        return {"status": "archived", "node_id": node_id, "persisted": persisted}
    except Exception as exc:
        log.exception("delete_node failed")
        return {"error": str(exc)}


GRAPH_TOOLS = [
    {
        "name": "add_entity",
        "description": "Add a node or entity (person, place, concept, organisation) to the knowledge graph.",
        "inputSchema": ADD_ENTITY,
        "_handler": handle_add_entity,
    },
    {
        "name": "add_relationship",
        "description": "Add a directed relationship (edge) between two entities in the knowledge graph.",
        "inputSchema": ADD_RELATIONSHIP,
        "_handler": handle_add_relationship,
    },
    {
        "name": "search_graph",
        "description": "Search nodes in the knowledge graph by label or ID substring.",
        "inputSchema": SEARCH_GRAPH,
        "_handler": handle_search_graph,
    },
    {
        "name": "get_graph_summary",
        "description": "Return a high-level summary of the knowledge graph: node count, edge count, decision count, node type breakdown.",
        "inputSchema": EMPTY,
        "_handler": handle_get_graph_summary,
    },
    {
        "name": "get_graph_analytics",
        "description": "Compute PageRank centrality, betweenness centrality, degree centrality, and community detection over the knowledge graph.",
        "inputSchema": GET_ANALYTICS,
        "_handler": handle_get_graph_analytics,
    },
    {
        "name": "query_graph",
        "description": "Query the live knowledge graph: get a node, traverse its neighbours (up to 5 hops), or keyword-search nodes by id and content.",
        "inputSchema": QUERY_GRAPH,
        "_handler": handle_query_graph,
    },
    {
        "name": "update_node",
        "description": "Update properties of an existing node (for example mark an action todo, doing or done with a note) and persist to SEMANTICA_KG_PATH.",
        "inputSchema": UPDATE_NODE,
        "_handler": handle_update_node,
    },
    {
        "name": "delete_node",
        "description": "Archive a node (soft delete: marks status='archived' and keeps it for history) and persist to SEMANTICA_KG_PATH.",
        "inputSchema": DELETE_NODE,
        "_handler": handle_delete_node,
    },
]
