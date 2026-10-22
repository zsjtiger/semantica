"""Wildcard ontology management endpoints; registered after the specific routes."""


import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from ...session import GraphSession
from ...dependencies import get_session

from . import _common

router = APIRouter()


# ---------------------------------------------------------------------------
# Wildcard management endpoints (must come after specific routes)
# ---------------------------------------------------------------------------

@router.delete("/{ontology_uri:path}")
async def remove_ontology(ontology_uri: str, request: Request):
    registry = _common._get_registry(request)
    if ontology_uri not in registry:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")
    del registry[ontology_uri]
    return {"status": "removed", "uri": ontology_uri}


@router.patch("/{ontology_uri:path}/toggle", response_model=_common.ToggleResponse)
async def toggle_ontology(ontology_uri: str, request: Request):
    registry = _common._get_registry(request)
    if ontology_uri not in registry:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")
    entry = registry[ontology_uri]
    entry.enabled = not entry.enabled
    return _common.ToggleResponse(uri=ontology_uri, enabled=entry.enabled)


@router.post("/{ontology_uri:path}/refresh", response_model=_common.RefreshResponse)
async def refresh_ontology(
    ontology_uri: str,
    request: Request,
    session: GraphSession = Depends(get_session),
):
    registry = _common._get_registry(request)
    if ontology_uri not in registry:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")
    entry = registry[ontology_uri]
    if not entry.source_url:
        raise HTTPException(status_code=422, detail="No source URL to refresh from.")

    raw = await asyncio.to_thread(_common._fetch_url_sync, entry.source_url)
    content_str = raw.decode("utf-8", errors="replace")

    try:
        nodes, edges, _ = await asyncio.to_thread(
            _common._parse_rdf_sync, content_str.encode("utf-8"), entry.format
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Refresh parse error: {exc}") from exc

    try:
        nodes_added, edges_added = await asyncio.to_thread(
            session.add_nodes_and_edges, nodes, edges
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    entry.loaded_at = datetime.now(timezone.utc).isoformat()

    return _common.RefreshResponse(uri=ontology_uri, nodes_added=nodes_added, edges_added=edges_added)
