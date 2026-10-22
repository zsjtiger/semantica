"""Ontology alignment endpoints backed by OntologyEngine."""


import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request

from ...session import GraphSession
from ...dependencies import get_session

from . import _common

router = APIRouter()


# ---------------------------------------------------------------------------
# Alignment endpoints using OntologyEngine
# ---------------------------------------------------------------------------

@router.post("/alignments", response_model=_common.AlignmentResponse)
async def create_alignment(
    body: _common.AlignmentRequest,
    request: Request,
    session: GraphSession = Depends(get_session),
):
    """Create an alignment between two ontology entities using OntologyEngine.create_alignment."""
    response = _common.AlignmentResponse(
        source=body.source_uri,
        predicate=body.predicate,
        target=body.target_uri,
    )
    try:
        from ....ontology import OntologyEngine
        
        # Initialize OntologyEngine with session's graph store
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Use OntologyEngine.create_alignment
        await asyncio.to_thread(
            engine.create_alignment,
            body.source_uri,
            body.target_uri,
            body.predicate
        )
    except Exception as exc:
        _common.logger.info("OntologyEngine alignment store unavailable; using route-level store: %s", exc)

    alignment_store = _common._get_alignment_store(request)
    alignment_store[_common._alignment_key(response.source, response.predicate, response.target)] = response
    return response


@router.get("/alignments/{entity_uri:path}", response_model=List[_common.AlignmentResponse])
async def get_alignments(
    entity_uri: str,
    request: Request,
    session: GraphSession = Depends(get_session),
):
    """Get all alignments for an entity using OntologyEngine.get_alignments."""
    route_alignments = [
        alignment
        for alignment in _common._get_alignment_store(request).values()
        if alignment.source == entity_uri or alignment.target == entity_uri
    ]
    try:
        from ....ontology import OntologyEngine
        
        # Initialize OntologyEngine with session's graph store
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Use OntologyEngine.get_alignments
        alignments = await asyncio.to_thread(engine.get_alignments, entity_uri)
        
        engine_alignments = [
            alignment
            for alignment in (_common._coerce_alignment(align) for align in alignments)
            if alignment is not None
        ]
        merged = {
            _common._alignment_key(alignment.source, alignment.predicate, alignment.target): alignment
            for alignment in route_alignments + engine_alignments
        }
        return list(merged.values())
    except Exception as exc:
        _common.logger.info("OntologyEngine alignment lookup unavailable; using route-level store: %s", exc)
        return route_alignments


@router.get("/alignments", response_model=List[_common.AlignmentResponse])
async def list_alignments(
    request: Request,
    ontology_uri: Optional[str] = Query(None),
    session: GraphSession = Depends(get_session),
):
    """List all alignments, optionally filtered by ontology URI using OntologyEngine.list_alignments."""
    route_alignments = list(_common._get_alignment_store(request).values())
    if ontology_uri:
        route_alignments = [
            alignment for alignment in route_alignments
            if ontology_uri in alignment.source or ontology_uri in alignment.target
        ]
    try:
        from ....ontology import OntologyEngine
        
        # Initialize OntologyEngine with session's graph store
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Use OntologyEngine.list_alignments
        alignments = await asyncio.to_thread(engine.list_alignments, ontology_uri=ontology_uri)
        
        engine_alignments = [
            alignment
            for alignment in (_common._coerce_alignment(align) for align in alignments)
            if alignment is not None
        ]
        merged = {
            _common._alignment_key(alignment.source, alignment.predicate, alignment.target): alignment
            for alignment in route_alignments + engine_alignments
        }
        return list(merged.values())
    except Exception as exc:
        _common.logger.info("OntologyEngine alignment listing unavailable; using route-level store: %s", exc)
        return route_alignments
