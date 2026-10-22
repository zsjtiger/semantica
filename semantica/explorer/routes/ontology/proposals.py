"""Ontology proposal endpoints."""


import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...session import GraphSession
from ...dependencies import get_session

from . import _common

router = APIRouter()


# ---------------------------------------------------------------------------
# Proposal endpoints
# ---------------------------------------------------------------------------

@router.post("/propose", response_model=_common.ProposalResponse)
async def submit_proposal(
    request: Request,
    body: _common.ProposalRequest,
    session: GraphSession = Depends(get_session),
):
    """Submit a change proposal with impact analysis and SHACL pre-validation."""
    drafts = _common._get_drafts(request)
    proposals = _common._get_proposals(request)

    if body.draft_id not in drafts:
        raise HTTPException(status_code=404, detail="Draft not found.")

    draft = drafts[body.draft_id]
    proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    # Compute impact analysis using VersionManager.diff_ontologies and OntologyEngine
    impact_analysis = {}
    shacl_validation = {}

    try:
        from ....ontology import OntologyEngine
        from ....change_management.ontology_version_manager import VersionManager
        
        # Build ontology dicts from draft diff for comparison
        base_ontology = {"uri": body.ontology_uri, "classes": [], "properties": []}
        target_ontology = {
            "uri": body.ontology_uri,
            "classes": [{"uri": uri} for uri in draft.diff.added_classes],
            "properties": [{"uri": uri} for uri in draft.diff.added_properties],
        }
        
        # Use VersionManager.diff_ontologies for structured diff
        version_manager = VersionManager(
            store=session.graph.store if hasattr(session.graph, "store") else None
        )
        diff_result = await asyncio.to_thread(
            version_manager.diff_ontologies,
            base_ontology,
            target_ontology
        )
        
        # Use OntologyEngine for validation and SHACL
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Run validation if available
        validation_results = {}
        try:
            val_res = await asyncio.to_thread(engine.validate, target_ontology)
            validation_results = {
                "valid": getattr(val_res, "valid", getattr(val_res, "is_valid", False)),
                "consistent": getattr(val_res, "consistent", True),
                "satisfiable": getattr(val_res, "satisfiable", True),
                "errors": getattr(val_res, "errors", []),
                "warnings": getattr(val_res, "warnings", [])
            }
        except Exception as val_exc:
            _common.logger.warning(f"Validation failed: {val_exc}")
            validation_results = {"error": str(val_exc)}
        
        # SHACL pre-validation using OntologyEngine.validate_graph
        if engine.store:
            try:
                # Generate SHACL from target ontology
                await asyncio.to_thread(
                    engine.to_shacl,
                    target_ontology,
                    format="turtle"
                )
                
                # Validate current graph data against new SHACL shapes
                all_nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)
                graph_data = {"nodes": all_nodes}
                
                validation_report = await asyncio.to_thread(
                    engine.validate_graph,
                    graph_data,
                    ontology=target_ontology,
                    explain=True
                )
                
                shacl_validation = {
                    "status": "validated",
                    "conforms": getattr(validation_report, "conforms", True),
                    "violations": [
                        {
                            "message": v.message,
                            "severity": v.severity,
                            "focus_node": v.focus_node,
                        }
                        for v in getattr(validation_report, "violations", [])
                    ]
                }
            except Exception as shacl_exc:
                _common.logger.warning(f"SHACL pre-validation failed: {shacl_exc}")
                shacl_validation = {"status": "error", "error": str(shacl_exc)}
        else:
            shacl_validation = {"status": "skipped", "reason": "No store configured"}
        
        impact_analysis = {
            "diff": diff_result,
            "validation_results": validation_results,
            "class_adds": len(draft.diff.added_classes),
            "class_removals": len(draft.diff.removed_classes),
            "property_changes": len(draft.diff.added_properties) + len(draft.diff.removed_properties),
            "restriction_changes": len(draft.diff.added_restrictions) + len(draft.diff.removed_restrictions),
        }
            
    except Exception as exc:
        _common.logger.warning(f"Impact analysis failed: {exc}")
        impact_analysis = {"error": str(exc)}
        shacl_validation = {"status": "error", "error": str(exc)}

    proposal = _common.ProposalResponse(
        proposal_id=proposal_id,
        draft_id=body.draft_id,
        ontology_uri=body.ontology_uri,
        summary=body.summary,
        author=draft.author,
        reviewer=body.reviewer,
        state="proposed",
        impact_analysis=impact_analysis,
        shacl_validation=shacl_validation,
        created_at=now,
        updated_at=now,
        comments=[],
    )
    proposals[proposal_id] = proposal
    return proposal


@router.get("/proposals", response_model=List[_common.ProposalResponse])
async def list_proposals(
    request: Request,
    ontology_uri: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
):
    """List proposals with optional filters."""
    proposals = _common._get_proposals(request)
    result = list(proposals.values())

    if ontology_uri:
        result = [p for p in result if p.ontology_uri == ontology_uri]
    if state:
        result = [p for p in result if p.state == state]

    return result


@router.get("/proposals/{proposal_id}", response_model=_common.ProposalResponse)
async def get_proposal(
    proposal_id: str,
    request: Request,
):
    """Get proposal detail."""
    proposals = _common._get_proposals(request)
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    return proposals[proposal_id]


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal(
    proposal_id: str,
    request: Request,
):
    """Approve a proposal."""
    proposals = _common._get_proposals(request)
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    proposal = proposals[proposal_id]
    proposal.state = "approved"
    proposal.updated_at = datetime.now(timezone.utc).isoformat()
    return {"status": "approved", "proposal_id": proposal_id}


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(
    proposal_id: str,
    request: Request,
):
    """Reject a proposal (can return to draft)."""
    proposals = _common._get_proposals(request)
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    proposal = proposals[proposal_id]
    proposal.state = "rejected"
    proposal.updated_at = datetime.now(timezone.utc).isoformat()
    return {"status": "rejected", "proposal_id": proposal_id}


@router.post("/proposals/{proposal_id}/publish")
async def publish_proposal(
    proposal_id: str,
    request: Request,
    session: GraphSession = Depends(get_session),
):
    """Publish an approved proposal using VersionManager.create_version()."""
    proposals = _common._get_proposals(request)
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    proposal = proposals[proposal_id]

    if proposal.state != "approved":
        raise HTTPException(status_code=400, detail="Only approved proposals can be published.")

    drafts = _common._get_drafts(request)
    if proposal.draft_id not in drafts:
        raise HTTPException(status_code=404, detail="Draft not found.")
    draft = drafts[proposal.draft_id]

    # Build ontology dict for version creation
    ontology_dict = {
        "uri": proposal.ontology_uri,
        "classes": [{"uri": uri} for uri in draft.diff.added_classes],
        "properties": [{"uri": uri} for uri in draft.diff.added_properties],
        "diff": draft.diff.model_dump(),
    }

    # Create version record using VersionManager
    try:
        from ....change_management.ontology_version_manager import VersionManager
        
        # Initialize VersionManager with proper config
        version_manager = VersionManager(
            store=session.graph.store if hasattr(session.graph, "store") else None
        )
        
        # Generate version string based on existing versions
        versions = _common._get_versions(request)
        existing_versions = versions.get(proposal.ontology_uri, [])
        version_num = len(existing_versions) + 1
        version_str = f"1.{version_num}.0"
        
        version_manager.create_version(
            version=version_str,
            ontology=ontology_dict,
            changes=[proposal.summary],
        )
        
        # Store version in app state
        if proposal.ontology_uri not in versions:
            versions[proposal.ontology_uri] = []
        
        versions[proposal.ontology_uri].append(
            _common.VersionEntry(
                version_id=version_str,
                ontology_uri=proposal.ontology_uri,
                state="published",
                author=proposal.author,
                date=datetime.now(timezone.utc).isoformat(),
                diff_summary=draft.diff.model_dump(),
            )
        )
        
        _common.logger.info(f"Created version {version_str} for ontology {proposal.ontology_uri}")
        
    except Exception as exc:
        _common.logger.warning(f"VersionManager.create_version failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Version creation failed: {exc}") from exc

    # Apply additions only after version creation succeeds so failed publishes do not
    # leave a partially mutated live graph.
    nodes_added = 0
    edges_added = 0

    class_nodes = []
    for class_uri in draft.diff.added_classes:
        label = class_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        class_nodes.append({
            "id": class_uri,
            "type": "owl:Class",
            "content": label,
            "properties": {"rdfs:label": label},
        })
    if class_nodes:
        nodes_added += await asyncio.to_thread(session.add_nodes, class_nodes)

    property_nodes = []
    for prop_uri in draft.diff.added_properties:
        label = prop_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        property_nodes.append({
            "id": prop_uri,
            "type": "owl:ObjectProperty",
            "content": label,
            "properties": {"rdfs:label": label},
        })
    if property_nodes:
        nodes_added += await asyncio.to_thread(session.add_nodes, property_nodes)

    proposal.state = "published"
    proposal.updated_at = datetime.now(timezone.utc).isoformat()

    return {
        "status": "published",
        "proposal_id": proposal_id,
        "version": version_str,
        "nodes_added": nodes_added,
        "edges_added": edges_added,
    }


@router.post("/proposals/{proposal_id}/comment")
async def add_comment(
    proposal_id: str,
    body: _common.CommentRequest,
    request: Request,
):
    """Add inline comment to a proposal."""
    proposals = _common._get_proposals(request)
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    proposal = proposals[proposal_id]
    comment = {
        "id": f"comment_{uuid.uuid4().hex[:8]}",
        "element_uri": body.element_uri,
        "text": body.text,
        "author": body.author,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    proposal.comments.append(comment)
    proposal.updated_at = datetime.now(timezone.utc).isoformat()
    return {"status": "commented", "comment_id": comment["id"]}
