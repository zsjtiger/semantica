"""Ontology version endpoints."""


import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request

from ...session import GraphSession
from ...dependencies import get_session

from . import _common

router = APIRouter()


# ---------------------------------------------------------------------------
# Version endpoints
# ---------------------------------------------------------------------------

@router.get("/versions/{ontology_uri:path}", response_model=List[_common.VersionEntry])
async def list_versions(
    ontology_uri: str,
    request: Request,
):
    """List version history for an ontology."""
    versions = _common._get_versions(request)
    return versions.get(ontology_uri, [])


@router.post("/versions/{ontology_uri:path}/compare", response_model=_common.VersionCompareResponse)
async def compare_versions(
    ontology_uri: str,
    body: _common.VersionCompareRequest,
    request: Request,
    session: GraphSession = Depends(get_session),
):
    """Compare two ontology versions using VersionManager.compare_versions()."""
    try:
        from ....change_management.ontology_version_manager import VersionManager
        
        # Initialize VersionManager with session's graph store
        version_manager = VersionManager(
            store=session.graph.store if hasattr(session.graph, "store") else None
        )
        
        # Get version data from app state
        versions = _common._get_versions(request)
        ontology_versions = versions.get(ontology_uri, [])
        
        # Find version records. Older in-memory state may contain plain dicts, while
        # new publishes store VersionEntry models.
        v1_record = next((v for v in ontology_versions if _common._version_field(v, "version_id") == body.version1), None)
        v2_record = next((v for v in ontology_versions if _common._version_field(v, "version_id") == body.version2), None)
        
        if not v1_record or not v2_record:
            raise HTTPException(status_code=404, detail="One or both versions not found.")
        
        # Build ontology dicts from version records
        v1_dict = {
            "uri": ontology_uri,
            "version": body.version1,
            "classes": [],
            "properties": [],
            "diff": _common._version_field(v1_record, "diff_summary", {}),
        }
        v2_dict = {
            "uri": ontology_uri,
            "version": body.version2,
            "classes": [],
            "properties": [],
            "diff": _common._version_field(v2_record, "diff_summary", {}),
        }
        
        # Use VersionManager.diff_ontologies for structured comparison
        diff_result = await asyncio.to_thread(
            version_manager.diff_ontologies,
            v1_dict,
            v2_dict
        )
        
        metadata_changes = {
            "version_id": {"from": body.version1, "to": body.version2},
            "author": {
                "from": _common._version_field(v1_record, "author"),
                "to": _common._version_field(v2_record, "author"),
            },
            "date": {
                "from": _common._version_field(v1_record, "date"),
                "to": _common._version_field(v2_record, "date"),
            },
            "state": {
                "from": _common._version_field(v1_record, "state"),
                "to": _common._version_field(v2_record, "state"),
            },
        }
        
        return _common.VersionCompareResponse(
            version1=body.version1,
            version2=body.version2,
            metadata_changes=metadata_changes,
            class_changes={
                "added": diff_result.get("added_classes", []),
                "removed": diff_result.get("removed_classes", []),
                "changed": diff_result.get("changed_classes", []),
            },
            property_changes={
                "added": diff_result.get("added_properties", []),
                "removed": diff_result.get("removed_properties", []),
                "changed": diff_result.get("changed_properties", []),
            },
            restriction_changes={
                "added": diff_result.get("added_axioms", []),
                "removed": diff_result.get("removed_axioms", []),
                "changed": diff_result.get("changed_axioms", []),
            },
            axiom_changes={
                "added": diff_result.get("added_axioms", []),
                "removed": diff_result.get("removed_axioms", []),
                "changed": diff_result.get("changed_axioms", []),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        _common.logger.error(f"Version comparison failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Version comparison failed: {exc}") from exc
