"""Ontology draft endpoints."""


import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request


from . import _common

router = APIRouter()


# ---------------------------------------------------------------------------
# Draft endpoints
# ---------------------------------------------------------------------------

@router.patch("/draft", response_model=_common.DraftResponse)
async def save_draft(
    request: Request,
    body: _common.DraftRequest,
):
    """Stage editor diffs as a draft with ChangeLogEntry metadata."""
    drafts = _common._get_drafts(request)
    draft_id = f"draft_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    # Create ChangeLogEntry for audit trail
    try:
        from ....change_management.change_log import ChangeLogEntry
        ChangeLogEntry.create_now(
            author=body.author,
            description=body.summary or f"Draft changes for {body.ontology_uri}",
            change_id=draft_id
        )
    except Exception as exc:
        _common.logger.warning(f"Failed to create ChangeLogEntry: {exc}")

    draft = _common.DraftResponse(
        draft_id=draft_id,
        ontology_uri=body.ontology_uri,
        diff=body.diff,
        author=body.author,
        summary=body.summary,
        created_at=now,
        updated_at=now,
    )
    drafts[draft_id] = draft
    return draft


@router.get("/drafts/{ontology_uri:path}", response_model=List[_common.DraftResponse])
async def list_drafts(
    ontology_uri: str,
    request: Request,
):
    """Get staged draft diffs for an ontology."""
    drafts = _common._get_drafts(request)
    return [d for d in drafts.values() if d.ontology_uri == ontology_uri]


@router.get("/draft/{draft_id}", response_model=_common.DraftResponse)
async def get_draft(
    draft_id: str,
    request: Request,
):
    """Get a specific draft by ID."""
    drafts = _common._get_drafts(request)
    if draft_id not in drafts:
        raise HTTPException(status_code=404, detail="Draft not found.")
    return drafts[draft_id]
