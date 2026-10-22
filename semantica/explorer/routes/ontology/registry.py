"""Ontology registry endpoints (specific paths; registered before the wildcard routes)."""


import os
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...session import GraphSession
from ...dependencies import get_session

from . import _common

router = APIRouter()


# ---------------------------------------------------------------------------
# Registry endpoints (all specific paths before wildcard)
# ---------------------------------------------------------------------------

@router.get("/registry", response_model=List[_common.OntologyEntry])
async def list_registry(
    request: Request,
    q: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    format: Optional[str] = Query(None),
    session: GraphSession = Depends(get_session),
):
    registry = _common._get_registry(request)

    # Discover ontology-type nodes from live graph not yet registered
    all_nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)

    # Count entity types per ontology URI via scheme_uri property
    class_counts: Dict[str, int] = {}
    concept_counts: Dict[str, int] = {}
    prop_counts: Dict[str, int] = {}
    implicit: Dict[str, Dict[str, Any]] = {}

    for node in all_nodes:
        ntype = node.get("type", "")
        nid = node.get("id", "")
        etype = _common._classify_node_type(ntype)
        scheme_uri = node.get("properties", {}).get("scheme_uri") or node.get("properties", {}).get("uri")

        if etype == "ontology" or etype == "scheme":
            if nid and nid not in registry:
                implicit[nid] = node
        elif scheme_uri:
            if etype == "class":
                class_counts[scheme_uri] = class_counts.get(scheme_uri, 0) + 1
            elif etype == "concept":
                concept_counts[scheme_uri] = concept_counts.get(scheme_uri, 0) + 1
            elif etype == "property":
                prop_counts[scheme_uri] = prop_counts.get(scheme_uri, 0) + 1

    result: List[_common.OntologyEntry] = []

    def _matches(name: str, uri: str, desc: str) -> bool:
        if not q:
            return True
        ql = q.lower()
        return any(ql in t.lower() for t in [name, uri, desc] if t)

    for entry in registry.values():
        if status and entry.status != status:
            continue
        if format and entry.format.lower() != format.lower():
            continue
        if not _matches(entry.name, entry.uri, entry.description or ""):
            continue
        updated = entry.model_copy(update={
            "class_count": class_counts.get(entry.uri, entry.class_count),
            "concept_count": concept_counts.get(entry.uri, entry.concept_count),
            "property_count": prop_counts.get(entry.uri, entry.property_count),
        })
        result.append(updated)

    for nid, node in implicit.items():
        props = node.get("properties", {})
        name = _common._node_label(node) or nid
        if not _matches(name, nid, props.get("description", "")):
            continue
        result.append(_common.OntologyEntry(
            uri=nid,
            name=name,
            description=props.get("description"),
            format=props.get("format", "unknown"),
            status="external",
            version=props.get("version") or props.get("owl:versionInfo"),
            class_count=class_counts.get(nid, 0),
            concept_count=concept_counts.get(nid, 0),
            property_count=prop_counts.get(nid, 0),
            loaded_at=props.get("loaded_at", ""),
            enabled=True,
        ))

    return result


@router.post("/preview", response_model=_common.OntologyPreview)
async def preview_ontology(body: _common.PreviewOntologyRequest):
    if not body.url and not body.content:
        raise HTTPException(status_code=422, detail="Provide either url or content.")

    if body.url:
        raw = await asyncio.to_thread(_common._fetch_url_sync, body.url)
        content_str = raw.decode("utf-8", errors="replace")
    else:
        content_str = body.content or ""

    fmt = _common._normalize_format(body.format) if body.format else _common._detect_format(content_str)

    try:
        _, _, metadata = await asyncio.to_thread(
            _common._parse_rdf_sync, content_str.encode("utf-8"), fmt
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse ontology: {exc}") from exc

    return _common.OntologyPreview(
        uri=metadata.get("uri", ""),
        name=metadata.get("name", ""),
        description=metadata.get("description"),
        namespace=_common._extract_namespace(metadata.get("uri", "")),
        version=metadata.get("version"),
        license=metadata.get("license"),
        format=fmt,
        estimated_triples=metadata.get("triple_count", 0),
        source_url=body.url,
    )


@router.post("/load", response_model=_common.LoadOntologyResponse)
async def load_ontology(
    request: Request,
    body: _common.LoadOntologyRequest,
    session: GraphSession = Depends(get_session),
):
    if not body.url and not body.content:
        raise HTTPException(status_code=422, detail="Provide either url or content.")

    if body.url:
        raw = await asyncio.to_thread(_common._fetch_url_sync, body.url)
        content_str = raw.decode("utf-8", errors="replace")
    else:
        content_str = body.content or ""

    fmt = _common._normalize_format(body.format) if body.format else _common._detect_format(content_str)
    suffix = _common._INGEST_FORMAT_SUFFIXES.get(fmt)
    if suffix is None:
        raise HTTPException(status_code=422, detail=f"Unsupported ontology format: {fmt}")

    try:
        # Try to use OntologyIngestor for proper parsing and conversion
        from ....ingest.ontology_ingestor import OntologyIngestor
        from tempfile import NamedTemporaryFile
        
        ingestor = OntologyIngestor()
        
        # Write content to temporary file for ingestion
        with NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as temp_file:
            temp_file.write(content_str)
            temp_path = temp_file.name
        
        try:
            # Use OntologyIngestor to parse and convert
            ontology_data = await asyncio.to_thread(
                ingestor.ingest_ontology,
                temp_path,
                format=fmt
            )
            if not ontology_data.data.get("classes") and not ontology_data.data.get("properties"):
                raise ValueError("No OWL classes or properties found by OntologyIngestor")
            
            # Convert to graph nodes/edges using ontology data
            nodes, edges = await asyncio.to_thread(
                _common._convert_ontology_to_graph,
                ontology_data.data
            )
            
            try:
                nodes_added, edges_added = await asyncio.to_thread(
                    session.add_nodes_and_edges, nodes, edges
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            
            # Register in registry
            registry = _common._get_registry(request)
            ontology_uri = ontology_data.data.get("uri", f"temp:{uuid.uuid4().hex[:12]}")
            registry[ontology_uri] = _common.OntologyEntry(
                uri=ontology_uri,
                name=body.name or ontology_data.data.get("name", "Imported Ontology"),
                description=body.description or ontology_data.data.get("description"),
                format=fmt,
                status="external",
                version=ontology_data.data.get("version", "1.0"),
                class_count=len([n for n in nodes if n.get("type") in _common._CLASS_TYPES]),
                concept_count=len([n for n in nodes if n.get("type") in _common._CONCEPT_TYPES]),
                property_count=len([n for n in nodes if n.get("type") in _common._PROPERTY_TYPES]),
                loaded_at=datetime.now(timezone.utc).isoformat(),
                enabled=True,
                tags=body.tags,
                source_url=body.url,
            )
            
            return _common.LoadOntologyResponse(
                uri=ontology_uri,
                name=body.name or ontology_data.data.get("name", "Imported Ontology"),
                nodes_added=nodes_added,
                edges_added=edges_added,
                format=fmt,
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError as cleanup_exc:
                _common.logger.debug("Failed to remove temporary ontology file: %s", cleanup_exc)
                
    except HTTPException:
        # Re-raise HTTPExceptions we deliberately raised above (e.g. the 422 from
        # SKOS cycle validation) instead of letting the broad `except Exception`
        # below mask them as an ingestor failure and silently retry via the
        # fallback parser.
        raise
    except Exception as ingest_exc:
        _common.logger.warning(f"OntologyIngestor failed, falling back to basic parsing: {ingest_exc}")

        # Fallback to basic parsing
        nodes, edges, metadata = await asyncio.to_thread(
            _common._parse_rdf_sync, content_str.encode("utf-8"), fmt
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse ontology: {exc}") from exc

    # Fallback path - use basic parsing
    try:
        nodes_added, edges_added = await asyncio.to_thread(
            session.add_nodes_and_edges, nodes, edges
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    registry = _common._get_registry(request)
    ontology_uri = metadata.get("uri", f"temp:{uuid.uuid4().hex[:12]}")
    registry[ontology_uri] = _common.OntologyEntry(
        uri=ontology_uri,
        name=body.name or metadata.get("name", "Imported Ontology"),
        description=body.description or metadata.get("description"),
        format=fmt,
        status="external",
        version=metadata.get("version", "1.0"),
        class_count=sum(1 for n in nodes if n.get("type") in _common._CLASS_TYPES),
        concept_count=sum(1 for n in nodes if n.get("type") in _common._CONCEPT_TYPES),
        property_count=sum(1 for n in nodes if n.get("type") in _common._PROPERTY_TYPES),
        loaded_at=datetime.now(timezone.utc).isoformat(),
        enabled=True,
        tags=body.tags,
        source_url=body.url,
    )

    return _common.LoadOntologyResponse(
        uri=ontology_uri,
        name=body.name or metadata.get("name", "Imported Ontology"),
        nodes_added=nodes_added,
        edges_added=edges_added,
        format=fmt,
    )


@router.post("/create", response_model=_common.LoadOntologyResponse)
async def create_ontology(
    request: Request,
    body: _common.CreateOntologyRequest,
    session: GraphSession = Depends(get_session),
):
    """Create ontology from scratch, sample data, or text using OntologyEngine."""
    ns = body.namespace.rstrip("/#")
    onto_uri = f"{ns}#ontology"
    
    # Initialize OntologyEngine with session's graph store
    engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
    if body.provider or body.model:
        engine_config["provider"] = body.provider
        engine_config["model"] = body.model
    
    nodes: List[Dict[str, Any]] = [{
        "id": onto_uri,
        "type": "owl:Ontology",
        "content": body.name,
        "properties": {
            "rdfs:label": body.name,
            "rdfs:comment": body.description or "",
            "namespace": body.namespace,
        },
    }]
    edges: List[Dict[str, Any]] = []

    if body.mode == "data" and body.sample_data:
        try:
            from ....ontology import OntologyEngine
            engine = OntologyEngine(**engine_config)
            result = await asyncio.to_thread(engine.from_data, body.sample_data)
            
            # Convert OntologyEngine result to graph nodes/edges
            if isinstance(result, dict):
                for cls in result.get("classes", []):
                    cls_uri = f"{ns}/{cls.get('name', uuid.uuid4().hex[:6])}"
                    nodes.append({
                        "id": cls_uri,
                        "type": "owl:Class",
                        "content": cls.get("name", ""),
                        "properties": {
                            "rdfs:label": cls.get("name", ""),
                            "rdfs:comment": cls.get("description", ""),
                        },
                    })
                
                # Add property edges
                for prop in result.get("properties", []):
                    prop_uri = f"{ns}/{prop.get('name', uuid.uuid4().hex[:6])}"
                    
                    nodes.append({
                        "id": prop_uri,
                        "type": "owl:ObjectProperty",
                        "content": prop.get("name", ""),
                        "properties": {"rdfs:label": prop.get("name", "")},
                    })
                    
                    # Only create domain/range edges if domain/range are specified
                    domain = prop.get('domain')
                    if domain and domain.strip():
                        domain_uri = f"{ns}/{domain}"
                        edges.append({
                            "source": prop_uri,
                            "target": domain_uri,
                            "type": "rdfs:domain",
                            "weight": 1.0,
                        })
                    
                    range_val = prop.get('range')
                    if range_val and range_val.strip():
                        range_uri = f"{ns}/{range_val}"
                        edges.append({
                            "source": prop_uri,
                            "target": range_uri,
                            "type": "rdfs:range",
                            "weight": 1.0,
                        })
                
                # Add subclass edges
                for cls in result.get("classes", []):
                    cls_uri = f"{ns}/{cls.get('name', '')}"
                    for parent in cls.get("superclasses", []):
                        parent_uri = f"{ns}/{parent}"
                        edges.append({
                            "source": cls_uri,
                            "target": parent_uri,
                            "type": "rdfs:subClassOf",
                            "weight": 1.0,
                        })
            
            _common.logger.info(f"Generated ontology from sample data with {len(nodes)} nodes, {len(edges)} edges")
            
        except Exception as exc:
            _common.logger.exception("Failed to generate ontology from sample data; aborting ontology creation.")
            raise HTTPException(status_code=500, detail=f"Ontology generation failed: {exc}") from exc

    elif body.mode == "text" and body.schema_text:
        try:
            from ....ontology import OntologyEngine
            engine = OntologyEngine(**engine_config)
            result = await asyncio.to_thread(engine.from_text, body.schema_text, provider=body.provider, model=body.model)
            
            # Convert OntologyEngine result to graph nodes/edges
            if isinstance(result, dict):
                for cls in result.get("classes", []):
                    cls_uri = f"{ns}/{cls.get('name', uuid.uuid4().hex[:6])}"
                    nodes.append({
                        "id": cls_uri,
                        "type": "owl:Class",
                        "content": cls.get("name", ""),
                        "properties": {
                            "rdfs:label": cls.get("name", ""),
                            "rdfs:comment": cls.get("description", ""),
                        },
                    })
                
                # Add property edges
                for prop in result.get("properties", []):
                    prop_uri = f"{ns}/{prop.get('name', uuid.uuid4().hex[:6])}"
                    
                    nodes.append({
                        "id": prop_uri,
                        "type": "owl:ObjectProperty",
                        "content": prop.get("name", ""),
                        "properties": {"rdfs:label": prop.get("name", "")},
                    })
                    
                    # Only create domain/range edges if domain/range are specified
                    domain = prop.get('domain')
                    if domain and domain.strip():
                        domain_uri = f"{ns}/{domain}"
                        edges.append({
                            "source": prop_uri,
                            "target": domain_uri,
                            "type": "rdfs:domain",
                            "weight": 1.0,
                        })
                    
                    range_val = prop.get('range')
                    if range_val and range_val.strip():
                        range_uri = f"{ns}/{range_val}"
                        edges.append({
                            "source": prop_uri,
                            "target": range_uri,
                            "type": "rdfs:range",
                            "weight": 1.0,
                        })
                
                # Add subclass edges
                for cls in result.get("classes", []):
                    cls_uri = f"{ns}/{cls.get('name', '')}"
                    for parent in cls.get("superclasses", []):
                        parent_uri = f"{ns}/{parent}"
                        edges.append({
                            "source": cls_uri,
                            "target": parent_uri,
                            "type": "rdfs:subClassOf",
                            "weight": 1.0,
                        })
            
            _common.logger.info(f"Generated ontology from text with {len(nodes)} nodes, {len(edges)} edges")
            
        except Exception as exc:
            _common.logger.exception("Failed to generate ontology from schema text; aborting ontology creation.")
            raise HTTPException(status_code=500, detail=f"Ontology generation failed: {exc}") from exc

    try:
        nodes_added, edges_added = await asyncio.to_thread(
            session.add_nodes_and_edges, nodes, edges
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    registry = _common._get_registry(request)
    registry[onto_uri] = _common.OntologyEntry(
        uri=onto_uri,
        name=body.name,
        description=body.description,
        format="turtle",
        status="draft",
        version="0.1.0",
        class_count=sum(1 for n in nodes if n.get("type") == "owl:Class"),
        loaded_at=datetime.now(timezone.utc).isoformat(),
        enabled=True,
        tags=body.tags,
    )

    return _common.LoadOntologyResponse(
        uri=onto_uri, name=body.name,
        nodes_added=nodes_added, edges_added=edges_added, format="turtle",
    )


@router.get("/search", response_model=List[_common.OntologySearchResult])
async def search_entities(
    q: str = Query(..., min_length=1),
    entity_type: Optional[str] = Query(None),
    limit: int = Query(default=50, ge=1, le=200),
    session: GraphSession = Depends(get_session),
):
    # Use the session's indexed search; over-fetch to allow post-filtering by entity type
    raw_hits = await asyncio.to_thread(session.search, q, limit * 6)
    results: List[_common.OntologySearchResult] = []

    for hit in raw_hits:
        node = hit.get("node", hit)  # session.search returns {"node": ..., "score": ...}
        ntype = node.get("type", "")
        if ntype not in _common._SEARCHABLE_TYPES:
            continue
        etype = _common._classify_node_type(ntype)
        if entity_type and etype != entity_type:
            continue

        label = _common._node_label(node)
        props = node.get("properties", {})
        definition = (
            props.get("rdfs:comment")
            or props.get("skos:definition")
            or props.get("description")
        )

        results.append(_common.OntologySearchResult(
            uri=node.get("id", ""),
            label=label,
            type=ntype,
            entity_type=etype,
            definition=definition,
            source_ontology=props.get("scheme_uri"),
            namespace_prefix=_common._extract_namespace(node.get("id", "")),
        ))
        if len(results) >= limit:
            break

    return results


@router.get("/graph", response_model=_common.OntologyGraphResponse)
async def get_ontology_graph(
    request: Request,
    uri: str = Query(..., min_length=1),
    session: GraphSession = Depends(get_session),
):
    """Return the editable schema subgraph for one registered ontology."""
    known_ontology_uris = await asyncio.to_thread(
        _common._known_ontology_uris, session, _common._get_registry(request)
    )
    if uri not in known_ontology_uris:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")

    try:
        core_nodes_by_id = await asyncio.to_thread(
            _common._collect_core_nodes, session, uri, known_ontology_uris
        )
        if not core_nodes_by_id:
            raise HTTPException(status_code=404, detail="Ontology graph not found.")
        core_node_ids = set(core_nodes_by_id)
        selected_edges = await asyncio.to_thread(
            _common._select_structure_edges, session, core_node_ids
        )
    except _common.GraphTruncationError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    # Invariant: the helpers raise the moment their accumulation passes
    # _MAX_ANALYSIS_NODES, so core_nodes_by_id and selected_edges are both
    # within the cap here; a post-filter re-check would be unreachable.
    external_node_ids = {
        node_id
        for edge in selected_edges
        for node_id in (str(edge.get("source", "")), str(edge.get("target", "")))
    } - core_node_ids
    external_nodes = await asyncio.gather(
        *(asyncio.to_thread(session.get_node, node_id) for node_id in external_node_ids)
    )

    selected_nodes = list(core_nodes_by_id.values())
    selected_nodes.extend(node for node in external_nodes if node is not None)
    selected_nodes.sort(key=lambda node: str(node.get("id", "")))
    selected_edges.sort(
        key=lambda edge: (
            str(edge.get("source", "")),
            str(edge.get("type", "")),
            str(edge.get("target", "")),
            str(edge.get("id", "")),
        )
    )
    # Annotate copies: session node dicts may be cached and shared with other callers.
    return _common.OntologyGraphResponse(
        uri=uri,
        nodes=[
            {**node, "entity_type": _common._classify_node_type(node.get("type", ""))}
            for node in selected_nodes
        ],
        edges=selected_edges,
    )


@router.get("/entity/{entity_uri:path}", response_model=_common.EntityDetailResponse)
async def get_entity_detail(
    entity_uri: str,
    request: Request,
    session: GraphSession = Depends(get_session),
):
    node = await asyncio.to_thread(session.get_node, entity_uri)
    if node is None:
        raise HTTPException(status_code=404, detail="Entity not found.")

    props = node.get("properties", {})
    ntype = node.get("type", "")
    label = _common._node_label(node)
    definition = props.get("rdfs:comment") or props.get("skos:definition") or props.get("description")

    out_edges, _ = await asyncio.to_thread(session.get_edges, source=entity_uri, skip=0, limit=9999)
    in_edges, _ = await asyncio.to_thread(session.get_edges, target=entity_uri, skip=0, limit=9999)

    superclasses = [e["target"] for e in out_edges if e.get("type") in {"rdfs:subClassOf", "skos:broader"}]
    subclasses = [e["source"] for e in in_edges if e.get("type") in {"rdfs:subClassOf", "skos:broader"}]
    domain = [e["target"] for e in out_edges if e.get("type") == "rdfs:domain"]
    range_ = [e["target"] for e in out_edges if e.get("type") == "rdfs:range"]

    all_nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)
    instance_count = sum(1 for n in all_nodes if n.get("type") == entity_uri)
    # Ownership must use the same candidate set as /graph, so it goes through the
    # same helper rather than being derived from all_nodes above: that scan is
    # capped at 999,999, and on a larger graph a truncated set would silently
    # drop ontologies and make the two endpoints disagree about who owns a node.
    # The helper iterates only the ontology node types, so it is not a full scan.
    known_ontology_uris = await asyncio.to_thread(
        _common._known_ontology_uris, session, _common._get_registry(request)
    )

    return _common.EntityDetailResponse(
        uri=entity_uri, label=label,
        type=ntype, entity_type=_common._classify_node_type(ntype),
        definition=definition,
        source_ontology=props.get("scheme_uri"),
        owning_ontology=_common._resolve_owning_ontology(node, known_ontology_uris),
        superclasses=superclasses, subclasses=subclasses,
        domain=domain, range=range_,
        instance_count=instance_count, properties=props,
    )


@router.get("/skos/schemes", response_model=List[_common.SKOSScheme])
async def list_skos_schemes(session: GraphSession = Depends(get_session)):
    """List SKOS concept schemes using OntologyEngine.list_vocabularies()."""
    try:
        from ....ontology import OntologyEngine
        
        # Initialize OntologyEngine with session's graph store
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Use OntologyEngine.list_vocabularies
        vocabularies = await asyncio.to_thread(engine.list_vocabularies)
        
        # Count concepts per scheme
        all_nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)
        concept_counts: Dict[str, int] = {}
        for node in all_nodes:
            scheme_uri = node.get("properties", {}).get("scheme_uri")
            if scheme_uri and node.get("type") in _common._CONCEPT_TYPES:
                concept_counts[scheme_uri] = concept_counts.get(scheme_uri, 0) + 1
        
        return [
            _common.SKOSScheme(
                uri=vocab["uri"],
                title=vocab["label"] or vocab["uri"].rsplit("/", 1)[-1].rsplit("#", 1)[-1],
                description=None,
                concept_count=concept_counts.get(vocab["uri"], 0),
            )
            for vocab in vocabularies
        ]
    except Exception as exc:
        _common.logger.warning(f"OntologyEngine.list_vocabularies failed, falling back to session: {exc}")
        # Fallback to session-based implementation
        nodes, _ = await asyncio.to_thread(
            session.get_nodes, node_type="skos:ConceptScheme", skip=0, limit=999_999
        )
        all_edges, _ = await asyncio.to_thread(session.get_edges, skip=0, limit=999_999)
        concept_counts: Dict[str, int] = {}
        for edge in all_edges:
            if edge.get("type") in {"skos:inScheme", "skos:topConceptOf"}:
                concept_counts[edge["target"]] = concept_counts.get(edge["target"], 0) + 1
            elif edge.get("type") == "skos:hasTopConcept":
                concept_counts[edge["source"]] = concept_counts.get(edge["source"], 0) + 1

        result = []
        for node in nodes:
            props = node.get("properties", {})
            nid = node.get("id", "")
            result.append(_common.SKOSScheme(
                uri=nid,
                title=_common._node_label(node),
                description=props.get("description") or props.get("skos:definition"),
                concept_count=concept_counts.get(nid, 0),
            ))
        return result


@router.post("/skos/search", response_model=List[_common.OntologySearchResult])
async def search_skos_concepts(
    body: _common.SKOSConceptSearchRequest,
    session: GraphSession = Depends(get_session),
):
    """Search SKOS concepts using OntologyEngine.search_concepts()."""
    try:
        from ....ontology import OntologyEngine
        
        # Initialize OntologyEngine with session's graph store
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Use OntologyEngine.search_concepts
        concepts = await asyncio.to_thread(
            engine.search_concepts,
            body.query,
            scheme_uri=body.scheme_uri
        )
        
        return [
            _common.OntologySearchResult(
                uri=concept["uri"],
                label=concept["label"],
                type="skos:Concept",
                entity_type="concept",
                definition=None,
                source_ontology=body.scheme_uri,
                namespace_prefix=_common._extract_namespace(concept["uri"]),
            )
            for concept in concepts
        ]
    except Exception as exc:
        _common.logger.warning(f"OntologyEngine.search_concepts failed, using fallback: {exc}")
        # Fallback to session-based search
        raw_hits = await asyncio.to_thread(session.search, body.query, 300)
        results: List[_common.OntologySearchResult] = []

        for hit in raw_hits:
            node = hit.get("node", hit)
            ntype = node.get("type", "")
            if ntype not in _common._CONCEPT_TYPES:
                continue
            if body.scheme_uri:
                scheme = node.get("properties", {}).get("scheme_uri")
                if scheme != body.scheme_uri:
                    continue

            label = _common._node_label(node)
            props = node.get("properties", {})
            definition = (
                props.get("rdfs:comment")
                or props.get("skos:definition")
                or props.get("description")
            )

            results.append(_common.OntologySearchResult(
                uri=node.get("id", ""),
                label=label,
                type=ntype,
                entity_type="concept",
                definition=definition,
                source_ontology=props.get("scheme_uri"),
                namespace_prefix=_common._extract_namespace(node.get("id", "")),
            ))
            if len(results) >= 50:
                break

        return results


@router.get("/skos/concept/{concept_uri:path}", response_model=_common.SKOSConceptDetail)
async def get_skos_concept(
    concept_uri: str,
    session: GraphSession = Depends(get_session),
):
    """Get SKOS concept detail using OntologyEngine.list_concepts() and search_concepts()."""
    try:
        from ....ontology import OntologyEngine
        
        # Initialize OntologyEngine with session's graph store
        engine_config = {"store": session.graph.store if hasattr(session.graph, "store") else None}
        engine = OntologyEngine(**engine_config)
        
        # Try to get scheme_uri from node first
        node = await asyncio.to_thread(session.get_node, concept_uri)
        if node is None:
            raise HTTPException(status_code=404, detail="Concept not found.")
        
        scheme_uri = node.get("properties", {}).get("scheme_uri")
        
        # Use OntologyEngine.list_concepts if scheme_uri is known
        if scheme_uri:
            concepts = await asyncio.to_thread(engine.list_concepts, scheme_uri)
            concept_data = next((c for c in concepts if c["uri"] == concept_uri), None)
            
            if concept_data:
                return _common.SKOSConceptDetail(
                    uri=concept_data["uri"],
                    pref_label=concept_data["pref_label"],
                    alt_labels=concept_data.get("alt_labels", []),
                    hidden_labels=[],
                    definition=None,
                    scope_note=None,
                    editorial_note=None,
                    broader=[],
                    narrower=[],
                    related=[],
                    exact_match=[],
                    close_match=[],
                    broad_match=[],
                    narrow_match=[],
                    scheme_uri=scheme_uri,
                )
        
        # Fallback to session-based implementation
        props = node.get("properties", {})
        out_edges, _ = await asyncio.to_thread(session.get_edges, source=concept_uri, skip=0, limit=9999)
        in_edges, _ = await asyncio.to_thread(session.get_edges, target=concept_uri, skip=0, limit=9999)

        def collect_out(rel: str) -> List[str]:
            return [e["target"] for e in out_edges if e.get("type") == rel]

        def collect_in(rel: str) -> List[str]:
            return [e["source"] for e in in_edges if e.get("type") == rel]

        pref_label = props.get("pref_label") or props.get("skos:prefLabel") or _common._node_label(node)
        alt_labels = props.get("alt_labels") or props.get("skos:altLabel") or []
        if isinstance(alt_labels, str):
            alt_labels = [alt_labels]
        hidden_labels = props.get("skos:hiddenLabel") or []
        if isinstance(hidden_labels, str):
            hidden_labels = [hidden_labels]

        if not scheme_uri:
            candidates = collect_out("skos:inScheme") or collect_out("skos:topConceptOf")
            scheme_uri = candidates[0] if candidates else None

        return _common.SKOSConceptDetail(
            uri=concept_uri,
            pref_label=pref_label,
            alt_labels=list(alt_labels),
            hidden_labels=list(hidden_labels),
            definition=props.get("definition") or props.get("skos:definition"),
            scope_note=props.get("skos:scopeNote"),
            editorial_note=props.get("skos:editorialNote"),
            broader=collect_out("skos:broader") + collect_in("skos:narrower"),
            narrower=collect_out("skos:narrower") + collect_in("skos:broader"),
            related=collect_out("skos:related"),
            exact_match=collect_out("skos:exactMatch"),
            close_match=collect_out("skos:closeMatch"),
            broad_match=collect_out("skos:broadMatch"),
            narrow_match=collect_out("skos:narrowMatch"),
            scheme_uri=scheme_uri,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _common.logger.warning(f"OntologyEngine SKOS methods failed, using fallback: {exc}")
        # Fallback to session-based implementation
        node = await asyncio.to_thread(session.get_node, concept_uri)
        if node is None:
            raise HTTPException(status_code=404, detail="Concept not found.")

        props = node.get("properties", {})
        out_edges, _ = await asyncio.to_thread(session.get_edges, source=concept_uri, skip=0, limit=9999)
        in_edges, _ = await asyncio.to_thread(session.get_edges, target=concept_uri, skip=0, limit=9999)

        def collect_out(rel: str) -> List[str]:
            return [e["target"] for e in out_edges if e.get("type") == rel]

        def collect_in(rel: str) -> List[str]:
            return [e["source"] for e in in_edges if e.get("type") == rel]

        pref_label = props.get("pref_label") or props.get("skos:prefLabel") or _common._node_label(node)
        alt_labels = props.get("alt_labels") or props.get("skos:altLabel") or []
        if isinstance(alt_labels, str):
            alt_labels = [alt_labels]
        hidden_labels = props.get("skos:hiddenLabel") or []
        if isinstance(hidden_labels, str):
            hidden_labels = [hidden_labels]

        scheme_uri = props.get("scheme_uri")
        if not scheme_uri:
            candidates = collect_out("skos:inScheme") or collect_out("skos:topConceptOf")
            scheme_uri = candidates[0] if candidates else None

        return _common.SKOSConceptDetail(
            uri=concept_uri,
            pref_label=pref_label,
            alt_labels=list(alt_labels),
            hidden_labels=list(hidden_labels),
            definition=props.get("definition") or props.get("skos:definition"),
            scope_note=props.get("skos:scopeNote"),
            editorial_note=props.get("skos:editorialNote"),
            broader=collect_out("skos:broader") + collect_in("skos:narrower"),
            narrower=collect_out("skos:narrower") + collect_in("skos:broader"),
            related=collect_out("skos:related"),
            exact_match=collect_out("skos:exactMatch"),
            close_match=collect_out("skos:closeMatch"),
            broad_match=collect_out("skos:broadMatch"),
            narrow_match=collect_out("skos:narrowMatch"),
            scheme_uri=scheme_uri,
        )


# alignments, health, and SHACL studio


@router.get("/alignments", response_model=List[_common.OntologyAlignment])
async def list_alignments(
    request: Request,
    uri: Optional[str] = Query(None),
):
    store = _common._get_alignment_store(request)
    alignments = list(store.values())
    if uri:
        alignments = [
            item for item in alignments
            if item.source_uri.startswith(uri) or item.target_uri.startswith(uri)
        ]
    return sorted(alignments, key=lambda item: (item.source_label, item.target_label, item.relation))


@router.post("/alignments", response_model=_common.OntologyAlignment)
async def upsert_alignment(
    request: Request,
    body: _common.OntologyAlignmentRequest,
    session: GraphSession = Depends(get_session),
):
    source_node = await asyncio.to_thread(session.get_node, body.source_uri)
    target_node = await asyncio.to_thread(session.get_node, body.target_uri)
    # External vocabulary URIs (e.g. schema.org, DBpedia) are not in the local
    # graph; fall back to a label derived from the URI or the caller-supplied label.
    source_label = (
        body.source_label
        or (_common._node_label(source_node) if source_node is not None else None)
        or _common._label_from_uri(body.source_uri)
    )
    target_label = (
        body.target_label
        or (_common._node_label(target_node) if target_node is not None else None)
        or _common._label_from_uri(body.target_uri)
    )

    now = datetime.now(timezone.utc).isoformat()
    store = _common._get_alignment_store(request)
    alignment_id = _common._alignment_id(body.source_uri, body.relation, body.target_uri)
    existing = store.get(alignment_id)
    alignment = _common.OntologyAlignment(
        id=alignment_id,
        source_uri=body.source_uri,
        source_label=source_label,
        target_uri=body.target_uri,
        target_label=target_label,
        relation=body.relation,
        predicate_uri=_common._ALIGNMENT_RELATIONS[body.relation],
        confidence=body.confidence,
        provenance=body.provenance,
        source=body.source,
        reviewer=body.reviewer,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    store[alignment_id] = alignment
    # OntologyEngine.create_alignment() requires a TripletStore (e.g. FalkorDB) which is
    # not configured in the explorer deployment. Alignments are intentionally stored only
    # in request.app.state (session memory). The ephemeral-storage banner in the UI
    # communicates this limitation to users.
    return alignment


@router.delete("/alignments")
async def delete_alignment(request: Request, id: str = Query(...)):
    store = _common._get_alignment_store(request)
    if id not in store:
        raise HTTPException(status_code=404, detail="Alignment not found.")
    del store[id]
    return {"status": "removed", "id": id}


@router.post("/suggest-alignments", response_model=List[_common.AlignmentSuggestion])
async def suggest_alignments(
    body: _common.AlignmentSuggestionRequest,
    session: GraphSession = Depends(get_session),
):
    nodes, total_count = await asyncio.to_thread(session.get_nodes, skip=0, limit=_common._MAX_ANALYSIS_NODES)
    if total_count > _common._MAX_ANALYSIS_NODES:
        _common.logger.warning(
            "suggest-alignments: graph has %d nodes; analysis capped at %d. "
            "Filter by source/target ontology URI for more accurate results.",
            total_count, _common._MAX_ANALYSIS_NODES,
        )
    source_nodes = _common._ontology_entities(nodes, body.source_ontology_uri)
    target_nodes = _common._ontology_entities(nodes, body.target_ontology_uri)
    if not body.source_ontology_uri:
        source_nodes = _common._ontology_entities(nodes)
    if not body.target_ontology_uri:
        target_nodes = _common._ontology_entities(nodes)

    # Per-side entity cap to bound the O(n²) comparison loop.
    if len(source_nodes) > _common._MAX_ENTITIES_PER_SIDE:
        _common.logger.warning("suggest-alignments: source side capped at %d entities.", _common._MAX_ENTITIES_PER_SIDE)
        source_nodes = source_nodes[:_common._MAX_ENTITIES_PER_SIDE]
    if len(target_nodes) > _common._MAX_ENTITIES_PER_SIDE:
        _common.logger.warning("suggest-alignments: target side capped at %d entities.", _common._MAX_ENTITIES_PER_SIDE)
        target_nodes = target_nodes[:_common._MAX_ENTITIES_PER_SIDE]

    # Build TF-IDF character-ngram embeddings for all candidate labels.
    all_entities = source_nodes + target_nodes
    all_labels = list({_common._node_label(n) for n in all_entities if _common._node_label(n)})
    embeddings: Optional[Dict[str, List[float]]] = await asyncio.to_thread(_common._tfidf_embedding_vectors, all_labels)
    has_embeddings = embeddings is not None

    # Pre-build token sets for targets to enable O(1) prefiltering (skip zero-overlap pairs).
    target_token_sets: Dict[str, frozenset] = {
        n.get("id", ""): _common._token_set(_common._node_label(n)) for n in target_nodes
    }

    suggestions: List[_common.AlignmentSuggestion] = []
    for source_node in source_nodes:
        source_id = source_node.get("id", "")
        source_label = _common._node_label(source_node)
        source_ontology = _common._node_source_ontology(source_node)
        source_tokens = _common._token_set(source_label)
        source_vec = embeddings.get(source_label) if has_embeddings else None

        for target_node in target_nodes:
            target_id = target_node.get("id", "")
            if not source_id or source_id == target_id:
                continue
            target_ontology = _common._node_source_ontology(target_node)
            if source_ontology and target_ontology and source_ontology == target_ontology:
                continue

            # Token-overlap prefilter: skip pairs with zero shared tokens (Jaccard=0 ⇒ label_sim≈0).
            target_tokens = target_token_sets.get(target_id, frozenset())
            if source_tokens and target_tokens and not (source_tokens & target_tokens):
                continue

            target_label = _common._node_label(target_node)
            label_score = _common._label_similarity(source_label, target_label)

            embedding_sim: Optional[float] = None
            if has_embeddings and source_vec is not None:
                target_vec = embeddings.get(target_label)
                if target_vec is not None:
                    try:
                        embedding_sim = round(_common._cosine_sim(source_vec, target_vec), 4)
                    except Exception:
                        embedding_sim = None

            # Combined score: average label and embedding similarity when both are available.
            if embedding_sim is not None:
                score = round(0.4 * label_score + 0.6 * embedding_sim, 4)
                reason = (
                    f"Label similarity {label_score:.2f}, embedding cosine similarity {embedding_sim:.2f} "
                    f"(TF-IDF character n-gram vectors via SimilarityCalculator)."
                )
            else:
                score = label_score
                reason = f"Label similarity {label_score:.2f}; embedding vectors unavailable."

            if score < body.threshold:
                continue

            relation = _common._candidate_relation(source_node, target_node)
            suggestions.append(_common.AlignmentSuggestion(
                source_uri=source_id,
                source_label=source_label,
                target_uri=target_id,
                target_label=target_label,
                relation=relation,
                score=score,
                label_similarity=label_score,
                embedding_similarity=embedding_sim,
                reason=reason,
            ))

    suggestions.sort(key=lambda item: item.score, reverse=True)
    return suggestions[:body.limit]


@router.get("/health", response_model=_common.OntologyHealthResponse)
async def ontology_health(
    request: Request,
    uri: str = Query(...),
    session: GraphSession = Depends(get_session),
):
    registry = {entry.uri: entry for entry in await _common._registry_entries(request, session)}
    entry = registry.get(uri)
    if entry is None:
        raise HTTPException(status_code=404, detail="Ontology not found in registry.")

    nodes, total_nodes = await asyncio.to_thread(session.get_nodes, skip=0, limit=_common._MAX_ANALYSIS_NODES)
    edges, _ = await asyncio.to_thread(session.get_edges, skip=0, limit=_common._MAX_ANALYSIS_NODES)
    if total_nodes > _common._MAX_ANALYSIS_NODES:
        _common.logger.warning("ontology-health: graph has %d nodes; analysis capped at %d.", total_nodes, _common._MAX_ANALYSIS_NODES)
    entities = _common._ontology_entities(nodes, uri)
    classes = [node for node in entities if _common._classify_node_type(node.get("type", "")) == "class"]
    properties = [node for node in entities if _common._classify_node_type(node.get("type", "")) == "property"]
    assessed = classes + properties

    issues: List[_common.HealthIssue] = []
    total = max(len(assessed), 1)
    with_label = sum(1 for node in assessed if _common._node_label(node))
    with_comment = sum(1 for node in assessed if _common._entity_description(node))
    with_definition = sum(1 for node in assessed if _common._entity_definition(node))
    completeness_score = ((with_label + with_comment + with_definition) / (total * 3)) * 100

    for node in assessed:
        label = _common._node_label(node)
        if not _common._entity_description(node):
            issues.append(_common.HealthIssue(
                id=f"doc:{node.get('id')}",
                severity="warning",
                category="Documentation",
                entity_uri=node.get("id"),
                entity_label=label,
                message=f"{label} is missing a comment or definition.",
                action="Add documentation in Editor.",
            ))

    property_range_edges = {
        edge.get("source")
        for edge in edges
        if edge.get("type") == "rdfs:range"
    }
    missing_range = [node for node in properties if node.get("id") not in property_range_edges]
    for node in missing_range[:25]:
        issues.append(_common.HealthIssue(
            id=f"range:{node.get('id')}",
            severity="info",
            category="Consistency",
            entity_uri=node.get("id"),
            entity_label=_common._node_label(node),
            message="Property has no explicit rdfs:range.",
            action="Review property range in Editor.",
        ))

    consistency_score = max(0.0, 100.0 - (len(missing_range) / max(len(properties), 1)) * 60.0)

    assessed_ids = {node.get("id") for node in assessed if node.get("id")}
    alignments = _common._get_alignment_store(request).values()
    aligned_sources = {
        item.source_uri for item in alignments if item.source_uri in assessed_ids
    } | {
        item.target_uri for item in alignments if item.target_uri in assessed_ids
    }
    alignment_score = (len(aligned_sources) / total) * 100
    if assessed and not aligned_sources:
        issues.append(_common.HealthIssue(
            id=f"alignment:{uri}",
            severity="warning",
            category="Alignment",
            message="No cross-ontology alignments are recorded for local classes or properties.",
            action="Review suggested alignments.",
        ))

    documentation_score = ((with_comment / total) * 80.0) + (20.0 if entry.version or entry.source_url else 0.0)

    try:
        health_nodes, health_edges = await _common._fetch_analysis_graph(session, uri, "shacl-health")
        shacl_turtle, _ = await _common._generated_shacl_for_uri(
            request, session, uri, nodes=health_nodes, edges=health_edges
        )
        data_graph_turtle = await _common._data_graph_turtle_for_uri(
            request, session, uri, nodes=health_nodes, edges=health_edges
        )
        from ....ontology import OntologyEngine
        engine = OntologyEngine()
        report = await asyncio.to_thread(
            engine.validate_graph,
            data_graph_turtle,
            shacl=shacl_turtle,
            data_graph_format="turtle",
            shacl_format="turtle",
        )
        warning_count = len(report.warnings)
        if not report.conforms:
            shacl_score = max(0.0, 100.0 - (report.violation_count * 20.0))
        elif warning_count:
            shacl_score = max(0.0, 100.0 - (warning_count * 5.0))
        else:
            shacl_score = 100.0
        shacl_status = "ok" if report.conforms and not warning_count else "warning"
        detail_parts = []
        if not report.conforms:
            detail_parts.append(f"{report.violation_count} SHACL violation(s)")
        if warning_count:
            detail_parts.append(f"{warning_count} SHACL warning(s)")
        shacl_detail = (
            "Graph conforms to all generated SHACL constraints."
            if not detail_parts
            else f"Graph has {', '.join(detail_parts)}."
        )
        shacl_dimension = _common.HealthDimension(
            key="shacl",
            label="SHACL Conformance",
            score=round(shacl_score, 1),
            status=shacl_status,
            detail=shacl_detail,
        )
    except _common.GraphTruncationError as exc:
        shacl_dimension = _common.HealthDimension(
            key="shacl",
            label="SHACL Conformance",
            score=0.0,
            status="critical",
            detail=str(exc),
        )
    except ImportError:
        shacl_dimension = _common.HealthDimension(
            key="shacl",
            label="SHACL Conformance",
            score=0.0,
            status="unavailable",
            detail="Live SHACL validation is available in SHACL Studio when optional validation dependencies are installed.",
        )
    except Exception as exc:
        shacl_dimension = _common.HealthDimension(
            key="shacl",
            label="SHACL Conformance",
            score=0.0,
            status="critical",
            detail=f"Live SHACL validation failed: {exc}",
        )

    dimensions = [
        _common.HealthDimension(
            key="completeness",
            label="Completeness",
            score=round(completeness_score, 1),
            status="ok" if completeness_score >= 80 else "warning",
            detail=f"{with_label}/{total} labeled, {with_comment}/{total} documented, {with_definition}/{total} defined.",
        ),
        _common.HealthDimension(
            key="consistency",
            label="Consistency",
            score=round(consistency_score, 1),
            status="ok" if consistency_score >= 80 else "warning",
            detail=f"{len(missing_range)} properties are missing explicit ranges.",
        ),
        shacl_dimension,
        _common.HealthDimension(
            key="alignment",
            label="Alignment Coverage",
            score=round(alignment_score, 1),
            status="ok" if alignment_score >= 50 else "warning",
            detail=f"{len(aligned_sources)}/{total} classes or properties have an alignment.",
        ),
        _common.HealthDimension(
            key="documentation",
            label="Documentation",
            score=round(documentation_score, 1),
            status="ok" if documentation_score >= 75 else "warning",
            detail="Measures comments plus source/version metadata.",
        ),
    ]
    scoreable = [dim for dim in dimensions if dim.status != "unavailable"]
    total_score = sum(dim.score for dim in scoreable) / max(len(scoreable), 1)

    return _common.OntologyHealthResponse(
        uri=uri,
        name=entry.name,
        total_score=round(total_score, 1),
        dimensions=dimensions,
        issues=issues[:100],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/shacl/generate", response_model=_common.ShaclGenerateResponse)
async def generate_shacl(
    request: Request,
    body: _common.ShaclGenerateRequest,
    session: GraphSession = Depends(get_session),
):
    try:
        shacl_turtle, shapes = await _common._generated_shacl_for_uri(request, session, body.uri, body.quality_tier)
    except _common.GraphTruncationError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return _common.ShaclGenerateResponse(
        uri=body.uri,
        shacl_turtle=shacl_turtle,
        shape_count=len(shapes),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/shacl/shapes", response_model=_common.ShaclShapesResponse)
async def list_shacl_shapes(
    request: Request,
    uri: str = Query(...),
    session: GraphSession = Depends(get_session),
):
    try:
        shacl_turtle, shapes = await _common._generated_shacl_for_uri(request, session, uri)
    except _common.GraphTruncationError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return _common.ShaclShapesResponse(
        uri=uri,
        shapes=shapes,
        shacl_turtle=shacl_turtle,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/shacl/validate", response_model=_common.ShaclValidationResponse)
async def validate_shacl(
    request: Request,
    body: _common.ShaclValidateRequest,
    session: GraphSession = Depends(get_session),
):
    if not body.shacl_turtle.strip():
        raise HTTPException(status_code=422, detail="SHACL Turtle cannot be empty.")

    _shacl_bytes = body.shacl_turtle.encode("utf-8")
    if len(_shacl_bytes) > _common._MAX_SHACL_TURTLE_BYTES:
        return _common.ShaclValidationResponse(
            uri=body.uri,
            conforms=False,
            status="error",
            message=(
                f"SHACL Turtle size ({len(_shacl_bytes)} bytes) "
                f"exceeds maximum allowed size ({_common._MAX_SHACL_TURTLE_BYTES} bytes); "
                f"set SEMANTICA_MAX_SHACL_TURTLE_BYTES to raise the limit."
            ),
            violations=[],
        )

    # Syntax-check the submitted Turtle with rdflib before claiming anything about it.
    try:
        import rdflib  # type: ignore
        g = rdflib.Graph()
        await asyncio.to_thread(g.parse, data=body.shacl_turtle, format="turtle")
        if len(g) > _common._MAX_SHACL_TRIPLES:
            return _common.ShaclValidationResponse(
                uri=body.uri,
                conforms=False,
                status="error",
                message=(
                    f"SHACL graph triple count ({len(g)}) "
                    f"exceeds maximum allowed limit ({_common._MAX_SHACL_TRIPLES}); "
                    f"set SEMANTICA_MAX_SHACL_TRIPLES to raise the limit."
                ),
                violations=[],
            )
    except ImportError:
        pass  # rdflib unavailable; skip syntax check
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Turtle syntax: {exc}",
        ) from exc

    if not body.uri:
        return _common.ShaclValidationResponse(
            uri=body.uri,
            conforms=False,
            status="unavailable",
            message=(
                "Turtle parsed successfully. "
                "No target ontology URI was provided — "
                "specify 'uri' to execute live SHACL validation against an ontology data graph."
            ),
            violations=[],
        )

    try:
        data_graph_turtle = await _common._data_graph_turtle_for_uri(request, session, body.uri)
        from ....ontology import OntologyEngine
        engine = OntologyEngine()
        semaphore = _common._get_shacl_semaphore()

        async def _run_validation():
            async with semaphore:
                return await asyncio.to_thread(
                    engine.validate_graph,
                    data_graph_turtle,
                    shacl=body.shacl_turtle,
                    data_graph_format="turtle",
                    shacl_format="turtle",
                )

        report = await asyncio.wait_for(
            _run_validation(),
            timeout=_common._MAX_SHACL_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _common.ShaclValidationResponse(
            uri=body.uri,
            conforms=False,
            status="error",
            message=(
                f"SHACL validation timed out after {_common._MAX_SHACL_TIMEOUT_SECONDS} seconds; "
                f"set SEMANTICA_MAX_SHACL_TIMEOUT to raise the timeout."
            ),
            violations=[],
        )
    except _common.GraphTruncationError as exc:
        return _common.ShaclValidationResponse(
            uri=body.uri,
            conforms=False,
            status="unavailable",
            message=str(exc),
            violations=[],
        )
    except ImportError as exc:
        return _common.ShaclValidationResponse(
            uri=body.uri,
            conforms=False,
            status="unavailable",
            message=(
                "Turtle parsed successfully. "
                f"Live SHACL validation is unavailable: {exc}"
            ),
            violations=[],
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _common.ShaclValidationResponse(
            uri=body.uri,
            conforms=False,
            status="error",
            message=f"SHACL validation error: {exc}",
            violations=[],
        )

    violations = [
        _common.ShaclViolation(
            node=str(v.focus_node) if v.focus_node is not None else None,
            path=str(v.result_path) if v.result_path is not None else None,
            severity=str(v.severity or "Violation"),
            message=str(
                v.message
                or v.explanation
                or f"SHACL constraint violation ({v.constraint}) on {v.focus_node}"
            ),
            focus_node=str(v.focus_node) if v.focus_node is not None else None,
            source_shape=str(v.shape) if v.shape is not None else None,
        )
        for v in [*report.violations, *report.warnings, *report.infos]
    ]
    _result_counts = []
    if report.violations:
        _result_counts.append(f"{len(report.violations)} violation(s)")
    if report.warnings:
        _result_counts.append(f"{len(report.warnings)} warning(s)")
    if report.infos:
        _result_counts.append(f"{len(report.infos)} info result(s)")
    summary_msg = (
        f"SHACL validation found {', '.join(_result_counts)}."
        if _result_counts
        else "Graph conforms to SHACL shapes."
    )
    return _common.ShaclValidationResponse(
        uri=body.uri,
        conforms=report.conforms,
        status="success",
        message=summary_msg,
        violations=violations,
        report_text=report.raw_report,
    )
