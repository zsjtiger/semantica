"""Module-level helpers and data types for ContextGraph."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

import yaml



class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> Dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


# Optional imports for advanced features
try:
    from ..kg import (
        CentralityCalculator,
        CommunityDetector,
        ConnectivityAnalyzer,
        GraphAnalyzer,
        GraphBuilder,
        LinkPredictor,
        NodeEmbedder,
        PathFinder,
        SimilarityCalculator,
    )
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False


class _CausalChain(dict):
    """Dict response that still iterates over hops for legacy callers."""

    def __iter__(self):
        return iter(self.get("hops", []))


def _parse_iso_dt(value: str) -> Optional[datetime]:
    """Parse an ISO datetime string into a tz-naive UTC datetime.

    Supported formats (in priority order):
        - Year-only shorthand:  "1990"  → "1990-01-01"
        - Date-only:            "1990-06-15"
        - Full ISO (with tz):   "1990-06-15T00:00:00+00:00" / "...Z"
        - Full ISO (naive):     "1990-06-15T00:00:00"

    Returns None on failure; callers must treat the node as Always-Active.
    """
    import logging
    import re as _re
    if not value:
        return None
    s = str(value).strip()
    if _re.fullmatch(r"\d{4}", s):
        s = f"{s}-01-01"
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError) as e:
        logging.getLogger("semantica.context").warning(
            "Malformed temporal value %r — treating node as Always-Active. (%s)", value, e
        )
        return None


def _normalize_temporal_input(value: Optional[Union[str, int, float, datetime]]) -> Optional[str]:
    """Normalize supported temporal inputs to ISO strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None).isoformat()
    if isinstance(value, str):
        parsed = _parse_iso_dt(value)
        if parsed is None:
            raise ValueError(f"Temporal value {value!r} is not a valid ISO datetime string")
        return parsed.isoformat()
    raise ValueError("Temporal values must be datetime, epoch seconds, ISO strings, or None")


def normalize_temporal_input(
    value: Optional[Union[str, int, float, datetime]]
) -> Optional[str]:
    """Normalize a temporal value to a tz-naive UTC ISO-8601 string.

    This is the public surface of the normalization logic used throughout
    :class:`ContextGraph` for retraction, purge, and decision timestamps.
    Exposing it lets sibling modules (e.g. :mod:`erasure`) share the same
    normalization without importing the private ``_normalize_temporal_input``.

    Args:
        value: Any of the following:

            * ``None`` — returned as-is (no timestamp).
            * :class:`~datetime.datetime` — converted to UTC if tz-aware,
              then serialized as a tz-naive ISO string
              (e.g. ``"2026-01-01T07:00:00"``).
            * :class:`int` or :class:`float` — interpreted as a POSIX epoch
              seconds value, converted to UTC, serialized as above.
            * :class:`str` — must be a valid ISO-8601 datetime string;
              offset-aware values (including ``Z``) are converted to UTC
              before serialization.  Year-only (``"2026"``) and date-only
              (``"2026-01-15"``) shorthand forms are also accepted.

    Returns:
        A tz-naive UTC ISO-8601 string (e.g. ``"2026-01-01T12:00:00"``),
        or ``None`` when *value* is ``None``.

    Raises:
        ValueError: If *value* is a string that cannot be parsed as an
            ISO-8601 datetime, or if *value* is a type that is not
            supported (e.g. a :class:`~datetime.date` object).
    """
    return _normalize_temporal_input(value)


def _closing_valid_until(current: Optional[str], at_iso: str) -> str:
    """Return the earlier of an existing end bound and a retraction time.

    Retraction closes a validity window and must never widen one: an entity
    added with ``valid_until`` already in the past would otherwise be reported
    active by ``is_active``/``state_at`` for the span between its original end
    and the retraction. An unparseable ``current`` imposes no end bound at all
    (see :func:`_parse_iso_dt`), so ``at_iso`` still closes it.
    """
    if current is None:
        return at_iso
    existing = _parse_iso_dt(current)
    if existing is None:
        return at_iso
    requested = _parse_iso_dt(at_iso)
    if requested is None or existing <= requested:
        return current
    return at_iso


def _pick_first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _default_edge_id(
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    metadata: Dict[str, Any],
    valid_from: Optional[str],
    valid_until: Optional[str],
) -> str:
    payload = json.dumps(
        {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "weight": weight,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "metadata": metadata,
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def _resolve_edge_identity(
    *,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    metadata: Dict[str, Any],
    valid_from: Optional[str],
    valid_until: Optional[str],
    edge_id: Any = None,
    family_id: Any = None,
) -> Tuple[str, str]:
    resolved_edge_id = str(
        _pick_first(
            edge_id,
            _default_edge_id(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata,
                valid_from=valid_from,
                valid_until=valid_until,
            ),
        )
    )
    resolved_family_id = str(_pick_first(family_id, resolved_edge_id))
    return resolved_edge_id, resolved_family_id


def _coerce_metadata_map(*values: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _coerce_node_id(raw_node: Dict[str, Any]) -> Optional[str]:
    value = _pick_first(
        raw_node.get("id"),
        raw_node.get("node_id"),
        raw_node.get("_id"),
        raw_node.get("uri"),
        raw_node.get("key"),
    )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_edge_endpoint(raw_edge: Dict[str, Any], prefix: str) -> Optional[str]:
    prefix = prefix.lower()
    candidates = {
        "source": ["source_id", "source", "start", "start_id", "from", "src", "START_ID", ":START_ID"],
        "target": ["target_id", "target", "end", "end_id", "to", "dst", "END_ID", ":END_ID"],
    }[prefix]
    value = _pick_first(*(raw_edge.get(candidate) for candidate in candidates))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any, default: float = 1.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ContextNode:
    """Context graph node (Internal implementation)."""

    node_id: str
    node_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[str] = None  
    valid_until: Optional[str] = None  

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """Return True if this node is active at the given time (defaults to now).

        Both ``at_time`` and stored bounds are normalized to tz-naive UTC so that
        callers may pass either aware or naive datetimes without raising TypeError.
        """
        if self.valid_from is None and self.valid_until is None:
            return True
        now = at_time if at_time is not None else datetime.utcnow()
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc).replace(tzinfo=None)
        start = _parse_iso_dt(self.valid_from) if self.valid_from is not None else None
        end = _parse_iso_dt(self.valid_until) if self.valid_until is not None else None
        if start is not None and now < start:
            return False
        if end is not None and now > end:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        props = self.properties.copy()
        props.update(self.metadata)
        props["content"] = self.content
        if self.valid_from is not None:
            props["valid_from"] = self.valid_from
        if self.valid_until is not None:
            props["valid_until"] = self.valid_until
        return {"id": self.node_id, "type": self.node_type, "properties": props}


@dataclass
class ContextEdge:
    """Context graph edge (Internal implementation)."""

    source_id: str
    target_id: str
    edge_type: str
    edge_id: str = ""
    weight: float = 1.0
    family_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[str] = None  
    valid_until: Optional[str] = None  

    def __post_init__(self) -> None:
        self.source_id = str(self.source_id)
        self.target_id = str(self.target_id)
        self.edge_type = str(self.edge_type or "related_to")
        if isinstance(self.weight, dict):
            if not isinstance(self.metadata, dict):
                self.metadata = {}
            dict_meta = dict(self.weight)
            self.weight = _coerce_float(dict_meta.pop("weight", 1.0), default=1.0)
            self.metadata = {**dict_meta, **self.metadata}
        else:
            self.weight = _coerce_float(self.weight, default=1.0)
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        self.edge_id, self.family_id = _resolve_edge_identity(
            source_id=self.source_id,
            target_id=self.target_id,
            edge_type=self.edge_type,
            weight=self.weight,
            metadata=self.metadata,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            edge_id=self.edge_id,
            family_id=self.family_id,
        )
    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """Return True if this edge is active at the given time (defaults to now).

        Both ``at_time`` and stored bounds are normalized to tz-naive UTC so that
        callers may pass either aware or naive datetimes without raising TypeError.
        """
        if self.valid_from is None and self.valid_until is None:
            return True
        now = at_time if at_time is not None else datetime.utcnow()
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc).replace(tzinfo=None)
        start = _parse_iso_dt(self.valid_from) if self.valid_from is not None else None
        end = _parse_iso_dt(self.valid_until) if self.valid_until is not None else None
        if start is not None and now < start:
            return False
        if end is not None and now > end:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d = {
            "id": self.edge_id,
            "familyId": self.family_id or self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.edge_type,
            "weight": self.weight,
            "properties": self.metadata,
        }
        if self.valid_from is not None:
            d["valid_from"] = self.valid_from
        if self.valid_until is not None:
            d["valid_until"] = self.valid_until
        return d


_ATTRS_MISSING = object()

#: Edge types that represent an explicitly recorded causal relationship between
#: two decisions. These are authoritative: they are what the caller asserted via
#: add_causal_relationship(), as opposed to relationships inferred from shared
#: entities and timestamps.
_CAUSAL_EDGE_TYPES = ("CAUSED", "INFLUENCED", "PRECEDENT_FOR")

# Causal edges circulate under two vocabularies: this module's canonical
# spellings above, and the present-tense spellings CausalChainAnalyzer also
# accepts ("causes", "influences", "leads_to", "supports"). The present-tense
# forms normalize onto the canonical types for storage; traversal accepts
# both vocabularies so an edge recorded either way is never invisible.
_CAUSAL_EDGE_ALIASES = {
    "CAUSES": "CAUSED",
    "CAUSED": "CAUSED",
    "INFLUENCES": "INFLUENCED",
    "INFLUENCED": "INFLUENCED",
    "PRECEDES": "PRECEDENT_FOR",
    "PRECEDENT_FOR": "PRECEDENT_FOR",
}
_CAUSAL_TRAVERSAL_TYPES = frozenset(_CAUSAL_EDGE_ALIASES) | {
    "LEADS_TO", "LEAD_TO", "SUPPORTS", "SUPPORT",
}
