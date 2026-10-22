"""``semantica temporal`` commands."""

from datetime import datetime, timezone
from typing import (Optional)
import click
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def temporal(ctx: click.Context) -> None:
    """Point-in-time queries and Allen interval algebra."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@temporal.command("snapshot")
@click.option("--at", "at_time", required=True, help="ISO 8601 datetime.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "table"]),
    default="json",
    show_default=True,
)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def temporal_snapshot(
    cli_ctx: _common.CLIContext, at_time: str, fmt: str, local_json: bool
) -> None:
    """Graph state at a specific point in time.

    \b
    Example:
      semantica temporal snapshot --at "2026-01-15T09:00:00Z"
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg import TemporalGraphQuery
        except ImportError as exc:
            raise click.ClickException(
                f"KG temporal module not available: {exc}"
            ) from exc
        graph = _common._load_temporal_kg(cli_ctx)
        tgq = TemporalGraphQuery()
        result = tgq.reconstruct_at_time(graph, at_time)
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(result if isinstance(result, dict) else {"snapshot": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@temporal.command("query")
@click.argument("query_str")
@click.option(
    "--at",
    "at_time",
    default=None,
    help="ISO 8601 datetime to query at (defaults to the current time).",
)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def temporal_query(
    cli_ctx: _common.CLIContext, query_str: str, at_time: Optional[str], local_json: bool
) -> None:
    """Temporal-aware graph query.

    \b
    Note: the query string is reserved for future query parsing; the result is
    currently the graph slice valid at --at regardless of its content.
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg import TemporalGraphQuery
        except ImportError as exc:
            raise click.ClickException(
                f"KG temporal module not available: {exc}"
            ) from exc
        graph = _common._load_temporal_kg(cli_ctx)
        tgq = TemporalGraphQuery()
        result = tgq.query_at_time(
            graph, query_str, at_time or datetime.now(timezone.utc)
        )
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(
                result if isinstance(result, (dict, list)) else {"result": str(result)}
            )
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@temporal.command("history")
@click.argument("entity_id")
@click.option("--since", default=None, help="ISO 8601 date filter.")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def temporal_history(
    cli_ctx: _common.CLIContext,
    entity_id: str,
    since: Optional[str],
    fmt: str,
    local_json: bool,
) -> None:
    """Change history for an entity.

    \b
    Example:
      semantica temporal history entity_alice --since 2025-01-01 --format table
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg.temporal_reasoning import TemporalReasoningEngine
            from ..kg.temporal_model import (
                parse_temporal_value,
                temporal_structure_to_json_ready,
            )
        except ImportError as exc:
            raise click.ClickException(
                f"KG temporal module not available: {exc}"
            ) from exc
        graph = _common._load_temporal_kg(cli_ctx)
        events = TemporalReasoningEngine().timeline_of(entity_id, graph)
        if since:
            since_dt = parse_temporal_value(since)
            if since_dt is not None:
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)

                def _aware(ts: datetime) -> datetime:
                    return (
                        ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
                    )

                events = [
                    event for event in events if _aware(event["timestamp"]) >= since_dt
                ]
        # Serialize through the kg temporal helpers so timestamps are ISO 8601
        # in every output mode, consistent with `temporal distance`, instead of
        # the space-separated str() form json.dumps(default=str) produces.
        ready = temporal_structure_to_json_ready(
            events if isinstance(events, list) else []
        )
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(ready)
        else:
            _common._pprint(cli_ctx, ready)

    _common._run_with_error_handling(_action)


@temporal.command("distance")
@click.option("--event1", required=True, help="First timestamp (ISO 8601).")
@click.option("--event2", required=True, help="Second timestamp (ISO 8601).")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def temporal_distance(
    cli_ctx: _common.CLIContext, event1: str, event2: str, local_json: bool
) -> None:
    """Temporal distance between two timestamps."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg.temporal_model import parse_temporal_value
        except ImportError as exc:
            raise click.ClickException(
                f"KG temporal module not available: {exc}"
            ) from exc
        # Exact values, not a granularity: rounding to whole seconds first
        # collapses any distance smaller than a second to zero.
        t1 = parse_temporal_value(event1)
        t2 = parse_temporal_value(event2)
        if t1 is None or t2 is None:
            raise click.ClickException(
                f"Could not parse timestamp(s): {event1!r}, {event2!r}"
            )
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
        if t2.tzinfo is None:
            t2 = t2.replace(tzinfo=timezone.utc)
        seconds = abs((t2 - t1).total_seconds())
        result = {
            "event1": t1.isoformat(),
            "event2": t2.isoformat(),
            "distance_seconds": seconds,
            "distance_days": seconds / 86400,
        }
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@temporal.command("allen")
@click.option(
    "--interval1", required=True, help="First interval as start/end (ISO 8601)."
)
@click.option(
    "--interval2", required=True, help="Second interval as start/end (ISO 8601)."
)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def temporal_allen(
    cli_ctx: _common.CLIContext, interval1: str, interval2: str, local_json: bool
) -> None:
    """Allen interval algebra relation between two intervals."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg.temporal_reasoning import TemporalReasoningEngine
        except ImportError as exc:
            raise click.ClickException(
                f"Reasoning module not available: {exc}"
            ) from exc
        engine = TemporalReasoningEngine()
        iv1 = _common._parse_allen_interval(interval1)
        iv2 = _common._parse_allen_interval(interval2)
        relation = engine.relation(iv1, iv2)
        relation_name = relation.value if hasattr(relation, "value") else str(relation)
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(
                {
                    "interval1": interval1,
                    "interval2": interval2,
                    "relation": relation_name,
                }
            )
        else:
            _common.console.print(f"{interval1} ── {relation_name} ──► {interval2}")

    _common._run_with_error_handling(_action)
