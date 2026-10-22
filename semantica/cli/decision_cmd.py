"""``semantica decision`` commands."""

from pathlib import Path
from typing import (Any, Dict, List,
                    Optional)
import click
from rich.table import Table
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def decision(ctx: click.Context) -> None:
    """Record, trace, and analyze decisions with causal provenance."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@decision.command("record")
@click.option("--title", required=True, help="Decision title.")
@click.option("--tags", default=None, help="Comma-separated tags.")
@click.option("--valid-from", default=None, help="ISO 8601 validity start.")
@click.option("--valid-until", default=None, help="ISO 8601 validity end.")
@click.option("--rationale", default=None, help="Decision rationale.")
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_record(cli_ctx: _common.CLIContext, title: str, tags: Optional[str],
                    valid_from: Optional[str], valid_until: Optional[str],
                    rationale: Optional[str], local_dry: bool, local_json: bool) -> None:
    """Record a new decision.

    \b
    Example:
      semantica decision record --title "Approve vendor X" --tags "finance,vendor"
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "record decision", json_out=_common._is_json(cli_ctx, local_json),
                 title=title, tags=tag_list)
            return
        try:
            from ..context.decision_methods import record_decision
            cross_ctx: Dict[str, Any] = {"tags": tag_list}
            if valid_from:
                cross_ctx["valid_from"] = valid_from
            if valid_until:
                cross_ctx["valid_until"] = valid_until
            # Map CLI flags to API: title→scenario, rationale→reasoning,
            # first tag (if any)→category, outcome and confidence use defaults.
            category = tag_list[0] if tag_list else "general"
            # --rationale is optional, but ContextGraph rejects empty reasoning.
            # Normalise once so both backends store the same thing rather than
            # one storing "" and the other refusing the command.
            reasoning = rationale or _common._NO_RATIONALE
            if _common._uses_memory_graph(cli_ctx):
                with _common._memory_graph_lock(cli_ctx):
                    graph = _common._load_context_graph(cli_ctx)
                    result = graph.record_decision(
                        category=category,
                        scenario=title,
                        reasoning=reasoning,
                        outcome="recorded",
                        confidence=1.0,
                        # ContextGraph tracks node validity itself, so pass the
                        # dates as validity rather than leaving them in
                        # metadata, where they would not bound the node.
                        # They are kept out of the metadata as well, since the
                        # graph ignores metadata keys that name node fields.
                        metadata={
                            k: v for k, v in cross_ctx.items()
                            if k not in ("valid_from", "valid_until")
                        },
                        valid_from=valid_from,
                        valid_until=valid_until,
                    )
                    _common._save_context_graph(cli_ctx, graph)
            else:
                result = record_decision(
                    _common._get_graph_store(cli_ctx),
                    category=category,
                    scenario=title,
                    reasoning=reasoning,
                    outcome="recorded",
                    confidence=1.0,
                    cross_system_context=cross_ctx,
                )
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho({"id": str(result)})
        else:
            _common._ok(cli_ctx, f"Decision recorded: {result}")

    _common._run_with_error_handling(_action)


@decision.command("list")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_list(cli_ctx: _common.CLIContext, limit: int, fmt: str, local_json: bool) -> None:
    """List recent decisions."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            if _common._uses_memory_graph(cli_ctx):
                results_raw = _common._memory_decision_records(cli_ctx, limit=limit)
            else:
                from ..context.decision_query import DecisionQuery
                dq = DecisionQuery(_common._get_graph_store(cli_ctx))
                results_raw = dq.find_by_time_range(
                    __import__("datetime").datetime.min,
                    __import__("datetime").datetime.now(),
                    limit=limit,
                )
            results = [
                {"id": d.decision_id, "title": d.scenario, "tags": [d.category]}
                for d in (results_raw or [])
            ]
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(results)
        else:
            table = Table(title="[bold]Recent Decisions[/bold]",
                          box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
            table.add_column("ID", style=_common._KEY, no_wrap=True)
            table.add_column("Title")
            table.add_column("Category")
            for d in results:
                table.add_row(str(d.get("id", "")), str(d.get("title", "")),
                              str(d.get("tags", [""])[0]))
            _common.console.print(table)

    _common._run_with_error_handling(_action)


@decision.command("query")
@click.option("--filter", "filter_str", default=None, help="e.g. tag:finance")
@click.option("--since", default=None, help="ISO 8601 date.")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_query(cli_ctx: _common.CLIContext, filter_str: Optional[str],
                   since: Optional[str], fmt: str, local_json: bool) -> None:
    """Filter decisions by tag, entity, or date."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        tag_value = _common._tag_filter_value(filter_str) if filter_str is not None else None
        try:
            import datetime as _dt
            since_dt = _dt.datetime.fromisoformat(since) if since else _dt.datetime.min
            if _common._uses_memory_graph(cli_ctx):
                raw = _common._memory_decision_records(
                    cli_ctx, limit=500,
                    since_ts=since_dt.timestamp() if since else None,
                )
            else:
                from ..context.decision_query import DecisionQuery
                dq = DecisionQuery(_common._get_graph_store(cli_ctx))
                raw = dq.find_by_time_range(since_dt, _dt.datetime.now(), limit=500)
            results: List[Dict[str, Any]] = [
                {"id": d.decision_id, "category": d.category, "scenario": d.scenario,
                 "outcome": d.outcome, "confidence": d.confidence}
                for d in (raw or [])
                if tag_value is None or tag_value in d.category.lower()
            ]
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(results)
        else:
            _common._pprint(cli_ctx, results)

    _common._run_with_error_handling(_action)


@decision.command("trace")
@click.argument("decision_id")
@click.option("--format", "fmt", type=click.Choice(["text", "mermaid", "json"]),
              default="text", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_trace(cli_ctx: _common.CLIContext, decision_id: str, fmt: str, local_json: bool) -> None:
    """Show the full causal decision chain for a decision ID.

    \b
    Example:
      semantica decision trace dec_abc123 --format mermaid
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            if _common._uses_memory_graph(cli_ctx):
                chain_raw = _common._load_context_graph(cli_ctx).get_causal_chain(decision_id)
            else:
                from ..context.decision_methods import get_causal_chain
                chain_raw = get_causal_chain(_common._get_graph_store(cli_ctx), decision_id)
            chain: Any = [
                {"id": d.decision_id, "scenario": d.scenario, "outcome": d.outcome}
                for d in chain_raw
            ]
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(chain if isinstance(chain, (dict, list)) else {"chain": str(chain)})
        else:
            _common.console.print(chain)

    _common._run_with_error_handling(_action)


@decision.command("similar")
@click.argument("decision_id")
@click.option("--top-k", default=3, type=int, show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_similar(cli_ctx: _common.CLIContext, decision_id: str, top_k: int, local_json: bool) -> None:
    """Find precedent decisions similar to the given ID."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..context.decision_methods import find_precedents
            if _common._uses_memory_graph(cli_ctx):
                raw = _common._load_context_graph(cli_ctx).find_precedents(
                    decision_id, limit=top_k
                )
            else:
                raw = find_precedents(
                    _common._get_graph_store(cli_ctx), decision_id, limit=top_k
                )
            results: List[Dict[str, Any]] = [
                {"id": d.decision_id, "scenario": d.scenario, "category": d.category,
                 "confidence": d.confidence}
                for d in (raw or [])
            ]
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(results)
        else:
            _common._pprint(cli_ctx, results)

    _common._run_with_error_handling(_action)


@decision.command("impact")
@click.argument("decision_id")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_impact(cli_ctx: _common.CLIContext, decision_id: str, local_json: bool) -> None:
    """Analyze downstream impact of a decision."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..context.decision_methods import analyze_decision_impact
            if _common._uses_memory_graph(cli_ctx):
                result = _common._load_context_graph(cli_ctx).analyze_decision_impact(
                    decision_id
                )
            else:
                result = analyze_decision_impact(
                    _common._get_graph_store(cli_ctx), decision_id
                )
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@decision.command("check")
@click.argument("decision_id")
@click.option("--rules", default=None, type=click.Path(exists=True),
              help="Policy rules YAML file.")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def decision_check(cli_ctx: _common.CLIContext, decision_id: str, rules: Optional[str],
                   local_json: bool) -> None:
    """Validate a decision against policy rules."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            if _common._uses_memory_graph(cli_ctx):
                graph = _common._load_context_graph(cli_ctx)
                decision = next(
                    (d for d in _common._memory_decisions(cli_ctx) if d["id"] == decision_id),
                    None,
                )
                if decision is None:
                    raise click.ClickException(
                        f"No decision {decision_id!r} in "
                        f"{_common._memory_graph_path(cli_ctx)}. Run 'semantica decision "
                        f"list' to see what is recorded."
                    )
                result = graph.check_decision_rules(
                    decision, rules=_common._load_policy_rules(rules) if rules else None
                )
            else:
                from ..context.decision_methods import check_decision_compliance
                # The API expects a policy_id string; derive it from the rules file
                # stem if provided, otherwise use the decision ID as the policy key.
                policy_id = Path(rules).stem if rules else decision_id
                result = check_decision_compliance(
                    _common._get_graph_store(cli_ctx), decision_id, policy_id
                )
        except ImportError as exc:
            raise click.ClickException(f"Context module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)
