"""``semantica kg`` commands."""

import json
from typing import (Optional, Sequence)
import click
from rich.table import Table
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def kg(ctx: click.Context) -> None:
    """Knowledge graph and semantic build commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@kg.command("build")
@click.option("--source", "-s", multiple=True, help="Data sources to process.")
@click.option(
    "-c",
    "--config",
    "command_config_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=str),
    default=None,
    help="Path to YAML/JSON config file.",
)
@click.pass_obj
def kg_build(
    cli_ctx: _common.CLIContext,
    source: Sequence[str],
    command_config_path: Optional[str],
):
    """Build a knowledge base from sources."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        _common._run_build_command(cli_ctx, source, command_config_path)

    _common._run_with_error_handling(_action)


# ─── Additional kg subcommands ────────────────────────────────────────────────


@kg.command("query")
@click.argument("query_str")
@click.option("--lang", type=click.Choice(["cypher", "sparql"]), default="cypher",
              show_default=True, help="Query language.")
@click.option("--limit", default=100, type=int, show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_query(cli_ctx: _common.CLIContext, query_str: str, lang: str, limit: int, local_json: bool) -> None:
    """Run a Cypher or SPARQL query against the knowledge graph."""
    cli_ctx = _common._require_ctx(cli_ctx)
    json_out = _common._is_json(cli_ctx, local_json)

    def _action() -> None:
        try:
            from ..graph_store import execute_query
        except ImportError as exc:
            raise click.ClickException(f"Graph store module not available: {exc}") from exc
        result = execute_query(query_str, lang=lang, limit=limit,
                               config=cli_ctx.config.to_dict())
        if json_out:
            _common._jecho(result if isinstance(result, (dict, list)) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@kg.command("stats")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table",
              show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_stats(cli_ctx: _common.CLIContext, fmt: str, local_json: bool) -> None:
    """Show node/edge counts, density, and graph metrics."""
    cli_ctx = _common._require_ctx(cli_ctx)
    json_out = _common._is_json(cli_ctx, local_json) or fmt == "json"

    def _action() -> None:
        try:
            from ..kg import GraphAnalyzer

            stats = GraphAnalyzer(
                config=cli_ctx.config.to_dict()
            ).compute_metrics()
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        if json_out:
            _common._jecho(stats)
        else:
            table = Table(title="[bold]Knowledge Graph Statistics[/bold]",
                          box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
            table.add_column("Metric", style=_common._KEY, no_wrap=True)
            table.add_column("Value", style=_common._VAL)
            for k, v in (stats.items() if isinstance(stats, dict) else []):
                table.add_row(str(k), str(v))
            _common.console.print(table)

    _common._run_with_error_handling(_action)


@kg.command("analyze")
@click.option("--mode", type=click.Choice(["centrality", "community", "connectivity", "all"]),
              default="all", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_analyze(cli_ctx: _common.CLIContext, mode: str, local_json: bool) -> None:
    """Run centrality, community detection, or connectivity analysis."""
    cli_ctx = _common._require_ctx(cli_ctx)
    json_out = _common._is_json(cli_ctx, local_json)

    def _action() -> None:
        try:
            from ..kg import GraphAnalyzer
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        result = GraphAnalyzer(config=cli_ctx.config.to_dict()).analyze(mode=mode)
        if json_out:
            _common._jecho(result if isinstance(result, dict) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@kg.command("find-path")
@click.option("--from", "from_entity", required=True, help="Source entity name.")
@click.option("--to", "to_entity", required=True, help="Target entity name.")
@click.option("--type", "path_type",
              type=click.Choice(["shortest", "semantic", "causal"]),
              default="shortest", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_find_path(cli_ctx: _common.CLIContext, from_entity: str, to_entity: str,
                 path_type: str, local_json: bool) -> None:
    """Find a path between two entities in the knowledge graph."""
    cli_ctx = _common._require_ctx(cli_ctx)
    json_out = _common._is_json(cli_ctx, local_json)

    def _action() -> None:
        try:
            from ..kg import PathFinder
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        path = PathFinder(config=cli_ctx.config.to_dict()).find_path(
            from_entity, to_entity, path_type=path_type
        )
        if json_out:
            _common._jecho(path if isinstance(path, dict) else {"path": path})
        else:
            _common._pprint(cli_ctx, path)

    _common._run_with_error_handling(_action)


@kg.command("resolve")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_resolve(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Run entity resolution across the knowledge graph."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg import EntityResolver
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        result = EntityResolver(config=cli_ctx.config.to_dict()).resolve()
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"result": str(result)})
        else:
            _common._ok(cli_ctx, f"Entity resolution complete: {result}")

    _common._run_with_error_handling(_action)


@kg.command("predict")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_predict(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Run link prediction on the knowledge graph."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg import LinkPredictor
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        result = LinkPredictor(config=cli_ctx.config.to_dict()).predict()
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@kg.command("validate")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_validate_cmd(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Run a graph integrity check."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg import GraphValidator
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        result = GraphValidator(config=cli_ctx.config.to_dict()).validate()
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"result": str(result)})
        else:
            _common._ok(cli_ctx, f"Graph validation: {result}")

    _common._run_with_error_handling(_action)


@kg.command("global")
@click.argument("query_str")
@click.option(
    "--reports",
    "reports_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file containing community reports.",
)
@click.option(
    "--hierarchy",
    "hierarchy_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file containing community hierarchy.",
)
@click.option(
    "--level",
    type=int,
    default=None,
    help="Coarsening level to query.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=4000,
    show_default=True,
    help="Token budget for retrieval context.",
)
@click.option(
    "--min-relevance",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum relevance score (0.0 to 10.0).",
)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_global_cmd(
    cli_ctx: _common.CLIContext,
    query_str: str,
    reports_path: Optional[str],
    hierarchy_path: Optional[str],
    level: Optional[int],
    max_tokens: int,
    min_relevance: float,
    local_json: bool,
) -> None:
    """Run global Map-Reduce search over hierarchical community reports."""
    cli_ctx = _common._require_ctx(cli_ctx)
    json_out = _common._is_json(cli_ctx, local_json)

    def _action() -> None:
        try:
            from ..context.methods import retrieve_global
            from ..kg.community_hierarchy import CommunityHierarchy
        except ImportError as exc:
            raise click.ClickException(f"Module not available: {exc}") from exc

        loaded_reports = None
        if reports_path:
            try:
                with open(reports_path, "r", encoding="utf-8") as f:
                    loaded_reports = json.load(f)
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to read reports file '{reports_path}': {exc}"
                ) from exc
        else:
            raise click.ClickException(
                "Global retrieval requires --reports. "
                "Please provide a path to community reports."
            )

        loaded_hierarchy = None
        if hierarchy_path:
            try:
                with open(hierarchy_path, "r", encoding="utf-8") as f:
                    h_data = json.load(f)
                    loaded_hierarchy = CommunityHierarchy.from_dict(h_data)
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to read hierarchy file '{hierarchy_path}': {exc}"
                ) from exc

        result = retrieve_global(
            query=query_str,
            reports=loaded_reports,
            hierarchy=loaded_hierarchy,
            level=level,
            max_context_tokens=max_tokens,
            min_relevance_score=min_relevance,
        )

        if json_out:
            _common._jecho(result.to_dict())
        else:
            _common._ok(
                cli_ctx,
                f"Global Search Results (Level {result.level}, "
                f"Reports: {len(result.community_reports_used)}):",
            )
            _common.console.print(f"\n{result.response}\n")
            if result.citations:
                _common.console.print(f"Citations: {', '.join(result.citations)}")
            _common.console.print(
                f"Key Points: {len(result.key_points)} | "
                f"Time: {result.metrics.get('time_taken', 0.0):.2f}s"
            )

    _common._run_with_error_handling(_action)


@kg.command("drift")
@click.argument("query_str")
@click.option(
    "--reports",
    "reports_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file containing community reports.",
)
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file containing knowledge graph.",
)
@click.option(
    "--depth",
    type=int,
    default=2,
    show_default=True,
    help="Traversal depth for entity exploration.",
)
@click.option(
    "--drift-threshold",
    type=float,
    default=0.35,
    show_default=True,
    help="Threshold for semantic drift pruning.",
)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def kg_drift_cmd(
    cli_ctx: _common.CLIContext,
    query_str: str,
    reports_path: Optional[str],
    graph_path: Optional[str],
    depth: int,
    drift_threshold: float,
    local_json: bool,
) -> None:
    """Run DRIFT hybrid search combining global framing and local exploration."""
    cli_ctx = _common._require_ctx(cli_ctx)
    json_out = _common._is_json(cli_ctx, local_json)

    def _action() -> None:
        try:
            from ..context.methods import retrieve_drift
        except ImportError as exc:
            raise click.ClickException(f"Module not available: {exc}") from exc

        loaded_reports = None
        if reports_path:
            try:
                with open(reports_path, "r", encoding="utf-8") as f:
                    loaded_reports = json.load(f)
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to read reports file '{reports_path}': {exc}"
                ) from exc

        loaded_graph = None
        if graph_path:
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    loaded_graph = json.load(f)
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to read graph file '{graph_path}': {exc}"
                ) from exc

        result = retrieve_drift(
            query=query_str,
            knowledge_graph=loaded_graph,
            reports=loaded_reports,
            max_depth=depth,
            drift_threshold=drift_threshold,
        )

        if json_out:
            _common._jecho(result.to_dict())
        else:
            _common._ok(
                cli_ctx,
                f"DRIFT Hybrid Search Results (Depth {result.depth_reached}, "
                f"Facts: {len(result.verified_local_contexts)}):",
            )
            _common.console.print(f"\n{result.answer}\n")
            if result.citations:
                _common.console.print(f"Citations: {', '.join(result.citations)}")
            _common.console.print(
                f"Facets Explored: {len(result.facets_explored)} | "
                f"Pruned Facts: {result.pruned_fact_count} | "
                f"Time: {result.metrics.get('time_taken', 0.0):.2f}s"
            )

    _common._run_with_error_handling(_action)
