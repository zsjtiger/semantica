"""``semantica validate`` commands."""

import json
from pathlib import Path
from typing import (Optional)
import click
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def validate(ctx: click.Context) -> None:
    """SHACL shapes and conflict detection."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@validate.command("shacl")
@click.option("--shapes", default=None, type=click.Path(exists=True),
              help="SHACL shapes file (auto-generated if omitted).")
@click.option("--strictness", type=click.Choice(["strict", "moderate", "lenient"]),
              default="moderate", show_default=True)
@click.option("--report", default=None, type=click.Path(),
              help="Write validation report to file.")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def validate_shacl(cli_ctx: _common.CLIContext, shapes: Optional[str], strictness: str,
                   report: Optional[str], local_json: bool) -> None:
    """Validate data against SHACL constraint shapes.

    \b
    Example:
      semantica validate shacl --strictness strict --report report.json
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import OntologyValidator
            v = OntologyValidator(config=cli_ctx.config.to_dict())
            result = v.validate_shacl(shapes_file=shapes, strictness=strictness)
        except ImportError as exc:
            raise click.ClickException(f"Ontology/validation module not available: {exc}") from exc
        payload = result if isinstance(result, dict) else {"valid": bool(result)}
        if report:
            Path(report).write_text(json.dumps(payload, default=str), encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {report}")
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(payload)
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@validate.command("conflicts")
@click.option("--strategy",
              type=click.Choice(["value", "property", "type", "relationship",
                                  "temporal", "logical", "entity", "all"]),
              default="all", show_default=True)
@click.option("--format", "fmt", type=click.Choice(["json", "table"]),
              default="json", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def validate_conflicts(cli_ctx: _common.CLIContext, strategy: str, fmt: str, local_json: bool) -> None:
    """Detect value, type, temporal, and logical conflicts.

    \b
    Example:
      semantica validate conflicts --format json | jq '.conflicts | length'
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..conflicts import detect_conflicts
            result = detect_conflicts(strategy=strategy, config=cli_ctx.config.to_dict())
        except ImportError as exc:
            raise click.ClickException(f"Conflicts module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(result if isinstance(result, dict) else {"conflicts": result})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@validate.command("integrity")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def validate_integrity(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Run graph integrity checks."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..kg import GraphValidator
            v = GraphValidator(config=cli_ctx.config.to_dict())
            result = v.integrity_check()
        except ImportError as exc:
            raise click.ClickException(f"KG module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"valid": bool(result)})
        else:
            _common._ok(cli_ctx, f"Integrity check: {result}")

    _common._run_with_error_handling(_action)
