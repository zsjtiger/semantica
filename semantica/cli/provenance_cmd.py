"""``semantica provenance`` commands."""

import json
from pathlib import Path
from typing import (Optional)
import click
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def provenance(ctx: click.Context) -> None:
    """Lineage, audit logs, and W3C PROV-O export."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@provenance.command("lineage")
@click.argument("entity_id")
@click.option("--depth", default=3, type=int, show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def provenance_lineage(cli_ctx: _common.CLIContext, entity_id: str, depth: int, local_json: bool) -> None:
    """Show data lineage for an entity.

    \b
    Example:
      semantica provenance lineage entity_alice --depth 3
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            result = pm.lineage(entity_id, depth=depth)
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"lineage": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@provenance.command("audit")
@click.option("--since", default=None, help="ISO 8601 date.")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "table"]),
              default="table", show_default=True)
@click.option("--output", default=None, type=click.Path())
@click.option("--json", "local_json", is_flag=True, default=False)
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.pass_obj
def provenance_audit(cli_ctx: _common.CLIContext, since: Optional[str], fmt: str,
                     output: Optional[str], local_json: bool,
                     local_dry: bool) -> None:
    """Export the audit log.

    \b
    Example:
      semantica provenance audit --since 2026-01-01 --format csv --output audit.csv
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "export audit log", since=since, format=fmt, output=output)
            return
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            result = pm.audit_log(since=since, format=fmt)
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        text = json.dumps(result, default=str) if _common._is_json(cli_ctx, local_json) else str(result)
        if output:
            Path(output).write_text(text, encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {output}")
        else:
            click.echo(text)

    _common._run_with_error_handling(_action)


@provenance.command("export")
@click.option("--format", "fmt", type=click.Choice(["turtle", "ntriples", "jsonld"]),
              default="turtle", show_default=True)
@click.option("--base-uri", "base_uri", default=None,
              help="Namespace URI entities/agents/activities are minted under "
                   "(default: ProvenanceManager.DEFAULT_BASE_URI).")
@click.option("--output", default=None, type=click.Path())
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.pass_obj
def provenance_export(cli_ctx: _common.CLIContext, fmt: str, base_uri: Optional[str],
                      output: Optional[str], local_dry: bool) -> None:
    """Export provenance as W3C PROV-O RDF.

    \b
    Example:
      semantica provenance export --format turtle --output prov.ttl
      semantica provenance export --base-uri https://example.org/kg# --output prov.ttl
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "export provenance", format=fmt, base_uri=base_uri, output=output)
            return
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            data = pm.export_prov(format=fmt, base_uri=base_uri)
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        if output:
            Path(output).write_text(data, encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {output}")
        else:
            click.echo(data)

    _common._run_with_error_handling(_action)


@provenance.command("check")
@click.option("--strict", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def provenance_check(cli_ctx: _common.CLIContext, strict: bool, local_json: bool) -> None:
    """Validate provenance integrity."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            result = pm.check(strict=strict)
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        is_valid = not isinstance(result, dict) or result.get("valid", True)
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"valid": bool(result)})
        elif is_valid:
            _common._ok(cli_ctx, f"Provenance check: {result}")
        else:
            _common._warn(cli_ctx, f"Provenance check: {result}")
        if strict and not is_valid:
            raise click.ClickException(
                f"Provenance integrity check failed: {result.get('errors')} error(s)"
            )

    _common._run_with_error_handling(_action)


@provenance.command("invalidate")
@click.argument("entity_id")
@click.option("--by", "agent_id", required=True, help="Agent responsible for the invalidation.")
@click.option("--reason", default=None, help="Human-readable reason for the invalidation.")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.pass_obj
def provenance_invalidate(cli_ctx: _common.CLIContext, entity_id: str, agent_id: str,
                          reason: Optional[str], local_json: bool,
                          local_dry: bool) -> None:
    """Mark a tracked entity as invalidated (tombstone, not a hard delete).

    \b
    Example:
      semantica provenance invalidate entity_alice --by reviewer_jane --reason "Source retracted"
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "invalidate provenance entity", entity_id=entity_id, by=agent_id, reason=reason)
            return
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            result = pm.invalidate(entity_id, agent_id=agent_id, reason=reason)
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result.to_dict())
        else:
            _common._ok(cli_ctx, f"Invalidated {entity_id} (by {agent_id})")

    _common._run_with_error_handling(_action)


@provenance.command("verify-chain")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def provenance_verify_chain(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Verify the hash chain across all provenance entries.

    Detects wholesale row deletion: per-row checksums alone only prove a
    surviving row wasn't edited in place, not that no row is missing.

    \b
    Example:
      semantica provenance verify-chain
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            result = pm.verify_chain()
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        elif result.get("valid"):
            _common._ok(cli_ctx, f"Chain verified: {result.get('total_entries')} entries, no breaks")
        else:
            _common._warn(cli_ctx, f"Chain verification failed: {len(result.get('broken_links', []))} broken link(s)")
            for link in result.get("broken_links", []):
                click.echo(f"  {link}")

    _common._run_with_error_handling(_action)


@provenance.command("descendants")
@click.argument("entity_id")
@click.option("--depth", default=None, type=int, show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def provenance_descendants(cli_ctx: _common.CLIContext, entity_id: str, depth: Optional[int],
                           local_json: bool) -> None:
    """Show downstream descendants (reverse lineage) for an entity.

    The counterpart to `provenance lineage`, which only traces upstream
    ancestors. Answers "entity X was wrong — what downstream facts used it?"

    \b
    Example:
      semantica provenance descendants entity_alice --depth 3
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..provenance import ProvenanceManager
            pm = ProvenanceManager(config=cli_ctx.config.to_dict())
            if depth is not None:
                entries = [e.to_dict() for e in pm.trace_descendants(entity_id, max_depth=depth)]
                result = {"entity_id": entity_id, "depth": depth, "entries": entries}
            else:
                result = pm.get_descendants(entity_id) or {"entity_id": entity_id, "entries": []}
        except ImportError as exc:
            raise click.ClickException(f"Provenance module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)
