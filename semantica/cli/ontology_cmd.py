"""``semantica ontology`` commands."""

import json
from pathlib import Path
from typing import (Optional)
import click
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def ontology(ctx: click.Context) -> None:
    """OWL generation, import, SHACL, and SKOS vocabularies."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@ontology.command("generate")
@click.option("--domain", default=None, help="Domain hint (e.g. healthcare).")
@click.option("--output", default=None, type=click.Path())
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_generate(cli_ctx: _common.CLIContext, domain: Optional[str], output: Optional[str],
                      local_dry: bool, local_json: bool) -> None:
    """Auto-generate OWL ontology from graph data.

    \b
    Example:
      semantica ontology generate --domain healthcare --output healthcare.ttl
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "generate ontology", json_out=_common._is_json(cli_ctx, local_json),
                 domain=domain, output=output)
            return
        try:
            from ..ontology import OntologyGenerator
            gen = OntologyGenerator(config=cli_ctx.config.to_dict())
            result = gen.generate(domain=domain)
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        if output:
            Path(output).write_text(str(result), encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {output}")
        elif _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"ontology": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@ontology.command("import")
@click.argument("source")
@click.option("--format", "fmt",
              type=click.Choice(["turtle", "rdfxml", "jsonld", "ntriples"]),
              default=None)
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_import(cli_ctx: _common.CLIContext, source: str, fmt: Optional[str],
                    local_dry: bool, local_json: bool) -> None:
    """Import OWL/RDF/Turtle/JSON-LD from a file or URL.

    \b
    Example:
      semantica ontology import schema.org.ttl --format turtle
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "import ontology", json_out=_common._is_json(cli_ctx, local_json),
                 source=source, format=fmt)
            return
        try:
            from ..ontology import ingest_ontology
            result = ingest_ontology(source, format=fmt, config=cli_ctx.config.to_dict())
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"status": "ok"})
        else:
            _common._ok(cli_ctx, f"Imported: {source}")

    _common._run_with_error_handling(_action)


@ontology.command("validate")
@click.option("--shapes", default=None, type=click.Path(exists=True))
@click.option("--strictness", type=click.Choice(["strict", "moderate", "lenient"]),
              default="moderate", show_default=True)
@click.option("--report", default=None, type=click.Path())
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_validate(cli_ctx: _common.CLIContext, shapes: Optional[str], strictness: str,
                      report: Optional[str], local_json: bool) -> None:
    """Run SHACL validation on the ontology."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import validate_ontology
            result = validate_ontology(shapes_file=shapes, strictness=strictness,
                                       config=cli_ctx.config.to_dict())
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        payload = result if isinstance(result, dict) else {"valid": bool(result)}
        if report:
            Path(report).write_text(json.dumps(payload, default=str), encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {report}")
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(payload)
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@ontology.command("shacl")
@click.option("--output", default=None, type=click.Path())
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_shacl(cli_ctx: _common.CLIContext, output: Optional[str], local_json: bool) -> None:
    """Generate SHACL shapes from the ontology."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import SHACLGenerator
            gen = SHACLGenerator(config=cli_ctx.config.to_dict())
            result = gen.generate()
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        text = str(result)
        if output:
            Path(output).write_text(text, encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {output}")
        elif _common._is_json(cli_ctx, local_json):
            _common._jecho({"shacl": text})
        else:
            click.echo(text)

    _common._run_with_error_handling(_action)


@ontology.group("skos", invoke_without_command=True)
@click.pass_context
def ontology_skos(ctx: click.Context) -> None:
    """Manage SKOS concept schemes and vocabulary hierarchies."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@ontology_skos.command("search")
@click.argument("term")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def skos_search(cli_ctx: _common.CLIContext, term: str, local_json: bool) -> None:
    """Search SKOS concept schemes for a term."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import OntologyGenerator
            gen = OntologyGenerator(config=cli_ctx.config.to_dict())
            result = gen.skos_search(term)
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, (dict, list)) else {"results": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@ontology_skos.command("hierarchy")
@click.argument("uri")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def skos_hierarchy(cli_ctx: _common.CLIContext, uri: str, local_json: bool) -> None:
    """Show concept hierarchy tree for a URI."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import OntologyGenerator
            gen = OntologyGenerator(config=cli_ctx.config.to_dict())
            result = gen.skos_hierarchy(uri)
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"hierarchy": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@ontology.command("align")
@click.option("--source", required=True, type=click.Path(exists=True))
@click.option("--target", required=True, type=click.Path(exists=True))
@click.option("--strategy", type=click.Choice(["semantic", "structural", "lexical"]),
              default="semantic", show_default=True)
@click.option("--output", default=None, type=click.Path())
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_align(cli_ctx: _common.CLIContext, source: str, target: str, strategy: str,
                   output: Optional[str], local_json: bool) -> None:
    """Align two ontologies.

    \b
    Example:
      semantica ontology align --source mine.ttl --target schema.ttl --strategy semantic
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import OntologyGenerator
            gen = OntologyGenerator(config=cli_ctx.config.to_dict())
            result = gen.align(source, target, strategy=strategy)
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        if output:
            _common._write_result_output(Path(output), result)
            _common._ok(cli_ctx, f"Wrote {output}")
        elif _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"alignments": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@ontology.command("health")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]),
              default="table", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_health(cli_ctx: _common.CLIContext, fmt: str, local_json: bool) -> None:
    """Show ontology health dashboard."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..ontology import OntologyValidator
            v = OntologyValidator(config=cli_ctx.config.to_dict())
            result = v.health()
        except ImportError as exc:
            raise click.ClickException(f"Ontology module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(result if isinstance(result, dict) else {"health": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@ontology.command("version")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ontology_version(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Manage ontology versions."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..change_management import OntologyVersionManager
            v = OntologyVersionManager(**cli_ctx.config.to_dict())
            result = v.current()
        except ImportError as exc:
            raise click.ClickException(f"Change management module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"version": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)
