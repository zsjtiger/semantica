"""``semantica visualize`` commands."""

import sys
from pathlib import Path
from typing import (Optional)
import click
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def visualize(ctx: click.Context) -> None:
    """Render interactive graph and ontology diagrams."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _viz_command(name: str, fn_name: str, title: str) -> None:
    @visualize.command(name)
    @click.option("--layout",
                  type=click.Choice(["forceatlas2", "spring", "circular", "hierarchical", "spectral"]),
                  default="spring", show_default=True)
    @click.option("--format", "fmt", type=click.Choice(["html", "svg", "png", "pdf"]),
                  default="html", show_default=True)
    @click.option("--filter", "filter_str", default=None)
    @click.option("--color-scheme",
                  type=click.Choice(["default", "dark", "categorical", "entity-type"]),
                  default="default", show_default=True)
    @click.option("--output", default=None, type=click.Path())
    @click.pass_obj
    def _cmd(cli_ctx: _common.CLIContext, layout: str, fmt: str, filter_str: Optional[str],
             color_scheme: str, output: Optional[str]) -> None:
        cli_ctx = _common._require_ctx(cli_ctx)

        def _action() -> None:
            try:
                from ..visualization import get_visualization_method
                viz_fn = get_visualization_method(fn_name)
                result = viz_fn(
                    layout=layout, format=fmt, filter=filter_str,
                    color_scheme=color_scheme, config=cli_ctx.config.to_dict(),
                )
            except ImportError as exc:
                raise click.ClickException(f"Visualization module not available: {exc}") from exc
            if output:
                if isinstance(result, bytes):
                    Path(output).write_bytes(result)
                else:
                    Path(output).write_text(str(result), encoding="utf-8")
                _common._ok(cli_ctx, f"Wrote {output}")
            else:
                # No --output: emit to stdout so the caller can redirect or pipe.
                if isinstance(result, bytes):
                    sys.stdout.buffer.write(result)
                else:
                    click.echo(result)

        _common._run_with_error_handling(_action)

    _cmd.__doc__ = title


_viz_command("kg", "visualize_kg", "Render an interactive knowledge graph.")


_viz_command("ontology", "visualize_ontology", "Render an OWL class and property diagram.")


_viz_command("embeddings", "visualize_embeddings", "Render the embedding space (UMAP/t-SNE).")


_viz_command("temporal", "visualize_temporal", "Render a timeline and temporal event chart.")


_viz_command("analytics", "visualize_analytics", "Render centrality and community analytics.")
