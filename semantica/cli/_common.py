"""Shared CLI infrastructure: the root ``main`` group, helpers and constants."""

import json
import contextlib
import os
import re
import sys
import time
import ast
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List,
                    Optional, Sequence, Tuple)
import yaml
import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from .. import __version__
from ..core.config_manager import Config, ConfigManager
from ..utils.exceptions import SemanticaError
from ..utils.logging import setup_logging

"""
Semantica CLI Entry Point

This module provides the command-line interface for the Semantica framework,
enabling users to interact with the framework via terminal commands.
"""


try:  # POSIX advisory locking
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


try:  # Windows advisory locking
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


# Reconfigure stdout/stderr to UTF-8 on Windows before any other import
# captures sys.stdout (Rich, Click). This prevents UnicodeEncodeError from
# box-drawing characters and emoji on the default cp1252 code page.
if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


if TYPE_CHECKING:
    from ..core.orchestrator import Semantica


console = Console()


# Supported interpreter range: ``requires-python = ">=3.10,<3.14"`` in
# pyproject.toml. tests/test_python_support_policy.py fails if the two drift.
MIN_PYTHON = (3, 10)


MAX_PYTHON_EXCLUSIVE = (3, 14)


def _python_version_check(version: Sequence[int]) -> Tuple[str, Optional[str]]:
    """Return ``(status, hint)`` for an interpreter ``(major, minor, ...)``."""
    supported = (
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        f"-{MAX_PYTHON_EXCLUSIVE[0]}.{MAX_PYTHON_EXCLUSIVE[1] - 1}"
    )
    major_minor = (version[0], version[1])
    if major_minor < MIN_PYTHON:
        return "fail", f"upgrade to Python {supported}"
    if major_minor >= MAX_PYTHON_EXCLUSIVE:
        return "warn", f"Python {supported} is supported; newer versions are untested"
    return "ok", None


# ─── Visual style constants ───────────────────────────────────────────────────
_BRAND    = "bold blue"


_KEY      = "cyan"


_VAL      = "green"


_DIM      = "dim"


_SUCCESS  = "bold green"


_WARN_STY = "bold yellow"


_TABLE_BOX = box.SIMPLE_HEAD   # single underline under headers; ASCII-safe


@dataclass
class CLIContext:
    """Shared runtime context for all CLI commands."""

    config_path: Optional[str]
    config: Config
    log_level: str
    log_level_override: Optional[str] = None
    framework: Optional["Semantica"] = None
    json_output: bool = False
    quiet: bool = False
    no_color: bool = False
    dry_run_global: bool = False
    store_backend: Optional[str] = None
    vector_store_backend: Optional[str] = None
    _start: float = field(default_factory=time.perf_counter)


def _require_ctx(cli_ctx: Optional[CLIContext]) -> CLIContext:
    """Guard against uninitialized CLI context.

    Under normal Click operation (standalone_mode=True) ctx.obj is always set
    before a subcommand runs. This guard protects embedded/library usage where
    standalone_mode=False might leave ctx.obj as None.
    """
    if cli_ctx is None:
        raise click.ClickException(
            "CLI context is uninitialized — this is a bug, please report it."
        )
    return cli_ctx


_ERROR_HINTS: Dict[type, str] = {
    ConnectionRefusedError: "run 'semantica doctor' to check backend connectivity",
    FileNotFoundError:      "check that the path exists and is readable",
    PermissionError:        "check file and directory permissions",
    ImportError:            "install the missing extras: pip install semantica[…]",
    TimeoutError:           "the backend may be overloaded — retry or increase timeout in config",
}


def _json_error_mode() -> bool:
    """True when this invocation promised machine-readable stdout.

    Covers both the global ``--json`` flag (stored on the CLI context) and a
    subcommand's local ``--json`` flag (uniformly named ``local_json``).
    """
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    if ctx.params.get("local_json"):
        return True
    return isinstance(ctx.obj, CLIContext) and ctx.obj.json_output


def _show_error_card(title: str, detail: str, hint: Optional[str] = None) -> None:
    if _json_error_mode():
        # --json promises machine-readable stdout with errors on stderr, so
        # emit a structured error line there instead of a Rich panel.
        click.echo(json.dumps({"error": detail, "type": title}), err=True)
        return
    body = f"[bold]{title}[/bold]\n[{_DIM}]{detail}[/{_DIM}]"
    if hint:
        body += f"\n\n[{_KEY}]→[/{_KEY}] [{_DIM}]{hint}[/{_DIM}]"
    console.print(
        Panel(body, title="[bold red] Error [/bold red]", border_style="red", padding=(0, 2))
    )


def _run_with_error_handling(action: Callable[[], None]) -> None:
    """Run a CLI action with error cards (or JSON-mode stderr errors) on failure."""
    try:
        action()
    except click.ClickException as exc:
        _show_error_card(type(exc).__name__, exc.format_message())
        raise SystemExit(exc.exit_code)
    except SemanticaError as exc:
        cause = exc.__cause__
        hint = _ERROR_HINTS.get(type(cause)) if cause else None
        _show_error_card("Semantica error", str(exc), hint=hint)
        raise SystemExit(1)
    except Exception as exc:
        hint = _ERROR_HINTS.get(type(exc), "run with --log-level DEBUG for a full traceback")
        _show_error_card(type(exc).__name__, str(exc), hint=hint)
        raise SystemExit(1)


def _load_config_data(file_path: Path) -> Dict[str, Any]:
    """Load and validate raw YAML/JSON config data."""
    suffix = file_path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            with file_path.open("r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle)
        elif suffix == ".json":
            with file_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        else:
            raise click.ClickException(
                "Unsupported configuration file format: "
                f"{suffix}. Supported formats: .yaml, .yml, .json"
            )
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise click.ClickException(
            f"Failed to parse configuration file '{file_path}': {exc}"
        ) from exc

    if config_data is None:
        config_data = {}

    if not isinstance(config_data, dict):
        raise click.ClickException(
            "Configuration file must contain a mapping/object at the root."
        )

    return config_data


def default_config_path() -> Path:
    """The config file `semantica init` writes, loaded when --config is absent."""
    return Path.home() / ".semantica" / "config.yaml"


def _resolve_config_path(config_path: Optional[str]) -> Optional[str]:
    """Return the config file to load: --config, else init's file if it exists.

    Without this, `semantica init` wrote ``~/.semantica/config.yaml`` and nothing
    ever read it — so accepting its defaults still left every command using the
    built-in fallback rather than the backend the user had just chosen (#1481).
    """
    if config_path:
        return config_path
    default = default_config_path()
    return str(default) if default.is_file() else None


def _build_runtime_config(
    config_path: Optional[str],
    log_level: Optional[str],
) -> Config:
    """Resolve CLI config from file plus global flag overrides."""
    config_manager = ConfigManager()

    resolved_path = _resolve_config_path(config_path)
    if resolved_path:
        config_data = _load_config_data(Path(resolved_path))
    else:
        config_data = {}

    logging_config = config_data.get("logging")
    if logging_config is not None and not isinstance(logging_config, dict):
        raise click.ClickException(
            "Logging configuration section must contain a mapping/object."
        )

    if log_level:
        if logging_config is None:
            logging_config = {}
            config_data["logging"] = logging_config

        logging_config["level"] = log_level.upper()

    # Keep validation disabled at CLI bootstrap to avoid blocking unrelated commands.
    return config_manager.load_from_dict(config_data, validate=False)


def _setup_cli_logging(
    logging_config: Dict[str, Any],
    *,
    quiet: bool = False,
    json_output: bool = False,
    allow_file_fallback: bool = True,
) -> None:
    """Initialize logging without making the default log file a startup blocker.

    The library logging default writes to ``semantica.log`` in the current
    working directory. CLI commands should still be usable from read-only or
    restricted directories, so default file-handler failures fall back to
    console-only logging. If the user explicitly configured a log file, keep the
    failure actionable instead of silently ignoring their configuration.
    """
    try:
        setup_logging(config=logging_config)
    except OSError as exc:
        if not allow_file_fallback:
            raise

        fallback_config = {**logging_config, "file": None}
        setup_logging(config=fallback_config)
        # Do not emit a warning: completion scripts and machine-readable
        # commands must keep stdout/stderr stable when the default log file is
        # unavailable. Explicitly configured log-file failures still raise.
        del quiet, json_output, exc  # suppress unused-variable lint


def _get_framework(cli_ctx: CLIContext) -> "Semantica":
    """Lazily initialize framework only when a command needs it."""
    if cli_ctx.framework is None:
        from ..core.orchestrator import Semantica

        cli_ctx.framework = Semantica(config=cli_ctx.config.to_dict())
    return cli_ctx.framework


def _check_log_directory(cli_ctx: CLIContext) -> str:
    """Verify the configured log directory is writable.

    The CLI may already hold an open file handle for ``semantica.log``. Probe
    with a unique temporary file so Windows can check writability without
    deleting the active log file.
    """
    log_file = cli_ctx.config.get("logging.file", "semantica.log")
    if not log_file:
        return "console-only logging"

    log_path = Path(log_file).expanduser()
    log_dir = log_path.parent
    if not log_dir or str(log_dir) == ".":
        log_dir = Path.cwd()
    elif not log_dir.is_absolute():
        log_dir = Path.cwd() / log_dir

    probe_path: Optional[Path] = None
    try:
        probe_path = (
            log_dir
            / f".semantica-doctor-{os.getpid()}-{time.time_ns()}.tmp"
        )
        probe_path.write_text("ok", encoding="utf-8")
        return f"{log_dir} writable"
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                # Some locked-down Windows environments allow writes but deny
                # unlinking. Logging can still proceed, so cleanup is best-effort.
                pass


def _run_build(cli_ctx: CLIContext, sources: Sequence[str]) -> None:
    """Thin wrapper around existing build orchestration flow.

    build_knowledge_base is expected to return a dict of the form::

        {
            "statistics": {
                "sources_processed": <int>,
                ...
            },
            ...
        }

    Both the top-level dict and the "statistics" key are optional; the
    function degrades gracefully when either is absent or None.
    """
    if not sources:
        raise click.UsageError(
            "At least one source is required. Use --source/-s one or more times."
        )

    if cli_ctx.dry_run_global:
        _dry(cli_ctx, "build knowledge base", sources=list(sources))
        return

    framework = _get_framework(cli_ctx)
    # Every build call is kept: the multi-source progress loop below calls
    # build_knowledge_base once per file, and keeping only the last result
    # would make the empty-graph check and the reported counts reflect one
    # file instead of the whole build.
    results: List[Dict[str, Any]] = []
    if cli_ctx.quiet or cli_ctx.json_output:
        results.append(framework.build_knowledge_base(sources=list(sources)))
    elif len(sources) == 1:
        with console.status(
            f"[{_DIM}]Building knowledge base from {Path(sources[0]).name}…[/{_DIM}]",
            spinner="dots",
        ):
            results.append(framework.build_knowledge_base(sources=list(sources)))
    elif not cli_ctx.json_output:
        with Progress(
            SpinnerColumn(),
            TextColumn("[{task.description}]", style=_DIM),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("waiting", total=len(sources))
            for src in sources:
                progress.update(task, description=Path(src).name)
                results.append(framework.build_knowledge_base(sources=[src]))
                progress.advance(task)

    results = [r for r in results if isinstance(r, dict)]
    processed_counts = [
        r.get("statistics", {}).get("sources_processed") for r in results
    ]
    processed_counts = [c for c in processed_counts if c is not None]
    processed = sum(processed_counts) if processed_counts else None

    # A source count alone does not mean anything was built: with no pipeline
    # configured the default pipeline passes its input through untouched, so
    # every source is "processed" and the graph stays empty. Reporting success
    # there hid the failure completely (#1352). Only results that actually
    # carry a knowledge_graph are judged; a result without one is not evidence
    # of an empty graph.
    graphs = [r["knowledge_graph"] or {} for r in results if "knowledge_graph" in r]
    entity_count = sum(len(g.get("entities") or []) for g in graphs)
    relationship_count = sum(len(g.get("relationships") or []) for g in graphs)

    if graphs and processed and not entity_count and not relationship_count:
        raise click.ClickException(
            f"{processed} source(s) processed but the knowledge graph is empty "
            "— no entities or relationships were extracted. This usually means "
            "no pipeline was configured, so the sources were read but never "
            "parsed or extracted. Supply a pipeline with --config, or use "
            "'semantica extract' to check the sources yield entities."
        )

    if processed is not None:
        _ok(
            cli_ctx,
            f"Knowledge base built — {processed} source(s) processed, "
            f"{entity_count} entities, {relationship_count} relationships.",
        )
    else:
        _ok(cli_ctx, "Knowledge base build completed.")


def _run_build_command(
    cli_ctx: CLIContext,
    source: Sequence[str],
    command_config_path: Optional[str],
) -> None:
    """Execute build command path with optional command-level config override.

    When a per-command config file is supplied, the logging configuration from
    that file is re-applied (setup_logging clears existing handlers before
    adding new ones, so there is no handler accumulation risk).
    """
    if command_config_path:
        cmd_config = _build_runtime_config(
            command_config_path, cli_ctx.log_level_override
        )
        # Re-apply logging so the command-level logging section takes effect.
        _setup_cli_logging(
            cmd_config.get("logging", {}),
            quiet=cli_ctx.quiet,
            json_output=cli_ctx.json_output,
            allow_file_fallback=False,
        )
        command_ctx = CLIContext(
            config_path=_resolve_config_path(command_config_path),
            config=cmd_config,
            log_level=cli_ctx.log_level,
            log_level_override=cli_ctx.log_level_override,
            # Propagate all global flag state so command-level config overrides
            # don't silently drop --json, --quiet, --dry-run, or backend flags.
            json_output=cli_ctx.json_output,
            quiet=cli_ctx.quiet,
            no_color=cli_ctx.no_color,
            dry_run_global=cli_ctx.dry_run_global,
            store_backend=cli_ctx.store_backend,
            vector_store_backend=cli_ctx.vector_store_backend,
        )
        _run_build(command_ctx, source)
    else:
        _run_build(cli_ctx, source)


# ─── Banner, startup dashboard, Rich help ─────────────────────────────────────

_BANNER = (
    " ███████╗███████╗███╗   ███╗ █████╗ ███╗   ██╗████████╗██╗ ██████╗  █████╗ \n"
    " ██╔════╝██╔════╝████╗ ████║██╔══██╗████╗  ██║╚══██╔══╝██║██╔════╝ ██╔══██╗\n"
    " ███████╗█████╗  ██╔████╔██║███████║██╔██╗ ██║   ██║   ██║██║      ███████║\n"
    " ╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║   ██║   ██║██║      ██╔══██║\n"
    " ███████║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║   ██║╚██████╗ ██║  ██║\n"
    " ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═╝"
)


_HELP_SECTIONS: List[Tuple[str, List[str]]] = [
    ("📥 Data Ingestion",   ["ingest", "watch", "parse", "split", "normalize"]),
    ("🧠 Intelligence",     ["extract", "deduplicate", "reason", "decision", "temporal"]),
    ("🕸️  Knowledge Graph", ["kg"]),
    ("📊 Analytics",        ["embed", "validate", "ontology", "provenance"]),
    ("📤 Export & Viz",     ["export", "visualize"]),
    ("⚙️  Infrastructure",  ["store", "backup", "pipeline", "config"]),
    ("🖥️  Services",        ["server", "explorer", "mcp"]),
    ("🛠️  Tools",           ["init", "doctor", "changelog", "info", "shell", "completion"]),
]


class RichGroup(click.Group):
    """click.Group that renders --help with Rich-formatted grouped sections."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        import io
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        width = min(getattr(formatter, "width", 100) or 100, 100)
        h = Console(file=io.StringIO(), no_color=not is_tty, width=width, highlight=False)
        buf = h.file  # type: ignore[attr-defined]

        h.print()
        h.print(
            Panel(
                Text.from_markup(
                    f"[{_BRAND}]semantica[/{_BRAND}]  [{_DIM}]v{__version__}[/{_DIM}]\n"
                    "Knowledge Intelligence Platform"
                ),
                box=box.ROUNDED, padding=(0, 2), expand=False,
            )
        )
        h.print(f"[{_DIM}]Usage:[/{_DIM}]  semantica [OPTIONS] COMMAND [ARGS]...\n")

        for section_title, cmd_names in _HELP_SECTIONS:
            rows: List[Tuple[str, str]] = []
            for name in cmd_names:
                cmd = self.get_command(ctx, name)
                if cmd is None or cmd.hidden:
                    continue
                rows.append((name, cmd.get_short_help_str(limit=55)))
            if not rows:
                continue
            h.print(f"  [{_KEY}]{section_title}[/{_KEY}]")
            tbl = Table(box=None, show_header=False, show_edge=False, padding=(0, 2, 0, 4))
            tbl.add_column("cmd", style=_KEY, no_wrap=True, min_width=14)
            tbl.add_column("desc", style=_DIM)
            for name, desc in rows:
                tbl.add_row(name, desc)
            h.print(tbl)

        h.print(f"\n  [{_KEY}]Options[/{_KEY}]")
        opt_tbl = Table(box=None, show_header=False, show_edge=False, padding=(0, 2, 0, 4))
        opt_tbl.add_column("flag", style=_KEY, no_wrap=True, min_width=28)
        opt_tbl.add_column("desc", style=_DIM)
        for param in ctx.command.params:
            if isinstance(param, click.Option) and not param.hidden:
                opt_tbl.add_row("  ".join(param.opts), param.help or "")
        h.print(opt_tbl)

        h.print(f"\n  [{_KEY}]Quick Start[/{_KEY}]")
        for cmd_str, desc in [
            ("semantica ingest data/",    "load documents into the knowledge base"),
            ("semantica extract doc.pdf", "extract entities and relations"),
            ("semantica kg build",         "build the knowledge graph"),
            ("semantica reason run",       "run the reasoning engine"),
            ("semantica shell",            "interactive REPL"),
        ]:
            h.print(f"    [{_VAL}]{cmd_str:<38}[/{_VAL}][{_DIM}]{desc}[/{_DIM}]")
        h.print()
        formatter.write(buf.getvalue())


def _show_startup(cli_ctx: CLIContext) -> None:
    """Dashboard printed when semantica is run with no subcommand."""
    if cli_ctx.quiet or cli_ctx.json_output:
        return
    cfg = cli_ctx.config.to_dict()
    graph_store = _resolve_graph_backend(cli_ctx)
    vector_store = (
        cli_ctx.vector_store_backend
        or cfg.get("vector_store", {}).get("backend", "faiss")
    )
    profile = cfg.get("profile", "default")

    console.print()
    console.print(Text(_BANNER, style=_BRAND), justify="center")
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[{_DIM}]Knowledge Intelligence Platform  •  v{__version__}[/{_DIM}]\n\n"
                f"[{_KEY}]🕸️  Context Graphs[/{_KEY}]      "
                f"[{_KEY}]⚡ Decision Intelligence[/{_KEY}]      "
                f"[{_KEY}]🔍 Provenance[/{_KEY}]\n"
                f"[{_KEY}]🧩 Knowledge Fusion[/{_KEY}]    "
                f"[{_KEY}]🧠 Reasoning Engine[/{_KEY}]          "
                f"[{_KEY}]📊 Explainability[/{_KEY}]"
            ),
            box=box.ROUNDED,
            padding=(1, 4),
            expand=False,
        )
    )
    console.print()
    tbl = Table(box=_TABLE_BOX, show_edge=False, padding=(0, 2))
    tbl.add_column("", style=_KEY, no_wrap=True)
    tbl.add_column("", style=_VAL)
    tbl.add_row("Graph Store",  graph_store)
    tbl.add_row("Vector Store", vector_store)
    tbl.add_row("Profile",      profile)
    tbl.add_row("Config",       cli_ctx.config_path or "(defaults)")
    console.print(tbl)
    console.print()
    console.print(
        f"  [{_DIM}]Run [bold]semantica --help[/bold] for all commands  •  "
        f"[bold]semantica shell[/bold] for interactive mode[/{_DIM}]"
    )
    console.print()


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    cls=RichGroup,
)
@click.version_option(version=__version__)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=str),
    default=None,
    help="Path to YAML/JSON config file.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False,
    ),
    default=None,
    help="Override logging level for this CLI invocation.",
)
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Machine-readable JSON to stdout; errors to stderr.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress all informational output.")
@click.option("--no-color", is_flag=True, default=False,
              help="Disable colored output.")
@click.option("--dry-run", "global_dry_run", is_flag=True, default=False,
              help="Preview without writing (all write commands).")
@click.option("--store", "store_backend", default=None,
              help="Override graph store backend.")
@click.option("--vector-store", "vector_store_backend", default=None,
              help="Override vector store backend.")
@click.option("--profile", default=None, help="Named config profile.")
@click.pass_context
def main(
    ctx: click.Context,
    config_path: Optional[str],
    log_level: Optional[str],
    json_output: bool,
    quiet: bool,
    no_color: bool,
    global_dry_run: bool,
    store_backend: Optional[str],
    vector_store_backend: Optional[str],
    profile: Optional[str],
) -> None:
    """Semantica - Semantic Layer & Knowledge Engineering Framework"""
    try:
        config = _build_runtime_config(config_path=config_path, log_level=log_level)
        # Always initialize logging so file handlers are installed; --quiet only
        # suppresses console output (controlled via _ok/_dry checks).
        _setup_cli_logging(
            config.get("logging", {}),
            quiet=quiet,
            json_output=json_output,
            allow_file_fallback=config_path is None,
        )
        effective_log_level = config.get("logging.level", "INFO")
        # Reinitialize the module-level console if --no-color was requested so
        # all subsequent console.print() calls in this invocation respect the flag.
        if no_color:
            global console  # noqa: PLW0603
            console = Console(no_color=True)
        ctx.obj = CLIContext(
            config_path=_resolve_config_path(config_path),
            config=config,
            log_level=effective_log_level,
            log_level_override=log_level.upper() if log_level else None,
            json_output=json_output,
            quiet=quiet,
            no_color=no_color,
            dry_run_global=global_dry_run,
            store_backend=store_backend,
            vector_store_backend=vector_store_backend,
        )
        if ctx.invoked_subcommand is None:
            _show_startup(ctx.obj)
    except click.ClickException:
        raise
    except SemanticaError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Failed to initialize CLI: {exc}") from exc


class _DeepEmbeddingFailure(Exception):
    """A deep-probe failure from doctor's embedding checks.

    Marks failures that happened AFTER the backend imported cleanly — model
    load, probe, or runtime problems — so the check's hint can point at the
    real remediation instead of `pip install`.
    """


# ─── Graph store helper ──────────────────────────────────────────────────────


MEMORY_GRAPH_BACKEND = "memory"


_NO_RATIONALE = "(no rationale provided)"


_LOCK_TIMEOUT_SECONDS = 10.0


# `semantica init` offers memory as its default and ContextGraph serves it
# without any server, so it is also the fallback when nothing is configured.
_DEFAULT_GRAPH_BACKEND = MEMORY_GRAPH_BACKEND


def _resolve_graph_backend(cli_ctx: CLIContext) -> str:
    """Return the graph backend this invocation will actually use.

    Single source of truth for the three places that used to resolve this
    independently (status, ``doctor`` and ``_get_graph_store``) with two
    different defaults — the inconsistency behind #1481.
    """
    graph_db = cli_ctx.config.to_dict().get("graph_db", {})
    return cli_ctx.store_backend or graph_db.get(
        "backend", _DEFAULT_GRAPH_BACKEND
    )


def _uses_memory_graph(cli_ctx: CLIContext) -> bool:
    """True when decisions and exports are served by an in-process ContextGraph.

    ``GraphStore`` has no ``memory`` backend — it supports neo4j, falkordb,
    age and neptune — so ``memory`` is served by ``ContextGraph``, which
    already works in memory and is what the Python quickstart uses.
    """
    return _resolve_graph_backend(cli_ctx) == MEMORY_GRAPH_BACKEND


def _memory_graph_path(cli_ctx: CLIContext) -> Path:
    """Where the memory backend's graph is persisted between invocations.

    The CLI is one process per command, so an in-memory graph that is never
    written back would make ``decision record`` pointless. Defaults to
    ``~/.semantica/context_graph.json`` (beside ``config.yaml``); override with
    ``graph_db.path`` in the config.
    """
    configured = cli_ctx.config.to_dict().get("graph_db", {}).get("path")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".semantica" / "context_graph.json"


def _load_context_graph(cli_ctx: CLIContext) -> Any:
    """Return the persisted ContextGraph, or an empty one if none exists yet."""
    from ..context import ContextGraph

    graph = ContextGraph()
    path = _memory_graph_path(cli_ctx)
    if path.exists():
        try:
            graph.load_from_file(path)
        except Exception as exc:
            raise click.ClickException(
                f"Could not read the graph at {path}: {exc}"
            ) from exc
    return graph


def _save_context_graph(cli_ctx: CLIContext, graph: Any) -> None:
    """Persist the ContextGraph so the next command sees this one's writes."""
    path = _memory_graph_path(cli_ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.save_to_file(path)


def _try_lock_exclusive(fd: int) -> bool:
    """Take an exclusive advisory lock on ``fd`` without blocking."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    if msvcrt is not None:  # pragma: no cover - Windows only
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    # No locking primitive available: proceed rather than refuse to record.
    return True


def _unlock(fd: int) -> None:
    """Release the advisory lock held on ``fd``."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows only
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def _memory_graph_lock(cli_ctx: CLIContext) -> Iterator[None]:
    """Serialise read-modify-write of the memory graph across CLI processes.

    ``decision record`` loads the whole graph, adds a node and writes it back.
    Two invocations that load before either saves would silently lose the
    earlier decision — ``save_to_file`` replaces the file atomically, so the
    write is never torn, but the lost update is invisible.

    Uses an OS advisory lock (``flock``, or ``msvcrt.locking`` on Windows) on a
    sidecar ``.lock`` file. The kernel releases it when the holder exits, so
    there is no staleness timeout to guess at and a crashed writer cannot wedge
    the CLI. Deliberately never unlinks the file: removing it would let a
    second process create a fresh inode and lock that instead, re-introducing
    the double-entry this prevents — which is exactly what an mtime-based
    "reclaim an old lock" rule does to a writer that is merely slow or
    suspended.
    """
    lock_path = _memory_graph_path(cli_ctx).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    try:
        while not _try_lock_exclusive(fd):
            if time.monotonic() >= deadline:
                raise click.ClickException(
                    f"Timed out waiting for another semantica command to finish "
                    f"writing {_memory_graph_path(cli_ctx)}."
                )
            time.sleep(0.05)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _decision_from_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a ContextGraph decision node to the CLI's decision shape.

    Mirrors the fields the ``GraphStore`` path reads off a ``Decision``
    (``decision_id``, ``scenario``, ``category``, ``outcome``, ``confidence``)
    so both backends print identically.
    """
    meta = node.get("metadata", {}) or {}
    # Carry every stored field, not just the ones the tables print:
    # check_decision_rules() indexes decision["reasoning"] and reads
    # decision_maker, valid_from/valid_until and nested metadata.
    decision = {
        "id": node.get("id", ""),
        "decision_id": node.get("id", ""),
        "scenario": meta.get("scenario", node.get("content", "")),
        "category": meta.get("category", ""),
        "outcome": meta.get("outcome", ""),
        "confidence": meta.get("confidence", 0.0),
        "timestamp": meta.get("timestamp", 0.0),
        "recorded_at": meta.get("recorded_at", ""),
        "reasoning": meta.get("reasoning", ""),
        "decision_maker": meta.get("decision_maker") or "",
    }
    for key in ("valid_from", "valid_until", "metadata"):
        if key in meta:
            decision[key] = meta[key]
    return decision


def _memory_decisions(cli_ctx: CLIContext, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Decisions from the memory backend, newest first."""
    graph = _load_context_graph(cli_ctx)
    decisions = [
        _decision_from_node(n) for n in graph.get_nodes_by_label("decision")
    ]
    decisions.sort(key=lambda d: d.get("timestamp") or 0.0, reverse=True)
    return decisions[:limit] if limit is not None else decisions


def _memory_decision_records(
    cli_ctx: CLIContext,
    limit: Optional[int] = None,
    since_ts: Optional[float] = None,
) -> List[SimpleNamespace]:
    """Memory-backend decisions as records with a ``Decision``-shaped surface.

    The commands read ``.decision_id`` / ``.scenario`` / ``.category`` /
    ``.outcome`` / ``.confidence`` off whatever the store returns, so matching
    that surface lets both backends share one formatting path.
    """
    rows = [
        d for d in _memory_decisions(cli_ctx)
        if since_ts is None or (d.get("timestamp") or 0.0) >= since_ts
    ]
    if limit is not None:
        rows = rows[:limit]
    return [
        SimpleNamespace(
            decision_id=d["id"], scenario=d["scenario"], category=d["category"],
            outcome=d["outcome"], confidence=d["confidence"],
        )
        for d in rows
    ]


def _get_graph_store(cli_ctx: CLIContext) -> Any:
    """Return a GraphStore instance wired from the current CLIContext config."""
    from ..graph_store import GraphStore
    cfg = cli_ctx.config.to_dict()
    graph_db = dict(cfg.get("graph_db", {}))
    graph_db.pop("path", None)
    # Pop unconditionally: `store_backend or graph_db.pop(...)` short-circuits,
    # so an override left the configured backend in the kwargs and GraphStore
    # received it twice.
    configured = graph_db.pop("backend", None)
    backend = cli_ctx.store_backend or configured or _DEFAULT_GRAPH_BACKEND
    if backend == MEMORY_GRAPH_BACKEND:
        # Reached only if a new caller forgets to route memory through
        # ContextGraph; GraphStore would raise "Unknown backend: memory".
        raise click.ClickException(
            "The 'memory' backend is served by ContextGraph, not GraphStore — "
            "this command has not been wired for it yet. Use --store neo4j "
            "(or falkordb/age/neptune) for a real graph database."
        )
    return GraphStore(backend=backend, **graph_db)


def _load_policy_rules(path: str) -> Dict[str, Any]:
    """Load a decision policy rules file as a mapping.

    Distinct from ``_load_rule_definitions()``, which loads reasoning rules as a
    list of strings. ``ContextGraph.check_decision_rules()`` takes a mapping of
    policy constraints, so anything else is a usage error worth naming.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise click.ClickException(
            f"Policy rules file '{path}' must be a YAML mapping of rules; "
            f"got {type(data).__name__}."
        )
    return data


def _load_rule_definitions(path: str) -> List[str]:
    """Load reasoning rule definitions from a YAML or plain-text rules file.

    YAML files may hold a list of rule strings or a mapping with a ``rules``
    list; anything else (e.g. Datalog) is read as one rule per non-comment
    line. The strings are handed to ``Reasoner.add_rule()`` untouched.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        rules_value = data.get("rules")
        if rules_value is None and "rules" not in data:
            raise click.ClickException(
                f"Rules file '{path}' is a YAML mapping but has no 'rules' key. "
                "Expected either a YAML list or a mapping with a 'rules' list."
            )
        data = rules_value
    if isinstance(data, list):
        return [str(item) for item in data]
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _load_premises(path: str) -> List[Any]:
    """Load DeductiveReasoner ``Premise`` objects from a YAML file.

    Mirrors _load_rule_definitions()'s format conventions: a YAML list of
    plain statement strings, a list of ``{statement, confidence}`` mappings,
    a top-level ``{"premises": [...]}`` mapping, or plain-text lines.

    ``confidence`` is accepted and parsed onto the Premise (the dataclass
    has the field, so a doc-shaped input shouldn't be rejected), but
    DeductiveReasoner.apply_logic() never reads premise.confidence -- every
    derived Conclusion's confidence comes from the firing rule's confidence
    instead. Setting it currently has no effect on the result.
    """
    from ..reasoning.deductive_reasoner import Premise
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        if "premises" not in data:
            raise click.ClickException(
                f"Premises file '{path}' is a YAML mapping but has no 'premises' key. "
                "Expected either a YAML list or a mapping with a 'premises' list."
            )
        premises_value = data["premises"]
        # A present-but-non-list value (e.g. "premises: null") must be a
        # clear error, not silently fall through to reinterpreting the raw
        # file text as one-statement-per-line plain text.
        if not isinstance(premises_value, list):
            raise click.ClickException(
                f"Premises file '{path}' has a 'premises' key that is not a list."
            )
        data = premises_value
    elif not isinstance(data, list):
        data = [line.strip() for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    premises = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            statement = item.get("statement")
            if not statement:
                raise click.ClickException(
                    f"Premise entry {i} in '{path}' is missing 'statement'.")
            premises.append(Premise(
                premise_id=str(item.get("id", f"premise_{i}")),
                statement=str(statement),
                confidence=float(item.get("confidence", 1.0)),
            ))
        else:
            premises.append(Premise(premise_id=f"premise_{i}", statement=str(item)))
    if not premises:
        raise click.ClickException(f"Premises file '{path}' contains no premises.")
    return premises


def _load_observations(path: str) -> List[Any]:
    """Load AbductiveReasoner ``Observation`` objects from a YAML file.

    Mirrors _load_rule_definitions()'s format conventions: a YAML list of
    plain description strings, a list of ``{description, facts}`` mappings,
    a top-level ``{"observations": [...]}`` mapping, or plain-text lines.

    ``facts`` is accepted and parsed onto the Observation (the dataclass
    has the field), but nothing in abductive_reasoner.py currently reads
    observation.facts -- hypothesis generation only matches a rule's
    conclusion against observation.description. Setting it currently has
    no effect on the result.
    """
    from ..reasoning.abductive_reasoner import Observation
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        if "observations" not in data:
            raise click.ClickException(
                f"Observations file '{path}' is a YAML mapping but has no 'observations' key. "
                "Expected either a YAML list or a mapping with an 'observations' list."
            )
        obs_value = data["observations"]
        # A present-but-non-list value (e.g. "observations: null") must be
        # a clear error, not silently fall through to reinterpreting the
        # raw file text as one-description-per-line plain text -- that
        # path could otherwise hand find_explanations() a single bogus
        # observation and report a misleading "success".
        if not isinstance(obs_value, list):
            raise click.ClickException(
                f"Observations file '{path}' has an 'observations' key that is not a list."
            )
        data = obs_value
    elif not isinstance(data, list):
        data = [line.strip() for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    observations = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            description = item.get("description")
            if not description:
                raise click.ClickException(
                    f"Observation entry {i} in '{path}' is missing 'description'.")
            observations.append(Observation(
                observation_id=str(item.get("id", f"obs_{i}")),
                description=str(description),
                facts=item.get("facts", []),
            ))
        else:
            observations.append(Observation(observation_id=f"obs_{i}", description=str(item)))
    if not observations:
        raise click.ClickException(f"Observations file '{path}' contains no observations.")
    return observations


def _graph_store_facts(cli_ctx: CLIContext) -> List[str]:
    """Read the configured graph store into Reasoner fact strings.

    Follows the same conventions ``Reasoner.add_fact()`` applies to
    KG-style dicts: nodes become ``Label(name)`` and relationships become
    ``TYPE(source, target)``, with internal node ids resolved to names.
    """
    gs = _get_graph_store(cli_ctx)
    nodes = gs.get_nodes(limit=sys.maxsize)
    relationships = gs.get_relationships(limit=sys.maxsize)
    names: Dict[Any, Any] = {}
    facts: List[str] = []
    for node in nodes:
        props = node.get("properties") or {}
        name = props.get("name") or props.get("id") or node.get("id")
        names[node.get("id")] = name
        for label in node.get("labels") or ["Entity"]:
            facts.append(f"{label}({name})")
    for rel in relationships:
        source = names.get(rel.get("start_node_id"), rel.get("start_node_id"))
        target = names.get(rel.get("end_node_id"), rel.get("end_node_id"))
        facts.append(f"{rel.get('type', 'RELATED_TO')}({source}, {target})")
    return facts


def _graph_store_as_context(cli_ctx: CLIContext) -> Dict[str, List[Dict[str, Any]]]:
    """Read the configured graph store into GraphReasoner's expected shape.

    GraphReasoner._prepare_graph_context() reads a plain
    ``{"entities": [...], "relationships": [...]}`` dict -- distinct from
    both the raw GraphStore rows and the ``Label(arg)`` fact strings
    ``_graph_store_facts()`` builds for the other engines.
    """
    gs = _get_graph_store(cli_ctx)
    nodes = gs.get_nodes(limit=sys.maxsize)
    relationships = gs.get_relationships(limit=sys.maxsize)
    names: Dict[Any, Any] = {}
    entities = []
    for node in nodes:
        props = node.get("properties") or {}
        name = props.get("name") or props.get("id") or node.get("id")
        names[node.get("id")] = name
        labels = node.get("labels") or ["Entity"]
        # Multiple labels are all real classifications (mirrors the one
        # fact-per-label convention _graph_store_facts() uses); joining them
        # keeps a node with e.g. ["Person", "Employee"] fully described
        # instead of silently dropping every label but the first.
        entities.append({
            "id": name, "name": name, "type": "/".join(labels), "properties": props,
        })
    rel_out = []
    for rel in relationships:
        source = names.get(rel.get("start_node_id"), rel.get("start_node_id"))
        target = names.get(rel.get("end_node_id"), rel.get("end_node_id"))
        rel_out.append({
            "source": source, "target": target,
            "type": rel.get("type", "RELATED_TO"),
            "properties": rel.get("properties") or {},
        })
    return {"entities": entities, "relationships": rel_out}


def _lowercase_datalog_args(fact_str: str) -> str:
    """Lowercase only a fact string's arguments, keeping the predicate's
    case untouched.

    DatalogReasoner reads a leading-uppercase *argument* as a variable
    (constants must be lowercase), but places no such constraint on the
    predicate. Lowercasing the whole string would still satisfy the
    constant check but would break matching against Datalog rules written
    against the graph's own label spelling, e.g. "Human(X) :- Person(X)."
    expects a "Person(...)" fact, not "person(...)".
    """
    match = re.match(r'^([a-zA-Z0-9_]+)\((.*)\)$', fact_str.strip())
    if not match:
        return fact_str
    predicate, args_str = match.groups()
    args = [arg.strip().lower() for arg in args_str.split(',')]
    return f"{predicate}({', '.join(args)})"


def _run_reasoning_with_status(
    cli_ctx: CLIContext, engine: str, local_json: bool, fn: Callable[[], Dict[str, Any]]
) -> Dict[str, Any]:
    """Run a reasoning computation, showing a spinner unless output is quiet/JSON.

    Checks _is_json() (global --json OR the command's own --json flag), not
    just cli_ctx.json_output -- otherwise `reason run --json` (the local
    flag, global --json omitted) would still print spinner status text to
    stdout ahead of the JSON payload, corrupting it for a parser.
    """
    if cli_ctx.quiet or _is_json(cli_ctx, local_json):
        return fn()
    with console.status(
        f"[{_DIM}]Running {engine} reasoning engine…[/{_DIM}]", spinner="dots"
    ):
        return fn()


# ─── Output helpers ──────────────────────────────────────────────────────────


def _jecho(data: Any) -> None:
    """Emit JSON to stdout."""
    click.echo(json.dumps(data, default=str))


def _ok(cli_ctx: CLIContext, text: str) -> None:
    if not cli_ctx.quiet:
        elapsed = time.perf_counter() - cli_ctx._start
        console.print(f"[{_SUCCESS}] ✓[/{_SUCCESS}] {text}  [{_DIM}]{elapsed:.1f}s[/{_DIM}]")


def _info(cli_ctx: CLIContext, text: str) -> None:
    if not cli_ctx.quiet:
        console.print(f"[{_DIM}] ·[/{_DIM}] {text}")


def _warn(cli_ctx: CLIContext, text: str) -> None:
    console.print(f"[{_WARN_STY}] ⚠[/{_WARN_STY}] {text}")


def _dry(cli_ctx: CLIContext, action: str, *, json_out: bool = False, **fields: Any) -> None:
    payload = {"dry_run": True, "action": action, **fields}
    if json_out or cli_ctx.json_output:
        _jecho(payload)
    elif not cli_ctx.quiet:
        console.print(f"[{_WARN_STY}] Dry run:[/{_WARN_STY}] would {action}: {fields}")


def _is_dry(cli_ctx: CLIContext, local_dry: bool) -> bool:
    return local_dry or cli_ctx.dry_run_global


def _is_json(cli_ctx: CLIContext, local_json: bool) -> bool:
    return local_json or cli_ctx.json_output


def _pprint(cli_ctx: CLIContext, data: Any) -> None:
    """Pretty-print a result to the terminal.

    Dicts and lists are rendered as syntax-highlighted JSON.
    Strings are printed as-is. Respects --quiet and --no-color.
    """
    if cli_ctx.quiet:
        return
    if isinstance(data, (dict, list)):
        console.print(
            Syntax(
                json.dumps(data, indent=2, default=str),
                "json",
                theme="monokai",
                word_wrap=True,
            )
        )
    else:
        console.print(str(data))


def _serialize_extract_result(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize_extract_result(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize_extract_result(v) for k, v in obj.items()}
    return obj


# ─── Data In ──────────────────────────────────────────────────────────────────


_INGEST_TYPES = [
    "file", "web", "parquet", "xml", "rest", "db", "duckdb", "elastic",
    "email", "feed", "gdrive", "huggingface", "mcp", "mongo", "repo",
    "snowflake", "stream",
]


_INGEST_FORMATS = [
    "pdf", "docx", "csv", "excel", "html", "json", "jsonl", "ndjson", "parquet", "xml", "rdf"
]


_GRAPH_STORE_ENV_BACKEND_HINTS = {
    "GRAPH_STORE_NEO4J_URI": "neo4j",
    "GRAPH_STORE_FALKORDB_HOST": "falkordb",
    "GRAPH_STORE_NEPTUNE_ENDPOINT": "neptune",
    "GRAPH_STORE_AGE_CONNECTION_STRING": "age",
}


def _configured_ingest_graph_backend(
    cli_ctx: CLIContext, store_override: Optional[str]
) -> Optional[str]:
    graph_db = dict(cli_ctx.config.to_dict().get("graph_db", {}))
    backend = store_override or cli_ctx.store_backend or graph_db.get("backend")
    if backend:
        return str(backend)

    env_backend = os.environ.get("GRAPH_STORE_DEFAULT_BACKEND")
    if env_backend:
        return env_backend

    for env_var, hinted_backend in _GRAPH_STORE_ENV_BACKEND_HINTS.items():
        if os.environ.get(env_var):
            return hinted_backend

    return None


def _json_default(obj) -> object:
    """JSON serialiser that converts NumPy scalars/arrays to native Python types.

    Also expands dataclasses (e.g. ``FileObject`` from ``ingest``) to plain
    dicts and decodes ``bytes`` as UTF-8 text where possible, so a domain
    object round-trips through ``--output`` as data instead of a repr string.
    Falls back to ``str()`` for everything else so the writer never crashes on
    unexpected types (e.g. ``datetime``).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()
    try:
        import numpy as np  # local import — only needed when result contains numpy
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return str(obj)


def _write_result_output(out_path: Path, result) -> None:
    """Serialize a structured CLI result (dict or list) for ``--output``.

    Domain commands like ``deduplicate`` and ``ontology align`` produce dicts
    and lists, not numeric matrices — routing them through the embeddings
    writer rejected their shapes and extensions (.csv is documented for
    deduplicate). JSON-family formats serialize anything; CSV serializes a
    list of dicts (or a single dict as one row).

    Accepted extensions: .json, .jsonl, .csv  (no-extension and .txt are
    rejected so the path reported to the caller always matches the file
    actually created, consistent with every other --output in the CLI).
    """
    import json as _json

    suffix = out_path.suffix.lower()

    # ── JSON ────────────────────────────────────────────────────────────────
    if suffix == ".json":
        with open(out_path, "w", encoding="utf-8") as fh:
            _json.dump(result, fh, indent=2, default=_json_default)
        return

    # ── JSON Lines ──────────────────────────────────────────────────────────
    # Every record must occupy exactly one line.  Wrap a bare dict in a list
    # so callers never need to know whether their result is singular or plural.
    if suffix == ".jsonl":
        items = result if isinstance(result, list) else [result]
        with open(out_path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(_json.dumps(item, default=_json_default) + "\n")
        return

    # ── CSV ─────────────────────────────────────────────────────────────────
    if suffix == ".csv":
        import pandas as pd

        rows = result if isinstance(result, list) else [result]
        if not rows:
            raise click.ClickException(
                "No results to write — output file not created."
            )
        # Normalise numpy scalars/arrays to Python natives so to_csv() does
        # not fall back to repr() strings for array-valued cells.
        def _normalise(row):
            if not isinstance(row, dict):
                return row
            out = {}
            for k, v in row.items():
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, np.generic):
                        v = v.item()
                except ImportError:
                    pass
                out[k] = v
            return out

        pd.DataFrame([_normalise(r) for r in rows]).to_csv(out_path, index=False)
        return

    # ── unsupported ─────────────────────────────────────────────────────────
    display = suffix if suffix else "(no extension)"
    raise click.ClickException(
        f"Unsupported output format '{display}'. Use .json, .jsonl, or .csv"
    )


def _tag_filter_value(filter_str: str) -> str:
    """Return the value part of a ``tag:<value>`` decision filter, lowercased.

    ``str.lstrip("tag:")`` strips any leading ``t``, ``a``, ``g`` or ``:``
    *characters*, not the prefix: ``tag:auth`` became ``uth`` and ``tag:git``
    became ``it``, so the filter silently matched the wrong text. Anything
    without the prefix is passed through unchanged. An empty value is rejected,
    since ``"" in category`` is true for every decision.
    """
    value = filter_str.removeprefix("tag:").lower()
    if not value:
        raise click.BadParameter(
            "tag: filter requires a non-empty value, e.g. tag:finance",
            param_hint="'--filter'",
        )
    return value


def _load_temporal_kg(cli_ctx: CLIContext) -> Dict[str, Any]:
    """Load the CLI's persisted graph in the ``{"entities", "relationships"}``
    shape the temporal engines consume."""
    if not _uses_memory_graph(cli_ctx):
        raise click.ClickException(
            "The temporal commands read the CLI's persisted memory-backend "
            "graph and are not wired for the "
            f"{_resolve_graph_backend(cli_ctx)!r} backend yet."
        )
    from ..kg.temporal_model import parse_temporal_value

    graph = _load_context_graph(cli_ctx).to_kg_dict()
    # The export keeps ``recorded_at`` inside properties/metadata, but the
    # temporal engines read temporal fields at the fact's top level. Lift it
    # (normalised to UTC) so a fact recorded without explicit validity dates
    # still has a history event.
    for fact in (*graph.get("entities", []), *graph.get("relationships", [])):
        if "recorded_at" in fact:
            continue
        recorded_at = (fact.get("properties") or {}).get("recorded_at") or (
            fact.get("metadata") or {}
        ).get("recorded_at")
        parsed = parse_temporal_value(recorded_at) if recorded_at is not None else None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            fact["recorded_at"] = parsed.isoformat()
    return graph


def _parse_allen_interval(spec: str) -> Any:
    """Parse a ``start/end`` interval spec into a ``TemporalInterval``.

    Values keep their exact precision instead of being normalised to a
    granularity: expanding the end to the last microsecond of its second (as
    ``normalize_interval`` does) makes an end that equals the next interval's
    start report ``overlaps`` where Allen algebra says ``meets``.
    """
    from ..kg.temporal_model import (
        TemporalBound,
        parse_temporal_bound,
        parse_temporal_value,
    )
    from ..kg.temporal_reasoning import TemporalInterval

    parts = spec.split("/", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise click.ClickException(
            f"Interval must be given as start/end, got: {spec!r}"
        )
    start = parse_temporal_value(parts[0].strip())
    if start is None:
        raise click.ClickException(f"Invalid interval start: {parts[0].strip()!r}")
    end: Any = parse_temporal_bound(parts[1].strip(), default=TemporalBound.OPEN)
    if end is not TemporalBound.OPEN:
        end = parse_temporal_value(end)
        if end is None:
            raise click.ClickException(f"Invalid interval end: {parts[1].strip()!r}")
    # Keep naive values comparable with aware ones inside relation().
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if isinstance(end, datetime) and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if isinstance(end, datetime) and end < start:
        raise click.ClickException(f"Interval end precedes its start: {spec!r}")
    return TemporalInterval(start=start, end=end)


# ─── Data Out ─────────────────────────────────────────────────────────────────


# Formats this command can build out of a graph-store dump. Three are left out
# because a dump of entities and relationships cannot feed them: OWL and SHACL
# are serialized from an ontology (`semantica ontology shacl` generates the
# shapes), and the distance matrix is computed from an Explorer session graph
# (POST /api/export/distance-enriched). Offering them advertised a failure, and
# offering OWL would advertise a document with no classes in it.
_EXPORT_FORMATS = [
    "turtle", "jsonld", "ntriples", "rdfxml",
    "parquet", "arrow", "csv", "json", "yaml",
    "graphml", "arangodb",
]


def _multi_file_destination_error(
    format_name: str, compress: bool
) -> click.ClickException:
    """The error for asking a multi-file format for a single destination."""
    if compress:
        return click.ClickException(
            f"--format {format_name} writes one file per collection, so "
            f"--compress has no single file to compress; use --output and leave "
            f"--compress off."
        )
    return click.ClickException(
        f"--format {format_name} writes one file per collection, so stdout "
        f"cannot carry it; pass --output to name where they go."
    )


# ─── Orchestration ────────────────────────────────────────────────────────────


# Must stay in step with PipelineTemplateManager.list_templates(). The CLI used to
# offer five names the manager has never registered, so every `pipeline init`
# reached "Template not found" once the wrapper called the real API. Pinned by
# tests/test_cli_commands.py::TestPipeline::test_init_templates_match_the_manager.
_PIPELINE_TEMPLATES = [
    "document_processing", "rag_pipeline", "kg_construction", "ontology_generation",
]


class PipelineFileError(click.ClickException):
    """A pipeline file that parses but is not a usable pipeline definition.

    ``errors`` keeps the per-entry messages so ``--json`` can report each
    offending entry instead of one flattened sentence.
    """

    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.errors = list(errors) if errors else [message]


def _pipeline_step_errors(payload: Dict[str, Any]) -> List[str]:
    """Return an error for every step ``PipelineBuilder`` would silently drop.

    ``PipelineBuilder.build_pipeline`` keeps a step only when both ``name`` and
    ``type`` are truthy (``pipeline_builder.py`` — ``if step_name and
    step_type``). Every other entry disappears before the validator sees it, so
    a partly malformed file used to be reported valid. Check the shape here,
    while the offending index is still known.
    """
    steps = payload.get("steps", [])
    if steps is None:
        steps = []
    if not isinstance(steps, list):
        return [f"'steps' must be a list, got {type(steps).__name__}"]
    errors: List[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(
                f"steps[{index}] is not a mapping (got {type(step).__name__})"
            )
            continue
        label = f"steps[{index}]"
        name = step.get("name")
        if isinstance(name, str) and name.strip():
            label += f" ('{name}')"
        for field_name in ("name", "type"):
            value = step.get(field_name)
            if not (isinstance(value, str) and value.strip()):
                errors.append(
                    f"{label} is missing a non-empty '{field_name}'"
                )
    return errors


def _pipeline_builder_error_messages(exc: BaseException) -> List[str]:
    """Split the builder's ``Pipeline validation failed: [...]`` message.

    ``PipelineBuilder.build_pipeline`` raises one ``ValidationError`` whose
    message embeds the list of validation errors as a Python literal, so a
    ``--json`` caller that only got that string would have to parse it back.
    """
    detail = str(exc)
    prefix = "Pipeline validation failed: "
    if detail.startswith(prefix):
        detail = detail[len(prefix):]
    if detail.startswith("[") and detail.endswith("]"):
        try:
            parsed = ast.literal_eval(detail)
        except (ValueError, SyntaxError):
            return [detail]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [detail]


def _load_pipeline_file(path: str) -> Any:
    """Load a pipeline config file into a ``Pipeline`` object.

    Accepts YAML, or JSON (which ``yaml.safe_load`` parses as a subset). Step
    handlers are process-local and are never restored from a file, so every step
    of a loaded pipeline has ``handler is None`` — see
    ``PipelineSerializer.deserialize_pipeline``.
    """
    try:
        from ..pipeline import PipelineSerializer
    except ImportError as exc:
        raise click.ClickException(f"Pipeline module not available: {exc}") from exc
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise click.ClickException(f"Could not parse {path} as YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineFileError(
            f"{path} is not a pipeline definition: expected a mapping with a "
            f"'steps' list, got {type(payload).__name__}."
        )
    step_errors = _pipeline_step_errors(payload)
    if step_errors:
        raise PipelineFileError(
            f"{path} is not a valid pipeline definition: " + "; ".join(step_errors),
            errors=step_errors,
        )
    return PipelineSerializer().deserialize_pipeline(payload)


_MIGRATE_SUPPORTED_BACKENDS = {"faiss", "sqlite", "pgvector"}


_MIGRATE_BATCH_SIZE = 500


def _migrate_backend_config(vs_cfg: Dict[str, Any], backend: str) -> Dict[str, Any]:
    """Resolve per-backend config out of the vector_store config section.

    Supports both a per-backend nested shape (``vector_store.faiss.dimension``)
    and the common flat single-backend shape (``vector_store.backend`` +
    sibling keys), since either can appear depending on how many backends a
    user has configured.
    """
    nested = vs_cfg.get(backend)
    if isinstance(nested, dict):
        return dict(nested)
    if vs_cfg.get("backend") == backend:
        return {k: v for k, v in vs_cfg.items() if k != "backend"}
    return {}


def _require_faiss_index_path(cfg: Dict[str, Any], role: str) -> str:
    """FAISS has no server to hold state between commands: a fresh FAISSStore
    starts empty and nothing outside the process persists it, so migration
    needs an explicit on-disk index to read from or write to."""
    index_path = cfg.get("index_path")
    if not index_path:
        raise click.ClickException(
            f"faiss as migration {role} requires 'index_path' in the vector_store "
            f"config (vector_store.faiss.index_path or vector_store.index_path "
            f"when faiss is the configured backend)."
        )
    return index_path


def _collect_backup_sources(
    cfg: Dict[str, Any],
    config_path: Optional[str],
    include: str,
    strip_config: bool,
) -> Tuple[List[Tuple[str, Path]], List[Dict[str, str]]]:
    """Return (arcname, local_path) pairs for local files to include in a backup.

    Also returns each configured store whose local path does not exist, so the
    caller can report it instead of writing a partial archive with no sign that
    a store was dropped (#2009).
    """
    sources: List[Tuple[str, Path]] = []
    skipped: List[Dict[str, str]] = []

    if not strip_config and include in ("config", "all"):
        candidates: List[str] = [c for c in [config_path, "semantica.yaml", "semantica.yml"] if c]
        for c in candidates:
            p = Path(c)
            if p.is_file():
                sources.append(("config/semantica.yaml", p))
                break

    if include in ("ontology", "all"):
        _ont: Any = cfg.get("ontology_path") or (cfg.get("ontology") or {}).get("path")
        ont_path: Optional[str] = str(_ont) if _ont else None
        if ont_path:
            p = Path(ont_path)
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        sources.append((f"ontology/{f.relative_to(p)}", f))
            elif p.is_file():
                sources.append((f"ontology/{p.name}", p))

    store_map = {"graph": "graph_db", "vector": "vector_store", "triplet": "triplet_store"}
    for store_type, cfg_key in store_map.items():
        if include not in (store_type, "all"):
            continue
        info = cfg.get(cfg_key)
        if not isinstance(info, dict):
            continue
        # Only file-addressable backends have a local path key
        _dp: Any = info.get("path") or info.get("data_path") or info.get("index_path")
        data_path: Optional[str] = str(_dp) if _dp else None
        if not data_path:
            # No local path key at all: a cloud/database backend, which
            # `backup info` reports as not file-copyable. Not a skip.
            continue
        p = Path(data_path)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    sources.append((f"{store_type}/{f.relative_to(p)}", f))
        elif p.is_file():
            sources.append((f"{store_type}/{p.name}", p))
        else:
            # Redacted: this is shown in a warning and written to the
            # manifest, even with --strip-config.
            skipped.append({
                "store": store_type,
                "backend": str(info.get("backend", "unknown")),
                "path": _redact(data_path),
            })

    return sources, skipped


def _redact(uri: str) -> str:
    """Redact passwords/tokens from a connection URI."""
    import re
    return re.sub(r"(:)[^@/]+(@)", r"\1••••••••\2", uri)


# ─── Services ─────────────────────────────────────────────────────────────────


def _pid_file(name: str) -> Path:
    return Path.home() / ".semantica" / f"{name}.pid"


def _write_pid(name: str, pid: int) -> None:
    pf = _pid_file(name)
    pf.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(pf.parent), 0o700)
    except OSError:
        pass
    # Use low-level open to avoid following symlinks (TOCTOU / symlink-clobber
    # attacks). O_NOFOLLOW is POSIX-only; fall back gracefully on Windows.
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # type: ignore[attr-defined]
    try:
        fd = os.open(str(pf), flags, 0o600)
    except OSError as exc:
        raise click.ClickException(f"Failed to write PID file {pf}: {exc}") from exc
    try:
        os.write(fd, str(pid).encode())
    finally:
        os.close(fd)


def _read_pid(name: str) -> Optional[int]:
    pf = _pid_file(name)
    if pf.exists():
        try:
            return int(pf.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return None


def _kill_service(name: str) -> bool:
    pid = _read_pid(name)
    if pid is None:
        return False
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
        _pid_file(name).unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        _pid_file(name).unlink(missing_ok=True)
        return False


def _service_status(name: str) -> str:
    pid = _read_pid(name)
    if pid is None:
        return "stopped"
    try:
        os.kill(pid, 0)
        return f"running (pid {pid})"
    except ProcessLookupError:
        _pid_file(name).unlink(missing_ok=True)
        return "stopped"
