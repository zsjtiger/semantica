"""Top-level commands: info, shell, doctor, init, watch, build, ingest, parse, split, normalize, extract."""

import json
import os
import sys
from pathlib import Path
from typing import (Any, Callable, Dict, List,
                    Optional, Sequence, Tuple)
import yaml
import click
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .. import __version__
from . import _common


@_common.main.command()
@click.pass_obj
def info(cli_ctx: _common.CLIContext):
    """Display information about Semantica."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        _common.console.print(
            Panel(
                Text.from_markup(
                    f"[{_common._BRAND}]Semantica Framework[/{_common._BRAND}]  v{__version__}\n"
                    f"[{_common._DIM}]Semantic Layer & Knowledge Engineering[/{_common._DIM}]"
                ),
                box=box.ROUNDED,
                padding=(0, 2),
                expand=False,
            )
        )

        table = Table(box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
        table.add_column("Component", style=_common._KEY, no_wrap=True)
        table.add_column("Status", style=_common._VAL)

        table.add_row("Core Orchestrator", "Active")
        table.add_row("Knowledge Graph Engine", "Active")
        table.add_row("Pipeline Execution", "Active")
        table.add_row("Vector Store Integration", "Active")
        table.add_row("CLI Config File", cli_ctx.config_path or "(none)")
        table.add_row("CLI Log Level", cli_ctx.log_level)

        _common.console.print(table)

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.pass_context
def shell(ctx: click.Context) -> None:
    """Interactive REPL — run subcommands without the 'semantica' prefix."""
    import shlex
    cli_ctx = _common._require_ctx(ctx.obj)
    try:
        import readline as _rl; del _rl  # side-effect: enables line editing on Unix
    except ImportError:
        # Optional dependency: readline may be unavailable on some platforms
        # (e.g., certain Windows/Python builds). Continue without line editing.
        pass

    if not cli_ctx.quiet:
        _common.console.print(
            Panel(
                Text.from_markup(
                    f"[{_common._BRAND}]Semantica Shell[/{_common._BRAND}]  [{_common._DIM}]v{__version__}[/{_common._DIM}]\n"
                    f"[{_common._DIM}]Type a subcommand (e.g. [bold]kg stats[/bold]), "
                    f"[bold]help[/bold], or [bold]exit[/bold] to quit.[/{_common._DIM}]"
                ),
                box=box.ROUNDED, padding=(0, 2), expand=False,
            )
        )

    while True:
        try:
            line = input("semantica> ").strip()
        except (EOFError, KeyboardInterrupt):
            if not cli_ctx.quiet:
                _common.console.print(f"\n[{_common._DIM}]Goodbye![/{_common._DIM}]")
            break
        if not line:
            continue
        if line in ("exit", "quit", "q", ":q"):
            if not cli_ctx.quiet:
                _common.console.print(f"[{_common._DIM}]Goodbye![/{_common._DIM}]")
            break
        if line in ("help", "?"):
            click.echo(ctx.find_root().get_help())
            continue
        try:
            args = shlex.split(line)
        except ValueError as exc:
            _common._warn(cli_ctx, f"Parse error: {exc}")
            continue
        cmd_name = args[0]
        cmd = _common.main.get_command(ctx, cmd_name)
        if cmd is None:
            _common._warn(cli_ctx, f"Unknown command: {cmd_name!r}")
            _common.console.print(f"  [{_common._DIM}]Type [bold]help[/bold] to list available commands.[/{_common._DIM}]")
            continue
        try:
            with cmd.make_context(cmd_name, args[1:], parent=ctx) as sub_ctx:
                cmd.invoke(sub_ctx)
        except click.exceptions.Exit:
            # Subcommands may request termination via Click's Exit; keep REPL alive.
            continue
        except SystemExit:
            # Some commands/libraries may raise SystemExit; do not exit the shell loop.
            continue
        except click.ClickException as exc:
            exc.show()
        except Exception as exc:
            _common._warn(cli_ctx, f"Error: {exc}")


@_common.main.command()
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def changelog(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Show release notes and check for a newer version."""
    import urllib.request
    import urllib.error

    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        api = "https://api.github.com/repos/semantica-agi/semantica/releases/latest"
        try:
            req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError:
            _common._warn(cli_ctx, "Could not reach GitHub — check your internet connection.")
            return

        tag: str = data.get("tag_name", "unknown")
        body: str = data.get("body", "").strip()
        html_url: str = data.get("html_url", "")
        latest = tag.lstrip("v")
        current = __version__

        if _common._is_json(cli_ctx, local_json):
            _common._jecho({"current": current, "latest": latest, "url": html_url, "notes": body})
            return

        up_to_date = latest == current
        status = (
            f"[{_common._SUCCESS}]You are on the latest version ({current})[/{_common._SUCCESS}]"
            if up_to_date
            else f"[{_common._WARN_STY}]Update available: {current} → {latest}[/{_common._WARN_STY}]\n"
                 f"[{_common._DIM}]pip install --upgrade semantica[/{_common._DIM}]"
        )
        _common.console.print(
            Panel(
                Text.from_markup(
                    f"[{_common._BRAND}]Semantica {tag}[/{_common._BRAND}]\n\n"
                    f"{body}\n\n"
                    f"{status}"
                ),
                title="[bold]Changelog[/bold]",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        if html_url and not up_to_date:
            _common.console.print(f"  [{_common._DIM}]Full notes: {html_url}[/{_common._DIM}]")

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.option("--json", "local_json", is_flag=True, default=False)
@click.option("--deep-embeddings", "deep_embeddings", is_flag=True, default=False,
              help="Also instantiate the local embedding backends and embed a probe "
                   "text (catches backends that import cleanly but cannot load).")
@click.pass_obj
def doctor(cli_ctx: _common.CLIContext, local_json: bool, deep_embeddings: bool) -> None:
    """Run a health check on all Semantica components and backends."""
    import importlib.metadata
    cli_ctx = _common._require_ctx(cli_ctx)

    Check = Tuple[str, str, str, Optional[str]]  # label, status, note, hint

    def _check(label: str, fn: "Callable[[], str]", hint: Optional[str] = None) -> Check:
        try:
            note = fn()
            return label, "ok", note, None
        except _common._DeepEmbeddingFailure as exc:
            # A deep-probe failure means the package IMPORTED fine: the pip
            # hint would be the wrong remediation for what is actually a
            # runtime/model-load problem (broken torch, failed model
            # download, missing shared libs).
            return label, "fail", str(exc), (
                "runtime/model-load failure — reinstalling the package usually "
                "does not help; check the warnings above (torch install, model "
                "download, disk space)"
            )
        except Exception as exc:
            return label, "fail", str(exc), hint

    def _action() -> None:
        checks: List[Check] = []

        # Python version
        pv = sys.version_info
        py_status, py_hint = _common._python_version_check(pv)
        checks.append(("Python", py_status,
                        f"{pv.major}.{pv.minor}.{pv.micro}", py_hint))

        # Semantica version
        checks.append(("semantica", "ok", __version__, None))

        # Rich version
        def _rich_ver() -> str:
            return importlib.metadata.version("rich")
        checks.append(_check("rich", _rich_ver))

        # Graph store reachability
        def _graph() -> str:
            backend = _common._resolve_graph_backend(cli_ctx)
            if backend == _common.MEMORY_GRAPH_BACKEND:
                # No server to reach; prove the graph file is readable instead,
                # which is the only way this backend can actually fail.
                # Keep this short: the Note column truncates at 80 columns.
                graph = _common._load_context_graph(cli_ctx)
                count = len(graph.get_nodes_by_label("decision"))
                return f"memory, {count} decision(s)"
            gs = _common._get_graph_store(cli_ctx)
            gs.ping() if hasattr(gs, "ping") else gs.connect()
            return f"{backend} reachable"
        checks.append(_check("Graph store", _graph, hint="run 'semantica store list' to see configured backends"))

        # Vector store
        def _vector() -> str:
            cfg = cli_ctx.config.to_dict()
            backend = cli_ctx.vector_store_backend or cfg.get("vector_store", {}).get("backend", "faiss")
            if backend == "faiss":
                import faiss  # noqa: F401
            return f"{backend} importable"
        checks.append(_check("Vector store", _vector, hint="pip install semantica[vectorstore-…]"))

        # Embedding backends (#994): `doctor` used to report all green while
        # every local embedding backend was non-functional — import success
        # says nothing about model loading. Default checks stay cheap
        # (import + version); --deep-embeddings (or
        # SEMANTICA_DOCTOR_DEEP_EMBEDDINGS=1) instantiates the backend through
        # TextEmbedder and embeds a probe, which is the only level that
        # catches a backend that imports cleanly but cannot actually load.
        deep = deep_embeddings or os.environ.get("SEMANTICA_DOCTOR_DEEP_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes", "on")

        def _embedding_backend(method: str) -> str:
            if method == "sentence_transformers":
                import sentence_transformers  # noqa: F401
                try:
                    ver = importlib.metadata.version("sentence-transformers")
                except Exception:
                    ver = getattr(sentence_transformers, "__version__", "installed")
                note = f"importable ({ver})"
            else:
                import fastembed  # noqa: F401
                try:
                    ver = importlib.metadata.version("fastembed")
                except Exception:
                    ver = getattr(fastembed, "__version__", "installed")
                note = f"importable ({ver})"
            if not deep:
                return note
            try:
                from ..embeddings import TextEmbedder
                embedder = TextEmbedder(method=method)
                if embedder.model is None and embedder.fastembed_model is None:
                    raise RuntimeError(
                        "model failed to load — the hash fallback is active "
                        "(see warnings above); embedding quality is degraded"
                    )
                probe = embedder.embed_text("semantica doctor embedding probe")
            except _common._DeepEmbeddingFailure:
                raise
            except Exception as exc:
                raise _common._DeepEmbeddingFailure(str(exc)) from exc
            return f"{note}; deep probe ok ({len(probe)}-dim)"

        checks.append(_check(
            "Embeddings (sentence-transformers)",
            lambda: _embedding_backend("sentence_transformers"),
            hint="pip install 'semantica[embeddings-local]'",
        ))
        checks.append(_check(
            "Embeddings (fastembed)",
            lambda: _embedding_backend("fastembed"),
            hint="pip install 'semantica[embeddings-local]'",
        ))

        # LLM provider keys
        for provider, var in [("OpenAI", "OPENAI_API_KEY"), ("Anthropic", "ANTHROPIC_API_KEY"),
                               ("Groq", "GROQ_API_KEY")]:
            val = os.environ.get(var, "")
            if val:
                checks.append((provider, "ok", f"{var} set ({val[:8]}…)", None))
            else:
                checks.append((provider, "warn", f"{var} not set", f"export {var}=…"))

        # Config file
        if cli_ctx.config_path:
            ok = Path(cli_ctx.config_path).is_file()
            checks.append(("Config file", "ok" if ok else "fail",
                            cli_ctx.config_path, None))
        else:
            checks.append(("Config file", "warn", "using defaults (no --config)", "run 'semantica init'"))

        checks.append(_check("Log directory", lambda: _common._check_log_directory(cli_ctx)))

        if _common._is_json(cli_ctx, local_json):
            _common._jecho([{"check": lbl, "status": st, "note": note, "hint": hint}
                    for lbl, st, note, hint in checks])
            return

        tbl = Table(box=_common._TABLE_BOX, show_edge=False, padding=(0, 2), expand=True)
        tbl.add_column("Check",  style=_common._KEY, no_wrap=True, min_width=34)
        tbl.add_column("Status", no_wrap=True, min_width=4)
        tbl.add_column("Note",   style=_common._DIM, min_width=15, ratio=2, overflow="fold")
        tbl.add_column("Hint",   style=_common._DIM, min_width=20, ratio=3, overflow="fold")

        icons = {"ok": f"[{_common._SUCCESS}] ✓[/{_common._SUCCESS}]",
                 "warn": f"[{_common._WARN_STY}] ⚠[/{_common._WARN_STY}]",
                 "fail": "[bold red] ✗[/bold red]"}

        errors = warns = 0
        for lbl, st, note, hint in checks:
            tbl.add_row(lbl, icons.get(st, st), note or "", hint or "")
            if st == "fail":
                errors += 1
            elif st == "warn":
                warns += 1

        _common.console.print()
        _common.console.print(tbl)
        _common.console.print()
        summary = (f"[{_common._SUCCESS}]All checks passed[/{_common._SUCCESS}]" if not errors and not warns
                   else f"[bold red]{errors} error(s)[/bold red]  [{_common._WARN_STY}]{warns} warning(s)[/{_common._WARN_STY}]")
        _common.console.print(f"  {summary}")
        _common.console.print()

    _common._run_with_error_handling(_action)


@_common.main.command(name="init")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing config.")
@click.pass_obj
def init_cmd(cli_ctx: _common.CLIContext, force: bool) -> None:
    """Interactive wizard to create ~/.semantica/config.yaml."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        config_dir = Path.home() / ".semantica"
        config_file = config_dir / "config.yaml"

        if config_file.exists() and not force:
            _common._warn(cli_ctx, f"Config already exists at {config_file}  (use --force to overwrite)")
            return

        if not cli_ctx.quiet:
            _common.console.print(
                Panel(
                    Text.from_markup(
                        f"[{_common._BRAND}]Semantica Setup Wizard[/{_common._BRAND}]\n"
                        f"[{_common._DIM}]Creates ~/.semantica/config.yaml  •  press Ctrl+C to abort[/{_common._DIM}]"
                    ),
                    box=box.ROUNDED, padding=(0, 2), expand=False,
                )
            )
            _common.console.print()

        graph_backend = click.prompt(
            "  Graph store backend",
            type=click.Choice(["memory", "neo4j", "falkordb", "neptune"], case_sensitive=False),
            default="memory",
        )
        vector_backend = click.prompt(
            "  Vector store backend",
            type=click.Choice(["faiss", "qdrant", "pinecone", "weaviate", "milvus"], case_sensitive=False),
            default="faiss",
        )
        llm_provider = click.prompt(
            "  LLM provider",
            type=click.Choice(["none", "openai", "anthropic", "groq", "ollama"], case_sensitive=False),
            default="none",
        )

        cfg: Dict[str, Any] = {
            "graph_db":     {"backend": graph_backend},
            "vector_store": {"backend": vector_backend},
        }

        if llm_provider != "none":
            env_var = f"{llm_provider.upper()}_API_KEY"
            existing = os.environ.get(env_var, "")
            prompt_str = f"  {env_var}" + (f" [{existing[:8]}…]" if existing else "")
            key = click.prompt(prompt_str, default=existing, hide_input=True, show_default=False)
            if key:
                cfg["llm"] = {"provider": llm_provider, "api_key_env": env_var}
                os.environ[env_var] = key

        config_dir.mkdir(parents=True, exist_ok=True)
        with config_file.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

        _common.console.print()
        _common._ok(cli_ctx, f"Config written to {config_file}")
        _common._info(cli_ctx, "Run 'semantica doctor' to verify your setup.")

    _common._run_with_error_handling(_action)


@_common.main.command(name="watch")
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--type", "ingestor_type", default=None, help="Force ingestor type.")
@click.option("--store", "store_override", default=None, help="Target graph backend.")
@click.option("--patterns", default="*.pdf,*.docx,*.txt,*.csv,*.json,*.jsonl,*.ndjson",
              show_default=True, help="Comma-separated glob patterns to match.")
@click.pass_obj
def watch_cmd(cli_ctx: _common.CLIContext, path: str, ingestor_type: Optional[str],
              store_override: Optional[str], patterns: str) -> None:
    """Watch a directory and auto-ingest new or changed files.

    \b
    Requires: pip install semantica[watch]
    Examples:
      semantica watch ./data/contracts/
      semantica watch ./reports/ --patterns "*.pdf,*.docx"
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileSystemEvent
    except ImportError:
        raise click.ClickException(
            "watchdog is required — install it with: pip install semantica[watch]"
        )

    cli_ctx = _common._require_ctx(cli_ctx)
    watch_path = Path(path).resolve()
    pat_list = [p.strip() for p in patterns.split(",")]

    class _Handler(FileSystemEventHandler):
        def on_created(self, event: "FileSystemEvent") -> None:
            self._handle(event)

        def on_modified(self, event: "FileSystemEvent") -> None:
            self._handle(event)

        def _handle(self, event: "FileSystemEvent") -> None:
            if event.is_directory:
                return
            p = Path(str(event.src_path))
            if not any(p.match(pat) for pat in pat_list):
                return
            _common._info(cli_ctx, f"Detected {p.name} — ingesting…")
            try:
                from ..ingest import ingest as _ingest
                kwargs: Dict[str, Any] = {}
                if ingestor_type:
                    kwargs["source_type"] = ingestor_type
                if store_override or cli_ctx.store_backend:
                    kwargs["store"] = store_override or cli_ctx.store_backend
                _ingest(str(p), **kwargs)
                _common._ok(cli_ctx, f"{p.name}")
            except Exception as exc:
                _common._warn(cli_ctx, f"{p.name}: {exc}")

    if not cli_ctx.quiet:
        _common.console.print(
            Panel(
                Text.from_markup(
                    f"[{_common._BRAND}]Watching[/{_common._BRAND}] {watch_path}\n"
                    f"[{_common._DIM}]Patterns: {patterns}  •  Ctrl+C to stop[/{_common._DIM}]"
                ),
                box=box.ROUNDED, padding=(0, 2), expand=False,
            )
        )

    observer = Observer()
    observer.schedule(_Handler(), str(watch_path), recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    if not cli_ctx.quiet:
        _common.console.print(f"\n[{_common._DIM}]Stopped watching.[/{_common._DIM}]")


@_common.main.command("build", hidden=True)
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
def build_alias(
    cli_ctx: _common.CLIContext,
    source: Sequence[str],
    command_config_path: Optional[str],
):
    """Backward-compatible alias for 'kg build'."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        _common._run_build_command(cli_ctx, source, command_config_path)

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.argument("source")
@click.option("--type", "ingestor_type", type=click.Choice(_common._INGEST_TYPES), default=None,
              help="Force ingestor type (auto-detected by default).")
@click.option("--format", "fmt", type=click.Choice(_common._INGEST_FORMATS), default=None,
              help="Format hint.")
@click.option("--recursive", is_flag=True, default=False, help="Recurse into directories.")
@click.option("--watch", is_flag=True, default=False, help="Re-ingest on file changes.")
@click.option("--batch-size", default=500, type=int, show_default=True)
@click.option("--store", "store_override", default=None,
              help="Target graph backend: neo4j falkordb age neptune. "
                   "Not yet implemented — ingest cannot persist to a graph "
                   "store, so this only determines whether the command "
                   "refuses to report false success; use --output instead.")
@click.option("--output", default=None, type=click.Path(),
              help="Write ingested content to a .json/.jsonl/.csv file "
                   "instead of the (unimplemented) graph store.")
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def ingest(
    cli_ctx: _common.CLIContext, source: str, ingestor_type: Optional[str], fmt: Optional[str],
    recursive: bool, watch: bool, batch_size: int, store_override: Optional[str],
    output: Optional[str], local_dry: bool, local_json: bool,
) -> None:
    """Load files, URLs, databases, or streams into the graph.

    \b
    Examples:
      semantica ingest ./reports/q1.pdf
      semantica ingest ./data/ --recursive --format csv
      semantica ingest https://feeds.example.com/news --type feed --watch
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "ingest", json_out=_common._is_json(cli_ctx, local_json),
                 source=source, type=ingestor_type, format=fmt)
            return
        graph_backend = _common._configured_ingest_graph_backend(cli_ctx, store_override)
        if graph_backend and graph_backend.lower() != "memory" and not output:
            raise click.ClickException(
                f"A graph backend is configured ({graph_backend}), but "
                "semantica ingest does not write to graph stores yet — no CLI "
                "command currently does (tracked in issues #1351, #1352). "
                "Pass --output <file>.json to save the ingested content "
                "instead, or build a GraphStore/GraphBuilder directly in "
                "Python."
            )
        # NOTE: --store/GRAPH_STORE_DEFAULT_BACKEND are read only to decide
        # whether to raise the error above — nothing downstream of this point
        # writes to a graph store, so neither is forwarded as an ingest kwarg.
        kwargs: Dict[str, Any] = {"batch_size": batch_size}
        if ingestor_type:
            kwargs["source_type"] = ingestor_type
        if fmt:
            kwargs["format"] = fmt
        if recursive:
            kwargs["recursive"] = True
        if watch:
            kwargs["watch"] = True
        try:
            from ..ingest import ingest as _ingest
            label = Path(source).name if Path(source).exists() else source
            if cli_ctx.quiet or cli_ctx.json_output:
                result = _ingest(source, **kwargs)
            else:
                with _common.console.status(
                    f"[{_common._DIM}]Ingesting {label}{'  (recursive)' if recursive else ''}…[/{_common._DIM}]",
                    spinner="dots",
                ):
                    result = _ingest(source, **kwargs)
        except ImportError as exc:
            raise click.ClickException(f"Ingest module not available: {exc}") from exc
        if output:
            _common._write_result_output(Path(output), result)
            _common._ok(cli_ctx, f"Wrote {output}")
        elif _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"status": "ok"})
        else:
            _common._ok(cli_ctx, f"Ingested: {source}")

    _common._run_with_error_handling(_action)


@_common.main.command("parse")
@click.argument("file", type=click.Path(exists=True))
@click.option("--parser",
              type=click.Choice(["pdf", "docx", "code", "csv", "email", "excel",
                                  "html", "image", "json", "pptx", "xml", "web"]),
              default=None, help="Parser override.")
@click.option("--format", "fmt", type=click.Choice(["json", "yaml", "table"]),
              default="json", show_default=True)
@click.pass_obj
def parse_cmd(cli_ctx: _common.CLIContext, file: str, parser: Optional[str], fmt: str) -> None:
    """Parse a document into structured content (stdout).

    \b
    Examples:
      semantica parse contract.pdf | jq '.entities'
      semantica parse source.py --parser code --format json > parsed.json
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        kwargs: Dict[str, Any] = {"file_path": file}
        if parser:
            kwargs["parser"] = parser
        try:
            from ..parse import parse_document
            if cli_ctx.quiet or cli_ctx.json_output or fmt == "json":
                result = parse_document(**kwargs)
            else:
                with _common.console.status(f"[{_common._DIM}]Parsing {Path(file).name}…[/{_common._DIM}]", spinner="dots"):
                    result = parse_document(**kwargs)
        except ImportError as exc:
            raise click.ClickException(f"Parse module not available: {exc}") from exc
        if fmt == "json" or _common._is_json(cli_ctx, False):
            click.echo(json.dumps(result, default=str))
        elif fmt == "yaml":
            click.echo(yaml.dump(result, default_flow_style=False))
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.argument("input_path")
@click.option("--strategy",
              type=click.Choice(["recursive", "semantic", "entity-aware", "relation-aware",
                                  "sliding-window", "structural", "table"]),
              default="recursive", show_default=True)
@click.option("--chunk-size", default=512, type=int, show_default=True)
@click.option("--overlap", default=64, type=int, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["json", "jsonl", "text"]),
              default="json", show_default=True)
@click.option("--output", default=None, type=click.Path())
@click.pass_obj
def split(
    cli_ctx: _common.CLIContext, input_path: str, strategy: str, chunk_size: int,
    overlap: int, fmt: str, output: Optional[str],
) -> None:
    """Chunk documents with configurable strategies.

    \b
    Examples:
      semantica split report.pdf --strategy semantic --chunk-size 256
      semantica split data.csv --strategy table --format jsonl > chunks.jsonl
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..split import split_recursive, get_split_method
            fn = get_split_method(strategy) if strategy != "recursive" else split_recursive
            chunks = fn(input_path, chunk_size=chunk_size, overlap=overlap)
        except ImportError as exc:
            raise click.ClickException(f"Split module not available: {exc}") from exc
        if fmt == "jsonl":
            lines = "\n".join(json.dumps(c, default=str) for c in (chunks or []))
        elif fmt == "json":
            lines = json.dumps(chunks, default=str)
        else:
            lines = "\n".join(str(c) for c in (chunks or []))
        if output:
            Path(output).write_text(lines, encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {output}")
        else:
            click.echo(lines)

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.argument("input_text")
@click.option("--mode", type=click.Choice(["text", "temporal", "entity", "all"]),
              default="all", show_default=True)
@click.option("--domain", type=click.Choice(["healthcare", "legal", "finance", "general"]),
              default="general", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def normalize(cli_ctx: _common.CLIContext, input_text: str, mode: str, domain: str,
              local_json: bool) -> None:
    """Normalize text and dates (deterministic, no LLM).

    \b
    Examples:
      semantica normalize "We met last Tuesday around noon." --mode temporal
      semantica normalize entities.json --mode entity --domain legal
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        src = Path(input_text)
        text = src.read_text(encoding="utf-8") if src.is_file() else input_text
        try:
            from ..normalize import normalize_text, normalize_date, normalize_entity
            if mode == "temporal":
                result = normalize_date(text, domain=domain)
            elif mode == "entity":
                result = normalize_entity(text, domain=domain)
            elif mode == "text":
                result = normalize_text(text)
            else:
                result = normalize_text(normalize_date(normalize_entity(text, domain=domain)))
        except ImportError as exc:
            raise click.ClickException(f"Normalize module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho({"result": result})
        else:
            _common.console.print(str(result))

    _common._run_with_error_handling(_action)


# ─── Processing ───────────────────────────────────────────────────────────────


@_common.main.command()
@click.argument("input_path")
@click.option("--mode",
              type=click.Choice(["ner", "relations", "triplets", "events", "coreference", "all"]),
              default="all", show_default=True)
@click.option("--method", type=click.Choice(["pattern", "ml", "llm"]),
              default="ml", show_default=True)
@click.option("--model", default=None, help="LLM model when using --method llm.")
@click.option("--confidence", default=0.5, type=float, show_default=True,
              help="Minimum confidence 0.0-1.0.")
@click.option("--temporal", is_flag=True, default=False, help="Also extract temporal bounds.")
@click.option("--format", "fmt",
              type=click.Choice(["json", "yaml", "table", "rdf"]),
              default="json", show_default=True)
@click.option("--output", default=None, type=click.Path())
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def extract(
    cli_ctx: _common.CLIContext, input_path: str, mode: str, method: str,
    model: Optional[str], confidence: float, temporal: bool,
    fmt: str, output: Optional[str], local_json: bool,
) -> None:
    """Run extraction (NER, relations, triplets, events) on text or files.

    \b
    Examples:
      semantica extract "Alice signed the contract with Acme Corp."
      semantica extract report.pdf --mode triplets --method llm --model claude-sonnet-4-6
      cat text.txt | semantica extract - --mode relations
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if input_path == "-":
            text = sys.stdin.read()
        else:
            p = Path(input_path)
            text = p.read_text(encoding="utf-8") if p.exists() else input_path
        try:
            from ..semantic_extract import (
                NERExtractor,
                RelationExtractor,
                TripletExtractor,
                EventDetector,
            )

            extractor_config: Dict[str, Any] = {"min_confidence": confidence}
            if model:
                extractor_config["llm_model"] = model

            def _run_single(
                target: str,
                *,
                entities: Optional[List[Any]] = None,
                relations: Optional[List[Any]] = None,
            ) -> Any:
                if target == "triplets":
                    extractor = TripletExtractor(
                        method=method, include_temporal=temporal, **extractor_config
                    )
                    return extractor.extract(text, entities=entities, relations=relations)
                if target == "relations":
                    if entities is None:
                        ner = NERExtractor(method=method, **extractor_config)
                        entities = ner.extract(text)
                    extractor = RelationExtractor(
                        method=method, confidence_threshold=confidence, **extractor_config
                    )
                    return extractor.extract(text, entities=entities)
                if target == "ner":
                    extractor = NERExtractor(method=method, **extractor_config)
                    return extractor.extract(text)
                if target == "events":
                    extractor = EventDetector(method=method, **extractor_config)
                    return extractor.extract(text)
                raise click.ClickException(
                    f"Extraction mode '{target}' is not yet wired to a runtime extractor."
                )

            def _run_extraction() -> Any:
                # 'all' is the default, so it must not be the one value that
                # raises. Run every mode that has a runtime extractor and nest
                # each result under its own key (#1789).
                if mode == "all":
                    # Reuse each stage's output instead of recomputing it: the
                    # triplet stage would otherwise run its own NER + relation
                    # pass on top of the ones already run here, tripling the
                    # inference cost of the default command.
                    entities = _run_single("ner")
                    relations = _run_single("relations", entities=entities)
                    triplets = _run_single(
                        "triplets", entities=entities, relations=relations
                    )
                    return {
                        "ner": entities,
                        "relations": relations,
                        "triplets": triplets,
                        "events": _run_single("events"),
                    }
                return _run_single(mode)

            if cli_ctx.quiet or cli_ctx.json_output:
                result = _run_extraction()
            else:
                with _common.console.status(
                    f"[{_common._DIM}]Running {mode} extraction ({method})…[/{_common._DIM}]",
                    spinner="dots",
                ):
                    result = _run_extraction()
        except ImportError as exc:
            raise click.ClickException(f"Extract module not available: {exc}") from exc
        serialized = _common._serialize_extract_result(result)
        json_out = _common._is_json(cli_ctx, local_json) or fmt == "json"
        if json_out:
            text_out = json.dumps(
                serialized if isinstance(serialized, dict) else {"result": serialized},
                default=str,
            )
        elif fmt == "yaml":
            text_out = yaml.dump(serialized, default_flow_style=False)
        else:
            text_out = str(serialized)
        if output:
            Path(output).write_text(text_out, encoding="utf-8")
            _common._ok(cli_ctx, f"Wrote {output}")
        else:
            click.echo(text_out)

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.option("--strategy", type=click.Choice(["blocking", "semantic", "hybrid"]),
              default="hybrid", show_default=True)
@click.option("--min-similarity", default=0.7, type=float, show_default=True)
@click.option("--action", "dedup_action",
              type=click.Choice(["detect", "merge", "report"]),
              default="detect", show_default=True)
@click.option("--sort-by",
              type=click.Choice(["similarity", "entity_count", "cluster_size"]),
              default="similarity")
@click.option("--output", default=None, type=click.Path())
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def deduplicate(
    cli_ctx: _common.CLIContext, strategy: str, min_similarity: float, dedup_action: str,
    sort_by: str, output: Optional[str], local_dry: bool, local_json: bool,
) -> None:
    """Detect and resolve duplicate entities in the knowledge graph.

    \b
    Examples:
      semantica deduplicate --strategy hybrid --min-similarity 0.8
      semantica deduplicate --action merge --dry-run
      semantica deduplicate --action report --output duplicates.csv
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    strategy_map = {
        "blocking": "blocking_v2",
        "semantic": "legacy",
        "hybrid": "hybrid_v2",
    }

    def _load_entities() -> List[Dict[str, Any]]:
        from ..graph_store import get_nodes
        from ..graph_store.config import graph_store_config

        graph_db = dict(cli_ctx.config.to_dict().get("graph_db", {}))
        backend = cli_ctx.store_backend or graph_db.pop("backend", None)
        previous_graph_config = graph_store_config.get_all()
        graph_store_config.update(graph_db)
        if backend:
            graph_store_config.set("default_backend", backend)
        try:
            return get_nodes(limit=sys.maxsize)
        finally:
            graph_store_config.update(previous_graph_config)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "deduplicate", json_out=_common._is_json(cli_ctx, local_json),
                 strategy=strategy, dedup_action=dedup_action)
            return
        try:
            from ..deduplication import detect_duplicates
            from ..deduplication.entity_merger import EntityMerger

            def _run_dedup() -> Any:
                entities = _load_entities()
                candidate_strategy = strategy_map.get(strategy, strategy)
                detector_sort_by = "similarity_score" if sort_by == "similarity" else "confidence"
                detection_kwargs: Dict[str, Any] = {
                    "candidate_strategy": candidate_strategy,
                    "sort_by": detector_sort_by,
                }
                if dedup_action == "detect":
                    return detect_duplicates(
                        entities,
                        method="group",
                        similarity_threshold=min_similarity,
                        **detection_kwargs,
                    )
                elif dedup_action == "merge":
                    merger = EntityMerger()
                    return merger.merge_duplicates(
                        entities,
                        threshold=min_similarity,
                        **detection_kwargs,
                    )
                else:  # report — pairwise pairs with similarity scores
                    pairs = detect_duplicates(
                        entities,
                        method="pairwise",
                        similarity_threshold=min_similarity,
                        **detection_kwargs,
                    )
                    return {
                        "total_entities": len(entities),
                        "duplicate_pairs": len(pairs),
                        "pairs": [
                            {
                                "entity_1": getattr(p, "entity1_id", None),
                                "entity_2": getattr(p, "entity2_id", None),
                                "similarity": getattr(p, "similarity_score", None),
                            }
                            for p in pairs
                        ],
                    }

            if cli_ctx.quiet or cli_ctx.json_output:
                result = _run_dedup()
            else:
                with _common.console.status(
                    f"[{_common._DIM}]Running deduplication ({strategy}, {dedup_action})…[/{_common._DIM}]",
                    spinner="dots",
                ):
                    result = _run_dedup()
        except ImportError as exc:
            raise click.ClickException(f"Deduplication module not available: {exc}") from exc
        if output:
            _common._write_result_output(Path(output), result)
            _common._ok(cli_ctx, f"Wrote {output}")
        elif _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, (dict, list)) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@_common.main.command()
@click.option("--format", "fmt", type=click.Choice(_common._EXPORT_FORMATS),
              default="json", show_default=True)
@click.option("--output", default=None, type=click.Path(),
              help="Write to file (stdout if omitted).")
@click.option("--with-provenance", is_flag=True, default=False,
              help="Embed W3C lineage metadata.")
@click.option("--filter", "filter_str", default=None, help="e.g. type:Person")
@click.option("--compress", is_flag=True, default=False, help="Gzip the output.")
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def export(
    cli_ctx: _common.CLIContext, fmt: str, output: Optional[str], with_provenance: bool,
    filter_str: Optional[str], compress: bool, local_dry: bool, local_json: bool,
) -> None:
    """Export the graph in 11 supported formats.

    \b
    arrow, csv and parquet write one file per collection, so --output names a
    base path there and these formats cannot go to stdout.

    \b
    Examples:
      semantica export --format turtle --output graph.ttl
      semantica export --format parquet --output graph.parquet
        writes graph_entities.parquet and graph_relationships.parquet
      semantica export --format csv --filter "type:Person" --output persons.csv
        writes persons_entities.csv and persons_relationships.csv
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "export", json_out=_common._is_json(cli_ctx, local_json),
                 format=fmt, output=output)
            return
        try:
            import tempfile

            from ..export import MULTI_FILE_FORMATS, get_export_method
            from ..graph_store import get_nodes, get_relationships
            from ..graph_store.config import graph_store_config

            fn = get_export_method("export", "knowledge_graph")
            if fn is None:
                raise click.ClickException("Export method not available: export/knowledge_graph")

            # Fail before the store is read. These routes write one file per
            # collection, which is neither one stream nor one compressed file,
            # and finding that out after a full export wastes the read.
            if fmt in MULTI_FILE_FORMATS and (compress or not output):
                raise _common._multi_file_destination_error(fmt, compress)

            if _common._uses_memory_graph(cli_ctx):
                # ContextGraph.to_kg_dict() already returns entities and
                # relationships in the shape assembled below.
                kg = _common._load_context_graph(cli_ctx).to_kg_dict()
                entities = kg.get("entities", [])
                relationships = kg.get("relationships", [])
            else:
                graph_db = dict(cli_ctx.config.to_dict().get("graph_db", {}))
                graph_db.pop("path", None)
                # Pop before choosing, or an override leaves the configured
                # backend in the mapping pushed into graph_store_config.
                configured = graph_db.pop("backend", None)
                backend = cli_ctx.store_backend or configured
                previous_graph_config = graph_store_config.get_all()
                graph_store_config.update(graph_db)
                if backend:
                    graph_store_config.set("default_backend", backend)
                try:
                    entities = get_nodes(limit=sys.maxsize)
                    relationships = get_relationships(limit=sys.maxsize)
                finally:
                    graph_store_config.update(previous_graph_config)

            knowledge_graph = {
                "entities": entities,
                "relationships": relationships,
                "nodes": entities,
                "edges": relationships,
            }

            kwargs: Dict[str, Any] = {"format": fmt}
            if with_provenance:
                kwargs["include_provenance"] = True
            if filter_str:
                kwargs["filter"] = filter_str

            temp_output = None
            target_output = output
            if target_output is None or compress:
                temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
                temp_output = temp_handle.name
                temp_handle.close()
                target_output = temp_output

            written = fn(knowledge_graph, target_output, **kwargs)
        except ImportError as exc:
            raise click.ClickException(f"Export module not available: {exc}") from exc

        # A route that decides its own output names the files it wrote: Arrow
        # and Parquet write one file per collection, so the path this command
        # was given is a base name and is never created. Report what is there
        # rather than the name that was asked for.
        produced = [Path(p) for p in written] if written else []
        try:
            if produced and (compress or not output):
                # A backstop: a route that starts writing several files is
                # caught here too, not only the two rejected above.
                raise _common._multi_file_destination_error(fmt, compress)
            if compress:
                import gzip

                assert temp_output is not None
                compressed = gzip.compress(Path(temp_output).read_bytes())
                if output:
                    Path(output).write_bytes(compressed)
                    _common._ok(cli_ctx, f"Wrote compressed {output}")
                else:
                    sys.stdout.buffer.write(compressed)
            elif output:
                if produced:
                    _common._ok(cli_ctx, f"Wrote {', '.join(str(p) for p in produced)}")
                elif written is None:
                    _common._ok(cli_ctx, f"Wrote {output}")
                else:
                    _common._warn(
                        cli_ctx,
                        "Nothing written: the store returned no entities or "
                        "relationships.",
                    )
            else:
                assert temp_output is not None
                try:
                    click.echo(Path(temp_output).read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    sys.stdout.buffer.write(Path(temp_output).read_bytes())
        finally:
            if temp_output:
                Path(temp_output).unlink(missing_ok=True)

    _common._run_with_error_handling(_action)


# ─── Shell completion ─────────────────────────────────────────────────────────


@_common.main.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish", "powershell"]))
def completion(shell: str) -> None:
    """Output shell completion script.

    \b
    Install:
      semantica completion bash        >> ~/.bashrc
      semantica completion zsh         >> ~/.zshrc
      semantica completion fish        > ~/.config/fish/completions/semantica.fish
      semantica completion powershell  >> $PROFILE
    """
    shell_map = {
        "bash": ("bash", "~/.bashrc"),
        "zsh": ("zsh", "~/.zshrc"),
        "fish": ("fish", "~/.config/fish/completions/semantica.fish"),
        "powershell": ("powershell", "$PROFILE"),
    }
    click_shell, install_path = shell_map[shell]
    env_var = "_SEMANTICA_COMPLETE"
    try:
        from click.shell_completion import ShellComplete
        complete = ShellComplete(_common.main, ctx_args={}, prog_name="semantica",
                                 complete_var=env_var)
        src = complete.source()
        click.echo(src, nl=False)
    except Exception:
        click.echo(
            f"# Add this to {install_path}:\n"
            f'eval "$({env_var}={click_shell}_source semantica)"'
        )
