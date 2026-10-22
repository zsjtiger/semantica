"""``semantica store`` commands."""

from pathlib import Path
from typing import (Any, Dict, List,
                    Optional)
import click
from rich.table import Table
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def store(ctx: click.Context) -> None:
    """Manage graph, vector, and triplet store backends."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@store.command("list")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def store_list(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """List all configured backends."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        cfg = cli_ctx.config.to_dict()
        store_cfg = cfg.get("store", {})
        backends = {k: v for k, v in store_cfg.items() if isinstance(v, dict)}
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(backends)
        else:
            table = Table(title="[bold]Configured Backends[/bold]",
                          box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
            table.add_column("Type", style=_common._KEY, no_wrap=True)
            table.add_column("Backend")
            table.add_column("URI/Host")
            for store_type, info in backends.items():
                table.add_row(
                    store_type,
                    str(info.get("backend", "-")),
                    str(info.get("uri", info.get("host", "-"))),
                )
            _common.console.print(table)

    _common._run_with_error_handling(_action)


@store.command("connect")
@click.option("--backend", required=True, help="Backend name.")
@click.option("--uri", default=None, help="Connection URI.")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def store_connect(cli_ctx: _common.CLIContext, backend: str, uri: Optional[str], local_json: bool) -> None:
    """Test a backend connection."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            # get_graph_store_method(task, method_name) is the method
            # registry, not a backend factory (#1354); build the store
            # through GraphStore, which resolves the backend by name.
            from ..graph_store import GraphStore
            cfg = dict(cli_ctx.config.to_dict().get("graph_db", {}))
            cfg.pop("backend", None)
            if uri:
                cfg["uri"] = uri
            # Instantiation only wires the backend; the probe below performs
            # the live connectivity check and raises if unreachable.
            store_instance = GraphStore(backend=backend, **cfg)
            for probe in ("health_check", "ping", "connect"):
                fn = getattr(store_instance, probe, None)
                if callable(fn):
                    fn()
                    break
            result = {"backend": backend, "connected": True}
        except ImportError as exc:
            result = {"backend": backend, "connected": False,
                      "error": f"Module not available: {exc}"}
        except Exception as exc:
            result = {"backend": backend, "connected": False, "error": str(exc)}
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        else:
            status = "[green]connected[/green]" if result.get("connected") else "[red]failed[/red]"
            _common.console.print(f"Backend {backend}: {status}")
            if not result.get("connected") and "error" in result:
                _common.console.print(f"  Error: {result['error']}")

    _common._run_with_error_handling(_action)


@store.command("stats")
@click.option("--backend", required=True)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def store_stats(cli_ctx: _common.CLIContext, backend: str, fmt: str, local_json: bool) -> None:
    """Show backend statistics."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..graph_store import run_analytics
            stats = run_analytics(backend=backend, config=cli_ctx.config.to_dict())
        except ImportError as exc:
            raise click.ClickException(f"Graph store module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(stats if isinstance(stats, dict) else {"stats": str(stats)})
        else:
            _common.console.print(stats)

    _common._run_with_error_handling(_action)


@store.command("migrate")
@click.option("--from", "from_backend", required=True)
@click.option("--to", "to_backend", required=True)
@click.option("--namespace", default=None)
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def store_migrate(cli_ctx: _common.CLIContext, from_backend: str, to_backend: str,
                  namespace: Optional[str], local_dry: bool, local_json: bool) -> None:
    """Migrate data between backends.

    Direct migration is only wired up between faiss, sqlite, and pgvector -
    these are the backends whose storage contract supports paging through
    every stored vector. Migrating to or from qdrant, pinecone, milvus, or
    weaviate still needs the export/reindex workaround below, since each of
    those needs its own enumeration design (Qdrant scroll, Pinecone list,
    etc.) that hasn't been built yet.

    \b
    Example:
      semantica store migrate --from faiss --to sqlite --namespace production --dry-run
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "migrate", from_backend=from_backend, to_backend=to_backend)
            return

        if from_backend not in _common._MIGRATE_SUPPORTED_BACKENDS or to_backend not in _common._MIGRATE_SUPPORTED_BACKENDS:
            raise click.ClickException(
                f"Direct backend migration ({from_backend} → {to_backend}) is only supported "
                f"between {', '.join(sorted(_common._MIGRATE_SUPPORTED_BACKENDS))}. To migrate involving "
                "another backend, export your data first:\n"
                "  semantica export --format parquet --output dump.parquet\n"
                f"  semantica embed index dump.parquet --store {to_backend}"
                + (f" --namespace {namespace}" if namespace else "")
            )

        from ..vector_store import VectorStore

        vs_cfg = cli_ctx.config.to_dict().get("vector_store", {}) or {}
        source_cfg = _common._migrate_backend_config(vs_cfg, from_backend)
        dest_cfg = _common._migrate_backend_config(vs_cfg, to_backend)

        source_index_path = None
        if from_backend == "faiss":
            source_index_path = _common._require_faiss_index_path(source_cfg, "source")
        dest_index_path = None
        if to_backend == "faiss":
            dest_index_path = _common._require_faiss_index_path(dest_cfg, "destination")

        source = VectorStore(backend=from_backend, config=source_cfg)
        if source_index_path:
            source._backend_store.load_index(source_index_path)

        source_dimension = getattr(source._backend_store, "dimension", None)
        if source_dimension and "dimension" not in dest_cfg:
            dest_cfg["dimension"] = source_dimension

        dest = VectorStore(backend=to_backend, config=dest_cfg)
        if dest_index_path and Path(dest_index_path).exists():
            dest._backend_store.load_index(dest_index_path)

        migrated = 0
        vectors_batch: List[Any] = []
        metadata_batch: List[Dict[str, Any]] = []
        ids_batch: List[str] = []

        def _flush() -> None:
            nonlocal migrated
            if not vectors_batch:
                return
            dest.store_vectors(list(vectors_batch), list(metadata_batch), ids=list(ids_batch))
            migrated += len(vectors_batch)
            vectors_batch.clear()
            metadata_batch.clear()
            ids_batch.clear()

        for item in source.iter_vectors(batch_size=_common._MIGRATE_BATCH_SIZE):
            meta = dict(item.get("metadata") or {})
            if namespace and "namespace" not in meta:
                meta["namespace"] = namespace
            vectors_batch.append(item["vector"])
            metadata_batch.append(meta)
            ids_batch.append(item["id"])
            if len(vectors_batch) >= _common._MIGRATE_BATCH_SIZE:
                _flush()
        _flush()

        if dest_index_path and migrated:
            dest._backend_store.save_index(dest_index_path)

        result = {"from": from_backend, "to": to_backend, "migrated": migrated}
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result)
        else:
            _common._ok(cli_ctx, f"Migrated {migrated} vectors from {from_backend} to {to_backend}")

    _common._run_with_error_handling(_action)


@store.command("flush")
@click.option("--namespace", default=None)
@click.option("--confirm", is_flag=True, default=False,
              help="Required safety confirmation.")
@click.pass_obj
def store_flush(cli_ctx: _common.CLIContext, namespace: Optional[str], confirm: bool) -> None:
    """Clear a namespace (requires --confirm)."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if not confirm:
            raise click.UsageError("--confirm is required to flush a namespace.")
        try:
            from ..vector_store import delete_vectors
            delete_vectors(namespace=namespace, config=cli_ctx.config.to_dict())
        except ImportError as exc:
            raise click.ClickException(f"Store module not available: {exc}") from exc
        _common._ok(cli_ctx, f"Flushed namespace: {namespace or 'default'}")

    _common._run_with_error_handling(_action)
