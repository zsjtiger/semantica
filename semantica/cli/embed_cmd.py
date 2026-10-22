"""``semantica embed`` commands."""

from pathlib import Path
from typing import (Optional)
import click
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Generate, index, and search embeddings."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@embed.command("generate")
@click.argument("input_path")
@click.option("--model",
              type=click.Choice(["sentence-transformers", "fastembed", "openai", "bge"]),
              default="sentence-transformers", show_default=True)
@click.option("--store", "store_backend",
              type=click.Choice(["faiss", "pinecone", "weaviate", "qdrant", "milvus", "pgvector"]),
              default=None)
@click.option("--namespace", default=None)
@click.option("--output", default=None, type=click.Path())
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def embed_generate(cli_ctx: _common.CLIContext, input_path: str, model: str,
                   store_backend: Optional[str], namespace: Optional[str],
                   output: Optional[str], local_json: bool) -> None:
    """Embed entities or documents and optionally index them.

    \b
    Example:
      semantica embed generate ./entities.json --model sentence-transformers --output embeddings.parquet
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..embeddings import generate_embeddings
            _gen = lambda: generate_embeddings(
                input_path, model=model,
                store=store_backend or cli_ctx.vector_store_backend,
                namespace=namespace,
            )
            if cli_ctx.quiet or cli_ctx.json_output:
                result = _gen()
            else:
                with _common.console.status(
                    f"[{_common._DIM}]Generating embeddings ({model})…[/{_common._DIM}]",
                    spinner="dots",
                ):
                    result = _gen()
        except ImportError as exc:
            raise click.ClickException(f"Embeddings module not available: {exc}") from exc
        if output:
            output_path = Path(output)
            suffix = output_path.suffix.lower()
            try:
                import numpy as np
                import pandas as pd
                arr = np.asarray(result)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                if arr.ndim != 2:
                    raise click.ClickException(
                        f"embed generate --output expects a 1-D or 2-D array, "
                        f"got {arr.ndim}-D (shape {arr.shape})"
                    )
                rows = [list(row) for row in arr]
                if suffix == ".parquet":
                    # Schema: single 'embedding' column (list[float] per row).
                    # embed index detects vector columns via
                    # isinstance(df[c].iloc[0], (list, np.ndarray)).
                    df = pd.DataFrame({"embedding": rows})
                    df.to_parquet(output_path, index=False)
                elif suffix in (".json", ".jsonl"):
                    df = pd.DataFrame({"embedding": rows})
                    df.to_json(
                        output_path,
                        orient="records",
                        lines=(suffix == ".jsonl"),
                    )
                else:
                    raise click.ClickException(
                        f"Unsupported output format '{suffix}'. "
                        "Use .parquet, .json, or .jsonl"
                    )
            except ImportError as exc:
                raise click.ClickException(
                    f"Missing dependency for --output: {exc}. "
                    "Install pyarrow with: pip install pyarrow"
                ) from exc
            _common._ok(cli_ctx, f"Wrote {output}")
        elif _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"status": "ok"})
        else:
            _common._ok(cli_ctx, "Embeddings generated.")

    _common._run_with_error_handling(_action)


@embed.command("search")
@click.argument("query_text")
@click.option("--store",
              type=click.Choice(["faiss", "pinecone", "weaviate", "qdrant", "milvus", "pgvector"]),
              default=None)
@click.option("--top-k", default=10, type=int, show_default=True)
@click.option("--hybrid", is_flag=True, default=False, help="Dense + sparse hybrid search.")
@click.option("--namespace", default=None)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def embed_search(cli_ctx: _common.CLIContext, query_text: str, store: Optional[str],
                 top_k: int, hybrid: bool, namespace: Optional[str], local_json: bool) -> None:
    """Semantic similarity search over the vector store.

    \b
    Example:
      semantica embed search "CEO of Acme Corp" --store qdrant --top-k 5 --hybrid
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..embeddings.methods import embed_text
            from ..vector_store import search_vectors
            query_vector = embed_text(query_text)
            results = search_vectors(
                query_vector,
                k=top_k,
                hybrid=hybrid,
                store=store or cli_ctx.vector_store_backend,
                namespace=namespace,
            )
        except ImportError as exc:
            raise click.ClickException(f"Embeddings/vector store module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(results if isinstance(results, (dict, list)) else {"results": str(results)})
        else:
            _common._pprint(cli_ctx, results)

    _common._run_with_error_handling(_action)


@embed.command("index")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--store",
              type=click.Choice(["faiss", "pinecone", "weaviate", "qdrant", "milvus", "pgvector"]),
              default="faiss", show_default=True)
@click.option("--namespace", default=None)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def embed_index(cli_ctx: _common.CLIContext, file_path: str, store: str,
                namespace: Optional[str], local_json: bool) -> None:
    """Index a Parquet/JSON embeddings file into a vector store.

    \b
    Example:
      semantica embed index embeddings.parquet --store faiss --namespace production
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            import numpy as np
            import pandas as pd
            from ..vector_store import create_index
        except ImportError as exc:
            raise click.ClickException(f"Vector store module not available: {exc}") from exc

        fp = Path(file_path)
        suffix = fp.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(fp)
        elif suffix in (".json", ".jsonl"):
            df = pd.read_json(fp, lines=(suffix == ".jsonl"))
        else:
            raise click.ClickException(
                f"Unsupported embeddings file format '{suffix}'. Use .parquet, .json, or .jsonl"
            )

        vector_col = next(
            (c for c in df.columns if isinstance(df[c].iloc[0] if len(df) else None, (list, np.ndarray))),
            None,
        ) if len(df) else None
        if vector_col is None:
            raise click.ClickException("No vector column found in embeddings file")

        id_col = next(
            (c for c in df.columns if c != vector_col and df[c].dtype == object), None
        )
        vectors = [np.array(v, dtype=np.float32) for v in df[vector_col]]
        ids = list(df[id_col].astype(str)) if id_col else None

        result = create_index(vectors, ids=ids, store=store, namespace=namespace)
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"status": "ok", "indexed": len(vectors)})
        else:
            _common._ok(cli_ctx, f"Indexed {len(vectors)} vectors from {file_path} into {store}")

    _common._run_with_error_handling(_action)
