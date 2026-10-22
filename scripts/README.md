# Scripts

Maintenance and manual verification tools. None of these run as part of the
pytest suite.

| Script | Purpose | Run from the repo root |
| --- | --- | --- |
| `docs_check.py` | Checks the docs site: `docs.json`, links, and API references. CI runs it on docs changes. | `python scripts/docs_check.py` |
| `poc_runner.py` | Proof-of-concept runs for the Explorer security fixes. Needs `httpx` and `fastapi`. | `python scripts/poc_runner.py` |
| `verify_backends.py` | Checks each vector store backend against a live service. Needs the services running. | `python scripts/verify_backends.py` |
| `verify_context_sync.py` | Manual walk-through of the context and vector store synchronisation. | `python scripts/verify_context_sync.py` |
| `verify_rich_cli.py` | Renders the CLI output for a visual check. | `python scripts/verify_rich_cli.py` |
