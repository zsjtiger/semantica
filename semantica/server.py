"""Compatibility entry point for the Semantica HTTP server.

The application itself lives in :mod:`semantica.explorer.app`. This module keeps
the ``semantica.server:app`` import path and the ``semantica-server`` script
working.
"""

import os

from .explorer.app import app, create_app  # noqa: F401
from .utils.logging import setup_logging


def main():
    """Run the server.

    Binds to loopback by default; set SEMANTICA_HOST to expose beyond
    localhost (e.g. behind a reverse proxy that terminates auth/TLS).
    """
    import uvicorn

    setup_logging()
    host = os.environ.get("SEMANTICA_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8000)


if __name__ == "__main__":
    main()
