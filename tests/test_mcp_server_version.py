"""Regression tests for MCP server version reporting (issue #863).

Both public MCP version surfaces must derive from the same authoritative
package version rather than a hardcoded stale literal:
  1. MCP ``initialize`` → ``serverInfo.version``
  2. ``semantica://schema/info`` → ``version``

The authoritative source of truth is ``semantica.__version__``, which is
maintained in sync with ``pyproject.toml``'s static ``version`` field by
the release process.  We assert equality against that value rather than
duplicating the version-resolution logic here, so the tests remain valid
through future version bumps without modification.

The ``assertNotEqual(..., "0.4.0")`` canaries guard against regression to
the original stale literal that triggered issue #863.
"""

import json
import unittest

import semantica
from semantica_mcp.mcp import server as mcp_server
from semantica_mcp.mcp.resources.registry import handle_resource_read

_EXPECTED = semantica.__version__


def _server_info():
    """serverInfo as reported by the initialize response."""
    response = mcp_server.SemanticaMCPServer().dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    )
    return response["result"]["serverInfo"]


def _read_resource(uri):
    """Parsed JSON body of a resource read."""
    return json.loads(handle_resource_read(uri)["text"])


class TestMCPServerVersion(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # SERVER_INFO (used directly in the initialize response)
    # ------------------------------------------------------------------ #

    def test_server_info_version_matches_package(self):
        """SERVER_INFO['version'] must equal the authoritative package version."""
        self.assertEqual(_server_info()["version"], _EXPECTED)

    def test_server_info_version_is_not_stale_literal(self):
        """Guard: SERVER_INFO must not report the original hardcoded 0.4.0."""
        self.assertNotEqual(_server_info()["version"], "0.4.0")

    # ------------------------------------------------------------------ #
    # MCP initialize → serverInfo.version
    # ------------------------------------------------------------------ #

    def test_initialize_server_info_version_matches_package(self):
        """The MCP initialize response must report the authoritative package version."""
        response = mcp_server.SemanticaMCPServer().dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        self.assertIsNotNone(response)
        self.assertEqual(
            response["result"]["serverInfo"]["version"],
            _EXPECTED,
        )

    def test_initialize_server_info_version_is_not_stale_literal(self):
        """Guard: initialize must not report the original hardcoded 0.4.0."""
        response = mcp_server.SemanticaMCPServer().dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        self.assertNotEqual(response["result"]["serverInfo"]["version"], "0.4.0")

    # ------------------------------------------------------------------ #
    # semantica://schema/info → version
    # ------------------------------------------------------------------ #

    def test_schema_info_resource_version_matches_package(self):
        """The semantica://schema/info resource must report the authoritative package version."""
        resource = _read_resource("semantica://schema/info")
        self.assertEqual(resource["version"], _EXPECTED)

    def test_schema_info_resource_version_is_not_stale_literal(self):
        """Guard: schema/info must not report the original hardcoded 0.4.0."""
        resource = _read_resource("semantica://schema/info")
        self.assertNotEqual(resource["version"], "0.4.0")

    # ------------------------------------------------------------------ #
    # Both surfaces must agree
    # ------------------------------------------------------------------ #

    def test_both_version_surfaces_are_identical(self):
        """SERVER_INFO and schema/info must report the exact same version string,
        confirming both surfaces derive from a single authoritative value."""
        init_response = mcp_server.SemanticaMCPServer().dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        schema_resource = _read_resource("semantica://schema/info")
        self.assertEqual(
            init_response["result"]["serverInfo"]["version"],
            schema_resource["version"],
        )


if __name__ == "__main__":
    unittest.main()
