"""
Semantica MCP Server

Exposes Semantica's knowledge graph, decision intelligence, semantic extraction,
reasoning, and analytics capabilities as an MCP (Model Context Protocol) server
over stdio — compatible with Claude Desktop, Windsurf, Cline, Continue, VS Code,
Roo Code, and any other MCP-aware tool.

This package is a compatibility entry point. The implementation lives in
``semantica_mcp.mcp``; keeping ``python -m semantica.mcp_server`` and the
``semantica-mcp`` script working means existing client configurations need no
change.

Usage
-----
Configure in your tool's MCP settings:

    Claude Desktop / Windsurf / Cline / Continue / VS Code:
    {
        "mcpServers": {
            "semantica": {
                "command": "semantica-mcp"
            }
        }
    }

Or using python -m:
    {
        "mcpServers": {
            "semantica": {
                "command": "python",
                "args": ["-m", "semantica.mcp_server"]
            }
        }
    }

Run directly for testing:
    semantica-mcp
    # or
    python -m semantica.mcp_server

Environment variables:
    SEMANTICA_KG_PATH   — path to a persisted graph to load on start (optional)
    SEMANTICA_LOG_LEVEL — log level: DEBUG, INFO, WARNING (default: WARNING)
"""

from semantica_mcp.mcp.server import (
    TOOL_DEFINITIONS as TOOLS,
    UnknownToolError,
    call_tool,
    main,
)

__all__ = ["TOOLS", "UnknownToolError", "call_tool", "main"]
