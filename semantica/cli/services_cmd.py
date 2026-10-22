"""``semantica services`` and the config, server, explorer and mcp groups."""

import json
import os
import sys
from typing import (Optional)
import click
from rich.table import Table
from . import _common


@_common.main.group(name="services", invoke_without_command=True)
@click.pass_context
def services(ctx: click.Context) -> None:
    """Service management commands (server, explorer, mcp — foundation placeholder).

    Subcommands will follow the spec layout::

        semantica services server  start|stop|status
        semantica services explorer start|stop|status
        semantica services mcp      start|stop|status
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@_common.main.group(name="config", invoke_without_command=True)
@click.pass_context
def config_group(ctx: click.Context) -> None:
    """Configuration command group."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@_common.main.group(invoke_without_command=True)
@click.pass_context
def server(ctx: click.Context) -> None:
    """Start/stop/status the REST API server (port 8000)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@server.command("start")
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--workers", default=1, type=int, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Enable hot reload.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.pass_obj
def server_start(cli_ctx: _common.CLIContext, port: int, workers: int, reload: bool, host: str) -> None:
    """Start the REST API server.

    \b
    Example:
      semantica server start --port 8000 --workers 4
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    _LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
    if host not in _LOOPBACK_HOSTS:
        _common.console.print(
            f"[{_common._WARN_STY}] ⚠[/{_common._WARN_STY}] Binding to [cyan]{host}[/cyan] exposes "
            "the server to the network. Set SEMANTICA_API_KEY before doing this "
            "in any reachable environment — without it, protected routes refuse "
            "all requests (503), and with SEMANTICA_ALLOW_ANONYMOUS=true they are "
            "wide open."
        )

    def _action() -> None:
        import subprocess as sp
        cmd = [
            sys.executable, "-m", "uvicorn", "semantica.server:app",
            "--host", host, "--port", str(port),
            "--workers", str(workers),
        ]
        if reload:
            cmd.append("--reload")
        proc = sp.Popen(cmd)
        _common._write_pid("server", proc.pid)
        _common._ok(cli_ctx, f"Server started on {host}:{port} (pid {proc.pid})")

    _common._run_with_error_handling(_action)


@server.command("stop")
@click.pass_obj
def server_stop(cli_ctx: _common.CLIContext) -> None:
    """Stop the REST API server."""
    cli_ctx = _common._require_ctx(cli_ctx)
    if _common._kill_service("server"):
        _common._ok(cli_ctx, "Server stopped.")
    else:
        _common.console.print(f"[{_common._WARN_STY}] ⚠[/{_common._WARN_STY}] Server is not running.")


@server.command("status")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def server_status(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Show REST API server status."""
    cli_ctx = _common._require_ctx(cli_ctx)
    status = _common._service_status("server")
    if _common._is_json(cli_ctx, local_json):
        _common._jecho({"service": "server", "status": status})
    else:
        _common.console.print(f"Server: {status}")


@_common.main.group(invoke_without_command=True)
@click.pass_context
def explorer(ctx: click.Context) -> None:
    """Start/stop/status the Knowledge Explorer UI (port 5173)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@explorer.command("start")
@click.option("--port", default=5173, type=click.IntRange(1, 65535), show_default=True)
@click.option("--api-url", default="http://localhost:8000", show_default=True)
@click.option("--graph", default=None, type=click.Path(exists=True),
              help="Optional graph JSON file to preload.")
@click.pass_obj
def explorer_start(cli_ctx: _common.CLIContext, port: int, api_url: str,
                   graph: Optional[str]) -> None:
    """Start the Knowledge Explorer dashboard.

    \b
    Example:
      semantica explorer start --port 5173 --api-url http://localhost:8000
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        import subprocess as sp
        cmd = [sys.executable, "-m", "semantica.explorer", "--port", str(port)]
        if graph:
            cmd += ["--graph", graph]
        env = os.environ.copy()
        env["SEMANTICA_API_URL"] = api_url
        proc = sp.Popen(cmd, env=env)
        _common._write_pid("explorer", proc.pid)
        _common._ok(cli_ctx, f"Explorer started on port {port} using API {api_url} (pid {proc.pid})")

    _common._run_with_error_handling(_action)


@explorer.command("stop")
@click.pass_obj
def explorer_stop(cli_ctx: _common.CLIContext) -> None:
    """Stop the Knowledge Explorer."""
    cli_ctx = _common._require_ctx(cli_ctx)
    if _common._kill_service("explorer"):
        _common._ok(cli_ctx, "Explorer stopped.")
    else:
        _common.console.print(f"[{_common._WARN_STY}] ⚠[/{_common._WARN_STY}] Explorer is not running.")


@explorer.command("status")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def explorer_status(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Show Explorer status."""
    cli_ctx = _common._require_ctx(cli_ctx)
    status = _common._service_status("explorer")
    if _common._is_json(cli_ctx, local_json):
        _common._jecho({"service": "explorer", "status": status})
    else:
        _common.console.print(f"Explorer: {status}")


@explorer.command("open")
@click.option("--port", default=5173, type=click.IntRange(1, 65535), show_default=True)
@click.pass_obj
def explorer_open(cli_ctx: _common.CLIContext, port: int) -> None:
    """Open the Explorer in the default browser."""
    cli_ctx = _common._require_ctx(cli_ctx)
    import webbrowser
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    _common._ok(cli_ctx, f"Opened {url}")


@_common.main.group(invoke_without_command=True)
@click.pass_context
def mcp(ctx: click.Context) -> None:
    """Start/stop the MCP server and call tools directly."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mcp.command("start")
@click.option("--transport", type=click.Choice(["stdio", "http"]),
              default="stdio", show_default=True)
@click.option("--port", default=3000, type=int, show_default=True)
@click.pass_obj
def mcp_start(cli_ctx: _common.CLIContext, transport: str, port: int) -> None:
    """Start the MCP server.

    \b
    Example:
      semantica mcp start --transport stdio
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        import subprocess as sp
        cmd = [sys.executable, "-m", "semantica_mcp.mcp.server"]
        if transport == "http":
            raise click.ClickException("HTTP transport is not supported (stdio only).")
        proc = sp.Popen(cmd)
        _common._write_pid("mcp", proc.pid)
        _common._ok(cli_ctx, f"MCP server started (pid {proc.pid})")

    _common._run_with_error_handling(_action)


@mcp.command("stop")
@click.pass_obj
def mcp_stop(cli_ctx: _common.CLIContext) -> None:
    """Stop the MCP server."""
    cli_ctx = _common._require_ctx(cli_ctx)
    if _common._kill_service("mcp"):
        _common._ok(cli_ctx, "MCP server stopped.")
    else:
        _common.console.print(f"[{_common._WARN_STY}] ⚠[/{_common._WARN_STY}] MCP server is not running.")


@mcp.command("status")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def mcp_status(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Show MCP server status."""
    cli_ctx = _common._require_ctx(cli_ctx)
    status = _common._service_status("mcp")
    if _common._is_json(cli_ctx, local_json):
        _common._jecho({"service": "mcp", "status": status})
    else:
        _common.console.print(f"MCP server: {status}")


@mcp.command("list-tools")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def mcp_list_tools(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """List available MCP tools."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        # Same catalog the server exposes via tools/list, so `list-tools`
        # and `mcp start` can't drift (issue #1355).
        from semantica_mcp.mcp.tools import TOOL_DEFINITIONS
        tools = [t["name"] for t in TOOL_DEFINITIONS]
        if _common._is_json(cli_ctx, local_json):
            _common._jecho({"tools": list(tools)})
        else:
            table = Table(title="[bold]MCP Tools[/bold]",
                          box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
            table.add_column("Tool", style=_common._KEY)
            for t in tools:
                table.add_row(str(t))
            _common.console.print(table)

    _common._run_with_error_handling(_action)


@mcp.command("call")
@click.argument("tool_name")
@click.option("--args", default="{}", help="JSON arguments string.")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def mcp_call(cli_ctx: _common.CLIContext, tool_name: str, args: str, local_json: bool) -> None:
    """Invoke an MCP tool directly from the terminal.

    \b
    Example:
      semantica mcp call extract_entities --args '{"text": "Alice works at Acme Corp."}'
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            tool_args = json.loads(args)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in --args: {exc}") from exc
        if not isinstance(tool_args, dict):
            raise click.ClickException("--args must be a JSON object")
        # Dispatch through the same server `mcp start` spawns; its session
        # module never defined MCPSession (issue #1355).
        from semantica_mcp.mcp.server import UnknownToolError, call_tool
        try:
            result = call_tool(tool_name, tool_args)
        except UnknownToolError as exc:
            raise click.ClickException(str(exc)) from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, (dict, list)) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


# ─── Wire services subgroups into the existing 'services' group ───────────────


services.add_command(server, name="server")


services.add_command(explorer, name="explorer")


services.add_command(mcp, name="mcp")
