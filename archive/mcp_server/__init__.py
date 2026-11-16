"""
EduLens Model Context Protocol (MCP) Integration Package.
"""

from backend.mcp_server.config import settings
from backend.mcp_server.server import create_mcp_server, create_http_app, mcp_server_instance

__all__ = [
    "settings",
    "create_mcp_server",
    "create_http_app",
    "mcp_server_instance",
]

