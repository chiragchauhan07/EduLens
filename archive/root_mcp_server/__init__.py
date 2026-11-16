"""
Top-level alias for backend.mcp_server.
"""
from backend.mcp_server import settings, create_mcp_server, create_http_app, mcp_server_instance

__all__ = [
    "settings",
    "create_mcp_server",
    "create_http_app",
    "mcp_server_instance",
]

