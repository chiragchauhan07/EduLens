"""
config.py — Configuration settings for EduLens MCP Server.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class MCPSettings(BaseModel):
    """Configuration settings for EduLens Model Context Protocol (MCP) Server."""

    # Server Identity
    server_name: str = Field(
        default=os.getenv("EDULENS_MCP_NAME", "edulens-mcp-server"),
        description="Name of the MCP server announced to clients",
    )
    server_version: str = Field(
        default=os.getenv("EDULENS_MCP_VERSION", "1.0.0"),
        description="Semantic version of EduLens MCP integration",
    )
    server_description: str = Field(
        default=(
            "EduLens Learning Intelligence MCP Server. Receives student-AI learning conversations, "
            "evaluates demonstrated conceptual understanding, diagnoses confusion, tracks actual quiz performance, "
            "and generates structured learning intelligence."
        ),
        description="Server capability description exposed in client discovery",
    )

    # Storage Configuration
    storage_dir: Path = Field(
        default=Path(os.getenv("EDULENS_STORAGE_DIR", str(ROOT_DIR / "data" / "storage"))),
        description="Local directory for persistent storage (SQLite and raw sessions)",
    )
    database_filename: str = Field(
        default=os.getenv("EDULENS_DB_NAME", "edulens.db"),
        description="SQLite database file name",
    )

    # Networking & Transport
    host: str = Field(
        default=os.getenv("EDULENS_MCP_HOST", "0.0.0.0"),
        description="Host address to bind for HTTP/SSE transport",
    )
    port: int = Field(
        default=int(os.getenv("EDULENS_MCP_PORT", "8001")),
        description="Port for HTTP/SSE transport",
    )
    transport: str = Field(
        default=os.getenv("EDULENS_MCP_TRANSPORT", "sse"),
        description="Default transport: 'sse' (Server-Sent Events), 'streamable' (Streamable HTTP), or 'stdio'",
    )

    # Authentication & Security
    auth_enabled: bool = Field(
        default=os.getenv("EDULENS_AUTH_ENABLED", "true").lower() in ("1", "true", "yes"),
        description="Whether to enforce API key / Bearer token authentication",
    )
    default_dev_user: str = Field(
        default=os.getenv("EDULENS_DEV_USER", "student_001"),
        description="Default student ID used in local development mode when auth is disabled",
    )
    default_dev_token: str = Field(
        default=os.getenv("EDULENS_DEV_TOKEN", "edulens-dev-token-student-001"),
        description="Pre-configured test token for local development and automated testing",
    )

    # Reliability & Limits
    max_payload_bytes: int = Field(
        default=int(os.getenv("EDULENS_MAX_PAYLOAD_BYTES", str(5 * 1024 * 1024))),  # 5MB
        description="Maximum accepted request payload size in bytes",
    )
    max_messages_per_session: int = Field(
        default=int(os.getenv("EDULENS_MAX_MESSAGES", "200")),
        description="Maximum turns/messages accepted in a single conversation",
    )

    @property
    def database_path(self) -> Path:
        """Returns the full path to the SQLite database file."""
        return self.storage_dir / self.database_filename


# Global singleton settings instance
settings = MCPSettings()

# Ensure storage directory exists
settings.storage_dir.mkdir(parents=True, exist_ok=True)

