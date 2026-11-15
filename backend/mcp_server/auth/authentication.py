"""
authentication.py — Authentication and user-isolation engine for EduLens MCP Server.
"""

import contextvars
import os
import secrets
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timezone

from backend.mcp_server.config import settings


class AuthenticationError(Exception):
    """Raised when request lacks valid authentication credentials."""
    pass


class AuthorizationError(Exception):
    """Raised when an authenticated user attempts to access another user's resources."""
    pass


@dataclass
class UserContext:
    """Represents the authenticated identity of the MCP client."""
    user_id: str
    username: str
    client_type: str = "generic"
    is_authenticated: bool = True
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ContextVar tracking the current user for the active execution context / task
_current_user_context: contextvars.ContextVar[Optional[UserContext]] = contextvars.ContextVar(
    "current_user_context", default=None
)


class TokenManager:
    """
    Manages API keys, bearer tokens, and user associations for MCP connections.
    Supports ChatGPT, Claude, and local developer clients with zero data leakage.
    """

    def __init__(self):
        # In-memory registry of active tokens: token_string -> UserContext
        self._tokens: Dict[str, UserContext] = {}
        self._seed_default_tokens()

    def _seed_default_tokens(self) -> None:
        """Seed default tokens for development and testing."""
        # 1. Primary dev token
        self._tokens[settings.default_dev_token] = UserContext(
            user_id=settings.default_dev_user,
            username="Local Developer",
            client_type="developer",
        )
        # 2. Secondary test user for cross-user isolation tests
        self._tokens["edulens-test-token-student-002"] = UserContext(
            user_id="student_002",
            username="Secondary Test Student",
            client_type="test",
        )
        # 3. Environment variable override
        custom_key = os.getenv("EDULENS_API_KEY")
        if custom_key:
            self._tokens[custom_key] = UserContext(
                user_id=os.getenv("EDULENS_USER_ID", "student_production"),
                username="Production User",
                client_type="production",
            )

    def register_token(self, token: str, user_id: str, username: str, client_type: str = "generic") -> None:
        """Registers a token associated with a specific user."""
        self._tokens[token] = UserContext(
            user_id=user_id,
            username=username,
            client_type=client_type,
        )

    def generate_token_for_user(self, user_id: str, client_type: str = "chatgpt") -> str:
        """Generates a secure cryptographically random token for an EduLens user."""
        new_token = f"edulens_{client_type}_{secrets.token_urlsafe(24)}"
        self.register_token(new_token, user_id=user_id, username=f"User {user_id}", client_type=client_type)
        return new_token

    def validate_token(self, token: Optional[str]) -> UserContext:
        """
        Validates token string against registered credentials.
        Raises AuthenticationError if invalid or missing when auth is enforced.
        """
        if not token:
            if not settings.auth_enabled:
                # Local development fallback
                return UserContext(
                    user_id=settings.default_dev_user,
                    username="Dev User (Unauthenticated Mode)",
                    client_type="local_dev",
                    is_authenticated=False,
                )
            raise AuthenticationError(
                "Missing authentication token. Provide an 'Authorization: Bearer <token>' header, "
                "?token=<token> parameter, or 'auth_token' argument."
            )

        # Normalize token: strip 'Bearer ' prefix if present
        clean_token = token.strip()
        if clean_token.lower().startswith("bearer "):
            clean_token = clean_token[7:].strip()

        if clean_token in self._tokens:
            return self._tokens[clean_token]

        raise AuthenticationError("Invalid or expired authentication token.")


# Global singleton token manager
token_manager = TokenManager()


def set_current_user(user: UserContext):
    """Sets the active UserContext in the current async context."""
    return _current_user_context.set(user)


def get_current_user() -> UserContext:
    """
    Returns the UserContext for the current request.
    If none set, falls back to dev user or raises AuthenticationError based on settings.
    """
    user = _current_user_context.get()
    if user is not None:
        return user

    if not settings.auth_enabled:
        return UserContext(
            user_id=settings.default_dev_user,
            username="Dev User (Default)",
            client_type="local_dev",
            is_authenticated=False,
        )

    raise AuthenticationError("No authenticated user context found for the current operation.")


def authenticate_token(token_or_header: Optional[str]) -> UserContext:
    """Helper to authenticate a raw token or header string and set context."""
    user = token_manager.validate_token(token_or_header)
    set_current_user(user)
    return user

