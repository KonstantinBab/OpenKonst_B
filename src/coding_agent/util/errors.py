"""Domain-specific exceptions."""

from __future__ import annotations


class CodingAgentError(Exception):
    """Base exception for the coding agent."""


class WorkspaceViolationError(CodingAgentError):
    """Raised when a path escapes the allowed workspace."""


class PolicyDeniedError(CodingAgentError):
    """Raised when an action is blocked by policy."""


class ApprovalRequiredError(CodingAgentError):
    """Raised when an action requires explicit approval."""


class ToolExecutionError(CodingAgentError):
    """Raised when a tool execution fails."""


class LLMResponseError(CodingAgentError):
    """Raised when a provider returns unusable content."""

