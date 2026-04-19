class AgentHarnessError(Exception):
    """Base exception for the harness."""


class ToolDefinitionError(AgentHarnessError):
    """Raised when a tool definition is invalid."""


class ToolRegistrationError(AgentHarnessError):
    """Raised when the tool registry cannot accept a tool."""


class ToolExecutionError(AgentHarnessError):
    """Raised when a tool fails during execution."""


class MaxTurnsExceededError(AgentHarnessError):
    """Raised when the agent exhausts its allowed tool loop turns."""


class ProviderError(AgentHarnessError):
    """Raised when the provider cannot complete a model request."""
