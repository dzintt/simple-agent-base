import pytest

from simple_agent_base.errors import ToolDefinitionError, ToolRegistrationError
from simple_agent_base.tools import ToolRegistry, tool


@tool
async def sample_tool(name: str, count: int = 1) -> str:
    """Return a simple greeting."""
    return f"{name}:{count}"


@tool
def sync_sample_tool(name: str) -> str:
    """Return a greeting from a sync tool."""
    return f"sync:{name}"


def test_decorator_accepts_valid_async_function() -> None:
    registry = ToolRegistry([sample_tool])
    definition = registry.get("sample_tool")

    assert definition.name == "sample_tool"
    assert definition.description == "Return a simple greeting."
    assert definition.is_async is True


def test_decorator_accepts_valid_sync_function() -> None:
    registry = ToolRegistry([sync_sample_tool])
    definition = registry.get("sync_sample_tool")

    assert definition.name == "sync_sample_tool"
    assert definition.description == "Return a greeting from a sync tool."
    assert definition.is_async is False


def test_missing_annotations_are_rejected() -> None:
    with pytest.raises(ToolDefinitionError):

        @tool
        async def invalid_tool(name) -> str:
            return str(name)


def test_duplicate_tool_names_are_rejected() -> None:
    registry = ToolRegistry([sample_tool])

    with pytest.raises(ToolRegistrationError):
        registry.register(sample_tool)


def test_json_schema_matches_parameters() -> None:
    registry = ToolRegistry([sample_tool])
    tool_schema = registry.to_openai_tools()[0]
    parameters = tool_schema["parameters"]

    assert set(parameters["properties"]) == {"name", "count"}
    assert parameters["required"] == ["name"]
