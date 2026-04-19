from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from simple_agent_base.errors import ToolDefinitionError
from simple_agent_base.types import ToolDefinition

TOOL_DEFINITION_ATTR = "__simple_agent_base_tool_definition__"
TOOL_METADATA_ATTR = "__simple_agent_base_tool_metadata__"


def extract_description(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    first_line = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return first_line or f"Run the {func.__name__} tool."


def build_arguments_model(func: Callable[..., Any]) -> type[BaseModel]:
    signature = inspect.signature(func)
    fields: dict[str, tuple[Any, Any]] = {}

    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            raise ToolDefinitionError(f"Tool '{func.__name__}' cannot use *args or **kwargs.")

        if parameter.annotation is inspect.Signature.empty:
            raise ToolDefinitionError(
                f"Tool '{func.__name__}' must declare a type annotation for '{parameter.name}'."
            )

        default = ... if parameter.default is inspect.Signature.empty else parameter.default
        fields[parameter.name] = (parameter.annotation, default)

    model_name = f"{func.__name__.title().replace('_', '')}Arguments"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def build_tool_definition(func: Callable[..., Any]) -> ToolDefinition:
    metadata = getattr(func, TOOL_METADATA_ATTR, {})
    description = metadata.get("description") or extract_description(func)
    name = metadata.get("name") or func.__name__
    arguments_model = build_arguments_model(func)
    parameters = arguments_model.model_json_schema()

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        func=func,
        arguments_model=arguments_model,
        is_async=inspect.iscoroutinefunction(func),
    )


def dump_tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, BaseModel):
        payload: Any = value.model_dump(mode="json")
    else:
        payload = value

    return json.dumps(payload, ensure_ascii=False, default=str)
