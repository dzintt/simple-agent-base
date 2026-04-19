from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import TOOL_DEFINITION_ATTR, TOOL_METADATA_ATTR, build_tool_definition, extract_description


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            inner,
            TOOL_METADATA_ATTR,
            {
                "name": name or inner.__name__,
                "description": description or extract_description(inner),
            },
        )
        setattr(inner, TOOL_DEFINITION_ATTR, build_tool_definition(inner))
        return inner

    if func is None:
        return decorator

    return decorator(func)
