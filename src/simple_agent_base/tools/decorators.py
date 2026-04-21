from __future__ import annotations

from collections.abc import Callable

from .base import TOOL_DEFINITION_ATTR, TOOL_METADATA_ATTR, build_tool_definition, extract_description


def tool(
    func: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., object] | Callable[[Callable[..., object]], Callable[..., object]]:
    def decorator(inner: Callable[..., object]) -> Callable[..., object]:
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
