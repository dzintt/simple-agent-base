from __future__ import annotations

import pytest
from pydantic import ValidationError

from simple_agent_base import AgentConfig


def test_agent_config_accepts_tool_timeout() -> None:
    assert AgentConfig(model="gpt-5", tool_timeout=1.5).tool_timeout == 1.5


def test_agent_config_allows_no_tool_timeout() -> None:
    assert AgentConfig(model="gpt-5", tool_timeout=None).tool_timeout is None


@pytest.mark.parametrize("tool_timeout", [0, -1])
def test_agent_config_rejects_non_positive_tool_timeout(tool_timeout: float) -> None:
    with pytest.raises(ValidationError):
        AgentConfig(model="gpt-5", tool_timeout=tool_timeout)
