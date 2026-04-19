from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    model: str = Field(validation_alias=AliasChoices("model", "OPENAI_MODEL"))
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("api_key", "OPENAI_API_KEY"),
    )
    base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("base_url", "OPENAI_BASE_URL"),
    )
    max_turns: int = Field(default=8, ge=1)
    parallel_tool_calls: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    timeout: float | None = Field(default=None, gt=0.0)
