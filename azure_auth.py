"""Centralized Azure identity helpers for the calibrator tools."""

import os
from typing import Optional

from azure.identity import DefaultAzureCredential, TokenCredential


_DEV_SENTINEL = "dev"
_SCOPE = "https://cognitiveservices.azure.com/.default"


def get_azure_credential(env_var_name: str = "AZURE_TOKEN_CREDENTIALS") -> TokenCredential:
    """Return a DefaultAzureCredential configured for dev or prod.

    - AZURE_TOKEN_CREDENTIALS=dev (or unset) -> DefaultAzureCredential()
    - Any other value -> DefaultAzureCredential(require_envvar=True)

    This allows local development to keep using CLI / interactive auth, while
    production (or any non-dev setting) becomes deterministic and compliant
    with security guidance requiring AZURE_TOKEN_CREDENTIALS to select the
    deployed-service credential.
    """

    mode: Optional[str] = os.getenv(env_var_name)
    if mode is None or mode.lower() == _DEV_SENTINEL:
        return DefaultAzureCredential()

    return DefaultAzureCredential(require_envvar=True)


def get_azure_scope() -> str:
    """Expose the shared Azure OpenAI scope used throughout the repo."""

    return _SCOPE
