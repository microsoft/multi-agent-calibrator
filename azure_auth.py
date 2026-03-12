"""Centralized Azure identity helpers for the calibrator tools."""

import os

from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    TokenCredential,
)


_SCOPE = "https://cognitiveservices.azure.com/.default"
_DEV_SENTINEL = "dev"


def get_azure_credential() -> TokenCredential:
    """Return an Azure credential based on the current environment.

    The AZURE_TOKEN_CREDENTIALS environment variable controls which credential
    is used at runtime:
    - Set to "dev" for local development (uses DefaultAzureCredential).
    - Set to any other value (or leave unset) in Azure environments to use
      ManagedIdentityCredential directly — no credential probing.
    """

    mode = (os.getenv("AZURE_TOKEN_CREDENTIALS") or "").lower()

    if mode == _DEV_SENTINEL:
        return DefaultAzureCredential(
            require_envvar=True,
            exclude_cli_credential=True,
            exclude_powershell_credential=True,
        )

    return ManagedIdentityCredential()


def get_azure_scope() -> str:
    """Expose the shared Azure OpenAI scope used throughout the repo."""

    return _SCOPE
