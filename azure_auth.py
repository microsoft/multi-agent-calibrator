"""Centralized Azure identity helpers for the calibrator tools."""

from azure.identity import DefaultAzureCredential, TokenCredential


_SCOPE = "https://cognitiveservices.azure.com/.default"


def get_azure_credential() -> TokenCredential:
    """Return a DefaultAzureCredential with require_envvar=True.

    The AZURE_TOKEN_CREDENTIALS environment variable controls which credential
    is used at runtime:
    - In Azure environments, set it to a deployed-service credential name
      (e.g. ManagedIdentityCredential, WorkloadIdentityCredential).
    - In local dev, set it to "dev" to exclude deployed-service credentials.

    require_envvar=True ensures deterministic credential selection and avoids
    probing development-time credentials in production.
    """

    return DefaultAzureCredential(require_envvar=True)


def get_azure_scope() -> str:
    """Expose the shared Azure OpenAI scope used throughout the repo."""

    return _SCOPE
