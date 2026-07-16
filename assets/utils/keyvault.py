"""Azure Key Vault helper - the Azure equivalent of ``assets/utils/ssm.py``.

Mirrors the small get/put/list/delete surface of the AWS SSM Parameter Store
helper so the setup notebooks read the same on both clouds. Authentication uses
``DefaultAzureCredential`` (Azure CLI login locally, Managed Identity on Azure ML).

Example
-------
    kv = key_vault("https://my-vault.vault.azure.net/")
    kv.put_params("PDF2LLM-REGION", "koreacentral")
    kv.get_params("PDF2LLM-REGION")

Note: Key Vault secret names may only contain alphanumerics and dashes, so use
``PREFIX-REGION`` style keys (dashes), not slashes.
"""

from __future__ import annotations

import os
from typing import List


class key_vault:
    def __init__(self, vault_url: str | None = None):
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
        if not vault_url:
            raise ValueError(
                "vault_url is required (or set AZURE_KEY_VAULT_URL)."
            )
        self.client = SecretClient(
            vault_url=vault_url, credential=DefaultAzureCredential()
        )

    def put_params(self, key: str, value: str, enc: bool = False) -> str:
        """Create/update a secret. ``enc`` is accepted for API parity (secrets
        are always encrypted at rest in Key Vault)."""
        try:
            self.client.set_secret(key, str(value))
            return "Store success"
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Error storing '{key}': {exc}")
            return "Error"

    def get_params(self, key: str, enc: bool = False) -> str:
        return self.client.get_secret(key).value

    def get_all_params(self) -> List[str]:
        return [s.name for s in self.client.list_properties_of_secrets()]

    def delete_param(self, keys) -> None:
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self.client.begin_delete_secret(key).wait()
        print(f"  secrets: {keys} deleted successfully")
