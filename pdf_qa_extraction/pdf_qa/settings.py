"""Environment-variable ledger loader, validator and ``.env`` generator.

Every setting the pipeline/providers/web app read is declared once in
``settings.yaml`` (next to this module). This module turns that ledger into:

* :func:`load_settings` / :func:`iter_settings` -- structured access,
* :func:`render_dotenv_example` -- a grouped, commented ``.env`` template,
* :func:`validate_env` / :func:`provider_configured` -- "is provider X ready?"
  checks used by the web app and the ``--check`` CLI.

Point ``PDF2LLM_SETTINGS_FILE`` at your own copy to manage a custom ledger.

CLI::

    python -m pdf_qa.settings --list            # show the ledger
    python -m pdf_qa.settings --check azure      # required vars for a provider
    python -m pdf_qa.settings --write-env .env.example
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

DEFAULT_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.yaml")


def _settings_file() -> str:
    return os.environ.get("PDF2LLM_SETTINGS_FILE") or DEFAULT_SETTINGS_FILE


@dataclass(frozen=True)
class Setting:
    """One declared environment variable."""

    name: str
    group: str = "core"
    default: str = ""
    description: str = ""
    secret: bool = False
    required_for: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def is_set(self, env: Optional[Dict[str, str]] = None) -> bool:
        """True if this var (or any alias) has a non-empty value in ``env``."""
        env = os.environ if env is None else env
        for key in (self.name, *self.aliases):
            if (env.get(key) or "").strip():
                return True
        return False


def load_settings(path: Optional[str] = None) -> List[Setting]:
    """Read the ledger and return the settings in declaration order."""
    ledger_path = path or _settings_file()
    with open(ledger_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    raw = data.get("settings")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Settings ledger '{ledger_path}' has no 'settings' list.")
    out: List[Setting] = []
    for spec in raw:
        if not isinstance(spec, dict) or not spec.get("name"):
            raise ValueError(f"Malformed setting entry in '{ledger_path}': {spec!r}")
        out.append(
            Setting(
                name=str(spec["name"]),
                group=str(spec.get("group", "core")),
                default="" if spec.get("default") is None else str(spec.get("default")),
                description=str(spec.get("description", "")),
                secret=bool(spec.get("secret", False)),
                required_for=[str(x) for x in (spec.get("required_for") or [])],
                aliases=[str(x) for x in (spec.get("aliases") or [])],
            )
        )
    return out


def iter_settings(path: Optional[str] = None) -> List[Setting]:
    """Alias for :func:`load_settings` (kept for readable call sites)."""
    return load_settings(path)


def grouped_settings(path: Optional[str] = None) -> "Dict[str, List[Setting]]":
    """Return the settings grouped by their ``group``, preserving order."""
    groups: "Dict[str, List[Setting]]" = {}
    for setting in load_settings(path):
        groups.setdefault(setting.group, []).append(setting)
    return groups


def validate_env(
    provider: str, env: Optional[Dict[str, str]] = None, path: Optional[str] = None
) -> List[str]:
    """Return the names of vars **required** for ``provider`` that are unset."""
    provider = (provider or "").strip().lower()
    missing: List[str] = []
    for setting in load_settings(path):
        if provider in [p.lower() for p in setting.required_for] and not setting.is_set(env):
            missing.append(setting.name)
    return missing


def provider_configured(
    provider: str, env: Optional[Dict[str, str]] = None, path: Optional[str] = None
) -> bool:
    """True when every var required for ``provider`` is set."""
    return not validate_env(provider, env=env, path=path)


_GROUP_TITLES = {
    "core": "Core",
    "extraction": "Extraction / chart-context",
    "io": "Input / output (local + batch)",
    "provider.azure": "Provider: Azure AI Foundry (Entra ID first-class)",
    "provider.openai": "Provider: OpenAI",
    "provider.bedrock": "Provider: AWS Bedrock",
    "provider.ollama": "Provider: Ollama (local, no credentials)",
    "quality": "Quality control (validation + de-duplication)",
    "webapp": "Web app",
}


def render_dotenv_example(path: Optional[str] = None) -> str:
    """Render a grouped, commented ``.env`` template from the ledger."""
    lines = [
        "# .env — copy to `.env` and fill in. Generated from pdf_qa/settings.yaml",
        "# Regenerate: python -m pdf_qa.settings --write-env .env.example",
        "",
    ]
    for group, settings in grouped_settings(path).items():
        title = _GROUP_TITLES.get(group, group)
        lines.append(f"# ===== {title} =====")
        for s in settings:
            if s.description:
                lines.append(f"# {s.description}")
            if s.required_for:
                lines.append(f"#   required for: {', '.join(s.required_for)}")
            if s.aliases:
                lines.append(f"#   aliases: {', '.join(s.aliases)}")
            if s.secret:
                lines.append(f"# {s.name}=")  # never seed a secret value
            elif s.default:
                lines.append(f"{s.name}={s.default}")
            else:
                lines.append(f"# {s.name}=")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_dotenv_example(dest: str, path: Optional[str] = None) -> str:
    """Write the rendered template to ``dest`` and return its content."""
    content = render_dotenv_example(path)
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write(content)
    return content


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="PDF2LLM settings ledger tool.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print the ledger.")
    group.add_argument(
        "--check", metavar="PROVIDER", help="Check required vars for a provider."
    )
    group.add_argument(
        "--write-env",
        nargs="?",
        const=".env.example",
        metavar="PATH",
        help="Write a .env template (default: .env.example).",
    )
    args = parser.parse_args(argv)

    if args.list:
        for group_name, settings in grouped_settings().items():
            print(f"[{group_name}]")
            for s in settings:
                mark = " (set)" if s.is_set() else ""
                req = f" required_for={s.required_for}" if s.required_for else ""
                print(f"  {s.name}={s.default!r}{req}{mark}")
        return 0

    if args.check:
        missing = validate_env(args.check)
        if missing:
            print(f"[{args.check}] missing required env: {', '.join(missing)}")
            return 1
        print(f"[{args.check}] OK — all required env present.")
        return 0

    dest = args.write_env
    write_dotenv_example(dest)
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
