"""Generation providers for the workflow.

Only the recorded-replay provider runs in the credential-free default path. Any
cloud/live provider MUST be gated by :func:`pdf_qa.policy.guard_provider_call`
before construction; see :class:`LiveOllamaProvider` which is local-only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .prompts import build_generation_prompt, prompt_sha256


class ProviderError(Exception):
    pass


@dataclass
class Generation:
    answer: str
    evidence: List[dict]
    category: str
    answerable: bool
    qa_id: str
    generation_mode: str


class RecordedReplayProvider:
    """Deterministic provider: replays recorded generations by prompt hash.

    No network, no credentials. Raises if a prompt was not recorded, so a replay
    run can never silently fall back to a live call.
    """

    name = "recorded-replay"
    generation_mode = "recorded_replay"

    def __init__(self, records: List[dict]):
        self._by_hash: Dict[str, dict] = {}
        self._by_qid: Dict[str, dict] = {}
        for r in records:
            h = (r.get("generation") or {}).get("prompt_sha256")
            if h:
                self._by_hash[h] = r
            self._by_qid[r["qa_id"]] = r

    @classmethod
    def from_jsonl(cls, path: str) -> "RecordedReplayProvider":
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        return cls(recs)

    def recorded_questions(self) -> List[dict]:
        return list(self._by_qid.values())

    def generate(self, question: str, document_text: str) -> Generation:
        prompt = build_generation_prompt(question, document_text)
        h = prompt_sha256(prompt)
        rec = self._by_hash.get(h)
        if rec is None:
            raise ProviderError(
                f"no recorded generation for prompt hash {h[:12]}… "
                "(replay is deterministic; rebuild the fixture if inputs changed)"
            )
        return Generation(
            answer=rec["answer"],
            evidence=rec.get("evidence", []),
            category=rec.get("category", "single_fact"),
            answerable=rec.get("answerable", True),
            qa_id=rec["qa_id"],
            generation_mode=self.generation_mode,
        )


class LiveOllamaProvider:
    """Local Ollama provider (no raw-content egress).

    Requires a running Ollama daemon. This is a thin optional path for
    ``make demo-live-ollama``; it is intentionally local-only so it is always
    egress-safe. It is not exercised by CI.
    """

    name = "ollama"
    generation_mode = "live"

    def __init__(self, model: str = "qwen2.5:0.5b", host: Optional[str] = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    def generate(self, question: str, document_text: str) -> Generation:  # pragma: no cover - needs daemon
        import urllib.request

        prompt = build_generation_prompt(question, document_text)
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(self.host.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read())
        return Generation(
            answer=out.get("response", "").strip(),
            evidence=[],  # live generations require downstream evidence grounding
            category="single_fact",
            answerable=True,
            qa_id="live-" + prompt_sha256(prompt)[:12],
            generation_mode=self.generation_mode,
        )
