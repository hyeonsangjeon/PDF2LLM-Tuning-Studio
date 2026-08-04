"""P1-3: minimal OpenAI-compatible client — one question against a served artifact.

Streams a single chat completion from any OpenAI-compatible endpoint (e.g. vLLM's
``api_server``) and prints the answer. No SDK dependency (httpx only).

    python quantization/serving/client.py \
        --base-url http://127.0.0.1:8000 \
        --model runs/<run_id>/model/sft \
        --question "광주는 어느 지역에 있는가?" \
        --context "광주는 대한민국 남서부의 광역시이다."
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def ask(base_url: str, model: str, question: str, *, context: Optional[str] = None,
        max_tokens: int = 256, api_key: str = "EMPTY", timeout: float = 120.0,
        stream: bool = True) -> str:
    import httpx

    content = (f"다음 문맥에서 질문에 답하세요.\n문맥: {context}\n질문: {question}"
               if context else question)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": stream}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if not stream:
        r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    out: List[str] = []
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line[5:].strip() if line.startswith("data:") else line.strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    out.append(piece)
                    sys.stdout.write(piece)
                    sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="quantization.serving.client",
                                 description="Ask one question against a served artifact (P1-3).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--context", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args(argv)
    ask(args.base_url, args.model, args.question, context=args.context,
        max_tokens=args.max_tokens, stream=not args.no_stream)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
