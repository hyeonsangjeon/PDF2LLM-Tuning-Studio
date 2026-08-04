"""Tiny, dependency-light SFT with completion-only masking (CPU-smoke capable).

``train_sft`` runs a short manual training loop so it works across transformers
versions without depending on the Trainer API. Loss is computed *only* on the
assistant completion tokens (the prompt is masked with ``-100``), which is the
correct objective for instruction tuning.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple


def format_chat(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """Return (prompt, completion) from a chat-style messages list.

    Everything up to the final assistant turn becomes the prompt; the final
    assistant content is the completion (the only part that carries loss).
    """
    sys_txt = ""
    turns = []
    completion = ""
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            sys_txt = content
        elif role == "user":
            turns.append(f"<|user|>\n{content}")
        elif role == "assistant":
            if i == len(messages) - 1:
                completion = content
            else:
                turns.append(f"<|assistant|>\n{content}")
    header = (f"<|system|>\n{sys_txt}\n" if sys_txt else "")
    prompt = header + "\n".join(turns) + "\n<|assistant|>\n"
    return prompt, completion


def _encode(tokenizer, prompt: str, completion: str, max_seq_len: int):
    import torch

    eos = tokenizer.eos_token_id
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids + [eos]
    ids = (p_ids + c_ids)[:max_seq_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_seq_len]
    # guarantee at least one supervised token
    if all(t == -100 for t in labels):
        labels[-1] = ids[-1]
    input_ids = torch.tensor([ids], dtype=torch.long)
    label_ids = torch.tensor([labels], dtype=torch.long)
    return input_ids, label_ids


def _load_rows(train_path: str) -> List[dict]:
    with open(train_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train_sft(
    train_path: str,
    model_id: str,
    out_dir: str,
    *,
    max_steps: int = 3,
    max_seq_len: int = 128,
    learning_rate: float = 5e-4,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict:
    """Run a short completion-only SFT loop and save the model. Returns metrics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    rows = _load_rows(train_path)
    examples = [format_chat(r["messages"]) for r in rows if r.get("messages")]
    if not examples:
        raise ValueError(f"no training examples in {train_path}")

    losses: List[float] = []
    step = 0
    while step < max_steps:
        random.shuffle(examples)
        for prompt, completion in examples:
            input_ids, labels = _encode(tokenizer, prompt, completion, max_seq_len)
            input_ids, labels = input_ids.to(device), labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(float(out.loss.item()))
            step += 1
            if step >= max_steps:
                break

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return {
        "model_id": model_id,
        "steps": step,
        "n_examples": len(examples),
        "loss_first": round(losses[0], 5) if losses else None,
        "loss_last": round(losses[-1], 5) if losses else None,
        "device": device,
        "out_dir": out_dir,
    }


def evaluate_sft(
    model_dir: str,
    eval_rows: List[Dict],
    *,
    max_new_tokens: int = 24,
    max_seq_len: int = 128,
) -> Dict:
    """Greedy-generate answers for a few eval rows; return generations only.

    Scoring is left to the caller (workflow scoring). This just proves the
    trained model produces text without error.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    gens = []
    for r in eval_rows:
        prompt, _ = format_chat(r["messages"])
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        gens.append({"qa_id": r.get("qa_id"), "generated": text.strip()})
    return {"n": len(gens), "generations": gens}
