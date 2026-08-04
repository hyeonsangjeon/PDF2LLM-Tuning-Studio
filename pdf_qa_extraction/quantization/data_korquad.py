"""KorQuAD data preparation for the quantization track (A/B/C shared).

Loads ``KorQuAD/squad_kor_v1`` at runtime (never committed) and converts it to a
generative instruction format. The same held-out val slice + seed are used by all
three methods so their EM/F1/perplexity numbers are directly comparable.

CLI:
    python -m quantization.data_korquad --smoke      # tiny subset, prints 1 example
    python -m quantization.data_korquad --stats      # full split sizes
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "config.yaml")


def load_config(path: str = DEFAULT_CONFIG, force_mode: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml and, when ``compute.mode == 'cpu'`` (or ``force_mode``),
    fold the ``compute.smoke`` overrides into the top-level ``data``/``train``/``eval``
    sections so the rest of the pipeline reads a single flat config."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mode = force_mode or cfg.get("compute", {}).get("mode", "gpu")
    cfg["compute"]["mode"] = mode
    if mode == "cpu":
        smoke = cfg.get("compute", {}).get("smoke", {}) or {}
        if smoke.get("use_smoke_model"):
            cfg["base_model"]["selected"] = cfg["base_model"]["smoke"]
        for key in ("train_subset", "eval_size", "max_seq_len"):
            if key in smoke:
                cfg["data"][key] = smoke[key]
        for key in ("backend", "precision", "max_steps", "per_device_batch_size", "grad_accum"):
            if key in smoke:
                cfg["train"][key] = smoke[key]
        if "ppl_samples" in smoke:
            cfg["eval"]["ppl_samples"] = smoke["ppl_samples"]
    return cfg


def build_prompt(context: str, question: str, template: str) -> str:
    """Render the generative instruction prompt (ends right after ``[답]``)."""
    return template.format(context=context.strip(), question=question.strip())


@dataclass
class QAExample:
    id: str
    prompt: str          # instruction text ending with "[답]\n"
    answer: str          # first gold answer (SFT target)
    answers: List[str]   # all gold answers (for EM/F1 max-over-refs)
    context: str
    question: str

    def to_text(self, eos: str) -> str:
        """Full SFT sequence: prompt + gold answer + EOS."""
        return f"{self.prompt} {self.answer}{eos}"


def _to_example(row: Dict[str, Any], template: str) -> QAExample:
    answers = list(row["answers"]["text"]) if row.get("answers") else []
    answers = [a for a in answers if a and a.strip()] or [""]
    return QAExample(
        id=str(row.get("id", "")),
        prompt=build_prompt(row["context"], row["question"], template),
        answer=answers[0],
        answers=answers,
        context=row["context"],
        question=row["question"],
    )


def load_korquad(cfg: Dict[str, Any], split: str = "final_holdout") -> Dict[str, List[QAExample]]:
    """Return {'train': [...], 'eval': [...]} of QAExample.

    - eval = a *fixed* seed-shuffled DISJOINT slice of the official validation
      split (P1-1): ``final_holdout`` ([800:1800]) for the final comparison, or
      ``selection_dev`` ([0:800]) for tuning. Identical across A/B/C.
    - train = the official train split, optionally truncated to data.train_subset.
    """
    from datasets import load_dataset
    from . import splits as S

    dcfg = cfg["data"]
    template = dcfg["prompt_template"]
    seed = int(dcfg.get("seed", 42))
    S.assert_config_splits_disjoint(cfg)

    ds = load_dataset(dcfg["dataset"])
    train_split = ds["train"].shuffle(seed=seed)
    val_split = ds["validation"].shuffle(seed=seed)

    start, end = S.split_bounds(cfg, split)
    size = end - start
    eval_size = dcfg.get("eval_size")
    if eval_size:
        size = min(size, int(eval_size))
    val_split = val_split.select(range(start, min(start + size, len(val_split))))

    subset = dcfg.get("train_subset")
    if subset:
        train_split = train_split.select(range(min(int(subset), len(train_split))))

    train = [_to_example(r, template) for r in train_split]
    ev = [_to_example(r, template) for r in val_split]
    return {"train": train, "eval": ev}


def to_hf_text_dataset(examples: List[QAExample], eos: str):
    """Build a datasets.Dataset with a single 'text' column for trl SFTTrainer."""
    from datasets import Dataset

    return Dataset.from_dict({"text": [ex.to_text(eos) for ex in examples]})


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true", help="force compute.mode=cpu overrides")
    ap.add_argument("--stats", action="store_true", help="print full split sizes")
    args = ap.parse_args()

    cfg = load_config(args.config, force_mode="cpu" if args.smoke else None)
    if args.stats:
        from datasets import load_dataset

        ds = load_dataset(cfg["data"]["dataset"])
        print({k: len(v) for k, v in ds.items()})

    data = load_korquad(cfg)
    print(f"[data] base={cfg['base_model']['selected']} mode={cfg['compute']['mode']}")
    print(f"[data] train={len(data['train'])} eval={len(data['eval'])} "
          f"seed={cfg['data']['seed']} eval_size={cfg['data'].get('eval_size')}")
    ex = data["eval"][0]
    print("\n----- example prompt -----")
    print(ex.prompt)
    print("----- gold answers -----")
    print(ex.answers)
    print("----- SFT text (prompt+answer+EOS, first 400 chars) -----")
    print(ex.to_text("<|eos|>")[:400])


if __name__ == "__main__":
    _main()
