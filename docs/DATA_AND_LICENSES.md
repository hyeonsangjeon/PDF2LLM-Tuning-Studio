# Data & Licenses

Authoritative provenance + license ledger for every third‑party or bundled
asset this repository uses or ships. It is the **single source** for asset
licensing; [`SECURITY.md`](../SECURITY.md) and
[`docs/TRUST_AND_DATA.md`](TRUST_AND_DATA.md) link here instead of repeating it.

> **Scope of the root [`LICENSE`](../LICENSE) (MIT).** MIT covers **this
> repository's own source code only**. It does **not** relicense the external
> datasets, model weights, or sample documents listed below, and it does **not**
> automatically cover the fine‑tuned model artifacts committed under
> `pdf_qa_extraction/quantization/artifacts/` (those inherit their base model
> and training‑data terms — see the tables). Do not assume MIT flows through to
> any row here.

## 1. Datasets (downloaded at runtime — never committed)

| Asset | Where | License | Redistribution note |
|---|---|---|---|
| **KorQuAD 1.0** (`KorQuAD/squad_kor_v1`) | Hugging Face Hub, fetched at run time by `quantization/data_korquad.py` | **CC BY‑ND 2.0 KR** (Attribution‑**NoDerivatives**) | The dataset is **not** copied into this repo. The **ND** clause governs redistributing *modified copies of the dataset*; it is a **known‑unknown** for redistributing models fine‑tuned on it — see §5. |

## 2. Base models (downloaded at runtime — never committed)

| Model | License | Access |
|---|---|---|
| `Qwen/Qwen3-8B` (default) | Apache‑2.0 | ungated |
| `Qwen/Qwen2.5-7B-Instruct` | Apache‑2.0 | ungated |
| `01-ai/Yi-1.5-9B-Chat` | Apache‑2.0 | ungated (automatic fallback) |
| `meta-llama/Llama-3.1-8B-Instruct` | **Llama 3.1 Community License** | gated; used **only** if an HF token is present, otherwise skipped |

Model weights are pulled from the Hugging Face Hub at run time. Pin a specific
revision (commit SHA) rather than a mutable tag when you need reproducibility.

## 3. Committed derived model artifacts

Located in `pdf_qa_extraction/quantization/artifacts/` (`A_bf16`,
`A_bf16_run`, `A_bf16_adapter`, …).

| Artifact | Derived from | Weight terms | Data terms |
|---|---|---|---|
| BF16 LoRA adapter / merged + INT4 weights | **Qwen3‑8B** (Apache‑2.0) fine‑tuned on **KorQuAD 1.0** | Model weights follow the **Apache‑2.0** base | Trained on KorQuAD (**CC BY‑ND 2.0 KR**) → redistribution question is a **known‑unknown**, see §5 |

The per‑artifact `README.md` files carry the Hugging Face model‑card metadata
(`license: apache-2.0`) inherited from the base.

## 4. Bundled sample assets (committed)

| File | Purpose | Provenance / License |
|---|---|---|
| `pdf_qa_extraction/data/fsi_data.pdf` | one‑click demo input ("International Finance") | **Provenance unrecorded** in git history (imported in the initial commit). **Known‑unknown** — confirm source + license before treating as freely redistributable (see §5). Used only as a *local demo input*. |
| `pdf_qa_extraction/data/samples/memoir_sample_ko.pdf`, `…_long.pdf` | persona demo inputs (Korean) | Demo documents; provenance to confirm (§5). |
| `pdf_qa_extraction/data/qa_pairs.jsonl`, `data/samples/*.jsonl` | generated Q&A outputs | Produced by this project's pipeline → **MIT** alongside the code. |
| `pdf_qa_extraction/webapp/static/index.html` | demo UI | Project‑authored; **no third‑party CDN, font, or script** is loaded (fully self‑contained). → **MIT**. |

## 5. Known unknowns (do not overstate)

These are **open items**, deliberately not resolved by assertion:

1. **`fsi_data.pdf` / `memoir_sample_ko*.pdf` provenance.** Original source and
   license are not recorded. Until confirmed, treat them as *demo‑only inputs*,
   not as redistributable datasets. If a source cannot be established, replace
   them with clearly‑licensed or synthetic equivalents before any packaged
   release that bundles them.
2. **KorQuAD ND clause vs. fine‑tuned artifacts.** Whether weights trained on a
   NoDerivatives dataset may be redistributed is unsettled and depends on
   jurisdiction/interpretation. This repo does **not** claim it is settled.
3. This ledger is **human‑maintained documentation**, not a machine‑verified
   SBOM or an attested release. No signing/attestation/SBOM gate is implied here
   (that is future supply‑chain work).

Related: runtime data handling, egress, and retention are documented in
[`docs/TRUST_AND_DATA.md`](TRUST_AND_DATA.md); vulnerability reporting in
[`SECURITY.md`](../SECURITY.md).
