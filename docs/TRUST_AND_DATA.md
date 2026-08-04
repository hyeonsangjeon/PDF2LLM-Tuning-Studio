# Trust & Data Handling

The **single source** for this project's runtime trust boundary, network egress,
upload handling, PII gate, retention, and threat model. Asset licensing lives in
[`docs/DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md); vulnerability reporting in
[`SECURITY.md`](../SECURITY.md). Those documents link here rather than repeat it.

## 1. Operating boundary — local, single‑node demo

This project is a **local, single‑node developer demo and research pipeline**.
It has **no** authentication, **no** authorization, **no** multi‑tenant
isolation, **no** per‑user sessions, and **no** rate limiting. Do not expose the
web app directly to untrusted networks or treat it as a hosted multi‑user
service. Run it on a machine (or container/VM) you control.

## 2. Network egress — what calls out, and when

| Path | Egress |
|---|---|
| Web app **`preview`** mode (`/api/extract`, `mode=preview`) | **None.** The PDF is partitioned locally; no LLM or network call. Works fully offline. |
| Web app **`full`** mode | Calls the **selected provider**: `azure` / `openai` / `bedrock` → **cloud egress** using *your* credentials/endpoints; `ollama` → **local**, no cloud. |
| CLI / quantization / workflows | Download models & datasets from the Hugging Face Hub **at run time** (see §6 for offline). Provider generation follows the same provider rules as above. |

Provider credentials are read from the environment / `.env`. The settings
endpoint reports only whether a variable **is set** — **secret values are never
returned** to the UI. For restricted or unknown documents the intended contract
is **zero cloud calls**; preview mode enforces this by never invoking a provider.

## 3. Upload handling (web app)

An uploaded file is validated and stored defensively before any heavy work:

- **Extension:** only `*.pdf` is accepted (otherwise `400`).
- **Content sniff:** the payload must carry the **`%PDF-`** signature within its
  first bytes, so a renamed non‑PDF is rejected (`415`) even with a `.pdf` name.
- **Size cap:** streamed writes are capped by **`PDFQA_MAX_UPLOAD_MB`**
  (default **25 MiB**; exact‑byte override `PDFQA_MAX_UPLOAD_BYTES`). Exceeding
  it returns `413`.
- **Bounded memory:** the upload is copied to disk in **≤1 MiB chunks** — it is
  **never read whole into memory**, so an oversized upload cannot exhaust RAM.
- **Empty upload:** rejected (`400`).

### Temp‑file lifetime & retention

Each request writes into a fresh `tempfile.mkdtemp(prefix="pdfqa_")` directory
that is **always removed in the handler's `finally`** (including on error).
**Retention == request lifetime.** The demo writes nothing to a database and
persists no upload server‑side. Generated Q&A / JSONL is returned in the HTTP
response (and offered as a download) — not stored.

## 4. Malicious‑PDF limits (be explicit)

The guards in §3 stop the common abuses: wrong type, oversize, empty, and
memory‑blowup uploads. **They do not sandbox the PDF parser.** Parsing uses
upstream libraries (`unstructured`, `pymupdf`); a crafted PDF that targets a
parser vulnerability is **out of scope** of these guards. Mitigate by running
the demo inside a container/VM you trust and keeping those libraries updated.

## 5. PII gate (what it is — and is not)

`pdf_qa/pii.py` implements a **baseline pattern detector** for high‑risk Korean
identifiers — resident registration number (with date/check validation), credit
card (Luhn), phone, email, and bank account — plus a mechanical‑fake/canary
validator.

- `has_real_pii()` **gates training export** in the PDF‑native workflow: rows
  containing real PII are **quarantined, not exported**.
- `redact()` masks matches as `[REDACTED_<KIND>]`.
- `scripts/scan_secrets.py` runs the same shapes in CI to keep secrets/PII out
  of commits.

**Honest limits:** this is a regex baseline, **not** a certified DLP or a
compliance control. It can miss real PII or over‑match. Real‑organization
denylists are injected **only in private CI**, never committed here.

## 6. Telemetry‑off / offline execution

The application adds **no telemetry of its own**. Upstream libraries (Hugging
Face Hub, `transformers`, `datasets`, `unstructured`) may fetch assets or emit
usage pings. To run fully offline once assets are cached:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

Web‑app **preview** mode requires no network at all.

## 7. Threat model

**Supported (in scope):**

- Reject wrong‑type / oversize / empty uploads; bound upload memory.
- No cloud call in `preview` mode or for restricted/unknown documents.
- Keep secret **values** out of the settings endpoint and out of logs.
- Baseline PII gate on training export; secret/PII shape scan in CI.
- Per‑request temp isolation with guaranteed cleanup.

**Not supported (out of scope — do not assume these exist):**

- Authentication, authorization, multi‑tenant isolation, per‑user sessions.
- Network rate limiting or DoS protection.
- Sandboxing/hardening of the third‑party PDF parser.
- Guaranteed/complete PII detection or removal.
- Any compliance certification (SOC 2, ISO 27001, HIPAA, GDPR guarantees).

To report a suspected vulnerability, see [`SECURITY.md`](../SECURITY.md).
