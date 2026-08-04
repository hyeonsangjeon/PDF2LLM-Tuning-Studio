# Security Policy

## Supported versions

This is a research / demo project released from the `main` branch. Security
fixes are applied to **`main` only**; there is no long‑term support branch or
back‑porting guarantee. Always run the latest `main` (pin a commit SHA for
reproducibility).

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| older tags / commits | ❌ (upgrade to latest `main`) |

## Reporting a vulnerability

**Please do not open a public issue for security reports.** Instead:

- Use **GitHub → Security → [Report a vulnerability](https://github.com/hyeonsangjeon/PDF2LLM-Tuning-Studio/security/advisories/new)**
  (private vulnerability reporting), or
- Contact the maintainer privately via their GitHub profile
  ([@hyeonsangjeon](https://github.com/hyeonsangjeon)).

Please include: affected component/path, version or commit SHA, reproduction
steps, and impact. We aim to **acknowledge within 7 days** and to coordinate a
fix and disclosure timeline with you. This is a best‑effort, volunteer‑run
project — there is no paid bug‑bounty.

## Scope

Before reporting, please read [`docs/TRUST_AND_DATA.md`](docs/TRUST_AND_DATA.md),
which defines the **operating boundary and threat model**. In particular, this
project is a **local single‑node demo** with **no** authentication, tenant
isolation, or compliance certification — issues that assume those features exist
are **out of scope** by design (they are documented non‑goals, not defects).

**In scope** (examples): committing secrets/credentials, upload guards that can
be trivially bypassed (size/type/memory), a `preview`‑mode path that makes an
unexpected network call, secret **values** leaking through an endpoint or logs,
or the PII training‑export gate failing to quarantine real PII.

**Out of scope** (examples): "there is no login", "no rate limiting", "the PDF
parser could be exploited by a crafted file" (the parser is third‑party and
un‑sandboxed by design — run in a trusted container/VM), or requests for
compliance attestations.

## Assets & data

Dataset, model, and bundled‑sample provenance and licensing — including known
unknowns — are tracked in
[`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md). The root
[`LICENSE`](LICENSE) (MIT) covers **this repository's source code only** and
does not relicense third‑party datasets or model weights.
