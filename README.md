# GreenLeaf Dashboard

Streamlit dashboard for the RBC x BCCAI x SFU Beedie Agribusiness Analytics Hackathon 2026.
Built around GreenLeaf CEA, a hypothetical B.C. controlled-environment agriculture operator.

## Folder layout

- `data/raw/` — GreenLeaf source CSVs and case PDF (read-only).
- `data/external/` — external BC benchmarks as YAML files, one per benchmark.
- `data/derived/` — parquets built by `build_derived.py`. Gitignored.
- `analysis/` — exploratory analyses and one-off scripts.
- `models/` — trained model artifacts. `*.pkl` is gitignored.
- `docs/` — supporting documentation.
- `docs/corpus/` — curated documents for the chat assistant. Contents gitignored.
- `app/` — Streamlit app entrypoint.
- `app/tabs/` — one module per dashboard tab.
- `tests/` — pytest suite.

## How to run

TBD — Streamlit entrypoint pending.

## How to develop

- `pytest tests/` — run the test suite.
- `streamlit run app/main.py` — launch the dashboard (once `app/main.py` exists).
