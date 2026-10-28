# Architecture

## The two parts
- **`control/`: GreenLeaf 2.0** (active). A uv project (Python 3.12) with its own `pyproject.toml`, `uv.lock`,
  tests and AGPL-3.0 license, because it depends on GreenLight-Gym, which is AGPL.
- **The repo root: the hackathon dashboard** (frozen). Streamlit (`app/streamlit_app/`), an agentic RAG chat
  (`app/chat/`, `app/rag/`), the spike early-warning model (`analysis/`, `models/`), and the organisers' synthetic
  data (`data/raw/`). Its own `pyproject.toml` and `.venv`.

## GreenLeaf 2.0 data flow
```
weather CSVs (data/weather, hashed) ─┐
GreenLight-Gym @2d3febb ─────────────┤
                                     ▼
scenarios.py ─► envs.py: make_raw_env ─► [SeasonProgress] ─► [ResidualRBWrapper ◄─ rule_based.py] ─► SolverFailureFlag
                    │ reward: rewards.ConstraintCostReward (profit − w·costs; costs labelled correctly)
                    ▼
train.py: SubprocVecEnv → VecMonitor → [VecLagrangianReward ◄─ lagrangian.PIDLagrangian] → VecNormalize → SB3 PPO
    │ writes runs/<exp>/<arm>/seed<k>/ (manifest.json, ckpt_*, vecnorm_*, tb/, lambdas.csv)
    ▼
selection.py (VALIDATION) ─► evaluate.run_policy (rebuilds the stack from the manifest, frozen stats)
                                   │ per-step states and controls
                                   ▼
                        metrics.episode_metrics (raw states; independent of rewards.py)
                                   ▼
               ledger.py result (rliable IQM + CIs via stats.py) ─► ledger/ledger.jsonl ─► results/
```

## Who works where

| Path | Written by | Notes |
|---|---|---|
| `control/greenleaf_control/{scenarios,rewards,envs,metrics}.py`, `docs/PROTOCOL.md` | builder, on a `core` issue | Scorer files: Aarmen approves every diff |
| `control/greenleaf_control/*` (other modules) | builder | Against `docs/API-CONTRACT.md` |
| `control/tests/` (including `golden/`, `probes/`) | test-writer | Golden values only from a number the issue gives |
| `control/configs/ppo.yml`, `rule_based_default.yml` | builder, once (copied from upstream with attribution) | Never tuned by hand |
| `control/configs/rule_based_tuned.yml` | `tune_rb` CLI only | Via `/experiment` E1 |
| `control/ledger/ledger.jsonl` | `ledger` CLI only | Append-only; `check_locks.py` |
| `control/results/` | `evaluate`, `stats`, `ledger` CLIs only | `final/` is empty until the seal |
| `control/runs/` | `train` CLI only | Git-ignored; mirrors the Modal volume |
| `control/data/weather/` | `scripts/fetch_weather.py`, once | Immutable; hashed into every run |
| `.claude/`, `AGENTS.md`, `CLAUDE.md`, `docs/`, `specs/` | the orchestrating session, with Aarmen | Harness and docs |
| repo root (the dashboard) | only its cleanup issue | Frozen |

## Key decisions
Dated, with who decided. Not re-argued without a new reason.

- **2026-10-27, Aarmen.** GreenLeaf 2.0 is the next portfolio project, chosen over river forecasting (Tributary, on
  hold). It lives in this repo, not a new one.
- **2026-10-27, Aarmen.** Training grids run on Modal CPU containers, total spend capped at $40. No GPU: the simulator
  and a small MLP are CPU-bound.
- **2026-10-27, Claude (veto open).** The contribution is a constrained residual PPO: a residual on a BO-tuned
  rule-based controller, with a separate PID-Lagrangian multiplier per constraint and budgets pegged to the
  controller's own violations.
  - Plain PPO-Lagrangian on this simulator is already published (SafeGreenRL, SSRN, Jun 2026).
  - An unconstrained residual on the rule-based controller is in arXiv 2607.11959.
- **2026-10-27, Claude.** GreenLight-Gym pinned at `2d3febb` (pure-Python CasADi; installs on the M3 with no build).
  Not the `AgriControl-2025` branch, which needs C++ and has build-failure issues. Speed: ~2,000 steps/s; one 60-day
  season in 2.9 s on the M3.
- **2026-10-27, Claude.** The upstream simulator is untrusted. Its lamp penalty never fires and its CO₂ and temperature
  penalty labels are swapped (both read from source). We relabel the costs and test every term ourselves.
- **2026-10-27, Claude.** Metrics are independent of the reward code: `metrics.py` never imports `rewards.py`, and
  profit is cross-checked against upstream's summed info.
- **2026-10-27, Claude.** Season-progress observation on every learning arm except `ppo_paper`; episodes end
  `terminated` at season end (Pardo et al. 2018).
- **2026-10-27, Claude.** Residual arms use `log_std_init = -1.0` and scale 0.15, so the policy at init stays near the
  rule-based controller.
- **2026-10-27, Claude.** Statistics: IQM with stratified-bootstrap CIs (rliable); 5 seeds per sweep arm; the chosen
  configuration re-run with 10 fresh seeds; crashed runs count (Agarwal 2021; Patterson 2024).
- **2026-10-27, Claude.** Harness ported from crivo and Edison: four specialists, work types, an append-only ledger
  with fingerprints, `check_locks`, and a sealed test split enforced in code, by hooks and through git history.
- **2026-10-27, Claude (veto open).** Deployment, like Edison and mostly static:
  - a static site on Vercel replays pre-computed seasons (hourly JSON);
  - a Modal web endpoint built from the training image runs the live season, with a per-visitor limit, sleeping when
    idle;
  - the repo holds the ledger, results and report, so every number on the site traces to a run;
  - the live run offers the three test cities and labels any other city "untested here".
- **2026-10-27, Claude (veto open).** Licenses: MIT at the repo root for the dashboard; AGPL-3.0-or-later in
  `control/`. The NASA POWER weather fetcher is our own code, not upstream's AGPL script.
- **2026-10-28, Aarmen.** Seasons are exactly N steps (5,760 for 60 days), matching PROTOCOL §1; upstream's env runs
  N+1, so `envs.py` ends the season itself. The paper's 3104.58 (5,761 steps) is re-measured in #11.
