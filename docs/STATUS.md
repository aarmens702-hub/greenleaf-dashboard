# Status and handoff

**Read this first when you pick the project up on any machine.** It's the shared memory: where things stand, what's next, what we learned, and how Aarmen likes to work. Update it at the end of each session.

*Last updated: 2026-10-27, Mac session.*

## Where things are
- **The repo** is `github.com/aarmens702-hub/greenleaf-dashboard`: public, default branch `master`, no license yet. It holds the hackathon dashboard that placed 2nd (RBC × BCCAI × SFU, May 2026). GreenLeaf 2.0 is being added in `control/`.
- **The harness was ported** from crivo and Edison on 2026-10-27, on a local `dev` branch that isn't pushed yet:
  - `AGENTS.md`, `CLAUDE.md`;
  - the four agents, five hooks and five skills under `.claude/`;
  - the issue templates;
  - `docs/PROTOCOL.md`, `API-CONTRACT.md`, `ARCHITECTURE.md`, `PRD.md`;
  - `specs/2026-10-27-the-plan.md`.
- **Local branch `prod-rag-pipeline`** (the dashboard's CI and RAG eval) is folded into `dev` and was never pushed.
- **Nothing in `control/` exists yet.** Stage 0 builds it.

## To do, in order
1. **Done 2026-10-27:**
   - `dev` pushed, with the dashboard's `prod-rag-pipeline` work;
   - labels created;
   - Stage 0 and Stage 1 filed as issues #1–#18.
2. **Stage 0, now. Start with `/start-task 1`:**
   - #1 `control/` scaffold, licenses, CI;
   - #2 manifests;
   - #3 the ledger;
   - #4 `check_locks`;
   - #5 the `guard_runs` hook.
   - #5 is independent and can run in parallel with #2–#4.
3. **Stage 1:**
   - #6 scenarios + weather;
   - #7 rewards;
   - #8 rule-based + configs;
   - #9 envs;
   - #10 metrics;
   - #11 golden numbers;
   - #12 Lagrangian;
   - #13 train/evaluate/selection;
   - #14 the probe ladder;
   - #15 residual and Lagrangian probes;
   - #16 overfit a trivial case;
   - #17 stats/tune/grid/Modal;
   - #18 plots.
   - No experiment before all of these are shipped.
4. **Stage 2** (E1 … E6): experiments, each pre-registered. File them with `/new-task` once Stage 1 is green. Ask
   GreenLeaf (D6) is built in parallel.
5. **Stage 3:** the final (once), then the analysis.
6. **Stage 4:** deploy and polish (D1–D7 in the plan). Visual roadmap: https://claude.ai/artifact/19obed2CuttzDAPZzbFDuw

**Run the harness from a Claude Code session started inside this repo** (`cd ~/Desktop/greenleaf-dashboard && claude`).
Its agents, hooks and skills load only there.

## What we've learned
- **GreenLight-Gym at `2d3febb`:**
  - Installs cleanly on the M3 (Python 3.12, no build step).
  - Runs ~2,000 steps/s; a 60-day season with the rule-based controller takes 2.9 s, return 3104.58.
  - The bench script is `bench.py` in the 2026-10-27 session scratchpad; S1-6 re-measures it in-repo.
- **Upstream bugs (read from source):**
  - The lamp penalty always returns 0.
  - The `temp_penalty` and `co2_penalty` info labels are swapped.
  - A solver error is swallowed by a bare `except` and the episode is truncated.
  - Its own tests may not test much (unverified).
- **Prior work that shaped the contribution:**
  - Plain PPO-Lagrangian on this simulator: SafeGreenRL (Jun 2026, "conservative economics").
  - Unconstrained residual on the rule-based controller: arXiv 2607.11959.
  - Simple optimizers beat RL in the Autonomous Greenhouse Challenge: iGrow, AggieAI.
  - The paper's rule-based baseline is untuned.
- **Compute:** CPU-bound, so Modal CPU containers (~$10 expected), capped at $40 total. GPUs don't help here.

## How Aarmen likes to work
- **Decisions stated as defaults**, not rounds of questions; he vetoes what he doesn't like.
- **Plans before changes;** casual tone; no em dashes in public-facing text.
- **He does logins, key pastes and anything outward-facing** (pushes, repo settings, publishing); merges to `master` need his explicit yes.
- **Small paid runs only;** check the real spend; estimates have under-counted ~3x before.

## Set up a new laptop
1. Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and the GitHub CLI; `gh auth login`.
2. `git clone https://github.com/aarmens702-hub/greenleaf-dashboard && cd greenleaf-dashboard && git switch dev`.
3. `cd control && uv sync` (once Stage 0 lands).
4. `modal setup` (Aarmen only; the token stays out of agent context).
