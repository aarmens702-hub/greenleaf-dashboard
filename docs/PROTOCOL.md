# Evaluation protocol (core)

**Version 1, 2026-10-27.** This file decides how every GreenLeaf 2.0 result is scored, so it is a scorer file:
- changing it needs Aarmen's yes;
- its hash stamps every ledger row, so results recorded under an older version stop being comparable.

Sources for each rule are in `docs/ARCHITECTURE.md` (Key decisions) and the harness design.

## 1. The environment
- **Simulator:** GreenLight-Gym (`BartvLaatum/GreenLight-Gym2`), pinned at commit `2d3febb`, env
  `gl_gym/GreenLightTomato-v0`. Treated as untrusted code: everything we rely on has our own test.
- **Season:** 60 days at dt = 900 s, so 5,760 steps. The episode ends `terminated` at the end of the season, because
  the season really ends. A `truncated` episode means a solver failure, counted, never a time limit.
- **Controls (6):** boiler, CO₂ dosing, thermal screen, roof vents, lamps, blackout screen. Each is in [0, 1].
- **Climate bounds:** CO₂ 300–1600 ppm; air temperature 15–34 °C; relative humidity 50–85 %.
- **Prices** (upstream defaults): tomatoes €1.60/kg fresh weight (dry-matter ratio 0.065); heat €0.09/kWh; electricity
  €0.30/kWh; CO₂ €0.30/kg.
- **Observations:**
  - The `ppo_paper` arm keeps upstream's default modules exactly (it reproduces the paper's fixed-scenario setting).
  - Every other learning arm adds a **season-progress** observation (fraction of the 60 days elapsed), because start
    days are randomized and day-of-year alone doesn't tell the agent how much season is left (Pardo et al. 2018).

## 2. Splits (weather scenarios)
A scenario is `(location, growth_year, start_day)`. Weather files are hashed (sha256) and the hashes stamp every run.

| Split | Scenarios | Used for |
|---|---|---|
| TRAIN | Amsterdam 2008, 2009, 2011; start days 45–75, sampled at random each reset | Training |
| TRAIN_EVAL | Amsterdam {2008, 2009, 2011} × start days {45, 60, 75} (9) | RB tuning, Lagrangian budgets |
| VALIDATION | Amsterdam 2010 × start days {45, 59, 75} (3) | Checkpoint selection, every experiment's result |
| PAPER | Amsterdam 2010, day 59 | The reproduction check only |
| TEST (sealed) | 2012 × start days {45, 49, 53, 57, 61, 65, 69, 73}, at Amsterdam, London, and Abbotsford BC (8 each) | The `final` evaluation, once |

VALIDATION contains PAPER's season. That is deliberate: PAPER is only a reproduction check, never a selection target.

## 3. Methods (arms)

| Arm | What it is |
|---|---|
| `zero_controls` | Every actuator at 0 all season. Lower bound: no energy spend |
| `random` | Uniform random controls each step, seeded |
| `rb_default` | Upstream's rule-based controller with its default parameters |
| `rb_tuned` | The same controller, its 12 setpoints tuned with Optuna TPE (seed 0, 200 trials) to maximize the mean paper-style return over TRAIN_EVAL |
| `ppo_paper` | SB3 PPO with upstream's config (`control/configs/ppo.yml`), upstream's action scheme (rate changes of ±0.1 per step), trained on PAPER only |
| `ppo` | The same PPO, trained on TRAIN |
| `residual` | Controls = clip(rb_tuned action + 0.15 × policy action, 0, 1); reward = scaled profit − w × Σ scaled costs; `log_std_init = -1.0`; trained on TRAIN |
| `residual_lag` | `residual` with w = 0 and a PID-Lagrangian multiplier per constraint, with budget = fraction × rb_tuned's mean per-step cost on TRAIN_EVAL |

**Equal effort:**
- The RB gets 200 tuning trials.
- RL arms use the paper's hyperparameters with no tuning.
- Any RL hyperparameter search must be registered and matched by RB tuning effort.
- Every learning arm trains for 2,000,000 steps unless its registration says otherwise.

## 4. Metrics (computed by `metrics.py` from raw states and controls, never from the shaped reward)
Per episode:

| Metric | Definition |
|---|---|
| `profit_eur_m2` (primary) | Fruit revenue (growth in fruit dry weight → fresh weight × price) minus heating, electricity and CO₂ costs, summed over the season |
| `viol_hours_<c>` | For c in {co2, temp, rh}: steps with the indoor value outside its bounds, × 0.25 h |
| `pct_in_bounds_<c>` | 100 × (1 − violating steps / steps) |
| `yield_kg_m2` | Fresh-weight fruit growth over the season |
| `heat_cost`, `elec_cost`, `co2_cost` | Season totals, €/m² |
| `paper_return` | Sum of upstream's per-step reward, with our corrected labels; secondary, for comparison with the paper |
| `solver_failures` | Steps where the integrator failed (the upstream truncation) |
| `steps` | Must equal 5,760 for a valid episode |

- **A cross-check, not shared code:** `metrics.profit_eur_m2` must agree with upstream's summed `profit` info to
  1e-6 on a golden season. They are written independently.
- An episode with `solver_failures > 0` or `steps < 5,760` is invalid. It's reported, never dropped.

## 5. Training, checkpoints, selection
- **Checkpoints:** every 250,000 steps and at the end, each with its frozen VecNormalize stats.
- **Selection on VALIDATION only,** deterministic actions, frozen stats:
  - `residual_lag`: the highest mean profit among checkpoints whose mean per-step cost on every constraint is ≤
    budget × 1.10; if none qualifies, the lowest total cost.
  - Every other learning arm: the highest mean `paper_return`.
- **Fresh runs only:** a run directory is written once and never resumed.

## 6. Statistics and reporting
- **Scores:** a method's scores are a matrix of seeds × scenarios. The headline statistic is the **IQM** with 95%
  stratified-bootstrap CIs (rliable, 2,000 reps, seed 0).
- **Also reported:**
  - per-scenario values (plotted; IQM can hide one bad scenario);
  - performance profiles;
  - probability of improvement over `rb_tuned`.
- **No standard errors.** No "best seed".
- **Seeds:**
  - 5 per arm in sweeps (seeds 0–4);
  - the configuration chosen from a sweep is re-run with 10 fresh seeds (100–109) before any claim (two-stage,
    Patterson et al. 2024);
  - deterministic arms (the RBs, `zero_controls`) need one run.
- **Crashed runs** count: a result lists them, and an arm with crashes or invalid episodes can't be adopted.
- **Diagnostics logged for every learning run:** approx-KL, clip fraction, entropy, explained variance, λ per
  constraint (Lagrangian arms). A negative KL invalidates the run.
- **Every claim names its ledger rows** and is graded verified, partial or unverified. Numbers that don't support the
  headline are reported too.

## 7. The debug ladder (no greenhouse training before it is green)
1. **Env sanity:**
   - `check_env` on every wrapper chain.
   - Full seasons of `rb_default`, `random` and `zero_controls` are finite and exactly 5,760 steps, ending
     `terminated`.
   - Every reward and cost term matches hand-computed values on a violating and a non-violating state.
2. **Golden numbers** (`control/tests/golden/`):
   - `rb_default` on PAPER has upstream return 3104.58 (measured 2026-10-27, gl-gym 2d3febb, Python 3.12, M3).
   - `rb_default`, `random` and `zero_controls` metrics on PAPER, recorded once.
3. **Invariants and properties:**
   - Residual controls stay in [0, 1] for any input.
   - A zero residual equals `rb_tuned`.
   - Cost labels match the bounds.
   - Solver failures are counted.
   - Splits are disjoint, and weather hashes match.
4. **Determinism:** the same seed gives the same trajectory hash; a different seed gives a different hash.
5. **Golden eval path:**
   - The RB through the full training stack (frozen VecNormalize) equals the RB standalone.
   - A reloaded checkpoint reproduces its logged evaluation.
6. **Probe ladder:** our exact wrapper chain plus SB3 PPO, each in seconds, each with its known answer in the test's
   docstring:
   - constant reward;
   - observation-dependent reward;
   - discounting;
   - action-dependent reward;
   - action × observation;
   - residual offset;
   - Lagrangian cost at budget.
7. **Overfit a trivial case:** one scenario (PAPER), a 2-day season, no constraint terms. Within 200,000 steps, PPO must clearly beat `random` and `zero_controls` on return. The gap to `rb_default` is recorded, not gated: a hand-tuned controller can legitimately win a 2-day season, so beating it is not a plumbing test.

## 8. The seal
- **What is sealed:** the TEST scenarios are scored only by `evaluate --final <row>`, where `<row>` is a
  `final-registered` ledger row that lists the selected checkpoints and both RBs and carries Aarmen's approval.
- **Once:** it runs once. The `final-result` row consumes the seal; the code and `check_locks.py` refuse a second.
- **What is not sealed:** loading the 2012 weather to check an env resets is not scoring and is allowed in tests.
  Nothing else may touch the TEST scenarios.

## 9. Experiment sequence (each an `experiment` issue, VALIDATION only)
- **E1:** tune the RB (`rb_tuned`).
- **E2:** reproduce PPO on PAPER (`ppo_paper`, 5 seeds): PPO beats `rb_default` on profit and penalties, as the
  paper's Figure 2 shows.
- **E3:** `ppo` on TRAIN (5 seeds) against `rb_tuned`.
- **E4:** `residual` at w ∈ {0.5, 1, 4} (5 seeds each).
- **E5:** `residual_lag` at budget fraction ∈ {1.0, 0.5, 0.25} (5 seeds each).
- **E6:** the chosen configuration with 10 fresh seeds.
- Then the `final`.
