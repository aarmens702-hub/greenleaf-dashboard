"""Evaluation: one path that scores a controller on fixed scenarios (docs/API-CONTRACT.md).

- run_rb plays the grower's rules on the raw env (absolute controls, raw state), as
  envs.run_baseline does, with any params.
- run_policy rebuilds a run's env stack from its manifest.json with train.build_vec_env, loads
  vecnorm_<checkpoint>.pkl onto the stack under VecNormalize, freezes it (training=False,
  norm_reward=False), loads ckpt_<checkpoint>.zip on cpu and predicts deterministically.

Both grade the season with metrics (raw states and controls, never the reward) and add
paper_reward_sum, info["paper_reward"] summed in step order. With record_steps they also return
one frame row per step t = 0..T-1, read from the state after the step.

Each seal's test splits (scenarios.SEALS: 2012's TEST, 2013's TEST_2013) are sealed (PROTOCOL §8):
only --final <final-registered row> may score them, once per seal. It refuses anything that could
break the seal before any env is built, then scores the row's runs at their registered checkpoints
and its baselines on its seal's splits, and writes FINAL_RESULTS/<row>/
(stamp.json, seasons.parquet, per_split.json) for `ledger final-result` to record, then replays/:
export_replay's hourly JSON of every season of each arm's lowest registered seed and of every
baseline, from the seasons it scored. make_raw_env
refuses TEST weather too. run_rb and run_policy pass allow_test through (default False); only the
--final path (_final) sets it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from greenleaf_control import envs, ledger, manifest, metrics, rule_based, scenarios, stats, train

SEASON_LENGTH = 60  # days; the CLI scores full seasons
# Upstream's control order (gl_gym @2d3febb): uBoil, uCO2, uThScr, uVent, uLamp, uBlScr.
CONTROLS = ("boil", "co2", "thscr", "vent", "lamp", "blscr")
# --final's outputs, written once; empty until the seal (built from this file, like selection's).
FINAL_RESULTS = Path(__file__).resolve().parents[1] / "results" / "final"
FINAL_PRIMARY = "profit_eur_m2"
FINAL_GUARDS = (
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "paper_return",
    "yield_kg_m2",
    "solver_failures",
)
NO_CLAIM = "n<5: no claim"
# The CLI's baselines: envs.BASELINES and the rules tune_rb --price-regime agc2020 tuned.
BASELINES = (*envs.BASELINES, "rb_tuned_agc2020")


def _grade(infos: list[dict], raw, price_regime: str = "sim") -> dict:
    """metrics.episode_metrics on the season's raw trace with raw's params and N, at price_regime's
    prices, plus paper_reward_sum (summed in step order, as train's final_eval.json sums
    paper_return)."""
    states, controls = metrics.trace_from_infos(infos)
    graded = metrics.episode_metrics(
        states,
        controls,
        np.asarray(raw.p),
        season_steps=raw.N,
        prices=metrics.PRICE_REGIMES[price_regime],
    )
    # A plain += in step order, as train does: Python 3.12's sum() of floats is compensated, so it
    # can differ from train's paper_return in the last bit.
    paper_reward_sum = 0.0
    for info in infos:
        paper_reward_sum += info["paper_reward"]
    return {**graded, "paper_reward_sum": paper_reward_sum}


def _frame(infos: list[dict]) -> pd.DataFrame:
    """One row per step: t, co2 [ppm], temp [C], rh [%] after the step, the applied u_*, the RB's
    u_rb_* (residual arms only), profit and cost_*."""
    x = np.stack([info["x"] for info in infos])
    u = np.stack([info["controls"] for info in infos])
    cols = {
        "t": np.arange(len(infos)),
        "co2": co2dens2ppm(x[:, 2], x[:, 0] * 1e-6),  # x[0] CO2 [mg/m3] -> [kg/m3] -> [ppm]
        "temp": x[:, 2],
        "rh": vaporPres2rh(x[:, 2], x[:, 15]),  # x[15] vapour pressure [Pa] -> RH [%]
        **{f"u_{name}": u[:, j] for j, name in enumerate(CONTROLS)},
    }
    if "u_rb" in infos[0]:
        u_rb = np.stack([info["u_rb"] for info in infos])
        cols.update({f"u_rb_{name}": u_rb[:, j] for j, name in enumerate(CONTROLS)})
    cols["profit"] = [info["profit"] for info in infos]
    cols.update({f"cost_{c}": [info[f"cost_{c}"] for info in infos] for c in metrics.CONSTRAINTS})
    return pd.DataFrame(cols)


def run_rb(
    params: dict,
    scenario: dict,
    season_length: int = 60,
    record_steps: bool = False,
    return_trace: bool = False,
    *,
    allow_test: bool = False,
    price_regime: str = "sim",
) -> tuple:
    """The rule-based controller with params, one season on the raw env, played and graded at
    price_regime's prices.

    With return_trace, also the graded season's (states, controls, params) as a third item."""
    env = envs.SolverFailureFlag(
        envs.make_raw_env(
            scenario=scenario,
            season_length=season_length,
            normalize_actions=False,
            allow_test=allow_test,
            price_regime=price_regime,
        )
    )
    controller = rule_based.make_controller(params)
    env.reset()
    infos, done = [], False
    while not done:
        _, _, terminated, truncated, info = env.step(rule_based.rb_action(env, controller))
        infos.append(info)
        done = terminated or truncated
    result = _grade(infos, env.unwrapped, price_regime)
    params = np.asarray(env.unwrapped.p, dtype=np.float64)
    env.close()
    frame = _frame(infos) if record_steps else None
    if return_trace:
        return result, frame, (*metrics.trace_from_infos(infos), params)
    return result, frame


def run_policy(
    run_dir: Path,
    checkpoint: str,
    scenario: dict,
    season_length: int = 60,
    record_steps: bool = False,
    return_trace: bool = False,
    *,
    allow_test: bool = False,
) -> tuple:
    """A run's checkpoint, one deterministic season on its training stack with frozen stats, played
    and graded at its manifest's price regime (sim if it records none).

    With return_trace, also the graded season's (states, controls, params) as a third item."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    stack = train.build_vec_env(
        config, 1, scenario=scenario, season_length=season_length, allow_test=allow_test
    )
    venv = VecNormalize.load(run_dir / f"vecnorm_{checkpoint}.pkl", stack.venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(run_dir / f"ckpt_{checkpoint}.zip", device="cpu")
    obs = venv.reset()
    infos, done = [], False
    while not done:  # DummyVecEnv resets itself when the season ends
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, step_infos = venv.step(action)
        infos.append(step_infos[0])
        done = bool(dones[0])
    raw = venv.unwrapped.envs[0].unwrapped
    result = _grade(infos, raw, config.get("price_regime", "sim"))
    params = np.asarray(raw.p, dtype=np.float64)
    venv.close()
    frame = _frame(infos) if record_steps else None
    if return_trace:
        return result, frame, (*metrics.trace_from_infos(infos), params)
    return result, frame


def _score(
    controller: str,
    scenario: dict,
    *,
    return_trace: bool = False,
    allow_test: bool = False,
    price_regime: str = "sim",
) -> dict | tuple:
    """One 60-day season of a BASELINES entry (played and graded at price_regime) or a run dir's
    final checkpoint (at its manifest's): its result, or with return_trace (result, frame, (states,
    controls, params)) from that same season."""
    if controller in ("rb_default", "rb_tuned", "rb_tuned_agc2020"):
        params = rule_based.load_rb_params(controller.removeprefix("rb_"))
        scored = run_rb(
            params,
            scenario,
            SEASON_LENGTH,
            record_steps=return_trace,
            return_trace=return_trace,
            allow_test=allow_test,
            price_regime=price_regime,
        )
        return scored if return_trace else scored[0]
    if controller in envs.BASELINES:  # zero_controls, random
        infos = envs.run_baseline(
            controller,
            scenario,
            SEASON_LENGTH,
            allow_test=allow_test,
            price_regime=price_regime,
        )
        env = envs.make_raw_env(
            scenario=scenario,
            season_length=SEASON_LENGTH,
            normalize_actions=False,
            allow_test=allow_test,
            price_regime=price_regime,
        )
        result = _grade(infos, env.unwrapped, price_regime)  # the model's params and N
        params = np.asarray(env.unwrapped.p, dtype=np.float64)
        env.close()
        if return_trace:
            return result, _frame(infos), (*metrics.trace_from_infos(infos), params)
        return result
    scored = run_policy(
        Path(controller),
        "final",
        scenario,
        SEASON_LENGTH,
        record_steps=return_trace,
        return_trace=return_trace,
        allow_test=allow_test,
    )
    return scored if return_trace else scored[0]


def _by_run(seasons: list[dict], split: str, arm: str, key: str) -> np.ndarray:
    """runs x scenarios of key for one arm on one split: runs as scored (by seed; a baseline is one
    run), scenarios in the split's order."""
    mine = [s for s in seasons if s["split"] == split and s["arm"] == arm]
    runs = dict.fromkeys(s["run"] for s in mine)
    return np.array([[s[key] for s in mine if s["run"] == r] for r in runs], dtype=np.float64)


def _ci(scores: np.ndarray) -> dict:
    iqm, low, high = stats.iqm_ci(scores)
    return {"iqm": iqm, "ci_low": low, "ci_high": high}


def _shown(ci: dict) -> str:
    return f"{ci['iqm']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def _final(row_id: str, ledger_path: Path, parser: argparse.ArgumentParser) -> int:
    """evaluate --final <row>: the refusals in order, before any env, weather read or write; then
    every registered run at its checkpoint and every baseline on each of its seal's splits, once to
    FINAL_RESULTS/<row>/, with replays/ of the traced seasons last. Writes no ledger row; prints the
    command that records per_split.json, or raises with it after the replays if a traced season
    ended early (it gets no replay)."""
    rows = ledger.rows(path=ledger_path)
    finals = [r for r in rows if r["kind"] == "final-registered"]
    row = next((r for r in finals if r["row_id"] == row_id), None)
    if row is None:
        parser.error(f"--final {row_id}: no final-registered row has that id in {ledger_path}")
    seal = ledger.seal_of(row)
    if ledger.seal_spent(rows, seal):
        parser.error(
            f"--final {row_id}: a final-result of the {seal} seal is in the ledger; the seal is spent"
        )
    newest = [r for r in finals if ledger.seal_of(r) == seal][-1]
    if row is not newest:
        parser.error(
            f"--final {row_id}: a newer final-registered row of the {seal} seal,"
            f" {newest['row_id']}, replaces it"
        )
    here, there = manifest.fingerprints(), row["fingerprints"]
    differ = sorted(k for k in here.keys() | there.keys() if here.get(k) != there.get(k))
    if differ:
        parser.error(
            f"--final {row_id}: the scorer files differ from the row's fingerprints:"
            f" {', '.join(differ)}"
        )
    # FINAL_RESULTS may hold only spent seals' finals: anything else is a final that opened its
    # seal's test weather and was never recorded, or a dir no final-registered row names.
    spent = {r["row_id"] for r in finals if ledger.seal_spent(rows, ledger.seal_of(r))}
    held = sorted(p.name for p in FINAL_RESULTS.iterdir()) if FINAL_RESULTS.exists() else []
    stray = [name for name in held if name not in spent]
    if stray:
        parser.error(
            f"--final {row_id}: {FINAL_RESULTS} is not empty: it holds {', '.join(stray)}, no spent"
            " seal's final; an earlier final already opened the test weather in this checkout"
        )
    seal_splits = scenarios.SEALS[seal]
    scenarios.require_weather([s for split in seal_splits.values() for s in split])
    arms = ledger.final_arms(row, rows)
    runs = {r["row_id"]: r for r in rows if r["kind"] == "run"}
    for arm, scored in arms.items():
        for run_id, c in scored.items():
            run_dir = Path(runs[run_id]["run_dir"])
            need = ("manifest.json", f"ckpt_{c}.zip", f"vecnorm_{c}.pkl", "selected.json")
            missing = [name for name in need if not (run_dir / name).is_file()]
            selected = run_dir / "selected.json"
            chosen = None if missing else json.loads(selected.read_text())["checkpoint"]
            if missing or chosen != c:
                why = f"no {', '.join(missing)}" if missing else f"selected.json says {chosen!r}"
                parser.error(
                    f"--final {row_id}: run {run_id} (arm {arm}, seed {runs[run_id]['seed']},"
                    f" {run_dir}): {why}, but the row registers checkpoint {c!r}"
                )
    stamp = manifest.code_stamp(smoke=False)  # a dirty tree raises

    learning = [
        (arm, runs[run_id], c)
        for arm in sorted(arms)
        for run_id, c in sorted(arms[arm].items(), key=lambda item: runs[item[0]]["seed"])
    ]
    # Each arm's lowest registered seed is traced: its seasons, and every baseline's, are replayed.
    traced = {arm: min(arms[arm], key=lambda run_id: runs[run_id]["seed"]) for arm in arms}
    # Imported here: export_replay imports evaluate, so a module-level import would be circular.
    from greenleaf_control import export_replay

    # The replays' stamp and each traced run's meta.run (export_replay's provenance), built before
    # the weather opens, so after it nothing but a season can fail.
    replay_stamp = {
        "commit": stamp["commit"],
        "dirty": stamp["dirty"],
        "gl_gym": manifest._versions()["gl-gym"],
    }
    provenance = {}
    for arm, run_id in traced.items():
        c = arms[arm][run_id]
        selected_by = next(
            r for r in rows if r["kind"] == "result" and r["selected"].get(run_id) == c
        )
        trained = json.loads((Path(runs[run_id]["run_dir"]) / "manifest.json").read_text())
        provenance[arm] = {
            "id": run_id,
            "registered": runs[run_id]["registered"],
            "result": selected_by["row_id"],
            "arm": runs[run_id]["arm"],
            "seed": runs[run_id]["seed"],
            "checkpoint": c,
            "commit": trained["commit"],
        }

    out = FINAL_RESULTS / row_id
    out.mkdir(parents=True, exist_ok=False)
    with (out / "stamp.json").open("x") as f:
        f.write(json.dumps(stamp, sort_keys=True, indent=2) + "\n")
    seasons, replays = [], []
    for split, split_scenarios in seal_splits.items():
        for s in split_scenarios:
            where = {"split": split, **s, "season_length": SEASON_LENGTH}
            for arm, run, c in learning:
                trace = run["row_id"] == traced[arm]
                scored = run_policy(
                    Path(run["run_dir"]),
                    c,
                    s,
                    SEASON_LENGTH,
                    record_steps=trace,
                    return_trace=trace,
                    allow_test=True,
                )
                identity = {"arm": arm, "run": run["row_id"], "seed": run["seed"], "checkpoint": c}
                seasons.append({**identity, **where, **scored[0]})
                if trace:
                    replays.append((arm, provenance[arm], split, s, scored))
            for b in row["baselines"]:
                scored = _score(b, s, return_trace=True, allow_test=True)
                identity = {"arm": b, "run": None, "seed": None, "checkpoint": None}
                seasons.append({**identity, **where, **scored[0]})
                replays.append((b, None, split, s, scored))
    with (out / "seasons.parquet").open("xb") as f:
        pd.DataFrame(seasons).to_parquet(f)

    per_split = {}
    for split in seal_splits:
        rb_tuned = _by_run(seasons, split, "rb_tuned", FINAL_PRIMARY)
        per_split[split] = {}
        for arm in [*sorted(arms), *row["baselines"]]:
            primary = _by_run(seasons, split, arm, FINAL_PRIMARY)
            numbers = {
                **_ci(primary),
                "guards": {g: _ci(_by_run(seasons, split, arm, g)) for g in FINAL_GUARDS},
                "invalid_episodes": sum(
                    not s["valid"] for s in seasons if s["split"] == split and s["arm"] == arm
                ),
                "prob_improvement": None
                if arm == "rb_tuned"
                else stats.prob_improvement(primary, rb_tuned),
                "scored": arms.get(arm, {}),
            }
            per_split[split][arm] = numbers
            guards = "  ".join(f"{g} {_shown(numbers['guards'][g])}" for g in FINAL_GUARDS)
            p = numbers["prob_improvement"]
            line = (
                f"{split}  {arm}  {FINAL_PRIMARY} {_shown(numbers)}  {guards}"
                f"  P(improvement) {'null' if p is None else f'{p:.3f}'}"
                f"  n_runs {len(primary)}  invalid_episodes {numbers['invalid_episodes']}"
            )
            if arm in arms and len(primary) < 5:
                line += f"  {NO_CLAIM}"
            print(line)
    text = json.dumps({"per_split": per_split}, sort_keys=True, indent=2, allow_nan=False)
    numbers_path = out / "per_split.json"
    with numbers_path.open("x") as f:
        f.write(text + "\n")

    record = (
        "python -m greenleaf_control.ledger final-result"
        f" --final-registered {row_id} --numbers {numbers_path}"
    )

    full, short = SEASON_LENGTH * export_replay.N_DAY, []
    (out / "replays").mkdir()
    for name, run, split, s, (result, frame, (states, controls, params)) in replays:
        if len(frame) != full:  # a solver failure ended it early: no replay of a partial season
            short.append(f"{name} {split} day {s['start_day']}: {len(frame)} of {full} steps")
            continue
        payload = export_replay.export(
            frame,
            result,
            states,
            controls,
            params,
            controller=name,
            scenario=s,
            stamp=replay_stamp,
            split=split,
            run=run,
        )
        named = (
            name if run is None else f"{name} seed {run['seed']} at checkpoint {run['checkpoint']}"
        )
        payload["meta"]["note"] = f"registered final {row_id}: {named}"
        path = out / "replays" / f"{name}_{s['location']}_{s['growth_year']}_{s['start_day']}.json"
        with path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(payload, allow_nan=False, separators=(",", ":")))
    if short:
        raise RuntimeError(
            f"--final {row_id}: {len(short)} traced season(s) ended early, so they have no replay:"
            f" {'; '.join(short)}. Every other replay is written, and seasons.parquet and"
            f" per_split.json are complete; record them with: {record}"
        )
    print(f"next: {record}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.evaluate")
    p.add_argument(
        "--controller",
        help=f"one of {', '.join(BASELINES)}, or a run dir (its final checkpoint)",
    )
    p.add_argument(
        "--split",
        choices=list(scenarios.SPLITS),
        help=f"the test splits ({', '.join(n for t in scenarios.SEALS.values() for n in t)}) need"
        " --final",
    )
    p.add_argument("--out", type=Path, help="the parquet to write; must not exist")
    p.add_argument(
        "--price-regime",
        choices=list(metrics.PRICE_REGIMES),
        help="a baseline's prices, PROTOCOL §1 (default sim); a run dir's are its manifest's",
    )
    p.add_argument(
        "--final",
        metavar="ROW",
        help="its seal's newest final-registered row: score its runs and baselines once on its"
        " seal's test splits into control/results/final/<ROW>/ (takes no --controller, --split,"
        " --out or --price-regime)",
    )
    return p


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    flags = {"--controller": args.controller, "--split": args.split, "--out": args.out}
    if args.final is not None:
        given = [
            flag
            for flag, value in {**flags, "--price-regime": args.price_regime}.items()
            if value is not None
        ]
        if given:
            parser.error(f"--final takes no {', '.join(given)}: it scores its row as registered")
        return _final(args.final, ledger_path, parser)
    missing = [flag for flag, value in flags.items() if value is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if any(args.split in t for t in scenarios.SEALS.values()):
        parser.error(
            f"--split {args.split} is TEST weather, sealed (PROTOCOL §8): only --final"
            " <final-registered row> may score it"
        )
    if args.out.exists():
        parser.error(f"--out {args.out} already exists; a result is written once")
    baseline = args.controller in BASELINES
    if not baseline and not (Path(args.controller) / "manifest.json").is_file():
        parser.error(
            f"--controller {args.controller}: not one of {BASELINES} and not a run dir"
            " with a manifest.json"
        )
    if not baseline and args.price_regime is not None:
        parser.error(
            f"--price-regime is for a baseline: run dir {args.controller} is scored at its"
            " manifest's price_regime"
        )
    price_regime = "sim" if args.price_regime is None else args.price_regime
    if not baseline:  # the regime the run dir is graded at, for its rows
        config = json.loads((Path(args.controller) / "manifest.json").read_text())["config"]
        price_regime = config.get("price_regime", "sim")

    rows = [
        {
            "controller": args.controller,
            "checkpoint": None if baseline else "final",
            "split": args.split,
            **s,
            "season_length": SEASON_LENGTH,
            "price_regime": price_regime,
            **_score(args.controller, s, price_regime=price_regime),
        }
        for s in scenarios.SPLITS[args.split]
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as f:
        pd.DataFrame(rows).to_parquet(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
