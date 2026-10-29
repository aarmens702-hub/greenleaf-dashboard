"""A recorded season as hourly JSON for the demo site (D2; the site, D3, replays it):

    python -m greenleaf_control.export_replay --controller rb_default --split paper --out site/

writes <controller>_<location>_<growth_year>_<start_day>.json per scenario of the split (a run
dir is named by its dir), each one 60-day season as {"meta", "hourly", "totals"}.

Hour h is steps 4h..4h+3. Its climate is the state at the end of the hour (frame row 4h + 3) to 3
significant digits, its actuators the hour's mean to 3 decimals, and its profit and violation hours
metrics (the grader) on the season's first 4h + 4 steps, to 4 decimals. totals is the season's
result, unrounded.

Baselines play on the raw env as evaluate scores them; a run dir goes through evaluate.run_policy,
the one evaluation path. The TEST splits are sealed (PROTOCOL §8) and every --final waits for F1,
as in evaluate's CLI.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from greenleaf_control import envs, evaluate, manifest, metrics, scenarios

SEASON_LENGTH = 60  # days, as evaluate's CLI scores them
N_DAY = 96  # steps a day: 86,400 s / 900 s
CLIMATE = ("co2", "temp", "rh")
U_COLS = [f"u_{name}" for name in evaluate.CONTROLS]  # upstream's control order
NOTE = "a simulator run, not a registered result"
UNITS = {"co2": "ppm", "temp": "°C", "rh": "%", "profit": "€/m²", "hours": "h"}


def export(frame, result, states, controls, params, *, controller, scenario, stamp) -> dict:
    """The site's payload for one season: evaluate's per-step frame and result, and the season's
    raw states (T + 1, nx), controls (T, 6) and model params for the running counters."""
    n_hours = len(frame) // 4
    end_of_hour = frame.iloc[3::4]
    means = frame[U_COLS].to_numpy(dtype=np.float64).reshape(n_hours, 4, len(U_COLS)).mean(axis=1)
    prefixes = [
        metrics.episode_metrics(states[: k + 1], controls[:k], params, season_steps=k)
        for k in range(4, 4 * n_hours + 1, 4)
    ]
    hourly = {
        "t_hour": list(range(n_hours)),
        **{
            c: [float(f"{v:.3g}") for v in end_of_hour[c].to_numpy(dtype=np.float64)]
            for c in CLIMATE
        },
        **{col: [round(float(v), 3) for v in means[:, j]] for j, col in enumerate(U_COLS)},
        "profit": [round(m["profit_eur_m2"], 4) for m in prefixes],
        **{
            f"viol_{c}": [round(m[f"viol_hours_{c}"], 4) for m in prefixes]
            for c in metrics.CONSTRAINTS
        },
    }
    meta = {
        "controller": controller,
        "scenario": {**scenario, "season_length": len(frame) // N_DAY},
        "commit": stamp["commit"],
        "dirty": stamp["dirty"],
        "gl_gym": stamp["gl_gym"],
        "note": NOTE,
        "bounds": {c: list(bounds) for c, bounds in metrics.BOUNDS.items()},
        "units": UNITS,
    }
    return {"meta": meta, "hourly": hourly, "totals": result}


def _season(controller: str, scenario: dict) -> tuple:
    """One 60-day season's frame, result and (states, controls, params): a baseline on the raw env,
    graded as evaluate grades it, or a run dir's final checkpoint through evaluate.run_policy."""
    if controller in envs.BASELINES:
        infos = envs.run_baseline(controller, scenario, SEASON_LENGTH)
        env = envs.make_raw_env(
            scenario=scenario, season_length=SEASON_LENGTH, normalize_actions=False
        )
        result = evaluate._grade(infos, env.unwrapped)  # the model's params and N
        params = np.asarray(env.unwrapped.p, dtype=np.float64)
        env.close()
        return evaluate._frame(infos), result, (*metrics.trace_from_infos(infos), params)
    result, frame, trace = evaluate.run_policy(
        Path(controller), "final", scenario, SEASON_LENGTH, record_steps=True, return_trace=True
    )
    return frame, result, trace


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m greenleaf_control.export_replay",
        description="Export recorded seasons as hourly JSON for the demo site.",
    )
    p.add_argument(
        "--controller",
        required=True,
        help=f"one of {', '.join(envs.BASELINES)}, or a run dir (its final checkpoint)",
    )
    p.add_argument(
        "--split",
        choices=list(scenarios.SPLITS),
        required=True,
        help=f"the test splits ({', '.join(scenarios.TEST)}) need --final",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="the dir to write into; no file in it is replaced"
    )
    p.add_argument(
        "--final", metavar="ROW", help="a final-registered row (F1, Stage 3; not implemented)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.final is not None:
        parser.error(
            f"--final {args.final}: the one-time final evaluation is F1's (Stage 3), not"
            " implemented yet"
        )
    if args.split in scenarios.TEST:
        parser.error(
            f"--split {args.split} is TEST weather, sealed (PROTOCOL §8): only --final"
            " <final-registered row> may export it"
        )
    baseline = args.controller in envs.BASELINES
    if not baseline and not (Path(args.controller) / "manifest.json").is_file():
        parser.error(
            f"--controller {args.controller}: not one of {envs.BASELINES} and not a run dir"
            " with a manifest.json"
        )
    name = args.controller if baseline else Path(args.controller).resolve().name
    split = scenarios.SPLITS[args.split]
    paths = [
        args.out / f"{name}_{s['location']}_{s['growth_year']}_{s['start_day']}.json" for s in split
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        parser.error(f"already exists, and an export is written once: {', '.join(existing)}")

    code = manifest.code_stamp(smoke=True)
    stamp = {
        "commit": code["commit"],
        "dirty": code["dirty"],
        "gl_gym": manifest._versions()["gl-gym"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for s, path in zip(split, paths, strict=True):
        frame, result, (states, controls, params) = _season(args.controller, s)
        payload = export(
            frame, result, states, controls, params, controller=name, scenario=s, stamp=stamp
        )
        with path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(payload, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
