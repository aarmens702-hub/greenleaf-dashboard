"""Evaluation: one path that scores a controller on fixed scenarios (docs/API-CONTRACT.md).

- run_rb plays the grower's rules on the raw env (absolute controls, raw state), as
  envs.run_baseline does, with any params.
- run_policy rebuilds a run's env stack from its manifest.json with train.build_vec_env, loads
  vecnorm_<checkpoint>.pkl onto the stack under VecNormalize, freezes it (training=False,
  norm_reward=False), loads ckpt_<checkpoint>.zip on cpu and predicts deterministically.

Both grade the season with metrics (raw states and controls, never the reward) and add
paper_reward_sum, info["paper_reward"] summed in step order. With record_steps they also return
one frame row per step t = 0..T-1, read from the state after the step.

The TEST splits are sealed (PROTOCOL §8): only --final <final-registered row> may score them, and
that is F1's (Stage 3), so every --final is refused here. make_raw_env refuses TEST weather too.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from greenleaf_control import envs, metrics, rule_based, scenarios, train

SEASON_LENGTH = 60  # days; the CLI scores full seasons
# Upstream's control order (gl_gym @2d3febb): uBoil, uCO2, uThScr, uVent, uLamp, uBlScr.
CONTROLS = ("boil", "co2", "thscr", "vent", "lamp", "blscr")


def _grade(infos: list[dict], raw) -> dict:
    """metrics.episode_metrics on the season's raw trace with raw's params and N, plus
    paper_reward_sum (summed in step order, as train's final_eval.json sums paper_return)."""
    states, controls = metrics.trace_from_infos(infos)
    graded = metrics.episode_metrics(states, controls, np.asarray(raw.p), season_steps=raw.N)
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
    params: dict, scenario: dict, season_length: int = 60, record_steps: bool = False
) -> tuple[dict, pd.DataFrame | None]:
    """The rule-based controller with params, one season on the raw env."""
    env = envs.SolverFailureFlag(
        envs.make_raw_env(scenario=scenario, season_length=season_length, normalize_actions=False)
    )
    controller = rule_based.make_controller(params)
    env.reset()
    infos, done = [], False
    while not done:
        _, _, terminated, truncated, info = env.step(rule_based.rb_action(env, controller))
        infos.append(info)
        done = terminated or truncated
    result = _grade(infos, env.unwrapped)
    env.close()
    return result, _frame(infos) if record_steps else None


def run_policy(
    run_dir: Path,
    checkpoint: str,
    scenario: dict,
    season_length: int = 60,
    record_steps: bool = False,
    return_trace: bool = False,
) -> tuple:
    """A run's checkpoint, one deterministic season on its training stack with frozen stats.

    With return_trace, also the graded season's (states, controls, params) as a third item."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    stack = train.build_vec_env(config, 1, scenario=scenario, season_length=season_length)
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
    result = _grade(infos, raw)
    params = np.asarray(raw.p, dtype=np.float64)
    venv.close()
    frame = _frame(infos) if record_steps else None
    if return_trace:
        return result, frame, (*metrics.trace_from_infos(infos), params)
    return result, frame


def _score(controller: str, scenario: dict) -> dict:
    """One 60-day season of a BASELINES entry or a run dir's final checkpoint."""
    if controller in ("rb_default", "rb_tuned"):
        params = rule_based.load_rb_params(controller.removeprefix("rb_"))
        return run_rb(params, scenario, SEASON_LENGTH)[0]
    if controller in envs.BASELINES:  # zero_controls, random
        infos = envs.run_baseline(controller, scenario, SEASON_LENGTH)
        env = envs.make_raw_env(
            scenario=scenario, season_length=SEASON_LENGTH, normalize_actions=False
        )
        result = _grade(infos, env.unwrapped)  # the model's params and N
        env.close()
        return result
    return run_policy(Path(controller), "final", scenario, SEASON_LENGTH)[0]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.evaluate")
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
    p.add_argument("--out", type=Path, required=True, help="the parquet to write; must not exist")
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
            " <final-registered row> may score it"
        )
    if args.out.exists():
        parser.error(f"--out {args.out} already exists; a result is written once")
    baseline = args.controller in envs.BASELINES
    if not baseline and not (Path(args.controller) / "manifest.json").is_file():
        parser.error(
            f"--controller {args.controller}: not one of {envs.BASELINES} and not a run dir"
            " with a manifest.json"
        )

    rows = [
        {
            "controller": args.controller,
            "checkpoint": None if baseline else "final",
            "split": args.split,
            **s,
            "season_length": SEASON_LENGTH,
            **_score(args.controller, s),
        }
        for s in scenarios.SPLITS[args.split]
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as f:
        pd.DataFrame(rows).to_parquet(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
