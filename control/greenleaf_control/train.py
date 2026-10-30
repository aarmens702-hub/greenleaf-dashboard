"""Training: one run of one method and seed, smoke or registered (docs/API-CONTRACT.md).

The stack is SubprocVecEnv (DummyVecEnv for one env) -> VecMonitor -> VecLagrangianReward
(residual_lag only) -> VecNormalize -> PPO on cpu, with configs/ppo.yml as written except
log_std_init = -1.0 for the residual arms (PROTOCOL §3). Checkpoints are saved with their
VecNormalize stats every CHECKPOINT_EVERY steps and at the end (PROTOCOL §5); then the end-of-training
model plays one deterministic PAPER season into final_eval.json.

- --smoke caps a run at 20,000 steps and a 2-day season, writes
  <runs-dir>/smoke/<method>-seed<S>-<UTC stamp>-<hex> and records no ledger row.
- --registered <row> runs one cell of that registration (an arm, a seed, its timesteps) into
  <runs-dir>/<row>/<arm>/seed<S> and records its run and run-end rows.
- --registered <row> --remote '{"stamp": ..., "registered": <row>}' is the same run where there's
  no git or ledger: the manifest records the passed stamp and the Mac writes the ledger rows. It's
  refused wherever manifest.REPO_ROOT holds a .git, so a checkout can't skip the dirty-tree check.

build_vec_env and build_model are the one way to build a run's envs and model; the probes (S1-9,
S1-10) will use them.
"""

import argparse
import copy
import functools
import inspect
import json
import math
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from greenleaf_control import envs, ledger, manifest, rule_based, scenarios
from greenleaf_control.lagrangian import (
    FixedMultiplier,
    LagrangianCallback,
    PIDLagrangian,
    VecLagrangianReward,
)
from greenleaf_control.rewards import constraint_costs

RUNS = Path(__file__).resolve().parents[1] / "runs"
CHECKPOINT_EVERY = 250_000  # PROTOCOL §5; read at call time, so tests can patch it
SMOKE_TIMESTEPS = 20_000
SMOKE_DAYS = 2
RESIDUAL = ("residual", "residual_lag")
RESIDUAL_LOG_STD_INIT = -1.0  # PROTOCOL §3: std 0.37 for the residual arms
# SB3's own defaults, stated so the manifest records them; gamma comes from configs/ppo.yml.
VECNORMALIZE = {"norm_obs": True, "norm_reward": True, "clip_obs": 10.0, "clip_reward": 10.0}
# make_env's co-pilot scale (PROTOCOL §3), recorded in the manifest, not set here.
RESIDUAL_SCALE = inspect.signature(envs.make_env).parameters["residual_scale"].default
# The run-shaping flags' CLI defaults; a registered run must be at one its arm doesn't name.
SEASON_LENGTH, N_ENVS, PENALTY_WEIGHT, RB = 60, 4, 1.0, "tuned"
# configs/ppo.yml's policy_kwargs strings, mapped explicitly; any other string raises.
FROM_YML = {
    "optimizer_class": {"adam": torch.optim.Adam},
    "activation_fn": {"silu": torch.nn.SiLU},
    "log_std_init": {"np.log(1)": 0.0},
}


def _ppo_yml() -> dict:
    """configs/ppo.yml's GreenLightEnv block, as written."""
    return yaml.safe_load((rule_based.CONFIGS / "ppo.yml").read_text())["GreenLightEnv"]


def _from_yml(key: str, value):
    """A policy_kwargs value from ppo.yml, with its string (if any) mapped by FROM_YML."""
    if not isinstance(value, str):
        return value
    table = FROM_YML.get(key, {})
    if value not in table:
        raise ValueError(f"configs/ppo.yml policy_kwargs {key}: no mapping for {value!r}")
    return table[value]


def build_vec_env(
    config: dict, n_envs: int, *, scenario: dict | None = None, season_length: int | None = None
) -> VecNormalize:
    """The run's env stack, env i = make_env(method, seed=seed + i), VecNormalize training.

    scenario and season_length override the config's weather and season (for evaluation).
    """
    # The residual arms' recorded scale; the other arms don't use one, so make_env's default stands.
    scale = config.get("residual_scale")
    fns = [
        functools.partial(
            envs.make_env,
            config["method"],
            seed=config["seed"] + i,
            scenario=scenario,
            season_length=config["season_length"] if season_length is None else season_length,
            penalty_weight=config["penalty_weight"],
            rb_params=config.get("rb_params"),
            **({} if scale is None else {"residual_scale": scale}),
        )
        for i in range(n_envs)
    ]
    venv = VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))
    if config["method"] == "residual_lag":
        venv = VecLagrangianReward(venv)
    return VecNormalize(venv, gamma=_ppo_yml()["gamma"], **VECNORMALIZE)


def build_model(config: dict, venv) -> PPO:
    """configs/ppo.yml's PPO on venv, on cpu, seeded; only log_std_init changes (residual arms)."""
    kwargs = _ppo_yml()
    for key in ("total_timesteps", "n_envs", "policy"):
        del kwargs[key]
    policy_kwargs = {k: _from_yml(k, v) for k, v in kwargs.pop("policy_kwargs").items()}
    if config["method"] in RESIDUAL:
        policy_kwargs["log_std_init"] = RESIDUAL_LOG_STD_INIT
    run_dir = config.get("run_dir")
    return PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        seed=config["seed"],
        device="cpu",
        tensorboard_log=str(Path(run_dir) / "tb") if run_dir else None,
        **kwargs,
    )


class _RunCallback(BaseCallback):
    """Saves ckpt_<steps>.zip + vecnorm_<steps>.pkl whenever num_timesteps crosses a multiple of
    CHECKPOINT_EVERY, counts solver failures, and keeps each PPO update's diagnostics.

    SB3 dumps the logger before each update, so an update's train/* values are still there at the
    next rollout's start and at the end of training; each update is read once.
    """

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.solver_failures = 0
        self.approx_kls: list[float] = []
        self.entropy: float | None = None
        self.explained_variance: float | None = None
        self._last = 0

    def _on_step(self) -> bool:
        self.solver_failures += sum(bool(info["solver_failure"]) for info in self.locals["infos"])
        every = CHECKPOINT_EVERY
        if self.num_timesteps // every > self._last // every:
            self.model.save(self.run_dir / f"ckpt_{self.num_timesteps}.zip")
            vecnorm = self.model.get_vec_normalize_env()
            vecnorm.save(self.run_dir / f"vecnorm_{self.num_timesteps}.pkl")
        self._last = self.num_timesteps
        return True

    def _read_update(self) -> None:
        values = self.logger.name_to_value  # a defaultdict: `in` doesn't insert
        if "train/approx_kl" in values:
            self.approx_kls.append(float(values["train/approx_kl"]))
            self.entropy = -float(values["train/entropy_loss"])
            self.explained_variance = float(values["train/explained_variance"])

    def _on_rollout_start(self) -> None:
        self._read_update()

    def _on_training_end(self) -> None:
        self._read_update()

    def diagnostics(self) -> dict:
        """The run-end's diagnostics; None where there's no finite value (or no update at all)."""
        return {
            "min_approx_kl": min(filter(math.isfinite, self.approx_kls), default=None),
            "final_entropy": _finite(self.entropy),
            "final_explained_variance": _finite(self.explained_variance),
        }


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _budgets(rb: str, budget_frac: float, season_length: int) -> list[float]:
    """budget_frac x rb_<rb>'s mean per-step cost over TRAIN_EVAL (pooled), in CONSTRAINTS order."""
    costs = np.array(
        [
            constraint_costs(info)
            for s in scenarios.TRAIN_EVAL
            for info in envs.run_baseline(f"rb_{rb}", s, season_length)
        ]
    )
    return [float(budget_frac * c) for c in costs.mean(axis=0)]


def _final_eval(model: PPO, trained: VecNormalize, config: dict, run_dir: Path) -> None:
    """One deterministic PAPER season with the in-memory model and frozen copies of its stats."""
    venv = build_vec_env(config, 1, scenario=scenarios.PAPER)
    venv.obs_rms = copy.deepcopy(trained.obs_rms)
    venv.ret_rms = copy.deepcopy(trained.ret_rms)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    paper_return, steps, done = 0.0, 0, False
    while not done:  # DummyVecEnv resets itself when the season ends
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        paper_return += infos[0]["paper_reward"]
        steps += 1
        done = bool(dones[0])
    venv.close()
    result = {
        "scenario": scenarios.PAPER,
        "season_length": config["season_length"],
        "checkpoint": "final",
        "paper_return": float(paper_return),
        "steps": steps,
    }
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    (run_dir / "final_eval.json").write_text(text + "\n")


def _train(config: dict, run_dir: Path, callback: _RunCallback) -> None:
    """Build, learn, save the final checkpoint and write final_eval.json."""
    venv = build_vec_env(config, config["n_envs"])
    try:
        model = build_model(config, venv)
        callbacks = [callback]
        if config["method"] == "residual_lag":
            # A zero budget gets a fixed multiplier of 1, where a PID's lambda would pin at its cap.
            pids = [PIDLagrangian(b) if b > 0 else FixedMultiplier(1.0) for b in config["budgets"]]
            # The VecLagrangianReward itself, which build_vec_env puts right under VecNormalize.
            callbacks.append(LagrangianCallback(venv.venv, pids, run_dir / "lambdas.csv"))
        model.learn(config["timesteps"], callback=callbacks)
        model.save(run_dir / "ckpt_final.zip")
        venv.save(run_dir / "vecnorm_final.pkl")
        _final_eval(model, venv, config, run_dir)
    finally:
        venv.close()


def _registration(
    row_id: str, config: dict, ledger_path: Path, remote: dict | None
) -> tuple[dict, dict]:
    """The registered row and the one arm this run is; refuses a run the row doesn't hold.

    An arm is its method and its config; a run-shaping flag the arm doesn't name must be where
    leaving the flag out puts it. With --remote, the row it carries stands in for the ledger.
    """
    if remote is None:
        registered, source = ledger.rows(kind="registered", path=ledger_path), ledger_path
    else:
        registered, source = [remote["registered"]], "--remote"
    found = [r for r in registered if r["row_id"] == row_id]
    if not found:
        raise ValueError(f"--registered {row_id}: no registered row with that id in {source}")
    [row] = found
    method = config["method"]
    unflagged = {
        "season_length": SEASON_LENGTH,
        "n_envs": N_ENVS,
        "rb": RB if method in RESIDUAL else None,  # --rb is ignored outside the residual arms
        "penalty_weight": 0.0 if method == "residual_lag" else PENALTY_WEIGHT,  # lag forces w = 0
        "budget_frac": None,  # residual_lag requires --budget-frac, so its arm must name it
    }
    same = [a for a in row["arms"] if a["method"] == method]
    differs = [
        [
            f"{k} = {config.get(k)!r}, not {v!r}"
            for k, v in {**unflagged, **a["config"]}.items()
            if k not in config or config[k] != v
        ]
        for a in same
    ]
    arms = [a for a, keys in zip(same, differs, strict=True) if not keys]
    if len(arms) != 1:
        why = "".join(
            f"; arm {a['name']} differs on {', '.join(keys)}"
            for a, keys in zip(same, differs, strict=True)
            if keys
        )
        raise ValueError(
            f"{row_id}: {len(arms)} arms match method {method} and this run's flags (a flag an arm"
            f" doesn't name must be at its CLI default){why}; a registered run must be exactly one"
            f" arm of {row['arms']}"
        )
    if config["seed"] not in row["seeds"]:
        raise ValueError(f"{row_id}: seed {config['seed']} is not in its seeds {row['seeds']}")
    if config["timesteps"] != row["timesteps"]:
        raise ValueError(
            f"{row_id}: --timesteps {config['timesteps']} differs from its timesteps"
            f" {row['timesteps']}"
        )
    return row, arms[0]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.train")
    p.add_argument("--method", choices=envs.METHODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--penalty-weight",
        type=float,
        help=f"w on the summed costs (default {PENALTY_WEIGHT}); not residual_lag",
    )
    p.add_argument(
        "--budget-frac",
        type=float,
        help="residual_lag only, required: budget = this x the RB's mean per-step cost on TRAIN_EVAL",
    )
    timesteps = _ppo_yml()["total_timesteps"]
    p.add_argument(
        "--timesteps", type=int, default=timesteps, help=f"default {timesteps:,} (configs/ppo.yml)"
    )
    p.add_argument(
        "--season-length", type=int, default=SEASON_LENGTH, help=f"days (default {SEASON_LENGTH})"
    )
    p.add_argument("--n-envs", type=int, default=N_ENVS, help=f"default {N_ENVS}")
    p.add_argument("--runs-dir", type=Path, default=RUNS, help="default control/runs")
    p.add_argument(
        "--rb",
        choices=rule_based.NAMES,
        default=RB,
        help=f"the residual arms' RB params (default {RB})",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke", action="store_true", help="20,000 steps, 2-day seasons, no ledger row"
    )
    mode.add_argument("--registered", metavar="ROW", help="the registered row this run belongs to")
    p.add_argument(
        "--remote",
        type=json.loads,
        metavar="JSON",
        help='a remote registered run: {"stamp": <code_stamp>, "registered": <its row>}; no git,'
        " no ledger rows",
    )
    return p


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    lag = args.method == "residual_lag"
    if lag and args.budget_frac is None:
        parser.error("residual_lag needs --budget-frac")
    if not lag and args.budget_frac is not None:
        parser.error(f"--budget-frac is for residual_lag only, not {args.method}")
    if lag and args.penalty_weight is not None:
        parser.error("--penalty-weight is refused for residual_lag: it trains with w = 0")
    if args.smoke and args.remote is not None:
        parser.error("--remote is for a registered run, not --smoke")
    # A file check, not a git call: only the container, which has no checkout, may pass a stamp.
    if args.remote is not None and (manifest.REPO_ROOT / ".git").exists():
        parser.error(
            f"--remote is the container's alone: {manifest.REPO_ROOT} is a git checkout, where a"
            " run stamps its own code (and refuses a dirty tree)"
        )

    residual = args.method in RESIDUAL
    timesteps, season_length = args.timesteps, args.season_length
    if args.smoke:
        timesteps, season_length = min(timesteps, SMOKE_TIMESTEPS), min(season_length, SMOKE_DAYS)
    weight = PENALTY_WEIGHT if args.penalty_weight is None else args.penalty_weight
    config = {
        "method": args.method,
        "seed": args.seed,
        "timesteps": timesteps,
        "season_length": season_length,
        "n_envs": args.n_envs,
        "penalty_weight": 0.0 if lag else weight,
        "budget_frac": args.budget_frac,
        "budgets": None,
        "rb": args.rb if residual else None,
        "rb_params": rule_based.load_rb_params(args.rb) if residual else None,
        "residual_scale": RESIDUAL_SCALE if residual else None,
        "vecnormalize": {**VECNORMALIZE, "gamma": _ppo_yml()["gamma"]},
    }
    if args.smoke:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{args.method}-seed{args.seed}-{stamp}-{secrets.token_hex(4)}"
        run_dir, row, arm = args.runs_dir / "smoke" / name, None, None
    else:
        row, arm = _registration(args.registered, config, ledger_path, args.remote)
        run_dir = args.runs_dir / row["row_id"] / arm["name"] / f"seed{args.seed}"
    config["run_dir"] = str(run_dir)
    if lag:
        config["budgets"] = _budgets(args.rb, args.budget_frac, season_length)

    manifest.write_manifest(
        run_dir,
        config=config,
        issue=0 if row is None else row["issue"],
        registered=None if row is None else row["row_id"],
        smoke=args.smoke,
        stamp=None if args.remote is None else args.remote["stamp"],
    )
    callback = _RunCallback(run_dir)
    start = time.monotonic()
    run, status = None, "crashed"
    try:
        if row is not None and args.remote is None:  # a remote run's rows are the Mac's to write
            run = ledger.run_start(
                registered=row["row_id"],
                arm=arm["name"],
                seed=args.seed,
                run_dir=str(run_dir),
                path=ledger_path,
            )
        _train(config, run_dir, callback)
        status = "finished"
    finally:  # a crash is recorded in the manifest and the run-end, then propagates
        end = {
            "status": status,
            "wall_s": time.monotonic() - start,
            "timesteps": callback.num_timesteps,
            "solver_failures": callback.solver_failures,
            "cost_usd": None,
            "diagnostics": callback.diagnostics(),
        }
        manifest.finish_manifest(run_dir, **end)
        if run is not None:
            ledger.run_end(run=run["row_id"], **end, path=ledger_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
