"""Tests for greenleaf_control.train (issue #13, part 13a): smoke training for every method, the env
stack and the model it trains, and its refusals.

train is the one way a run is made (docs/API-CONTRACT.md). Its stack is DummyVecEnv (one env) or
SubprocVecEnv (more), then VecMonitor, then VecLagrangianReward (residual_lag only), then
VecNormalize, then PPO on cpu with configs/ppo.yml unchanged except log_std_init = -1.0 for the
residual arms (PROTOCOL §3). --smoke caps a run at 20,000 steps and a 2-day season, writes
<runs-dir>/smoke/<method>-seed<S>-<UTC stamp>-<hex> and records no ledger row. Without --smoke a
run needs --registered <row>: its arm (method and config), seed and timesteps must be in that
registration, a run-shaping flag its arm doesn't name must be at its CLI default, and it writes
<runs-dir>/<row>/<arm>/seed<S>, which must not exist yet.

Issue #19 (part 19a): a remote run gets --remote '{"stamp": <manifest.code_stamp>, "registered":
<row>}' and reads neither git nor the ledger; its manifest takes the stamp verbatim and, like every
registered run's, carries the run's diagnostics. Part 19b: only the container may pass --remote,
so train refuses it where manifest.REPO_ROOT holds a .git (a checkout).

Known answers used throughout:
- One day is 96 steps (86,400 s / 900 s), so --season-length 1 is a 96-step season and 2 is 192.
- With --n-envs 1 a PPO rollout is n_steps = 2,048 env steps, so --timesteps 2048 is exactly one
  rollout and one update, and every checkpoint step is hit exactly.
- configs/rule_based_tuned.yml doesn't exist until E1, so every run here passes --rb default.

Nothing here may write the real ledger or control/runs/. Every run gets --runs-dir and ledger_path
under tmp_path, every run that could go non-smoke gets a temp git repo as manifest.REPO_ROOT, an
autouse fixture refuses any ledger append to the real file, and the last test checks the real
ledger and control/runs/ are as they were when this module was collected.
"""

import csv
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import gl_gym.environments  # noqa: F401  (first: gl_gym.components.weather alone is circular)
import numpy as np
import pytest
import torch
import yaml
from gl_gym.components.weather import FixedWeatherSampler, RandomWeatherSampler
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnvWrapper,
    VecMonitor,
    VecNormalize,
)

from greenleaf_control import envs, ledger, manifest, rule_based, scenarios
from greenleaf_control.lagrangian import LagrangianCallback, PIDLagrangian, VecLagrangianReward
from greenleaf_control.rewards import CONSTRAINTS, constraint_costs

# greenleaf_control.train is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger and runs dir, read when this module is collected, before any test in it runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)

METHODS = ("ppo_paper", "ppo", "residual", "residual_lag")
RESIDUAL = ("residual", "residual_lag")
N_DAY = 96  # 86,400 s / 900 s
DIMS = {"ppo_paper": 22, "ppo": 23, "residual": 29, "residual_lag": 29}  # as in test_envs.py
RB_DEFAULT = rule_based.load_rb_params("default")
# configs/ppo.yml's GreenLightEnv block, copied by hand, minus total_timesteps, n_envs and policy
# (train's own) and policy_kwargs (below). The test checks the file against it, then the model.
PPO_YML = {
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 8,
    "gamma": 0.9631,
    "gae_lambda": 0.9167,
    "clip_range": 0.2,
    "normalize_advantage": True,
    "ent_coef": 0.05434,
    "vf_coef": 0.8225,
    "max_grad_norm": 0.3,
    "use_sde": False,
    "sde_sample_freq": 8,
    "target_kl": None,
    "learning_rate": 2e-5,
}
PPO_YML_POLICY_KWARGS = {
    "net_arch": {"pi": [256, 256, 256], "vf": [512, 512, 512]},
    "optimizer_class": "adam",
    "optimizer_kwargs": {"amsgrad": True},
    "activation_fn": "silu",
    "log_std_init": "np.log(1)",
}
# The same policy_kwargs with the strings mapped: adam -> torch.optim.Adam, silu -> torch.nn.SiLU.
POLICY_KWARGS = {
    "net_arch": {"pi": [256, 256, 256], "vf": [512, 512, 512]},
    "optimizer_class": torch.optim.Adam,
    "optimizer_kwargs": {"amsgrad": True},
    "activation_fn": torch.nn.SiLU,
}
# np.log(1) = 0.0 (std 1); PROTOCOL §3 sets -1.0 (std 0.37) for the residual arms.
LOG_STD_INIT = {"ppo_paper": 0.0, "ppo": 0.0, "residual": -1.0, "residual_lag": -1.0}
FINAL_EVAL_KEYS = {"scenario", "season_length", "checkpoint", "paper_return", "steps"}
CLI_FLAGS = [
    "--method",
    "--seed",
    "--penalty-weight",
    "--budget-frac",
    "--timesteps",
    "--season-length",
    "--n-envs",
    "--runs-dir",
    "--smoke",
    "--registered",
    "--rb",
    "--remote",  # #19a: only train_remote sets it
]
# One arm, one seed, 2,048 steps: what the registered tests run, or refuse to. The arm names every
# run-shaping flag the registered argv sets away from its CLI default (--rb is ignored for ppo).
REGISTRATION = {
    "issue": 13,
    "hypothesis": "plumbing: a registered run goes end to end",
    "arms": [
        {
            "name": "a",
            "method": "ppo",
            "config": {"penalty_weight": 1.0, "season_length": 1, "n_envs": 1},
        }
    ],
    "seeds": [0],
    "timesteps": 2048,
    "scenarios": ["validation"],
    "primary": "profit_eur_m2",
    "guards": ["viol_hours_temp"],
    "rule": "none: a test registration",
    "budget": {"cpu_hours": 0.01, "usd": 0.0, "backend": "local"},
}


@pytest.fixture(autouse=True)
def _never_the_real_ledger(monkeypatch):
    """A ledger call that forgets path= would append to the real ledger; this refuses it."""
    real_append = ledger.append

    def append(row: dict, path: Path = ledger.LEDGER) -> None:
        if Path(path).resolve() == REAL_LEDGER.resolve():
            raise AssertionError(f"a test tried to append a {row['kind']} row to the real ledger")
        real_append(row, path)

    monkeypatch.setattr(ledger, "append", append)


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], check=True, cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(repo: Path) -> None:
    """Stage and commit everything. CI has no git identity and may sign commits globally."""
    _git(repo, "add", ".")
    _git(
        repo,
        *("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"),
        *("commit", "-q", "-m", "commit"),
    )


def _make_repo(tmp_path: Path, monkeypatch) -> Path:
    """A temp git repo with one committed file, control/README.md, set as manifest.REPO_ROOT.

    runs/ and the ledger live beside it under tmp_path, so a run leaves its tree clean.
    """
    # A git hook's environment would point these git calls at the real repo.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    (repo / "control").mkdir(parents=True)
    (repo / "control" / "README.md").write_text("temp repo\n")
    _git(repo, "init", "-q")
    _commit(repo)
    monkeypatch.setattr(manifest, "REPO_ROOT", repo)
    return repo


def _smoke_argv(method: str, runs: Path, *extra: str) -> list[str]:
    """A 2,048-step smoke run of method, seed 0, one env, 1-day seasons; extra flags win."""
    lag = ["--budget-frac", "0.5"] if method == "residual_lag" else []
    return [
        *("--method", method),
        *("--seed", "0"),
        "--smoke",
        *("--runs-dir", str(runs)),
        *("--n-envs", "1"),
        *("--timesteps", "2048"),
        *("--season-length", "1"),
        *("--rb", "default"),
        *lag,
        *extra,
    ]


def _registered_argv(row_id: str, runs: Path, **flags: str | None) -> list[str]:
    """Arm a, seed 0, as REGISTRATION holds it; flags (e.g. seed="1") replace a flag's value, and
    a None drops the flag."""
    values = {
        "method": "ppo",
        "seed": "0",
        "penalty_weight": "1.0",
        "timesteps": "2048",
        "season_length": "1",
        "n_envs": "1",
        "rb": "default",
        "runs_dir": str(runs),
        "registered": row_id,
        **flags,
    }
    return [
        arg
        for key, value in values.items()
        if value is not None
        for arg in (f"--{key.replace('_', '-')}", value)
    ]


def _registered(tmp_path: Path, monkeypatch, **over) -> tuple[Path, dict]:
    """A clean temp repo as REPO_ROOT and a tmp ledger holding REGISTRATION, with over replacing
    its fields (e.g. timesteps=4096): (ledger path, row)."""
    _make_repo(tmp_path, monkeypatch)
    path = tmp_path / "ledger.jsonl"
    return path, ledger.register(**{**REGISTRATION, **over}, path=path)


def _config(method: str, tmp_path: Path, **over) -> dict:
    """A run's config as train records it in manifest.json, which build_vec_env and build_model read.

    residual_lag's weight is 0.0 (make_env forces it; --penalty-weight is refused for it). run_dir
    is there for build_model's TensorBoard logs (<run_dir>/tb).
    """
    return {
        "method": method,
        "seed": 0,
        "penalty_weight": 0.0 if method == "residual_lag" else 1.0,
        "budget_frac": 0.5 if method == "residual_lag" else None,
        "timesteps": 2048,
        "season_length": 1,
        "n_envs": 1,
        "rb": "default",
        "rb_params": RB_DEFAULT,
        "run_dir": str(tmp_path / "run"),
        **over,
    }


def _smoke_dir(runs: Path) -> Path:
    """The one run dir a smoke run made; nothing else is under runs/."""
    assert sorted(p.name for p in runs.iterdir()) == ["smoke"]
    [run_dir] = list((runs / "smoke").iterdir())
    return run_dir


def _files(root: Path) -> list[Path]:
    """Every file under root; none if root doesn't exist."""
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


def _chain(venv) -> list:
    """The VecEnv stack from the outside in, ending at the DummyVecEnv or SubprocVecEnv."""
    chain = [venv]
    while isinstance(chain[-1], VecEnvWrapper):
        chain.append(chain[-1].venv)
    return chain


def _refusal(excinfo, capsys) -> str:
    """A refusal's text: a ValueError's message, or a SystemExit's code plus argparse's stderr."""
    if isinstance(excinfo.value, SystemExit):
        assert excinfo.value.code not in (0, None), "a refusal must exit nonzero"
    return str(excinfo.value) + capsys.readouterr().err


def _no_learning(monkeypatch) -> dict:
    """PPO.learn becomes a no-op that records the steps it was asked for and the training envs'
    season length (N), and sets num_timesteps as if it had trained them."""
    seen = {}

    def learn(self, total_timesteps, *args, **kwargs):
        seen["total_timesteps"] = total_timesteps
        seen["N"] = [env.N for env in self.get_env().get_attr("unwrapped")]
        self.num_timesteps = total_timesteps
        return self

    monkeypatch.setattr(PPO, "learn", learn)
    return seen


def _assert_lag_budgets_and_pids(config: dict, lambdas_csv: Path) -> None:
    """residual_lag's budgets are budget_frac x rb_default's mean per-step cost over TRAIN_EVAL,
    stored as config["budgets"] in CONSTRAINTS order; a nonzero budget gets a PID with the contract
    defaults, and a zero budget a fixed multiplier of 1 (issue #38).

    The budgets are recomputed here from envs.run_baseline and rewards.constraint_costs: 9 seasons
    of 96 steps, pooled (equal lengths, so a mean of per-season means agrees). Known answer for the
    multipliers: on a nonzero budget, a fresh PIDLagrangian(budget) (kp 0.5, ki 0.05, kd 0.1,
    lambda_max 100) fed the one rollout's logged mean cost gives the logged lambda; on a zero
    budget (co2's here: rb_default has no co2 cost on these 1-day seasons) the logged lambda is
    1.0 whatever the cost, where a PIDLagrangian(0.0) gives 0 at zero cost and
    min(100, 0.65 x cost / 1e-8) at a positive one (e = cost / 1e-8; I + kp*e + kd*d = 0.65e).
    """
    episodes = [envs.run_baseline("rb_default", s, 1) for s in scenarios.TRAIN_EVAL]
    assert [len(e) for e in episodes] == [N_DAY] * 9
    costs = np.array([constraint_costs(info) for e in episodes for info in e])  # (864, 3)
    assert config["budget_frac"] == 0.5
    np.testing.assert_allclose(config["budgets"], 0.5 * costs.mean(axis=0), rtol=1e-9, atol=0)

    with lambdas_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # 2,048 steps is one rollout, so one rollout end
    assert int(rows[0]["timesteps"]) == 2048
    for c, budget in zip(CONSTRAINTS, config["budgets"], strict=True):
        cost = float(rows[0][f"cost_{c}"])
        expected = PIDLagrangian(budget).update(cost) if budget > 0 else 1.0
        assert float(rows[0][f"lambda_{c}"]) == pytest.approx(expected, rel=1e-9, abs=1e-12), c


@pytest.mark.parametrize("method", METHODS)
def test_smoke_run_writes_every_output_under_runs_smoke(method, tmp_path):
    """train --smoke, for each method (Done when): one run dir, runs/smoke/<method>-seed0-<stamp>-<hex>,
    holding manifest.json (smoke, finished), ckpt_final.zip, vecnorm_final.pkl, final_eval.json,
    TensorBoard events under tb/ and, for residual_lag only, lambdas.csv. No ledger row is written.

    Known answers: 2,048 steps is one rollout; the end-of-training evaluation is one deterministic
    season on PAPER at the run's 1-day season, so 96 steps. The residual arms record the RB params
    they resolved; residual_lag records its budgets, which its PIDs use.
    """
    import greenleaf_control.train as train

    runs, path = tmp_path / "runs", tmp_path / "ledger.jsonl"
    assert train.main(_smoke_argv(method, runs), ledger_path=path) == 0

    run_dir = _smoke_dir(runs)
    assert re.fullmatch(rf"{re.escape(method)}-seed0-.+-[0-9a-f]+", run_dir.name), run_dir.name
    for name in ("manifest.json", "ckpt_final.zip", "vecnorm_final.pkl", "final_eval.json"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "lambdas.csv").is_file() == (method == "residual_lag")
    assert any((run_dir / "tb").rglob("events.out.tfevents.*"))
    assert not path.exists() or path.read_text() == ""

    m = json.loads((run_dir / "manifest.json").read_text())
    assert (m["smoke"], m["registered"], m["issue"]) == (True, None, 0)
    assert (m["status"], m["timesteps"]) == ("finished", 2048)
    config = m["config"]
    assert (config["method"], config["seed"]) == (method, 0)
    assert (config["timesteps"], config["season_length"], config["n_envs"]) == (2048, 1, 1)
    if method in RESIDUAL:
        assert (config["rb"], config["rb_params"]) == ("default", RB_DEFAULT)

    final = json.loads((run_dir / "final_eval.json").read_text())
    assert set(final) == FINAL_EVAL_KEYS
    assert final["scenario"] == scenarios.PAPER
    assert (final["season_length"], final["checkpoint"], final["steps"]) == (1, "final", N_DAY)
    assert isinstance(final["paper_return"], float) and math.isfinite(final["paper_return"])

    if method == "residual_lag":
        _assert_lag_budgets_and_pids(config, run_dir / "lambdas.csv")


def test_smoke_caps_timesteps_at_20000_and_the_season_at_2_days(tmp_path, monkeypatch):
    """--smoke --timesteps 50000 --season-length 10 trains 20,000 steps on 2-day seasons (192 steps)
    and records both in the manifest config; the end-of-training evaluation is 2 days too.

    PPO.learn is a no-op here (recording what it was asked), so nothing trains 20,000 steps.
    """
    import greenleaf_control.train as train

    seen = _no_learning(monkeypatch)
    runs = tmp_path / "runs"
    argv = _smoke_argv("ppo", runs, "--timesteps", "50000", "--season-length", "10")
    assert train.main(argv, ledger_path=tmp_path / "ledger.jsonl") == 0

    assert seen == {"total_timesteps": 20_000, "N": [2 * N_DAY]}
    run_dir = _smoke_dir(runs)
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    assert (config["timesteps"], config["season_length"]) == (20_000, 2)
    final = json.loads((run_dir / "final_eval.json").read_text())
    assert (final["season_length"], final["steps"]) == (2, 2 * N_DAY)


def test_two_smoke_runs_get_two_run_dirs(tmp_path, monkeypatch):
    """A smoke run never reuses a run dir (PROTOCOL §5: fresh runs only): the same command twice
    makes two run dirs, each with its own manifest. PPO.learn is a no-op here."""
    import greenleaf_control.train as train

    _no_learning(monkeypatch)
    runs = tmp_path / "runs"
    for _ in range(2):
        assert train.main(_smoke_argv("ppo", runs), ledger_path=tmp_path / "ledger.jsonl") == 0

    run_dirs = sorted((runs / "smoke").iterdir())
    assert len(run_dirs) == 2
    assert all((d / "manifest.json").is_file() for d in run_dirs)


def test_checkpoints_every_checkpoint_every_steps_each_with_its_vecnormalize(tmp_path, monkeypatch):
    """CHECKPOINT_EVERY is 250,000 (PROTOCOL §5). Patched to 1,024, a 2,048-step run saves ckpt_1024
    and ckpt_2048, each with vecnorm_<steps>.pkl, then ckpt_final and vecnorm_final.

    Known answers (one env): the model in ckpt_<k> has trained k steps (num_timesteps = k), and the
    VecNormalize saved with it has seen the first reset plus k steps: obs_rms.count = 1e-4 + 1 + k
    (abs 2 allows one more reset).
    """
    import greenleaf_control.train as train

    assert train.CHECKPOINT_EVERY == 250_000
    monkeypatch.setattr(train, "CHECKPOINT_EVERY", 1024)
    runs = tmp_path / "runs"
    assert train.main(_smoke_argv("ppo", runs), ledger_path=tmp_path / "ledger.jsonl") == 0

    run_dir = _smoke_dir(runs)
    saved = {p.name for p in run_dir.iterdir() if p.name.startswith(("ckpt_", "vecnorm_"))}
    assert saved == {
        name
        for tag in ("1024", "2048", "final")
        for name in (f"ckpt_{tag}.zip", f"vecnorm_{tag}.pkl")
    }
    venv = DummyVecEnv([lambda: envs.make_env("ppo", seed=0, season_length=1)])
    for tag, steps in (("1024", 1024), ("2048", 2048), ("final", 2048)):
        assert PPO.load(run_dir / f"ckpt_{tag}.zip", device="cpu").num_timesteps == steps, tag
        vecnorm = VecNormalize.load(str(run_dir / f"vecnorm_{tag}.pkl"), venv)
        assert vecnorm.obs_rms.count == pytest.approx(1e-4 + 1 + steps, abs=2), tag


CLI_REFUSALS = [
    pytest.param(["--method", "ppo"], "--registered", id="neither-smoke-nor-registered"),
    pytest.param(
        ["--method", "ppo", "--smoke", "--registered", "registered-0badc0de"],
        "--smoke",
        id="both-smoke-and-registered",
    ),
    pytest.param(
        ["--method", "ppo", "--smoke", "--budget-frac", "0.5"], "--budget-frac", id="frac-on-ppo"
    ),
    pytest.param(
        ["--method", "residual", "--smoke", "--budget-frac", "0.5"],
        "--budget-frac",
        id="frac-on-residual",
    ),
    pytest.param(
        ["--method", "residual_lag", "--smoke"], "--budget-frac", id="residual-lag-without-frac"
    ),
    pytest.param(
        ["--method", "residual_lag", "--smoke", "--budget-frac", "0.5", "--penalty-weight", "1.0"],
        "--penalty-weight",
        id="penalty-weight-on-residual-lag",
    ),
    pytest.param(["--method", "dqn", "--smoke"], "dqn", id="unknown-method"),
    pytest.param(["--method", "residual", "--smoke", "--rb", "best"], "best", id="unknown-rb"),
]


@pytest.mark.parametrize(("flags", "named"), CLI_REFUSALS)
def test_command_line_refusals(flags, named, tmp_path, monkeypatch, capsys):
    """Done when: train without --smoke and without --registered refuses. So do both together,
    --budget-frac on any arm but residual_lag, residual_lag without it, --penalty-weight on
    residual_lag (it forces w = 0), an unknown method and an unknown --rb.

    Each raises SystemExit (argparse, nonzero) or ValueError naming its rule, before anything is
    written: no file under runs/ and no ledger.
    """
    import greenleaf_control.train as train

    _make_repo(tmp_path, monkeypatch)
    runs, path = tmp_path / "runs", tmp_path / "ledger.jsonl"
    argv = [
        *("--seed", "0"),
        *("--timesteps", "2048"),
        *("--season-length", "1"),
        *("--n-envs", "1"),
        *("--rb", "default"),
        *("--runs-dir", str(runs)),
        *flags,
    ]

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        train.main(argv, ledger_path=path)
    assert named in _refusal(excinfo, capsys)
    assert _files(runs) == []
    assert not path.exists()


REGISTERED_REFUSALS = [
    pytest.param({"registered": "registered-deadbeef"}, "registered-deadbeef", id="no-such-row"),
    pytest.param({"method": "residual"}, "arm", id="method-is-no-arm"),
    pytest.param({"penalty_weight": "2.0"}, "arm", id="config-differs-from-the-arm"),
    pytest.param({"seed": "1"}, "seed", id="seed-not-registered"),
    pytest.param({"timesteps": "4096"}, "timesteps", id="timesteps-differ"),
]


@pytest.mark.parametrize(("flags", "named"), REGISTERED_REFUSALS)
def test_registered_run_refuses_what_its_registration_does_not_hold(
    flags, named, tmp_path, monkeypatch, capsys
):
    """A registered run must be one cell of its registration: an arm whose method and every config
    value match the flags, a registered seed, the registered timesteps. REGISTRATION holds arm a =
    ppo at penalty weight 1.0 on 1-day seasons, seed 0, 2,048 steps.

    Each mismatch raises naming its rule, before any run dir or run row: the ledger holds only the
    registration.
    """
    import greenleaf_control.train as train

    path, row = _registered(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        train.main(_registered_argv(row["row_id"], runs, **flags), ledger_path=path)
    assert named in _refusal(excinfo, capsys).lower()
    assert _files(runs) == []
    assert ledger.rows(path=path) == [row]


def test_registered_run_refuses_an_existing_run_dir(tmp_path, monkeypatch):
    """Done when: an existing run dir refuses. <runs>/<row>/a/seed0 already exists, so the run
    raises FileExistsError naming it (from write_manifest), leaves it as it was and adds no run row
    (PROTOCOL §5: a run dir is written once and never resumed)."""
    import greenleaf_control.train as train

    path, row = _registered(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    run_dir = runs / row["row_id"] / "a" / "seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "notes.txt").write_text("an earlier run\n")

    with pytest.raises(FileExistsError, match=re.escape(str(run_dir))):
        train.main(_registered_argv(row["row_id"], runs), ledger_path=path)
    assert [p.name for p in run_dir.iterdir()] == ["notes.txt"]
    assert (run_dir / "notes.txt").read_text() == "an earlier run\n"
    assert ledger.rows(path=path) == [row]


def test_registered_run_records_its_run_and_run_end_rows(tmp_path, monkeypatch):
    """A registered 2,048-step run of arm a, seed 0, from a clean temp repo: it returns 0, writes
    <runs>/<row>/a/seed0 with a finished, clean, non-smoke manifest naming the registration and its
    issue, and appends a run row, then a finished run-end row. No smoke dir.

    Known answers: the run-end's timesteps are the registered 2,048; a local run costs None; one
    PPO update logs a finite approx-KL, entropy and explained variance. The manifest carries the
    same diagnostics as the run-end (#19a: a remote run's run-end is written from its manifest).
    """
    import greenleaf_control.train as train

    path, row = _registered(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    assert train.main(_registered_argv(row["row_id"], runs), ledger_path=path) == 0

    run_dir = runs / row["row_id"] / "a" / "seed0"
    reg, run, end = ledger.rows(path=path)
    assert reg == row
    assert (run["kind"], run["registered"], run["arm"], run["seed"]) == (
        "run",
        row["row_id"],
        "a",
        0,
    )
    assert Path(run["run_dir"]) == run_dir
    assert (end["kind"], end["run"], end["status"], end["timesteps"]) == (
        "run-end",
        run["row_id"],
        "finished",
        2048,
    )
    assert end["cost_usd"] is None
    assert end["wall_s"] > 0
    assert type(end["solver_failures"]) is int and end["solver_failures"] >= 0
    diagnostics = end["diagnostics"]
    assert all(isinstance(v, float) and math.isfinite(v) for v in diagnostics.values()), diagnostics

    m = json.loads((run_dir / "manifest.json").read_text())
    assert (m["registered"], m["smoke"], m["dirty"], m["issue"]) == (
        row["row_id"],
        False,
        False,
        13,
    )
    assert (m["status"], m["timesteps"]) == ("finished", 2048)
    assert m["diagnostics"] == diagnostics
    config = m["config"]
    assert (config["method"], config["seed"]) == ("ppo", 0)
    assert (config["penalty_weight"], config["season_length"]) == (1.0, 1)
    for name in ("ckpt_final.zip", "vecnorm_final.pkl", "final_eval.json"):
        assert (run_dir / name).is_file(), name
    assert not (runs / "smoke").exists()


def test_registered_grid_runs_every_seed_with_the_ledger_inside_the_repo(tmp_path, monkeypatch):
    """Regression for #13 (review, item 1): the ledger is control/ledger/ledger.jsonl, inside the
    repo, so seed 0's run and run-end rows dirtied the tree and seed 1 of the same grid was refused.

    Laid out like this repo: the ledger is tracked (committed with its registration of seeds
    [0, 1]) and run dirs go under control/runs/. Seed 0 then seed 1 both return 0 with clean
    manifests, and the ledger ends registered, run, run-end, run, run-end, each run-end closing
    its run as finished. PPO.learn is a no-op here.
    """
    import greenleaf_control.train as train

    _no_learning(monkeypatch)
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / "control" / "ledger" / "ledger.jsonl"
    path.parent.mkdir()
    row = ledger.register(**{**REGISTRATION, "seeds": [0, 1]}, path=path)
    _commit(repo)
    runs = repo / "control" / "runs"

    for seed in ("0", "1"):
        argv = _registered_argv(row["row_id"], runs, seed=seed)
        assert train.main(argv, ledger_path=path) == 0, f"seed {seed}"

    rows = ledger.rows(path=path)
    assert [r["kind"] for r in rows] == ["registered", "run", "run-end", "run", "run-end"]
    _, run0, end0, run1, end1 = rows
    assert (run0["seed"], run1["seed"]) == (0, 1)
    assert (end0["run"], end1["run"]) == (run0["row_id"], run1["row_id"])
    assert (end0["status"], end1["status"]) == ("finished", "finished")
    for seed in (0, 1):
        m = json.loads((runs / row["row_id"] / "a" / f"seed{seed}" / "manifest.json").read_text())
        assert (m["dirty"], m["status"]) == (False, "finished"), f"seed {seed}"


def _refused_before_training(self, *args, **kwargs):
    """PPO.learn for a run that must be refused: reaching it fails the test."""
    raise AssertionError("PPO.learn was reached: this run should have been refused")


# A one-arm registration whose arm doesn't name `key`, and a run setting that flag away from its
# CLI default; every other flag is the arm's value. Defaults: --season-length 60, --n-envs 4,
# --rb tuned (residual arms only), --penalty-weight 1.0. residual_lag's --budget-frac has no
# default (it is required), so its arm must name budget_frac.
UNLISTED_FLAGS = [
    pytest.param(
        "ppo",
        {"penalty_weight": 1.0, "n_envs": 1},
        {"season_length": "2"},
        "season_length",
        id="season-length-2-not-the-default-60",
    ),
    pytest.param(
        "ppo",
        {"penalty_weight": 1.0, "season_length": 1},
        {"n_envs": "2"},
        "n_envs",
        id="n-envs-2-not-the-default-4",
    ),
    pytest.param(
        "residual",
        {"penalty_weight": 1.0, "season_length": 1, "n_envs": 1},
        {"rb": "default"},
        "rb",
        id="rb-default-not-the-default-tuned",
    ),
    pytest.param(
        "ppo",
        {"season_length": 1, "n_envs": 1},
        {"penalty_weight": "0.5"},
        "penalty_weight",
        id="penalty-weight-0.5-not-the-default-1",
    ),
    pytest.param(
        "residual_lag",
        {"rb": "default", "season_length": 1, "n_envs": 1},
        {"penalty_weight": None, "budget_frac": "0.5"},
        "budget_frac",
        id="budget-frac-not-in-the-arm",
    ),
]


@pytest.mark.parametrize(("method", "arm_config", "flags", "key"), UNLISTED_FLAGS)
def test_registered_run_refuses_an_unlisted_flag_away_from_its_default(
    method, arm_config, flags, key, tmp_path, monkeypatch, capsys
):
    """#13 review: a registered run is one cell of its registration, so a run-shaping flag its arm
    doesn't name must be at its CLI default; otherwise the run isn't the arm that was registered.

    Each raises SystemExit or ValueError whose reason names the key (as season_length or
    --season-length), before any run dir or run row: the ledger holds only the registration.
    """
    import greenleaf_control.train as train

    path, row = _registered(
        tmp_path, monkeypatch, arms=[{"name": "a", "method": method, "config": arm_config}]
    )
    runs = tmp_path / "runs"
    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    argv = _registered_argv(row["row_id"], runs, method=method, **flags)

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        train.main(argv, ledger_path=path)
    reason = _refusal(excinfo, capsys).split("error:")[-1]  # argparse's usage lists every flag
    flag = "--" + key.replace("_", "-")
    assert re.search(rf"\b{key}\b", reason) or flag in reason, reason
    assert _files(runs) == []
    assert ledger.rows(path=path) == [row]


UNLISTED_DEFAULTS = [
    pytest.param(
        "ppo",
        {"season_length": 1, "n_envs": 1},
        {"penalty_weight": None},
        id="ppo-penalty-weight-absent",
    ),
    pytest.param(
        "ppo",
        {"season_length": 1, "n_envs": 1},
        {"penalty_weight": "1.0"},
        id="ppo-penalty-weight-1.0-given-is-the-default",
    ),
    pytest.param(
        "residual_lag",
        {"budget_frac": 0.5, "rb": "default", "season_length": 1, "n_envs": 1},
        {"penalty_weight": None, "budget_frac": "0.5"},
        id="residual-lag-trains-at-w-0-not-the-default-1",
    ),
]


@pytest.mark.parametrize(("method", "arm_config", "flags"), UNLISTED_DEFAULTS)
def test_registered_run_accepts_unlisted_flags_at_their_defaults(
    method, arm_config, flags, tmp_path, monkeypatch
):
    """The other side of the rule: a flag the arm doesn't name is fine at its CLI default, given
    or not. --rb default on a ppo arm is ignored (--rb is for the residual arms), and residual_lag's
    forced penalty weight 0.0 isn't held against the 1.0 default (--penalty-weight is refused for
    it). Each run returns 0 and closes its run row as finished. PPO.learn is a no-op here.
    """
    import greenleaf_control.train as train

    _no_learning(monkeypatch)
    path, row = _registered(
        tmp_path, monkeypatch, arms=[{"name": "a", "method": method, "config": arm_config}]
    )
    argv = _registered_argv(row["row_id"], tmp_path / "runs", method=method, **flags)

    assert train.main(argv, ledger_path=path) == 0
    assert [(r["kind"], r["status"]) for r in ledger.rows(path=path)[1:]] == [
        ("run", "running"),
        ("run-end", "finished"),
    ]


def test_residual_lag_lambdas_reach_the_env_stack_the_model_trains_on(tmp_path, monkeypatch):
    """residual_lag's LagrangianCallback updates the VecLagrangianReward inside the model's own
    stack (right under VecNormalize), not VecNormalize or another copy, so its lambdas reach the
    reward PPO trains on.

    Known answer: a 2,048-step smoke run with one env is one rollout, so the callback's last update
    is lambdas.csv's last row, and the wrapper holds exactly those lambdas when training ends.
    """
    import greenleaf_control.train as train

    made = []

    class Recording(LagrangianCallback):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(train, "LagrangianCallback", Recording)
    runs = tmp_path / "runs"
    argv = _smoke_argv("residual_lag", runs)
    assert train.main(argv, ledger_path=tmp_path / "ledger.jsonl") == 0

    [cb] = made
    assert type(cb.wrapper) is VecLagrangianReward
    assert cb.wrapper is _chain(cb.model.get_env())[1]
    with (_smoke_dir(runs) / "lambdas.csv").open(newline="") as f:
        last = list(csv.DictReader(f))[-1]
    np.testing.assert_array_equal(
        cb.wrapper.lambdas, [float(last[f"lambda_{c}"]) for c in CONSTRAINTS]
    )


def test_residual_lag_gives_a_zero_budget_a_fixed_multiplier_and_the_others_pids(
    tmp_path, monkeypatch
):
    """Issue #38: _train gives each constraint with budget b > 0 a PIDLagrangian(b) and each with
    b == 0 a FixedMultiplier(1.0), in CONSTRAINTS order; the manifest's budgets are unchanged, so
    co2's still shows 0.0.

    Known answer: budgets (0.0, 0.0012, 0.0038), patched in for _budgets (so no TRAIN_EVAL season
    plays), give [FixedMultiplier with lam 1.0, PIDLagrangian(0.0012), PIDLagrangian(0.0038)], the
    PIDs on the contract defaults (kp 0.5, ki 0.05, kd 0.1, lambda_max 100). Today all three are
    PIDLagrangians. PPO.learn is a no-op here.
    """
    import greenleaf_control.train as train

    budgets = [0.0, 0.0012, 0.0038]
    monkeypatch.setattr(train, "_budgets", lambda rb, budget_frac, season_length: list(budgets))
    _no_learning(monkeypatch)
    made = []

    class Recording(LagrangianCallback):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(train, "LagrangianCallback", Recording)
    runs = tmp_path / "runs"
    argv = _smoke_argv("residual_lag", runs)
    assert train.main(argv, ledger_path=tmp_path / "ledger.jsonl") == 0

    [cb] = made
    # by name first, so while FixedMultiplier is missing this fails here, on the assertion
    assert [type(p).__name__ for p in cb.pids] == [
        "FixedMultiplier",
        "PIDLagrangian",
        "PIDLagrangian",
    ]
    from greenleaf_control.lagrangian import FixedMultiplier

    fixed, *pids = cb.pids
    assert type(fixed) is FixedMultiplier
    assert fixed.lam == 1.0
    assert [type(p) for p in pids] == [PIDLagrangian, PIDLagrangian]
    assert [p.budget for p in pids] == [0.0012, 0.0038]
    assert [(p.kp, p.ki, p.kd, p.lambda_max) for p in pids] == [(0.5, 0.05, 0.1, 100.0)] * 2
    config = json.loads((_smoke_dir(runs) / "manifest.json").read_text())["config"]
    assert config["budgets"] == budgets


def test_registered_run_that_crashes_records_crashed_and_propagates(tmp_path, monkeypatch):
    """PPO.learn raising RuntimeError("boom") in a registered run: the error propagates, the
    manifest says crashed, and the ledger holds the registration, the run row and a crashed
    run-end closing it. No final checkpoint or evaluation is written.

    Known answers: learn raised before any env step, so the run-end records 0 timesteps, 0 solver
    failures and, with no PPO update, every diagnostic None; the manifest records the same
    diagnostics (#19a).
    """
    import greenleaf_control.train as train

    def learn(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(PPO, "learn", learn)
    path, row = _registered(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="boom"):
        train.main(_registered_argv(row["row_id"], runs), ledger_path=path)

    run_dir = runs / row["row_id"] / "a" / "seed0"
    m = json.loads((run_dir / "manifest.json").read_text())
    assert (m["status"], m["timesteps"], m["solver_failures"]) == ("crashed", 0, 0)
    reg, run, end = ledger.rows(path=path)
    assert reg == row
    assert (run["kind"], run["seed"], Path(run["run_dir"])) == ("run", 0, run_dir)
    assert (end["kind"], end["run"], end["status"]) == ("run-end", run["row_id"], "crashed")
    assert (end["timesteps"], end["solver_failures"]) == (0, 0)
    assert end["diagnostics"] == dict.fromkeys(ledger.DIAGNOSTIC_KEYS)
    assert m["diagnostics"] == end["diagnostics"]
    assert not (run_dir / "ckpt_final.zip").exists()
    assert not (run_dir / "final_eval.json").exists()


FLAG_EVERY = 512  # inject a solver failure on every 512th env step


def test_registered_run_end_records_exact_solver_failures_and_update_diagnostics(
    tmp_path, monkeypatch
):
    """A registered 4,096-step run with one env is two rollouts of n_steps 2,048, so two PPO
    updates; its run-end's numbers are exactly the ones the run produced.

    Known answers:
    - SolverFailureFlag.step, patched to set info["solver_failure"] on every 512th call without
      truncating, flags 4,096 / 512 = 8 training steps (any 4,096 consecutive calls hold exactly 8
      multiples of 512), so solver_failures = 8. The 96 evaluation steps after training aren't
      training steps and don't count.
    - min_approx_kl is the smaller of the two updates' train/approx_kl, read off SB3's logger right
      after each PPO.train; final_entropy and final_explained_variance are the second update's
      -train/entropy_loss and train/explained_variance.
    """
    import greenleaf_control.train as train

    truncated_steps = []
    real_step = envs.SolverFailureFlag.step

    def step(self, action):
        obs, reward, terminated, truncated, info = real_step(self, action)
        truncated_steps.append(truncated)
        if len(truncated_steps) % FLAG_EVERY == 0:
            info["solver_failure"] = True
        return obs, reward, terminated, truncated, info

    updates = []
    real_train = PPO.train

    def train_and_record(self):
        real_train(self)
        values = self.logger.name_to_value
        updates.append(
            {
                "approx_kl": float(values["train/approx_kl"]),
                "entropy": -float(values["train/entropy_loss"]),
                "explained_variance": float(values["train/explained_variance"]),
            }
        )

    monkeypatch.setattr(envs.SolverFailureFlag, "step", step)
    monkeypatch.setattr(PPO, "train", train_and_record)
    path, row = _registered(tmp_path, monkeypatch, timesteps=4096)
    argv = _registered_argv(row["row_id"], tmp_path / "runs", timesteps="4096")
    assert train.main(argv, ledger_path=path) == 0

    assert len(truncated_steps) == 4096 + N_DAY  # training, then the 1-day PAPER evaluation
    assert not any(truncated_steps), "a real solver failure would change the known count"
    assert len(updates) == 2
    _, _, end = ledger.rows(path=path)
    assert (end["status"], end["timesteps"]) == ("finished", 4096)
    assert end["solver_failures"] == 4096 // FLAG_EVERY  # 8
    assert end["diagnostics"] == {
        "min_approx_kl": min(u["approx_kl"] for u in updates),
        "final_entropy": updates[-1]["entropy"],
        "final_explained_variance": updates[-1]["explained_variance"],
    }


# Issue #19, part 19a: a remote run's code stamp is passed in. The container that runs
# `train --registered <row> --remote '{"stamp": ..., "registered": <row>}'` has no git checkout and
# no ledger: the Mac computes manifest.code_stamp, passes it and the registered row, and writes the
# run and run-end rows itself (19b).

STAMP_KEYS = {"commit", "dirty", "fingerprints", "weather_hashes"}
OTHER_ROW = "registered-0badc0de"  # an id no test registers


def _outside_git(tmp_path: Path, monkeypatch) -> Path:
    """manifest.REPO_ROOT becomes tmp_path/image, a directory in no git repo (like the container's
    /root): git may not search above tmp_path for one."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    image = tmp_path / "image"
    image.mkdir()
    monkeypatch.setattr(manifest, "REPO_ROOT", image)
    return image


def _no_git_or_ledger(*args, **kwargs):
    raise AssertionError("a --remote run read git or the ledger")


def _remote(tmp_path: Path, monkeypatch, *, checkout: bool = False) -> tuple[Path, dict, dict]:
    """The container's side of a remote run: (tmp ledger path, the registered row, the stamp).

    REPO_ROOT is outside any git repo (with checkout, a clean temp git repo, as on the Mac).
    REGISTRATION is built dry, so the tmp ledger never exists. The stamp is what the Mac would
    pass: a fake 40-hex commit, fake fingerprints (the sha256 of each key's name, never of a file)
    and this checkout's real weather hashes. Then manifest._git, ledger.rows and ledger.append all
    raise.
    """
    if checkout:
        _make_repo(tmp_path, monkeypatch)
    else:
        _outside_git(tmp_path, monkeypatch)
    path = tmp_path / "ledger.jsonl"
    row = ledger.register(**REGISTRATION, dry=True, path=path)
    stamp = {
        "commit": "0123456789abcdef" * 2 + "01234567",  # 40 hex, no commit anywhere
        "dirty": False,
        "fingerprints": {
            key: hashlib.sha256(key.encode()).hexdigest() for key in manifest.FINGERPRINTED
        },
        "weather_hashes": scenarios.weather_hashes(),
    }
    for module, name in ((manifest, "_git"), (ledger, "rows"), (ledger, "append")):
        monkeypatch.setattr(module, name, _no_git_or_ledger)
    return path, row, stamp


def _payload(stamp: dict, row: dict) -> str:
    """--remote's JSON, as train_remote passes it."""
    return json.dumps({"stamp": stamp, "registered": row})


def test_remote_run_writes_the_passed_stamp_verbatim_and_never_reads_git_or_the_ledger(
    tmp_path, monkeypatch
):
    """Done when (#19): train with passed-in fingerprints writes them to the manifest verbatim and
    never calls git. With --remote, train reads neither git nor the ledger (both raise here) and
    writes no ledger row: the tmp ledger is never created.

    The manifest holds the stamp's commit, dirty, fingerprints and weather hashes exactly as
    passed. It names the passed row and its issue (13), is finished at 0 steps (the no-op learn
    never steps an env), and carries the run's diagnostics for the Mac's run-end. Known answer:
    PPO.learn is a no-op here, so no update ran and every diagnostic is None (the same as a run
    that crashed before training).
    """
    import greenleaf_control.train as train

    _no_learning(monkeypatch)
    path, row, stamp = _remote(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    argv = _registered_argv(row["row_id"], runs, remote=_payload(stamp, row))
    assert train.main(argv, ledger_path=path) == 0

    assert not path.exists()
    assert [p.name for p in runs.iterdir()] == [row["row_id"]]
    m = json.loads((runs / row["row_id"] / "a" / "seed0" / "manifest.json").read_text())
    assert {key: m[key] for key in STAMP_KEYS} == stamp
    assert (m["registered"], m["issue"], m["smoke"]) == (row["row_id"], 13, False)
    # 0: the no-op learn never steps an env or calls the callback, whose count train records
    assert (m["status"], m["timesteps"]) == ("finished", 0)
    assert m["diagnostics"] == dict.fromkeys(ledger.DIAGNOSTIC_KEYS)


STAMP_WEATHER = [
    pytest.param(
        lambda hashes: {**hashes, "Amsterdam/2010.csv": "0" * 64},
        "Amsterdam/2010.csv",
        id="one-hash-differs",
    ),
    pytest.param(
        lambda hashes: {k: v for k, v in hashes.items() if k != "Amsterdam/2009.csv"},
        "Amsterdam/2009.csv",
        id="a-file-the-stamp-lacks",
    ),
    # The Mac has a weather file the container doesn't: the image didn't ship it.
    pytest.param(
        lambda hashes: {**hashes, "Amsterdam/2013.csv": "0" * 64},
        "Amsterdam/2013.csv",
        id="a-file-this-machine-lacks",
    ),
]


@pytest.mark.parametrize(("change", "named"), STAMP_WEATHER)
def test_remote_run_refuses_weather_that_differs_from_the_stamp(
    change, named, tmp_path, monkeypatch
):
    """The stamp's weather hashes are the Mac's; the container's weather files must hash the same
    (scenarios.weather_hashes() here), or the run trains on other data than the manifest records.

    One file differs (its hash, it's missing from the stamp, or it's missing here because the image
    didn't ship it): a RuntimeError names that file and no other, before any run dir exists, PPO
    trains nothing and no ledger is written.
    """
    import greenleaf_control.train as train

    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    path, row, stamp = _remote(tmp_path, monkeypatch)
    here = scenarios.weather_hashes()
    assert change(here).get(named) != here.get(named)  # the change is to that file
    stamp = {**stamp, "weather_hashes": change(here)}
    runs = tmp_path / "runs"

    with pytest.raises(RuntimeError, match=re.escape(named)) as excinfo:
        argv = _registered_argv(row["row_id"], runs, remote=_payload(stamp, row))
        train.main(argv, ledger_path=path)
    others = [key for key in here if key != named and key in str(excinfo.value)]
    assert not others, f"names files that don't differ: {others}"
    assert _files(runs) == []
    assert not path.exists()


def test_remote_run_refuses_a_dirty_stamp(tmp_path, monkeypatch):
    """A stamp from a dirty tree records code no commit holds; a registered run refuses it, as
    write_manifest refuses a dirty tree, with a RuntimeError saying dirty, before any run dir."""
    import greenleaf_control.train as train

    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    path, row, stamp = _remote(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="dirty"):
        argv = _registered_argv(row["row_id"], runs, remote=_payload({**stamp, "dirty": True}, row))
        train.main(argv, ledger_path=path)
    assert _files(runs) == []
    assert not path.exists()


REMOTE_REFUSALS = [
    pytest.param(
        lambda runs, remote: _smoke_argv("ppo", runs, "--remote", remote),
        ("--remote",),
        id="with-smoke",
    ),
    pytest.param(
        lambda runs, remote: _registered_argv(OTHER_ROW, runs, remote=remote),
        ("--remote", OTHER_ROW),
        id="for-another-row",
    ),
]


@pytest.mark.parametrize(("make_argv", "named"), REMOTE_REFUSALS)
def test_remote_flag_is_refused_with_smoke_and_for_another_row(
    make_argv, named, tmp_path, monkeypatch, capsys
):
    """--remote needs --registered, so --smoke --remote refuses; and the row --remote carries
    must be the one --registered names (here OTHER_ROW, not the passed row).

    Each raises SystemExit (nonzero) or ValueError whose reason names --remote (and, for another
    row, the --registered id), not argparse's unknown-flag error, before anything is written: no
    file under runs/ and no ledger.
    """
    import greenleaf_control.train as train

    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    path, row, stamp = _remote(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        train.main(make_argv(runs, _payload(stamp, row)), ledger_path=path)
    reason = _refusal(excinfo, capsys).split("error:")[-1]  # argparse's usage lists every flag
    assert "unrecognized arguments" not in reason, reason
    for text in named:
        assert text in reason, reason
    assert _files(runs) == []
    assert not path.exists()


def test_remote_flag_is_refused_inside_a_git_repo(tmp_path, monkeypatch, capsys):
    """#19b: --remote takes a passed stamp instead of reading git, so only the container, which has
    no git checkout, may use it; on the Mac it would skip the dirty-tree check. train refuses
    --remote where manifest.REPO_ROOT holds a .git, and checks the filesystem for it, not git:
    here git, the ledger and PPO.learn all raise.

    REPO_ROOT is a clean temp git repo and the payload is the one the container's run accepts, so
    the .git is the only reason to refuse: SystemExit (nonzero), ValueError or RuntimeError whose
    reason names --remote, before any run dir, and no ledger is written.
    """
    import greenleaf_control.train as train

    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    path, row, stamp = _remote(tmp_path, monkeypatch, checkout=True)
    assert (manifest.REPO_ROOT / ".git").exists()
    runs = tmp_path / "runs"

    with pytest.raises((SystemExit, ValueError, RuntimeError)) as excinfo:
        train.main(
            _registered_argv(row["row_id"], runs, remote=_payload(stamp, row)), ledger_path=path
        )
    reason = _refusal(excinfo, capsys).split("error:")[-1]  # argparse's usage lists every flag
    assert "--remote" in reason, reason
    assert _files(runs) == []
    assert not path.exists()


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        pytest.param({"seed": "1"}, "seed 1", id="seed-not-in-the-row"),
        pytest.param({"timesteps": "4096"}, "--timesteps 4096", id="timesteps-differ"),
    ],
)
def test_remote_run_checks_its_flags_against_the_passed_row(
    flags, named, tmp_path, monkeypatch, capsys
):
    """With --remote the registration check runs against the passed row, with the ledger never
    read: REGISTRATION holds seed 0 and 2,048 steps, so --seed 1 and --timesteps 4096 each refuse
    as they would locally, naming the value, before any run dir."""
    import greenleaf_control.train as train

    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    path, row, stamp = _remote(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    argv = _registered_argv(row["row_id"], runs, remote=_payload(stamp, row), **flags)

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        train.main(argv, ledger_path=path)
    reason = _refusal(excinfo, capsys).split("error:")[-1]
    assert "unrecognized arguments" not in reason, reason
    assert named in reason, reason
    assert _files(runs) == []
    assert not path.exists()


def test_registered_run_outside_a_git_repo_refuses_without_the_remote_flag(tmp_path, monkeypatch):
    """Done when (#19): without passed-in fingerprints, inside a non-repo, train refuses. A
    registered run with no --remote and REPO_ROOT in no git repo raises (git's own error or a
    RuntimeError) before any run dir or run row: the ledger holds only the registration."""
    import greenleaf_control.train as train

    _outside_git(tmp_path, monkeypatch)
    path = tmp_path / "ledger.jsonl"
    row = ledger.register(**REGISTRATION, path=path)
    monkeypatch.setattr(PPO, "learn", _refused_before_training)
    runs = tmp_path / "runs"

    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        train.main(_registered_argv(row["row_id"], runs), ledger_path=path)
    assert _files(runs) == []
    assert ledger.rows(path=path) == [row]


# A file written after the temp repo's commit, and whether the tree is then dirty. Run dirs under
# control/runs/ and the ledger file itself don't count; anything else, in control/ledger/ too, does.
CODE_STAMP_TREES = [
    pytest.param("control/new.py", True, id="an-untracked-file"),
    pytest.param("control/README.md", True, id="a-modified-file"),
    pytest.param("control/ledger/notes.txt", True, id="another-file-in-the-ledger-dir"),
    pytest.param("control/runs/r/a/seed0/manifest.json", False, id="a-run-dir"),
    pytest.param("control/ledger/ledger.jsonl", False, id="the-ledger-file"),
]


@pytest.mark.parametrize(("written", "dirty"), CODE_STAMP_TREES)
def test_code_stamp_refuses_a_dirty_tree_unless_smoke(written, dirty, tmp_path, monkeypatch):
    """manifest.code_stamp(smoke=...) is write_manifest's git logic on its own, with the same
    exemptions. In a temp repo laid out like this one (control/ledger/.gitkeep committed), a dirty
    tree raises RuntimeError naming the file unless smoke, and smoke records dirty True; an exempt
    file leaves the tree clean either way."""
    repo = _make_repo(tmp_path, monkeypatch)
    (repo / "control" / "ledger").mkdir()
    (repo / "control" / "ledger" / ".gitkeep").write_text("")
    _commit(repo)
    file = repo / written
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("written after the commit\n")

    if dirty:
        with pytest.raises(RuntimeError, match=re.escape(written)):
            manifest.code_stamp(smoke=False)
    else:
        assert manifest.code_stamp(smoke=False)["dirty"] is False
    assert manifest.code_stamp(smoke=True)["dirty"] is dirty


def test_code_stamp_matches_what_write_manifest_records(tmp_path, monkeypatch):
    """code_stamp returns exactly {commit, dirty, fingerprints, weather_hashes} for this checkout,
    and write_manifest with no stamp records those same four.

    Known answers, in a clean temp repo whose one scorer file is docs/PROTOCOL.md = b"protocol\\n":
    commit is its HEAD, dirty False, fingerprints the sha256 of those 9 bytes for PROTOCOL.md and
    None for every other scorer file (not in the repo), and the weather hashes this checkout's.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    (repo / "docs").mkdir()
    (repo / "docs" / "PROTOCOL.md").write_bytes(b"protocol\n")
    _commit(repo)

    stamp = manifest.code_stamp(smoke=False)
    assert set(stamp) == STAMP_KEYS
    assert (stamp["commit"], stamp["dirty"]) == (_git(repo, "rev-parse", "HEAD"), False)
    protocol = hashlib.sha256(b"protocol\n").hexdigest()
    assert stamp["fingerprints"] == {
        key: protocol if key == "docs/PROTOCOL.md" else None for key in manifest.FINGERPRINTED
    }
    assert stamp["weather_hashes"] == scenarios.weather_hashes()

    m = manifest.write_manifest(
        tmp_path / "run", config=_config("ppo", tmp_path), issue=19, registered=None, smoke=False
    )
    assert {key: m[key] for key in STAMP_KEYS} == stamp


@pytest.mark.parametrize("method", METHODS)
def test_build_model_uses_ppo_yml_unchanged(method, tmp_path):
    """build_model is configs/ppo.yml's PPO on cpu with nothing changed but log_std_init: 0.0 (from
    np.log(1)) for ppo_paper and ppo, -1.0 for the residual arms (PROTOCOL §3).

    The file is checked against this test's copy first, so a changed file and a changed model both
    show. Observable on the model: every hyperparameter, policy_kwargs with nothing added, SiLU
    layers 3 x 256 (pi) and 3 x 512 (vf), Adam with amsgrad at lr 2e-5, the log_std values, the
    seed and the device.
    """
    import greenleaf_control.train as train

    yml = yaml.safe_load((CONTROL / "configs" / "ppo.yml").read_text())["GreenLightEnv"]
    assert (yml.pop("total_timesteps"), yml.pop("n_envs"), yml.pop("policy")) == (
        2_000_000,
        4,
        "MlpPolicy",
    )
    assert yml.pop("policy_kwargs") == PPO_YML_POLICY_KWARGS
    assert yml == PPO_YML

    venv = VecNormalize(
        DummyVecEnv([lambda: envs.make_env(method, seed=0, season_length=1, rb_params=RB_DEFAULT)]),
        gamma=0.9631,
    )
    model = train.build_model(_config(method, tmp_path), venv)

    got = {key: getattr(model, key) for key in PPO_YML}
    got["clip_range"] = model.clip_range(1.0)  # SB3 turns it into a schedule
    assert got == PPO_YML
    assert model.lr_schedule(1.0) == 2e-5
    assert model.seed == 0
    assert model.device == torch.device("cpu")
    assert model.policy_kwargs == {**POLICY_KWARGS, "log_std_init": LOG_STD_INIT[method]}

    policy = model.policy
    assert type(policy) is ActorCriticPolicy  # PPO's "MlpPolicy"
    assert policy.activation_fn is torch.nn.SiLU
    nets = ((policy.mlp_extractor.policy_net, 256), (policy.mlp_extractor.value_net, 512))
    for net, width in nets:
        assert [type(layer) for layer in net] == [torch.nn.Linear, torch.nn.SiLU] * 3
        assert [layer.out_features for layer in net if isinstance(layer, torch.nn.Linear)] == [
            width
        ] * 3
    assert type(policy.optimizer) is torch.optim.Adam
    assert policy.optimizer.defaults["amsgrad"] is True
    assert policy.optimizer.param_groups[0]["lr"] == 2e-5
    assert torch.equal(policy.log_std.detach(), torch.full((6,), LOG_STD_INIT[method]))


@pytest.mark.parametrize(
    ("key", "value"),
    [("activation_fn", "relu"), ("optimizer_class", "sgd"), ("log_std_init", "np.log(2)")],
)
def test_build_model_refuses_a_ppo_yml_string_it_has_no_mapping_for(
    key, value, tmp_path, monkeypatch
):
    """configs/ppo.yml's policy_kwargs strings are mapped explicitly (silu, adam, np.log(1)); any
    other string raises ValueError naming the key and the string, rather than reaching SB3 as text
    or being guessed at."""
    import greenleaf_control.train as train

    def ppo_yml() -> dict:
        yml = yaml.safe_load((CONTROL / "configs" / "ppo.yml").read_text())["GreenLightEnv"]
        yml["policy_kwargs"][key] = value
        return yml

    monkeypatch.setattr(train, "_ppo_yml", ppo_yml)
    venv = VecNormalize(DummyVecEnv([lambda: envs.make_env("ppo", seed=0, season_length=1)]))

    with pytest.raises(ValueError, match=re.escape(value)) as excinfo:
        train.build_model(_config("ppo", tmp_path), venv)
    assert key in str(excinfo.value)


@pytest.mark.parametrize("method", METHODS)
def test_build_vec_env_stack_order(method, tmp_path):
    """build_vec_env(config, 1), from the outside in: VecNormalize (training, gamma 0.9631, obs and
    reward normalized, SB3's default clips of 10), VecLagrangianReward with lambdas at 0 (residual_lag
    only), VecMonitor, then a DummyVecEnv of one make_env(method, seed=0) env.

    Known answers: the observation is 22 dims (ppo_paper), 23 (ppo) or 29 (residual arms); the season
    is the config's 1 day (96 steps); the reward's weight is the config's 0.5 (0.0 for residual_lag);
    the env's RNG is in the state make_env(method, seed=0) leaves it in.
    """
    import greenleaf_control.train as train

    weight = 0.0 if method == "residual_lag" else 0.5
    venv = train.build_vec_env(_config(method, tmp_path, penalty_weight=weight), 1)

    chain = _chain(venv)
    lag = [VecLagrangianReward] if method == "residual_lag" else []
    assert [type(v) for v in chain] == [VecNormalize, *lag, VecMonitor, DummyVecEnv]
    vecnorm, inner = chain[0], chain[-1]
    assert (vecnorm.training, vecnorm.norm_obs, vecnorm.norm_reward) == (True, True, True)
    assert vecnorm.gamma == 0.9631
    assert (vecnorm.clip_obs, vecnorm.clip_reward) == (10.0, 10.0)
    if lag:
        np.testing.assert_array_equal(chain[1].lambdas, np.zeros(3))
    assert vecnorm.num_envs == 1
    assert vecnorm.observation_space.shape == (DIMS[method],)

    raw = inner.envs[0].unwrapped
    assert raw.N == N_DAY
    assert raw.reward_fn.penalty_weight == weight
    ref = envs.make_env(method, seed=0, season_length=1, rb_params=RB_DEFAULT).unwrapped
    assert raw.np_random.bit_generator.state == ref.np_random.bit_generator.state


def test_build_vec_env_uses_subproc_envs_seeded_seed_plus_i(tmp_path):
    """Two envs run in subprocesses: VecNormalize, VecMonitor, then a SubprocVecEnv whose env i is
    make_env("ppo", seed=3 + i). Known answer: env i's RNG is in the state make_env(seed=3 + i)
    leaves it in, and those two states differ."""
    import greenleaf_control.train as train

    refs = [
        envs.make_env("ppo", seed=3 + i, season_length=1).unwrapped.np_random.bit_generator.state
        for i in range(2)
    ]
    assert refs[0] != refs[1]
    venv = train.build_vec_env(_config("ppo", tmp_path, seed=3), 2)
    try:
        chain = _chain(venv)
        assert [type(v) for v in chain] == [VecNormalize, VecMonitor, SubprocVecEnv]
        assert venv.num_envs == 2
        assert [g.bit_generator.state for g in chain[-1].get_attr("np_random")] == refs
    finally:
        venv.close()


def test_build_vec_env_scenario_and_season_length_override_the_config(tmp_path):
    """For evaluation: build_vec_env(config, 1, scenario=s, season_length=2) fixes ppo's weather on s
    (VALIDATION's first scenario, not ppo's TRAIN sampler) for a 2-day season (192 steps). Without
    them it keeps the method's TRAIN sampler and the config's 1-day season (96 steps)."""
    import greenleaf_control.train as train

    config = _config("ppo", tmp_path)
    s = scenarios.VALIDATION[0]
    default = _chain(train.build_vec_env(config, 1))[-1].envs[0].unwrapped
    fixed = (
        _chain(train.build_vec_env(config, 1, scenario=s, season_length=2))[-1].envs[0].unwrapped
    )

    assert isinstance(default.weather_scenario_sampler, RandomWeatherSampler)
    assert default.N == N_DAY
    assert isinstance(fixed.weather_scenario_sampler, FixedWeatherSampler)
    assert vars(fixed.weather_scenario_sampler) == s
    assert fixed.N == 2 * N_DAY


def test_build_functions_match_the_contract_signatures():
    """build_vec_env(config, n_envs, *, scenario=None, season_length=None) and
    build_model(config, venv), exposed for evaluate and the probes."""
    import greenleaf_control.train as train

    params = inspect.signature(train.build_vec_env).parameters
    assert list(params) == ["config", "n_envs", "scenario", "season_length"]
    for name in ("scenario", "season_length"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params[name].default is None, name
    assert list(inspect.signature(train.build_model).parameters) == ["config", "venv"]


def test_cli_runs_as_a_module_and_lists_the_contract_flags(tmp_path):
    """python -m greenleaf_control.train --help exits 0 and lists every contract flag plus --rb."""
    result = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.train", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    missing = [flag for flag in CLI_FLAGS if flag not in result.stdout]
    assert not missing, f"--help doesn't list {missing}"


def test_main_takes_ledger_path_and_no_test_wrote_the_real_ledger_or_runs():
    """main(argv=None, *, ledger_path=ledger.LEDGER): every test here passes a tmp ledger_path.

    This test runs last in the module and checks the real ledger's size and mtime, and the
    listing of control/runs/, are as they were when the module was collected.
    """
    import greenleaf_control.train as train

    params = inspect.signature(train.main).parameters
    assert list(params) == ["argv", "ledger_path"]
    assert params["argv"].default is None
    assert params["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["ledger_path"].default == ledger.LEDGER
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE
