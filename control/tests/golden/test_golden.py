"""Golden numbers on PAPER (issue #11, probe; PROTOCOL §7, ladder rung 2).

The numbers every later change is checked against. values.json beside this file holds them, copied
at full precision from the orchestrator's measurement on #11 (no training, deterministic seasons):

- Measured 2026-10-28T19:27:16Z on commit 964a194 (gl-gym 2d3febb), arm64 Darwin 24.5.0,
  Python 3.12.7. Scenario PAPER = Amsterdam 2010, start day 59, a 60-day season.
- Changing a value needs Aarmen's yes: tests/golden/ is scorer material, behind /ship's gate.

Two numbers, two env lengths (decision B, Aarmen 2026-10-28):
- Upstream's own env (its GreenhouseReward, normalize_actions=False, the fixed PAPER sampler) runs
  N + 1 = 5,761 steps before it ends `terminated`. Driven by rb_default, its summed reward
  reproduces PROTOCOL §7's 3104.58 (the 2026-10-27 bench; Done-when, rel 1e-4) and is pinned at
  3104.5770693 (rel 1e-6). 3104.58 is not a published number: bench.py printed it to 2 decimals
  (gl-gym 2d3febb, Python 3.12, M3), with no repo commit; values.json records that source.
- Our env ends the season on step N, exactly 5,760 steps, so our metrics are over 5,760 steps:
  envs.run_baseline -> metrics.trace_from_infos -> metrics.episode_metrics with the model's params.
  rb_default's paper_return is 3104.2159 there, not PROTOCOL §7's 3104.58 (the 2026-10-27 bench),
  because the 5,761st step is cut. test_decision_b_only_cuts_upstreams_extra_final_step pins that.

viol_hours_<c> is a violating-step count x 0.25 h, exactly representable, so it is compared with ==.
The other floats are compared at rel 1e-6. Each season takes about 3-5 s, so they are slow and
CI (Linux x86_64) doesn't run them; they are pinned on the machine above.
"""

import functools
import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from greenleaf_control import envs, metrics, rule_based, scenarios

VALUES = json.loads(Path(__file__).with_name("values.json").read_text(encoding="utf-8"))
BASELINES = ("rb_default", "zero_controls", "random")
N_SEASON = 5_760  # 60 days x 96 steps (86,400 s / 900 s)
FLOAT_KEYS = ("profit_eur_m2", "yield_kg_m2", "paper_return")
VIOL_KEYS = ("viol_hours_co2", "viol_hours_temp", "viol_hours_rh")


@functools.cache
def _upstream_rb_default_season():
    """Upstream's own env on PAPER, driven by rb_default until it ends: its per-step rewards and the
    final (terminated, truncated). Cached, so the slow tests below run the season once (about 4 s)."""
    env = gym.make(
        envs.ENV_ID,
        weather_data_dir=str(scenarios.WEATHER_DIR),
        normalize_actions=False,
        weather_scenario_sampler="fixed",
        weather_scenario_sampler_kwargs=scenarios.PAPER,
    )
    controller = rule_based.make_controller(rule_based.load_rb_params("default"))
    env.reset(seed=0)
    rewards, terminated, truncated = [], False, False
    while not (terminated or truncated):
        _, r, terminated, truncated, _ = env.step(rule_based.rb_action(env, controller))
        rewards.append(float(r))
    env.close()
    return tuple(rewards), (terminated, truncated)


def test_values_json_names_its_provenance_and_every_pinned_number():
    """values.json carries its provenance (a commit, gl-gym's commit, a machine, the date, Python,
    PAPER itself), PROTOCOL §7's 3104.58 (the 2026-10-27 bench) with its own source beside upstream's
    measured return, and exactly the pinned metrics for each of rb_default, zero_controls and random."""
    prov = VALUES["provenance"]
    assert re.fullmatch(r"[0-9a-f]{7,40}", prov["commit"])
    assert re.fullmatch(r"[0-9a-f]{7,40}", prov["gl_gym"])
    assert prov["machine"] and prov["measured_at"] and prov["python"]
    assert "Aarmen" in prov["note"]
    assert prov["scenario"] == scenarios.PAPER
    up = VALUES["upstream_rb_default_return"]
    assert set(up) == {"value", "steps", "protocol_reference"}
    assert up["steps"] == N_SEASON + 1
    assert isinstance(up["value"], float)
    ref = up["protocol_reference"]
    assert set(ref) == {"value", "source", "gl_gym", "python", "machine", "commit", "note"}
    assert ref["value"] == 3104.58
    assert ref["source"].startswith("PROTOCOL §7")
    assert ref["gl_gym"] == prov["gl_gym"]  # the bench and our measurement ran the same simulator
    assert ref["commit"] is None  # the bench ran on no repo commit; its note says so
    assert ref["python"] and ref["machine"] and ref["note"]
    assert set(VALUES["ours"]) == set(BASELINES)
    for name in BASELINES:
        assert set(VALUES["ours"][name]) == {*FLOAT_KEYS, *VIOL_KEYS, "steps"}, name
        assert VALUES["ours"][name]["steps"] == N_SEASON, name


@pytest.mark.slow
def test_upstream_rb_default_return_on_paper():
    """Upstream's own env on PAPER, driven by rb_default until it terminates: 5,761 steps, never
    truncated, and a return equal to PROTOCOL §7's 3104.58 (the 2026-10-27 bench) at rel 1e-4 and
    to the pinned 3104.5770693 at rel 1e-6 (about 4 s)."""
    up = VALUES["upstream_rb_default_return"]
    rewards, ended = _upstream_rb_default_season()
    assert ended == (True, False)
    assert len(rewards) == up["steps"]
    assert sum(rewards) == pytest.approx(up["protocol_reference"]["value"], rel=1e-4)
    assert sum(rewards) == pytest.approx(up["value"], rel=1e-6)


@pytest.mark.slow
def test_decision_b_only_cuts_upstreams_extra_final_step():
    """Decision B only cuts upstream's extra final step. Upstream's first 5,760 rewards sum exactly
    to our rb_default season's sum of info["paper_reward"] (the reviewer measured both at
    3104.215895035613), and upstream's 5,761st reward, the step our env drops, is a positive 0.3612.
    So our 3104.2159 is PROTOCOL §7's 3104.58 (the 2026-10-27 bench) less that one step (about 4 s,
    plus 4 s for upstream's season unless the test above already ran it)."""
    rewards, _ = _upstream_rb_default_season()
    infos = envs.run_baseline("rb_default", scenarios.PAPER)
    assert len(infos) == N_SEASON
    assert sum(rewards[:N_SEASON]) == sum(info["paper_reward"] for info in infos)
    # 3104.5770693 (5,761 steps) - 3104.2158950 (5,760 steps) = 0.3611743, the dropped step's reward
    assert rewards[N_SEASON] > 0
    assert rewards[N_SEASON] == pytest.approx(0.3612, rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("name", BASELINES)
def test_our_paper_season_metrics(name):
    """A full PAPER season of each baseline on our env (5,760 steps), graded by metrics: valid,
    viol_hours_<c> equal to the pinned values exactly, and profit_eur_m2, yield_kg_m2 and
    paper_return at rel 1e-6 (about 4 s each)."""
    want = VALUES["ours"][name]
    params = envs.make_raw_env(scenario=scenarios.PAPER, normalize_actions=False).unwrapped.p
    states, controls = metrics.trace_from_infos(envs.run_baseline(name, scenarios.PAPER))
    m = metrics.episode_metrics(states, controls, np.asarray(params), season_steps=N_SEASON)
    assert m["steps"] == want["steps"]
    assert m["valid"] is True
    assert {k: m[k] for k in VIOL_KEYS} == {k: want[k] for k in VIOL_KEYS}
    for k in FLOAT_KEYS:
        assert m[k] == pytest.approx(want[k], rel=1e-6), k
