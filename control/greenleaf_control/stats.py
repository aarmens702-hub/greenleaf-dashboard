"""Statistics on runs x tasks score arrays (PROTOCOL §6): the IQM with its stratified-bootstrap
95% CI, and the probability of improvement.

iqm_ci doesn't call rliable.library, which can't be imported in the locked env: rliable 1.2.0 pins
arch < 8, and arch 7.2.0 calls pandas 3's changed deprecate_kwarg. It writes out rliable 1.2.0's
percentile bootstrap instead (StratifiedBootstrap + get_interval_estimates(method="percentile")),
on the same random stream: pinned to a written-out replica of rliable 1.2.0's percentile stratified
bootstrap in tests/test_stats.py; checked once against rliable.library itself with pandas'
deprecate_kwarg shimmed in memory (2026-10-28, max abs diff 6e-17). rliable.metrics needs only
numpy and scipy, so the IQM and the probability of improvement are rliable's own.
"""

import numpy as np
from rliable import metrics


def _runs_by_tasks(x: np.ndarray, name: str) -> np.ndarray:
    """x as an array, refused unless it is 2-D (runs x tasks)."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, runs x tasks; got shape {x.shape}")
    return x


def iqm_ci(scores: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(IQM, ci_low, ci_high) of a runs x tasks array: the 25% trimmed mean over every score, and
    the 2.5th and 97.5th percentiles of reps bootstrap IQMs.

    Each rep resamples runs with replacement within every task (column): one
    RandomState(seed).choice(runs, (runs, tasks)) draw, the stream rliable gets from
    np.random.seed(seed). The global np.random is never touched.
    """
    scores = _runs_by_tasks(scores, "scores")
    runs, tasks = scores.shape
    rng = np.random.RandomState(seed)
    iqms = [
        metrics.aggregate_iqm(scores[rng.choice(runs, (runs, tasks)), np.arange(tasks)])
        for _ in range(reps)
    ]
    low, high = np.percentile(iqms, [2.5, 97.5])
    return float(metrics.aggregate_iqm(scores)), float(low), float(high)


def prob_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a = b) over every (run of a, run of b) pair within a task, averaged over
    tasks (rliable's probability_of_improvement); a and b may have different numbers of runs.

    Tasks pair up column by column, so a and b need the same number. rliable reads a's count only
    (fewer in a silently drops b's other tasks), so a mismatch is refused here.
    """
    a, b = _runs_by_tasks(a, "a"), _runs_by_tasks(b, "b")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"a has {a.shape[1]} tasks and b has {b.shape[1]}; tasks pair up column by column"
        )
    return float(metrics.probability_of_improvement(a, b))
