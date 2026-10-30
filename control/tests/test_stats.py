"""Tests for greenleaf_control.stats (issue #17, part 17a): the IQM with its stratified-bootstrap CI,
and the probability of improvement (docs/API-CONTRACT.md, PROTOCOL §6).

iqm_ci(scores, reps=2000, seed=0) -> (iqm, ci_low, ci_high) takes a 2-D runs x tasks array. The IQM
is rliable 1.2.0's aggregate_iqm (the 25% trimmed mean over every score). The CI is a stratified
percentile 95% bootstrap that iqm_ci writes out itself, drawing from a local
np.random.RandomState(seed): rliable.library can't be imported in the locked env (rliable 1.2.0 pins
arch < 8, and arch 7.2.0 breaks on pandas 3). That local stream is the one rliable's
get_interval_estimates gets from np.random.seed(seed), so _replica_ci pins the two as equal to
1e-12. The global np.random is never touched.

prob_improvement(a, b) is rliable's probability_of_improvement as a Python float: per task,
P(a > b) + 0.5 P(a = b) over every pair of a run of a and a run of b, averaged over tasks.

greenleaf_control.stats is imported inside each test body, so while it is missing each test FAILS
rather than ERRORs at collection.
"""

import inspect

import numpy as np
import pytest
import scipy.stats


def _scores(seed: int = 0, shape: tuple[int, int] = (5, 8)) -> np.ndarray:
    """A runs x tasks array of U[0, 1) scores from a seeded Generator (not the global np.random)."""
    return np.random.default_rng(seed).random(shape)


def _replica_ci(scores: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    """The stratified-bootstrap percentile 95% CI of the IQM, written out here without rliable.

    Each rep draws, within every task (column), runs with replacement: one
    RandomState(seed).choice(runs, (runs, tasks)) call per rep, the stream np.random.seed(seed)
    gives. The CI is the 2.5th and 97.5th percentiles of the reps' 25% trimmed means. Checked to
    reproduce rliable 1.2.0's get_interval_estimates(method="percentile") to 1e-12.
    """
    rng = np.random.RandomState(seed)
    runs, tasks = scores.shape
    iqms = [
        scipy.stats.trim_mean(
            scores[rng.choice(runs, (runs, tasks), replace=True), np.arange(tasks)],
            0.25,
            axis=None,
        )
        for _ in range(reps)
    ]
    low, high = np.percentile(iqms, [2.5, 97.5])
    return float(low), float(high)


def test_iqm_ci_of_constant_scores_is_exactly_the_constant():
    """Done when: iqm_ci on constant scores gives (c, c, c).

    Known answer: 5 x 8 scores all 2.5. Every bootstrap resample is all 2.5, so every resampled IQM
    is 2.5 and both percentiles are 2.5: exactly (2.5, 2.5, 2.5).
    """
    from greenleaf_control import stats

    assert stats.iqm_ci(np.full((5, 8), 2.5)) == (2.5, 2.5, 2.5)


def test_iqm_ci_brackets_its_iqm_on_random_scores():
    """Done when: on a random 5 x 8 array the IQM lies inside its CI.

    The IQM is the mean of the middle 20 of the 40 sorted scores (25% trimmed from each end), and
    the CI has width: ci_low < iqm < ci_high.
    """
    from greenleaf_control import stats

    x = _scores()
    iqm, low, high = stats.iqm_ci(x)

    assert iqm == pytest.approx(np.sort(x.ravel())[10:30].mean(), rel=1e-12)
    assert low < iqm < high


def test_iqm_of_0_to_39_is_19_5():
    """Known answer: np.arange(40) as 5 x 8. Trimming 10 scores from each end leaves 10..29, whose
    mean is (10 + 29) / 2 = 19.5. The CI contains it."""
    from greenleaf_control import stats

    iqm, low, high = stats.iqm_ci(np.arange(40, dtype=float).reshape(5, 8))

    assert iqm == 19.5
    assert low <= 19.5 <= high


@pytest.mark.parametrize(
    ("shape", "reps", "seed"),
    [((5, 8), 2000, 0), ((5, 8), 2000, 1), ((10, 3), 500, 7)],
    ids=["5x8-2000-reps-seed-0", "5x8-2000-reps-seed-1", "10x3-500-reps-seed-7"],
)
def test_iqm_ci_is_a_stratified_percentile_bootstrap_over_runs(shape, reps, seed):
    """The CI is the percentile 95% CI of a bootstrap that resamples runs within each task (not
    tasks), reps times, from a local RandomState(seed), the same stream np.random.seed(seed) gives.

    Known answer: _replica_ci, the same bootstrap written out with scipy and a RandomState.
    """
    from greenleaf_control import stats

    x = _scores(seed=3, shape=shape)
    _, low, high = stats.iqm_ci(x, reps=reps, seed=seed)

    assert (low, high) == pytest.approx(_replica_ci(x, reps, seed), rel=1e-12, abs=1e-12)


def test_iqm_ci_defaults_to_2000_reps_and_seed_0():
    """PROTOCOL §6: 2,000 bootstrap reps, seed 0, as the contract's defaults; a call with no
    reps or seed is the call with reps=2000, seed=0."""
    from greenleaf_control import stats

    params = inspect.signature(stats.iqm_ci).parameters
    assert list(params) == ["scores", "reps", "seed"]
    assert (params["reps"].default, params["seed"].default) == (2000, 0)
    x = _scores()
    assert stats.iqm_ci(x) == stats.iqm_ci(x, reps=2000, seed=0)


def test_iqm_ci_depends_on_its_seed_not_on_the_global_rng():
    """The same scores give the same CI whatever the global np.random was seeded with before the
    call (1 or 12345), and a different seed argument gives a different CI."""
    from greenleaf_control import stats

    x = _scores()
    np.random.seed(1)
    first = stats.iqm_ci(x)
    np.random.seed(12345)
    second = stats.iqm_ci(x)

    assert first == second
    assert stats.iqm_ci(x, seed=1)[1:] != first[1:]


def test_iqm_ci_leaves_the_global_rng_as_it_found_it():
    """iqm_ci draws from its own RandomState(seed) and never touches the global np.random: the
    global state after the call equals the state before it, so the caller's next draws are the
    ones it would have got without the call."""
    from greenleaf_control import stats

    np.random.seed(7)
    expected = np.random.random(3)

    np.random.seed(7)
    before = np.random.get_state()
    stats.iqm_ci(_scores())
    after = np.random.get_state()

    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_iqm_ci_returns_three_python_floats():
    """The three numbers go into ledger rows as JSON, so they are float, not np.float64."""
    from greenleaf_control import stats

    result = stats.iqm_ci(_scores())

    assert type(result) is tuple
    assert [type(v) for v in result] == [float, float, float]


@pytest.mark.parametrize(
    "scores",
    [np.arange(5, dtype=float), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
    ids=["5-scores", "8-scores"],
)
def test_iqm_ci_refuses_a_1d_array(scores):
    """scores is runs x tasks. A 1-D array raises ValueError naming the shape it needs, rather than
    rliable's IndexError ("tuple index out of range") from inside StratifiedBootstrap."""
    from greenleaf_control import stats

    with pytest.raises(ValueError, match=r"2-D|runs"):
        stats.iqm_ci(scores)


def test_prob_improvement_of_x_plus_1_over_x_is_1():
    """Done when: prob_improvement(x + 1, x) == 1.0.

    Known answer: x is U[0, 1), so within each task every score of x + 1 exceeds every score of x
    (each column's range is under 1): P = 25 / 25 = 1 per task, 1.0 averaged.
    """
    from greenleaf_control import stats

    x = _scores()
    assert np.ptp(x, axis=0).max() < 1.0
    assert stats.prob_improvement(x + 1, x) == 1.0


def test_prob_improvement_of_x_over_itself_is_one_half():
    """Known answer: identical scores tie on every task, so P = 0.5."""
    from greenleaf_control import stats

    x = _scores()
    assert stats.prob_improvement(x, x) == 0.5


def test_prob_improvement_of_a_over_b_and_b_over_a_sum_to_1():
    """With no ties (continuous scores), every pair is won by exactly one side, so
    P(a > b) + P(b > a) = 1 on each task and on the average."""
    from greenleaf_control import stats

    a, b = _scores(seed=0), _scores(seed=1)
    assert stats.prob_improvement(a, b) + stats.prob_improvement(b, a) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # 1 task: of the 4 (a, b) pairs, 1 > 0, 3 > 2 and 3 > 0 win: 3 / 4 = 0.75
        ([[1], [3]], [[2], [0]], 0.75),
        # task 1 as above, 0.75; task 2: 10 = 10 ties (0.5), 10 > 5, 20 > 10, 20 > 5 win:
        # 3.5 / 4 = 0.875; the mean over tasks is (0.75 + 0.875) / 2 = 0.8125
        ([[1, 10], [3, 20]], [[2, 10], [0, 5]], 0.8125),
        # 3 runs of a against 2 of b: 1 > 0, 2 > 0, 3 > 0 and 3 > 2.5 win: 4 / 6 = 0.6667
        ([[1], [2], [3]], [[0], [2.5]], 4 / 6),
    ],
    ids=["one-task", "two-tasks-with-a-tie", "unequal-run-counts"],
)
def test_prob_improvement_hand_computed(a, b, expected):
    """P(a > b) + 0.5 P(a = b) over every (run of a, run of b) pair within a task, averaged over
    tasks; a and b may have different numbers of runs."""
    from greenleaf_control import stats

    got = stats.prob_improvement(np.array(a, dtype=float), np.array(b, dtype=float))
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("a_tasks", "b_tasks"), [(3, 8), (8, 3)], ids=["a-has-fewer-tasks", "b-has-fewer-tasks"]
)
def test_prob_improvement_refuses_mismatched_task_counts(a_tasks, b_tasks):
    """Tasks pair up column by column, so a and b need the same number of tasks. rliable reads
    a's task count only: with fewer tasks in a it silently compares just those columns
    (prob_improvement(x[:, :3], x) == 0.5), and with fewer in b it raises IndexError. Each order
    raises ValueError naming the task counts. Different run counts stay allowed: 3 runs of x
    against all 5 gives a float."""
    from greenleaf_control import stats

    x = _scores()
    with pytest.raises(ValueError, match="task"):
        stats.prob_improvement(x[:, :a_tasks], x[:, :b_tasks])
    assert type(stats.prob_improvement(x[:3], x)) is float


def test_prob_improvement_returns_a_python_float():
    """rliable returns np.float64; prob_improvement returns float, for the ledger's JSON."""
    from greenleaf_control import stats

    x = _scores()
    assert type(stats.prob_improvement(x + 1, x)) is float
    assert type(stats.prob_improvement(x, _scores(seed=1))) is float
