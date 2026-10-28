def test_tomato_env_makes_resets_and_steps():
    """Smoke test for the control/ scaffold (issue #1).

    Proves our package and the pinned simulator import together, that
    gl_gym/GreenLightTomato-v0 builds for a one-day season, resets with seed 0 and takes one
    step with a zero action, returning gymnasium's 5-tuple. Structure only: upstream is
    untrusted and its default weather is unchecked, so no reward or state value is asserted.
    """
    # Local imports, so a missing package FAILS in this body instead of ERRORing at collection.
    import greenleaf_control

    assert greenleaf_control.__name__ == "greenleaf_control"

    import gl_gym  # noqa: F401  (importing gl_gym registers its env ids with gymnasium)
    import gymnasium as gym
    import numpy as np

    env = gym.make("gl_gym/GreenLightTomato-v0", season_length=1)
    obs, info = env.reset(seed=0)
    assert env.unwrapped.N == 96  # 1 day = 86,400 s / 900 s per step = 96 steps
    flat = gym.spaces.flatten(env.observation_space, obs)
    assert flat.shape == (gym.spaces.flatdim(env.observation_space),)
    assert np.all(np.isfinite(flat))
    assert isinstance(info, dict)

    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

    flat = gym.spaces.flatten(env.observation_space, obs)
    assert flat.shape == (gym.spaces.flatdim(env.observation_space),)
    assert np.all(np.isfinite(flat))
    assert isinstance(reward, float | np.floating)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool | np.bool_)
    assert isinstance(truncated, bool | np.bool_)
    assert not terminated and not truncated  # step 1 of 96: no season end, no solver failure
    assert isinstance(info, dict)
