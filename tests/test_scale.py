import numpy as np

from kinect3dgs.dataset.scale import estimate_from_samples


def samples(true_s, n=3000, noise=0.03, session="a", seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.6, 3.5, n)                    # SfM depth in model units
    d = true_s * z * (1 + rng.normal(0, noise, n))  # measured metres
    return [(session, float(di), float(zi)) for di, zi in zip(d, z)]


def test_recovers_known_scale():
    r = estimate_from_samples(samples(2.37))
    assert r.ok and abs(r.scale - 2.37) / 2.37 < 0.01 and r.residual_m < 0.1


def test_robust_to_outliers():
    s = samples(1.8)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(s), 600, replace=False):          # 20 % garbage depth
        s[i] = (s[i][0], float(rng.uniform(0.5, 4.0)), s[i][2])
    r = estimate_from_samples(s)
    assert r.ok and abs(r.scale - 1.8) / 1.8 < 0.05


def test_too_few_samples_rejected():
    r = estimate_from_samples(samples(1.0, n=500))
    assert not r.ok and "only" in r.reason


def test_noisy_depth_rejected():
    r = estimate_from_samples(samples(1.0, noise=0.6))
    assert not r.ok and "disagree" in r.reason


def test_sessions_disagreeing_rejected():
    r = estimate_from_samples(samples(1.0, session="a") + samples(1.3, session="b", seed=2))
    assert not r.ok and "sessions" in r.reason


def test_sessions_agreeing_accepted():
    r = estimate_from_samples(samples(1.0, session="a") + samples(1.04, session="b", seed=2))
    assert r.ok and set(r.per_session) == {"a", "b"}
