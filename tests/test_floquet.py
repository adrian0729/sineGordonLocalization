"""Floquet checks: the swing period, which ring modes grow, the instability threshold, and why the global term must be strong enough for type III."""
import numpy as np

from sgloc import floquet, sim


def test_period_matches_a_direct_simulation():
    A = 1.75
    theta, omega, t, dt = np.array([[A]]), np.array([[0.0]]), 0.0, 1e-3
    while True:  # first downward zero crossing of the angle is a quarter period
        theta, omega = sim.rk4_step(theta, omega, dt, 0.0, 0.0)
        t += dt
        if theta[0, 0] <= 0:
            break
    assert abs(4 * t - floquet.period(A)) < 4 * dt


def test_only_the_five_longest_waves_grow_at_A_175():
    sigma = floquet.growth_rates(1.75, np.arange(1, 51))
    assert (sigma[:5] > 0.05).all()
    assert (np.abs(sigma[5:]) < 1e-6).all()
    assert abs(sigma[0] - 0.1185) < 1e-3


def test_instability_threshold_is_near_the_papers_asymptote():
    assert 1.57 < floquet.threshold_amplitude() < 1.585


def test_small_global_term_would_not_be_type_three():
    # with h = 0.05 (the paper's typo) the chain is unstable below 1.59 and the fastest mode is short
    sigma = floquet.growth_rates(1.50, np.arange(1, 51), h=0.05)
    assert sigma.max() > 0.1 and 5 <= sigma.argmax() + 1 <= 12
