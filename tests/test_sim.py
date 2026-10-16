import numpy as np
import pytest

from sgloc import data, floquet, sim

P = sim.Params()


@pytest.fixture(scope="module")
def runs():
    A, theta0 = data.initial_conditions(24, seed=123)
    return theta0, sim.localize(theta0, P)


def total_energy(theta, omega):
    return sim.site_energy(theta, omega, P.g, P.h, "sym").sum(axis=-1)


def test_uniform_swing_stays_uniform_and_has_the_pendulum_period():
    A = 1.75
    steps = round(floquet.period(A) / P.dt)
    theta, omega = np.full((1, 100), A), np.zeros((1, 100))
    for _ in range(steps):
        theta, omega = sim.rk4_step(theta, omega, P.dt, P.g, P.h)
    assert np.ptp(theta) < 1e-12  # no spatial structure appears without noise
    # after one period (to within one RK4 step) the swing is back near its turning point
    assert abs(theta[0, 0] - A) < 1e-3 and abs(omega[0, 0]) < 0.05


def test_energy_drift_is_tiny_and_fourth_order():
    # RK4 is not symplectic, so energy drifts slowly; it must stay far below anything that could
    # move a 10%-of-total-energy threshold, and shrink like dt^4 when the step is halved.
    _, theta0 = data.initial_conditions(8, seed=5)
    drift = []
    for dt in (0.05, 0.025):
        theta, omega = sim.evolve(theta0, 301, sim.Params(dt=dt))  # tau = 0 .. 150
        E = total_energy(theta, omega)
        drift.append(np.abs(E / E[:, :1] - 1).max())
    assert drift[0] < 1e-5
    assert drift[0] / drift[1] > 8


def test_bond_conventions_share_the_total_energy():
    _, theta0 = data.initial_conditions(4, seed=6)
    theta, omega = sim.evolve(theta0, 30, P)
    totals = [sim.site_energy(theta, omega, P.g, P.h, b).sum(axis=-1) for b in sim.BONDS]
    np.testing.assert_allclose(totals[0], totals[1], rtol=1e-12)
    np.testing.assert_allclose(totals[0], totals[2], rtol=1e-12)


def test_rotating_the_ring_rotates_every_label(runs):
    theta0, labels = runs
    shift = 37
    rotated = sim.localize(np.roll(theta0, shift, axis=1), P)
    for name in sim.LABELS:
        np.testing.assert_array_equal(rotated[f"site_{name}"], (labels[f"site_{name}"] + shift) % 100)
        np.testing.assert_array_equal(rotated[f"time_{name}"], labels[f"time_{name}"])


def test_mirroring_the_ring_swaps_left_and_right_conventions(runs):
    theta0, labels = runs
    mirrored = sim.localize(theta0[:, ::-1], P)
    np.testing.assert_array_equal(mirrored["site_right"], 99 - labels["site_left"])
    np.testing.assert_array_equal(mirrored["site_left"], 99 - labels["site_right"])
    np.testing.assert_array_equal(mirrored["site_sym"], 99 - labels["site_sym"])
    np.testing.assert_array_equal(mirrored["site_avg"], 99 - labels["site_avg"])


def test_labels_do_not_depend_on_the_step_size(runs):
    theta0, labels = runs
    fine = sim.localize(theta0, sim.Params(dt=0.025))
    for name in sim.LABELS:
        np.testing.assert_array_equal(fine[f"site_{name}"], labels[f"site_{name}"])


def test_every_run_localizes_in_the_data_range(runs):
    _, labels = runs
    assert (labels["site_right"] >= 0).all()
    assert 30 < np.nanmean(labels["time_right"]) < 90


def test_evolve_frames_match_the_v1_input_layout():
    _, theta0 = data.initial_conditions(3, seed=8)
    x = data.v1_inputs(theta0, P)
    assert x.shape == (3, 10, 100, 2) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, :, :, 1], np.repeat(theta0[:, None, :], 10, axis=1), rtol=1e-6)
    theta, omega = sim.evolve(theta0, 10, P)
    np.testing.assert_allclose(x[..., 0], sim.site_energy(theta, omega, P.g, P.h, "right"), rtol=1e-6)


def test_damping_removes_exactly_gamma_times_the_kinetic_integral():
    # dE/dtau = -gamma * sum(omega^2) for linear damping, so the energy lost over a run must match
    # gamma times the integral of sum(omega^2), here over tau = 0 .. 40 with gamma = 1e-2
    from scipy.integrate import simpson
    gamma, sample = 1e-2, 0.1
    _, theta0 = data.initial_conditions(4, seed=5)
    theta, omega = sim.evolve(theta0, 401, sim.Params(sample=sample), gamma=gamma)
    E = total_energy(theta, omega)
    lost = E[:, 0] - E[:, -1]
    np.testing.assert_allclose(lost, gamma * simpson((omega**2).sum(-1), dx=sample, axis=1), rtol=1e-5)
    assert (lost / E[:, 0] > 0.25).all()  # about 30% of the energy is gone by tau = 40


def test_per_run_damping_matches_running_each_chain_alone():
    _, theta0 = data.initial_conditions(4, seed=11)
    gamma = np.array([0.0, 1e-3, 0.0, 3e-3])
    together = sim.localize(theta0, P, gamma=gamma)
    for i, g in enumerate(gamma):
        alone = sim.localize(theta0[i:i + 1], P, gamma=g)
        for key in together:
            np.testing.assert_array_equal(together[key][i:i + 1], alone[key])
    undamped = sim.localize(theta0, P, gamma=np.zeros(4))  # zeros take the conservative path
    for key in together:
        np.testing.assert_array_equal(undamped[key], sim.localize(theta0, P)[key])


def test_damping_delays_localization_and_strong_damping_prevents_it(runs):
    theta0, labels = runs
    weak = sim.localize(theta0, P, ("right",), gamma=1e-3)
    assert (weak["site_right"] >= 0).all()  # the swing stays above threshold long enough
    assert np.median(weak["time_right"] - labels["time_right"]) > 0
    # at gamma = 1e-2 the swing decays below A_c = 1.577 within about 20 time units
    assert (sim.localize(theta0, P, ("right",), gamma=1e-2)["site_right"] < 0).all()
