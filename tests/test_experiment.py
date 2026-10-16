import numpy as np
import pytest
import torch

from sgloc import ablations, data, experiment, models, sim, views

VIEW = {"channels": ["theta"], "frames": [0, 1, 2], "theta0": True, "layout": "channels"}


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    """A tiny pool and the pool of its mirror-image chains, simulated independently; the trajectory
    cache goes to a temporary folder."""
    cache, views.CACHE = views.CACHE, tmp_path_factory.mktemp("cache")
    _, theta0 = data.initial_conditions(4, seed=11)
    pool = {"theta0": theta0, "meta": {}, **sim.localize(theta0)}
    flipped = np.ascontiguousarray(theta0[:, ::-1])
    mirrored = {"theta0": flipped, "meta": {}, **sim.localize(flipped)}
    yield pool, mirrored
    views.CACHE = cache


@pytest.mark.parametrize("label", ["right", "avg"])
def test_mirror_augmentation_adds_the_mirror_image_runs(pools, label):
    pool, mirrored = pools
    cfg = {"view": VIEW, "loss": {"time": 0.1}, "label": label}
    augmented, _ = experiment.prepare(dict(cfg, augment="mirror"), {s: pool for s in ("train", "val", "test")})
    truth, _ = experiment.prepare(cfg, {s: mirrored for s in ("train", "val", "test")})
    plain, _ = experiment.prepare(cfg, {s: pool for s in ("train", "val", "test")})
    train = augmented["train"]
    assert len(train["x"]) == 8 and len(augmented["val"]["x"]) == len(augmented["test"]["x"]) == 4
    for key in ("x", "site", "localizes", "log_time"):
        torch.testing.assert_close(train[key][:4], plain["train"][key])  # the original runs come first, unchanged
    # the appended runs are what simulating the mirrored chains gives: inputs, site and time labels
    torch.testing.assert_close(train["x"][4:], truth["train"]["x"], rtol=0, atol=1e-5)
    assert torch.equal(train["site"][4:], truth["train"]["site"])
    assert torch.equal(train["log_time"][4:], truth["train"]["log_time"])


def test_mirror_augmentation_refuses_inputs_that_do_not_reverse_exactly():
    x = np.zeros((2, 3, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        experiment.mirror_inputs(views.View(channels=("E",)), x)  # one-sided site energies
    with pytest.raises(ValueError):
        experiment.mirror_inputs(views.View(channels=("theta",), stride=2), x)  # sensors on even sites only
    np.testing.assert_array_equal(experiment.mirror_inputs(views.View(channels=("theta",)), x), x[..., ::-1])


def test_dropout_keeps_the_saved_layout_and_only_acts_in_training():
    spec = dict(ablations.ARCH, aux=["time"])
    plain = models.build(spec, (10, 100))
    assert models.count_parameters(plain) == 22_786  # the final model's size, as reported
    assert list(plain.state_dict()) == [f"local.{2 * i}.{p}" for i in range(7) for p in ("weight", "bias")] + [
        "head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias", "aux.time.weight", "aux.time.bias"]
    dropped = models.build(dict(spec, dropout=0.2), (10, 100))
    assert models.count_parameters(dropped) == 22_786
    dropped.load_state_dict(dict(zip(dropped.state_dict(), plain.state_dict().values())))
    x = torch.randn(3, 10, 100)
    plain.eval()
    dropped.eval()
    torch.testing.assert_close(dropped(x)["site"], plain(x)["site"])
    dropped.train()
    assert not torch.equal(dropped(x)["site"], dropped(x)["site"])


def test_phase8_reuses_phase6_runs_and_leaves_earlier_config_ids_unchanged():
    # ids of recorded runs (runs/<name>_s<seed>_<id>): the final model and the data group at 1,000 runs
    final = dict(ablations.final_recipe(), seed=0)
    assert experiment.config_id(final) == "864be0d3c4"
    assert experiment.config_id(dict(final, n_train=1000)) == "6878a7c70f"
    phase6 = {experiment.config_id(c) for c in ablations.phase6()}
    phase8 = ablations.phase8()
    controls = [c for c in phase8 if c["name"].endswith("_control")]
    assert controls and {experiment.config_id(c) for c in controls} <= phase6
    others = {experiment.config_id(c) for c in phase8 if c not in controls}
    assert len(others) == len(phase8) - len(controls) and not others & phase6
