import numpy as np
import torch

from sgloc import losses, metrics, models


def test_v1_has_the_papers_parameter_count_and_output_shapes():
    model = models.V1CnnBiLstm()
    # conv 129,696 + 2 x LSTM 13,173,248 + dense 16,448 + heads 6,565
    assert models.count_parameters(model) == 26_499_205
    localizes, site = model(torch.randn(4, 10, 100, 2))
    assert localizes.shape == (4,) and site.shape == (4, 100)


def test_keras_lstm_matches_torch_lstm_without_dropout():
    torch.manual_seed(0)
    ours = models.KerasLSTM(7, 5).eval()
    ref = torch.nn.LSTM(7, 5, batch_first=True)
    with torch.no_grad():  # PyTorch uses the same (i, f, g, o) gate order
        ref.weight_ih_l0.copy_(ours.kernel.weight)
        ref.bias_ih_l0.copy_(ours.kernel.bias)
        ref.weight_hh_l0.copy_(ours.recurrent.weight)
        ref.bias_hh_l0.zero_()
    x = torch.randn(3, 6, 7)
    torch.testing.assert_close(ours(x), ref(x)[0][:, -1], rtol=1e-5, atol=1e-6)


def test_keras_lstm_initialization():
    lstm = models.KerasLSTM(10, 4)
    bias = lstm.kernel.bias.detach()
    assert torch.equal(bias[4:8], torch.ones(4)) and torch.equal(bias[:4], torch.zeros(4))
    w = lstm.recurrent.weight.detach()  # (16, 4) with orthonormal columns
    torch.testing.assert_close(w.T @ w, torch.eye(4), rtol=1e-5, atol=1e-5)


def test_v1_loss_ignores_runs_that_never_localize():
    site_logits = torch.randn(3, 100)
    outputs = (torch.zeros(3), site_logits)
    y_site = torch.tensor([5, -1, 7])
    total, parts = losses.v1_loss(outputs, (y_site >= 0).float(), y_site)
    expected_site = torch.nn.functional.cross_entropy(site_logits[[0, 2]], y_site[[0, 2]])
    torch.testing.assert_close(parts["site"], expected_site)
    torch.testing.assert_close(total, 0.5 * parts["localizes"] + expected_site)


def test_ring_distance_and_metrics():
    assert metrics.ring_distance(99, 0) == 1 and metrics.ring_distance(10, 60) == 50
    m = metrics.site_metrics(np.array([0, 50, 3]), np.array([99, 52, -1]))
    assert m["n"] == 2 and m["mae"] == 1.5 and m["mae_linear"] == 50.5
    assert m["acc_1"] == 0.5 and m["acc_2"] == 1.0 and m["acc_1_linear"] == 0.0
