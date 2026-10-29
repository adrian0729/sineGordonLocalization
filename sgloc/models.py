"""Neural-network models.

V1CnnBiLstm re-implements the paper's network (Sec. 2.3 and Appendix) in PyTorch, keeping the
tf.keras defaults it was built with: glorot-uniform kernels, zero biases, orthogonal recurrent
kernels, forget-gate bias 1, and LSTM dropout masks that stay fixed across timesteps.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dropout_mask(like, rate):
    keep = 1.0 - rate
    return torch.bernoulli(torch.full_like(like, keep)) / keep


class KerasLSTM(nn.Module):
    """One-direction LSTM returning the final hidden state, matching tf.keras.layers.LSTM.

    Gate order is (i, f, c, o) with sigmoid gates. `dropout` masks the inputs and
    `recurrent_dropout` masks the previous hidden state, each with one mask per sequence.
    """

    def __init__(self, input_size, units, dropout=0.0, recurrent_dropout=0.0):
        super().__init__()
        self.units, self.dropout, self.recurrent_dropout = units, dropout, recurrent_dropout
        self.kernel = nn.Linear(input_size, 4 * units)
        self.recurrent = nn.Linear(units, 4 * units, bias=False)
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.orthogonal_(self.recurrent.weight)
        with torch.no_grad():
            self.kernel.bias.zero_()
            self.kernel.bias[units:2 * units] = 1.0

    def forward(self, x):  # (batch, time, features) -> (batch, units)
        batch, steps, _ = x.shape
        if self.training and self.dropout > 0:
            x = x * _dropout_mask(x[:, :1], self.dropout)
        projected = self.kernel(x)  # all timesteps in one matrix multiply
        h = x.new_zeros(batch, self.units)
        c = x.new_zeros(batch, self.units)
        rec_mask = None
        if self.training and self.recurrent_dropout > 0:
            rec_mask = _dropout_mask(h, self.recurrent_dropout)
        for t in range(steps):
            z = projected[:, t] + self.recurrent(h if rec_mask is None else h * rec_mask)
            i, f, g, o = z.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        return h


class V1CnnBiLstm(nn.Module):
    """The paper's network.

    TimeDistributed Conv1D x4 (32, 64, 128, 256 filters, kernel 3, same padding, ReLU)
    -> TimeDistributed Flatten -> Bidirectional LSTM(128, dropout 0.2, recurrent dropout 0.2)
    -> Dense(64, ReLU) -> a "does it localize" logit and 100 site logits.
    Input shape (batch, time, sites, channels), as in the paper.
    """

    def __init__(self, n_sites=100, in_channels=2, filters=(32, 64, 128, 256), kernel=3,
                 lstm_units=128, dense=64, dropout=0.2, recurrent_dropout=0.2):
        super().__init__()
        layers, channels = [], in_channels
        for f in filters:
            layers += [nn.Conv1d(channels, f, kernel, padding="same"), nn.ReLU()]
            channels = f
        self.conv = nn.Sequential(*layers)
        self.forward_lstm = KerasLSTM(channels * n_sites, lstm_units, dropout, recurrent_dropout)
        self.backward_lstm = KerasLSTM(channels * n_sites, lstm_units, dropout, recurrent_dropout)
        self.dense = nn.Linear(2 * lstm_units, dense)
        self.localizes = nn.Linear(dense, 1)
        self.site = nn.Linear(dense, n_sites)
        for layer in [*self.conv, self.dense, self.localizes, self.site]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch, steps, sites, channels = x.shape
        z = self.conv(x.reshape(batch * steps, sites, channels).transpose(1, 2))
        z = z.transpose(1, 2).reshape(batch, steps, -1)  # Keras Flatten order: site-major
        h = torch.cat([self.forward_lstm(z), self.backward_lstm(z.flip(1))], dim=1)
        h = F.relu(self.dense(h))
        return self.localizes(h).squeeze(1), self.site(h)


class RingConv1d(nn.Module):
    """Conv1d over sites that keeps the number of sites (circular or zero padding).

    Computed as one batched matrix multiply, which is about twice as fast as nn.Conv1d on
    Apple-silicon CPUs (PyTorch has no oneDNN there). Initialized like nn.Conv1d.
    """

    def __init__(self, c_in, c_out, kernel, dilation=1, padding="circular"):
        super().__init__()
        self.kernel, self.dilation, self.padding = kernel, dilation, padding
        bound = 1 / math.sqrt(c_in * kernel)
        self.weight = nn.Parameter(torch.empty(c_out, c_in * kernel).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(c_out).uniform_(-bound, bound))

    def forward(self, x):  # (batch, channels, N)
        n_sites = x.shape[-1]
        pad = self.dilation * (self.kernel // 2)
        if self.padding == "circular" and pad > n_sites:  # a shift by d on a ring is a shift by d mod N
            cols = torch.stack([x.roll((self.kernel // 2 - j) * self.dilation, dims=-1) for j in range(self.kernel)], dim=2)
            return self.weight @ cols.reshape(x.shape[0], -1, n_sites) + self.bias[:, None]
        if pad:
            x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1) if self.padding == "circular" else F.pad(x, (pad, pad))
        cols = torch.stack([x[..., j * self.dilation: j * self.dilation + n_sites] for j in range(self.kernel)], dim=2)
        return self.weight @ cols.reshape(x.shape[0], -1, n_sites) + self.bias[:, None]


def ring_conv(c_in, c_out, kernel, dilation=1, padding="circular"):
    return RingConv1d(c_in, c_out, kernel, dilation, padding)


class FastGRU(nn.Module):
    """GRU (PyTorch's equations) written with plain matrix multiplies; returns the last hidden
    state. Much faster than nn.GRU on CPU for the many short sequences used here."""

    def __init__(self, input_size, hidden):
        super().__init__()
        self.hidden = hidden
        self.ih = nn.Linear(input_size, 3 * hidden)
        self.hh = nn.Linear(hidden, 3 * hidden)
        bound = 1 / math.sqrt(hidden)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x):  # (batch, time, features) -> (batch, hidden)
        gates_x = self.ih(x)
        h = x.new_zeros(x.shape[0], self.hidden)
        for t in range(x.shape[1]):
            xr, xz, xn = gates_x[:, t].chunk(3, dim=1)
            hr, hz, hn = self.hh(h).chunk(3, dim=1)
            r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
            h = (1 - z) * torch.tanh(xn + r * hn) + z * h
        return h


class SpectralConv(nn.Module):
    """Global layer: keep ring modes m = 1..modes and mix channels with one complex weight per mode.

    This is the form of the linear (Floquet) growth stage, which multiplies each ring mode by
    its own gain; the mean (m = 0) is left to the local branch.
    """

    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(2, modes, c_in, c_out) / (c_in * c_out) ** 0.5)

    def basis(self, n_sites, x):
        # Fourier modes 1..M as cosine and sine columns: identical to rfft/irfft restricted to those
        # modes, but plain matrix multiplies (fast on CPU and on the Apple GPU, which lacks complex support)
        n = torch.arange(n_sites, dtype=x.dtype, device=x.device)[:, None]
        m = torch.arange(1, self.modes + 1, dtype=x.dtype, device=x.device)[None]
        phase = 2 * math.pi * n * m / n_sites
        return torch.cos(phase), torch.sin(phase)

    def forward(self, x):  # (batch, channels, N)
        n_sites = x.shape[-1]
        C, S = self.basis(n_sites, x)
        Xr, Xi = x @ C, -(x @ S)
        Wr, Wi = self.weight[0], self.weight[1]
        Yr = torch.einsum("bcm,mcd->bdm", Xr, Wr) - torch.einsum("bcm,mcd->bdm", Xi, Wi)
        Yi = torch.einsum("bcm,mcd->bdm", Xr, Wi) + torch.einsum("bcm,mcd->bdm", Xi, Wr)
        return (2.0 / n_sites) * (Yr @ C.T - Yi @ S.T)


class RingNet(nn.Module):
    """v2 backbone: a local circular CNN and a global Fourier branch, then one score per site.

    head="site" keeps the model shift-equivariant on the ring; head="dense" flattens the features
    into a v1-style dense head instead. Optional auxiliary outputs: "time" (log localization
    time), "localizes" (binary logit), "profile" (later energy profile, one value per site).
    `dropout` > 0 adds dropout after every hidden activation (none by default, so the layer
    indices and saved weights of earlier models are unchanged).
    """

    def __init__(self, c_in, n_sites=100, width=32, depth=2, kernel=5, dilations=None, modes=6,
                 local=True, spectral=True, padding="circular", head="site", aux=(), dropout=0.0):
        super().__init__()
        assert local or spectral

        def drop():
            return [nn.Dropout(dropout)] if dropout else []

        self.local = None
        if local:
            layers, channels = [], c_in
            for d in dilations or [1] * depth:
                layers += [ring_conv(channels, width, kernel, d, padding), nn.GELU(), *drop()]
                channels = width
            self.local = nn.Sequential(*layers)
        self.spectral = SpectralConv(c_in, width, modes) if spectral else None
        self.spectral_drop = nn.Dropout(dropout) if dropout else nn.Identity()
        features = width * (int(local) + int(spectral))
        if head == "site":
            self.head = nn.Sequential(ring_conv(features, width, 3, 1, padding), nn.GELU(), *drop(), ring_conv(width, 1, 1))
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(features * n_sites, 64), nn.ReLU(), *drop(),
                                      nn.Linear(64, n_sites))
        self.head_kind = head
        self.aux = nn.ModuleDict()
        for name in aux:
            self.aux[name] = ring_conv(features, 1, 1) if name == "profile" else nn.Linear(features, 1)

    def forward(self, x):
        parts = []
        if self.local is not None:
            parts.append(self.local(x))
        if self.spectral is not None:
            parts.append(self.spectral_drop(F.gelu(self.spectral(x))))
        h = torch.cat(parts, dim=1)
        scores = self.head(h)
        out = {"site": scores.squeeze(1) if self.head_kind == "site" else scores}
        for name, layer in self.aux.items():
            out[name] = layer(h).squeeze(1) if name == "profile" else layer(h.mean(dim=-1)).squeeze(1)
        return out


class FrameGRU(nn.Module):
    """Multi-frame front end: a shared circular conv per frame, then a GRU over time at every
    site (weights shared across sites), feeding the last hidden state to a RingNet."""

    def __init__(self, c_frame, n_frames, hidden=16, features=16, **backbone):
        super().__init__()
        self.frame_conv = nn.Sequential(ring_conv(c_frame, features, 5), nn.GELU())
        self.gru = FastGRU(features, hidden)
        self.backbone = RingNet(hidden, **backbone)

    def forward(self, x):  # (batch, frames, channels, N)
        b, t, c, n = x.shape
        f = self.frame_conv(x.reshape(b * t, c, n))
        f = f.reshape(b, t, -1, n).permute(0, 3, 1, 2).reshape(b * n, t, -1)
        h = self.gru(f)
        return self.backbone(h.reshape(b, n, -1).transpose(1, 2))


class Conv2dFront(nn.Module):
    """Multi-frame front end: one 2D convolution over (time, sites), valid in time and circular
    in sites, then the remaining time steps are stacked as channels for a RingNet."""

    def __init__(self, c_frame, n_frames, features=8, kt=3, kx=5, **backbone):
        super().__init__()
        self.kt = kt
        self.conv = ring_conv(c_frame * kt, features, kx)  # a 2D kernel = a ring conv over kt stacked frames
        self.backbone = RingNet(features * (n_frames - kt + 1), **backbone)

    def forward(self, x):  # (batch, frames, channels, N)
        b, t, c, n = x.shape
        windows = torch.stack([x[:, i:i + self.kt].reshape(b, self.kt * c, n) for i in range(t - self.kt + 1)], dim=1)
        z = F.gelu(self.conv(windows.reshape(-1, self.kt * c, n)))
        return self.backbone(z.reshape(b, -1, n))


class PointwiseFront(nn.Module):
    """Multi-frame front end without recurrence: a small MLP applied to each site's stacked
    frames (1x1 convolutions), then the ring backbone."""

    def __init__(self, c_in, hidden=32, layers=2, **backbone):
        super().__init__()
        mods, channels = [], c_in
        for _ in range(layers):
            mods += [ring_conv(channels, hidden, 1), nn.GELU()]
            channels = hidden
        self.mlp = nn.Sequential(*mods)
        self.backbone = RingNet(hidden, **backbone)

    def forward(self, x):  # (batch, channels, N), frames stacked as channels
        return self.backbone(self.mlp(x))


def build(spec, input_shape):
    """Model from a config dict. input_shape is (C, N) for "channels" views and (T, C, N) for "frames"."""
    spec = dict(spec)
    kind = spec.pop("kind", "ring")
    front = spec.pop("front", "early")
    if kind == "v1":
        return V1Adapter(V1CnnBiLstm(n_sites=input_shape[-1], in_channels=input_shape[-2]))
    if front == "gru":
        return FrameGRU(input_shape[-2], input_shape[0], n_sites=input_shape[-1], **spec)
    if front == "conv2d":
        return Conv2dFront(input_shape[-2], input_shape[0], n_sites=input_shape[-1], **spec)
    if front == "mlp":
        return PointwiseFront(input_shape[0], n_sites=input_shape[-1], **spec)
    return RingNet(input_shape[0], n_sites=input_shape[-1], **spec)


class V1Adapter(nn.Module):
    """V1CnnBiLstm with the dict output used by the v2 training loop (inputs in "frames" layout)."""

    def __init__(self, v1):
        super().__init__()
        self.v1 = v1

    def forward(self, x):  # (batch, frames, channels, N) -> v1 expects (batch, frames, N, channels)
        localizes, site = self.v1(x.transpose(2, 3))
        return {"site": site, "localizes": localizes}


class RingLinear(nn.Module):
    """Linear site scores from one full-ring circular filter per input channel.

    score_n = sum_c sum_j w[c, j] x[c, (n + j) mod N], plus a[i] times a second filter u when
    `amplitude=True` (a = standardized swing amplitude, so the filter can change with A).
    With a softmax over sites this is multinomial logistic regression with weight sharing, a
    convex problem. In Fourier space each filter is one gain per ring mode: the learned
    counterpart of the Floquet filter.
    """

    def __init__(self, channels, n_sites=100, amplitude=False, dtype=torch.float64):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(channels, n_sites, dtype=dtype))
        self.u = nn.Parameter(torch.zeros(channels, n_sites, dtype=dtype)) if amplitude else None

    def scores_from_spectrum(self, X, a=None):  # X = rfft of the inputs, (batch, channels, N//2+1)
        n_sites = self.w.shape[-1]
        spec = (X * torch.fft.rfft(self.w, dim=-1).conj()).sum(dim=1)
        if self.u is not None:
            spec = spec + a[:, None] * (X * torch.fft.rfft(self.u, dim=-1).conj()).sum(dim=1)
        return torch.fft.irfft(spec, n=n_sites, dim=-1)

    def forward(self, x, a=None):  # x: (batch, channels, N)
        return self.scores_from_spectrum(torch.fft.rfft(x, dim=-1), a)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
