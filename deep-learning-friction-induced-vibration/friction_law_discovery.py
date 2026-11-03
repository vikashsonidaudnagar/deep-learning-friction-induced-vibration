#!/usr/bin/env python3
"""
Friction-law discovery with a hybrid neural ODE (physics + neural network).

Model of a mass on a moving belt / disc (pin-on-disc, stick-slip rig):

    m x'' + c x' + k x = N * mu_theta(|v_r|) * tanh(v_r / eps),   v_r = v_b - x'

m, c, k are physics (known from a separate test, or learned with --learn-ck, see the warning
below). Only the friction coefficient mu_theta is a neural network. An ensemble of networks,
each trained on a block-bootstrap resample, shows how much the learned curve varies.

USAGE
-----
  Demo with synthetic data (hidden Stribeck law, so the result can be checked):
      python friction_law_discovery.py --demo

  Your own measurements:
      python friction_law_discovery.py --data path/to/folder --m 1.0 --k 20 --c 0.2

  Faster test run (2 networks instead of 5):
      python friction_law_discovery.py --demo --quick

  Redo the plots from saved models without retraining:
      python friction_law_discovery.py --demo --load friction_outputs/models.pt

Runtime on a laptop CPU: about 25 min for the default 5-network ensemble, 10 min with --quick.
A GPU helps little here (the tensors are small and the time loop is sequential).

DATA FORMAT (for --data)
------------------------
Folder containing experiments.csv with one row per test:

    file,vb,N,role
    run_015.csv,0.15,9.81,train
    run_030.csv,0.30,9.81,train
    run_065.csv,0.65,9.81,test

  file : CSV with columns  t,x  (and optionally v). SI units: s, m, m/s.
         If v is missing it is computed with a Savitzky-Golay derivative (noisier; prefer measured v).
  vb   : belt / disc sliding speed at the contact (m/s)  (for a disc: omega * radius)
  N    : normal load (N)
  role : train or test (test runs are never used for training or model selection)

VALIDATION DESIGN
-----------------
The last --val-frac of every training record (in time) is held out. Training windows come only
from the first part, so no validation sample is ever seen in training. Test records are separate
experiments and are used only for the final rollouts.

KNOWN LIMITATIONS
-----------------
* Friction at |v_r| close to 0 is weakly identified: during stick the slip speed is below the
  velocity noise. The region |v_r| < max(5 eps, 10 sigma_v) is shaded in the plots.
* With --learn-ck, damping c and the viscous slope of mu are nearly interchangeable
  (c x' = c v_b - c v_r), so c is fixed only by the mean position offset. Measure c separately
  (e.g. free decay with the contact lifted) whenever possible.
* The ensemble spread is not a calibrated confidence interval; on the demo it is reported
  together with how often it actually covers the true law.

All outputs (plots, learned curve CSV, summary JSON, model weights) go to --out.
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.signal import savgol_coeffs, savgol_filter

torch.set_default_dtype(torch.float64)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})

# demo "experiment": true system, hidden from the model except for m, c, k
DEMO_C, DEMO_K, DEMO_EPS = 0.2, 20.0, 0.005     # truth uses a sharper sign() than the model (eps 0.02)
DEMO_SPEEDS = [(0.15, "train"), (0.30, "train"), (0.50, "train"), (0.80, "train"), (1.20, "train"),
               (0.65, "test"), (1.00, "test"), (0.10, "test")]
DEMO_SIG_X, DEMO_SIG_V = 0.002, 0.01


# ----------------------------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------------------------
def get_args():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="synthetic data from a hidden Stribeck law")
    src.add_argument("--data", type=str, help="folder with experiments.csv")
    a.add_argument("--out", type=str, default="friction_outputs")
    a.add_argument("--m", type=float, default=1.0, help="mass (kg)")
    a.add_argument("--c", type=float, default=0.2, help="damping (N s/m)")
    a.add_argument("--k", type=float, default=20.0, help="stiffness (N/m)")
    a.add_argument("--learn-ck", action="store_true", help="also identify c and k (see KNOWN LIMITATIONS)")
    a.add_argument("--eps", type=float, default=0.02, help="tanh regularisation of sign(v_r) in the model (m/s)")
    a.add_argument("--fs", type=float, default=50.0, help="resample measurements to this rate (Hz)")
    a.add_argument("--window", type=float, default=0.5, help="multiple-shooting window length (s)")
    a.add_argument("--stride", type=int, default=5, help="samples between window starts")
    a.add_argument("--block", type=float, default=1.5, help="block length for the block bootstrap (s)")
    a.add_argument("--ensemble", type=int, default=5, help="number of networks")
    a.add_argument("--iters", type=int, default=600)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=3e-3)
    a.add_argument("--width", type=int, default=32)
    a.add_argument("--val-frac", type=float, default=0.15, help="last fraction of each training record held out")
    a.add_argument("--quick", action="store_true", help="2 networks, 400 iterations")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    a.add_argument("--load", type=str, default=None,
                   help="models.pt from an earlier run: skip training and only redo the plots")
    args = a.parse_args()
    if args.quick:
        args.ensemble, args.iters = 2, 400
    return args


# ----------------------------------------------------------------------------------------------
# simulator (batched, differentiable, fixed-step RK4)
# ----------------------------------------------------------------------------------------------
def make_rhs(mu_fn, m, eps):
    def rhs(x, v, p):
        vr = p["vb"] - v
        F = p["N"] * mu_fn(vr) * torch.tanh(vr / eps)
        return v, (F - p["c"] * v - p["k"] * x) / m
    return rhs


def simulate(rhs, p, n_samples, substeps, dt, x0, v0):
    """Integrate n_samples sample intervals (each = substeps RK4 steps). Returns [n_samples+1, B]."""
    x, v = x0, v0
    xs, vs = [x], [v]
    for _ in range(n_samples):
        for _ in range(substeps):
            k1x, k1v = rhs(x, v, p)
            k2x, k2v = rhs(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, p)
            k3x, k3v = rhs(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, p)
            k4x, k4v = rhs(x + dt * k3x, v + dt * k3v, p)
            x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs.append(x)
        vs.append(v)
    return torch.stack(xs), torch.stack(vs)


def substeps_for(N_max, m, eps, fs, mu_max=0.8):
    """RK4 sub-steps per sample so that dt * |d(accel)/dv| stays below ~1 (RK4 limit is 2.78)."""
    lam = N_max * mu_max / (m * eps)
    return max(1, math.ceil((1 / fs) * lam))


# ----------------------------------------------------------------------------------------------
# friction models
# ----------------------------------------------------------------------------------------------
class MuNet(nn.Module):
    """mu(|v_r|) >= 0, even in v_r (so the friction force is odd)."""
    def __init__(self, width, v_scale):
        super().__init__()
        self.v_scale = v_scale
        self.f = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                               nn.Linear(width, 1))
        with torch.no_grad():                       # start near mu = 0.35
            self.f[-1].bias.fill_(math.log(math.expm1(0.35)))

    def forward(self, vr):
        return nn.functional.softplus(self.f((vr.abs() / self.v_scale).unsqueeze(-1)).squeeze(-1))


class MuExp(nn.Module):
    """Classical exponential law mu_k + (mu_s - mu_k) exp(-alpha |v_r|), as a baseline."""
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor([math.log(math.expm1(0.3)), math.log(math.expm1(0.1)), 0.0]))

    def forward(self, vr):
        mk, dm, al = nn.functional.softplus(self.raw)
        return mk + dm * torch.exp(-al * vr.abs())

    def params(self):
        mk, dm, al = [float(z) for z in nn.functional.softplus(self.raw.detach())]
        return dict(mu_k=mk, mu_s=mk + dm, alpha=al)


class Mech(nn.Module):
    """c and k, optionally trainable (kept positive)."""
    def __init__(self, c, k, learn):
        super().__init__()
        self.lc = nn.Parameter(torch.tensor(math.log(c)), requires_grad=learn)
        self.lk = nn.Parameter(torch.tensor(math.log(k)), requires_grad=learn)

    @property
    def c(self):
        return self.lc.exp()

    @property
    def k(self):
        return self.lk.exp()


def stribeck(v, mu_k, dmu, vs, delta, beta):
    """Generalised Stribeck curve used to give the learned network a closed form."""
    return mu_k + dmu * np.exp(-(np.abs(v) / vs) ** delta) + beta * np.abs(v)


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def mu_true_demo(vr):
    a = vr.abs()
    return 0.25 + 0.20 * torch.exp(-(a / 0.25) ** 2) + 0.08 * a


def demo_data(args):
    """Synthetic rig: hidden Stribeck + viscous law, sharper sign() than the model, sensor noise."""
    g = torch.Generator().manual_seed(11)
    vbs = [s for s, _ in DEMO_SPEEDS]
    roles = [r for _, r in DEMO_SPEEDS]
    N, T, fs = args.m * 9.81, 15.0, args.fs
    sub = substeps_for(N, args.m, DEMO_EPS, fs, mu_max=0.5)
    dt = 1 / fs / sub
    rhs = make_rhs(mu_true_demo, args.m, DEMO_EPS)
    p = dict(c=torch.tensor(DEMO_C), k=torch.tensor(DEMO_K), vb=torch.tensor(vbs), N=torch.tensor(N))
    with torch.no_grad():
        X, V = simulate(rhs, p, int(T * fs), sub, dt, torch.zeros(len(vbs)), torch.zeros(len(vbs)))
    t = np.arange(X.shape[0]) / fs
    exps = []
    for j, (vb, role) in enumerate(zip(vbs, roles)):
        x = X[:, j] + DEMO_SIG_X * torch.randn(X.shape[0], generator=g)
        v = V[:, j] + DEMO_SIG_V * torch.randn(X.shape[0], generator=g)
        exps.append(dict(name=f"vb={vb:.2f}", t=t, x=x.numpy(), v=v.numpy(), vb=vb, N=N, role=role,
                         v_from_deriv=False, x_clean=X[:, j].numpy(), v_clean=V[:, j].numpy()))
    print(f"demo: hidden law mu = 0.25 + 0.20 exp(-(|v|/0.25)^2) + 0.08|v|; true c = {DEMO_C}, k = {DEMO_K}, "
          f"eps = {DEMO_EPS} (model uses eps = {args.eps}); noise sigma_x = {DEMO_SIG_X}, sigma_v = {DEMO_SIG_V}")
    return exps


def load_data(args):
    import csv
    man = os.path.join(args.data, "experiments.csv")
    if not os.path.exists(man):
        raise SystemExit(f"missing {man} (see the DATA FORMAT section at the top of this script)")
    exps = []
    with open(man) as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            path = os.path.join(args.data, row["file"])
            d = np.genfromtxt(path, delimiter=",", names=True)
            cols = [c.lower() for c in d.dtype.names]
            t = np.asarray(d[d.dtype.names[cols.index("t")]], float)
            x = np.asarray(d[d.dtype.names[cols.index("x")]], float)
            v = np.asarray(d[d.dtype.names[cols.index("v")]], float) if "v" in cols else None
            ok = np.isfinite(t) & np.isfinite(x) & (np.isfinite(v) if v is not None else True)
            t, x = t[ok], x[ok]
            v = v[ok] if v is not None else None
            fs_raw = 1 / np.median(np.diff(t))
            if fs_raw < 0.99 * args.fs:
                print(f"WARNING {row['file']}: recorded at {fs_raw:.1f} Hz, below --fs {args.fs}; upsampling adds no information")
            # resample to a uniform grid at args.fs (light low-pass first when downsampling)
            if fs_raw > 1.5 * args.fs:
                w = max(5, int(fs_raw / args.fs) * 2 + 1)
                x = savgol_filter(x, w, 3)
                v = savgol_filter(v, w, 3) if v is not None else None
            tu = np.arange(t[0], t[-1], 1 / args.fs)
            xu = np.interp(tu, t, x)
            if v is None:
                win = max(5, int(0.1 * args.fs) | 1)
                vu = savgol_filter(xu, win, 3, deriv=1, delta=1 / args.fs)
            else:
                vu = np.interp(tu, t, v)
            exps.append(dict(name=os.path.splitext(row["file"])[0], t=tu - tu[0], x=xu, v=vu,
                             vb=float(row["vb"]), N=float(row["n"]), role=row.get("role", "train").lower(),
                             v_from_deriv=v is None))
    print(f"loaded {len(exps)} experiments: "
          f"{sum(e['role'] == 'train' for e in exps)} train, {sum(e['role'] == 'test' for e in exps)} test")
    return exps


def noise_level(y, fs):
    """Measurement-noise std: robust (MAD) std of the residual from a light Savitzky-Golay smoothing,
    corrected for the part of the noise the filter itself keeps (factor 1/sqrt(1 - h0))."""
    win = max(5, int(0.1 * fs) | 1)
    h0 = savgol_coeffs(win, 3)[win // 2]
    r = y - savgol_filter(y, win, 3)
    s = 1.4826 * np.median(np.abs(r - np.median(r))) / math.sqrt(1 - h0)
    return max(s, 1e-3 * np.std(y), 1e-9)


# ----------------------------------------------------------------------------------------------
# helpers for analysis
# ----------------------------------------------------------------------------------------------
def osc_stats(x, fs, noise):
    """Vibration amplitude in the last third of a record (after light smoothing), whether it is
    sustained (not below the noise and not still decaying), and its period (hysteresis crossings)."""
    win = max(5, int(0.2 * fs) | 1)
    xs = savgol_filter(np.asarray(x, float), win, 3)
    n = len(xs)
    mid, last = xs[n // 3: 2 * n // 3], xs[2 * n // 3:]
    a_mid, a_last = (mid.max() - mid.min()) / 2, (last.max() - last.min()) / 2
    sustained = bool(a_last > 5 * noise and a_last > 0.7 * a_mid)
    period = np.nan
    if sustained:                                   # period from the last half (long stick phases at low speed)
        half = xs[n // 2:]
        y, th = half - half.mean(), 0.3 * a_last
        up, ups = y[0] > 0, []
        for i, yi in enumerate(y):
            if up and yi < -th:
                up = False
            elif not up and yi > th:
                up = True
                ups.append(i)
        if len(ups) > 1:
            period = float(np.diff(ups).mean() / fs)
    return float(a_last), period, sustained


def inverse_dynamics_mu(e, m, c, k, eps, fs):
    """Direct friction estimate from the signals (needs m, c, k): mu = (m a + c v + k x) / (N sign(v_r))."""
    win = max(7, int(0.12 * fs) | 1)
    x_s = savgol_filter(e["x"], win, 3)
    v_s = savgol_filter(e["x"], win, 3, deriv=1, delta=1 / fs) if e["v_from_deriv"] else savgol_filter(e["v"], win, 3)
    a_s = savgol_filter(v_s, win, 3, deriv=1, delta=1 / fs)
    vr = e["vb"] - v_s
    F = m * a_s + c * v_s + k * x_s
    slip = np.abs(vr) > 5 * eps                     # only where sliding is clearly happening
    return vr, F, slip, F / (e["N"] * np.sign(vr))


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    args = get_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {dev}")

    if args.demo and not args.learn_ck and (args.c, args.k) != (DEMO_C, DEMO_K):
        print(f"demo: using the demo rig's known c = {DEMO_C}, k = {DEMO_K} (ignoring --c/--k)")
        args.c, args.k = DEMO_C, DEMO_K
    if args.learn_ck:
        print("WARNING --learn-ck: c and the viscous slope of mu are nearly interchangeable; c is pinned only by the\n"
              "        mean position offset of each record. Use a separately measured c whenever possible.")

    exps = demo_data(args) if args.demo else load_data(args)
    train = [e for e in exps if e["role"] == "train"]
    if not train:
        raise SystemExit("no experiments with role=train")
    fs = args.fs

    N_max = max(e["N"] for e in exps)
    sub = substeps_for(N_max, args.m, args.eps, fs)
    dt = 1 / fs / sub
    print(f"RK4: {sub} sub-steps per sample, dt = {dt * 1e3:.2f} ms")

    sig_x = float(np.mean([noise_level(e["x"], fs) for e in train]))
    sig_v = float(np.mean([noise_level(e["v"], fs) for e in train]))
    print(f"estimated measurement noise: sigma_x = {sig_x:.2e} m, sigma_v = {sig_v:.2e} m/s")
    v_weak = max(5 * args.eps, 10 * sig_v)         # below this |v_r| the law is weakly identified

    # multiple-shooting windows; validation = last val_frac of every training record (no overlap)
    WL = max(2, int(round(args.window * fs)))
    blk = max(WL, int(round(args.block * fs)))
    W = {"train": [], "val": []}
    for j, e in enumerate(train):
        n = len(e["x"])
        n_split = int(round((1 - args.val_frac) * n))
        for part, starts in (("train", range(0, n_split - WL, args.stride)), ("val", range(n_split, n - WL, args.stride))):
            for s in starts:
                W[part].append((e["x"][s:s + WL + 1], e["v"][s:s + WL + 1], e["vb"], e["N"], j * 10 ** 6 + s // blk))
    rows = W["train"] + W["val"]
    W_x = torch.tensor(np.array([r[0] for r in rows]).T, device=dev)
    W_v = torch.tensor(np.array([r[1] for r in rows]).T, device=dev)
    W_vb = torch.tensor([r[2] for r in rows], device=dev)
    W_N = torch.tensor([r[3] for r in rows], device=dev)
    n_tr, n_va = len(W["train"]), len(W["val"])
    trn_idx = torch.arange(n_tr, device=dev)
    val_idx = torch.arange(n_tr, n_tr + n_va, device=dev)
    block_of = np.array([r[4] for r in W["train"]])
    blocks = [np.where(block_of == b)[0] for b in np.unique(block_of)]
    print(f"windows of {WL / fs:.2f} s: {n_tr} train ({len(blocks)} bootstrap blocks of {blk / fs:.1f} s), "
          f"{n_va} validation (last {args.val_frac:.0%} of each training record)")

    vr_obs = np.concatenate([e["vb"] - e["v"] for e in train])
    vr_max = float(np.percentile(np.abs(vr_obs), 99.5))
    v_scale = max(vr_max, 1e-3)

    def window_loss(rhs, idx, c, k, n_sub=sub, h=dt):
        p = dict(c=c, k=k, vb=W_vb[idx], N=W_N[idx])
        Xp, Vp = simulate(rhs, p, WL, n_sub, h, W_x[0, idx], W_v[0, idx])
        return (((Xp[1:] - W_x[1:, idx]) / sig_x) ** 2 + ((Vp[1:] - W_v[1:, idx]) / sig_v) ** 2).mean()

    def train_one(mu_model, mech, iters, lr, seed, label):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        rhs = make_rhs(mu_model, args.m, args.eps)
        params = [p for p in list(mu_model.parameters()) + list(mech.parameters()) if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        pick = rng.integers(0, len(blocks), len(blocks))                      # block bootstrap
        boot = torch.tensor(np.concatenate([blocks[i] for i in pick]), device=dev)
        hist_t, hist_v, best, best_state = [], [], 1e99, None
        t0 = time.time()
        for it in range(iters):
            b = boot[torch.randint(0, len(boot), (args.batch,), device=dev)]
            loss = window_loss(rhs, b, mech.c, mech.k)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step(); sched.step()
            hist_t.append(loss.item())
            if it % 10 == 0 or it == iters - 1:
                with torch.no_grad():
                    vl = window_loss(rhs, val_idx, mech.c, mech.k).item()
                hist_v.append((it, vl))
                if vl < best:
                    best = vl
                    best_state = ({k: v.clone() for k, v in mu_model.state_dict().items()},
                                  {k: v.clone() for k, v in mech.state_dict().items()})
            if it % 100 == 0 or it == iters - 1:
                print(f"  [{label}] iter {it:4d}  train {loss.item():8.3f}  val {hist_v[-1][1]:8.3f}  "
                      f"c={mech.c.item():.3f} k={mech.k.item():.2f}  ({time.time() - t0:.0f} s)")
        mu_model.load_state_dict(best_state[0]); mech.load_state_dict(best_state[1])
        return dict(train=hist_t, val=hist_v, best_val=best)

    # ---------------- ensemble of hybrid neural ODEs + exponential baseline ----------------
    members, mechs, hists = [], [], []
    if args.load:
        ck = torch.load(args.load, map_location=dev, weights_only=False)
        v_scale = ck["v_scale"]
        for sd, msd in zip(ck["members"], ck["mechs"]):
            mu_i = MuNet(ck["width"], v_scale).to(dev); mu_i.load_state_dict(sd)
            mech_i = Mech(args.c, args.k, False).to(dev); mech_i.load_state_dict(msd)
            members.append(mu_i.eval()); mechs.append(mech_i)
        hists = ck["hists"]
        mu_ex = MuExp().to(dev); mu_ex.load_state_dict(ck["exp_model"])
        mech_ex = Mech(args.c, args.k, False).to(dev); mech_ex.load_state_dict(ck["exp_mech"])
        hist_ex = ck["hist_exp"]
        print(f"loaded {len(members)} trained networks from {args.load}")
    else:
        for i in range(args.ensemble):
            mu_i = MuNet(args.width, v_scale).to(dev)
            mech_i = Mech(args.c, args.k, args.learn_ck).to(dev)
            print(f"training network {i + 1}/{args.ensemble}")
            hists.append(train_one(mu_i, mech_i, args.iters, args.lr, args.seed + 100 * i, f"NN {i + 1}"))
            members.append(mu_i.eval()); mechs.append(mech_i)
        print("fitting the exponential-law baseline (same iteration budget)")
        mu_ex = MuExp().to(dev)
        mech_ex = Mech(args.c, args.k, args.learn_ck).to(dev)
        hist_ex = train_one(mu_ex, mech_ex, args.iters, 3e-2, args.seed + 999, "exp")
    exp_par = mu_ex.params()

    floor_note = None
    if args.demo:   # loss of the TRUE system (true law, true eps) on the same validation windows
        sub_t = substeps_for(N_max, args.m, DEMO_EPS, fs, mu_max=0.5)
        with torch.no_grad():
            floor_note = window_loss(make_rhs(mu_true_demo, args.m, DEMO_EPS), val_idx, torch.tensor(DEMO_C),
                                     torch.tensor(DEMO_K), n_sub=sub_t, h=1 / fs / sub_t).item()

    c_id = float(np.mean([mm.c.item() for mm in mechs]))
    k_id = float(np.mean([mm.k.item() for mm in mechs]))

    # ---------------- evaluate the learned law ----------------
    vgrid = torch.linspace(0, 1.25 * vr_max, 500, device=dev)
    with torch.no_grad():
        MU = torch.stack([mu(vgrid) for mu in members]).cpu().numpy()   # [K, n]
        MU_ex = mu_ex(vgrid).cpu().numpy()
    vg = vgrid.cpu().numpy()
    mu_mean, mu_std = MU.mean(0), MU.std(0)
    band = 2 * mu_std
    inside = vg <= vr_max
    ident = inside & (vg >= v_weak)
    weak = vg < v_weak
    mu_true_g = mu_true_demo(vgrid).cpu().numpy() if args.demo else None

    # closed-form fit: to the ensemble mean (the curve shown) and to every member (parameter spread)
    def fit_cf(curve):
        p0 = [curve[inside].min(), max(curve[0] - curve[inside].min(), 1e-3), 0.2 * vr_max, 1.5, 0.0]
        popt, _ = curve_fit(stribeck, vg[inside], curve[inside], p0=p0,
                            bounds=([0, 0, 1e-4, 0.3, -1], [2, 2, 10, 4, 1]), maxfev=20000)
        return popt
    try:
        popt = fit_cf(mu_mean)
        fit_curve = stribeck(vg, *popt)
        r2_fit = 1 - ((mu_mean[inside] - fit_curve[inside]) ** 2).sum() / ((mu_mean[inside] - mu_mean[inside].mean()) ** 2).sum()
        P = np.array([fit_cf(MU[i]) for i in range(len(members))])
        to_named = lambda q: dict(mu_k=q[0], mu_s=q[0] + q[1], v_s=q[2], delta=q[3], beta=q[4])
        names = ["mu_s", "mu_k", "v_s", "delta", "beta"]
        per = [to_named(q) for q in P]
        cf = dict(**to_named(popt), R2_vs_network=r2_fit,
                  member_std={n_: float(np.std([d[n_] for d in per])) for n_ in names})
    except Exception as err:   # noqa: BLE001
        print(f"closed-form fit failed: {err}")
        popt, fit_curve, cf = None, None, None

    # rollouts from the measured initial state, every member and the baseline, all experiments
    def rollout(mu_model, mech, e):
        rhs = make_rhs(mu_model, args.m, args.eps)
        p = dict(c=mech.c, k=mech.k, vb=torch.tensor(e["vb"], device=dev), N=torch.tensor(e["N"], device=dev))
        with torch.no_grad():
            X, V = simulate(rhs, p, len(e["x"]) - 1, sub, dt,
                            torch.tensor(e["x"][0], device=dev), torch.tensor(e["v"][0], device=dev))
        return X.cpu().numpy(), V.cpu().numpy()

    roll = {}
    for e in exps:
        R = [rollout(mu, mm, e) for mu, mm in zip(members, mechs)]
        Xe, Ve = rollout(mu_ex, mech_ex, e)
        roll[e["name"]] = dict(X=np.array([r[0] for r in R]), V=np.array([r[1] for r in R]), Xe=Xe, Ve=Ve)

    stats = []
    for e in exps:
        r = roll[e["name"]]
        a_m, p_m, s_m = osc_stats(e["x"], fs, sig_x)
        nn_ = [osc_stats(x, fs, sig_x) for x in r["X"]]
        a_e, p_e, s_e = osc_stats(r["Xe"], fs, sig_x)
        pn = [q[1] for q in nn_ if np.isfinite(q[1])]
        d = dict(name=e["name"], role=e["role"], vb=e["vb"],
                 amp_meas=a_m, period_meas=p_m, sustained_meas=s_m,
                 amp_nn=float(np.mean([q[0] for q in nn_])), amp_nn_min=float(min(q[0] for q in nn_)),
                 amp_nn_max=float(max(q[0] for q in nn_)), period_nn=float(np.mean(pn)) if pn else np.nan,
                 sustained_nn=f"{sum(q[2] for q in nn_)}/{len(nn_)}",
                 amp_exp=a_e, period_exp=p_e, sustained_exp=s_e)
        if args.demo:
            a_t, p_t, s_t = osc_stats(e["x_clean"], fs, sig_x)
            d.update(amp_true=a_t, period_true=p_t, sustained_true=s_t)
        stats.append(d)

    # linear stability of steady sliding for the MODEL being simulated (tanh term included):
    # m x'' + (c + F'(v_b)) x' + k x = ...,  F(v) = N mu(v) tanh(v/eps)  ->  unstable if c < -F'(v_b)
    N_ref = float(np.mean([e["N"] for e in train]))

    def ccrit_of(mu_fn, eps, v):
        v = v.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad((N_ref * mu_fn(v) * torch.tanh(v / eps)).sum(), v)
        return -g.detach().cpu().numpy()
    vb_line = torch.linspace(max(5 * args.eps, 0.02), vr_max, 300, device=dev)
    vbl = vb_line.cpu().numpy()
    ccrit = np.array([ccrit_of(mu, args.eps, vb_line) for mu in members])
    ccrit_ex = ccrit_of(mu_ex, args.eps, vb_line)
    ccrit_true = ccrit_of(mu_true_demo, DEMO_EPS, vb_line) if args.demo else None

    # demo-only accuracy numbers
    acc = None
    if args.demo:
        err = np.abs(mu_mean - mu_true_g)
        hi_ = vbl >= v_weak
        acc = dict(curve_rmse_identified=float(np.sqrt((err[ident] ** 2).mean())),
                   curve_rmse_weak=float(np.sqrt((err[weak] ** 2).mean())),
                   curve_rmse_exp_identified=float(np.sqrt(((MU_ex - mu_true_g)[ident] ** 2).mean())),
                   band_coverage_identified=float((err[ident] <= band[ident]).mean()),
                   band_coverage_weak=float((err[weak] <= band[weak]).mean()),
                   ccrit_rmse_identified=float(np.sqrt(((ccrit.mean(0) - ccrit_true)[hi_] ** 2).mean())),
                   ccrit_rmse_exp_identified=float(np.sqrt(((ccrit_ex - ccrit_true)[hi_] ** 2).mean())),
                   mu0_network=float(mu_mean[0]), mu0_true=float(mu_true_g[0]))

    # ======================================================================================
    # PLOTS
    # ======================================================================================
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(exps)))
    saved = []
    weak_lbl = f"weakly identified (|v_r| < {v_weak:.2f} m/s: stick/creep)"

    def save(fig, name):
        path = os.path.join(args.out, name)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    # 01: measured data
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for e, col in zip(exps, colors):
        ls = "-" if e["role"] == "train" else "--"
        ax[0].plot(e["t"], e["x"], ls, color=col, lw=1, label=f"{e['name']} ({e['role']})")
        ax[1].plot(e["t"], e["v"], ls, color=col, lw=1)
    t_split = (1 - args.val_frac) * train[0]["t"][-1]
    for a_ in ax:
        a_.axvspan(t_split, train[0]["t"][-1], color="orange", alpha=0.12)
    ax[0].set(ylabel="x (m)", title="Measured records (dashed = test experiments; orange = validation part of training records)")
    ax[1].set(xlabel="t (s)", ylabel="ẋ (m/s)")
    ax[0].legend(fontsize=7, ncol=4)
    save(fig, "01_measured_data.png")

    # 02: training curves
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i, h in enumerate(hists):
        ax[0].semilogy(np.convolve(h["train"], np.ones(10) / 10, "valid"), color=f"C{i}", alpha=0.8, label=f"NN {i + 1}")
        it, vl = zip(*h["val"])
        ax[1].semilogy(it, vl, color=f"C{i}", label=f"NN {i + 1}")
    ax[0].semilogy(np.convolve(hist_ex["train"], np.ones(10) / 10, "valid"), "k--", label="exponential law")
    it, vl = zip(*hist_ex["val"])
    ax[1].semilogy(it, vl, "k--", label="exponential law")
    if floor_note:
        ax[1].axhline(floor_note, color="r", ls=":", label="true system (noise floor)")
    ax[0].set(xlabel="iteration", ylabel="normalised loss", title="Training loss (10-point average)")
    ax[1].set(xlabel="iteration", ylabel="normalised loss",
              title=f"Validation loss (last {args.val_frac:.0%} of each training record)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    save(fig, "02_training_curves.png")

    # 03: the discovered friction law (main result)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)
    id_vals = []
    for e in train:
        vr, F, slip, mu_id = inverse_dynamics_mu(e, args.m, c_id, k_id, args.eps, fs)
        id_vals.append(mu_id[slip])
        ax[0].plot(np.abs(vr[slip]), mu_id[slip], ".", ms=2, color="0.65", alpha=0.4,
                   label="inverse dynamics (direct estimate, uses m, c, k)" if e is train[0] else None)
    id_vals = np.concatenate(id_vals)
    ax[0].fill_between(vg, mu_mean - band, mu_mean + band, color="C0", alpha=0.25,
                       label=f"ensemble spread ±2σ ({len(members)} nets; not a calibrated CI)")
    ax[0].plot(vg, mu_mean, "C0", lw=2.5, label="neural μθ (ensemble mean)")
    ax[0].plot(vg, MU_ex, "C3--", lw=2, label="fitted exponential law")
    if fit_curve is not None:
        ax[0].plot(vg, fit_curve, "C2:", lw=2, label=f"closed-form fit to the network (R² = {cf['R2_vs_network']:.3f} vs network)")
    if args.demo:
        ax[0].plot(vg, mu_true_g, "k", lw=1.5, label="true (hidden) law")
    ax[0].axvspan(0, v_weak, color="#f4d6a0", alpha=0.6, label=weak_lbl)
    ax[0].axvspan(vr_max, vg[-1], color="0.88", label="no data (extrapolation)")
    pool = [mu_mean[inside] - band[inside], mu_mean[inside] + band[inside], MU_ex[inside],
            np.percentile(id_vals, [3, 97]) if len(id_vals) else []]
    if args.demo:
        pool.append(mu_true_g[inside])
    lo, hi = min(np.min(p_) for p_ in pool if len(p_)), max(np.max(p_) for p_ in pool if len(p_))
    ax[0].set(ylabel="μ", title="Discovered friction law μ(|v_r|)", ylim=(lo - 0.08 * (hi - lo), hi + 0.45 * (hi - lo)))
    ax[0].legend(fontsize=7.5, loc="upper right")
    ax[1].hist(np.abs(vr_obs), bins=80, color="C0", alpha=0.7)
    ax[1].axvspan(0, v_weak, color="#f4d6a0", alpha=0.6)
    ax[1].set(xlabel="|v_r| (m/s)", ylabel="samples", title="Where the data are (training slip speeds)")
    save(fig, "03_friction_law.png")

    # 04: friction force vs slip velocity, both signs
    fig, ax = plt.subplots(figsize=(10, 5))
    for e, col in zip(train, colors):
        vr, F, slip, _ = inverse_dynamics_mu(e, args.m, c_id, k_id, args.eps, fs)
        ax.plot(vr, F / e["N"], ".", ms=2, color=col, alpha=0.35, label=f"{e['name']} (inverse dynamics)")
    vv = torch.linspace(-vr_max, vr_max, 801, device=dev)
    with torch.no_grad():
        Fm = np.array([(mu(vv) * torch.tanh(vv / args.eps)).cpu().numpy() for mu in members])
        Fe = (mu_ex(vv) * torch.tanh(vv / args.eps)).cpu().numpy()
    vvn = vv.cpu().numpy()
    ax.fill_between(vvn, Fm.mean(0) - 2 * Fm.std(0), Fm.mean(0) + 2 * Fm.std(0), color="C0", alpha=0.25)
    ax.plot(vvn, Fm.mean(0), "C0", lw=2.5, label="neural model")
    ax.plot(vvn, Fe, "C3--", lw=1.5, label="exponential law")
    if args.demo:
        with torch.no_grad():
            ax.plot(vvn, (mu_true_demo(vv) * torch.tanh(vv / DEMO_EPS)).cpu().numpy(), "k", lw=1, label="true")
    ax.set(xlabel="slip velocity v_r (m/s)", ylabel="F / N",
           title="Friction force characteristic (stick phases sit at v_r ≈ 0; v_r < 0 never observed)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "04_friction_force_characteristic.png")

    # 05: closed-form fit residual / parameters
    if fit_curve is not None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(vg[inside], mu_mean[inside] - fit_curve[inside], "C2")
        ax[0].fill_between(vg[inside], -band[inside], band[inside], color="C0", alpha=0.2, label="network spread ±2σ")
        ax[0].axvspan(0, v_weak, color="#f4d6a0", alpha=0.6)
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set(xlabel="|v_r| (m/s)", ylabel="μ_network − μ_fit", title="How well the closed form describes the network")
        ax[0].legend(fontsize=8)
        ax[1].axis("off")
        sd = cf["member_std"]
        fmt = lambda key, unit="": f"{cf[key]:.3f} ± {sd[key]:.3f}{unit}"
        rows_ = [["μ_s (static, weakly identified)", fmt("mu_s")], ["μ_k (kinetic)", fmt("mu_k")],
                 ["v_s (Stribeck speed)", fmt("v_s", " m/s")], ["δ (shape exponent)", fmt("delta")],
                 ["β (viscous)", fmt("beta", " s/m")], ["R² (fit vs network mean)", f"{cf['R2_vs_network']:.4f}"]]
        if args.demo:
            rows_ = [r + [t] for r, t in zip(rows_, ["0.45", "0.25", "0.25 m/s", "2", "0.08 s/m", "—"])]
        tb = ax[1].table(cellText=rows_, colLabels=["parameter", "fitted (± spread over nets)"] + (["true"] if args.demo else []),
                         loc="center", cellLoc="left")
        tb.scale(1, 1.6); tb.set_fontsize(9.5)
        ax[1].set_title("μ = μ_k + (μ_s−μ_k)·exp(−(|v|/v_s)^δ) + β|v|", fontsize=10)
        save(fig, "05_closed_form_fit.png")

    # 06 / 07: free rollouts for every experiment
    for role, fname in [("train", "06_rollouts_train.png"), ("test", "07_rollouts_test.png")]:
        sel = [e for e in exps if e["role"] == role]
        if not sel:
            continue
        fig, ax = plt.subplots(len(sel), 1, figsize=(12, 2.6 * len(sel)), squeeze=False)
        for a_, e in zip(ax[:, 0], sel):
            r = roll[e["name"]]
            a_.plot(e["t"], e["x"], ".", ms=2, color="0.5", label="measured")
            a_.fill_between(e["t"], r["X"].min(0), r["X"].max(0), color="C0", alpha=0.25)
            a_.plot(e["t"], np.median(r["X"], 0), "C0", lw=1.5, label="hybrid neural ODE (median; band = ensemble range)")
            a_.plot(e["t"], r["Xe"], "C3--", lw=1, label="exponential law")
            if role == "train":
                a_.axvspan(t_split, e["t"][-1], color="orange", alpha=0.12)
            a_.set(ylabel="x (m)", title=f"{e['name']}  ({role}, v_b = {e['vb']} m/s): free rollout from the first sample")
        ax[-1, 0].set_xlabel("t (s)")
        ax[0, 0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
        save(fig, fname)

    # 08: phase portraits (one network: a median across nets would mix cycles that drift out of phase)
    best_i = int(np.argmin([h["best_val"] for h in hists]))
    ncol = min(4, len(exps))
    nrow = math.ceil(len(exps) / ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow), squeeze=False)
    for a_, e in zip(ax.flat, exps):
        h = len(e["x"]) // 2
        r = roll[e["name"]]
        a_.plot(e["x"][h:], e["v"][h:], ".", ms=1.5, color="0.55", label="measured")
        a_.plot(r["X"][best_i][h:], r["V"][best_i][h:], "C0", lw=1.5, label="hybrid neural ODE (best-validation net)")
        a_.plot(r["Xe"][h:], r["Ve"][h:], "C3--", lw=1, label="exponential")
        a_.axhline(e["vb"], color="k", ls=":", lw=0.8)
        a_.set(title=f"{e['name']} ({e['role']})", xlabel="x (m)", ylabel="ẋ (m/s)")
    for a_ in ax.flat[len(exps):]:
        a_.axis("off")
    ax.flat[0].legend(fontsize=7)
    fig.suptitle("Phase portraits, second half of each record (dotted line = belt speed)")
    save(fig, "08_phase_portraits.png")

    # 09: amplitude and period vs belt speed
    S = sorted(stats, key=lambda s: s["vb"])
    vbx = np.array([s["vb"] for s in S])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    am = np.array([s["amp_meas"] for s in S]) * 1e3
    sus = np.array([s["sustained_meas"] for s in S])
    ax[0].plot(vbx[sus], am[sus], "ko", ms=8, label="measured: sustained stick-slip")
    ax[0].plot(vbx[~sus], am[~sus], "o", mfc="none", mec="k", ms=8, label="measured: decays / steady sliding")
    if args.demo:
        ax[0].plot(vbx, [s["amp_true"] * 1e3 for s in S], "k+", ms=10, label="true (noise-free)")
    nn_mean = np.array([s["amp_nn"] for s in S]) * 1e3
    nn_err = np.array([[s["amp_nn"] - s["amp_nn_min"], s["amp_nn_max"] - s["amp_nn"]] for s in S]).T * 1e3
    ax[0].errorbar(vbx + 0.012, nn_mean, yerr=nn_err, fmt="s", color="C0", capsize=3, label="hybrid neural ODE (ensemble range)")
    ax[0].plot(vbx - 0.012, [s["amp_exp"] * 1e3 for s in S], "^", color="C3", label="exponential law")
    ax[0].axhline(5 * sig_x * 1e3, color="0.4", ls=":", lw=1, label="detection threshold (5σ_x)")
    ax[1].plot(vbx, [s["period_meas"] for s in S], "ko", ms=8, label="measured")
    ax[1].plot(vbx + 0.012, [s["period_nn"] for s in S], "s", color="C0", label="hybrid neural ODE")
    ax[1].plot(vbx - 0.012, [s["period_exp"] for s in S], "^", color="C3", label="exponential law")
    for a_ in ax:
        for s in S:
            if s["role"] == "test":
                a_.axvline(s["vb"], color="orange", alpha=0.3, lw=8)
    ax[0].set(xlabel="belt speed v_b (m/s)", ylabel="amplitude, last third of record (mm)",
              title="Vibration amplitude (orange = test speeds)")
    ax[1].set(xlabel="belt speed v_b (m/s)", ylabel="period (s)", title="Stick-slip period (sustained cases only)")
    ax[0].set_ylim(top=1.45 * max(np.nanmax(am), np.nanmax(nn_mean + nn_err[1])))
    ax[0].legend(fontsize=7, loc="upper left", ncol=2); ax[1].legend(fontsize=8)
    save(fig, "09_amplitude_period_vs_speed.png")

    # 10: stability boundary predicted by the learned law, with the observed regimes
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(vbl[0], max(v_weak, vbl[0]), color="#f4d6a0", alpha=0.6, label=weak_lbl)
    ax.fill_between(vbl, ccrit.min(0), ccrit.max(0), color="C0", alpha=0.25)
    ax.plot(vbl, ccrit.mean(0), "C0", lw=2, label="from neural μθ (band = ensemble range)")
    ax.plot(vbl, ccrit_ex, "C3--", label="from exponential law")
    if args.demo:
        ax.plot(vbl, ccrit_true, "k", lw=1.2, label="true system")
    ax.axhline(c_id, color="g", ls="-.", label=f"rig damping c = {c_id:.3f}")
    ax.axhline(0, color="k", lw=0.6)
    for s in S:
        ax.plot(s["vb"], c_id, "o", ms=9, mfc="C3" if s["sustained_meas"] else "none", mec="C3", zorder=5)
    ax.plot([], [], "o", mfc="C3", mec="C3", label="observed: sustained stick-slip")
    ax.plot([], [], "o", mfc="none", mec="C3", label="observed: no sustained vibration")
    ax.set(xlabel="belt speed v_b (m/s)", ylabel="c_crit = −dF/dv_r at v_b, per N·s/m",
           title="Linear stability of steady sliding (unstable if c < c_crit). Stick-slip can also coexist\n"
                 "with stable sliding (bistability), so filled markers above the curve are expected.")
    ax.legend(fontsize=7.5)
    save(fig, "10_stability_boundary.png")

    # 11: residuals on validation windows
    with torch.no_grad():
        rhs0 = make_rhs(members[0], args.m, args.eps)
        p = dict(c=mechs[0].c, k=mechs[0].k, vb=W_vb[val_idx], N=W_N[val_idx])
        Xp, Vp = simulate(rhs0, p, WL, sub, dt, W_x[0, val_idx], W_v[0, val_idx])
        rx = (Xp - W_x[:, val_idx]).cpu().numpy()[1:].ravel()
        rv = (Vp - W_v[:, val_idx]).cpu().numpy()[1:].ravel()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].hist(rx * 1e3, bins=80, color="C0")
    ax[0].set(xlabel="x error (mm)", title=f"Validation residuals: σ = {rx.std() * 1e3:.2f} mm (noise est. {sig_x * 1e3:.2f})")
    ax[1].hist(rv, bins=80, color="C1")
    ax[1].set(xlabel="ẋ error (m/s)", title=f"σ = {rv.std():.4f} m/s (noise est. {sig_v:.4f})")
    save(fig, "11_residuals.png")

    # ---------------- files ----------------
    np.savetxt(os.path.join(args.out, "learned_friction_curve.csv"),
               np.column_stack([vg, mu_mean, mu_std, mu_mean - band, mu_mean + band, MU_ex, (vg < v_weak).astype(int)]),
               delimiter=",", header="v_r_abs,mu_mean,mu_std,mu_lower_2sd,mu_upper_2sd,mu_exponential_fit,weakly_identified",
               comments="")
    torch.save(dict(members=[m.state_dict() for m in members], mechs=[m.state_dict() for m in mechs],
                    v_scale=v_scale, width=members[0].f[0].out_features, c=c_id, k=k_id, eps=args.eps,
                    exp_params=exp_par, exp_model=mu_ex.state_dict(), exp_mech=mech_ex.state_dict(),
                    hists=hists, hist_exp=hist_ex), os.path.join(args.out, "models.pt"))
    summary = dict(mode="demo" if args.demo else "data", device=str(dev), n_train_windows=n_tr, n_val_windows=n_va,
                   observed_vr_max=vr_max, weakly_identified_below=v_weak,
                   noise_sigma_x=sig_x, noise_sigma_v=sig_v, c=c_id, k=k_id, learned_ck=args.learn_ck,
                   best_val_loss_members=[h["best_val"] for h in hists], best_val_loss_exponential=hist_ex["best_val"],
                   noise_floor_val_loss=floor_note, exponential_params=exp_par, closed_form_fit=cf,
                   demo_accuracy=acc, experiments=stats)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---------------- console summary ----------------
    print("\n=================== RESULTS ===================")
    member_vals = ", ".join("%.2f" % h["best_val"] for h in hists)
    print(f"validation loss (held-out last {args.val_frac:.0%} of training records): neural "
          f"{np.mean([h['best_val'] for h in hists]):.2f} (members {member_vals})"
          f"   exponential {hist_ex['best_val']:.2f}" + (f"   true system {floor_note:.2f}" if floor_note else ""))
    if args.learn_ck:
        print(f"identified c = {c_id:.4f}, k = {k_id:.3f}  (see the --learn-ck warning)")
    if cf:
        sd = cf["member_std"]
        print("closed form (fit to network, ± spread over nets): " + ", ".join(
            f"{n_}={cf[n_]:.3f}±{sd[n_]:.3f}" for n_ in ["mu_s", "mu_k", "v_s", "delta", "beta"]))
    print(f"exponential law: " + ", ".join(f"{k_}={v_:.3f}" for k_, v_ in exp_par.items()))
    if acc:
        print(f"curve RMSE vs true law, |v_r| >= {v_weak:.2f}: neural {acc['curve_rmse_identified']:.4f}, "
              f"exponential {acc['curve_rmse_exp_identified']:.4f};  below {v_weak:.2f} (weak): neural {acc['curve_rmse_weak']:.4f}")
        print(f"±2σ ensemble band covers the true law on {acc['band_coverage_identified']:.0%} of the identified range "
              f"and {acc['band_coverage_weak']:.0%} of the weak range;  mu(0): network {acc['mu0_network']:.3f}, true {acc['mu0_true']:.3f}")
        print(f"c_crit RMSE vs truth for v_b >= {v_weak:.2f}: neural {acc['ccrit_rmse_identified']:.3f}, "
              f"exponential {acc['ccrit_rmse_exp_identified']:.3f} N·s/m")
    hdr = f"{'experiment':<12}{'role':<7}{'state (meas)':<15}{'amp meas':>9}" + (f"{'amp true':>9}" if args.demo else "") + \
          f"{'amp NN':>9}{'amp exp':>9}{'T meas':>8}{'T NN':>8}{'T exp':>8}  NN sustained  exp sustained"
    print(hdr)
    for s in S:
        print(f"{s['name']:<12}{s['role']:<7}{'stick-slip' if s['sustained_meas'] else 'decays/steady':<15}"
              f"{s['amp_meas'] * 1e3:9.1f}" + (f"{s['amp_true'] * 1e3:9.1f}" if args.demo else "") +
              f"{s['amp_nn'] * 1e3:9.1f}{s['amp_exp'] * 1e3:9.1f}{s['period_meas']:8.3f}{s['period_nn']:8.3f}"
              f"{s['period_exp']:8.3f}  {s['sustained_nn']:>12}  {str(s['sustained_exp']):>13}")
    print(f"\nsaved {len(saved)} plots + learned_friction_curve.csv, summary.json, models.pt in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
