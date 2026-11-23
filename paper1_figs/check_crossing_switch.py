"""MSBQA paper 1 -- direct check that classical KPO trajectories do not switch configuration at the crossing p_c,
and (optional, --exact-sigma) that TWA equals classical noise when the noise amplitudes are exactly equal.
New classical simulations (small; CPU).  Run from msbqa_phase1/:
    python paper1_figs/check_crossing_switch.py --probe                       # timing estimate, seconds
    python paper1_figs/check_crossing_switch.py --exact-sigma --workers 180   # both checks, one instance per task
(121 + 2 x 20 tasks; with >= 121 workers the wall time is about one task, i.e. a few minutes.)
    python paper1_figs/check_crossing_switch.py --workers 180                 # switching check only
    python paper1_figs/check_crossing_switch.py --exact-sigma --skip-switch --workers 180   # exact-sigma only
Writes paper1_figs/report_crossing_switch.txt, per_traj_switch_summary.csv (and report_exact_sigma.txt).

Model: Goto 2016 App. E classical KPO network = weyl-family Weyl flow (detuning Delta_i = xi0 sum_j |J_ij|),
    dz/dt = -i[(Delta + K|z|^2) z - p(t) z* - xi0 J z],  p = pf t / T,  K = 1, xi0 = 0.5, pf = 7, T = 700, RK4,
the same equations as kpo_common.classical_run (layer L2g).  Instances: G4 regenerated with seed 2016 (identical to
gen_instances.py; the trap column is checked against branches_weyl.csv).
Switching check: crossing instances (branches_weyl.csv cross = 1) + controls (trap without crossing; 30 instances with
neither).  R trajectories per instance, initial noise uniform(+-eps) on both quadratures.  Every 0.01 in p the sign
pattern of Re z averaged over that interval (Z2 representative with s_0 = +1) is recorded; departure = first checkpoint with max|z| > 0.1.
Per trajectory: configuration 0.2 after departure, at p_c - 0.1, at p_c + 0.1, final; number of configuration changes
after that point.
Note (run of 10-31 with the INSTANTANEOUS sign and a 0.05 settling window): 97 % of crossing-instance trajectories
showed 'switches' and the configuration at departure matched the final one for only 58 % -- the instantaneous sign
flickers, so that version could not answer the question; this version averages the sign.
Prediction (paper 1, C2): on crossing instances the configuration chosen at departure is kept through p_c, i.e.
almost no trajectory switches into the ground configuration at p_c; success = share that departed into the ground
configuration.
--exact-sigma: 20 instances (the 8 with |z| >= 3 in check_twa_vs_noise + 12 others), R = 1000:
uniform(+-0.3) [sigma = 0.1732] vs Gaussian sigma = 0.1732 (TWA at h = 0.12), same code and dt (--dt-exact,
default 0.005; dt 0.0025 vs 0.005 changes P by <= 0.006, phase 1 G1); also compares the
uniform run with the existing NS eps = 0.3 result (integrator sanity).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = lambda *a: os.path.join(ROOT, *a)
K, XI0, PF, T = 1.0, 0.5, 7.0, 700.0
ap = argparse.ArgumentParser()
ap.add_argument("--R", type=int, default=200)
ap.add_argument("--eps", type=float, default=1e-6)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--exact-sigma", action="store_true")
ap.add_argument("--probe", action="store_true", help="time 2000 steps and exit")
ap.add_argument("--skip-switch", action="store_true", help="only the exact-sigma check")
ap.add_argument("--dt-exact", type=float, default=0.005)
ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes (one instance per task)")
a = ap.parse_args()
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


# ------------------------------------------------------------------ instances (gen_instances.py recipe)
n = 4
rng0 = np.random.default_rng(2016)
iu = np.triu_indices(n, 1)
Js = np.zeros((1000, n, n))
for k in range(1000):
    J = np.zeros((n, n))
    J[iu] = rng0.uniform(-1.0, 1.0, len(iu[0]))
    Js[k] = J + J.T
Ds = XI0 * np.abs(Js).sum(2)
S = np.array([[1 - 2 * ((b >> i) & 1) for i in range(n)] for b in range(2 ** n)], float)
E = -0.5 * np.einsum("bi,kij,bj->kb", S, Js, S)
br = pd.read_csv(P("run_v4", "branches", "branches_weyl.csv")).set_index("inst")


def rep(idx):
    """Z2 representative with s_0 = +1 (bit 0 clear)."""
    idx = np.asarray(idx)
    return np.where(idx & 1, idx ^ (2 ** n - 1), idx)


ground = np.array([rep(int(np.argmin(E[k]))) for k in range(1000)])
trap_re = []
for k in range(1000):
    w, U = np.linalg.eigh(np.diag(Ds[k]) - XI0 * Js[k])
    b = int(((U[:, 0] < 0).astype(int) << np.arange(n)).sum())
    trap_re.append(int(E[k, b] > E[k].min() + 1e-12))
assert np.array_equal(np.array(trap_re), br.trap.to_numpy()), "regenerated instances do not match branches_weyl.csv"


def make_init(spec):
    kind, val = spec
    if kind == "u":
        return lambda rng, sh: rng.uniform(-val, val, sh) + 1j * rng.uniform(-val, val, sh)
    return lambda rng, sh: val * (rng.standard_normal(sh) + 1j * rng.standard_normal(sh))


def _block(args):
    """worker: RK4 for instances idx, R trajectories each; returns patterns (B,R,ncp), maxabs, p grid."""
    idx, R, spec, dt, rec_every_p, seed, probe = args
    init = make_init(spec)
    rng = np.random.default_rng(seed)
    B = len(idx)
    J = Js[idx]                                      # (B,n,n)
    D = Ds[idx][:, None, :]                          # (B,1,n)
    z = init(rng, (B, R, n))
    nsteps = int(round(T / dt))
    every = max(1, int(round(rec_every_p * T / PF / dt)))
    ncp = nsteps // every
    pats = np.zeros((B, R, ncp), np.int8)
    mx = np.zeros((B, R, ncp), np.float32)
    pg = np.zeros(ncp)
    bits = (1 << np.arange(n))

    def f(z, t):
        p = PF * t / T
        return -1j * ((D + K * (z.real ** 2 + z.imag ** 2)) * z - p * np.conj(z) - XI0 * (z @ J))

    t = 0.0
    c = 0
    xs = np.zeros(z.shape)                           # Re z summed since the last checkpoint
    for s in range(1, nsteps + 1):
        k1 = f(z, t)
        k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(z + dt * k3, t + dt)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = s * dt
        xs += z.real
        if s % every == 0 and c < ncp:
            # configuration = sign of Re z AVERAGED over the last checkpoint interval (the instantaneous sign
            # oscillates around the cat fixed points shortly after departure)
            pats[:, :, c] = rep(((xs < 0) * bits).sum(-1))
            xs[:] = 0.0
            mx[:, :, c] = np.abs(z).max(-1)
            pg[c] = PF * t / T
            c += 1
        if probe and s == 2000:
            return None
    return pats, mx, pg


def integrate(idx, R, spec, dt, rec_every_p=0.01, seed=0):
    """one task per instance on a.workers processes; spec = ("u", eps) uniform(+-eps) or ("g", sigma) Gaussian.
    Seed of instance k: seed * 100003 + k (results do not depend on the number of workers)."""
    tasks = [(np.array([k]), R, spec, dt, rec_every_p, seed * 100003 + int(k), False) for k in idx]
    if a.workers <= 1:
        out = [_block(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(tasks)), mp_context=mp.get_context("fork")) as ex:
            out = list(ex.map(_block, tasks))
    return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), out[0][2])


if a.probe:
    t0 = time.time()
    _block((np.array([0]), a.R, ("u", a.eps), a.dt, 0.01, 0, True))
    dtw = time.time() - t0
    per = dtw * T / a.dt / 2000
    nt = 121
    print(f"probe: one instance x R {a.R}, 2000 steps in {dtw:.2f} s -> one task ~{per:.0f} s; switching check with "
          f"{a.workers} workers ~{per * -(-nt // a.workers) / 60:.1f} min (121 tasks)")
    raise SystemExit

# ------------------------------------------------------------------ switching check
cross = br.index[br.cross == 1].to_numpy()
trapnc = br.index[(br.cross == 0) & (br.trap == 1)].to_numpy()
neither = br.index[(br.cross == 0) & (br.trap == 0)].to_numpy()[:30]
idx = np.concatenate([cross, trapnc, neither])
grp = np.array(["crossing"] * len(cross) + ["trap, no crossing"] * len(trapnc) + ["neither"] * len(neither))
def run_switch():
    say(f"== classical configuration tracking (Goto App. E flow, T = {T:g}, pf = {PF:g}, dt = {a.dt}, R = {a.R}, "
        f"noise uniform +-{a.eps:g})")
    say(f"   instances: crossing {len(cross)}, trap without crossing {len(trapnc)}, neither {len(neither)}")
    t0 = time.time()
    pats, mx, pg = integrate(idx, a.R, ("u", a.eps), a.dt, seed=1)
    say(f"   integration {time.time() - t0:.0f} s")

    rows = []
    for b, k in enumerate(idx):
        pc = br.p_c[k] if br.cross[k] == 1 else np.nan
        fc = rep(int(br.first_cfg[k]))
        for r in range(a.R):
            dep = np.argmax(mx[b, r] > 0.1) if (mx[b, r] > 0.1).any() else -1
            if dep < 0:
                rows.append(dict(inst=k, group=grp[b], departed=0))
                continue
            settle = dep + 20                              # 0.2 in p after departure
            seq = pats[b, r, min(settle, len(pg) - 1):]
            nsw = int((np.diff(seq) != 0).sum())
            c_dep = int(pats[b, r, min(settle, len(pg) - 1)])
            c_fin = int(pats[b, r, -1])
            if np.isfinite(pc):
                i_bef = max(np.searchsorted(pg, pc - 0.1) - 1, 0)
                i_aft = min(np.searchsorted(pg, pc + 0.1), len(pg) - 1)
                c_bef, c_aft = int(pats[b, r, i_bef]), int(pats[b, r, i_aft])
            else:
                c_bef = c_aft = -1
            rows.append(dict(inst=k, group=grp[b], departed=1, p_dep=pg[dep], p_c=pc, c_dep=c_dep, c_bef=c_bef,
                             c_aft=c_aft, c_fin=c_fin, n_switch=nsw, ground=int(ground[k]), first_cfg=fc,
                             dep_ground=int(c_dep == ground[k]), dep_first=int(c_dep == fc), fin_ground=int(c_fin == ground[k]),
                             switch_at_pc=int(c_bef != c_aft) if c_bef >= 0 else -1,
                             to_ground_at_pc=int(c_bef != ground[k] and c_aft == ground[k]) if c_bef >= 0 else -1))
    tr = pd.DataFrame(rows)
    say()
    say("   group                N inst  departed  P_succ(final)  dep. into ground  dep. into first cfg  "
        "trajectories with any switch after departure")
    for g in ("crossing", "trap, no crossing", "neither"):
        s = tr[(tr.group == g) & (tr.departed == 1)]
        say(f"   {g:20s} {s.inst.nunique():4d}   {len(s) / max(1, (tr.group == g).sum()):.3f}     {s.fin_ground.mean():.3f}"
            f"          {s.dep_ground.mean():.3f}             {s.dep_first.mean():.3f}               "
            f"{(s.n_switch > 0).mean():.4f}")
    s = tr[(tr.group == "crossing") & (tr.departed == 1)]
    say()
    say("-- crossing instances: what happens at p_c")
    say(f"   trajectories whose configuration changes between p_c - 0.1 and p_c + 0.1: {(s.switch_at_pc == 1).mean():.4f}"
        f"  ({int((s.switch_at_pc == 1).sum())} of {len(s)})")
    say(f"   ... of which into the ground configuration: {int((s.to_ground_at_pc == 1).sum())}")
    say(f"   final success = departure into ground: identical for {(s.fin_ground == s.dep_ground).mean():.4f} of trajectories")
    say(f"   mean final success {s.fin_ground.mean():.3f};  mean share departing into the ground configuration "
        f"{s.dep_ground.mean():.3f}")
    per = s.groupby("inst").agg(P=("fin_ground", "mean"), dep_g=("dep_ground", "mean"), dep_f=("dep_first", "mean"),
                                sw=("n_switch", lambda x: (x > 0).mean()), p_c=("p_c", "first"),
                                p_dep=("p_dep", "median"))
    per.to_csv(os.path.join(HERE, "per_instance_switch_crossing.csv"), float_format="%.4f")
    say("   per instance (first 15): inst  P_final  dep->ground  dep->first_cfg  any-switch  p_dep  p_c")
    for k, r in per.head(15).iterrows():
        say(f"     {k:4d}   {r.P:.3f}    {r.dep_g:.3f}       {r.dep_f:.3f}        {r.sw:.3f}    {r.p_dep:.2f}  {r.p_c:.2f}")
    tr.groupby(["inst", "group"]).agg(P=("fin_ground", "mean"), dep_ground=("dep_ground", "mean"),
                                       any_switch=("n_switch", lambda x: (x > 0).mean())).to_csv(
        os.path.join(HERE, "per_traj_switch_summary.csv"), float_format="%.4f")
    with open(os.path.join(HERE, "report_crossing_switch.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


# ------------------------------------------------------------------ exact-sigma TWA check
def run_exact():
    L.clear()
    flagged = [9, 37, 51, 53, 57, 79, 90, 98]
    others = [i for i in range(100) if i not in flagged][:12]
    ex = np.array(flagged + others)
    sig = 0.3 / np.sqrt(3.0)
    say(f"== exact-sigma check: uniform(+-0.3) vs Gaussian sigma = {sig:.4f} (= TWA at h = {4 * sig ** 2:.3f}), "
        f"R = 1000, dt = {a.dt_exact}, {len(ex)} instances")
    t0 = time.time()
    pu = integrate(ex, 1000, ("u", 0.3), a.dt_exact, rec_every_p=7.0, seed=11)[0][:, :, -1]
    pgau = integrate(ex, 1000, ("g", sig), a.dt_exact, rec_every_p=7.0, seed=12)[0][:, :, -1]
    say(f"   integration {time.time() - t0:.0f} s")
    Pu = (pu == ground[ex][:, None]).mean(1)
    Pg = (pgau == ground[ex][:, None]).mean(1)
    old = np.array([[e["P_succ"] for e in json.load(open(P("run_v4", "parts", f"NS_{i}.json")))["layers"]["L2g"]
                     if abs(e["eps"] - 0.3) < 1e-12][0] for i in ex])
    se = np.sqrt(np.clip((Pu + Pg) / 2, 5e-4, 1 - 5e-4) * (1 - np.clip((Pu + Pg) / 2, 5e-4, 1 - 5e-4)) * 2 / 1000)
    say("   inst  uniform(new)  NS eps=0.3 (old)  Gaussian(new)  z(Gauss-unif)")
    for i, a1, a0, a2, s_ in zip(ex, Pu, old, Pg, se):
        say(f"   {i:4d}   {a1:.3f}         {a0:.3f}            {a2:.3f}          {(a2 - a1) / s_:+.1f}")
    say(f"   mean: uniform {Pu.mean():.4f}, old NS {old.mean():.4f}, Gaussian {Pg.mean():.4f};  "
        f"mean(Gauss - unif) {np.mean(Pg - Pu):+.4f}, share |z| < 2: {(np.abs((Pg - Pu) / se) < 2).mean():.2f}")
    with open(os.path.join(HERE, "report_exact_sigma.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if not a.skip_switch:
    run_switch()
if a.exact_sigma:
    run_exact()
print("written:", HERE)
