"""MSBQA paper 1 -- figure "classical failure modes" (publication version of geo_decisive/fig_geo_decisive.png).
Reads existing results only (no simulation).  Run from msbqa_phase1/:
    python paper1_figs/fig_classical_failure.py
Writes paper1_figs/fig_classical_failure.pdf / .png and paper1_figs/fig_classical_failure_numbers.txt.

Panels
 (a) departure pump p_dep (median over trajectories, measured: first max_i |z_i| > ell = 0.1, noise_scan.py) vs the
     linear prediction mu_dec(sigma): Lambda_1(mu_dec) = ln(ell / sigma), all NS instances, eps <= 1e-3.
     Lambda_k(p) = (1/r) int_{lam_k}^{p} sqrt(q^2 - lam_k^2) dq: amplification of linear mode k of
     M = diag(Delta) - xi0 J around the unstable origin; r = pf / T.
 (b) success of the classical KPO (L2g) vs x = ln(sigma / eps*), eps* = ell exp(-Lambda_1(lam_2)) (decision exactly
     when mode 2 turns unstable); all NS instances, eps <= 1e-2; colour = configuration crossing (branches_weyl.csv).
 (c) classical L2g (noise uniform +-eps, sigma = eps/sqrt 3) and TWA (L1, Gaussian vacuum, sigma = sqrt(h)/2) on one
     noise axis: mean over the SAME instances (those with NS and all TWA tags; G4 0-99), error bar = s.e.m.
     The quantum L0 at h = 1 is shown as a reference line.
Noise per quadrature: sigma.  Flow: Goto App. E classical model (= weyl-family Weyl flow), T = 700, pf = 7.
Wording for the paper: "linear amplification around the unstable origin" (slow passage through a bifurcation);
no geodesic / Jacobi language here -- that interpretation belongs to the geodesic paper and is only cited.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = lambda *a: os.path.join(ROOT, *a)
ELL = 0.1
TWA_TAGS = {"Wc": 1.0, "Wc07": 0.7, "Wc05": 0.5, "Wc035": 0.35, "Wc025": 0.25, "Wc010": 0.1, "Wc005": 0.05,
            "Wc002": 0.02, "Wc0005": 0.005, "Wc0001": 0.001}          # plan.txt: weyl classical lines, T = 700
NUM = []


def say(s=""):
    print(s)
    NUM.append(s)


def Lam(p, lam, r):
    p = np.atleast_1d(np.asarray(p, float))
    lam = max(abs(float(lam)), 1e-12)          # lam_1 can be ~ -1e-16 (M is PSD by construction)
    out = np.zeros_like(p)
    m = p > lam
    s = np.sqrt(p[m] ** 2 - lam ** 2)
    out[m] = 0.5 * (p[m] * s - lam ** 2 * np.log((p[m] + s) / lam)) / r
    return out


def mu_dec(sigma, lam1, r):
    target = np.log(ELL / sigma)
    if target <= 0:
        return max(abs(lam1), 0.0)
    lo, hi = max(abs(lam1), 1e-12), 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if Lam(mid, lam1, r)[0] < target else (lo, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ noise scan (all instances found)
rows = []
for f in sorted(glob.glob(P("run_v4", "parts", "NS_*.json"))):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if r.get("tag") != "NS":
        continue
    rr = r["pf"] / r["T"]
    for e in r["layers"]["L2g"]:
        rows.append(dict(inst=r["inst"], lam1=r["lam1"], lam2=r["lam2"], r=rr, eps=e["eps"], P=e["P_succ"],
                         pdep=e["dep"]["p_dep_med"], fdep=e["dep"]["frac_departed"]))
ns = pd.DataFrame(rows)
if ns.empty:
    raise SystemExit("no run_v4/parts/NS_*.json found (run from msbqa_phase1/)")
br = pd.read_csv(P("run_v4", "branches", "branches_weyl.csv")).set_index("inst")
inst = ns.groupby("inst")[["lam1", "lam2", "r"]].first()
inst["ln_eps_star"] = [np.log(ELL) - Lam(l2, l1, r)[0] for l1, l2, r in zip(inst.lam1, inst.lam2, inst.r)]
inst["cross"] = br.cross.reindex(inst.index).clip(lower=0).fillna(0).astype(int)
ns = ns.merge(inst[["ln_eps_star", "cross"]], left_on="inst", right_index=True)
ns["sigma"] = ns.eps / np.sqrt(3.0)
ns["x"] = np.log(ns.sigma) - ns.ln_eps_star
ns["mu_pred"] = [mu_dec(s, l1, r) for s, l1, r in zip(ns.sigma, ns.lam1, ns.r)]

A = ns[(ns.eps <= 1e-3) & (ns.fdep > 0.99)]
B = ns[ns.eps <= 1e-2]
say(f"instances in the noise scan: {ns.inst.nunique()}  (crossing: {int(inst.cross.sum())})")
dv = A.pdep - A.mu_pred
say(f"(a) p_dep - mu_dec: median {dv.median():+.3f}, 90% |.| {dv.abs().quantile(0.9):.3f}, "
    f"Spearman {spearmanr(A.pdep, A.mu_pred).correlation:+.3f}, n = {len(A)}")
for cc, nm in ((0, "no crossing"), (1, "crossing")):
    s = B[B.cross == cc]
    say(f"(b) {nm:12s}: P mean at x > 0 {s.P[s.x > 0].mean():.3f} (n {int((s.x > 0).sum())}), "
        f"at x < 0 {s.P[s.x < 0].mean():.3f} (n {int((s.x < 0).sum())})")

# ------------------------------------------------------------------ TWA per instance, same instances
twa = {}
for tag, h in TWA_TAGS.items():
    d = {}
    for f in glob.glob(P("run_v4", "parts", f"{tag}_*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("tag") == tag:
            d[r["inst"]] = r["layers"]["L1"]["P_succ"]
    twa[h] = pd.Series(d, dtype=float)
qref = {}
for f in glob.glob(P("run_v4", "parts", "Wq_*.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if r.get("tag") == "Wq":
        qref[r["inst"]] = r["P_succ"]
common = set(ns.inst.unique())
for h, s in twa.items():
    if len(s):
        common &= set(s.index)
common = sorted(common)
say(f"(c) instances with NS and all available TWA tags: {len(common)}; TWA tags found: "
    + ", ".join(f"h={h:g}:{len(s)}" for h, s in twa.items()))
cl = ns[ns.inst.isin(common)].groupby("eps").P.agg(["mean", "sem"])
cl.index = cl.index / np.sqrt(3.0)
tw = pd.DataFrame({np.sqrt(h) / 2: [s.reindex(common).mean(), s.reindex(common).sem()]
                   for h, s in twa.items() if len(s)}, index=["mean", "sem"]).T.sort_index()
q1 = np.mean([qref[i] for i in common if i in qref]) if qref else np.nan
say("    sigma      classical L2g          TWA")
for sgm, v in sorted(list(cl["mean"].items()) + [(k, None) for k in tw.index], key=lambda t: t[0]):
    if v is None:
        say(f"    {sgm:9.2e}                         {tw.loc[sgm, 'mean']:.3f} +- {tw.loc[sgm, 'sem']:.3f}")
    else:
        say(f"    {sgm:9.2e}   {v:.3f} +- {cl.loc[sgm, 'sem']:.3f}")
say(f"    quantum L0 (h = 1) on the same instances: {q1:.4f}")

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42})
BLUE, RED, GREY = "#2B6CB0", "#C0392B", "#555555"
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))

norm = Normalize(np.log10(A.eps.min()), np.log10(A.eps.max()))
sc = ax[0].scatter(A.mu_pred, A.pdep, s=4, c=np.log10(A.eps), cmap="viridis", norm=norm, lw=0)
lim = [min(A.mu_pred.min(), A.pdep.min()) * 0.95, max(A.mu_pred.max(), A.pdep.max()) * 1.03]
ax[0].plot(lim, lim, "k--", lw=0.7)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel(r"predicted $\mu_{\rm dec}$"); ax[0].set_ylabel(r"measured $p_{\rm dep}$")
cb = fig.colorbar(sc, ax=ax[0], pad=0.02, fraction=0.06)
cb.set_label(r"$\log_{10}\varepsilon$", fontsize=8); cb.ax.tick_params(labelsize=7)

for cc, col, lab, z in ((0, BLUE, "no crossing", 1), (1, RED, "crossing", 2)):
    s = B[B.cross == cc]
    ax[1].scatter(s.x, s.P, s=4 if cc == 0 else 7, c=col, alpha=0.45 if cc == 0 else 0.8, lw=0, label=lab, zorder=z)
ax[1].axvline(0, c=GREY, lw=0.6, ls=":")
ax[1].set_xlabel(r"$x=\ln(\sigma/\varepsilon^*)$"); ax[1].set_ylabel(r"$P_{\rm succ}$ (classical KPO)")
ax[1].set_ylim(-0.02, 1.04); ax[1].legend(loc="lower right", frameon=False, markerscale=2)

ax[2].errorbar(cl.index, cl["mean"], yerr=cl["sem"], fmt="o-", ms=3, lw=1, c=BLUE, capsize=1.5,
               label=r"classical, $\sigma=\varepsilon/\sqrt{3}$")
ax[2].errorbar(tw.index, tw["mean"], yerr=tw["sem"], fmt="s-", ms=3, lw=1, c=RED, capsize=1.5,
               label=r"TWA, $\sigma=\sqrt{h}/2$")
if np.isfinite(q1):
    ax[2].axhline(q1, c="k", lw=0.8, ls="--", label=r"quantum, $h=1$")
ax[2].set_xscale("log")
ax[2].set_xlabel(r"noise $\sigma$ per quadrature"); ax[2].set_ylabel(r"mean $P_{\rm succ}$")
ax[2].legend(loc="lower left", frameon=False)
for a_, t in zip(ax, "abc"):
    a_.text(-0.02, 1.02, f"({t})", transform=a_.transAxes, va="bottom", ha="left", fontweight="bold")
fig.tight_layout(w_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_classical_failure.{ext}"), dpi=300)
with open(os.path.join(HERE, "fig_classical_failure_numbers.txt"), "w") as fh:
    fh.write("\n".join(NUM) + "\n")
print("written:", HERE)
