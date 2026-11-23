"""MSBQA paper 1 -- per-instance check that TWA acts as classical noise of amplitude sigma = sqrt(h)/2.
Reads existing results only.  Run from msbqa_phase1/:
    python paper1_figs/check_twa_vs_noise.py
Writes paper1_figs/report_twa_vs_noise.txt and fig_twa_vs_noise.png.

Same flow (Goto App. E classical model = weyl-family Weyl flow, T = 700, pf = 7, dt = 0.0025, R = 1000 each),
different initial distributions:
  classical L2g : z_i = uniform(+-eps) + i uniform(+-eps)   -> sigma = eps / sqrt(3) per quadrature  (run_v4/parts/NS_i)
  TWA (L1)      : z_i = sqrt(h)/2 (N(0,1) + i N(0,1))         -> sigma = sqrt(h) / 2                 (run_v4/parts/Wc*_i)
Matched pairs (closest sigma available): eps 0.03 <-> h 0.001 (sigma 0.0173 / 0.0158) and eps 0.3 <-> h 0.1 (0.173 / 0.158).
Per instance: d = P_TWA - P_L2g, z = d / sqrt(P(1-P)(1/R + 1/R)) with P the pooled value (floor 1/(2R)).
If TWA = classical noise of the same amplitude, |z| should look like |N(0,1)| (about 95 % below 2), apart from the
small sigma mismatch.  Contrasts: an unmatched pair (eps 1e-6 vs h 0.001) and quantum vs TWA at h = 1.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = lambda *a: os.path.join(ROOT, *a)
NINST = 100
L = []


def say(s=""):
    print(s)
    L.append(s)


def load_ns(i):
    r = json.load(open(P("run_v4", "parts", f"NS_{i}.json")))
    return {e["eps"]: e["P_succ"] for e in r["layers"]["L2g"]}, r["R"]


def load_twa(tag, i):
    r = json.load(open(P("run_v4", "parts", f"{tag}_{i}.json")))
    return r["layers"]["L1"]["P_succ"], r["R"], r["h"]


br = pd.read_csv(P("run_v4", "branches", "branches_weyl.csv")).set_index("inst")
aw = pd.read_csv(P("run_v4", "analysis_weyl", "per_instance.csv")).set_index("inst")
pairs = [("matched", 0.03, "Wc0001"), ("matched", 0.3, "Wc010"), ("unmatched", 1e-6, "Wc0001")]
rows = []
for i in range(NINST):
    ns, R1 = load_ns(i)
    for kind, eps, tag in pairs:
        pt, R2, h = load_twa(tag, i)
        pc = ns[eps]
        pp = min(max((pc * R1 + pt * R2) / (R1 + R2), 0.5 / R1), 1 - 0.5 / R1)
        se = np.sqrt(pp * (1 - pp) * (1 / R1 + 1 / R2))
        rows.append(dict(inst=i, kind=kind, eps=eps, h=h, sig_c=eps / np.sqrt(3), sig_t=np.sqrt(h) / 2,
                         P_cl=pc, P_twa=pt, d=pt - pc, z=(pt - pc) / se, cross=int(max(br.cross[i], 0))))
d = pd.DataFrame(rows)

say("== TWA vs classical noise of matched amplitude, per instance (G4 0-99, weyl flow, T = 700, R = 1000 each)")
say("   pair                         sigma_cl  sigma_TWA |  mean d   mean|d|  max|d|  share |z|<2  share |z|<3  "
    "Spearman(P_cl,P_TWA)")
for kind, eps, tag in pairs:
    g = d[(d.kind == kind) & (d.eps == eps)]
    rho = spearmanr(g.P_cl, g.P_twa).correlation
    say(f"   {kind:9s} eps={eps:<6g} h={g.h.iloc[0]:<6g}  {g.sig_c.iloc[0]:.4f}    {g.sig_t.iloc[0]:.4f}   | "
        f"{g.d.mean():+.4f}  {g.d.abs().mean():.4f}  {g.d.abs().max():.3f}    {(g.z.abs() < 2).mean():.2f}"
        f"         {(g.z.abs() < 3).mean():.2f}        {rho:+.3f}")
    for cc, nm in ((1, "crossing"), (0, "no crossing")):
        s = g[g.cross == cc]
        if len(s):
            say(f"      {nm:12s} N {len(s):3d}: mean|d| {s.d.abs().mean():.4f}, share |z|<2 {(s.z.abs() < 2).mean():.2f}")
say()
say("   instances with |z| >= 3 in the matched pairs:")
m = d[(d.kind == "matched") & (d.z.abs() >= 3)]
for r in m.itertuples():
    say(f"     inst {r.inst:3d}  eps {r.eps:g} / h {r.h:g}:  P_cl {r.P_cl:.3f}  P_TWA {r.P_twa:.3f}  z {r.z:+.1f}  cross {r.cross}")
if m.empty:
    say("     none")
say()
qa = aw.reindex(range(NINST))
dq = (qa.P_L0 - qa.P_L1)
say("-- contrast: quantum (h = 1) vs TWA (h = 1) on the same instances")
say(f"   mean |P_L0 - P_L1| {dq.abs().mean():.3f}; on crossing instances "
    f"{dq[[i for i in range(NINST) if br.cross[i] == 1]].abs().mean():.3f}")

with open(os.path.join(HERE, "report_twa_vs_noise.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
for a_, (kind, eps, tag) in zip(ax, pairs):
    g = d[(d.kind == kind) & (d.eps == eps)]
    for cc, col, lab in ((0, "#2B6CB0", "no crossing"), (1, "#C0392B", "crossing")):
        s = g[g.cross == cc]
        a_.scatter(s.P_cl, s.P_twa, s=10, c=col, lw=0, label=lab)
    a_.plot([0, 1], [0, 1], "k--", lw=0.7)
    a_.set_xlabel(rf"classical, $\varepsilon$={eps:g} ($\sigma$={eps / np.sqrt(3):.3g})")
    a_.set_ylabel(rf"TWA, h={g.h.iloc[0]:g} ($\sigma$={np.sqrt(g.h.iloc[0]) / 2:.3g})")
    a_.set_title(kind)
    a_.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_twa_vs_noise.png"), dpi=150)
print("written:", HERE)
