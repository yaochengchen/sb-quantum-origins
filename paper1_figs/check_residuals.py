"""MSBQA paper 1: what the failures outside the Landau-Zener estimate are (numbers quoted in Sec. IV C, Fig. 5(d)).
Reads existing results only (run_v4/parts: Wq, WqT200, WqT2000, cvqw_cut, Wg, G0q, GqT200, GqT2000, Gg) and, for a few
instances, diagonalizes H(p_f) in the even sector (seconds each) to read out the ground state at the end of the ramp.
 A. weyl, instances with P_LZ < 1e-6: failure against the ramp time and the truncation; ground state at p_f
 B. weyl, instances with P_LZ > 0.01: failure at T = 200, 700, 2000 against P_LZ; ground state at p_f
 C. goto: instances whose minimum gap is at p = 0 (start of the ramp), failure against the ramp time
Run from msbqa_phase1/:
    python paper1_figs/check_residuals.py
Writes paper1_figs/report_residuals.txt.
"""
import json
import os

import numpy as np

import p1common as pc
import kpo_common as kc

say = pc.Numbers("check_residuals")
PF, K, H, D_CUT = 7.0, 1.0, 1.0, 23
N = 300


def part(tag, i):
    f = pc.P("run_v4", "parts", f"{tag}_{i}.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def fail(tag, i):
    r = part(tag, i)
    return np.nan if r is None else 1.0 - r["P_succ"]


def ground_readout(i, family="weyl"):
    """1 - P_succ of the even-sector ground state of H(p_f), with the read-out of the propagation runs."""
    import scipy.sparse as sps
    import scipy.sparse.linalg as sla
    J, Delta, meta = kc.load_instance(pc.P("instances"), "G4", i)
    sec = kc.KPOSector(J, Delta, K, meta["xi0"], H, D_CUT, family)
    w, v = sla.eigsh((sps.diags(sec.diag) + sec.Hhop + PF * sec.Hp).tocsr(), k=2, which="SA", tol=1e-10, ncv=40)
    psi = sec.embed(v[:, int(np.argmin(w))].astype(np.complex128))
    model = kc.KPOQuantum(J, Delta, K, meta["xi0"], H, D_CUT, family)
    P = kc.pattern_probabilities(model, psi, kc.theta_matrix(D_CUT))
    return 1.0 - kc.summarize_patterns(P, kc.ising_energies(J))[0]


# ------------------------------------------------------------------ A. weyl residuals
xlz = np.array([part("Wg", i)["x_LZ"] for i in range(N)])
plz = np.exp(-xlz * 700.0)
F = {t: np.array([fail(t, i) for i in range(N)]) for t in ("WqT200", "Wq", "WqT2000")}
low = plz < 1e-6
say(f"A. weyl, h = 1: {int(low.sum())} instances with P_LZ(T=700) < 1e-6")
say(f"   mean failure  T=200 {F['WqT200'][low].mean():.2e}   T=700 {F['Wq'][low].mean():.2e}   T=2000 {F['WqT2000'][low].mean():.2e}")
rel = np.abs(F["Wq"][low] - F["WqT2000"][low]) / F["Wq"][low]
say(f"   T=700 against T=2000, per instance: |difference| / failure median {np.median(rel):.3f}, max {rel.max():.3f}; "
    f"within 20 % on {int((rel <= 0.2).sum())} of {int(low.sum())}")
cut = [(i, fail("Wq", i), fail("cvqw_cut", i)) for i in range(10) if low[i]]
say(f"   truncation 22 -> 26 on instances {[c[0] for c in cut]}: relative change of the failure max "
    f"{max(abs(b - a) / a for _, a, b in cut):.3f}")
say("   ground state of H(p_f) read out directly (failure): instance, ground state, propagation T=700, T=2000")
gr = []
for i in [c[0] for c in cut] + [18, 164]:
    g = ground_readout(i)
    gr.append(abs(g - F["WqT2000"][i]) / F["WqT2000"][i])
    say(f"      #{i:<4d} {g:.3e}   {F['Wq'][i]:.3e}   {F['WqT2000'][i]:.3e}")
say(f"   ground state against T=2000: relative difference max {max(gr):.3f}")

# ------------------------------------------------------------------ B. weyl, P_LZ > 0.01
say("")
say("B. weyl, instances with P_LZ(T=700) > 0.01: failure measured / P_LZ at T = 200, 700, 2000; ground state at p_f")
for i in np.flatnonzero(plz > 0.01):
    g = part("Wg", i)
    row = "   ".join(f"T={T:g}: {F[t][i]:.2e} / {np.exp(-xlz[i] * T):.2e}" for t, T in (("WqT200", 200), ("Wq", 700), ("WqT2000", 2000)))
    say(f"   #{i:<4d} p* {g['p_star']:.2f}  Delta_min {g['Delta_min']:.4f}   {row}   ground state {ground_readout(i):.2e}")

# ------------------------------------------------------------------ C. goto
say("")
gg = [part("Gg", i) for i in range(N)]
p0 = np.array([g["p_star"] == 0.0 for g in gg])
dmin = np.array([g["Delta_min"] for g in gg])
plzg = np.exp(-np.array([g["x_LZ"] for g in gg]) * 700.0)
G = {t: np.array([fail(t, i) for i in range(N)]) for t in ("GqT200", "G0q", "GqT2000")}
info = np.genfromtxt(pc.P("instances", "G4_info.csv"), delimiter=",", names=True)
igap = np.asarray(info["gap"][:N], float)                      # Ising gap E1 - E0
say(f"C. goto, h = 1: minimum gap at p = 0 on {int(p0.sum())} of {N} instances (Delta_min {dmin[p0].min():.3f}-{dmin[p0].max():.3f})")
hi = p0 & (G["G0q"] > 1e-3)
lo = p0 & ~hi
say(f"   failure at T=700 above 1e-3 on {int(hi.sum())}: {G['G0q'][hi].min():.3f}-{G['G0q'][hi].max():.3f}, Ising gap "
    f"{igap[hi].min():.2f}-{igap[hi].max():.2f};  below on {int(lo.sum())}: max {G['G0q'][lo].max():.1e}, Ising gap "
    f"{igap[lo].min():.2f}-{igap[lo].max():.2f}")
sel = hi & np.isfinite(G["GqT2000"])
say(f"   ramp-time scan on {int(sel.sum())} of them: failure(700)/failure(2000) median "
    f"{np.median(G['G0q'][sel] / G['GqT2000'][sel]):.2f} [(2000/700)^2 = {(2000 / 700) ** 2:.2f}], failure(200)/failure(700) median "
    f"{np.median(G['GqT200'][sel] / G['G0q'][sel]):.2f} [(700/200)^2 = {(700 / 200) ** 2:.2f}]")
i5 = int(np.flatnonzero(sel)[0])
say(f"   example #{i5}: T=200 {G['GqT200'][i5]:.4f}, T=700 {G['G0q'][i5]:.4f}, T=2000 {G['GqT2000'][i5]:.4f}")
inner = ~p0
say(f"   interior minimum ({int(inner.sum())}): P_LZ >= 1e-5 on {int((inner & (plzg >= 1e-5)).sum())}; with P_LZ > 0.01, measured / P_LZ: "
    + ", ".join(f"#{i} {G['G0q'][i]:.3f}/{plzg[i]:.3f}" for i in np.flatnonzero(inner & (plzg > 0.01))))
lowg = plzg < 1e-6
say(f"   P_LZ < 1e-6: {int(lowg.sum())} instances, mean failure {G['G0q'][lowg].mean():.1e}; interior minimum only "
    f"({int((lowg & inner).sum())}): {G['G0q'][lowg & inner].mean():.1e}")
s2 = lowg & inner & np.isfinite(G["GqT2000"])
say(f"   those with a ramp-time scan ({int(s2.sum())}): mean failure T=200 {G['GqT200'][s2].mean():.1e}, T=700 "
    f"{G['G0q'][s2].mean():.1e}, T=2000 {G['GqT2000'][s2].mean():.1e}; failure(700)/failure(2000) median "
    f"{np.median(G['G0q'][s2] / G['GqT2000'][s2]):.2f}")
with open(os.path.join(pc.HERE, "report_residuals.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(say.lines) + "\n")
