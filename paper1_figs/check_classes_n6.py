"""MSBQA paper 1: numbers quoted in the text that are not shown in a figure (reads phase-1 results only).
 (1) Sec. IV.A, trap instances WITHOUT a crossing (G4 test set 0-299, weyl): is the first-born branch the ground
     configuration, and does noise repair the classical KPO (L2 at eps = 1e-6 vs eps*, L3 at eps*)?
 (2) Sec. VI.B, parameters not normalized with n: mean detuning, p_c - Dbar on crossing instances and the Ising gap,
     n = 4 (G4) vs n = 6 (G6), test sets 0-299.
Inputs: run_v4/branches/branches_weyl.csv, run_v4/branches_G6/branches_weyl.csv, run_v4/phase1b/per_instance_1b.csv,
instances/G4.npz, G4_info.csv, G6.npz, G6_info.csv (G6: code/gen_instances.py --set G6 --n 6 --ninst 1000 --seed 2016
--xi0 0.5, as in run_gpu_1c.sh). Run from msbqa_phase1/:
    python paper1_figs/check_classes_n6.py
"""
import numpy as np
import pandas as pd

import p1common as pc

say = pc.Numbers("check_classes_n6")

# (1) trap without crossing, n = 4 test set
b = pc.load_1b()
br = pc.load_branches().loc[b.index]
m = (b.trap == 1) & (b.cross == 0)
say(f"(1) G4 test set: trap without crossing N {int(m.sum())}; first-born branch = ground configuration in "
    f"{int(br.loc[m, 'first_is_ground'].sum())} of {int(m.sum())}")
say(f"    L2 eps=1e-6 {b.loc[m, pc.C_L2G_GOTO].mean():.3f} -> L2 eps* {b.loc[m, pc.C_L2G].mean():.3f};  "
    f"L3 eps* {b.loc[m, pc.C_L3].mean():.3f};  L4 eps* {b.loc[m, pc.C_L4].mean():.3f};  quantum {b.loc[m, 'Q_weyl'].mean():.3f}")
mc = b.cross == 1
say(f"    for comparison, crossing instances (N {int(mc.sum())}): first-born = ground in "
    f"{int(br.loc[mc, 'first_is_ground'].sum())}; L2 eps* {b.loc[mc, pc.C_L2G].mean():.3f}")

# (2) n = 4 vs n = 6
rows = []
for set_name, bfile, n in (("G4", ("branches", "branches_weyl.csv"), 4), ("G6", ("branches_G6", "branches_weyl.csv"), 6)):
    d = np.load(pc.P("instances", f"{set_name}.npz"))
    info = pd.read_csv(pc.P("instances", f"{set_name}_info.csv"))
    bw = pd.read_csv(pc.P("run_v4", *bfile)).set_index("inst")
    idx = bw.index[bw.index < 300]
    Dbar = d["Delta"][idx].mean(axis=1)
    cr = (bw.loc[idx, "cross"].clip(lower=0) == 1).values
    gap = info.set_index("inst").loc[idx, "gap"].values if "inst" in info.columns else info["gap"].values[idx]
    pcd = bw.loc[idx, "p_c"].values - Dbar
    rows.append(dict(n=n, N=len(idx), Dbar=Dbar.mean(), ncross=int(cr.sum()), pc_minus_Dbar=np.median(pcd[cr]),
                     gap_all=np.median(gap), gap_cross=np.median(gap[cr]),
                     not_lowest_at_pf=float((bw.loc[idx, "gs_lowest_at_pf"] != 1).mean())))
for r in rows:
    say(f"(2) n = {r['n']}: instances {r['N']}, mean detuning {r['Dbar']:.3f}, crossing {r['ncross']}, "
        f"median p_c - Dbar on crossing {r['pc_minus_Dbar']:.3f}, median Ising gap all {r['gap_all']:.3f} / crossing "
        f"{r['gap_cross']:.3f}, ground not the lowest branch at pf {r['not_lowest_at_pf']:.3f}")
say(f"    Ising gap ratio n=6/n=4: all {rows[1]['gap_all'] / rows[0]['gap_all']:.2f}, crossing "
    f"{rows[1]['gap_cross'] / rows[0]['gap_cross']:.2f}")
say.write()
