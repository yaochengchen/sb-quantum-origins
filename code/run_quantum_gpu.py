"""L0 quantum runs of the plan on one GPU (CuPy engine kpo_gpu), instances batched; same output as run_quantum.py.

usage (from msbqa_phase1/; gpu/run_gpu.sh starts one of these per GPU):
  CUDA_VISIBLE_DEVICES=3 python code/run_quantum_gpu.py --plan slurm/plan.txt --run run_main --inst-dir instances \
      --gpu 3 --ngpu 8 [--tags G0q,Wq] [--batch 64]
Takes every quantum line of the plan (or only --tags), keeps the instances i with i % ngpu == gpu whose part file
does not exist yet, and evolves the instances of one plan line together (at most --batch at a time).
Writes RUN/parts/<tag>_<inst>.json with the same fields as run_quantum.py (engine = "gpu"; wall_s is the wall
time of the whole batch, batch = its size) and appends one line per instance to RUN/parts/<tag>.log.
Restart-safe: finished parts are skipped. Progress (simulated time, steps, wall time) every 10% of T on stdout.
"""
import argparse
import os
import time

import cupy as cp
import numpy as np

import kpo_common as kc
import kpo_gpu as kg

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--ngpu", type=int, default=1)
ap.add_argument("--tags", default="")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--krylov", type=int, default=24)
a = ap.parse_args()
parts = os.path.join(a.run, "parts")
os.makedirs(parts, exist_ok=True)
want = set(a.tags.split(",")) if a.tags else None

lines = []
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if not f:
        continue
    if len(f) != 14:
        raise SystemExit(f"plan line needs 14 columns: {raw.strip()}")
    tag, kind, family, sset, lo, hi, T, pf, h, cutoff, R, tol, dtmax, dtcl = f
    if kind != "quantum" or (want and tag not in want):
        continue
    lines.append(dict(tag=tag, family=family, set=sset, lo=int(lo), hi=int(hi), T=float(T), pf=float(pf),
                      h=float(h), cutoff=int(cutoff), tol=float(tol), dtmax=float(dtmax)))

gpu_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name)
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} worker {a.gpu}/{a.ngpu} on {gpu_name}: "
      f"plan lines {[L['tag'] for L in lines]}", flush=True)


def out_path(tag, i):
    return os.path.join(parts, f"{tag}_{i}.json")


def run_batch(L, insts):
    t0 = time.time()
    tag = L["tag"]
    loaded = [kc.load_instance(a.inst_dir, L["set"], i) for i in insts]
    d = L["cutoff"] + 1
    eng = kg.SectorBatchGPU(np.stack([x[0] for x in loaded]), np.stack([x[1] for x in loaded]), a.K,
                            np.array([x[2]["xi0"] for x in loaded]), L["h"], d, L["family"], m=a.krylov)
    t_build = time.time() - t0
    head = f"[{tag} inst {insts[0]}..{insts[-1]} B={len(insts)}]"

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {head} {msg}", flush=True)

    log(f"start: sector dim {eng.N}, build {t_build:.1f} s")
    psi_b, st = eng.propagate(eng.vacuum(), L["T"], kc.linear_schedule(L["pf"], L["T"]), tol=L["tol"],
                              dt_max=L["dtmax"], log=log)
    psi_b = cp.asnumpy(psi_b)
    wall = time.time() - t0
    Theta = kc.theta_matrix(d)
    for b, i in enumerate(insts):
        J, Delta, meta = loaded[b]
        model = kc.KPOQuantum(J, Delta, a.K, meta["xi0"], L["h"], d, L["family"])
        psi = eng.embed(psi_b[b])
        P = kc.pattern_probabilities(model, psi, Theta)
        psucc, eres = kc.summarize_patterns(P, kc.ising_energies(J))
        pops = model.mode_populations(psi)
        nbar = (pops * np.arange(d)[None, :]).sum(1)
        res = dict(tag=tag, kind="quantum", layer="L0", family=L["family"], engine="gpu", set=L["set"], inst=i,
                   T=L["T"], pf=L["pf"], K=a.K, h=L["h"], cutoff=L["cutoff"], tol=L["tol"], dtmax=L["dtmax"],
                   krylov=a.krylov, P=P.tolist(), P_succ=psucc, E_res=eres, P_sum=float(P.sum()),
                   norm=float(np.linalg.norm(psi)), parity_leak=model.parity_leak(psi),
                   top2_pop_max=float(pops[:, -2:].sum(1).max()), nbar=nbar.tolist(),
                   steps=st["steps"], rejects=st["rejects"], matvecs=st["matvecs"], err_sum=float(st["err_sum"][b]),
                   build_s=t_build, wall_s=wall, batch=len(insts), gpu=gpu_name)
        kc.write_json_atomic(out_path(tag, i), res)
        line = (f"{L['set']}#{i} {L['family']}/gpu T={L['T']:g} h={L['h']:g} cut={L['cutoff']}: P_succ={psucc:.4f} "
                f"E_res={eres:.4f} top2={res['top2_pop_max']:.1e} steps={st['steps']} mv={st['matvecs']} "
                f"batch={len(insts)} {wall:.0f}s")
        with open(os.path.join(parts, f"{tag}.log"), "a") as fh:
            fh.write(line + "\n")
    log(f"done: {st['steps']} steps, {st['rejects']} rejects, {wall:.0f} s")


for L in lines:
    todo = [i for i in range(L["lo"], L["hi"] + 1) if i % a.ngpu == a.gpu and not os.path.exists(out_path(L["tag"], i))]
    for c0 in range(0, len(todo), a.batch):
        run_batch(L, todo[c0:c0 + a.batch])
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} worker {a.gpu}/{a.ngpu} done", flush=True)
