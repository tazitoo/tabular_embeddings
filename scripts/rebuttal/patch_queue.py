#!/usr/bin/env python3
"""Concept-level work queue for patch sweeps: keep every GPU full.

Static arm-per-host allocation left a 26h-to-5-day completion spread (user,
2026-08-21): concept costs vary by orders of magnitude, so whole arms pinned
to single GPUs guarantee idle tails. Concepts are independent -- roots are
embarrassingly parallel and a concept is the natural work unit -- so this
dispatcher runs from the Mac, polls the fleet, and launches ONE CONCEPT per
free GPU slot until the pool drains. Per-concept outputs land in
output/rebuttal/<run>/ (local pulls on completion are the source of truth);
analysis loaders already merge and dedup by (donor, feat).

Launch mechanics mirror orch.sh exactly: ssh -f + setsid nohup + env with
CUDA pinning, expandable segments, and the thread caps. tabicl_v2 concepts
are constrained to tfm2-capable hosts. A slot whose process died without
writing its output re-queues the concept once, then marks it failed loudly.

Usage:
    python -m scripts.rebuttal.patch_queue \
        --run v30q --pool-from "output/rebuttal/patchv28clf_*.json" \
        --done-from "output/rebuttal/patchv30clf_*.json" \
        --flags "--rank-by effectiveness_raw --exponents 1,0,1,0 \
                 --blast-form delta --beam all --window 3 --patience 3 \
                 --n-datasets 3 --n-rows 10 --device cuda"
"""
import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

REPO = "/home/brian/src/tabular_embeddings"
PY_TFM = "/home/brian/anaconda3/envs/tfm/bin/python"
PY_TFM2 = "/home/brian/anaconda3/envs/tfm2/bin/python"
SLOTS = [("morg.local", 0), ("morg.local", 2), ("morg.local", 3), ("morg.local", 4),
         ("surfer4", 0), ("octo4", 0), ("terrax4", 0), ("firelord4", 0)]
TFM2_HOSTS = {"morg.local"}          # tabicl_v2 runs only where tfm2 exists
ENV = ("CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
       "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
       "NUMEXPR_NUM_THREADS=8")


def sh(cmd, timeout=45):
    """None on ANY transport failure -- a slow poll must cost one cycle, not the
    queue (the first run died overnight on a single 30s morg timeout after 50
    clean completions)."""
    try:
        return subprocess.run(["ssh", "-o", "ConnectTimeout=10", *cmd],
                              capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return None


def busy_gpus(host):
    """CUDA indices with a live patch_search on `host` (parsed from the env
    args that our own launch convention puts on the command line)."""
    r = sh([host, "pgrep -af '[p]atch_search' 2>/dev/null"])
    if r is None:
        return None          # unknown -- caller must treat as fully busy
    busy = set()
    for line in r.stdout.splitlines():
        for tok in line.split():
            if tok.startswith("CUDA_VISIBLE_DEVICES="):
                busy.add(int(tok.split("=")[1]))
    # a python proc launched without the env prefix on its own line (the child)
    # still shows the parent env line; unparseable lines mean SOMETHING runs --
    # treat gpu0 as busy for single-gpu hosts to stay safe
    if r.stdout.strip() and not busy:
        busy.add(0)
    return busy


def concept_id(donor, feat):
    return f"{donor}_f{feat}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="subdir under output/rebuttal/")
    ap.add_argument("--pool-from", nargs="+", required=True,
                    help="sweep files/globs whose concept lists define the pool")
    ap.add_argument("--done-from", nargs="*", default=[],
                    help="additional files whose concepts count as already done")
    ap.add_argument("--flags", required=True, help="patch_search flags (verbatim)")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    outdir = PROJECT_ROOT / "output" / "rebuttal" / args.run
    outdir.mkdir(parents=True, exist_ok=True)

    pool = []
    seen = set()
    for pat in args.pool_from:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                k = (c["donor"], int(c["feat"]))
                if k not in seen:
                    seen.add(k)
                    pool.append(k)
    # STARTUP SYNC: pull any per-concept outputs still sitting on hosts (a prior
    # dispatcher may have died between a job finishing and its pull) so finished
    # work is never relaunched
    for host in {h for h, _ in SLOTS}:
        subprocess.run(["rsync", "-a",
                        f"{host}:{REPO}/output/rebuttal/{args.run}/",
                        str(outdir) + "/"], capture_output=True, timeout=300)

    done = set()
    for pat in args.done_from:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                done.add((c["donor"], int(c["feat"])))
    done |= {tuple(f.stem.rsplit("_f", 1)) and (f.stem.rsplit("_f", 1)[0],
             int(f.stem.rsplit("_f", 1)[1])) for f in outdir.glob("*_f*.json")}
    todo = [k for k in pool if k not in done]
    print(f"pool {len(pool)}, already done {len(pool) - len(todo)}, to run {len(todo)}",
          flush=True)

    inflight = {}          # (host, gpu) -> (donor, feat, remote_out, log)
    attempts = {}
    failed = []
    while todo or inflight:
        # ---- reap ------------------------------------------------------------
        for slot, (donor, feat, rout, log) in list(inflight.items()):
            host, gpu = slot
            bg = busy_gpus(host)
            if bg is None:
                continue      # host unreachable this cycle; check again next one
            alive = gpu in bg
            hr = sh([host, f"test -s {rout} && echo yes"])
            if hr is None:
                continue
            have = hr.stdout.strip() == "yes"
            if have:
                local = outdir / f"{concept_id(donor, feat)}.json"
                try:
                    subprocess.run(["rsync", "-a", f"{host}:{rout}", str(local)],
                                   check=True, timeout=120)
                except (subprocess.SubprocessError, OSError):
                    continue  # pull failed; retry next cycle, job stays accounted
                print(f"DONE {concept_id(donor, feat)} on {host}:{gpu} "
                      f"({len(todo)} queued)", flush=True)
                del inflight[slot]
            elif not alive:
                a = attempts.get((donor, feat), 1)
                if a >= 2:
                    print(f"FAILED twice: {concept_id(donor, feat)} (last {host}:{gpu}, "
                          f"log {log}) -- marked failed, NOT retried", flush=True)
                    failed.append((donor, feat))
                else:
                    attempts[(donor, feat)] = a + 1
                    todo.insert(0, (donor, feat))
                    print(f"CRASHED {concept_id(donor, feat)} on {host}:{gpu} -- "
                          f"requeued (attempt {a + 1})", flush=True)
                del inflight[slot]
        # ---- dispatch --------------------------------------------------------
        for slot in SLOTS:
            if not todo:
                break
            if slot in inflight:
                continue
            host, gpu = slot
            bg = busy_gpus(host)
            if bg is None or gpu in bg:
                continue          # unreachable or occupied (e.g. the control arm)
            idx = next((i for i, (d, _) in enumerate(todo)
                        if d != "tabicl_v2" or host in TFM2_HOSTS), None)
            if idx is None:
                continue
            donor, feat = todo.pop(idx)
            py = PY_TFM2 if donor == "tabicl_v2" else PY_TFM
            cid = concept_id(donor, feat)
            rout = f"{REPO}/output/rebuttal/{args.run}/{cid}.json"
            log = f"/tmp/{args.run}_{cid}.log"
            sh([host, f"mkdir -p {REPO}/output/rebuttal/{args.run}"])
            cmd = (f"cd {REPO} && setsid nohup env CUDA_VISIBLE_DEVICES={gpu} {ENV} "
                   f"{py} -m scripts.rebuttal.patch_search --concepts {donor}:{feat} "
                   f"{args.flags} --out {rout} > {log} 2>&1 < /dev/null &")
            try:
                subprocess.run(["ssh", "-f", "-n", "-o", "ConnectTimeout=10", host, cmd],
                               check=True, timeout=45)
            except (subprocess.SubprocessError, OSError):
                todo.insert(0, (donor, feat))
                continue      # launch failed; concept back on the queue
            attempts.setdefault((donor, feat), 1)
            inflight[slot] = (donor, feat, rout, log)
            print(f"LAUNCH {cid} -> {host}:{gpu} ({len(todo)} queued, "
                  f"{len(inflight)} running)", flush=True)
        time.sleep(args.poll)

    print(f"queue drained: {len(pool) - len(failed)} done, {len(failed)} failed "
          f"{[concept_id(*k) for k in failed]}", flush=True)


if __name__ == "__main__":
    main()
