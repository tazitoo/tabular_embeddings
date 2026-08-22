#!/usr/bin/env python3
"""Concept-level work queue for patch sweeps: keep every GPU full.

Static arm-per-host allocation left a 26h-to-5-day completion spread (user,
2026-08-21): concept costs vary by orders of magnitude, so whole arms pinned
to single GPUs guarantee idle tails. Concepts are independent -- roots are
embarrassingly parallel and a concept is the natural work unit -- so this
dispatcher runs from the Mac and fills one concept per free GPU slot until
the pool drains.

STATELESS BY DESIGN (v2): the first version kept an in-memory in-flight
table; after a crash+restart it could neither see leftover jobs' completions
nor know their concepts were running, so it relaunched work into duplicates
(observed: tabdpt:22 on two hosts, tabdpt:8 twice on one). Every cycle now
derives the ENTIRE state from observable truth: running concepts parsed from
the fleet's process lists, done concepts from the synced per-concept output
files, todo = pool - done - running. A restart at any moment reconstructs
exactly; a transport hiccup costs one cycle. Crashed concepts (not running,
no output) fall back into todo automatically, capped at --max-attempts then
reported failed, loudly.

Launch mechanics mirror orch.sh (ssh -f, setsid, CUDA pinning, thread caps);
tabicl_v2 concepts are constrained to tfm2-capable hosts.

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
import re
import subprocess
import time

from scripts._project_root import PROJECT_ROOT

REPO = "/home/brian/src/tabular_embeddings"
PY_TFM = "/home/brian/anaconda3/envs/tfm/bin/python"
PY_TFM2 = "/home/brian/anaconda3/envs/tfm2/bin/python"
SLOTS = [("morg.local", 0), ("morg.local", 2), ("morg.local", 3), ("morg.local", 4),
         ("surfer4", 0), ("octo4", 0), ("terrax4", 0), ("firelord4", 0)]
TFM2_HOSTS = {"morg.local"}
ENV = ("CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
       "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
       "NUMEXPR_NUM_THREADS=8")
CONCEPT_RE = re.compile(r"--concepts (\w+):(\d+)")
CUDA_RE = re.compile(r"CUDA_VISIBLE_DEVICES=(\d+)")


def sh(host, remote, timeout=45):
    try:
        return subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, remote],
                              capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        return None


def observe(host):
    """(busy_gpus, running_concepts) from the host's process list; None if
    unreachable (callers treat unreachable as fully busy / unknown)."""
    r = sh(host, "pgrep -af '[p]atch_search' 2>/dev/null")
    if r is None:
        return None
    busy, running = set(), set()
    for line in r.stdout.splitlines():
        m = CONCEPT_RE.search(line)
        g = CUDA_RE.search(line)
        if g:
            busy.add(int(g.group(1)))
        if m:
            running.add((m.group(1), int(m.group(2))))
    if r.stdout.strip() and not busy:
        busy.add(0)          # something unparseable runs; assume gpu0 on workers
    return busy, running


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool-from", nargs="+", required=True)
    ap.add_argument("--done-from", nargs="*", default=[])
    ap.add_argument("--flags", required=True)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    outdir = PROJECT_ROOT / "output" / "rebuttal" / args.run
    outdir.mkdir(parents=True, exist_ok=True)

    pool, seen = [], set()
    for pat in args.pool_from:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                k = (c["donor"], int(c["feat"]))
                if k not in seen:
                    seen.add(k)
                    pool.append(k)
    static_done = set()
    for pat in args.done_from:
        for p in sorted(glob.glob(pat)):
            for c in json.load(open(p)):
                static_done.add((c["donor"], int(c["feat"])))

    attempts = {}
    reported_done = set()
    print(f"pool {len(pool)}, static done {len(static_done & set(pool))}", flush=True)

    while True:
        # ---- observe everything -----------------------------------------------
        for host in {h for h, _ in SLOTS}:
            subprocess.run(["rsync", "-a", f"{host}:{REPO}/output/rebuttal/{args.run}/",
                            str(outdir) + "/"], capture_output=True, timeout=300)
        file_done = set()
        for f in outdir.glob("*_f*.json"):
            donor, feat = f.stem.rsplit("_f", 1)
            file_done.add((donor, int(feat)))
        done = static_done | file_done
        for k in sorted(file_done - reported_done):
            print(f"DONE {k[0]}_f{k[1]}", flush=True)
            reported_done.add(k)

        state = {}
        running = set()
        for host in {h for h, _ in SLOTS}:
            state[host] = observe(host)
            if state[host] is not None:
                running |= state[host][1]

        exhausted = {k for k, a in attempts.items()
                     if a >= args.max_attempts and k not in done}
        todo = [k for k in pool if k not in done and k not in running
                and k not in exhausted]

        if not todo and not (running & set(pool)):
            failed = sorted(exhausted)
            print(f"queue drained: {len(done & set(pool))} done, "
                  f"{len(failed)} failed {[f'{d}_f{f}' for d, f in failed]}", flush=True)
            break

        # ---- dispatch ----------------------------------------------------------
        for host, gpu in SLOTS:
            if not todo:
                break
            if state.get(host) is None or gpu in state[host][0]:
                continue
            idx = next((i for i, (d, _) in enumerate(todo)
                        if d != "tabicl_v2" or host in TFM2_HOSTS), None)
            if idx is None:
                continue
            donor, feat = todo.pop(idx)
            py = PY_TFM2 if donor == "tabicl_v2" else PY_TFM
            cid = f"{donor}_f{feat}"
            rout = f"{REPO}/output/rebuttal/{args.run}/{cid}.json"
            cmd = (f"mkdir -p {REPO}/output/rebuttal/{args.run} && cd {REPO} && "
                   f"setsid nohup env CUDA_VISIBLE_DEVICES={gpu} {ENV} "
                   f"{py} -m scripts.rebuttal.patch_search --concepts {donor}:{feat} "
                   f"{args.flags} --out {rout} > /tmp/{args.run}_{cid}.log 2>&1 "
                   f"< /dev/null &")
            try:
                subprocess.run(["ssh", "-f", "-n", "-o", "ConnectTimeout=10",
                                host, cmd], check=True, timeout=45)
            except (subprocess.SubprocessError, OSError):
                continue
            attempts[cid_k := (donor, feat)] = attempts.get(cid_k, 0) + 1
            state[host][0].add(gpu)
            print(f"LAUNCH {cid} -> {host}:{gpu} (attempt {attempts[cid_k]}, "
                  f"{len(todo)} queued)", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
