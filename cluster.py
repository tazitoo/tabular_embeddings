#!/usr/bin/env python3
"""
Lightweight distributed compute infrastructure for tabular embeddings.

Adapted from finance project's library/distributed.py. Provides distributed
execution across GPU workers without Prefect dependency.

Usage:
    # Context manager for ad-hoc distributed compute
    from cluster import gpu_cluster

    with gpu_cluster() as client:
        futures = client.map(my_task_func, task_args)
        results = client.gather(futures)

    # Run tasks across workers with progress reporting
    from cluster import run_on_workers

    results = run_on_workers(my_func, args_list)

    # Check worker health
    python cluster.py --check
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# GPU worker configuration
GPU_WORKERS: Dict[str, Dict] = {
    "surfer4": {
        "ip": "192.168.10.6",
        "gpu": "RTX 3090",
        "vram_gb": 24,
        "enabled": True,
    },
    "terrax4": {
        "ip": "192.168.10.4",
        "gpu": "RTX 2080 Ti",
        "vram_gb": 11,
        "enabled": True,
    },
    "octo4": {
        "ip": "192.168.10.12",
        "gpu": "RTX 3070",
        "vram_gb": 8,
        "enabled": True,
    },
    "firelord4": {
        "ip": "192.168.10.8",
        "gpu": "RTX 4090",
        "vram_gb": 24,
        "enabled": True,
    },
}

# Orchestrator (scheduler) configuration
SCHEDULER_IP = "192.168.10.50"  # galactus on compute VLAN
SCHEDULER_PORT = 8786
DASHBOARD_PORT = 8787

# Worker environment (Linux GPU workers — using finance env which has torch+CUDA+model deps)
WORKER_PYTHON = "/home/brian/anaconda3/envs/finance/bin/python"
WORKER_DASK_WORKER = "/home/brian/anaconda3/envs/finance/bin/dask-worker"
WORKER_BASE_DIR = "/home/brian/src/tabular_embeddings"


@dataclass
class ClusterConfig:
    """Configuration for distributed cluster."""

    scheduler_ip: str = SCHEDULER_IP
    scheduler_port: int = SCHEDULER_PORT
    dashboard_port: int = DASHBOARD_PORT
    workers: Dict[str, Dict] = field(default_factory=lambda: GPU_WORKERS.copy())
    memory_limit: str = "16GB"
    n_threads: int = 1
    n_workers_per_host: int = 1
    worker_timeout: int = 30  # seconds to wait for workers to connect

    @property
    def scheduler_address(self) -> str:
        return f"tcp://{self.scheduler_ip}:{self.scheduler_port}"

    @property
    def dashboard_address(self) -> str:
        return f"http://{self.scheduler_ip}:{self.dashboard_port}/status"

    @property
    def enabled_workers(self) -> Dict[str, Dict]:
        """Return only enabled workers."""
        return {k: v for k, v in self.workers.items() if v.get("enabled", True)}

    @property
    def worker_ips(self) -> List[str]:
        """Return list of enabled worker IPs."""
        return [w["ip"] for w in self.enabled_workers.values()]

    @property
    def worker_hosts(self) -> List[str]:
        """Return list of enabled worker hostnames."""
        return list(self.enabled_workers.keys())


# Default cluster configuration
DEFAULT_CONFIG = ClusterConfig()


def get_enabled_workers(include_unstable: bool = False) -> List[str]:
    """Get list of enabled GPU worker hostnames."""
    workers = []
    for name, info in GPU_WORKERS.items():
        if info.get("enabled", True) or include_unstable:
            workers.append(name)
    return workers


def get_worker_ip(hostname: str) -> Optional[str]:
    """Get IP address for a worker hostname."""
    worker = GPU_WORKERS.get(hostname)
    return worker["ip"] if worker else None


def sync_code_to_workers(
    workers: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Sync code to GPU workers via git pull.

    Returns:
        Tuple of (successful_workers, failed_workers)
    """
    if workers is None:
        workers = get_enabled_workers()

    if verbose:
        print("Syncing code to GPU workers...")

    successful = []
    failed = []

    for worker in workers:
        if verbose:
            print(f"  {worker}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                ["ssh", worker, f"cd {WORKER_BASE_DIR} && GIT_TERMINAL_PROMPT=0 git pull --ff-only"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                if "Already up to date" in result.stdout:
                    if verbose:
                        print("(up to date)")
                else:
                    if verbose:
                        print("(updated)")
                successful.append(worker)
            else:
                if verbose:
                    print(f"FAILED: {result.stderr.strip()[:50]}")
                failed.append(worker)
        except subprocess.TimeoutExpired:
            if verbose:
                print("FAILED: timeout")
            failed.append(worker)
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            failed.append(worker)

    return successful, failed


def check_worker_health(
    workers: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Check if workers are reachable and have torch+CUDA available.

    Returns:
        Dict mapping worker hostname to health status
    """
    if workers is None:
        workers = get_enabled_workers()

    health = {}

    for worker in workers:
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=5", worker,
                    f"{WORKER_PYTHON} -c "
                    "\"import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')\""
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            ok = result.returncode == 0 and "cuda=True" in result.stdout
            health[worker] = ok
            if verbose:
                if ok:
                    print(f"  {worker}: OK ({result.stdout.strip()})")
                else:
                    stderr = result.stderr.strip()[:60] if result.stderr else "no CUDA"
                    print(f"  {worker}: FAIL ({stderr})")
        except Exception as e:
            health[worker] = False
            if verbose:
                print(f"  {worker}: FAIL ({e})")

    return health


@contextmanager
def gpu_cluster(
    config: Optional[ClusterConfig] = None,
    sync_code: bool = True,
    verbose: bool = True,
    prevent_sleep: bool = True,
) -> Generator:
    """
    Context manager for ad-hoc distributed GPU cluster.

    Creates a LocalCluster scheduler on the orchestrator and connects
    remote workers via SSH. Automatically cleans up on exit.

    Yields:
        dask.distributed.Client connected to the cluster
    """
    from dask.distributed import Client, LocalCluster
    import platform

    if config is None:
        config = DEFAULT_CONFIG

    # Sync code to workers
    if sync_code:
        successful, failed = sync_code_to_workers(config.worker_hosts, verbose=verbose)
        if failed and verbose:
            print(f"  Warning: {len(failed)} workers failed to sync")

    worker_procs = []
    cluster = None
    client = None
    caffeinate_proc = None

    # Prevent macOS from sleeping during cluster operation
    if prevent_sleep and platform.system() == "Darwin":
        try:
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-s"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if verbose:
                print("Sleep prevention enabled (caffeinate)")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not start caffeinate: {e}")

    try:
        # Start local scheduler (no local workers)
        if verbose:
            print(f"Starting scheduler on {config.scheduler_ip}...")

        cluster = LocalCluster(
            n_workers=0,
            scheduler_port=config.scheduler_port,
            host=config.scheduler_ip,
            dashboard_address=f":{config.dashboard_port}",
        )
        client = Client(cluster)

        if verbose:
            print(f"  Scheduler: {config.scheduler_address}")
            print(f"  Dashboard: {config.dashboard_address}")

        # Start remote workers via SSH
        for name, info in config.enabled_workers.items():
            ip = info["ip"]
            if verbose:
                print(f"  Starting worker on {name} ({ip})...")

            worker_cmd = (
                f"cd {WORKER_BASE_DIR} && "
                f"PYTHONPATH={WORKER_BASE_DIR}:$PYTHONPATH "
                f"{WORKER_DASK_WORKER} {config.scheduler_address}"
                f" --nthreads {config.n_threads}"
                f" --nworkers {config.n_workers_per_host}"
                f" --no-dashboard"
                f" --memory-limit {config.memory_limit}"
                f" --name {name}"
            )
            proc = subprocess.Popen(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ServerAliveInterval=60",
                    "-o", "ServerAliveCountMax=10",
                    ip,
                    worker_cmd,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            worker_procs.append(proc)

        # Wait for workers to connect
        if verbose:
            print("Waiting for workers to connect...")

        expected_workers = len(config.enabled_workers)
        for _ in range(config.worker_timeout):
            connected = len(client.scheduler_info()["workers"])
            if connected >= expected_workers:
                break
            time.sleep(1)

        connected = len(client.scheduler_info()["workers"])
        if verbose:
            print(f"  Connected workers: {connected}/{expected_workers}")

        if connected == 0:
            raise RuntimeError("No workers connected to scheduler")

        yield client

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

        if cluster is not None:
            try:
                cluster.close()
            except Exception:
                pass

        for proc in worker_procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if caffeinate_proc is not None:
            try:
                caffeinate_proc.terminate()
                caffeinate_proc.wait(timeout=2)
            except Exception:
                pass


def run_on_workers(
    func,
    args_list: List,
    config: Optional[ClusterConfig] = None,
    sync_code: bool = True,
    verbose: bool = True,
    timeout_per_task: int = 600,
) -> List:
    """
    Run a function across GPU workers with automatic cluster management.

    Args:
        func: Function to execute (must be importable on workers)
        args_list: List of argument tuples for each task
        config: Cluster configuration
        sync_code: Sync code to workers before starting
        verbose: Print progress
        timeout_per_task: Timeout in seconds per task

    Returns:
        List of results in same order as args_list
    """
    with gpu_cluster(config=config, sync_code=sync_code, verbose=verbose) as client:
        from dask.distributed import as_completed

        n_tasks = len(args_list)
        if verbose:
            print(f"Submitting {n_tasks} tasks...")

        futures = client.map(func, args_list)

        # Map futures back to their original indices for ordered results
        future_to_idx = {f: i for i, f in enumerate(futures)}
        results = [None] * n_tasks

        completed = 0
        failed = 0

        # Progress frequency: every 5 tasks or 2%, whichever is more frequent
        progress_interval = min(5, max(1, n_tasks // 50))

        for future in as_completed(futures, with_results=False):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=timeout_per_task)
                results[idx] = result
                completed += 1

                if verbose and (completed % progress_interval == 0 or completed == n_tasks):
                    info = ""
                    if isinstance(result, dict):
                        model = result.get("model", "")
                        dataset = result.get("dataset", "")
                        worker = result.get("worker", "")
                        if model and dataset:
                            info = f" | {model} @ {dataset} ({worker})"
                    print(f"  [{completed}/{n_tasks}] {completed*100//n_tasks}% done{info}")

            except Exception as e:
                results[idx] = None
                failed += 1
                if verbose:
                    print(f"  Task {idx} failed: {e}")

        if verbose:
            print(f"Completed: {completed}/{n_tasks} tasks ({failed} failed)")

        return results


# Worker-side functions

def is_running_on_worker() -> bool:
    """Check if code is running on a GPU worker (vs orchestrator)."""
    hostname = socket.gethostname()
    return hostname in GPU_WORKERS


def get_worker_device() -> str:
    """Get appropriate torch device for current host."""
    if is_running_on_worker():
        return "cuda"
    else:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def extract_embeddings_task(args: dict) -> dict:
    """
    Worker-side task: extract embeddings for one (model, dataset) pair.

    Args:
        args: dict with keys:
            - model_name: str
            - X_context: np.ndarray
            - y_context: np.ndarray
            - X_query: np.ndarray
            - dataset_name: str

    Returns:
        dict with keys: model, dataset, embeddings, layer_embeddings, worker, status
    """
    import socket
    import numpy as np

    model_name = args["model_name"]
    X_context = args["X_context"]
    y_context = args["y_context"]
    X_query = args["X_query"]
    dataset_name = args["dataset_name"]
    worker = socket.gethostname()

    try:
        # Import inside worker to avoid serialization issues
        sys.path.insert(0, WORKER_BASE_DIR)
        from models import get_extractor

        device = get_worker_device()
        extractor = get_extractor(model_name, device=device)
        extractor.load_model()
        result = extractor.extract_embeddings(X_context, y_context, X_query)

        layer_embs = {k: v for k, v in result.layer_embeddings.items()}

        return {
            "model": model_name,
            "dataset": dataset_name,
            "embeddings": result.embeddings,
            "layer_embeddings": layer_embs,
            "worker": worker,
            "status": "ok",
        }
    except Exception as e:
        return {
            "model": model_name,
            "dataset": dataset_name,
            "embeddings": None,
            "layer_embeddings": {},
            "worker": worker,
            "status": f"failed: {e}",
        }


def extract_sae_embeddings_task(args: dict) -> dict:
    """
    Worker-side task: extract sliding-window embeddings for SAE training.

    Args:
        args: dict with keys:
            - model_name: str
            - X_all: np.ndarray
            - y_all: np.ndarray
            - dataset_name: str

    Returns:
        dict with keys: model, dataset, embeddings, worker, status
    """
    import socket
    import numpy as np

    model_name = args["model_name"]
    X_all = args["X_all"]
    y_all = args["y_all"]
    dataset_name = args["dataset_name"]
    worker = socket.gethostname()

    try:
        sys.path.insert(0, WORKER_BASE_DIR)
        from models import get_extractor

        device = get_worker_device()
        extractor = get_extractor(model_name, device=device)
        extractor.load_model()

        emb_list = []
        window_size = min(500, len(X_all) // 3)
        step = window_size // 2

        for start in range(0, len(X_all) - window_size, step):
            ctx_end = start + window_size
            query_end = min(ctx_end + step, len(X_all))

            result = extractor.extract_embeddings(
                X_all[start:ctx_end],
                y_all[start:ctx_end],
                X_all[ctx_end:query_end],
            )
            emb_list.append(result.embeddings)

        if emb_list:
            all_emb = np.vstack(emb_list)
        else:
            all_emb = None

        return {
            "model": model_name,
            "dataset": dataset_name,
            "embeddings": all_emb,
            "worker": worker,
            "status": "ok",
        }
    except Exception as e:
        return {
            "model": model_name,
            "dataset": dataset_name,
            "embeddings": None,
            "worker": worker,
            "status": f"failed: {e}",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed compute utilities")
    parser.add_argument("--check", action="store_true", help="Check worker health")
    parser.add_argument("--sync", action="store_true", help="Sync code to workers")
    args = parser.parse_args()

    if args.check:
        print("Checking worker health...")
        health = check_worker_health()
        n_healthy = sum(health.values())
        print(f"\n{n_healthy}/{len(health)} workers healthy")
        sys.exit(0 if n_healthy > 0 else 1)

    if args.sync:
        successful, failed = sync_code_to_workers()
        print(f"\n{len(successful)} synced, {len(failed)} failed")
        sys.exit(0 if not failed else 1)

    parser.print_help()
