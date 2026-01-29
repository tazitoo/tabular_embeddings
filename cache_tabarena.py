#!/usr/bin/env python3
"""Pre-cache all TabArena datasets from OpenML to local disk."""

import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.extended_loader import TABARENA_DATASETS


def timeout_handler(signum, frame):
    raise TimeoutError("Download timed out")


def cache_dataset(name: str, info: dict, timeout_sec: int = 120) -> bool:
    """Download one dataset with a timeout."""
    from sklearn.datasets import fetch_openml

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        data = fetch_openml(data_id=info["openml_id"], as_frame=True, parser="auto")
        signal.alarm(0)
        n_rows = len(data.data)
        n_cols = data.data.shape[1]
        print(f"  OK ({n_rows} x {n_cols}, {info['task']}, {info['domain']})")
        return True
    except TimeoutError:
        print(f"  TIMEOUT (>{timeout_sec}s)")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"  FAILED: {e}")
        return False


def main():
    print(f"Pre-caching {len(TABARENA_DATASETS)} TabArena datasets from OpenML...\n")

    ok, fail, timeout = [], [], []
    for i, (name, info) in enumerate(TABARENA_DATASETS.items(), 1):
        print(f"[{i}/{len(TABARENA_DATASETS)}] {name}...", end="", flush=True)
        success = cache_dataset(name, info)
        if success:
            ok.append(name)
        else:
            fail.append(name)

    print(f"\n{'='*50}")
    print(f"Cached: {len(ok)}/{len(TABARENA_DATASETS)}")
    if fail:
        print(f"Failed: {', '.join(fail)}")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
