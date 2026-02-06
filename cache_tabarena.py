#!/usr/bin/env python3
"""Pre-cache all TabArena datasets to data/cache/tabarena/ as .npz files.

After running this, load_tabarena_dataset() loads from cache (~100ms)
instead of downloading from OpenML (~30-60s per dataset).
"""

import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.extended_loader import TABARENA_DATASETS, load_tabarena_dataset, _TABARENA_CACHE_DIR


def timeout_handler(signum, frame):
    raise TimeoutError("Download timed out")


def main():
    print(f"Pre-caching {len(TABARENA_DATASETS)} TabArena datasets")
    print(f"Cache dir: {_TABARENA_CACHE_DIR}\n")

    ok, fail = [], []
    for i, (name, info) in enumerate(TABARENA_DATASETS.items(), 1):
        cache_path = _TABARENA_CACHE_DIR / f"{name}.npz"
        if cache_path.exists():
            print(f"[{i}/{len(TABARENA_DATASETS)}] {name}: cached")
            ok.append(name)
            continue

        print(f"[{i}/{len(TABARENA_DATASETS)}] {name}...", end="", flush=True)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)
        try:
            result = load_tabarena_dataset(name, max_samples=999_999)
            signal.alarm(0)
            if result is not None:
                X, y, meta = result
                print(f"  OK ({meta.n_samples} x {meta.n_features}, {info['task']})")
                ok.append(name)
            else:
                print("  FAILED: loader returned None")
                fail.append(name)
        except TimeoutError:
            print("  TIMEOUT (>120s)")
            fail.append(name)
        except Exception as e:
            signal.alarm(0)
            print(f"  FAILED: {e}")
            fail.append(name)

    print(f"\n{'='*50}")
    print(f"Cached: {len(ok)}/{len(TABARENA_DATASETS)}")
    if fail:
        print(f"Failed: {', '.join(fail)}")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
