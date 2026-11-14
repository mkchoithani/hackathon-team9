#!/usr/bin/env python3
"""Drop the Linux page cache of the model files, without root.

Why: GB10 (DGX Spark class) has ONE memory pool shared by CPU and GPU. Reading the 59 GB of weights
leaves ~59 GB of file pages in the page cache, and CUDA allocations on this unified memory do not
reclaim it: the first GB10 smoke run died with "CUDA out of memory ... 1.82 GiB free" while the
process itself held 47 GiB. The usual fix (echo 3 > /proc/sys/vm/drop_caches) needs root; this uses
posix_fadvise(POSIX_FADV_DONTNEED) on each file instead, which any user that can open the file may do.

    python scripts/evict_page_cache.py /tmp/nemotron-local                     # once
    python scripts/evict_page_cache.py /tmp/nemotron-local --every 2 --for 900 # keep evicting while loading
"""
import argparse
import os
import time
from pathlib import Path


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return -1.0


def evict(dirs):
    n = 0
    for d in dirs:
        for p in Path(d).rglob("*"):
            if not p.is_file():
                continue
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fdatasync(fd)  # dirty pages (fresh copy) must be written before they can be dropped
            except OSError:
                pass
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--every", type=float, default=0, help="repeat every N seconds (0 = once)")
    ap.add_argument("--for", dest="duration", type=float, default=900, help="stop repeating after N seconds")
    a = ap.parse_args()
    before = mem_available_gib()
    n = evict(a.dirs)
    print(f"[evict] {n} files, MemAvailable {before:.1f} -> {mem_available_gib():.1f} GiB", flush=True)
    end = time.time() + a.duration
    while a.every and time.time() < end:
        time.sleep(a.every)
        evict(a.dirs)


if __name__ == "__main__":
    main()
