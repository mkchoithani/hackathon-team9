"""Fast model loading from the shared NFS volume on the fb-hack GB10 nodes.

Loaded automatically (via PYTHONPATH) by every Python process that scripts/launch_train.sh starts.

Why: NeMo AutoModel's HF checkpoint reader (_backports/hf_storage.py: read_data) opens each
safetensors shard with ``mmap.mmap(fd, 0, access=ACCESS_COPY)`` and faults pages in one at a time.
Over NFS that runs at ~25 MB/s, so loading the 59 GB Nemotron-3-Nano took ~49 min. Plain sequential
reads of the same files run at ~1-2 GB/s. Pre-warming the page cache did not help (the mmap faults
still went to the server), and copying the whole model to /tmp does not work either: /tmp in a
Fuzzball job is RAM, which on the GB10 is also GPU memory, so the 59 GB copy OOM-killed the trainer.

What: when mmap is asked to map a ``$FASTLOAD_SRC/*.safetensors`` file, copy that ONE shard into
``$FASTLOAD_DIR`` (RAM) with a sequential read, then map the copy instead. The previous shard's copy
is deleted first, so only ~5 GB of extra RAM is used at a time. Measured on 2026-11-14: 13 shards in
~1 min instead of ~49 min, same step-0 loss (1.167) and memory (61.9 GiB).

Env:
  FASTLOAD_SRC  path prefix to accelerate (set by launch_train.sh to the model dir); unset = no-op
  FASTLOAD_DIR  where shard copies go (default /tmp/fastload_shards)
"""
import os
import shutil
import sys
import threading
import time

_SRC = os.environ.get("FASTLOAD_SRC", "")
_DIR = os.environ.get("FASTLOAD_DIR", "/tmp/fastload_shards")
_lock = threading.Lock()


def _log(msg):
    print(f"[fastload] {msg}", file=sys.stderr, flush=True)


def _stage(path):
    """Copy one shard into RAM (dropping any other staged shard) and return the copy's path."""
    name = os.path.basename(path)
    dst = os.path.join(_DIR, name)
    with _lock:
        os.makedirs(_DIR, exist_ok=True)
        if not os.path.exists(dst):
            for other in os.listdir(_DIR):
                if other != name:
                    try:
                        os.remove(os.path.join(_DIR, other))
                    except OSError:
                        pass
            t0 = time.time()
            tmp = dst + ".part"
            with open(path, "rb") as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 64 << 20)
            os.rename(tmp, dst)
            gib = os.path.getsize(dst) / 2**30
            dt = time.time() - t0
            _log(f"{name}: {gib:.1f} GiB copied in {dt:.1f}s ({gib / max(dt, 1e-3):.2f} GiB/s)")
    return dst


if _SRC:
    try:
        import mmap as _mmap_mod

        _Base = _mmap_mod.mmap

        class _FastMmap(_Base):
            def __new__(cls, fileno, length, *args, **kwargs):
                path = ""
                if isinstance(fileno, int) and fileno >= 0:
                    try:
                        path = os.readlink(f"/proc/self/fd/{fileno}")
                    except OSError:
                        path = ""
                if path.startswith(_SRC) and path.endswith(".safetensors"):
                    try:
                        fd = os.open(_stage(path), os.O_RDONLY)
                    except Exception as e:  # fall back to the slow NFS mmap rather than fail
                        _log(f"staging failed for {path}, using NFS: {e!r}")
                    else:
                        try:
                            return _Base.__new__(cls, fd, length, *args, **kwargs)
                        finally:
                            os.close(fd)
                return _Base.__new__(cls, fileno, length, *args, **kwargs)

        _mmap_mod.mmap = _FastMmap
        _log(f"mmap hook active in pid {os.getpid()} for {_SRC}")
    except Exception as e:
        _log(f"hook NOT installed: {e!r}")

# Chain to any other sitecustomize.py this one shadows (e.g. the DUMP_STACKS_EVERY one).
_me = os.path.dirname(os.path.abspath(__file__))
for _d in sys.path:
    _f = os.path.join(_d or ".", "sitecustomize.py")
    if os.path.abspath(_d or ".") != _me and os.path.isfile(_f):
        with open(_f) as _fh:
            exec(compile(_fh.read(), _f, "exec"), {"__name__": "sitecustomize_chained", "__file__": _f})
        break
