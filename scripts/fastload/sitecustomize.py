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

Hugging Face loading (scripts/07_predict.py --backend hf) goes through safetensors.safe_open instead;
that is replaced by a pread()-based reader (see below). To use it outside launch_train.sh:
  FASTLOAD_SRC=/shared/nemotron-3-nano-30b-a3b-bf16 PYTHONPATH=scripts/fastload python3 scripts/07_predict.py ...

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

    # Hugging Face transformers (scripts/07_predict.py --backend hf) loads shards with
    # safetensors.safe_open, which mmaps in Rust and never calls Python's mmap. Staging those shards
    # in RAM does NOT work: transformers keeps every shard open until loading ends, so the staged
    # copies can't be freed (59 GB of copies + 59 GB of weights -> OOM-killed, seen 2026-11-14), and
    # pre-warming the page cache doesn't hold either (on GB10 the GPU's own allocation evicts it).
    # Instead, read each tensor with plain pread() in one large read (~1.5 GB/s over NFS) and keep
    # nothing resident: verified byte-identical to safe_open, model ready in ~70 s (2026-11-15).
    try:
        import json
        import struct

        import safetensors as _st
        import torch

        _orig_safe_open = _st.safe_open
        _DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
               "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
               "BOOL": torch.bool}
        for _k, _a in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
            if hasattr(torch, _a):
                _DT[_k] = getattr(torch, _a)

        class _Slice:
            def __init__(self, f, name):
                self._f, self._name = f, name
                self._info = f._hdr[name]

            def get_shape(self):
                return list(self._info["shape"])

            def get_dtype(self):
                return self._info["dtype"]

            @property
            def shape(self):
                return self.get_shape()

            def __getitem__(self, idx):
                return self._f.get_tensor(self._name)[idx]

        class _PreadSafeOpen:
            """The subset of safetensors.safe_open that transformers uses, backed by pread()."""

            def __init__(self, path, framework="pt", device="cpu"):
                self._path, self._device = os.fspath(path), device
                self._fd = os.open(self._path, os.O_RDONLY)
                (n,) = struct.unpack("<Q", os.pread(self._fd, 8, 0))
                hdr = json.loads(os.pread(self._fd, n, 8))
                self._meta = hdr.pop("__metadata__", None)
                self._hdr, self._base = hdr, 8 + n

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
                return False

            def close(self):
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None

            def __del__(self):
                try:
                    self.close()
                except Exception:
                    pass

            def keys(self):
                return list(self._hdr.keys())

            def offset_keys(self):
                return sorted(self._hdr, key=lambda k: self._hdr[k]["data_offsets"][0])

            def metadata(self):
                return self._meta

            def get_slice(self, name):
                return _Slice(self, name)

            def get_tensor(self, name):
                info = self._hdr[name]
                a, b = info["data_offsets"]
                buf = bytearray(b - a)
                mv, got = memoryview(buf), 0
                while got < len(buf):  # pread may return short reads on large sizes
                    chunk = os.pread(self._fd, min(len(buf) - got, 1 << 30), self._base + a + got)
                    if not chunk:
                        raise IOError(f"short read in {self._path} for {name}")
                    mv[got:got + len(chunk)] = chunk
                    got += len(chunk)
                dt = _DT[info["dtype"]]
                t = torch.frombuffer(buf, dtype=dt) if len(buf) else torch.empty(0, dtype=dt)
                t = t.reshape(info["shape"])
                return t.to(self._device) if self._device not in (None, "cpu") else t

        def safe_open(filename, framework="pt", device="cpu", *args, **kwargs):
            path = os.fspath(filename)
            if path.startswith(_SRC) and path.endswith(".safetensors") and framework in ("pt", "torch"):
                return _PreadSafeOpen(path, framework, device)
            return _orig_safe_open(filename, framework, device, *args, **kwargs)

        _st.safe_open = safe_open
        try:
            import safetensors.torch as _stt
            _stt.safe_open = safe_open
        except Exception:
            pass
        _log(f"pread safe_open active in pid {os.getpid()}")
    except Exception as e:
        _log(f"safe_open hook NOT installed: {e!r}")

# Chain to any other sitecustomize.py this one shadows (e.g. the DUMP_STACKS_EVERY one).
_me = os.path.dirname(os.path.abspath(__file__))
for _d in sys.path:
    _f = os.path.join(_d or ".", "sitecustomize.py")
    if os.path.abspath(_d or ".") != _me and os.path.isfile(_f):
        with open(_f) as _fh:
            exec(compile(_fh.read(), _f, "exec"), {"__name__": "sitecustomize_chained", "__file__": _f})
        break
