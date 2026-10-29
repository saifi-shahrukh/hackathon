"""Tiny cross-venv codec for numpy arrays (RCS venv <-> lerobot venv) over multiprocessing.connection.

Arrays travel as (dtype, shape, bytes) so differing numpy versions never have to unpickle each
other's ndarray objects.
"""

from __future__ import annotations

import numpy as np

PORT = 6111
AUTHKEY = b"fr3duo"


def pack(x) -> tuple:
    a = np.ascontiguousarray(x)
    return ("nd", str(a.dtype), a.shape, a.tobytes())


def unpack(t) -> np.ndarray:
    _, dtype, shape, buf = t
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()


def pack_obs(images: dict, state: np.ndarray) -> dict:
    return {"images": {k: pack(v) for k, v in images.items()}, "state": pack(state)}


def unpack_obs(msg: dict) -> tuple[dict, np.ndarray]:
    return {k: unpack(v) for k, v in msg["images"].items()}, unpack(msg["state"])
