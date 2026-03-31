"""
cache.py — Request-level analysis cache.
Key = SHA256(resume_text + jd_text + role)
Stores full analysis result in memory.
For production: swap _STORE for Redis with TTL.
"""
import hashlib
import time
from typing import Optional

_STORE: dict = {}          # key → {"result": dict, "ts": float}
_MAX_ENTRIES  = 500        # evict oldest when full
_TTL_SECONDS  = 3600       # 1 hour TTL


def _make_key(resume_text: str, jd_text: str, role: str) -> str:
    raw = f"{resume_text.strip()}{jd_text.strip()}{role.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get(resume_text: str, jd_text: str, role: str) -> Optional[dict]:
    """Return cached result if it exists and is not expired."""
    key = _make_key(resume_text, jd_text, role)
    entry = _STORE.get(key)
    if entry is None:
        return None
    if time.time() - entry["ts"] > _TTL_SECONDS:
        del _STORE[key]
        return None
    print(f"[cache] HIT  key={key[:12]}...", flush=True)
    return entry["result"]


def set(resume_text: str, jd_text: str, role: str, result: dict) -> None:
    """Store result. Evict oldest entries if over limit."""
    if len(_STORE) >= _MAX_ENTRIES:
        # Evict oldest 20%
        sorted_keys = sorted(_STORE, key=lambda k: _STORE[k]["ts"])
        for old_key in sorted_keys[:_MAX_ENTRIES // 5]:
            del _STORE[old_key]

    key = _make_key(resume_text, jd_text, role)
    _STORE[key] = {"result": result, "ts": time.time()}
    print(f"[cache] SET  key={key[:12]}... entries={len(_STORE)}", flush=True)


def stats() -> dict:
    return {"entries": len(_STORE), "keys": [k[:12] for k in list(_STORE)[-5:]]}