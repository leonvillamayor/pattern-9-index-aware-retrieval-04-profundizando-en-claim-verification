import hashlib

def chunk_id(source: str, position: int, version: str) -> str:
    raw = f"{source}:{position}:{version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:10]  # ej. "7a3f9b2c1d"