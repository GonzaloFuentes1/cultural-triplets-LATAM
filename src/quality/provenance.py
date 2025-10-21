def attach_provenance(r: dict, source: str):
    r.setdefault("provenance", []).append(source)
    return r
