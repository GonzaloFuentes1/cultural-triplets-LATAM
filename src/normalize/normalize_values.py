from src.utils.text import to_iso_date

def normalize_value(rel: str, val: str) -> str:
    if rel in {"fecha_nacimiento","fecha_muerte"}:
        return to_iso_date(val) or val
    return val.strip()
