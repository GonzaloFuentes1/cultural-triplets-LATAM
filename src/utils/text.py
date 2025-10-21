import re

def to_iso_date(s: str) -> str | None:
    s = s.strip()
    # dd de MMMM de yyyy
    m = re.search(r"(\d{1,2})\s+de\s+([A-Za-záéíóúñ]+)\s+de\s+(\d{4})", s, re.I)
    months = {
        "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
        "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
    }
    if m:
        d, mon, y = int(m.group(1)), months[m.group(2).lower()], int(m.group(3))
        return f"{y:04d}-{mon:02d}-{d:02d}"
    # yyyy
    m = re.search(r"\b(1\d{3}|20\d{2})\b", s)
    if m:
        return m.group(1)
    return None
