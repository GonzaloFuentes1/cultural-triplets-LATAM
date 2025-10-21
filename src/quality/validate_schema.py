from typing import Dict

RELS = {"fecha_nacimiento","lugar_nacimiento","fecha_muerte","lugar_muerte","ocupacion","nombre_padre","nombre_madre","conyuge","fecha_separacion","instrumento","premio_recibido","hermana","residio_en","primer_disco","trabajo_en","cargo","autor","miembro_de"}

def is_valid_record(r: Dict) -> bool:
    for k in ("entity","relation","value"):
        if not r.get(k):
            return False
    if r["relation"] not in RELS:
        return False
    if float(r.get("confidence",0)) < 0.5:
        return False
    return True
