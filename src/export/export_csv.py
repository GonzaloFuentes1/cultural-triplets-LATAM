import json, pandas as pd
from pathlib import Path

def export_csv(canon_path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(canon_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    # ensure columns exist
    for col in ["entity","relation","value","confidence"]:
        if col not in df.columns:
            df[col] = None
    df_out = df[["entity","relation","value","confidence"]].rename(columns={
        "entity":"Entidad","relation":"Relacion","value":"Valor","confidence":"Confidence"
    })
    df_out.to_csv(out_dir/"tripletas.csv", index=False)
