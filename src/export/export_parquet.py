import json, pandas as pd
from pathlib import Path

def export_parquet(canon_path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(canon_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    pd.DataFrame(rows).to_parquet(out_dir/"tripletas.parquet", index=False)
