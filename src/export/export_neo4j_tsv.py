import json
from pathlib import Path

def export_neo4j(canon_path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = {}; rels = []
    def nid(name):
        if name not in nodes:
            nodes[name] = f"N{len(nodes)+1}"
        return nodes[name]
    for line in Path(canon_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        h = nid(r["entity"])
        # Para literales simples, igual los representamos como nodos de texto
        t = nid(r["value"])
        rels.append((h, r["relation"], t))
    (out_dir/"nodes.tsv").write_text("id\tname\n" + "\n".join(f"{i}\t{n}" for n,i in nodes.items()), encoding="utf-8")
    (out_dir/"rels.tsv").write_text(":START_ID\t:TYPE\t:END_ID\n" + "\n".join(f"{h}\t{rel}\t{t}" for h,rel,t in rels), encoding="utf-8")
