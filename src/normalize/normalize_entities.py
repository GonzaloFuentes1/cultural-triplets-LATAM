import yaml, regex as re
ALIASES = yaml.safe_load(open("config/entity_aliases.yaml","r",encoding="utf-8"))

def normalize_entity(name: str) -> str:
    for canon, alist in ALIASES.items():
        if name == canon or name in alist:
            return canon
    return re.sub(r"\s+", " ", name).strip()
