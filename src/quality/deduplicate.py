from collections import defaultdict

def _key(r):
    return (r["entity"].lower(), r["relation"], r["value"].lower())

def deduplicate(records: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for r in records:
        buckets[_key(r)].append(r)
    canon = []
    for _, rows in buckets.items():
        best = max(rows, key=lambda x: x.get("confidence",0))
        canon.append(best)
    return canon
