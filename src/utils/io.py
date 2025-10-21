from pathlib import Path
from dataclasses import dataclass

@dataclass
class Paths:
    raw: Path
    interim: Path
    triples: Path
    exports: Path

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(Path(cfg["raw"]), Path(cfg["interim"]), Path(cfg["triples"]), Path(cfg["exports"]))
