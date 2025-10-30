"""
Módulo de control de calidad integrado para tripletas.
Incluye normalización de entidades y filtros de calidad en un solo proceso.
"""

from .quality_filter import apply_quality_filters, save_canonical_triplets, load_config

__all__ = [
    "apply_quality_filters",
    "save_canonical_triplets", 
    "load_config"
]
