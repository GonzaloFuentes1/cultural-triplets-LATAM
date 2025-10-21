# Tripletas Culturales LATAM

Pipeline para extraer tripletas (Entidad–Relación–Valor) desde Wikipedia/WEB con GLiNER (NER) + GLiREL (RE) y export a CSV/Neo4j.

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -e .
# Instalar GLiREL (como submódulo o editable)
git clone https://github.com/jackboyla/GLiREL.git external/GLiREL
pip install -e external/GLiREL
```

## Configurar
Copia `.env.example` a `.env` si quieres definir variables (API keys LLM, etc.). Revisa `config/config.yaml` para parámetros.

## Uso rápido
```bash
# 1) Ingesta: Wikipedia categoría (ej. Músicos de Chile)
tripletas ingest --source wiki --category "Músicos de Chile" --limit 100

# 2) Preprocesar (HTML->texto->chunks)
tripletas preprocess

# 3) Extraer entidades (GLiNER) + relaciones (GLiREL) + atributos
tripletas extract

# 4) Normalizar + validar + deduplicar
tripletas normalize-validate

# 5) Exportar CSV/Neo4j
tripletas export --fmt csv
tripletas export --fmt neo4j
```

## Salida (CSV)
`data/exports/tripletas.csv` con columnas: `Entidad,Relacion,Valor,Confidence,SourceURL`

## Notas
- GLiREL es un clasificador zero‑shot de relaciones (multi‑relación en un forward). Debe estar instalado.
- Para atributos literales (fechas/premios), usamos regex/heurísticas; opcionalmente se puede activar un paso LLM ligero.
