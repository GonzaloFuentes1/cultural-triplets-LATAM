import typer, yaml, json, pathlib
from .utils.io import Paths
from .ingest.wiki_api import wiki_ingest_category
from .preprocess.clean_text import html_to_text
from .preprocess.segment import segment_text
from .ingest.category_scraper import scrape_categories_by_country

from .export.export_csv import export_csv
from tqdm import tqdm

app = typer.Typer()

CONFIG = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
PATHS = Paths.from_config(CONFIG["paths"])

@app.command(name="scrape-categories")
def scrape_categories():
    """Scrape categories from Wikipedia (not implemented yet)."""
    print("⚠️  Comando no implementado aún")

@app.command(name="full-pipeline")
def full_pipeline():
    """
    Ejecuta el pipeline completo desde cero: ingest → preprocess → extract → quality (integrado).
    """
    print("🚀 INICIANDO PIPELINE COMPLETO DESDE CERO")
    print("=" * 50)
    
    # 1. Ingest (método API mejorado por defecto)
    raw_json_files = list(pathlib.Path(PATHS.raw).glob("*.json"))
    if not raw_json_files:
        print("1️⃣  INGESTA - Descargando con API Wikipedia (wikitext completo)...")
        ingest(method="api", limit=50)  # Usar método API por defecto
    else:
        print(f"1️⃣  INGESTA - ✓ Ya hay {len(raw_json_files)} archivos JSON de API")
    
    # 2. Preprocess (maneja JSON de API automáticamente)
    interim_files = list(pathlib.Path(PATHS.interim).glob("*.json"))
    if not interim_files:
        print("2️⃣  PREPROCESSING - Procesando contenido limpio de API...")
        preprocess()
    else:
        print(f"2️⃣  PREPROCESSING - ✓ Ya hay {len(interim_files)} archivos procesados")
    
    # 3. Extract (si no hay tripletas)
    raw_triplet_files = [f for f in pathlib.Path(PATHS.triples).glob("*.jsonl") 
                        if f.name not in ["canon.jsonl"]]
    if not raw_triplet_files:
        print("3️⃣  EXTRACCIÓN - Extrayendo tripletas...")
        extract()
    else:
        print(f"3️⃣  EXTRACCIÓN - ✓ Ya hay {len(raw_triplet_files)} archivos de tripletas")
    
    # 4. Quality integrado (normalización + filtros)
    canon_file = pathlib.Path("data") / "canon.jsonl"
    if not canon_file.exists():
        print("4️⃣  CALIDAD INTEGRADA - Normalizando y filtrando...")
        quality()
    else:
        print("4️⃣  CALIDAD INTEGRADA - ✓ Ya hay archivo canónico")
    
    print("\n🎉 PIPELINE COMPLETO COMPLETADO")
    print("=" * 50)
    print("📁 Archivos generados:")
    print(f"   • Datos API limpios: {len(list(pathlib.Path(PATHS.raw).glob('*.json')))} archivos JSON")
    print(f"   • Datos procesados: {len(list(pathlib.Path(PATHS.interim).glob('*.json')))} archivos JSON")
    print(f"   • Tripletas extraídas: {len([f for f in pathlib.Path(PATHS.triples).glob('*.jsonl') if f.name != 'canon.jsonl'])} archivos JSONL")
    
    if canon_file.exists():
        with open(canon_file, "r", encoding="utf-8") as f:
            canon_count = sum(1 for line in f if line.strip())
        print(f"   • Tripletas canónicas: {canon_count:,} en data/canon.jsonl")
    
    print("\n🚀 Para exportar, usa: python -m src.cli export --fmt [csv|parquet|neo4j]")
    """Scrape categories for LATAM countries and save them to a file."""
    all_categories = []

    for country in CONFIG["latam_search"]:
        print(f"Scraping categories for {country}")
        categories = scrape_categories_by_country(country)
        all_categories.extend(categories)
    all_categories = list(set(all_categories))
    
    with open("categories.txt", "w", encoding="utf-8") as f:
        for category in all_categories:
            # Remove the "Categoría:" prefix
            category_name = category.replace("Categoría:", "")
            f.write(f"{category_name}\n")
    print(f"Scraped {len(all_categories)} categories and saved them to categories.txt")

@app.command()
def ingest(source: str = "wiki", limit: int = 300, method: str = "api"): 
    """
    Ingesta contenido desde fuentes externas.
    
    Args:
        source: Fuente de datos (solo 'wiki' por ahora)
        limit: Límite de páginas por categoría
        method: Método de extracción ('api' para Wikipedia API limpia, 'html' para HTML+trafilatura)
    """
    PATHS.raw.mkdir(parents=True, exist_ok=True)
    if source == "wiki":
        with open("categories.txt", "r", encoding="utf-8") as f:
            categories = [line.strip() for line in f.readlines()]
        
        if method == "api":
            from .ingest.wiki_api_clean import scrape_wikipedia_category_clean
            
            print(f"🚀 Iniciando ingesta con Wikipedia API (CONTENIDO LIMPIO)")
            print(f"📚 Procesando {len(categories)} categorías con límite de {limit} páginas c/u")
            print(f"✨ API oficial de Wikipedia → Texto estructurado y limpio")
            
            total_successful = 0
            for category in tqdm(categories, desc="Categorías"):
                successful = scrape_wikipedia_category_clean(category, limit, PATHS.raw, language="es")
                total_successful += successful
                
            print(f"\n🎉 INGESTA COMPLETADA CON API")
            print(f"   • Total páginas exitosas: {total_successful:,}")
            print(f"   • Formato: JSON estructurado (listo para extracción)")
            
        elif method == "html":
            print(f"🚀 Iniciando ingesta masiva con HTML completo")
            print(f"📚 Procesando {len(categories)} categorías con límite de {limit} páginas c/u")
            print(f"🎯 Trafilatura se encargará de extraer TODO el contenido útil después")
            
            for category in tqdm(categories, desc="Categorías"):
                wiki_ingest_category(category, limit, PATHS.raw, method=method)
        else:
            raise typer.BadParameter("Método no soportado. Use 'api' o 'html'")
    else:
        raise typer.BadParameter("Fuente no soportada aún")

@app.command()
def preprocess():
    PATHS.interim.mkdir(parents=True, exist_ok=True)
    
    # Obtener configuración de procesamiento de texto
    text_config = CONFIG.get("text_processing", {})
    use_trafilatura = text_config.get("use_trafilatura", True)
    use_improved_segmentation = text_config.get("use_improved_segmentation", True)
    
    # Procesar archivos HTML (método tradicional)
    html_files = list(pathlib.Path(PATHS.raw).glob("*.html"))
    json_files = list(pathlib.Path(PATHS.raw).glob("*.json"))
    
    total_processed = 0
    
    # Procesar archivos HTML existentes
    for html_file in html_files:
        print(f"Procesando HTML: {html_file.name}...")
        
        # Leer y limpiar HTML
        html_content = html_file.read_text(encoding="utf-8")
        text = html_to_text(html_content, use_trafilatura=use_trafilatura)
        
        if not text or len(text.strip()) < 100:
            print(f"  ⚠️  Advertencia: Texto muy corto o vacío para {html_file.name}")
            continue
            
        # Segmentar texto
        chunks = segment_text(
            text, 
            chunk_chars=CONFIG["chunk_chars"],
            use_improved=use_improved_segmentation
        )
        
        if not chunks:
            print(f"  ⚠️  Advertencia: No se generaron chunks para {html_file.name}")
            continue
            
        print(f"  ✓ Extraído texto de {len(text)} caracteres en {len(chunks)} chunks")
        
        out = {
            "source": html_file.stem, 
            "chunks": chunks,
            "metadata": {
                "original_length": len(html_content),
                "text_length": len(text),
                "num_chunks": len(chunks),
                "processing_method": "trafilatura" if use_trafilatura else "beautifulsoup"
            }
        }
        
        (pathlib.Path(PATHS.interim) / f"{out['source']}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total_processed += 1
    
    # Procesar archivos JSON de Wikipedia API (contenido ya limpio)
    for json_file in json_files:
        print(f"Procesando JSON API: {json_file.name}...")
        
        try:
            # Cargar datos de la API
            api_data = json.loads(json_file.read_text(encoding="utf-8"))
            text = api_data.get('content', '')
            
            if not text or len(text.strip()) < 100:
                print(f"  ⚠️  Advertencia: Contenido muy corto para {json_file.name}")
                continue
            
            # Segmentar texto limpio de la API
            chunks = segment_text(
                text, 
                chunk_chars=CONFIG["chunk_chars"],
                use_improved=use_improved_segmentation
            )
            
            if not chunks:
                print(f"  ⚠️  Advertencia: No se generaron chunks para {json_file.name}")
                continue
                
            print(f"  ✅ Procesado contenido API: {len(text)} caracteres en {len(chunks)} chunks")
            
            out = {
                "source": json_file.stem,
                "chunks": chunks,
                "metadata": {
                    "original_length": len(text),
                    "text_length": len(text),
                    "num_chunks": len(chunks),
                    "processing_method": "wikipedia_api_clean",
                    "title": api_data.get('title', ''),
                    "categories": api_data.get('categories', []),
                    "url": api_data.get('url', '')
                }
            }
            
            (pathlib.Path(PATHS.interim) / f"{out['source']}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_processed += 1
            
        except Exception as e:
            print(f"  ❌ Error procesando {json_file.name}: {e}")
            continue
    
    print(f"\n🎉 PREPROCESSING COMPLETADO")
    print(f"   • Archivos HTML procesados: {len(html_files)}")
    print(f"   • Archivos JSON API procesados: {len(json_files)}")
    print(f"   • Total exitosos: {total_processed}")
    
    if json_files:
        print(f"   ✨ Archivos de API ya tenían contenido limpio y estructurado")

@app.command()
def extract():
    """
    Extrae tripletas usando el wrapper optimizado de mREBEL con sistema de confianza.
    """
    from .extract.mrebel_wrapper_simple import load_mrebel_simple as load_mrebel, run_mrebel_simple as run_mrebel
    
    print("🚀 EXTRACCIÓN DE TRIPLETAS CON SISTEMA DE CONFIANZA")
    print("=" * 60)
    
    # Cargar modelo usando el wrapper optimizado
    print("🔄 Cargando modelo mREBEL optimizado...")
    model, tokenizer = load_mrebel("Babelscape/mrebel-large")
    
    # Asegurar que el directorio de salida exista
    PATHS.triples.mkdir(parents=True, exist_ok=True)
    
    # Obtener archivos a procesar
    interim_files = list(pathlib.Path(PATHS.interim).glob("*.json"))
    print(f"📁 Encontrados {len(interim_files)} archivos para procesar")
    
    total_triplets = 0
    processed_files = 0
    
    # Procesar archivos
    for interim_file in tqdm(interim_files, desc="Procesando archivos"):
        try:
            # Cargar datos del archivo
            data = json.loads(interim_file.read_text(encoding="utf-8"))
            
            # Preparar archivo de salida
            output_path = pathlib.Path(PATHS.triples) / f"{interim_file.stem}.jsonl"
            file_triplets = 0
            
            # Procesar cada chunk
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk_idx, chunk in enumerate(data["chunks"]):
                    if chunk.strip():
                        try:
                            # Extraer tripletas con el wrapper optimizado
                            triplets = run_mrebel(chunk, model, tokenizer)
                            
                            # Guardar cada triplete con metadatos adicionales
                            for triplet in triplets:
                                # Añadir metadatos del archivo fuente
                                triplet_with_meta = {
                                    **triplet,
                                    'source_file': interim_file.stem,
                                    'chunk_index': chunk_idx,
                                    'chunk_text': chunk[:100] + "..." if len(chunk) > 100 else chunk
                                }
                                
                                f.write(json.dumps(triplet_with_meta, ensure_ascii=False) + "\n")
                                file_triplets += 1
                                total_triplets += 1
                                
                        except Exception as e:
                            print(f"   ⚠️  Error procesando chunk {chunk_idx}: {e}")
                            continue
            
            processed_files += 1
            print(f"   ✅ {interim_file.name}: {file_triplets} tripletas extraídas")
            
        except Exception as e:
            print(f"   ❌ Error procesando {interim_file.name}: {e}")
            continue
    
    print(f"\n🎉 EXTRACCIÓN COMPLETADA")
    print(f"   • Archivos procesados: {processed_files}/{len(interim_files)}")
    print(f"   • Total tripletas extraídas: {total_triplets:,}")
    print(f"   • Sistema de confianza: ✅ Activado (0.0-1.0)")
    print(f"\n💡 Las tripletas incluyen campo 'confidence' para filtrado posterior")
    print(f"📁 Archivos guardados en: {PATHS.triples}")
    
    # Limpiar memoria GPU
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Memoria GPU liberada")




@app.command()
def quality(relax: float = 1.0):
    """
    Aplica normalización integrada + filtros de calidad.
    Argumento:
      --relax FLOAT  Factor para relajar reglas (>1 relaja confianza y conectividad).
    """
    from .quality.quality_filter import apply_quality_filters, save_canonical_triplets, load_config
    
    # Verificar tripletas
    triplets_dir = pathlib.Path(PATHS.triples)
    if not triplets_dir.exists() or not list(triplets_dir.glob("*.jsonl")):
        print("❌ No se encontraron archivos de tripletas.")
        print("   Ejecuta primero: python -m src.cli extract")
        return
        
    # Cargar configuración
    config = load_config()
    
    print(f"🎯 INICIANDO CALIDAD (relax={relax})")
    filtered_triplets, stats = apply_quality_filters(triplets_dir, config, relax=relax)
    
    data_dir = pathlib.Path("data"); data_dir.mkdir(exist_ok=True)
    canon_file = data_dir / "canon.jsonl"
    save_canonical_triplets(filtered_triplets, canon_file)
    
    print("\n✅ ESTADÍSTICAS FINALES")
    print(f"  • Tripletas iniciales: {stats.get('initial_count',0):,}")
    print(f"  • Tripletas finales: {stats.get('final_count',0):,}")
    print(f"  • Filtradas por validación: {stats.get('filtered_basic',0):,}")
    print(f"  • Top razones de filtrado: {stats.get('filter_reasons_top',[])[:8]}")
    print(f"  • Entidades normalizadas: {stats.get('entity_mappings',0):,}")
    print(f"  • Duplicados por unicidad (removidos): {stats.get('uniqueness_duplicates',0):,}")
    print(f"  • Entidades eliminadas por conectividad: {stats.get('connectivity',{}).get('removed_entities',0):,}")
    print(f"  • Reducción total: {stats.get('reduction_percent',0):.1f}%")
    print(f"  • Calidad final: {stats.get('quality_percent',0):.1f}%")
    print(f"\n💾 Archivo canónico guardado en: {canon_file}")

@app.command()  
def pipeline():
    """
    Ejecuta todo el pipeline completo: extract → quality (normalización + filtros integrados)
    """
    print("🔄 EJECUTANDO PIPELINE SIMPLIFICADO")
    print("=" * 60)
    
    print("\n📝 PASO 1: Verificando extracción...")
    triplets_dir = pathlib.Path(PATHS.triples)
    raw_triplets = [f for f in triplets_dir.glob("*.jsonl") if f.name not in ["canon.jsonl"]]
    
    if not triplets_dir.exists() or not raw_triplets:
        print("❌ No se encontraron archivos de tripletas extraídas.")
        print("   Ejecuta primero: python -m src.cli extract")
        return
    else:
        print(f"✅ Encontrados {len(raw_triplets)} archivos de tripletas")
        
    print("\n🔧 PASO 2: Aplicando proceso integrado (normalización + calidad)...")
    try:
        quality()
    except Exception as e:
        print(f"❌ Error en proceso de calidad integrado: {e}")
        return
        
    print("\n🎉 ¡PIPELINE SIMPLIFICADO EJECUTADO CON ÉXITO!")
    print("📄 Archivo final: data/canon.jsonl")
    
    # Mostrar estadísticas finales
    canon_path = pathlib.Path("data") / "canon.jsonl"
    if canon_path.exists():
        with open(canon_path, "r", encoding="utf-8") as f:
            canon_count = sum(1 for line in f if line.strip())
        print(f"✨ Total tripletas canónicas: {canon_count:,}")
    
    print(f"\n� Para exportar, usa: python -m src.cli export --fmt [csv|parquet|neo4j]")

@app.command()
def export(fmt: str = "csv"):
    """
    Exporta las tripletas canónicas a diferentes formatos.
    """
    canon_path = pathlib.Path("data")/"canon.jsonl"
    
    if not canon_path.exists():
        print("❌ No se encontró el archivo canon.jsonl")
        print("   Ejecuta primero: python -m src.cli pipeline")
        print("   O por separado: extract → normalize → quality")
        return
        
    if fmt == "csv":
        export_csv(canon_path, pathlib.Path(CONFIG["paths"]["exports"]))
    elif fmt == "parquet":
        export_parquet(canon_path, pathlib.Path(CONFIG["paths"]["exports"]))
    elif fmt == "neo4j":
        export_neo4j(canon_path, pathlib.Path(CONFIG["paths"]["exports"]))
    else:
        raise typer.BadParameter("Formato no soportado")

@app.command(name="filter")
def filter_triplets(input: str, output: str, config: str = None, relax: float = 1.0, explain: bool = False):
	"""
	Filtra un archivo JSONL de tripletas con trazabilidad.
	Ejemplo:
	  python -m src.cli filter data/triples/myfile.jsonl data/filtered.jsonl --relax 2.0 --explain
	"""
	from quality.quality_filter import filter_file
	filter_file(input, output, config_path=config, relax=relax, explain=explain)

if __name__ == "__main__":
    app()

import argparse
from pathlib import Path
from quality.quality_filter import filter_file

def main():
	parser = argparse.ArgumentParser(description="Filtrar tripletas con trazabilidad y opción de relajar reglas.")
	parser.add_argument("input", help="Archivo JSONL de entrada")
	parser.add_argument("output", help="Archivo JSONL de salida")
	parser.add_argument("--config", "-c", help="Ruta a config YAML (opcional)", default=None)
	parser.add_argument("--relax", "-r", type=float, default=1.0, help="Factor de relajación (>1 relaja filtros)")
	parser.add_argument("--explain", action="store_true", help="Imprime ejemplos de por qué se descartaron tripletas")
	args = parser.parse_args()
	
	inp = Path(args.input)
	outp = Path(args.output)
	if not inp.exists():
		print("Archivo de entrada no existe:", inp)
		return
	
	filter_file(str(inp), str(outp), config_path=args.config, relax=args.relax, explain=args.explain)

if __name__ == "__main__":
	main()
