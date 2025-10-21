import typer, yaml, json, pathlib
from .utils.io import Paths
from .ingest.wiki_api import wiki_ingest_category
from .preprocess.clean_text import html_to_text
from .preprocess.segment import segment_text
from .extract.mrebel_wrapper import run_mrebel
from .ingest.category_scraper import scrape_categories_by_country
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .export.export_csv import export_csv
from .export.export_parquet import export_parquet
from .export.export_neo4j_tsv import export_neo4j
from tqdm import tqdm

app = typer.Typer()

CONFIG = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
PATHS = Paths.from_config(CONFIG["paths"])

@app.command(name="scrape-categories")
def scrape_categories():
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
def ingest(source: str = "wiki", limit: int = 300): # Changed limit to 300
    PATHS.raw.mkdir(parents=True, exist_ok=True)
    if source == "wiki":
        with open("categories.txt", "r", encoding="utf-8") as f:
            categories = [line.strip() for line in f.readlines()]
        
        for category in tqdm(categories):
            wiki_ingest_category(category, limit, PATHS.raw)
    else:
        raise typer.BadParameter("Fuente no soportada aún")

@app.command()
def preprocess():
    PATHS.interim.mkdir(parents=True, exist_ok=True)
    for html_file in pathlib.Path(PATHS.raw).glob("*.html"):
        text = html_to_text(html_file.read_text(encoding="utf-8"))
        chunks = segment_text(text, chunk_chars=CONFIG["chunk_chars"])
        out = {"source": html_file.stem, "chunks": chunks}
        (pathlib.Path(PATHS.interim) / f"{out['source']}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )

@app.command()
def extract():
    # Carga el modelo y el tokenizador de mREBEL
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", src_lang="es_XX", tgt_lang="tp_XX")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
    
    # Asegúrate de que el directorio de salida para las tripletas exista
    PATHS.triples.mkdir(parents=True, exist_ok=True)

    # Itera sobre los archivos JSON preprocesados en el directorio intermedio
    for interim_file in tqdm(list(pathlib.Path(PATHS.interim).glob("*.json"))):
        # Carga los datos del archivo JSON
        data = json.loads(interim_file.read_text(encoding="utf-8"))
        
        # Prepara la ruta para el archivo de salida de las tripletas
        output_path = pathlib.Path(PATHS.triples) / f"{interim_file.stem}.jsonl"
        
        # Procesa cada chunk de texto para extraer tripletas
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in data["chunks"]:
                # Solo procesa chunks que no estén vacíos
                if chunk.strip():
                    # Extrae las tripletas usando el modelo mREBEL
                    triplets = run_mrebel(chunk, model, tokenizer)
                    
                    # Escribe cada tripleta como una línea JSON en el archivo de salida
                    for triplet in triplets:
                        f.write(json.dumps(triplet, ensure_ascii=False) + "\n")


@app.command()
def export(fmt: str = "csv"):
    canon_path = pathlib.Path(PATHS.triples)/"canon.jsonl"
    if fmt == "csv":
        export_csv(canon_path, pathlib.Path(CONFIG["paths"]["exports"]))
    elif fmt == "parquet":
        export_parquet(canon_path, pathlib.Path(CONFIG["paths"]["exports"]))
    elif fmt == "neo4j":
        export_neo4j(canon_path, pathlib.Path(CONFIG["paths"]["exports"]))
    else:
        raise typer.BadParameter("Formato no soportado")

if __name__ == "__main__":
    app()
