import time, requests, re
from pathlib import Path

API = "https://{lang}.wikipedia.org/w/api.php"

def sanitize_filename(filename: str) -> str:
    """Sanitiza un nombre de archivo removiendo o reemplazando caracteres inválidos."""
    # Caracteres inválidos en Windows: < > : " | ? * \
    # También reemplazamos / que ya se estaba manejando
    invalid_chars = r'[<>:"|?*\\/]'
    # Reemplazar caracteres inválidos con guiones bajos
    sanitized = re.sub(invalid_chars, '_', filename)
    # Remover espacios al inicio y final
    sanitized = sanitized.strip()
    # Remover puntos al final (no válidos en Windows)
    sanitized = sanitized.rstrip('.')
    # Asegurar que no esté vacío
    if not sanitized:
        sanitized = "unnamed_file"
    return sanitized

def page_titles_from_category(s: requests.Session, category: str, lang: str = "es", limit: int = 200):
    params = {"action":"query","list":"categorymembers","cmtitle":f"Category:{category}",
              "format":"json","cmlimit":50}
    out, cmcontinue = [], None
    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = s.get(API.format(lang=lang), params=params).json()
        out += [m["title"] for m in r["query"]["categorymembers"]]
        cmcontinue = r.get("continue", {}).get("cmcontinue")
        if not cmcontinue or len(out) >= limit:
            break
        time.sleep(0.2)
    return out[:limit]

def fetch_page_html(s: requests.Session, title: str, lang: str = "es") -> str:
    """Obtiene el HTML renderizado de una página de Wikipedia"""
    r = s.get(API.format(lang=lang), params={
        "action":"parse","page":title,"prop":"text","format":"json"
    })
    return r.json()["parse"]["text"]["*"]

def fetch_page_extract(s: requests.Session, title: str, lang: str = "es") -> str:
    """Obtiene el texto limpio usando extracts API (mejor para trafilatura)"""
    r = s.get(API.format(lang=lang), params={
        "action":"query",
        "format":"json",
        "titles":title,
        "prop":"extracts",
        "exintro":False,  # Incluir toda la página, no solo intro
        "explaintext":False,  # Mantener algo de HTML para trafilatura
        "exsectionformat":"wiki"
    })
    pages = r.json()["query"]["pages"]
    for page_id, page_data in pages.items():
        if page_id != "-1" and "extract" in page_data:
            return page_data["extract"]
    return ""

def fetch_page_wikitext(s: requests.Session, title: str, lang: str = "es") -> str:
    """Obtiene el wikitext fuente de la página"""
    r = s.get(API.format(lang=lang), params={
        "action":"query",
        "format":"json",
        "titles":title,
        "prop":"revisions",
        "rvprop":"content",
        "rvslots":"main"
    })
    pages = r.json()["query"]["pages"]
    for page_id, page_data in pages.items():
        if page_id != "-1" and "revisions" in page_data:
            return page_data["revisions"][0]["slots"]["main"]["*"]
    return ""

def wiki_ingest_category(category: str, limit: int, out_dir: str|Path, lang: str = "es", method: str = "html"):
    """
    Ingesta páginas de Wikipedia desde una categoría.
    Descarga HTML completo para que trafilatura se encargue de todo.
    
    Args:
        category: Nombre de la categoría
        limit: Límite de páginas a descargar
        out_dir: Directorio de salida
        lang: Idioma de Wikipedia
        method: Método de extracción (siempre 'html' para máximo contenido)
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({'User-Agent': 'CulturalTripletsLATAM/1.0 (Wikipedia content extraction)'})
    
    page_titles = page_titles_from_category(s, category, lang=lang, limit=limit)
    print(f"  📥 Descargando {len(page_titles)} páginas de la categoría '{category}'...")
    
    for i, title in enumerate(page_titles, 1):
        print(f"    [{i:3d}/{len(page_titles)}] {title}")
        
        # Siempre usar HTML completo - que trafilatura se encargue después
        content = fetch_page_html(s, title, lang=lang)
        
        if not content or len(content.strip()) < 50:  # Umbral muy bajo
            print(f"      ⚠️  Sin contenido, saltando...")
            continue
            
        filename = sanitize_filename(title)
        output_file = out / f"{filename}.html"
        output_file.write_text(content, encoding="utf-8")
        print(f"      ✓ Guardado: {len(content):,} caracteres")
        
        # Pequeña pausa para ser respetuosos con la API
        time.sleep(0.05)  # Más rápido
