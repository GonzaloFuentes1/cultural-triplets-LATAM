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
    r = s.get(API.format(lang=lang), params={
        "action":"parse","page":title,"prop":"text","format":"json"
    })
    return r.json()["parse"]["text"]["*"]

def wiki_ingest_category(category: str, limit: int, out_dir: str|Path, lang: str = "es"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({'User-Agent': 'GeminiCodeHelper/1.0 (https://gemini.google.com/)'})
    for t in page_titles_from_category(s, category, lang=lang, limit=limit):
        html = fetch_page_html(s, t, lang=lang)
        filename = sanitize_filename(t)
        (out / f"{filename}.html").write_text(html, encoding="utf-8")
