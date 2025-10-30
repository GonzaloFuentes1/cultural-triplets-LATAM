import re
import trafilatura
from trafilatura.settings import use_config

def clean_wikipedia_text(text: str) -> str:
    """
    Limpieza AGRESIVA específica para contenido de Wikipedia extraído con trafilatura.
    Remueve formato HTML/Wiki residual y deja texto limpio para NLP.
    """
    # 1. Remover referencias numéricas [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # 2. Remover referencias problemáticas específicas
    text = re.sub(r'\[cita requerida\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sin referencias\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[¿cuándo\?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[¿cuál\?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[¿dónde\?\]', '', text, flags=re.IGNORECASE)
    
    # 3. Limpiar enlaces wiki-style [texto](/wiki/enlace) -> texto
    text = re.sub(r'\[([^\]]+)\]\(/wiki/[^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Enlaces generales
    
    # 4. Remover formato de tabla HTML residual
    text = re.sub(r'\|+.*?\|+', ' ', text)  # Líneas con pipes |
    text = re.sub(r'\|---\|', ' ', text)    # Separadores de tabla
    text = re.sub(r'\|\s*\|', ' ', text)    # Pipes vacíos
    
    # 5. Limpiar caracteres de formato HTML residual
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    
    # 6. Remover líneas que son obviamente navegación/estructura
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Saltar líneas muy cortas o vacías
        if len(line) < 3:
            continue
            
        # Saltar líneas que son solo símbolos/formato
        if re.match(r'^[|\-=+\s]*$', line):
            continue
            
        # Saltar líneas de navegación típicas
        skip_patterns = [
            r'^(véase también|enlaces externos|referencias|categorías?):?\s*$',
            r'^tabla de contenidos?:?\s*$',
            r'^menú de navegación:?\s*$',
            r'^\s*\|\s*$',  # Líneas solo con pipes
            r'^\s*\d+\s*$',  # Líneas solo con números
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue
            
        cleaned_lines.append(line)
    
    # 7. Reunir líneas y limpiar espacios
    text = ' '.join(cleaned_lines)
    
    # 8. Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # 9. Limpiar caracteres problemáticos pero mantener acentos españoles
    # Mantener: letras, números, espacios, puntuación básica, acentos
    text = re.sub(r'[^\w\s.,;:!?¿¡()"\'-áéíóúüñÁÉÍÓÚÜÑ]', ' ', text)
    
    # 10. Limpieza final de espacios
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def html_to_text(html: str, use_trafilatura: bool = True) -> str:
    """
    Extrae TODO el texto útil de HTML usando trafilatura configurado agresivamente.
    Trafilatura va a sacar todo el contenido principal sin filtros excesivos.
    """
    # Configuración agresiva para trafilatura - queremos TODO
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "60")  # Más tiempo
    config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "50")   # Umbral muy bajo 
    config.set("DEFAULT", "MAX_EXTRACTED_SIZE", "2000000")  # Sin límite práctico
    
    # Configuración agresiva - extraer TODO lo posible
    extracted_text = trafilatura.extract(
        html,
        config=config,
        include_comments=False,     # No queremos comentários HTML
        include_tables=False,       # NO tablas - traen mucho ruido
        include_formatting=False,   # NO formato - traen pipes y basura
        include_links=False,        # NO enlaces - traen formato wiki
        include_images=False,       # No necesitamos imágenes
        deduplicate=True,          # Eliminar duplicados
        favor_precision=True,      # PRECISION > RECALL para texto más limpio
        favor_recall=False,        # No favor recall - preferir calidad
        with_metadata=False
    )
    
    if extracted_text and len(extracted_text.strip()) > 20:
        # Limpieza agresiva para texto súper limpio
        text = clean_wikipedia_text(extracted_text)
        return text
    else:
        # Si trafilatura falla completamente, al menos intentar extraer algo básico
        print("⚠️  Trafilatura no extrajo contenido suficiente")
        return ""
