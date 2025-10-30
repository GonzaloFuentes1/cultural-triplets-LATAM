import re
from typing import List

def segment_text_simple(text: str, chunk_chars: int = 1800) -> List[str]:
    """
    Segmentación simple original como fallback.
    """
    chunks, current = [], []
    for sent in text.replace("?!",".").replace("?",".").split('.'):
        s = (sent.strip()+'.').strip()
        if not s or s == '.':
            continue
        if sum(len(x) for x in current)+len(s) <= chunk_chars:
            current.append(s)
        else:
            chunks.append(' '.join(current)); current=[s]
    if current:
        chunks.append(' '.join(current))
    return chunks

def segment_text_improved(text: str, chunk_chars: int = 1800) -> List[str]:
    """
    Segmentación mejorada que maneja mejor el español y considera más patrones de puntuación.
    """
    # Normalizar el texto
    text = text.strip()
    if not text:
        return []
    
    # Patrones de fin de oración en español
    sentence_endings = r'[.!?](?:\s|$)'
    
    # Dividir en oraciones respetando abreviaciones comunes en español
    abbreviations = r'\b(?:Sr|Sra|Dr|Dra|Prof|Lic|Ing|Arq|etc|vs|p\.ej|cf|art|núm|pág|vol|ed|coord|dir)\.'
    
    # Reemplazar abreviaciones temporalmente para evitar división incorrecta
    temp_text = re.sub(abbreviations, lambda m: m.group().replace('.', '<<<DOT>>>'), text, flags=re.IGNORECASE)
    
    # Encontrar posiciones de fin de oración
    sentences = []
    last_pos = 0
    
    for match in re.finditer(sentence_endings, temp_text):
        end_pos = match.end()
        sentence = temp_text[last_pos:end_pos].strip()
        if sentence:
            # Restaurar puntos en abreviaciones
            sentence = sentence.replace('<<<DOT>>>', '.')
            sentences.append(sentence)
        last_pos = end_pos
    
    # Agregar el texto restante si existe
    if last_pos < len(temp_text):
        remaining = temp_text[last_pos:].strip()
        if remaining:
            remaining = remaining.replace('<<<DOT>>>', '.')
            sentences.append(remaining)
    
    # Si no se encontraron oraciones, usar división por párrafos
    if not sentences:
        sentences = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Agrupar oraciones en chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # Si una sola oración es muy larga, dividirla
        if sentence_length > chunk_chars:
            # Guardar chunk actual si existe
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Dividir oración larga por comas o punto y coma
            sub_parts = re.split(r'[,;]', sentence)
            temp_part = []
            temp_length = 0
            
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                    
                part_length = len(part)
                if temp_length + part_length <= chunk_chars:
                    temp_part.append(part)
                    temp_length += part_length
                else:
                    if temp_part:
                        chunks.append(' '.join(temp_part))
                    temp_part = [part]
                    temp_length = part_length
            
            if temp_part:
                chunks.append(' '.join(temp_part))
        
        # Verificar si la oración cabe en el chunk actual
        elif current_length + sentence_length <= chunk_chars:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Guardar chunk actual y empezar uno nuevo
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    # Agregar el último chunk si existe
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Filtrar chunks muy pequeños (menos de 50 caracteres)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 50]
    
    return chunks

def segment_text(text: str, chunk_chars: int = 1800, use_improved: bool = True) -> List[str]:
    """
    Función principal de segmentación de texto.
    
    Args:
        text: Texto a segmentar
        chunk_chars: Tamaño máximo de cada chunk en caracteres
        use_improved: Si usar la segmentación mejorada (True) o simple (False)
    
    Returns:
        Lista de chunks de texto
    """
    if use_improved:
        try:
            return segment_text_improved(text, chunk_chars)
        except Exception:
            # Fallback al método simple si hay algún error
            return segment_text_simple(text, chunk_chars)
    else:
        return segment_text_simple(text, chunk_chars)
