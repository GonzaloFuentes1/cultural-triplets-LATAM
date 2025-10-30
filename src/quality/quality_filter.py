"""
Módulo compacto de control de calidad para tripletas.
Versión optimizada con todas las funcionalidades en código más conciso.
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Any, Union
from pathlib import Path
import unicodedata
import difflib
from tqdm import tqdm
from functools import partial
import yaml
import multiprocessing as mp
import statistics

def _compute_similarity(e1_norm: str, e2_norm: str) -> float:
	"""
	Similitud combinada:
	  - Si iguales tras normalizar => 1.0
	  - Token Jaccard (palabras) + SequenceMatcher ratio
	  - Ponderación: 0.6 * jaccard + 0.4 * seq_ratio
	"""
	if not e1_norm or not e2_norm:
		return 0.0
	if e1_norm == e2_norm:
		return 1.0
	# tokens
	t1 = set([w for w in re.split(r'\W+', e1_norm) if w])
	t2 = set([w for w in re.split(r'\W+', e2_norm) if w])
	union = t1 | t2
	inter = t1 & t2
	jaccard = (len(inter) / len(union)) if union else 0.0
	seq = difflib.SequenceMatcher(None, e1_norm, e2_norm).ratio()
	return 0.6 * jaccard + 0.4 * seq

def _process_cluster_by_first_letter(args):
	"""
	Procesa un cluster de entidades con mejoras:
	  - Si cluster grande (> neighbor_window * 4) se compara cada entidad solo contra sus N vecinos
	  - Usa _compute_similarity para decidir merges
	Args esperado: (cluster, threshold, neighbor_window)
	"""
	cluster, threshold, neighbor_window = args
	if len(cluster) < 2:
		return {}
	mapping = {}
	processed = set()
	# pre-normalizar cluster para evitar recalcular
	# usar normalize_entity desde el módulo (asume disponible)
	# si normalize_entity no está en el scope, se usa identity
	try:
		normalizer = normalize_entity
	except NameError:
		def normalizer(x, cfg=None): return x.lower()
	# crear lista de tuplas (original, normalized)
	norm_list = [(ent, normalizer(ent, {})) for ent in cluster]
	# ordenar para ventana vecina (alfabético)
	norm_list.sort(key=lambda x: (len(x[1]), x[1]))
	N = len(norm_list)
	use_window = N > (neighbor_window * 4)
	for i, (entity1, norm1) in enumerate(norm_list):
		if entity1 in processed:
			continue
		similar = [entity1]
		# elegir candidatos: vecinos si cluster grande, sino todo resto
		if use_window:
			# crear ventana alrededor de i
			start = max(0, i - neighbor_window)
			end = min(N, i + neighbor_window + 1)
			candidates = norm_list[start:end]
		else:
			candidates = norm_list[i+1:]
		for entity2, norm2 in candidates:
			if entity2 in processed or entity2 == entity1:
				continue
			sim = _compute_similarity(norm1, norm2)
			if sim >= threshold:
				similar.append(entity2)
				processed.add(entity2)
		if len(similar) > 1:
			representative = _choose_best_representative(similar)
			for ent in similar:
				if ent != representative:
					mapping[ent] = representative
		processed.add(entity1)
	return mapping

def _choose_best_representative(entities: List[str]) -> str:
    """
    Elige el mejor representante usando criterios inteligentes:
    1. Mayúsculas correctas (Buenos Aires > buenos aires)
    2. Completitud (Buenos Aires > Bs. As.)
    3. Frecuencia de uso común
    4. Longitud apropiada (ni muy corto ni muy largo)
    """
    if len(entities) == 1:
        return entities[0]
    
    scored_entities = []
    
    for entity in entities:
        score = 0
        
        # 1. Penalizar todo en minúsculas (excepto siglas)
        if entity.islower() and len(entity) > 3:
            score -= 20
        
        # 2. Premiar capitalización correcta
        if entity[0].isupper() and not entity.isupper():
            score += 15
        
        # 3. Penalizar abreviaciones (contienen puntos)
        if '.' in entity:
            score -= 10
        
        # 4. Penalizar nombres muy cortos (probable abreviación)
        if len(entity) < 3:
            score -= 15
        elif len(entity) < 5:
            score -= 5
        
        # 5. Premiar longitud moderada (más información)
        if 5 <= len(entity) <= 50:
            score += 10
        elif len(entity) > 50:
            score -= 5  # Muy largo, posible ruido
        
        # 6. Penalizar caracteres especiales excesivos
        special_chars = sum(1 for c in entity if not c.isalnum() and c not in ' -áéíóúüñÁÉÍÓÚÜÑ')
        score -= special_chars * 2
        
        # 7. Premiar nombres que parecen formales/oficiales
        if any(word in entity.lower() for word in ['república', 'estado', 'ciudad', 'provincia', 'departamento']):
            score += 8
        
        # 8. Penalizar entidades que parecen fragmentos
        if entity.startswith(('de ', 'del ', 'la ', 'el ', 'los ', 'las ')):
            score -= 10
        
        scored_entities.append((score, -len(entity), entity.lower(), entity))  # Desempatar por longitud y alfabético
    
    # Elegir el de mayor puntaje
    best = max(scored_entities, key=lambda x: x[0])
    return best[3]

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Carga la configuración desde el archivo YAML."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_entity(entity: str, config: Dict[str, Any]) -> str:
    """Normaliza una entidad según la configuración."""
    norm_config = config.get("normalization", {}).get("entity_normalization", {})
    
    if not norm_config.get("enabled", True):
        return entity
        
    entity = entity.strip()
    
    # SIEMPRE aplicar normalización básica
    # 1. Convertir a minúsculas
    entity = entity.lower()
    
    # 2. Remover acentos/tildes
    entity = ''.join(c for c in unicodedata.normalize('NFD', entity) if unicodedata.category(c) != 'Mn')
    
    # 3. Remover stop words (determinantes, preposiciones, etc.)
    stop_words = [
        'el', 'la', 'los', 'las',           # artículos definidos
        'un', 'una', 'unos', 'unas',        # artículos indefinidos
        'de', 'del', 'de la', 'de los', 'de las',  # preposiciones
        'en', 'con', 'por', 'para', 'sin',  # más preposiciones
        'y', 'e', 'o', 'u',                 # conjunciones
        'que', 'como', 'pero', 'si',        # conectores
    ]
    
    # Dividir en palabras y filtrar stop words
    words = entity.split()
    filtered_words = []
    
    for word in words:
        # Mantener la palabra si no es stop word O si es la única palabra
        if word not in stop_words or len(words) == 1:
            filtered_words.append(word)
    
    # Si después de filtrar no queda nada, usar la entidad original normalizada
    if not filtered_words:
        return entity
    
    return ' '.join(filtered_words)

def is_valid_triplet(triplet: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Validación compacta de tripleta usando todas las reglas configuradas."""
    quality_config = config.get("quality_control", {})
    validation_patterns = quality_config.get("validation_patterns", {})
    
    head, relation, tail = triplet["head"], triplet["type"], triplet["tail"]
    head_type, tail_type = triplet.get("head_type", ""), triplet.get("tail_type", "")
    confidence = triplet.get("confidence", 1.0)
    
    # 1. Filtro de confianza (solo si existe el campo confidence)
    min_confidence = quality_config.get("min_confidence_threshold", 0.5)
    if "confidence" in triplet and confidence < min_confidence:
        return False
    
    # 2. Filtros básicos de longitud y contenido
    if (len(head) < quality_config.get("min_entity_length", 2) or 
        len(head) > quality_config.get("max_entity_length", 100) or
        len(tail) < quality_config.get("min_entity_length", 2) or 
        len(tail) > quality_config.get("max_entity_length", 100)):
        return False
    
    # 3. Entidades genéricas
    generic_entities = quality_config.get("generic_entities", [])
    if head.lower() in generic_entities or tail.lower() in generic_entities:
        return False
    
    # 4. Relaciones filtradas
    filtered_relations = quality_config.get("filtered_relations", [])
    if relation.lower() in filtered_relations:
        return False
    
    # 5. Tipos incompatibles (solo si ambos tipos están definidos)
    incompatible_pairs = quality_config.get("incompatible_type_pairs", [])
    if head_type and tail_type:
        for pair in incompatible_pairs:
            if len(pair) >= 2:
                # Verificar reglas simples de tipo (ej: ["concept", "concept"])
                if len(pair) == 2 and head_type == pair[0] and tail_type == pair[1]:
                    return False
                
                # Verificar reglas de autoreferencia (ej: ["loc", "loc", "contains", "self"])
                if (len(pair) >= 4 and pair[3] == "self" and 
                    head_type == pair[0] and tail_type == pair[1] and
                    head.lower().strip() == tail.lower().strip()):  # Solo rechazar si es la MISMA entidad
                    return False
    
    # 6. Validaciones semánticas específicas (solo si tipos están definidos)
    semantic_validation = validation_patterns.get("semantic_validation", {})
    relation_compatibility = semantic_validation.get("relation_entity_compatibility", {})
    
    if relation in relation_compatibility and head_type and tail_type:
        allowed_types = relation_compatibility[relation]
        if head_type not in allowed_types and tail_type not in allowed_types:
            return False
    
    # 7. Validación de ganador único para eventos
    event_consistency = validation_patterns.get("event_consistency", {})
    if (event_consistency.get("unique_winner_per_event", False) and 
        head_type == "eve" and relation == "winner"):
        # Esta validación se hace a nivel de dataset, no de tripleta individual
        pass
    
    # 8. Validación temporal básica
    temporal_consistency = validation_patterns.get("temporal_consistency", {})
    if temporal_consistency.get("century_validation", False) and tail_type == "date":
        # Verificar fechas razonables (no antes de 1500, no después de 2100)
        if re.search(r'\b(14\d{2}|13\d{2}|12\d{2})\b', tail) or re.search(r'\b(21[1-9]\d|22\d{2})\b', tail):
            return False
    
    # 9. Validación de género en ocupaciones
    gender_consistency = validation_patterns.get("gender_consistency", {})
    if gender_consistency and relation == "occupation":
        masculine_occupations = gender_consistency.get("masculine_occupations", [])
        feminine_occupations = gender_consistency.get("feminine_occupations", [])
        
        # Heurística simple: nombres terminados en 'o' son masculinos
        if (head.endswith('o') and tail in feminine_occupations) or \
           (head.endswith('a') and tail in masculine_occupations):
            return False
    
    return True

def find_similar_entities_optimized(entities: List[str], config: Dict[str, Any]) -> Dict[str, str]:
    """Encuentra entidades similares usando normalización básica + fuzzy matching paralelizado por primera letra."""
    norm_config = config.get("normalization", {}).get("entity_normalization", {})
    if not norm_config.get("enabled", True):
        return {}
    
    print("Iniciando normalización básica y detección de duplicados (fuzzy paralelizado)")

    # Umbral para fuzzy matching (usar la clave definida en config)
    fuzzy_threshold = norm_config.get("similarity_threshold", 0.85)
    neighbor_window = norm_config.get("neighbor_window", 50)  # nuevo parámetro: cuántos vecinos comparar
    large_cluster_hint = norm_config.get("large_cluster_threshold", 400)  # info de debug

    # 1. Normalizar todas las entidades
    print("Paso 1: normalizando entidades")
    
    normalized_entities = {}
    for entity in entities:
        normalized = normalize_entity(entity, config)
        normalized_entities[entity] = normalized
    
    # 2. Buscar duplicados exactos después de normalización
    print("Paso 2: detectando duplicados exactos")
    
    normalized_to_originals = defaultdict(list)
    for original, normalized in normalized_entities.items():
        normalized_to_originals[normalized].append(original)
    
    exact_mappings = {}
    exact_duplicates = 0
    
    for normalized, originals in normalized_to_originals.items():
        if len(originals) > 1:
            representative = _choose_best_representative(originals)
            for original in originals:
                if original != representative:
                    exact_mappings[original] = representative
                    exact_duplicates += 1
    
    print(f"Duplicados exactos encontrados: {exact_duplicates}")
    
    # 3. Agrupar entidades restantes por primera letra
    print("Paso 3: agrupando por primera letra para paralelización")
    
    remaining_entities = [entity for entity in entities if entity not in exact_mappings]
    
    # Agrupar por primera letra (normalizada)
    clusters_by_letter = defaultdict(list)
    for entity in remaining_entities:
        first_letter = normalize_entity(entity, config)[0] if normalize_entity(entity, config) else 'z'
        clusters_by_letter[first_letter].append(entity)
    
    print(f"Creados {len(clusters_by_letter)} clusters por primera letra")
    
    # 4. Paralelizar fuzzy matching por cluster
    print("Paso 4: aplicando fuzzy matching paralelizado")
    
    # Preparar argumentos para paralelización
    cluster_args = [(cluster, fuzzy_threshold, neighbor_window) for cluster in clusters_by_letter.values() if len(cluster) > 1]
    
    # Configurar número de procesos
    max_processes = config.get("parallelization", {}).get("max_processes", mp.cpu_count())
    num_processes = min(max_processes, len(cluster_args), mp.cpu_count())
    
    fuzzy_mappings = {}
    
    if cluster_args and num_processes > 1:
        print(f"Usando {num_processes} procesos para {len(cluster_args)} clusters")
        
        with mp.Pool(processes=num_processes) as pool:
            cluster_results = list(tqdm(
                pool.imap(_process_cluster_by_first_letter, cluster_args),
                total=len(cluster_args),
                desc="clusters"
            ))
        
        # Combinar resultados de todos los clusters
        for cluster_mapping in cluster_results:
            fuzzy_mappings.update(cluster_mapping)
    else:
        print("Procesamiento secuencial (pocos clusters o proceso único)")
        for cluster_arg in tqdm(cluster_args, desc="clusters"):
            cluster_mapping = _process_cluster_by_first_letter(cluster_arg)
            fuzzy_mappings.update(cluster_mapping)
    
    fuzzy_duplicates = len(fuzzy_mappings)
    print(f"Duplicados fuzzy encontrados: {fuzzy_duplicates}")

    # 5. Combinar todos los mapeos
    all_mappings = {**exact_mappings, **fuzzy_mappings}

    # Resumen compacto
    print(f"Entidades originales: {len(entities)}, mapeos: {len(all_mappings)}")
    # mostrar hasta 10 ejemplos de mapeo
    examples = list(all_mappings.items())[:10]
    if examples:
        print("Ejemplos de mapeos fuzzy (original -> representante):")
        for orig, rep in examples:
            print(f"  {orig} -> {rep}")
    return all_mappings

def explain_triplet(triplet: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    """Devuelve lista de razones por las que una tripleta sería rechazada (vacía => válida)."""
    reasons = []
    quality_config = config.get("quality_control", {})
    validation_patterns = quality_config.get("validation_patterns", {})
    
    head = (triplet.get("head") or "").strip()
    tail = (triplet.get("tail") or "").strip()
    relation = (triplet.get("type") or "").strip()
    head_type = triplet.get("head_type", "")
    tail_type = triplet.get("tail_type", "")
    confidence = triplet.get("confidence", None)
    
    min_conf = quality_config.get("min_confidence_threshold", 0.4)
    if confidence is not None and confidence < min_conf:
        reasons.append("low_confidence")
    
    min_len = quality_config.get("min_entity_length", 2)
    max_len = quality_config.get("max_entity_length", 100)
    if not head:
        reasons.append("head_empty")
    else:
        if len(head) < min_len:
            reasons.append("head_too_short")
        if len(head) > max_len:
            reasons.append("head_too_long")
    if not tail:
        reasons.append("tail_empty")
    else:
        if len(tail) < min_len:
            reasons.append("tail_too_short")
        if len(tail) > max_len:
            reasons.append("tail_too_long")
    
    generic = set(map(str.lower, quality_config.get("generic_entities", [])))
    if head.lower() in generic or tail.lower() in generic:
        reasons.append("generic_entity")
    
    filtered_rel = set(map(str.lower, quality_config.get("filtered_relations", [])))
    if relation and relation.lower() in filtered_rel:
        reasons.append("filtered_relation")
    
    # tipos incompatibles básicos (mirar pairs en config)
    incompatible_pairs = quality_config.get("incompatible_type_pairs", [])
    if head_type and tail_type:
        for pair in incompatible_pairs:
            if len(pair) >= 2 and head_type == pair[0] and tail_type == pair[1]:
                # si hay marca 'self' y mismo head/tail
                if len(pair) >= 4 and pair[3] == "self" and head.lower().strip() == tail.lower().strip():
                    reasons.append("self_incompatible_type")
                elif len(pair) == 2:
                    reasons.append("incompatible_types")
    
    # validación semántica simple
    semantic = validation_patterns.get("semantic_validation", {}).get("relation_entity_compatibility", {})
    if relation in semantic and head_type and tail_type:
        allowed = semantic[relation]
        if head_type not in allowed and tail_type not in allowed:
            reasons.append("semantic_incompatibility")
    
    # NUEVA VALIDACIÓN: Filtrar entidades temporales mal clasificadas
    temporal_consistency = validation_patterns.get("temporal_consistency", {})
    if temporal_consistency.get("year_as_event_filter", False):
        # Detectar años clasificados incorrectamente como eventos
        invalid_patterns = temporal_consistency.get("invalid_year_patterns", [])
        incompatible_relations = temporal_consistency.get("temporal_incompatible_relations", [])
        
        # Verificar si head parece un año/fecha pero está clasificado como evento
        if head_type == "eve" and relation in incompatible_relations:
            for pattern in invalid_patterns:
                if re.match(pattern, head):
                    reasons.append("year_classified_as_event")
                    break
        
        # Verificar si head es explícitamente temporal pero tiene relaciones de evento
        if head_type in ["time", "date", "num"] and relation in incompatible_relations:
            reasons.append("temporal_entity_invalid_relation")
        
        # Verificar conceptos temporales básicos
        if (head_type == "concept" and relation in incompatible_relations and 
            any(temporal_word in head.lower() for temporal_word in 
                ["año", "mes", "día", "fecha", "tiempo", "época", "periodo", "temporada"])):
            reasons.append("temporal_concept_invalid_relation")
    
    # Filtrar casos específicos como "2019 winner miguel indurain"
    if (head_type in ["eve", "time", "date", "num", "concept"] and 
        relation in ["winner", "participant", "performer", "competitor", "champion"] and
        re.match(r'^\d{4}$', head)):  # Años de 4 dígitos
        reasons.append("pure_year_invalid_event")
    
    return sorted(set(reasons))

def apply_uniqueness_rules(triplets: List[Dict], config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
    """Aplica reglas de unicidad de forma simple: solo sustituye si mejora significativamente la confianza."""
    margin = config.get("normalization", {}).get("uniqueness_confidence_margin", 0.05)
    kept = {}
    removed = 0
    
    for t in triplets:
        # Validación defensiva: asegurar que t es un diccionario
        if not isinstance(t, dict):
            continue
            
        key = (t.get("head"), t.get("type"))
        if key not in kept:
            kept[key] = t
        else:
            existing = kept[key]
            # Validación defensiva: asegurar que existing es un diccionario
            if not isinstance(existing, dict):
                kept[key] = t
                continue
                
            # Comparar confianza
            current_conf = t.get("confidence", 0.0)
            existing_conf = existing.get("confidence", 0.0)
            
            try:
                current_conf = float(current_conf)
                existing_conf = float(existing_conf)
            except (ValueError, TypeError):
                current_conf = 0.0
                existing_conf = 0.0
            
            # Solo reemplazar si la mejora es significativa
            if current_conf > existing_conf + margin:
                kept[key] = t
                removed += 1
            else:
                removed += 1
    
    # Convertir a lista final
    final_triplets = list(kept.values())
    
    stats = {"removed_duplicates": removed, "kept": len(final_triplets)}
    return final_triplets, stats

def apply_connectivity_filter(triplets: List[Dict], config: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
    """Aplica filtro de conectividad con debug detallado y lógica corregida."""
    min_connections_cfg = config.get("quality_control", {}).get("min_connections_per_entity", 3)
    min_connections = max(1, int(min_connections_cfg))
    
    print(f"\n🔍 DEBUG CONECTIVIDAD:")
    print(f"   Umbral mínimo: {min_connections} conexión(es)")
    print(f"   Tripletas a analizar: {len(triplets):,}")
    
    # Contar apariciones de cada entidad
    entity_connections = Counter()
    valid_triplets = []
    
    for t in triplets:
        if not isinstance(t, dict):
            continue
            
        valid_triplets.append(t)
        h = t.get("head", "").strip()
        ta = t.get("tail", "").strip()
        
        # Contar cada aparición (una entidad puede aparecer múltiples veces)
        if h:
            entity_connections[h] += 1
        if ta:
            entity_connections[ta] += 1
    
    print(f"   Entidades únicas encontradas: {len(entity_connections):,}")
    
    # Analizar distribución de conectividades
    if entity_connections:
        counts = list(entity_connections.values())
        print(f"   Conectividad mínima real: {min(counts)}")
        print(f"   Conectividad máxima real: {max(counts)}")
        print(f"   Conectividad promedio: {statistics.mean(counts):.1f}")
        print(f"   Conectividad mediana: {statistics.median(counts)}")
        
        # Histograma básico
        hist_counts = Counter(counts)
        print(f"   Distribución de conectividades:")
        for conn, num_entities in sorted(hist_counts.items())[:10]:  # Top 10
            pct = (num_entities / len(entity_connections)) * 100
            print(f"     {conn} conexión(es): {num_entities:,} entidades ({pct:.1f}%)")
        if len(hist_counts) > 10:
            print(f"     ... y {len(hist_counts) - 10} niveles más")
    
    # Identificar entidades a eliminar
    entities_to_remove = {e for e, c in entity_connections.items() if c < min_connections}
    
    print(f"   Entidades que NO cumplen umbral ({min_connections}): {len(entities_to_remove):,}")
    print(f"   Porcentaje a eliminar: {(len(entities_to_remove) / len(entity_connections) * 100):.1f}%")
    
    # Mostrar ejemplos de entidades eliminadas
    if entities_to_remove:
        sample_removed = sorted([(e, entity_connections[e]) for e in list(entities_to_remove)[:10]], 
                               key=lambda x: -x[1])
        print(f"   Ejemplos de entidades eliminadas:")
        for ent, conn in sample_removed:
            print(f"     '{ent}' (conectividad: {conn})")
    
    # Filtrar tripletas
    filtered = []
    for t in valid_triplets:
        h = t.get("head", "").strip()
        ta = t.get("tail", "").strip()
        if h not in entities_to_remove and ta not in entities_to_remove:
            filtered.append(t)
    
    removed_triplets = len(valid_triplets) - len(filtered)
    print(f"   Tripletas eliminadas: {removed_triplets:,}")
    print(f"   Tripletas mantenidas: {len(filtered):,}")
    
    # Estadísticas detalladas
    counts = list(entity_connections.values()) if entity_connections else [0]
    hist = {
        "total_entities": len(entity_connections),
        "removed_entities": len(entities_to_remove),
        "min_connections_threshold": min_connections,
        "median_connections": statistics.median(counts),
        "mean_connections": statistics.mean(counts) if counts else 0,
        "percent_below_threshold": (len(entities_to_remove) / len(entity_connections) * 100) if entity_connections else 0,
        "distribution": dict(Counter(counts).most_common(20)),
        "top_removed_entities_sample": sorted([(e, entity_connections[e]) for e in list(entities_to_remove)[:20]], key=lambda x: -x[1])
    }
    
    return filtered, {
        "removed_entities": hist["removed_entities"],
        "removed_triplets": removed_triplets,
        "kept_triplets": len(filtered),
        "diagnosis": hist
    }

def apply_quality_filters(input_data: Union[Path, List[Dict[str, Any]]], config: Dict[str, Any], relax: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aplica normalización y filtros con debug mejorado."""
    # ajustar config temporalmente según relax
    cfg = dict(config)
    qc = cfg.get("quality_control", {}).copy()
    qc["min_confidence_threshold"] = max(0.0, qc.get("min_confidence_threshold", 0.4) / max(1.0, relax))
    qc["min_connections_per_entity"] = max(1, int(qc.get("min_connections_per_entity", 3) / max(1.0, relax)))
    cfg["quality_control"] = qc
    
    print(f"Usando relax={relax:.2f} -> min_confidence={qc['min_confidence_threshold']:.3f}, min_connections={qc['min_connections_per_entity']}")
    
    # cargar tripletas
    if isinstance(input_data, Path):
        triplets = []
        for jsonl_file in input_data.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        triplet = json.loads(line)
                        # Validación defensiva: solo añadir si es diccionario
                        if isinstance(triplet, dict):
                            triplets.append(triplet)
                    except json.JSONDecodeError:
                        continue
    else:
        # Filtrar solo diccionarios válidos
        triplets = [t for t in input_data if isinstance(t, dict)]
    
    initial_count = len(triplets)
    # 1) Validación con explicación
    reasons_counter = Counter()
    valid_triplets = []
    filtered_basic = 0
    
    for t in tqdm(triplets, desc="validar"):
        # Validación defensiva adicional
        if not isinstance(t, dict):
            filtered_basic += 1
            reasons_counter["invalid_format"] += 1
            continue
            
        reasons = explain_triplet(t, qc)
        if reasons:
            filtered_basic += 1
            for r in reasons:
                reasons_counter[r] += 1
        else:
            valid_triplets.append(t)
    
    # 2) Normalización y mapping (sin cambios)
    all_entities = set()
    for triplet in valid_triplets:
        all_entities.add(triplet["head"])
        all_entities.add(triplet["tail"])
    entity_mapping = find_similar_entities_optimized(list(all_entities), qc)
    for triplet in valid_triplets:
        head_mapped = entity_mapping.get(triplet["head"], triplet["head"])
        tail_mapped = entity_mapping.get(triplet["tail"], triplet["tail"])
        triplet["head"] = normalize_entity(head_mapped, qc)
        triplet["tail"] = normalize_entity(tail_mapped, qc)
    
    # 3) Unicidad (con margen)
    final_triplets, uniqueness_stats = apply_uniqueness_rules(valid_triplets, qc)
    
    # 4) Conectividad (diagnóstico incluido) - AÑADIR DEBUG ANTES
    print(f"\n📊 ANTES del filtro de conectividad:")
    print(f"   Tripletas disponibles: {len(final_triplets):,}")
    
    # Hacer un mini-análisis previo
    pre_entities = set()
    for t in final_triplets:
        if isinstance(t, dict):
            h = t.get("head", "").strip()
            ta = t.get("tail", "").strip()
            if h: pre_entities.add(h)
            if ta: pre_entities.add(ta)
    
    print(f"   Entidades únicas antes: {len(pre_entities):,}")
    
    final_triplets, connectivity_stats = apply_connectivity_filter(final_triplets, cfg)
    
    # 5) Duplicados exactos con validación
    seen = set()
    unique_triplets = []
    duplicates = 0
    
    for triplet in final_triplets:
        # Validación defensiva
        if not isinstance(triplet, dict):
            continue
            
        head = triplet.get("head", "")
        relation = triplet.get("type", "")
        tail = triplet.get("tail", "")
        
        key = (head, relation, tail)
        if key not in seen:
            seen.add(key)
            unique_triplets.append(triplet)
        else:
            duplicates += 1
    
    final_count = len(unique_triplets)
    
    stats = {
        "initial_count": initial_count,
        "final_count": final_count,
        "filtered_basic": filtered_basic,
        "filter_reasons_top": reasons_counter.most_common(20),
        "entity_mappings": len(entity_mapping),
        "uniqueness_duplicates": uniqueness_stats.get("removed_duplicates", 0),
        "exact_duplicates": duplicates,
        "connectivity": connectivity_stats,
        "reduction_percent": ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0,
        "quality_percent": (final_count / initial_count * 100) if initial_count > 0 else 0
    }
    
    # imprimir resumen diagnóstico breve
    print("Diagnóstico de filtrado:")
    print(f"  - Filtrados por validación: {filtered_basic:,}")
    print(f"  - Razones principales: {stats['filter_reasons_top'][:8]}")
    print(f"  - Entidades mapeadas: {len(entity_mapping):,}")
    print(f"  - Duplicados por unicidad (removidos): {uniqueness_stats.get('removed_duplicates',0):,}")
    print(f"  - Entidades eliminadas por conectividad: {connectivity_stats.get('removed_entities',0):,} ({connectivity_stats.get('diagnosis',{}).get('percent_below_threshold',0):.1f}%)")
    print(f"  - Tripletas finales: {final_count:,} ({stats['quality_percent']:.1f}%)")
    
    # Sugerencia rápida basada en estadísticas
    if connectivity_stats.get('diagnosis', {}).get('percent_below_threshold', 0) > 50:
        print("Sugerencia: más del 50% de entidades tienen menos conexiones que el umbral.")
        print("  • Prueba reducir 'min_connections_per_entity' a 1 o 2 en config o usar --relax 2.0 en CLI.")
    
    return unique_triplets, stats

def save_canonical_triplets(triplets: List[Dict], output_path: Path):
    """Guarda las tripletas canónicas en formato JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
    
    print(f"Guardado: {len(triplets):,} tripletas en {output_path}")