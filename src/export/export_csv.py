"""
Exportador CSV para tripletas culturales LATAM.
Soporta múltiples formatos y opciones de filtrado.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import argparse


def load_triplets(input_path: Path) -> List[Dict[str, Any]]:
    """Carga tripletas desde archivo JSONL."""
    triplets = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triplet = json.loads(line)
                triplets.append(triplet)
            except json.JSONDecodeError:
                continue
    return triplets


def export_basic_csv(triplets: List[Dict], output_path: Path):
    """Exporta formato CSV básico: entidad, relacion, valor."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['entidad', 'relacion', 'valor'])
        
        for triplet in triplets:
            entidad = triplet.get('head', '').strip()
            relacion = triplet.get('type', '').strip()
            valor = triplet.get('tail', '').strip()
            
            if entidad and relacion and valor:
                writer.writerow([entidad, relacion, valor])
    
    print(f"✅ CSV básico exportado: {output_path} ({len(triplets):,} tripletas)")


def export_extended_csv(triplets: List[Dict], output_path: Path):
    """Exporta CSV extendido con todos los campos en español."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'entidad', 'tipo_entidad', 'relacion', 'valor', 'tipo_valor', 
            'confianza', 'archivo_fuente', 'indice_chunk'
        ])
        
        for triplet in triplets:
            writer.writerow([
                triplet.get('head', ''),
                triplet.get('head_type', ''),
                triplet.get('type', ''),
                triplet.get('tail', ''),
                triplet.get('tail_type', ''),
                triplet.get('confidence', ''),
                triplet.get('source_file', ''),
                triplet.get('chunk_index', '')
            ])
    
    print(f"✅ CSV extendido exportado: {output_path} ({len(triplets):,} tripletas)")


def export_entities_csv(triplets: List[Dict], output_path: Path):
    """Exporta lista única de entidades con estadísticas en español."""
    entity_stats = {}
    
    for triplet in triplets:
        entidad = triplet.get('head', '').strip()
        valor = triplet.get('tail', '').strip()
        tipo_entidad = triplet.get('head_type', '')
        tipo_valor = triplet.get('tail_type', '')
        
        if entidad:
            if entidad not in entity_stats:
                entity_stats[entidad] = {
                    'entidad': entidad,
                    'tipo': tipo_entidad,
                    'como_entidad': 0,
                    'como_valor': 0,
                    'apariciones_totales': 0,
                    'relaciones_como_entidad': set(),
                    'relaciones_como_valor': set()
                }
            entity_stats[entidad]['como_entidad'] += 1
            entity_stats[entidad]['apariciones_totales'] += 1
            entity_stats[entidad]['relaciones_como_entidad'].add(triplet.get('type', ''))
        
        if valor:
            if valor not in entity_stats:
                entity_stats[valor] = {
                    'entidad': valor,
                    'tipo': tipo_valor,
                    'como_entidad': 0,
                    'como_valor': 0,
                    'apariciones_totales': 0,
                    'relaciones_como_entidad': set(),
                    'relaciones_como_valor': set()
                }
            entity_stats[valor]['como_valor'] += 1
            entity_stats[valor]['apariciones_totales'] += 1
            entity_stats[valor]['relaciones_como_valor'].add(triplet.get('type', ''))
    
    # Convertir sets a conteos
    for stats in entity_stats.values():
        stats['relaciones_unicas_como_entidad'] = len(stats['relaciones_como_entidad'])
        stats['relaciones_unicas_como_valor'] = len(stats['relaciones_como_valor'])
        stats['total_relaciones_unicas'] = len(stats['relaciones_como_entidad'] | stats['relaciones_como_valor'])
        del stats['relaciones_como_entidad']
        del stats['relaciones_como_valor']
    
    # Exportar
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'entidad', 'tipo', 'como_entidad', 'como_valor', 'apariciones_totales',
            'relaciones_unicas_como_entidad', 'relaciones_unicas_como_valor', 'total_relaciones_unicas'
        ])
        
        # Ordenar por total de apariciones
        sorted_entities = sorted(entity_stats.values(), key=lambda x: x['apariciones_totales'], reverse=True)
        
        for stats in sorted_entities:
            writer.writerow([
                stats['entidad'], stats['tipo'], stats['como_entidad'], stats['como_valor'],
                stats['apariciones_totales'], stats['relaciones_unicas_como_entidad'],
                stats['relaciones_unicas_como_valor'], stats['total_relaciones_unicas']
            ])
    
    print(f"✅ CSV de entidades exportado: {output_path} ({len(entity_stats):,} entidades únicas)")


def export_relations_csv(triplets: List[Dict], output_path: Path):
    """Exporta estadísticas de relaciones en español."""
    relation_stats = Counter()
    relation_types = {}
    
    for triplet in triplets:
        relacion = triplet.get('type', '').strip()
        tipo_entidad = triplet.get('head_type', '')
        tipo_valor = triplet.get('tail_type', '')
        
        if relacion:
            relation_stats[relacion] += 1
            
            if relacion not in relation_types:
                relation_types[relacion] = {
                    'tipos_entidad': Counter(),
                    'tipos_valor': Counter(),
                    'pares_tipos': Counter()
                }
            
            relation_types[relacion]['tipos_entidad'][tipo_entidad] += 1
            relation_types[relacion]['tipos_valor'][tipo_valor] += 1
            relation_types[relacion]['pares_tipos'][f"{tipo_entidad}->{tipo_valor}"] += 1
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'relacion', 'cantidad', 'porcentaje', 'tipo_entidad_mas_comun', 
            'tipo_valor_mas_comun', 'par_tipos_mas_comun'
        ])
        
        total_triplets = len(triplets)
        
        for relacion, cantidad in relation_stats.most_common():
            porcentaje = (cantidad / total_triplets) * 100
            
            tipo_entidad_mas_comun = relation_types[relacion]['tipos_entidad'].most_common(1)[0][0] if relation_types[relacion]['tipos_entidad'] else ''
            tipo_valor_mas_comun = relation_types[relacion]['tipos_valor'].most_common(1)[0][0] if relation_types[relacion]['tipos_valor'] else ''
            par_tipos_mas_comun = relation_types[relacion]['pares_tipos'].most_common(1)[0][0] if relation_types[relacion]['pares_tipos'] else ''
            
            writer.writerow([
                relacion, cantidad, f"{porcentaje:.2f}%", 
                tipo_entidad_mas_comun, tipo_valor_mas_comun, par_tipos_mas_comun
            ])
    
    print(f"✅ CSV de relaciones exportado: {output_path} ({len(relation_stats):,} relaciones únicas)")


def export_countries_csv(triplets: List[Dict], output_path: Path):
    """Exporta tripletas agrupadas por país LATAM en español."""
    countries_data = {}
    
    # Mapeo de entidades a países LATAM
    latam_countries = {
        'argentina', 'bolivia', 'brasil', 'brazil', 'chile', 'colombia', 
        'costa rica', 'cuba', 'ecuador', 'el salvador', 'guatemala', 
        'honduras', 'mexico', 'nicaragua', 'panama', 'paraguay', 'peru', 
        'puerto rico', 'republica dominicana', 'uruguay', 'venezuela'
    }
    
    for triplet in triplets:
        entidad = triplet.get('head', '').lower()
        valor = triplet.get('tail', '').lower()
        relacion = triplet.get('type', '')
        
        # Buscar país en la tripleta
        pais = None
        if 'country' in relacion.lower():
            if valor in latam_countries:
                pais = valor
        
        # Buscar por entidades conocidas
        if not pais:
            for latam_country in latam_countries:
                if latam_country in entidad or latam_country in valor:
                    pais = latam_country
                    break
        
        if pais:
            if pais not in countries_data:
                countries_data[pais] = []
            countries_data[pais].append(triplet)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pais', 'cantidad_tripletas', 'entidades_unicas', 'relaciones_unicas',
            'entidad_ejemplo', 'relacion_ejemplo', 'valor_ejemplo'
        ])
        
        for pais, triplets_list in sorted(countries_data.items()):
            entidades = set()
            relaciones = set()
            
            for t in triplets_list:
                entidades.add(t.get('head', ''))
                entidades.add(t.get('tail', ''))
                relaciones.add(t.get('type', ''))
            
            # Muestra
            sample = triplets_list[0] if triplets_list else {}
            
            writer.writerow([
                pais.title(), len(triplets_list), len(entidades), len(relaciones),
                sample.get('head', ''), sample.get('type', ''), sample.get('tail', '')
            ])
    
    print(f"✅ CSV por países exportado: {output_path} ({len(countries_data):,} países)")


def filter_triplets(triplets: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
    """Aplica filtros a las tripletas."""
    filtered = triplets
    
    # Filtro por confianza mínima
    if 'min_confidence' in filters:
        min_conf = float(filters['min_confidence'])
        filtered = [t for t in filtered if t.get('confidence', 0) >= min_conf]
    
    # Filtro por tipo de entidad
    if 'entity_types' in filters:
        allowed_types = set(filters['entity_types'])
        filtered = [t for t in filtered if 
                   t.get('head_type', '') in allowed_types or 
                   t.get('tail_type', '') in allowed_types]
    
    # Filtro por relaciones
    if 'relations' in filters:
        allowed_relations = set(filters['relations'])
        filtered = [t for t in filtered if t.get('type', '') in allowed_relations]
    
    # Filtro por país
    if 'countries' in filters:
        allowed_countries = set(c.lower() for c in filters['countries'])
        filtered = [t for t in filtered if 
                   any(country in t.get('head', '').lower() or 
                       country in t.get('tail', '').lower() 
                       for country in allowed_countries)]
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Exportar tripletas a CSV')
    parser.add_argument('input', help='Archivo JSONL de entrada')
    parser.add_argument('--output-dir', default='data/exports', help='Directorio de salida')
    parser.add_argument('--format', choices=['basic', 'extended', 'entities', 'relations', 'countries', 'all'], 
                       default='all', help='Formato de exportación')
    parser.add_argument('--min-confidence', type=float, help='Confianza mínima')
    parser.add_argument('--entity-types', nargs='+', help='Tipos de entidad a incluir')
    parser.add_argument('--relations', nargs='+', help='Relaciones a incluir')
    parser.add_argument('--countries', nargs='+', help='Países a incluir')
    
    args = parser.parse_args()
    
    # Cargar tripletas
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ No se encuentra el archivo: {input_path}")
        return
    
    print(f"📂 Cargando tripletas desde: {input_path}")
    triplets = load_triplets(input_path)
    print(f"📊 Cargadas {len(triplets):,} tripletas")
    
    # Aplicar filtros
    filters = {}
    if args.min_confidence:
        filters['min_confidence'] = args.min_confidence
    if args.entity_types:
        filters['entity_types'] = args.entity_types
    if args.relations:
        filters['relations'] = args.relations
    if args.countries:
        filters['countries'] = args.countries
    
    if filters:
        print(f"🔍 Aplicando filtros: {filters}")
        triplets = filter_triplets(triplets, filters)
        print(f"✅ Tripletas filtradas: {len(triplets):,}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exportar según formato
    base_name = input_path.stem
    
    if args.format == 'basic' or args.format == 'all':
        export_basic_csv(triplets, output_dir / f"{base_name}_basic.csv")
    
    if args.format == 'extended' or args.format == 'all':
        export_extended_csv(triplets, output_dir / f"{base_name}_extended.csv")
    
    if args.format == 'entities' or args.format == 'all':
        export_entities_csv(triplets, output_dir / f"{base_name}_entities.csv")
    
    if args.format == 'relations' or args.format == 'all':
        export_relations_csv(triplets, output_dir / f"{base_name}_relations.csv")
    
    if args.format == 'countries' or args.format == 'all':
        export_countries_csv(triplets, output_dir / f"{base_name}_countries.csv")
    
    print(f"🎉 Exportación completada en: {output_dir}")


if __name__ == "__main__":
    main()
