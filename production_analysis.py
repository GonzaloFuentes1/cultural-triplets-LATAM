#!/usr/bin/env python3
"""
Script de producción para análisis completo del dataset con mREBEL
Genera estadísticas detalladas para toma de decisiones sobre umbrales de confianza
"""

import os
import sys
import json
import time
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extract'))

def production_analysis():
    """Análisis completo del dataset para producción"""
    
    print("🚀 ANÁLISIS DE PRODUCCIÓN - DATASET COMPLETO")
    print("=" * 80)
    print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Importar después de configurar el path
        from mrebel_wrapper import load_mrebel, run_mrebel
        
        # Cargar modelo
        print("🔄 Cargando modelo mREBEL...")
        start_time = time.time()
        model, tokenizer = load_mrebel(
            "Babelscape/mrebel-large",
            verify_cuda=False
        )
        load_time = time.time() - start_time
        print(f"✅ Modelo cargado en {load_time:.1f} segundos")
        
        # Leer dataset
        print("📖 Cargando dataset...")
        dataset_dir = "data/interim"
        
        if not os.path.exists(dataset_dir):
            print(f"❌ Directorio no encontrado: {dataset_dir}")
            return
        
        # Leer archivos JSON del directorio interim
        texts = []
        text_ids = []
        json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')][:50]  # Limitar a 50 archivos
        
        print(f"📁 Encontrados {len(json_files)} archivos JSON, procesando primeros 50...")
        
        for json_file in json_files:
            file_path = os.path.join(dataset_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extraer texto de chunks
                    if 'chunks' in data and data['chunks']:
                        # Unir todos los chunks en un solo texto
                        full_text = ' '.join(data['chunks'])
                        texts.append(full_text)
                        text_ids.append(data.get('source', json_file.replace('.json', '')))
                        
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"   ⚠️  Error procesando {json_file}: {e}")
                continue
        
        if not texts:
            print("❌ No se pudieron cargar textos del dataset")
            return
        
        print(f"📊 Dataset cargado: {len(texts)} textos para análisis")
        print(f"📝 Muestra: {texts[0][:100]}...")
        
        # === PROCESAMIENTO ===
        print("\\n🔄 PROCESANDO TEXTOS...")
        start_processing = time.time()
        
        all_results = []
        confidence_stats = {
            'all_confidences': [],
            'by_relation_type': defaultdict(list),
            'by_text_length': defaultdict(list),
            'triplets_per_text': [],
            'failed_extractions': 0,
            'processing_times': []
        }
        
        # Procesar textos
        for i, (text_id, text) in enumerate(zip(text_ids, texts)):
            text_start = time.time()
            
            # Progreso cada 10 textos
            if i % 10 == 0:
                elapsed = time.time() - start_processing
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(texts) - i - 1) / rate if rate > 0 else 0
                print(f"   {i+1:3d}/{len(texts)} | {rate:.1f} txt/s | ETA: {eta:.0f}s | {text[:50]}...")
            
            try:
                # Procesar texto
                results = run_mrebel(text, model, tokenizer)
                text_time = time.time() - text_start
                confidence_stats['processing_times'].append(text_time)
                
                if results:
                    # Información del texto
                    text_length = len(text.split())
                    confidence_stats['triplets_per_text'].append(len(results))
                    
                    # Clasificar por longitud de texto
                    if text_length < 20:
                        length_category = 'short'
                    elif text_length < 50:
                        length_category = 'medium'
                    else:
                        length_category = 'long'
                    
                    # Procesar cada triplete
                    for triplet in results:
                        conf = triplet.get('confidence', 0.5)
                        relation_type = triplet['type']
                        
                        # Almacenar resultado completo
                        result_entry = {
                            'text_id': text_id,
                            'text': text,
                            'text_length': text_length,
                            'length_category': length_category,
                            'head': triplet['head'],
                            'head_type': triplet.get('head_type', ''),
                            'relation': relation_type,
                            'tail': triplet['tail'],
                            'tail_type': triplet.get('tail_type', ''),
                            'confidence': conf,
                            'processing_time': text_time
                        }
                        all_results.append(result_entry)
                        
                        # Estadísticas
                        confidence_stats['all_confidences'].append(conf)
                        confidence_stats['by_relation_type'][relation_type].append(conf)
                        confidence_stats['by_text_length'][length_category].append(conf)
                else:
                    confidence_stats['failed_extractions'] += 1
                    
            except Exception as e:
                print(f"      ❌ Error procesando texto {i}: {e}")
                confidence_stats['failed_extractions'] += 1
        
        processing_time = time.time() - start_processing
        print(f"\\n✅ Procesamiento completado en {processing_time:.1f} segundos")
        
        # === ANÁLISIS ESTADÍSTICO ===
        print("\\n📊 GENERANDO ESTADÍSTICAS...")
        
        total_texts = len(texts)
        total_triplets = len(all_results)
        successful_texts = total_texts - confidence_stats['failed_extractions']
        
        # Estadísticas básicas
        basic_stats = {
            'dataset_info': {
                'total_texts': total_texts,
                'successful_extractions': successful_texts,
                'failed_extractions': confidence_stats['failed_extractions'],
                'success_rate': successful_texts / total_texts * 100,
                'total_triplets': total_triplets,
                'avg_triplets_per_text': sum(confidence_stats['triplets_per_text']) / len(confidence_stats['triplets_per_text']) if confidence_stats['triplets_per_text'] else 0,
                'avg_processing_time': sum(confidence_stats['processing_times']) / len(confidence_stats['processing_times']) if confidence_stats['processing_times'] else 0
            }
        }
        
        # Distribución de confianza
        confidences = confidence_stats['all_confidences']
        if confidences:
            # Percentiles
            confidences_sorted = sorted(confidences)
            n = len(confidences_sorted)
            
            confidence_distribution = {
                'total_triplets': len(confidences),
                'mean': sum(confidences) / len(confidences),
                'median': confidences_sorted[n//2],
                'min': min(confidences),
                'max': max(confidences),
                'percentiles': {
                    'p10': confidences_sorted[int(n * 0.1)],
                    'p25': confidences_sorted[int(n * 0.25)],
                    'p50': confidences_sorted[int(n * 0.5)],
                    'p75': confidences_sorted[int(n * 0.75)],
                    'p90': confidences_sorted[int(n * 0.9)],
                    'p95': confidences_sorted[int(n * 0.95)],
                    'p99': confidences_sorted[int(n * 0.99)]
                }
            }
            
            # Distribución por rangos
            ranges = {
                'excellent_099_100': sum(1 for c in confidences if c >= 0.99),
                'very_high_095_099': sum(1 for c in confidences if 0.95 <= c < 0.99),
                'high_090_095': sum(1 for c in confidences if 0.90 <= c < 0.95),
                'good_080_090': sum(1 for c in confidences if 0.80 <= c < 0.90),
                'medium_070_080': sum(1 for c in confidences if 0.70 <= c < 0.80),
                'low_050_070': sum(1 for c in confidences if 0.50 <= c < 0.70),
                'very_low_000_050': sum(1 for c in confidences if c < 0.50)
            }
            
            confidence_distribution['ranges'] = ranges
            confidence_distribution['ranges_percentages'] = {
                k: v / len(confidences) * 100 for k, v in ranges.items()
            }
        
        # Análisis por tipo de relación
        relation_analysis = {}
        for relation_type, confs in confidence_stats['by_relation_type'].items():
            if len(confs) >= 2:  # Solo analizar relaciones con al menos 2 casos
                relation_analysis[relation_type] = {
                    'count': len(confs),
                    'mean_confidence': sum(confs) / len(confs),
                    'min_confidence': min(confs),
                    'max_confidence': max(confs),
                    'high_confidence_count': sum(1 for c in confs if c >= 0.9),
                    'high_confidence_percentage': sum(1 for c in confs if c >= 0.9) / len(confs) * 100
                }
        
        # Top relaciones por frecuencia y confianza
        top_relations_by_frequency = sorted(
            relation_analysis.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:10]
        
        top_relations_by_confidence = sorted(
            relation_analysis.items(), 
            key=lambda x: x[1]['mean_confidence'], 
            reverse=True
        )[:10]
        
        # === GUARDAR RESULTADOS ===
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"analysis_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar datos completos
        results_file = os.path.join(output_dir, "complete_results.jsonl")
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\\n')
        
        # Guardar estadísticas
        stats_file = os.path.join(output_dir, "statistics.json")
        full_stats = {
            'basic_stats': basic_stats,
            'confidence_distribution': confidence_distribution,
            'relation_analysis': relation_analysis,
            'top_relations_by_frequency': top_relations_by_frequency,
            'top_relations_by_confidence': top_relations_by_confidence,
            'processing_info': {
                'total_processing_time': processing_time,
                'texts_per_second': len(texts) / processing_time,
                'triplets_per_second': total_triplets / processing_time,
                'timestamp': timestamp
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(full_stats, f, indent=2, ensure_ascii=False)
        
        # === GENERAR REPORTE VISUAL ===
        print("\\n📈 REPORTE DE ANÁLISIS")
        print("=" * 80)
        
        print(f"📊 INFORMACIÓN BÁSICA:")
        print(f"   • Textos procesados: {total_texts}")
        print(f"   • Extracciones exitosas: {successful_texts} ({successful_texts/total_texts*100:.1f}%)")
        print(f"   • Total tripletes: {total_triplets}")
        print(f"   • Promedio tripletes/texto: {basic_stats['dataset_info']['avg_triplets_per_text']:.1f}")
        print(f"   • Tiempo promedio/texto: {basic_stats['dataset_info']['avg_processing_time']:.2f}s")
        
        if confidences:
            print(f"\\n🎯 DISTRIBUCIÓN DE CONFIANZA:")
            print(f"   • Confianza promedio: {confidence_distribution['mean']:.3f}")
            print(f"   • Mediana: {confidence_distribution['median']:.3f}")
            print(f"   • Rango: {confidence_distribution['min']:.3f} - {confidence_distribution['max']:.3f}")
            
            print(f"\\n📊 PERCENTILES CLAVE:")
            for p, v in confidence_distribution['percentiles'].items():
                print(f"   • {p.upper()}: {v:.3f}")
            
            print(f"\\n🎨 DISTRIBUCIÓN POR RANGOS:")
            for range_name, percentage in confidence_distribution['ranges_percentages'].items():
                count = confidence_distribution['ranges'][range_name]
                range_display = range_name.replace('_', ' ').title().replace(' ', ' ')
                print(f"   • {range_display:20}: {count:4d} ({percentage:5.1f}%)")
            
            print(f"\\n🏆 TOP 5 RELACIONES MÁS FRECUENTES:")
            for i, (rel_type, stats) in enumerate(top_relations_by_frequency[:5]):
                print(f"   {i+1}. {rel_type:30} | {stats['count']:3d} casos | conf: {stats['mean_confidence']:.3f}")
            
            print(f"\\n⭐ TOP 5 RELACIONES MÁS CONFIABLES:")
            for i, (rel_type, stats) in enumerate(top_relations_by_confidence[:5]):
                print(f"   {i+1}. {rel_type:30} | {stats['count']:3d} casos | conf: {stats['mean_confidence']:.3f}")
        
        print(f"\\n💾 RESULTADOS GUARDADOS EN: {output_dir}/")
        print(f"   • complete_results.jsonl - Todos los tripletes extraídos")
        print(f"   • statistics.json - Estadísticas completas")
        
        # === RECOMENDACIONES ===
        print(f"\\n🎯 RECOMENDACIONES PARA UMBRALES:")
        if confidences:
            p95 = confidence_distribution['percentiles']['p95']
            p90 = confidence_distribution['percentiles']['p90']
            p75 = confidence_distribution['percentiles']['p75']
            p50 = confidence_distribution['percentiles']['p50']
            
            print(f"   • ALTA CONFIANZA (usar directamente): ≥ {p90:.3f} ({sum(1 for c in confidences if c >= p90)}/{len(confidences)} casos, {sum(1 for c in confidences if c >= p90)/len(confidences)*100:.1f}%)")
            print(f"   • CONFIANZA MEDIA (revisar): {p75:.3f} - {p90:.3f} ({sum(1 for c in confidences if p75 <= c < p90)}/{len(confidences)} casos)")
            print(f"   • BAJA CONFIANZA (verificar): < {p75:.3f} ({sum(1 for c in confidences if c < p75)}/{len(confidences)} casos)")
        
        print(f"\\n✅ ANÁLISIS COMPLETADO")
        print(f"Tiempo total: {time.time() - start_time:.1f} segundos")
        print("=" * 80)
        
        # Limpiar memoria
        del model
        import torch
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    production_analysis()