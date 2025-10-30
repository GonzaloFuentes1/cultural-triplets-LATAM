#!/usr/bin/env python3
"""
Scraper optimizado para Wikipedia usando su API oficial.
Extrae contenido limpio y estructurado directamente desde MediaWiki API.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote
import re

class WikipediaAPIScraper:
    def __init__(self, language: str = "es"):
        """
        Inicializa el scraper de Wikipedia API.
        
        LÍMITES DE LA API DE WIKIPEDIA:
        - 500 peticiones por segundo para usuarios anónimos
        - 5000 peticiones por segundo para usuarios registrados
        - Máximo 50 páginas por petición con prop=extracts
        - Máximo 500 miembros por categoría por petición
        - Rate limiting automático si se supera
        
        Args:
            language: Código de idioma (es, en, pt, etc.)
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Cultural-Triplets-LATAM/1.0 (https://github.com/your-repo; your.email@domain.com)'
        })
        
        # Contadores para monitoreo de límites
        self.requests_made = 0
        self.start_time = time.time()
    
    def get_page_content(self, title: str) -> Optional[Dict]:
        """
        Obtiene el contenido COMPLETO de una página usando múltiples métodos.
        
        Args:
            title: Título de la página
            
        Returns:
            Dict con contenido estructurado o None si falla
        """
        # MÉTODO 1: Obtener contenido completo con wikitext
        params_wikitext = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'revisions|info|categories',
            'rvprop': 'content',
            'rvslots': 'main',
            'inprop': 'url',
            'cllimit': 50
        }
        
        # MÉTODO 2: Obtener extracto como respaldo
        params_extract = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info|categories',
            'exintro': False,
            'explaintext': True,
            'exsectionformat': 'wiki',
            'inprop': 'url',
            'cllimit': 50
        }
        
        # Intentar MÉTODO 1: Wikitext completo (más contenido)
        try:
            self.requests_made += 1
            response = self.session.get(self.base_url, params=params_wikitext, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                page_id = list(pages.keys())[0]
                
                if page_id != '-1' and 'revisions' in pages[page_id]:
                    page_data = pages[page_id]
                    revision = page_data['revisions'][0]
                    wikitext = revision['slots']['main']['*']
                    
                    # Convertir wikitext a texto plano
                    clean_content = self._wikitext_to_plain(wikitext)
                    
                    if clean_content and len(clean_content.strip()) > 200:
                        # Extraer categorías
                        categories = []
                        if 'categories' in page_data:
                            categories = [cat['title'].replace('Categoría:', '') for cat in page_data['categories']]
                        
                        return {
                            'title': page_data['title'],
                            'pageid': page_data['pageid'],
                            'url': page_data.get('fullurl', ''),
                            'content': clean_content,
                            'categories': categories,
                            'length': len(clean_content),
                            'source': 'wikipedia_api_wikitext'
                        }
        except Exception as e:
            print(f"   ⚠️ Método wikitext falló: {e}")
        
        # MÉTODO 2: Extracto como respaldo
        try:
            self.requests_made += 1
            response = self.session.get(self.base_url, params=params_extract, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'query' not in data or 'pages' not in data['query']:
                return None
                
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            
            if page_id == '-1':  # Página no encontrada
                return None
                
            page_data = pages[page_id]
            
            # Verificar que tenemos contenido
            if 'extract' not in page_data or not page_data['extract'].strip():
                return None
            
            # Extraer categorías
            categories = []
            if 'categories' in page_data:
                categories = [cat['title'].replace('Categoría:', '') for cat in page_data['categories']]
            
            return {
                'title': page_data['title'],
                'pageid': page_data['pageid'],
                'url': page_data.get('fullurl', ''),
                'content': page_data['extract'],
                'categories': categories,
                'length': len(page_data['extract']),
                'source': 'wikipedia_api_extract'
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo {title}: {e}")
            return None
    
    def search_pages(self, query: str, limit: int = 50) -> List[str]:
        """
        Busca páginas relacionadas con una consulta.
        
        Args:
            query: Término de búsqueda
            limit: Número máximo de resultados
            
        Returns:
            Lista de títulos de páginas
        """
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'srnamespace': 0  # Solo artículos principales
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'query' not in data or 'search' not in data['query']:
                return []
                
            return [result['title'] for result in data['query']['search']]
            
        except Exception as e:
            print(f"❌ Error buscando '{query}': {e}")
            return []
    
    def get_category_members(self, category: str, limit: int = 500) -> List[str]:
        """
        Obtiene miembros de una categoría de Wikipedia.
        
        Args:
            category: Nombre de la categoría (sin "Categoría:")
            limit: Número máximo de páginas
            
        Returns:
            Lista de títulos de páginas
        """
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': f'Categoría:{category}',
            'cmlimit': limit,
            'cmnamespace': 0  # Solo artículos principales
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'query' not in data or 'categorymembers' not in data['query']:
                return []
                
            return [member['title'] for member in data['query']['categorymembers']]
            
        except Exception as e:
            print(f"❌ Error obteniendo categoría '{category}': {e}")
            return []
    
    def _wikitext_to_plain(self, wikitext: str) -> str:
        """
        Convierte wikitext a texto plano limpio.
        EXTRAE TODO EL CONTENIDO incluyendo todas las secciones.
        
        Args:
            wikitext: Código wikitext crudo
            
        Returns:
            Texto plano limpio con TODO el contenido
        """
        if not wikitext:
            return ""
        
        text = wikitext
        
        # Remover elementos de wikitext que no aportan contenido
        patterns = [
            r'\{\{[^}]*\}\}',  # Plantillas {{}}
            r'\[\[Archivo:.*?\]\]',  # Enlaces a archivos
            r'\[\[File:.*?\]\]',  # Enlaces a archivos (inglés)
            r'\[\[Imagen:.*?\]\]',  # Enlaces a imágenes
            r'\[\[Category:.*?\]\]',  # Categorías
            r'\[\[Categoría:.*?\]\]',  # Categorías (español)
            r'<ref[^>]*>.*?</ref>',  # Referencias completas
            r'<ref[^>]*/>',  # Referencias vacías
            r'<.*?>',  # Tags HTML
            r'\{\|.*?\|\}',  # Tablas wikitext
            r"'''([^']+)'''",  # Negritas -> texto normal
            r"''([^']+)''",  # Cursivas -> texto normal
        ]
        
        for pattern in patterns:
            if pattern in ["'''([^']+)'''", "''([^']+)''"]:
                text = re.sub(pattern, r'\1', text, flags=re.DOTALL)
            else:
                text = re.sub(pattern, ' ', text, flags=re.DOTALL)
        
        # Limpiar enlaces internos [[Texto|Mostrar]] -> Mostrar
        text = re.sub(r'\[\[([^|\]]+)\|([^]]+)\]\]', r'\2', text)
        # Limpiar enlaces simples [[Texto]] -> Texto
        text = re.sub(r'\[\[([^]]+)\]\]', r'\1', text)
        
        # Limpiar títulos de secciones
        text = re.sub(r'={2,6}([^=]+)={2,6}', r'\1', text)
        
        # Normalizar espacios y saltos de línea
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        return text

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído de Wikipedia (para método de extracto).
        
        Args:
            text: Texto crudo de Wikipedia
            
        Returns:
            Texto limpio
        """
        if not text:
            return ""
            
        # Remover patrones comunes de Wikipedia
        patterns = [
            r'\[editar\]',  # Enlaces de edición
            r'\[cita requerida\]',  # Citas requeridas
            r'\[.*?\]',  # Referencias en corchetes
            r'==+ .*? ==+',  # Títulos de sección (opcional)
            r'\n\s*\n\s*\n',  # Múltiples saltos de línea
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

def scrape_wikipedia_category_clean(category: str, limit: int = 300, output_dir: Path = None, language: str = "es") -> int:
    """
    Extrae contenido limpio de una categoría de Wikipedia usando la API oficial.
    INTELIGENTE: Solo descarga páginas que no existen o están vacías/incompletas.
    
    Args:
        category: Nombre de la categoría
        limit: Límite de páginas por categoría
        output_dir: Directorio de salida
        language: Idioma de Wikipedia
        
    Returns:
        Número de páginas procesadas exitosamente
    """
    if output_dir is None:
        output_dir = Path("data/raw")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scraper = WikipediaAPIScraper(language)
    
    print(f"🔍 Obteniendo páginas de la categoría: {category}")
    page_titles = scraper.get_category_members(category, limit)
    
    if not page_titles:
        print(f"❌ No se encontraron páginas en la categoría: {category}")
        return 0
    
    # Filtrar páginas que ya están procesadas correctamente
    pages_to_process = []
    existing_valid = 0
    
    for title in page_titles:
        safe_filename = re.sub(r'[^\w\s-]', '', title).strip()
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
        output_file = output_dir / f"{safe_filename}.json"
        
        # Verificar si el archivo ya existe y es válido
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Verificar que el archivo tenga contenido válido
                content = existing_data.get('content', '')
                if content and len(content.strip()) >= 200:
                    existing_valid += 1
                    continue  # Saltar, ya está procesado correctamente
                else:
                    print(f"   � Reprocesar archivo incompleto: {title}")
                    pages_to_process.append(title)
            except:
                print(f"   🔄 Reprocesar archivo corrupto: {title}")
                pages_to_process.append(title)
        else:
            pages_to_process.append(title)
    
    print(f"📚 Categoría '{category}': {len(page_titles)} páginas encontradas")
    print(f"   ✅ Ya procesadas correctamente: {existing_valid}")
    print(f"   🆕 Por procesar: {len(pages_to_process)}")
    
    if not pages_to_process:
        print(f"   🎉 Todas las páginas ya están procesadas!")
        return existing_valid
    
    successful = 0
    
    for i, title in enumerate(pages_to_process, 1):
        print(f"[{i}/{len(pages_to_process)}] Procesando: {title}")
        
        # Obtener contenido limpio
        page_data = scraper.get_page_content(title)
        
        if not page_data:
            print(f"   ⚠️ Sin contenido válido")
            continue
            
        # Limpiar texto
        clean_content = scraper.clean_text(page_data['content'])
        
        if len(clean_content) < 200:  # Filtrar páginas muy cortas
            print(f"   ⚠️ Contenido muy corto ({len(clean_content)} chars)")
            continue
        
        # Guardar en formato JSON limpio
        output_data = {
            'title': page_data['title'],
            'url': page_data['url'],
            'content': clean_content,
            'categories': page_data['categories'],
            'metadata': {
                'source': 'wikipedia_api_clean',
                'language': language,
                'category_source': category,
                'original_length': page_data['length'],
                'cleaned_length': len(clean_content),
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Nombre de archivo seguro
        safe_filename = re.sub(r'[^\w\s-]', '', title).strip()
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
        output_file = output_dir / f"{safe_filename}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ Guardado: {len(clean_content):,} caracteres")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error guardando: {e}")
            continue
        
        # Pausa para no sobrecargar la API
        time.sleep(0.1)
    
    total_valid = existing_valid + successful
    print(f"\n🎉 Procesamiento completado:")
    print(f"   • Páginas ya existentes válidas: {existing_valid}")
    print(f"   • Páginas nuevas procesadas: {successful}")
    print(f"   • Total páginas válidas: {total_valid}")
    print(f"   • Archivos guardados en: {output_dir}")
    
    return total_valid

if __name__ == "__main__":
    # Ejemplo de uso
    scrape_wikipedia_category_clean("Historia de Argentina", limit=50)