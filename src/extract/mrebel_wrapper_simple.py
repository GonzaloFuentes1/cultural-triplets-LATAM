#!/usr/bin/env python3
"""
Wrapper simple de mREBEL que funciona correctamente
Basado en el código original que sabemos que funciona
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Union
import torch.nn.functional as F


def extract_triplets_typed(text: str, confidence: float = 0.5) -> list:
    """
    Parser original que funciona correctamente
    """
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    # Limpiar tokens especiales
    clean_text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "")
    
    for token in clean_text.split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({
                    'head': subject.strip(), 
                    'head_type': subject_type, 
                    'type': relation.strip(),
                    'tail': object_.strip(), 
                    'tail_type': object_type,
                    'confidence': confidence
                })
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({
                        'head': subject.strip(), 
                        'head_type': subject_type, 
                        'type': relation.strip(),
                        'tail': object_.strip(), 
                        'tail_type': object_type,
                        'confidence': confidence
                    })
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    
    # Agregar el último triplete si está completo
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({
            'head': subject.strip(), 
            'head_type': subject_type, 
            'type': relation.strip(),
            'tail': object_.strip(), 
            'tail_type': object_type,
            'confidence': confidence
        })
    
    return triplets


def load_mrebel_simple(model_name: str = "Babelscape/mrebel-large"):
    """
    Carga mREBEL con la configuración exacta que funciona
    """
    print(f"🔄 Cargando tokenizer desde {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="tp_XX") 
    
    print(f"🔄 Cargando modelo desde {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Mover a CUDA si está disponible
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Modelo movido a CUDA")
    
    print("✅ Modelo cargado y listo")
    return model, tokenizer


def run_mrebel_simple(
    texts: Union[str, List[str]], 
    model, 
    tokenizer,
    max_length: int = 256,
    num_beams: int = 3,
    num_return_sequences: int = 1
) -> Union[List[dict], List[List[dict]]]:
    """
    Ejecuta mREBEL con la configuración que sabemos que funciona
    """
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]
    
    # Configuración de generación
    gen_kwargs = {
        "max_length": max_length,
        "length_penalty": 0,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "forced_bos_token_id": None,
        "return_dict_in_generate": True,
        "output_scores": True
    }
    
    all_results = []
    
    for text in texts:
        # Tokenizar
        model_inputs = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        
        # Mover a CUDA si es necesario
        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
        
        # Generar
        with torch.inference_mode():
            generated_output = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
                **gen_kwargs
            )
        
        # Decodificar
        decoded_preds = tokenizer.batch_decode(generated_output.sequences, skip_special_tokens=False)
        
        # Calcular confianza promedio
        confidence = _calculate_simple_confidence(generated_output, tokenizer)
        
        # Extraer tripletes
        text_triplets = []
        for sentence in decoded_preds:
            triplets = extract_triplets_typed(sentence, confidence)
            text_triplets.extend(triplets)
        
        all_results.append(text_triplets)
    
    return all_results[0] if single_input else all_results


def _calculate_simple_confidence(generated_output, tokenizer) -> float:
    """
    Calcula confianza simple basada en softmax promedio
    """
    if not hasattr(generated_output, 'scores') or not generated_output.scores:
        return 0.95  # Confianza por defecto alta
    
    probs = []
    sequences = generated_output.sequences
    scores = generated_output.scores
    
    # Procesar primera secuencia
    for step_idx, step_scores in enumerate(scores):
        if step_idx + 1 >= sequences.shape[1]:
            break
            
        # Token generado en este paso
        generated_token_id = sequences[0, step_idx + 1].item()
        
        # Saltar tokens especiales
        token_str = tokenizer.decode([generated_token_id])
        if token_str in ['<s>', '</s>', '<pad>', 'tp_XX', '__en__']:
            continue
        
        # Calcular probabilidad softmax
        step_probs = F.softmax(step_scores[0], dim=-1)
        token_prob = step_probs[generated_token_id].item()
        probs.append(token_prob)
    
    if not probs:
        return 0.95
        
    return sum(probs) / len(probs)


# Mantener compatibilidad con el wrapper anterior
def load_mrebel(*args, **kwargs):
    """Compatibilidad con la API anterior"""
    return load_mrebel_simple()

def run_mrebel(*args, **kwargs):
    """Compatibilidad con la API anterior"""
    return run_mrebel_simple(*args, **kwargs)