from typing import List, Union, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------- Chequeo breve de GPU (una vez) ----------------------

def _print_cuda_check(model, prefix="[GPU]"):
    try:
        print(f"{prefix} torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cur = torch.cuda.current_device()
            print(f"{prefix} model_device={next(model.parameters()).device} | current=cuda:{cur} ({torch.cuda.get_device_name(cur)})")
        else:
            print(f"{prefix} model_device={next(model.parameters()).device}")
    except Exception:
        pass


# ---------------------- Parser (igual que antes) ----------------------

def extract_triplets_typed(text: str) -> list:
    triplets = []
    text = text.strip()

    current = 'x'  # t=subject, s=object, o=relation
    subject, relation, object_, subject_type, object_type = '', '', '', '', ''

    tokens = (
        text.replace("<s>", "")
            .replace("<pad>", "")
            .replace("</s>", "")
            .replace("tp_XX", "")
            .replace("__en__", "")
            .split()
    )

    for token in tokens:
        if token in ("<triplet>", "<relation>"):
            if relation:
                triplets.append({
                    'head': subject.strip(),
                    'head_type': subject_type,
                    'type': relation.strip(),
                    'tail': object_.strip(),
                    'tail_type': object_type
                })
            current = 't'
            subject, relation, object_, subject_type, object_type = '', '', '', '', ''
        elif token.startswith("<") and token.endswith(">"):
            if current in ('t', 'o'):
                current = 's'
                if relation:
                    triplets.append({
                        'head': subject.strip(),
                        'head_type': subject_type,
                        'type': relation.strip(),
                        'tail': object_.strip(),
                        'tail_type': object_type
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

    if subject and relation and object_ and subject_type and object_type:
        triplets.append({
            'head': subject.strip(),
            'head_type': subject_type,
            'type': relation.strip(),
            'tail': object_.strip(),
            'tail_type': object_type
        })

    return triplets


# ---------------------- Carga optimizada (A100) ----------------------

def load_mrebel(
    model_name: str,
    device: str = "cuda",                          # "cuda", "cuda:0", etc.
    dtype: Optional[torch.dtype] = torch.bfloat16, # A100 → bf16
    verify_cuda: bool = True
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=(dtype if device.startswith("cuda") else torch.float32)
    ).to(device)
    model.eval()

    if verify_cuda:
        _print_cuda_check(model)

    return model, tokenizer


# ---------------------- Inference en batch (silenciosa) ----------------------

def run_mrebel(
    texts: Union[str, List[str]],
    model,
    tokenizer,
    *,
    batch_size: int = 32,               # Sube en A100 si hay VRAM (48, 64…)
    max_input_tokens: int = 512,        # Sube si tu checkpoint lo permite (p.ej., 1024)
    max_new_tokens: int = 128,
    num_beams: int = 3,
    num_return_sequences: int = 1,
    pad_to_multiple_of: int = 8
) -> Union[List[dict], List[List[dict]]]:
    """
    Devuelve:
      - si texto único: list[dict]
      - si lista de textos: list[list[dict]]
    """
    # ✅ Auto-fix: si por alguna razón el modelo quedó en CPU, súbelo.
    try:
        if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
            model.to("cuda")
            model.eval()
    except Exception:
        pass

    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    device = next(model.parameters()).device
    start_token_id = tokenizer.convert_tokens_to_ids("tp_XX")

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        length_penalty=0.0,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        use_cache=True
    )

    results: List[List[dict]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            max_length=max_input_tokens,
            padding=True,
            truncation=True,
            return_tensors='pt',
            pad_to_multiple_of=pad_to_multiple_of
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        if device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.generate(
                    **enc,
                    decoder_start_token_id=start_token_id,
                    **gen_kwargs
                )
        else:
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    decoder_start_token_id=start_token_id,
                    **gen_kwargs
                )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)
        k = num_return_sequences

        if k == 1:
            for seq in decoded:
                trips = extract_triplets_typed(seq)
                results.append(trips)
        else:
            for j in range(0, len(decoded), k):
                seqs = decoded[j:j + k]
                merged = []
                for s in seqs:
                    merged.extend(extract_triplets_typed(s))
                results.append(merged)

    return results[0] if single_input else results
