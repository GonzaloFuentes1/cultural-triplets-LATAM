def segment_text(text: str, chunk_chars: int = 1800):
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
