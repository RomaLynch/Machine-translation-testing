import numpy as np

if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.bool = bool
    np.int_ = np.int64

import torch
import os
import csv
import json
import re
from typing import List, Tuple
from simalign import SentenceAligner
from sacrebleu.tokenizers import tokenizer_13a
import jieba


# After execution, execute addMutAlignResToCSV.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE}")

LOCAL_MODEL_PATH = "bert-base-multilingual-cased"
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"The local model path does not exist:{LOCAL_MODEL_PATH}")

aligner_global = SentenceAligner(
    model=LOCAL_MODEL_PATH,
    token_type="bpe",
    matching_methods="mai",
    device=DEVICE,
)

en_tokenizer = tokenizer_13a.Tokenizer13a()
print("Tokenizer13a loading completed!")

def en_ch_align(en_sentence: str, zh_sentence: str, tgt_poly: str) -> Tuple[str, List[str], List[str], str]:
    if not zh_sentence.strip():
        return "", [], [], ""

    en_tok = en_tokenizer(en_sentence).split() if en_sentence else []
    zh_tok = list(jieba.cut_for_search(zh_sentence)) if zh_sentence else []

    if not en_tok or not zh_tok:
        return "", [], [], ""

    try:
        align_res = aligner_global.get_word_aligns(en_tok, zh_tok)
        itermax_align = align_res.get("itermax", [])
    except Exception as e:
        print(f"Alignment error (sentence index not displayed):{e}")
        return "", [], [], ""

    if not itermax_align:
        return "", [], [], ""
    en_to_zh_map = {}
    for e_idx, z_idx in itermax_align:
        if 0 <= e_idx < len(en_tok) and 0 <= z_idx < len(zh_tok):
            en_to_zh_map.setdefault(e_idx, []).append(z_idx)

    target_en_idx = None
    tgt_lower = tgt_poly.lower()
    for i, tok in enumerate(en_tok):
        if tok.lower() == tgt_lower:
            target_en_idx = i
            break

    if target_en_idx is None:
        return "", [], [], ""

    zh_indices = sorted(set(en_to_zh_map.get(target_en_idx, [])))

    ch_word = zh_tok[zh_indices[0]] if zh_indices else ""
    align_pairs_str = ", ".join([
        f"{en_tok[e]}({e})¡ú{zh_tok[z]}({z})"
        for e, z in itermax_align
        if 0 <= e < len(en_tok) and 0 <= z < len(zh_tok)
    ])

    return ch_word, en_tok, zh_tok, align_pairs_str

def batch_en_ch_align(
        en_sentences: List[str],
        zh_sentences: List[str],
        tgt_polys: List[str]
) -> List[Tuple[str, List[str], List[str], str]]:

    assert len(en_sentences) == len(zh_sentences) == len(tgt_polys), \
        f"The number of lines in the three files is inconsistent!EN: {len(en_sentences)}, ZH: {len(zh_sentences)}, POLY: {len(tgt_polys)}"

    results = []
    total = len(en_sentences)
    print(f"Start aligning {total} sentences (sentence by sentence processing)")

    for i in range(total):
        en_sent = en_sentences[i]
        zh_sent = zh_sentences[i]
        tgt_poly = tgt_polys[i]

        res = en_ch_align(en_sent, zh_sent, tgt_poly)
        results.append(res)

        if (i + 1) % 5000 == 0:
            print(f"Processed  {i + 1} / {total} rows")

    return results

def read_lines_keep_empty(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.rstrip('\n\r') for line in lines]

if __name__ == "__main__":
    en_sents = read_lines_keep_empty("bugDetectionData/src_sent.txt")
    zh_sents = read_lines_keep_empty("bugDetectionData/cleaned_mut3_ch_sent.txt")
    tgt_words = read_lines_keep_empty("bugDetectionData/polywordInMut.txt")

    print(f"Input file line count:EN: {len(en_sents)}, ZH: {len(zh_sents)}, POLY: {len(tgt_words)}")

    results = batch_en_ch_align(en_sents, zh_sents, tgt_words)

    output_csv = "bugDetectionData/mut_align_results/mut3_align.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ch_word", "en_tokens", "zh_tokens", "alignment_pairs"])
        for ch_word, en_tok, zh_tok, align_str in results:
            en_tok_str = json.dumps(en_tok, ensure_ascii=False)
            zh_tok_str = json.dumps(zh_tok, ensure_ascii=False)
            writer.writerow([ch_word, en_tok_str, zh_tok_str, align_str])
