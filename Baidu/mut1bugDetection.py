import numpy as np

if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.bool = bool
    np.int_ = np.int64

import json
import re
import torch
import os
import csv
from typing import List, Optional, Tuple, Dict
from transformers import AutoTokenizer, AutoModel
import time

# 执行完执行 mut1bugCount.py
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SBERT_V2_LOCAL_PATH = "SBERT_chinese_v2"
    POLYSEMY_JSON_PATH = "sense_dicts.json"
    INPUT_CSV_PATH = "bugDetectionData/bug/mut1_data4Detection.csv"
    OUTPUT_DIR = "bugDetectionData/bug"
    SIM_THRESHOLD_WORD = 0.8
    CSV_HEADER = [
        "en_sent", "ch_sent", "polyword", "ch_word",
        "en_tokens", "zh_tokens", "alignment_pairs", "poly_sense",
        "matched_gloss", "idx", "word2word_sim", "other_gloss", "mut1_sent",
        "mut2_sent", "mut3_sent", "status", "Mut1_ch", "cleaned_mut1_ch", "Mut2_ch", "cleaned_mut2_ch",
        "Mut3_ch", "cleaned_mut3_ch", "mut1_ch_word"
    ]


tokenizer_sbert = None
model_sbert = None
try:
    tokenizer_sbert = AutoTokenizer.from_pretrained(
        Config.SBERT_V2_LOCAL_PATH,
        local_files_only=True
    )
    model_sbert = AutoModel.from_pretrained(
        Config.SBERT_V2_LOCAL_PATH,
        local_files_only=True
    ).to(Config.DEVICE)
    model_sbert.eval()
except Exception as e:
    raise RuntimeError(f"{e}")


# 多义词库加载
def load_polysemy_json(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"poly dict not found：{json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


POLYSEMY_MAP = load_polysemy_json(Config.POLYSEMY_JSON_PATH)
print(f"The loading of the polysemous word library is complete, containing {len(POLYSEMY_MAP)} words in total")

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(Config.DEVICE)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def get_batch_embeddings(sentences: List[str]) -> torch.Tensor:
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    if not valid_sentences:
        return torch.tensor([]).to(Config.DEVICE)

    encoded_input = tokenizer_sbert(
        valid_sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(Config.DEVICE)

    with torch.no_grad():
        model_output = model_sbert(**encoded_input)

    embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

POLYSEMY_EMBEDDING_CACHE = {}


def preload_polysemy_embeddings():
    all_senses = []
    sense_keys = []

    for word, entries in POLYSEMY_MAP.items():
        if not isinstance(entries, dict):
            continue
        for idx_str, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            ch_senses = entry.get("[SENSES]", [])
            for sense in ch_senses:
                if isinstance(sense, str) and sense.strip():
                    sense_key = (word, idx_str, sense.strip())
                    sense_keys.append(sense_key)
                    all_senses.append(sense.strip())
    embeddings = get_batch_embeddings(all_senses)

    for i, sense_key in enumerate(sense_keys):
        word, idx_str, sense = sense_key
        if word not in POLYSEMY_EMBEDDING_CACHE:
            POLYSEMY_EMBEDDING_CACHE[word] = {}
        if idx_str not in POLYSEMY_EMBEDDING_CACHE[word]:
            POLYSEMY_EMBEDDING_CACHE[word][idx_str] = []
        POLYSEMY_EMBEDDING_CACHE[word][idx_str].append({
            "sense": sense,
            "embedding": embeddings[i:i + 1]  # [1, 768]
        })

    print(f"Preloading completed! Cached {len(sense_keys)} sense embeddings")

preload_polysemy_embeddings()

def calculate_batch_similarity(ch_word: str, sense_ch_list: List[str], word: str, target_idx_str: str = None) -> List[
    float]:
  
    ch_word_clean = ch_word.strip()
    if not ch_word_clean or not sense_ch_list:
        return [0.0] * len(sense_ch_list)

    ch_emb = get_batch_embeddings([ch_word_clean])
    if ch_emb.shape[0] == 0:
        return [0.0] * len(sense_ch_list)

  
    sense_embs = []
    for sense in sense_ch_list:
        sense_clean = sense.strip()
        emb = None

        word_cache = POLYSEMY_EMBEDDING_CACHE.get(word, {})
        idx_strs = [target_idx_str] if (target_idx_str and target_idx_str in word_cache) else word_cache.keys()

        for idx_str in idx_strs:
            for item in word_cache.get(idx_str, []):
               
                item_sense_clean = re.sub(r'[\s，。、；：""''()（）\n\r\t]', '', item["sense"].strip())
                input_sense_clean = re.sub(r'[\s，。、；：""''()（）\n\r\t]', '', sense_clean)

                if item_sense_clean == input_sense_clean:
                    emb = item["embedding"]
                    break
            if emb is not None:
                break

        if emb is None:
            sense_emb = get_batch_embeddings([sense_clean])
            emb = sense_emb if sense_emb.shape[0] > 0 else torch.zeros(1, 768).to(Config.DEVICE)

        sense_embs.append(emb)

    if not sense_embs:
        return [0.0] * len(sense_ch_list)
    sense_embs_tensor = torch.cat(sense_embs, dim=0)
    similarities = torch.mm(ch_emb, sense_embs_tensor.t()).squeeze(0).cpu().numpy().tolist()

    similarities = [max(0.0, min(1.0, float(s))) for s in similarities]
    if len(similarities) < len(sense_ch_list):
        similarities += [0.0] * (len(sense_ch_list) - len(similarities))

    return similarities

def get_idx(
        poly_sense: str,
        ch_word: str
) -> Tuple[Optional[int], float]:
    if not poly_sense.strip() or not ch_word.strip() or poly_sense not in POLYSEMY_MAP:
        return None, 0.0

    entries = POLYSEMY_MAP[poly_sense]
    if isinstance(entries, list):
        entries = {str(i): entry for i, entry in enumerate(entries) if entry is not None}
    if not isinstance(entries, dict):
        return None, 0.0

    for idx_str, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        ch_senses = entry.get("[SENSES]", [])
        for sense_ch in ch_senses:
            if not isinstance(sense_ch, str):
                continue
            sense_ch_clean = re.sub(r'[\s，。、；：""''()（）\n\r\t]', '', sense_ch.strip())
            ch_word_clean = re.sub(r'[\s，。、；：""''()（）\n\r\t]', '', ch_word.strip())

            if sense_ch_clean == ch_word_clean:
                idx = int(idx_str) if idx_str.isdigit() else None
                return idx, 1.0

    best_sim, best_idx = -1.0, None

    candidates = []
    sense_ch_list = []
    idx_str_list = []

    for idx_str, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        ch_senses = entry.get("[SENSES]", [])
        for sense_ch in ch_senses:
            if isinstance(sense_ch, str) and sense_ch.strip():
                candidates.append((idx_str, entry))
                sense_ch_list.append(sense_ch.strip())
                idx_str_list.append(idx_str)

    if candidates:
        for i, (idx_str, entry) in enumerate(candidates):
            sense_ch = sense_ch_list[i]
            sim = calculate_batch_similarity(ch_word.strip(), [sense_ch], poly_sense, idx_str)[0]

            if sim > best_sim:
                best_sim = sim
                best_idx = int(idx_str) if idx_str.isdigit() else None

    final_sim = best_sim if best_sim >= 0 else 0.0
    final_idx = best_idx if best_sim >= Config.SIM_THRESHOLD_WORD else None

    return final_idx, final_sim


def main():
    total_start_time = time.time()
    print("===== (mut1 only） =====")
    if not os.path.exists(Config.INPUT_CSV_PATH):
        print(f"Error: Input CSV  {Config.INPUT_CSV_PATH} not found")
        return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_csv_path = os.path.join(Config.OUTPUT_DIR, "sim0.8_mut1_bug.csv")

    processed_count = 0
    error_count = 0
    skip_count = 0
    no_match_count = 0  

    with open(Config.INPUT_CSV_PATH, 'r', encoding='utf-8', newline='') as infile, \
            open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)

        if set(reader.fieldnames) != set(Config.CSV_HEADER):
            print(f"Warning: CSV header mismatch!")
            print(f"Expected header:{Config.CSV_HEADER}")
            print(f"Actual header: {reader.fieldnames}")

        writer_fieldnames = reader.fieldnames + ["mut1_idx", "mut1_word2wordSim"]
        writer = csv.DictWriter(outfile, fieldnames=writer_fieldnames)
        writer.writeheader()

        for row_idx, row in enumerate(reader, start=1):
            try:
                poly_sense = row["poly_sense"].strip() if "poly_sense" in row else ""
                mut1_ch_word = row["mut1_ch_word"].strip() if "mut1_ch_word" in row else ""

                result_row = row.copy()
                if not poly_sense or not mut1_ch_word:
                    result_row["mut1_idx"] = ""
                    result_row["mut1_word2wordSim"] = ""
                    skip_count += 1
                    writer.writerow(result_row)
                    continue

                matched_idx, word2word_sim = get_idx(poly_sense, mut1_ch_word)

                result_row["mut1_idx"] = str(matched_idx) if matched_idx is not None else ""
                result_row["mut1_word2wordSim"] = round(word2word_sim, 4) if word2word_sim else ""

                writer.writerow(result_row)
                processed_count += 1

                if (processed_count + skip_count + no_match_count) % 5000 == 0:
                    print(
                        f"Processed {processed_count + skip_count + no_match_count} line（success:{processed_count}， skip:{skip_count}）")

            except Exception as e:
                error_count += 1
                result_row = row.copy()
                result_row["mut1_idx"] = ""
                result_row["mut1_word2wordSim"] = ""
                writer.writerow(result_row)
                print(f"Processing error on line {row_idx}：{str(e)[:100]}")

    total_time = time.time() - total_start_time
    minutes = total_time / 60
    hours = minutes / 60
    total_rows = processed_count + skip_count + no_match_count + error_count
    print(f"Total data rows:{total_rows}")
    print(f"number of successful lines:{processed_count}")
    print(f"Skip line count (field is empty):{skip_count}")
    print(f"number of error lines: {error_count}")
    print(f"Save the results to: {output_csv_path}")

    print(f"\nOverall execution time:")
    print(f"   - Total seconds: {total_time:.2f} seconds")
    print(f"   -  Total minutes:{minutes:.2f} minutes")
    if total_rows > 0:
        print(f"   -  Average time per line：{total_time / total_rows:.4f} seconds per line")

if __name__ == "__main__":
    main()
