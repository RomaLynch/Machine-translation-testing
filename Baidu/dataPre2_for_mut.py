import spacy
import pandas as pd
import os
from typing import Optional
import json
# After execution, execute mutation.py
SPACY_MODEL_PATH = 'en_core_web_sm'
POLYSEMY_JSON_PATH = "/sense_dicts.json"
INPUT_CSV_PATH = "mutation_csvFile/data4mut.csv"
OUTPUT_CSV_PATH = "mutation_csvFile/data4mut.csv"


def init_spacy(model_path: str):
    try:
        nlp = spacy.load(model_path, disable=['parser', 'textcat', 'ner'])
        print(f"SpaCy model loaded successfully£º{model_path}")
        return nlp
    except Exception as e:
        raise FileNotFoundError(f"SpaCy model loading failed£º{e}")

def load_polysemy_map(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The polysemous word library file does not exist£º{json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        polysemy_map = json.load(f)
    print(f"The polysemous word library has been successfully loaded, with a total of {len(polysemy_map)} words")
    return polysemy_map


def normalize_word_key(word: str, nlp, polysemy_map: dict) -> str:
    if not word or not isinstance(word, str):
        return ""

    clean_word = word.strip().strip("'\"`").strip()
    if not clean_word:
        return ""

    word_lower = clean_word.lower()
    doc = nlp(word_lower) 
    word_lemma = ""
    if doc:
        token = doc[0]
        word_lemma = token.lemma_.lower()
        if word_lemma.startswith("-") and len(clean_word) == 1:
            word_lemma = word_lower
    if not word_lemma:
        word_lemma = word_lower

    if word_lemma in polysemy_map:
        return word_lemma
    elif word_lower in polysemy_map:
        return word_lower
    else:
        return ""

def process_csv(
        input_path: str,
        output_path: str,
        nlp,
        polysemy_map: dict
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CSV input does not exist£º{input_path}")
    print(f"Read input CSV£º{input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='utf-8-sig')

    if "polyword" not in df.columns:
        raise ValueError(f"{df.columns.tolist()}")
    df["poly_sense"] = df["polyword"].apply(lambda x: normalize_word_key(x, nlp, polysemy_map))

    total_rows = len(df)
    empty_polyword = df["polyword"].isna() | df["polyword"].str.strip().eq("")
    non_empty_polyword = total_rows - empty_polyword.sum()
    matched_rows = df["poly_sense"].str.strip().ne("").sum()
    unmatched_rows = non_empty_polyword - matched_rows

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    nlp = init_spacy(SPACY_MODEL_PATH)
    polysemy_map = load_polysemy_map(POLYSEMY_JSON_PATH)

    process_csv(
        input_path=INPUT_CSV_PATH,
        output_path=OUTPUT_CSV_PATH,
        nlp=nlp,
        polysemy_map=polysemy_map
    )
