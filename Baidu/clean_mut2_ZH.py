import re
import os
import json
import csv
from pathlib import Path

# After execution, execute mut_Zh_Align.py
def clean_gloss(gloss: str) -> str:
    if not isinstance(gloss, str):
        return ""

    text = gloss.strip()
    bracket_pairs = [('(', ')'), ('（', '）')]

    for left, right in bracket_pairs:
        
        pattern = re.escape(left) + r'[^\s' + re.escape(left) + re.escape(right) + r']*' + re.escape(right)
        while re.search(pattern, text):
            text = re.sub(pattern, '', text)

    text = re.sub(r'[()（）]', '', text)

    text = re.sub(r'[.。!！?？;；:：,，\s]+$', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    if text:
        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()

    return text


def clean_text_light(sentence: str) -> str:
    if not sentence:
        return ""

    cleaned = re.sub(r'[\n\r\t]', ' ', sentence)

    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)

    cleaned = re.sub(r'\s+', ' ', cleaned)

    cleaned = cleaned.strip()

    return cleaned


def extract_brackets_from_text(text: str) -> list:
    if not text or not isinstance(text, str):
        return []


    zh_brackets = re.findall(r'（[^）]+）', text)

    en_brackets = re.findall(r'\([^)]+\)', text)

    all_brackets = list(set(zh_brackets + en_brackets))
    return all_brackets


def clean_mut2_ch_with_keep_original_brackets(mut2_text: str, keep_brackets: list) -> str:

    if not mut2_text:
        return ""

    cleaned_mut = clean_text_light(mut2_text)

    bracket_placeholder = {}
    valid_keep_brackets = [b for b in keep_brackets if b and isinstance(b, str)]
    for idx, bracket in enumerate(valid_keep_brackets):
        placeholder = f"__BRACKET_{idx}__"
        bracket_placeholder[placeholder] = bracket
        cleaned_mut = cleaned_mut.replace(bracket, placeholder)

    cleaned_mut = re.sub(r'（[^）]+）', '', cleaned_mut)

    cleaned_mut = re.sub(r'\([^)]+\)', '', cleaned_mut)

    cleaned_mut = re.sub(r'[()（）]', '', cleaned_mut)

    for placeholder, bracket in bracket_placeholder.items():
        cleaned_mut = cleaned_mut.replace(placeholder, bracket)

    final_cleaned = clean_text_light(cleaned_mut)

    return final_cleaned


def extract_column_from_csv(csv_path: str, output_txt_path: str, column_name: str) -> None:
    csv_path = Path(csv_path)
    output_txt_path = Path(output_txt_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    column_values = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f: 
        reader = csv.DictReader(f)
        if column_name not in reader.fieldnames:
            raise ValueError(f"CSV not found {column_name}，available：{reader.fieldnames}")

        for row in reader:
            col_value = row[column_name].strip()
            column_values.append(col_value)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for value in column_values:
            f.write(value + '\n')


def clean_file(input_filepath: str, output_filepath: str, use_gloss_clean: bool = False) -> None:
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"file not exist：{input_filepath}")

    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
            open(output_filepath, 'w', encoding='utf-8') as f_out:

        line_count = 0
        for line in f_in:
            if use_gloss_clean:
                line = clean_gloss(line)
            cleaned_line = clean_text_light(line)
            f_out.write(cleaned_line + '\n')
            line_count += 1

def clean_mut2_ch_from_csv(csv_path: str, raw_zh_file: str, output_zh_file: str) -> list:

    csv_path = Path(csv_path)
    raw_zh_path = Path(raw_zh_file)
    output_zh_path = Path(output_zh_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not exist：{csv_path}")
    if not raw_zh_path.exists():
        raise FileNotFoundError(f"mut2 original file not exist：{raw_zh_path}")

    output_zh_path.parent.mkdir(parents=True, exist_ok=True)

    original_brackets_list = []  
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required_cols = ["ch_sent", "en_sent"]
        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"CSV not found {column_name}，available：{reader.fieldnames}")

        for row in reader:
           
            ch_sent = row["ch_sent"].strip() if row["ch_sent"] else ""
            ch_brackets = extract_brackets_from_text(ch_sent)

            en_sent = row["en_sent"].strip() if row["en_sent"] else ""
            en_brackets = extract_brackets_from_text(en_sent)

     
            all_original_brackets = list(set(ch_brackets + en_brackets))
            original_brackets_list.append(all_original_brackets)


    mut2_lines = []
    with open(raw_zh_path, 'r', encoding='utf-8') as f:
        mut2_lines = [line.strip() if line else "" for line in f]


    if len(mut2_lines) != len(original_brackets_list):
        raise ValueError(f"The number of rows does not match! The CSV file has{len(original_brackets_list)}lines, while the mut2 file has{len(mut2_lines)}lines")

    cleaned_lines = []
    for idx, (mut_line, keep_brackets) in enumerate(zip(mut2_lines, original_brackets_list)):
        cleaned_line = clean_mut2_ch_with_keep_original_brackets(mut_line, keep_brackets)
        cleaned_lines.append(cleaned_line)

    with open(output_zh_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')


    return cleaned_lines


def read_cleaned_zh_txt(txt_path: str) -> list:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"The cleaned mut2 file does not exist：{txt_path}")

    zh_sentences = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            zh_sent = line.strip()
            zh_sentences.append(zh_sent)
    return zh_sentences


def add_mut2_to_csv(csv_path: str, zh_sentences: list, new_column: str = "cleaned_mut2_ch",
                  output_csv: str = None) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"The original CSV file does not exist：{csv_path}")
    if output_csv is None:
        output_csv = csv_path.parent / f"{csv_path.stem}_with_mut2{csv_path.suffix}"
    else:
        output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    header = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames.copy()
        for row in reader:
            csv_rows.append(row)
    csv_row_count = len(csv_rows)
    zh_row_count = len(zh_sentences)
    if csv_row_count != zh_row_count:
        raise ValueError(f"The number of rows does not match! CSV has {csv_row_count} rows, while mut2 TXT has {zh_row_count} rows")

    if new_column not in header:
        header.append(new_column)

    for idx, row in enumerate(csv_rows):
        row[new_column] = zh_sentences[idx]

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)



if __name__ == "__main__":
    CSV_FILE = "mut_translation_results/success_mutations.csv"
    RAW_MUT2_ZH_FILE = "mut_translation_results/mut2_zh.txt"  
    EXTRACTED_EN_TXT = "bugDetectionData/src_sent.txt"
    EXTRACTED_POLYWORD_TXT = "bugDetectionData/polywordInMut.txt"

    CLEANED_EN_FILE = "bugDetectionData/mut_src_sent.txt"
    CLEANED_MUT2_ZH_FILE = "bugDetectionData/cleaned_mut2_ch_sent.txt"  
    OUTPUT_CSV_WITH_MUT2 = "mut_translation_results/success_mutations.csv"

    NEW_MUT2_COLUMN = "cleaned_mut2_ch"

    cleaned_mut2_sentences = clean_mut2_ch_from_csv(
        csv_path=CSV_FILE,
        raw_zh_file=RAW_MUT2_ZH_FILE,
        output_zh_file=CLEANED_MUT2_ZH_FILE
    )

    add_mut2_to_csv(
        csv_path=CSV_FILE,
        zh_sentences=cleaned_mut2_sentences,
        new_column=NEW_MUT2_COLUMN,
        output_csv=OUTPUT_CSV_WITH_MUT2
    )
