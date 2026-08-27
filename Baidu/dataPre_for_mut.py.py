import csv
import json
import os

# After execution, execute dataPre2_for_mut.exe
def read_lines_keep_empty(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.rstrip('\n\r') for line in lines]


def add_columns_to_align_csv(
        align_csv_path: str,
        en_txt_path: str,
        zh_txt_path: str,
        poly_txt_path: str,
        output_csv_path: str
):
    print("Read the original text file")
    en_sents = read_lines_keep_empty(en_txt_path)
    zh_sents = read_lines_keep_empty(zh_txt_path)
    poly_words = read_lines_keep_empty(poly_txt_path)

    print(f"Read alignment result CSV: {align_csv_math}")
    align_rows = []
    with open(align_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_headers = ["ch_word", "en_tokens", "zh_tokens", "alignment_pairs"]
        if not all(h in reader.fieldnames for h in expected_headers):
            raise ValueError(f"CSV header does not meet expectations! Expected to include: {expected-headers}, actually includes: {reader. fieldnames}")

        for row in reader:
            align_rows.append(row)

    total_align = len(align_rows)
    total_en = len(en_sents)
    total_zh = len(zh_sents)
    total_poly = len(poly_words)
    if not (total_align == total_en == total_zh == total_poly):
        raise ValueError(
            f"The number of rows is inconsistent \n"
            f"Align CSV rows: {total_alignment} \n"
            f"Number of English text lines: {total_en\n}"
            f"Number of Chinese text lines: {total_zh}\n"
            f"Target word text line count£º{total_poly}"
        )
    print(f"Verified: All file lines are {total_align}")

    print(f"Write CSV after adding new columns£º{output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        new_headers = ["en_sent", "ch_sent", "polyword", "ch_word", "en_tokens", "zh_tokens", "alignment_pairs"]
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        for i in range(total_align):
            new_row = {
                "en_sent": en_sents[i],
                "ch_sent": zh_sents[i],
                "polyword": poly_words[i],
                "ch_word": align_rows[i]["ch_word"],
                "en_tokens": align_rows[i]["en_tokens"],
                "zh_tokens": align_rows[i]["zh_tokens"],
                "alignment_pairs": align_rows[i]["alignment_pairs"]
            }
            writer.writerow(new_row)

            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1} / {total_align} rows")

    print(f"New column added. Save results to£º{output_csv_path}")


if __name__ == "__main__":
    ALIGN_CSV_PATH = "aligned_results/align_res.csv"

    EN_TXT_PATH = "align_cleaned_corpus/cleaned_3w_um_src.txt"
    ZH_TXT_PATH = "align_cleaned_corpus/cleaned_3w_um_zh.txt"
    POLY_TXT_PATH = "corpus/tgt_word_cleaned.txt"

    OUTPUT_CSV_PATH = "mutation_csvFile/data4mut.csv"

    add_columns_to_align_csv(
        align_csv_path=ALIGN_CSV_PATH,
        en_txt_path=EN_TXT_PATH,
        zh_txt_path=ZH_TXT_PATH,
        poly_txt_path=POLY_TXT_PATH,
        output_csv_path=OUTPUT_CSV_PATH
    )
