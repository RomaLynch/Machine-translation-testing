import csv
from pathlib import Path
from typing import List

INPUT_CSV = "mut_translation_results/success_mutations.csv" 
# INPUT_TRANSLATE_TXT = "mut_translation_results/mut1_zh_pairs.txt"
# INPUT_TRANSLATE_TXT = "mut_translation_results/mut2_zh.txt"
INPUT_TRANSLATE_TXT = "mut_translation_results/mut3_zh.txt"
OUTPUT_CSV = "mut_translation_results/success_mutations.csv"
# NEW_COLUMN_NAME = "Mut1_ch"
# NEW_COLUMN_NAME = "Mut2_ch"
NEW_COLUMN_NAME = "Mut3_ch"
ENCODING = "utf-8" 

# After execution, execute clean_mut1_Zh.py clean_mut2_Zh.py clean_mut3_Zh.py

def read_translation_txt(txt_path: str) -> List[str]:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Translation TXT file does not exist:{txt_path}")

    with open(txt_path, 'r', encoding=ENCODING) as f:
        translations = [line.rstrip('\n') for line in f]

    print(f"Successfully read translation TXT: a total of {len(translations)}")
    return translations


def merge_translation_to_csv(
        csv_path: str,
        translations: List[str],
        output_csv: str,
        new_col_name: str
):
    csv_path = Path(csv_path)
    output_csv = Path(output_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"The original CSV file does not exist: {csv_path}")

    with open(csv_path, 'r', encoding=ENCODING, newline='') as f:
        reader = csv.DictReader(f)
        original_headers = reader.fieldnames
        csv_rows = [row for row in reader]  

    csv_row_count = len(csv_rows)
    trans_row_count = len(translations)
    if csv_row_count != trans_row_count:
        raise ValueError(
            f"The number of rows does not match! CSV has  {csv_row_count} rows, while TXT has {trans_row_count} rows for translation"
        )

    new_headers = original_headers + [new_col_name]

    for idx, row in enumerate(csv_rows):
        row[new_col_name] = translations[idx]  

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', encoding=ENCODING, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Merge completed! Add column {new_col_name} and save the result to {output_csv}")


if __name__ == "__main__":
    try:
        translations = read_translation_txt(INPUT_TRANSLATE_TXT)

        merge_translation_to_csv(
            csv_path=INPUT_CSV,
            translations=translations,
            output_csv=OUTPUT_CSV,
            new_col_name=NEW_COLUMN_NAME
        )
    except Exception as e:
        print(f"{str(e)}")
