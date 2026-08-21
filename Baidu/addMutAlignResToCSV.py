# Execute bugDetection.py after completion
import csv
import os
from pathlib import Path


def add_ch_word_column(
        align_csv_path: str,
        target_csv_path: str,
        output_csv_path: str,
        source_col: str = "ch_word",
        new_col: str = "mut3_ch_word"
) -> None:
    align_path = Path(align_csv_path)
    target_path = Path(target_csv_path)
    if not align_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {align_csv_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file does not exist: {target_csv_path}")


    ch_word_list = []
    with open(align_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        if source_col not in reader.fieldnames:
            raise ValueError(f"Column name{source_col}not found in source CSV, available column names: {reader.fieldnames}")

        for row in reader:
            ch_word = row[source_col]
            ch_word_list.append(ch_word)

    print(f"Read source CSV completed: Extract {len(ch_word_list)}rows of chw_ord data (including null values)")

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, 'r', encoding='utf-8', newline='') as in_f, \
            open(output_path, 'w', encoding='utf-8', newline='') as out_f:

        reader = csv.DictReader(in_f)
        new_fieldnames = reader.fieldnames + [new_col]
        writer = csv.DictWriter(out_f, fieldnames=new_fieldnames)
        writer.writeheader()

        row_count = 0
        for row_idx, row in enumerate(reader):
            new_row = row.copy()
            if row_idx < len(ch_word_list):
                new_row[new_col] = ch_word_list[row_idx]
            else:
                new_row[new_col] = ""
            writer.writerow(new_row)
            row_count += 1

        if row_count != len(ch_word_list):
            print(f"Warning: The number of rows does not match! The target CSV has {row_count}  rows, and the source CSV has{len(ch_word_list)} rows")
            print(f"   Description: Insufficient rows have been filled with null values, excess source data has not been used")
        else:
            print(f"Row number matching: Processing  {row_count}rows, all ch-words (including null values) have been added")


if __name__ == "__main__":
    ALIGN_CSV = "bugDetectionData/mut_align_results/mut3_align.csv" 
    TARGET_CSV = "mut_translation_results/success_mutations.csv" 
    OUTPUT_CSV = "bugDetectionData/bug/mut3_data4Detection.csv" 

    try:
        add_ch_word_column(
            align_csv_path=ALIGN_CSV,
            target_csv_path=TARGET_CSV,
            output_csv_path=OUTPUT_CSV,
            source_col="ch_word",
            new_col="mut3_ch_word"
        )
    except Exception as e:
        print(f"{e}")
