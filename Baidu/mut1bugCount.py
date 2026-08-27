import csv
import os
from pathlib import Path


def compare_and_generate_csv(
        input_csv_path: str,
        output_all_csv_path: str,
        output_diff_csv_path: str
) -> None:
    
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"file not found£º{input_csv_path}")

    total_non_empty_rows = 0  
    different_count = 0  

    all_valid_data = []  
    diff_data = []  
    core_cols = ["idx", "mut1_idx", "ch_word", "mut1_ch_word"]

    with open(input_path, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        all_original_cols = reader.fieldnames
        missing_core_cols = [col for col in core_cols if col not in all_original_cols]
        if missing_core_cols:
            raise ValueError(f"The input CSV is missing a core column:{missing_core_cols}")

        remaining_cols = [col for col in all_original_cols if col not in core_cols]
        output_header = core_cols + remaining_cols

        for row_idx, row in enumerate(reader, start=1):
            sense_idx = row["idx"].strip()
            mut1_idx = row["mut1_idx"].strip()  
            ch_word = row["ch_word"].strip()
            mut1_ch_word = row["mut1_ch_word"].strip()

            if not mut1_idx:
                continue

            total_non_empty_rows += 1

            current_row = {}
            current_row["idx"] = sense_idx
            current_row["mut1_idx"] = mut1_idx
            current_row["ch_word"] = ch_word
            current_row["mut1_ch_word"] = mut1_ch_word
            for col in remaining_cols:
                current_row[col] = row[col]
            all_valid_data.append(current_row)

            if sense_idx != mut1_idx:
                different_count += 1
                diff_data.append(current_row)

    output_all_path = Path(output_all_csv_path)
    output_all_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_all_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_header)
        writer.writeheader()
        writer.writerows(all_valid_data)

    output_diff_path = Path(output_diff_csv_path)
    output_diff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_diff_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_header)
        writer.writeheader()
        writer.writerows(diff_data)

    print("====Mut1 Data Comparison and Statistical Results=====")
    print(f"Total rows excluding mut1_idx null values: {total_non_empty_rows}")
    print(f"The number of times sense_idx is different from mut1_idx: {different_count}")
    if total_non_empty_rows > 0:
        different_rate = (different_count / total_non_empty_rows) * 100
        print(f"difference rate: {different_rate:.2f}%")
    else:
        print(f"Variance rate 0% (no valid data)")

    print(f"\nMut1 full effective row CSV has been generated: {output_all_csv_path}")
    print(f"   contains columns: {len(output_header)} | contains rows:{len(all_valid_data)}")
    print(f"\nMut1 only difference row CSV has been generated: {output_diff_csv_path}")
    print(f"   contains columns: {len(output_header)} | contains rows:{len(diff_data)}")
    print(f"\nOutput column order:")
    print(f"   Core column (front): £º{core_cols}")
    print(f"   Remaining Column (Post):{remaining_cols}")


if __name__ == "__main__":
    # INPUT_CSV = "bugDetectionData/bug/sim0.9_mut1_bug.csv"  
    # OUTPUT_ALL_CSV = "bugDetectionData/bug/mut1/sim0.9_bugCompare.csv"  
    # OUTPUT_DIFF_CSV = "bugDetectionData/bug/mut1/sim0.9_bugCount.csv"  

    INPUT_CSV = "Threshold analysis/sim0.95_mut1_bug.csv"  
    OUTPUT_ALL_CSV = "Threshold analysis/bug/mut1/sim0.95_bugCompare.csv"  
    OUTPUT_DIFF_CSV = "Threshold analysis/bug/mut1/sim0.95_bugCount.csv"  
    # INPUT_CSV = "bugDetectionData/bug/sim0.9_mut2_bug.csv"  
    # OUTPUT_ALL_CSV = "bugDetectionData/bug/mut2/sim0.9_bugCompare.csv"  
    # OUTPUT_DIFF_CSV = "bugDetectionData/bug/mut2/sim0.9_bugCount.csv"  
    #
    # INPUT_CSV = "bugDetectionData/bug/sim0.9_mut3_bug.csv"  
    # OUTPUT_ALL_CSV = "bugDetectionData/bug/mut3/sim0.9_bugCompare.csv"  
    # OUTPUT_DIFF_CSV = "bugDetectionData/bug/mut3/sim0.9_bugCount.csv"  

    try:
        compare_and_generate_csv(
            input_csv_path=INPUT_CSV,
            output_all_csv_path=OUTPUT_ALL_CSV,
            output_diff_csv_path=OUTPUT_DIFF_CSV
        )
    except Exception as e:
        print(f"{e}")
        import traceback
        traceback.print_exc()
