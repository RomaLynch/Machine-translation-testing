import csv
import os

# After execution, execute mutTXT_clean.exe

def extract_mut_to_txt_and_count_chars(input_csv):
    
    if not os.path.exists(input_csv):
        print(f"Error: The input file {input_csv} does not exist")
        return

    output_dir = "mutation_txt"
    os.makedirs(output_dir, exist_ok=True)
    mut1_path = os.path.join(output_dir, "mut1.txt")
    mut2_path = os.path.join(output_dir, "mut2.txt")
    mut3_path = os.path.join(output_dir, "mut3.txt")

    mut1_total_chars = 0
    mut2_total_chars = 0
    mut3_total_chars = 0
    line_count = 0

    with open(input_csv, 'r', encoding='utf-8', newline='') as csvfile, \
            open(mut1_path, 'w', encoding='utf-8') as f1, \
            open(mut2_path, 'w', encoding='utf-8') as f2, \
            open(mut3_path, 'w', encoding='utf-8') as f3:

        reader = csv.DictReader(csvfile)

        for row in reader:
            if row.get("status", "").strip() != "success":
                continue

            line_count += 1

            mut1 = row.get("mut1_sent", "").strip()
            mut2 = row.get("mut2_sent", "").strip()
            mut3 = row.get("mut3_sent", "").strip()

            f1.write(mut1 + "\n")
            f2.write(mut2 + "\n")
            f3.write(mut3 + "\n")

            mut1_total_chars += len(mut1)
            mut2_total_chars += len(mut2)
            mut3_total_chars += len(mut3)

    print("=" * 50)
    print(f"total number of lines£¨success£©£º{line_count}")
    print("-" * 30)
    print(f"mut1_sent.txt total number of characters£º{mut1_total_chars}")
    print(f"mut2_sent.txt total number of characters£º{mut2_total_chars}")
    print(f"mut3_sent.txt total number of characters£º{mut3_total_chars}")
    print("-" * 30)
    print(f"file path£º")
    print(f"- {mut1_path}")
    print(f"- {mut2_path}")
    print(f"- {mut3_path}")
    print("=" * 50)


INPUT_CSV = "mutation_results/mutations.csv"

if __name__ == "__main__":
    extract_mut_to_txt_and_count_chars(INPUT_CSV)
