import csv
import os

# After executing this py, execute add_MutTransToCSV.py
def filter_success_rows(input_csv: str, output_csv: str):
   
    if not os.path.exists(input_csv):
        print(f"Error: The input file {input_csv} does not exist")
        return

    success_count = 0
    total_count = 0

    with open(input_csv, 'r', encoding='utf-8', newline='') as infile, \
            open(output_csv, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        writer.writeheader()

        for row in reader:
            total_count += 1
            if row.get("status", "").strip() == "success":
                writer.writerow(row)
                success_count += 1

    print(f"Total number:{total_count}")
    print(f"Success lines:{success_count}")
    print(f" Filter results saved to:{output_csv}")


INPUT_CSV = "mutation_results/mutations.csv"  
OUTPUT_CSV = "mut_translation_results/success_mutations.csv" 

if __name__ == "__main__":
    filter_success_rows(INPUT_CSV, OUTPUT_CSV)
