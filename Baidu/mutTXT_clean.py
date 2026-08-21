import os

# After execution, execute mut_maidu_trans.py
def filter_long_sentences_and_count(input_txt, output_txt, max_chars=6000):
    if not os.path.exists(input_txt):
        print(f"Error: The input file {input_txt} does not exist")
        return 0, 0

    kept_count = 0  
    deleted_count = 0 

    with open(input_txt, 'r', encoding='utf-8') as infile, \
            open(output_txt, 'w', encoding='utf-8') as outfile:

        for line in infile:
            sentence = line.rstrip('\n')
            char_count = len(sentence)

            if char_count <= max_chars:
                outfile.write(sentence + '\n')
                kept_count += 1
            else:
                deleted_count += 1

    return kept_count, deleted_count


def main():
    base_dir = "mutation_txt"
    input_files = {
        "mut1.txt": "mut1_cleaned.txt",
        "mut2.txt": "mut2_cleaned.txt",
        "mut3.txt": "mut3_cleaned.txt"
    }
    max_char_limit = 6000  

    os.makedirs(base_dir, exist_ok=True)

    print("=" * 60)
    print(f"Start filtering: Delete sentences with more than {max_char_limit} characters")
    print("=" * 60)

    for input_name, output_name in input_files.items():
        input_path = os.path.join(base_dir, input_name)
        output_path = os.path.join(base_dir, output_name)

        kept, deleted = filter_long_sentences_and_count(
            input_path, output_path, max_char_limit
        )
        total = kept + deleted

        print(f"\n File£º{input_name}")
        print(f"Original total sentence count£º{total}")
        print(f"Number of deleted sentences£º{deleted}£¨characters>6000£©")
        print(f"number of reserved sentences£º{kept}")
        print(f"filtered file£º{output_path}")

    print("\n" + "=" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()
