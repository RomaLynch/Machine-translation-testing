import re
import os

# After execution, execute align.py
def clean_text_light(sentence: str) -> str:
  
    if not sentence:
        return ""
    cleaned = re.sub(r'[\n\r\t]', ' ', sentence)
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()

    return cleaned

def clean_file(input_filepath: str, output_filepath: str) -> None:
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"input file not found£º{input_filepath}")

    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
            open(output_filepath, 'w', encoding='utf-8') as f_out:

        line_count = 0
        for line in f_in:
            cleaned_line = clean_text_light(line)
            f_out.write(cleaned_line + '\n')
            line_count += 1


if __name__ == "__main__":
    RAW_EN_FILE = r"baidu\corpus\3w_um_src_cleaned.txt"
    RAW_ZH_FILE = r"baidu\corpus\3w_um_zh_new_cleaned.txt"

    CLEANED_EN_FILE = r"baidu\align_cleaned_corpus\cleaned_3w_um_src.txt"
    CLEANED_ZH_FILE = r"baidu\align_cleaned_corpus\cleaned_3w_um_zh.txt"

    clean_file(RAW_EN_FILE, CLEANED_EN_FILE)

    clean_file(RAW_ZH_FILE, CLEANED_ZH_FILE)

