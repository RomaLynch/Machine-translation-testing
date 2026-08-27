# After execution, execute clean_before_align.py
def clean_parallel_files(en_file: str, zh_file: str, poly_file: str):
    with open(en_file, "r", encoding="utf-8") as f_en, \
         open(zh_file, "r", encoding="utf-8") as f_zh, \
         open(poly_file, "r", encoding="utf-8") as f_poly:
        en_lines = f_en.readlines()
        zh_lines = f_zh.readlines()
        poly_lines = f_poly.readlines()

    if not (len(en_lines) == len(zh_lines) == len(poly_lines)):
        raise ValueError(
            f"EN: {len(en_lines)}, ZH: {len(zh_lines)}, POLY: {len(poly_lines)}"
        )

    cleaned_en = []
    cleaned_zh = []
    cleaned_poly = []

    for i, (en_line, zh_line, poly_line) in enumerate(zip(en_lines, zh_lines, poly_lines)):
        zh_stripped = zh_line.rstrip('\n\r')

        if zh_stripped.strip(): 
            cleaned_en.append(en_line)
            cleaned_zh.append(zh_line)
            cleaned_poly.append(poly_line)
        else:
            en_preview = en_line.rstrip('\n\r')
            poly_preview = poly_line.rstrip('\n\r')
            
    def add_cleaned_suffix(filepath):
        if filepath.endswith(".txt"):
            return filepath[:-4] + "_cleaned.txt"
        else:
            return filepath + "_cleaned"

    en_cleaned = add_cleaned_suffix(en_file)
    zh_cleaned = add_cleaned_suffix(zh_file)
    poly_cleaned = add_cleaned_suffix(poly_file)

    with open(en_cleaned, "w", encoding="utf-8") as f:
        f.writelines(cleaned_en)
    with open(zh_cleaned, "w", encoding="utf-8") as f:
        f.writelines(cleaned_zh)
    with open(poly_cleaned, "w", encoding="utf-8") as f:
        f.writelines(cleaned_poly)

if __name__ == "__main__":
    EN_FILE   = r"baidu\corpus\3w_um_src.txt"
    ZH_FILE   = r"baidu\corpus\3w_um_zh_new.txt"
    POLY_FILE = r"baidu\corpus\tgt_word.txt"

    clean_parallel_files(EN_FILE, ZH_FILE, POLY_FILE)

