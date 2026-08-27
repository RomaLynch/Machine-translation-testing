import os
import json
import re
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# After execution, execute
USAGE_LOCK = threading.Lock()

USAGE_FILE = Path("cache/youdao_usage.json")
USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

APP_KEY = ''
APP_SECRET = ''

def _load_usage():
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("current_hour", ""), data.get("used_this_hour", 0)
        except Exception:
            pass
    return "", 0

def _save_usage(current_hour: str, used: int):
    with open(USAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"current_hour": current_hour, "used_this_hour": used}, f)

def _get_current_hour():
    return datetime.now().strftime("%Y-%m-%d-%H")

def truncate(q):
    if q is None:
        return ""
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

def addAuthParams(appKey, appSecret, params):
    salt = str(uuid.uuid4())
    curtime = str(int(time.time()))
    q = params.get('q', '')
    signStr = appKey + truncate(q) + salt + curtime + appSecret
    sign = hashlib.sha256(signStr.encode('utf-8')).hexdigest()
    params['appKey'] = appKey
    params['salt'] = salt
    params['curtime'] = curtime
    params['signType'] = 'v3'
    params['sign'] = sign

def translate_one_with_quota(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    clean_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', t)
    if len(clean_text) > 5000:
        clean_text = clean_text[:5000]

    char_count = len(clean_text)  

    with USAGE_LOCK:
        current_hour, used_this_hour = _load_usage()
        now_hour = _get_current_hour()

        if now_hour != current_hour:
            print(f"[Youdao] coming new calculate {now_hour}£¬reset")
            current_hour = now_hour
            used_this_hour = 0

        if used_this_hour + char_count > 4_950_000:
            now = datetime.now()
            next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            wait_sec = (next_hour - now).total_seconds()
            if wait_sec > 0:
                print(f"\n[Youdao] Quota is about to exceed limit, sleep {wait_sec:.1f} seconds to {next_hour.strftime('%H:%M')} ...\n")

                USAGE_LOCK.release()
                time.sleep(wait_sec + 2)
                USAGE_LOCK.acquire()  

                current_hour, used_this_hour = _load_usage()
                now_hour = _get_current_hour()
                if now_hour != current_hour:
                    current_hour = now_hour
                    used_this_hour = 0

        used_this_hour += char_count
        _save_usage(current_hour, used_this_hour)

    data = {'q': clean_text, 'from': 'en', 'to': 'zh-CHS'}
    addAuthParams(APP_KEY, APP_SECRET, data)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        res = requests.post('https://openapi.youdao.com/api', data=data, headers=headers, timeout=15)
        res.raise_for_status()
        result = res.json()
        if result.get('errorCode') == '0':
            trans = result.get('translation', [clean_text])
            return trans[0] if trans else clean_text
        else:
            print(f"[Youdao Error] code={result.get('errorCode')}")
            return clean_text
    except Exception as e:
        print(f"[Youdao Exception] {str(e)[:100]}")
        return clean_text

def translate_text_with_youdao_batch(texts: List[str], n_jobs: int = 16) -> List[str]:
    print(f"A total of  {len(texts)} sentences£¬n_jobs={n_jobs}")
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        future_to_index = {
            executor.submit(translate_one_with_quota, text): i
            for i, text in enumerate(texts)
        }

        for future in tqdm(as_completed(future_to_index), total=len(texts), desc="Translation progress"):
            i = future_to_index[future]
            try:
                results[i] = future.result(timeout=30)
            except Exception as e:
                print(f"\n[ERROR] The translation of sentence {i}failed, revert back to the original text: {e}")
                results[i] = texts[i]

    return results


def batch_translate_with_cache(sentences: List[str], cache_file: str = "cache/translation_cache.json",
                               n_jobs: int = 16) -> List[str]:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    results = [None] * len(sentences)
    to_translate = []
    indices = []

    for i, sent in enumerate(sentences):
        key = hashlib.md5(sent.encode('utf-8')).hexdigest()
        if key in cache:
            results[i] = cache[key]
        else:
            to_translate.append(sent)
            indices.append(i)

    if to_translate:
        print(f"Translate {len(to_translate)} new sentences")
        translated = translate_text_with_youdao_batch(to_translate, n_jobs=n_jobs)
        for idx, orig, trans in zip(indices, to_translate, translated):
            key = hashlib.md5(orig.encode('utf-8')).hexdigest()
            cache[key] = trans
            results[idx] = trans
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    for i in range(len(results)):
        if results[i] is None:
            results[i] = sentences[i] or ""

    return results

def translate_txt_file(
    input_path: str,
    output_path: str,
    cache_file: str = "cache/translation_cache.json",
    n_jobs: int = 16
):

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        en_sentences = [line.rstrip('\n') for line in f]

    print(f"Load {len(en_sentences)} rows of English sentences together")

    zh_translations = batch_translate_with_cache(
        sentences=en_sentences,
        cache_file=cache_file,
        n_jobs=n_jobs
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for zh in zh_translations:
            f.write(zh + '\n')

    print(f"Translation completed! The result has been saved to: {output_path}")

if __name__ == "__main__":

    INPUT_TXT = "3w_um_src.txt"      
    OUTPUT_TXT = "3w_um_zh.txt"    

    translate_txt_file(
        input_path=INPUT_TXT,
        output_path=OUTPUT_TXT,
        cache_file="cache/translation_cache.json",
        n_jobs=16
    )