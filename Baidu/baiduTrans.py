# -*- coding: utf-8 -*-
import requests
import random
import re
import time
import threading
import queue
from hashlib import md5
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed

appid = ''
appkey = ''
from_lang = 'en'
to_lang = 'zh'
endpoint = 'https://api.fanyi.baidu.com'
path = ''
url = endpoint + path


QPS_LIMIT = 10  
REQUEST_INTERVAL = 1.0 / QPS_LIMIT  
SAFE_BUFFER = 0.0 
THREAD_NUM = 10  
PRINT_INTERVAL = 1000

request_counter = 0
qps_lock = threading.Lock()
last_request_time = time.time()
counter_lock = threading.Lock()

def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def createRequest(query):
    global request_counter, last_request_time

    with qps_lock:
        current_time = time.time()
        sleep_time = REQUEST_INTERVAL - (current_time - last_request_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_request_time = time.time()

    q = query.strip()
    if not q:
        print("[Warning] Empty text, skip translation")
        return ""
    if len(q) > 6000:
        q = q[:6000]
        print(f"[Warning] Text is too long, truncated to 6000 characters: {q [: 50]}")
    q = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', q)

    try:
        salt = random.randint(32768, 65536)
        sign = make_md5(appid + q + str(salt) + appkey)
        payload = {'appid': appid, 'q': q, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        r = requests.post(url, data=payload, headers=headers, timeout=10)
        r.raise_for_status()
        result = r.json()

        if 'error_code' in result:
            error_msg = {
                '52001': '请求超时', '52002': '系统错误', '52003': '未授权用户',
                '54000': '必填参数为空', '54001': '签名错误', '54003': '访问频率受限',
                '54004': '账户余额不足', '54005': '长query请求频繁', '58000': '客户端IP非法',
                '58001': '译文语言方向不支持', '58002': '服务当前已关闭'
            }.get(result['error_code'], f'错误码{result["error_code"]}')
            print(f"[Baidu API Error] {correr_msg} | Original text: {q [: 50]}")
            return q

        translation = [item['dst'] for item in result.get('trans_result', [])]
        trans_text = ''.join(translation) if translation else q

        with counter_lock:
            request_counter += 1
            if request_counter % PRINT_INTERVAL == 0:
                print(f"[Progress] Completed {request_comunter} translation requests")

        return trans_text

    except Exception as e:
        print(f"[trans errors] {str(e)} | details：{q[:50]}...")
        return q


def translate_worker(task_queue: queue.Queue, result_dict: dict):
    while not task_queue.empty():
        try:
            idx, sentence = task_queue.get()
            trans_text = createRequest(sentence)
            result_dict[idx] = trans_text
            task_queue.task_done()
        except Exception as e:
            print(f"{str(e)}")
            task_queue.task_done()


def main():
    input_path = Path(INPUT_TXT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"input failed :{INPUT_TXT_PATH}")

    with open(input_path, 'r', encoding='utf-8') as f:
        en_sentences = [line.rstrip('\n') for line in f]
    total_sentences = len(en_sentences)

    task_queue = queue.Queue()
    result_dict = {}
    for idx, sent in enumerate(en_sentences):
        task_queue.put((idx, sent))

    start_time = time.time()
    threads = []
    for i in range(THREAD_NUM):
        t = threading.Thread(target=translate_worker, args=(task_queue, result_dict))
        t.start()
        threads.append(t)

    task_queue.join()
    for t in threads:
        t.join()
    end_time = time.time()

    cn_sentences = [result_dict.get(i, "") for i in range(total_sentences)]

    with open(OUTPUT_CN_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cn_sentences))

    mix_lines = []
    for en, cn in zip(en_sentences, cn_sentences):
        mix_lines.append(en)
        mix_lines.append("")
        mix_lines.append(cn)
        mix_lines.append("")
    with open(OUTPUT_MIX_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mix_lines))

    total_time = end_time - start_time
    fail_num = sum(1 for cn, en in zip(cn_sentences, en_sentences) if cn == en or cn == "")
    success_num = total_sentences - fail_num
    print(f"sentence count：{total_sentences}")
    print(f"trans succeed：{success_num}")
    print(f"trans failed：{fail_num}")
    print(f"total time：{total_time:.2f} seconds")
    print(f"ave.speed：{total_sentences / total_time:.2f} lines/seconds")
    print(f"succeed rate：{success_num / total_sentences * 100:.2f}%")


if __name__ == "__main__":
    try:
        from tenacity import retry, stop_after_attempt, wait_fixed
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "tenacity"])
        from tenacity import retry, stop_after_attempt, wait_fixed

    INPUT_TXT_PATH = 'corpus/3w_um_src.txt'
    OUTPUT_CN_TXT_PATH = 'corpus/del/3w_um_zh.txt'
    OUTPUT_MIX_TXT_PATH = 'corpus/3w_um_pairs.txt'

    main()