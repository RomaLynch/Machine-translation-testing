# Execution Order
The full execution pipeline for the Baidu MT system is listed below (run scripts sequentially):
baiduTrans.py
→ trans_res_cleaned.py
→ clean_before_align.py
→ align.py
→ dataPre_for_mut.py & dataPre2_for_mut.py
→ mutation.py
→ gen_mutTXT4Trans.py
→ mutTXT_clean.py
→ mut_baidu_trans.py
→ extract_mutSuccess.py
→ add_MutTransToCSV.py
→ clean_mut1_ZH.py, clean_mut2_ZH.py, clean_mut3_ZH.py
→ mut_Zh_Align.py
→ addMutAlignResToCSV.py
→ mut1bugDetection.py
→ mut1bugCount.py

## Brief script‑wise function summary
1. `baiduTrans.py`: Call Baidu translation API to obtain initial translation results for raw test sentences
2. `trans_res_cleaned.py`: Clean raw API translation outputs
3. `clean_before_align.py`: Preprocess English‑Chinese sentence pairs for word alignment
4. `align.py`: Perform bilingual word‑level alignment using SimAlign
5. `dataPre_for_mut.py` / `dataPre2_for_mut.py`: Prepare dataset and gloss‑index mapping for subsequent mutation
6. `mutation.py`: Core mutation module, generate three kinds of mutated sentences (Mut1 / Mut2 / Mut3)
7. `gen_mutTXT4Trans.py`: Export mutated sentences into text files for MT inference
8. `mutTXT_clean.py`: Clean text format of mutated‑sentence files
9. `mut_baidu_trans.py`: Call Baidu API to translate mutated sentences
10. `extract_mutSuccess.py`: Filter valid mutation samples that complete translation normally
11. `add_MutTransToCSV.py`: Write mutation‑related translation results into CSV file
12. `clean_mut1_ZH.py` / `clean_mut2_ZH.py` / `clean_mut3_ZH.py`: Remove parenthetical gloss annotations from Chinese translated texts for Mut1‑Mut3
13. `mut_Zh_Align.py`: Re‑run bilingual alignment for post‑mutation translation results
14. `addMutAlignResToCSV.py`: Append alignment results to CSV
15. `mut1bugDetection.py`: WSD bug detection, compare predicted sense index against ground‑truth index
16. `mut1bugCount.py`: Count and output final bug statistics
