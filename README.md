# Context Information Injection-Based Mutation Testing to Detect Word Sense Disambiguation Bugs
This repository hosts the experimental code developed in conjunction with our paper "Context‑Information‑Injection‑Based Mutation Testing for Word Sense Disambiguation Bug Detection".
# Datasets
We adopt UM‑Corpus, a mainstream dataset for machine translation tasks, as our experimental dataset. We randomly sample 30,000 parallel sentence pairs covering eight topics from this corpus to construct our test set.
# Models
We evaluate two commercial machine translation systems, Baidu Translate and Youdao Translate, as well as the open‑source translation model OPUS. We utilize the paid online APIs for the two commercial systems, while the open‑source model is downloaded and deployed locally. Models can also be conveniently accessed via HuggingFace ([https://github.com/huggingface/](https://github.com/huggingface/)).
# Approach
This repository implements a mutation‑testing framework for detecting word‑sense disambiguation (WSD) bugs in machine‑translation (MT) systems. The pipeline consists of four main stages: **Pre‑processing**, **Initial Translation & Gloss Location**, **Mutation**, and **Bug Detection**.
## Pre‑processing
Three sub‑steps prepare inputs for subsequent mutation:
1. **Polysemous dictionary refinement**: Adopt the existing JSON‑formatted polyseme dictionary from prior work. Manually clean and filter inconsistent / misclassified sense‑gloss pairs to build a more robust polysemous resource.
2. **Raw corpus cleaning**: Process UM‑Corpus with regular expressions to strip invalid symbols, garbled text and redundant whitespace, ensuring valid input sentences for MT systems.
3. **Polysemous‑sentence extraction**: Use SpaCy lemmatization to match polysemous words (covering uppercase, singular and plural forms) and filter English sentences containing target polysemous words from UM‑Corpus.
## Initial Translation and Gloss Location
Prepare reference information before mutation:
1. Send cleaned sentences to target MT systems and collect initial Chinese translations as reference outputs.
2. Perform word‑level alignment between original English sentences and their Chinese translations using SimAlign.
3. Compute cosine similarity via SBERT‑Chinese between aligned Chinese translations and dictionary sense entries. A predefined similarity threshold (`0.9`) retrieves the ground‑truth sense index for each polysemous word.
## Mutation (three mutation operators: Mut1 / Mut2 / Mut3)
Generate test‑case mutated sentences by inserting gloss annotations in parentheses after target polysemous words:
**Mut1 (Insert the correct gloss)**: Randomly sample one valid correct gloss from `[SOURCE GLOSSES]` of the target sense and inject it into the original sentence. The injected correct gloss should guide the MT system to preserve the original translation.
**Mut2 (Negation of the incorrect gloss)**: Randomly select an irrelevant incorrect gloss, insert a negation prompt to rule out this wrong meaning. The expected behavior is that MT maintains the original correct translation.
**Mut3 (Prompt between correct and incorrect glosses)**: Inject both one correct gloss and one incorrect gloss with a multiple‑choice prompt. The MT system is expected to select the correct sense and keep consistent with the original translation.
All mutated sentences are fed into MT systems to obtain new Chinese translations for bug checking.
## Bug Detection
Identify WSD bugs by comparing post‑mutation outputs against the ground‑truth sense index:
1. Use regular expressions to remove parenthesized gloss annotations from the translated mutated sentences.
2. Re‑run English‑Chinese word alignment and use SBERT‑Chinese to obtain the predicted sense index for the polysemous word in mutated‑sentence translations.
3. **WSD bug**: Triggered when the predicted sense index mismatches the ground‑truth sense index obtained in the *Initial Translation and Gloss Location* stage.
