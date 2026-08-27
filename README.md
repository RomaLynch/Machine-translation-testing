# Context Information Injection-Based Mutation Testing to Detect Word Sense Disambiguation Bugs
This repository hosts the experimental code developed in conjunction with our paper "Context‑Information‑Injection‑Based Mutation Testing for Word Sense Disambiguation Bug Detection".
# Datasets
We adopt UM‑Corpus, a mainstream dataset for machine translation tasks, as our experimental dataset. We randomly sample 30,000 parallel sentence pairs covering eight topics from this corpus to construct our test set.
# Models
We evaluate two commercial machine translation systems, Baidu Translate and Youdao Translate, as well as the open‑source translation model OPUS. We utilize the paid online APIs for the two commercial systems, while the open‑source model is downloaded and deployed locally. Models can also be conveniently accessed via HuggingFace ([https://github.com/huggingface/](https://github.com/huggingface/)).
# Approach

