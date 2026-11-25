The right dataset depends on whether you are **pretraining a small model from scratch** or **fine-tuning an existing open-weights model** (like Llama, Qwen, or Mistral) on your local hardware. Pretraining needs raw, clean text to teach language and facts, while post-training needs formatted prompt-response pairs to teach instruction following and reasoning.

## Pretraining Small Models From Scratch

Training a small model locally (tens to hundreds of millions of parameters) requires narrow, clean text—sprawling web dumps spread a small model's limited capacity too thin and prevent it from locking in clean grammar.

* **TinyStories:** A synthetic corpus of simple short stories built with a restricted vocabulary. Models under 100M parameters trained on TinyStories produce fluent, coherent narratives even on a single consumer GPU or Apple Silicon.
* **FineWeb-Edu:** A classifier-filtered subset of Hugging Face's FineWeb focused strictly on high-value educational web pages. It is a top choice for small pretraining runs because educational density boosts reasoning and factual recall per token.
* **SmolLM-Corpus and Cosmopedia:** Curated specifically for compact local models, combining synthetic textbooks (Cosmopedia) with educational Python code (`Python-Edu`).
* **English Wikipedia and OpenWebText:** Wikipedia provides over 5.08 billion words of encyclopedic text under CC BY-SA for factual grounding, while OpenWebText offers tens of gigabytes of public-domain (CC0) web prose linked from highly upvoted Reddit posts.

## Large-Scale and Specialist Pretraining Corpora

If you are training a larger model or streaming subsets directly from Hugging Face without downloading terabytes to disk, these corpora offer stronger filtering than raw Common Crawl (whose monthly web archives exceed 340 TiB):

* **FineWeb and FineWeb2:** FineWeb spans 15 trillion deduplicated English tokens (ODC-By), while FineWeb2 applies an open curation pipeline across multiple languages.
* **Dolma (Allen AI):** A 3-trillion-token mixture (ODC-By) blending web text, scientific papers, code, books, and Wikipedia with fully documented provenance and filtering steps.
* **RedPajama-Data v2:** Over 100 billion tokens of filtered web text under a commercial-friendly Apache 2.0 license, bundled with quality signals so you can tune your own filtering thresholds.
* **The Pile (EleutherAI):** An 825 GB MIT-licensed English corpus combining 22 diverse sources—including academic papers, books, legal text, and GitHub—useful when you only want to pull specific domain components.
* **Common Corpus (Pleias):** A heavily curated open multilingual dataset spanning public-domain books, open science, open-source text, and over 407 billion tokens of legal and financial government records.
* **Code and Math:** **The Stack v2** (67.5 TB across 600+ languages, Apache 2.0) is the standard open code corpus, and **FineMath** extracts curated math pages directly from Common Crawl.

## Fine-Tuning and Alignment Datasets

For most local setups, adapting an existing base model via SFT and DPO yields far better practical results than pretraining from scratch.

* **OpenHermes 2.5:** Roughly 1 million curated synthetic instructions (Apache 2.0) covering chat, coding, math, and reasoning—widely used to make smaller local models perform above their parameter count.
* **FLAN v2:** Google's multi-task instruction collection (permissive CC license) covering classification, summarization, Q&A, and translation for strong zero-shot generalization.
* **Dolci-Instruct-SFT:** 2.15 million instruction-following samples across math, science, coding, safety, and 70+ languages (non-commercial CC-BY-NC-4.0).
* **Reasoning-Trace Corpora:** Datasets like **SYNTHETIC-2-SFT-verified** (4 million verified reasoning traces) and **Nemotron-Cascade-2-SFT-Data** (15.87 million mixed reasoning and agentic samples) teach step-by-step problem solving in math and code.
* **UltraFeedback:** 64,000 preference-ranked instructions (MIT license) commonly used to align local chatbots for helpfulness and instruction adherence.

## Practical Tips for Local Hardware

1. **Stream instead of downloading:** Use Hugging Face's `datasets` library in streaming mode (`streaming=True`) to pull slices of massive corpora like FineWeb-Edu or Dolma without filling your SSD.
2. **Prioritize quality over raw token count:** A few hundred million tokens of FineWeb-Edu or OpenHermes 2.5 will outperform tens of billions of noisy scraped pages on consumer GPUs.
3. **Check licensing early:** Stick to CC0, MIT, Apache 2.0, or ODC-By datasets if you plan to ship a commercial product; avoid NC-licensed sets (like CC-BY-NC-4.0) for commercial weights.

---

Use the interactive comparator `interactive_visual_im_9602518c3c5be223.html` to filter these datasets by training stage, domain focus, or commercial license compatibility to find the best fit for your setup.

---

Even on modest hardware, starting with a small streamed subset of **FineWeb-Edu** (for from-scratch experiments) or **OpenHermes 2.5** paired with **UltraFeedback** (for fine-tuning an existing open-weights model) gives the highest return per compute hour.