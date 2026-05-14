# 01 — Data Pipeline

Data quality dominates final model quality more than architecture choices. Budget 30–50% of total project effort here.

## Stages

1. **Acquisition** — Common Crawl WARC files, GitHub mirrors, arXiv, books, Wikipedia, multilingual corpora (mC4, CulturaX), licensed data (Stack Exchange, news), synthetic data.
2. **Extraction** — `trafilatura` / `resiliparse` for HTML→text; `pdfplumber`/`nougat` for PDFs; AST-aware extraction for code.
3. **Language ID** — `fasttext` lid.176; route per-language pipelines.
4. **Quality filtering** —
   - Heuristic (Gopher rules: mean word length, % alphabetic, stopword ratio, perplexity bands).
   - Classifier (small fastText classifier trained on "high quality" labels e.g. Wikipedia vs random Common Crawl). FineWeb-Edu style.
   - Toxicity classifier with calibrated thresholds.
5. **Deduplication** —
   - **Exact**: SHA-1 of normalized text.
   - **Near-dup**: MinHash-LSH on shingles (datasketch / `text-dedup`).
   - **Substring**: suffix-array based (Lee et al. 2022).
   - Dedup is global across the corpus, not per-shard.
6. **PII scrubbing** — regex + NER for emails, phones, SSNs, credit cards, API keys.
7. **Decontamination** — n-gram overlap removal vs every eval set (MMLU, HumanEval, etc.). Document each scrub.
8. **Mixing** — domain weights (e.g. 50% web, 15% code, 10% books, 10% papers, 10% multilingual, 5% math). Tunable; controlled by `configs/data_mix.yaml`.
9. **Tokenization & sharding** — tokenize once, write fixed-size shards (e.g. 1GB) of `uint32` token IDs to object storage. Index file lists shard URIs + token counts.
10. **Streaming dataloader** — Mosaic Streaming or WebDataset; deterministic shuffle with a global seed; resumable mid-epoch.

## Throughput target

For a 10T-token run you must tokenize ~50 TB of text. At 50 MB/s/node CPU throughput you need ~200 nodes for ~1 week. Plan for it.

## Skeleton modules

- `platform/data/acquire.py` — source connectors
- `platform/data/extract.py` — HTML/PDF/code extraction
- `platform/data/filter.py` — quality + toxicity
- `platform/data/dedup.py` — minhash + suffix array
- `platform/data/decontaminate.py` — eval-set scrub
- `platform/data/mix.py` — domain weighting
- `platform/data/shard.py` — tokenize + write shards
- `platform/data/loader.py` — streaming dataloader
