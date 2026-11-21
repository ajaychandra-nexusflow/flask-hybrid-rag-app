# Hybrid RAG design notes

## Pipeline
1. **Ingest**: extract text -> chunk (default 900 chars, 150 overlap, paragraph aware) -> Titan V2 embeddings (normalized) -> `collection.add` in Chroma with metadata `doc_id`, `filename`, `chunk_index`.
2. **Retrieve**: run dense and BM25 each over a candidate pool of `TOP_K * 3`, then fuse.
3. **Fuse (weighted RRF)**: `score(d) = Wd/(RRF_K + rank_dense(d)) + Wk/(RRF_K + rank_keyword(d))`. Rank-based fusion avoids having to normalise cosine scores against BM25 scores.
4. **Generate**: numbered context blocks go into the prompt; Nova Pro answers with `[n]` citations via the Converse API.

## Why these choices
- **Chroma persistent client** keeps the app single-process and dependency-light.
- **BM25 in memory** is rebuilt lazily from the Chroma collection after any ingest/delete. Fine to roughly tens of thousands of chunks; beyond that move keyword search to OpenSearch.
- **Dense + keyword** catches both semantic paraphrases and exact identifiers (SKUs, error codes, names) that embeddings blur.

## Tuning
| Symptom | Adjust |
|---|---|
| Misses exact terms / IDs | raise `HYBRID_KEYWORD_WEIGHT` |
| Misses paraphrased questions | raise `HYBRID_DENSE_WEIGHT` |
| Answers lack detail | raise `TOP_K` or `CHUNK_SIZE` |
| Answers drift off-topic | lower `TOP_K`, tighten `SYSTEM_PROMPT` |

## Quick evaluation
Keep 10 to 20 question/expected-source pairs; for each mode (`dense`, `keyword`, `hybrid`) check whether the expected file appears in the top `TOP_K` sources. Hybrid should match or beat both.
