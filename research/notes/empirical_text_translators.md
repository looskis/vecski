# Empirical results and practical architectures for learned text-embedding translators (alternative to re-embedding)

Scope: text/sentence/document embedding models used for semantic search and RAG. Each finding is tagged **[peer-reviewed]**, **[preprint]**, **[open-source repo / model card]**, or **[blog / anecdotal]**. Where I computed a derived number (e.g., "% of the source→target gap recovered") from a paper's table, I say so explicitly.

Quick orientation on what exists (as of Oct 2026):

| Work | Venue / type | Direction | Translator | Paired data | Headline retrieval number |
|---|---|---|---|---|---|
| Embedding-Converter (Yoon & Arik, Google) | ACL 2025 [peer-reviewed] | old docs → new space (queries native new); also query-side | 4-layer MLP (SELU), ~35M params, L1 + global + local relational losses | ~millions of BEIR passages + 500K MS MARCO queries | gecko003→004: nDCG@10 avg 0.5067 (src) / 0.5362 (converted) / 0.5609 (target) on 13 BEIR sets |
| Drift-Adapter (Vejendla, Rutgers) | EMNLP 2025 [peer-reviewed] | new queries → old index | Procrustes / low-rank affine / 256-unit residual MLP, MSE | 20,000 pairs | MLP recovers 98.4–99.2% of full re-embedding R@10 (MiniLM-L6→mpnet-base, AG-News/DBpedia/Emotion) |
| vec2vec (Jha, Shmatikov, Morris) | arXiv 2505.12540 [preprint] | any↔any, unsupervised | adapters + shared MLP backbone, GAN + cycle + VSP | 0 pairs (1M unpaired each) | cos 0.92 / top-1 1.00 GTE↔Stella; 1–7 GPU-days/pair; no nDCG reported |
| mini-vec2vec (Dar) | arXiv 2510.02348 [preprint] | any↔any, unsupervised | orthogonal Procrustes + iterative refinement | 0 pairs (60K unpaired) | top-1 0.96–1.00, <10 CPU-min |
| Procrustes Bounds (Maystre et al., Spotify/UiPath) | arXiv 2510.13406 [preprint] | new queries → old docs ("partial upgrade") | orthogonal Procrustes (+centering, zero-pad dims) | ~10,000 pairs | unaligned ≈ 0 nDCG@10; aligned gives gains over 49 pairs (heatmaps, no table extracted) |
| HMoE translation (Yang & Cao) | ICML 2026 [peer-reviewed, abstract only] | any↔any, 10 models | hierarchical MoE adapters + confidence metric | n/a | mixing/chaining loses 0.5–2.6% R@100 vs 7.2–92.3% for baselines |
| Maiorca et al. | NeurIPS 2023 [peer-reviewed] | any↔any | affine / linear / orthogonal closed form | anchors ≈ embedding dim | classification accuracy after stitching (no retrieval) |
| makltd123/embedding-translator | GitHub [open-source, anecdotal] | old docs → new; also reverse | ridge, Procrustes, MLP, kNN, relreps, listwise | 800 pairs | closes only ~16–18% of the gap to re-embedding; negative on unseen topic |
| QuerynAi adapters | HF Hub [model cards] | 49 directed pairs incl. ada-002→te3-small | linear or 1-hidden GELU MLP, cosine loss | ~350K rows | test cosine 0.87 (ada-002→te3-small); no retrieval metrics |

---

## Key Question 1: What have academic papers (2023–2026) actually measured for text-embedding-model translation / conversion?

### Takeaway
Two peer-reviewed papers give real retrieval numbers: Google's Embedding-Converter (ACL 2025) shows that a large MLP trained on millions of unlabeled passages recovers roughly a quarter to a half of the old→new nDCG@10 gap on BEIR (never matching re-embedding), while Drift-Adapter (EMNLP 2025) shows that for the easier query-side direction (new queries → old index) a tiny MLP on 20K pairs recovers 98–99% of full re-embedding's Recall@10. Unsupervised methods (vec2vec, mini-vec2vec) and Procrustes-bound theory establish that spaces are near-isometric enough for high top-1 matching, but they mostly report cosine/top-1 matching rather than end-to-end nDCG.

### Cited Findings

**Embedding-Converter (Yoon & Arik, Google Cloud AI, ACL 2025) [peer-reviewed]**
- Problem framing: learn h such that h(f(t)) ≈ g(t) using only an unlabeled corpus; a distinct converter per source→target pair; "operates with any embedding model, even those accessible only through prediction APIs". — [ACL Anthology PDF](https://aclanthology.org/2025.acl-long.1237.pdf)
- Training data: passages and queries from 14 BEIR datasets ("half the corpus for datasets with under 1 million passages, and 500,000 randomly sampled passages" for larger ones, plus the entire ~500K MS MARCO query set); MS MARCO excluded from evaluation. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Evaluation protocol for corpus conversion: converted corpus embeddings are searched with **queries encoded natively by the target model** ("Queries are consistently encoded using the target model (gecko004) to isolate the impact of corpus embedding conversion"). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Table 1 (in-domain, 13 BEIR, nDCG@10 averages): gecko003→gecko004: source 0.5067, **converter 0.5362**, target 0.5609. openai-3-small→gecko004: source 0.5209, **converter 0.5314**, target 0.5609. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Per-dataset (gecko003→004): converter beats target on Arguana (0.6103 vs 0.6070) and Trec-covid (0.8079 vs 0.7840) but is *below the source* on HotpotQA (0.5923 vs source 0.6248) and Quora (0.8392 vs 0.8626). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Table 2 (out-of-domain, 12 CQADupStack, nDCG@10 averages): gecko003→004: source 0.4271, **converter 0.4542**, target 0.4814. openai-3-small→gecko004: source 0.4250, **converter 0.4408**, target 0.4814. Authors: "the performance gap compared to the target model is larger than in the in-domain setting". — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Table 16 (GTE-Large → gecko004, 8 BEIR sets): source 0.5200, converter 0.5355, target 0.5477. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Table 19 (GTE-Large → Gemini embedding, BERT-encoder to decoder-only): SciFact 0.7402 (src) / 0.9218 (conv) / 0.9680 (tgt); NFCorpus 0.3391 / 0.3923 / 0.4131; Arguana 0.5928 / 0.6469 / 0.6539; SciDocs 0.2330 / 0.2280 / 0.2396 (converter below source). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Table 11 "downgrading" (gecko004→gecko003): source 0.5609, converter 0.5032, target 0.5067, i.e., converter ≈ old model. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Table 17 multilingual (gecko-multilingual-001→002, MIRACL): converter lands between source and target in most languages (e.g., French 0.4098 / 0.4634 / 0.4924; German 0.4295 / 0.4742 / 0.4928). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Use as a cheap model-selection proxy: "the relative performance of source and target models is accurately predicted by Embedding-Converter on 11 of the 13 datasets" (in-domain) and "perfectly predicted" out-of-domain. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Stated limitation: "converted embedding performance does not fully match that of the target model, meaning re-embedding is still necessary to achieve the full potential of a new model." — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Cost (Appendix A, Table 7): re-embedding 1B docs with openai-3-large ≈ $16,640 and 4,266 hours (rate-limited) vs. converter ≈ $185 and 37 hours; for 50M docs with openai-3-small: $128 / 213 h vs $6 / 1.2 h; converter inference "takes only 20 minutes" for 50M docs on 2 V100s ($4.96/h). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)

**Drift-Adapter (Vejendla, Rutgers, EMNLP 2025) [peer-reviewed]**
- Direction: maps *new* query embeddings into the *legacy* space, "gθ(f_new(q)) ≈ f_old(q)", so the existing ANN index is untouched. — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1); [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.805/)
- Text pair: all-MiniLM-L6-v2 (384-d) → all-mpnet-base-v2 (768-d), evaluated on MTEB AG-News, DBpedia-14, Emotion corpora; image pair CLIP ViT-B/32 → ViT-L/14 on a 1M-item LAION subset. — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- Training: N_p = 20,000 paired embeddings (2% of the 1M corpus), MSE loss, 80/20 split; Procrustes solved in closed form by SVD; training time ~1–2 min. — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- Results as *ratio to full re-embedding* (ARR): Residual-MLP R@10 ARR 0.992 (AG-News), 0.990 (DBpedia), 0.984 (Emotion); MRR ARR 0.988 / 0.983 / 0.976; CLIP 0.978 / 0.972; added query latency ~8 µs. Misaligned (no adapter) R@10 ARR reported at 0.589–0.723 on the text datasets. — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- Severe-drift case GloVe 300-d → MPNet 768-d: only 0.715 R@10 ARR. — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- Cost claim: ~0.5 GPU-hours vs ~100 GPU-hours for full re-indexing ("over 100×"). Limitations stated: needs paired data, assumes smooth drift, "downstream task impact unmeasured". — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)

**vec2vec (Jha, Shmatikov, Morris, 2025) [preprint]**
- First *unsupervised* translator (no pairs, no encoders): input/output adapters + shared backbone MLPs with residuals, LayerNorm, SiLU; losses = adversarial (latent + output discriminators) + reconstruction + cycle-consistency + vector-space-preservation (pairwise distances). — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Data: 1M unpaired NQ embeddings per model; max 64 tokens; "1–7 days per model pair" (176 GPU-days total); GAN training unstable—"only 3/15 cross-backbone runs achieved convergence". — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Matching quality (8,192 held-out NQ embeddings): GTE→Stella cos 0.92 / top-1 1.00; E5→Stella 0.86 / 1.00; Granite→GTE 0.87 / 0.95 (mean rank 1.18); GTE→GTR 0.75 / 0.91 (mean rank 2.64). Naive identity baseline cos ~0.01–0.04, top-1 0. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Data-efficiency ablation (GTE→GTR): 10K embeddings → cos 0.57, top-1 0.01; 50K → 0.74 / 0.81; 500K → 0.75 / 0.92; 1M → 0.75 / 0.92. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Removing VSP loss, cycle loss, or latent GAN each collapses top-1 to 0.00. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- The only retrieval-style metric is cross-modal MS-COCO Recall@16 0.23 (Granite→CLIP) vs 0.75 native CLIP; no text nDCG/recall reported. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)

**mini-vec2vec (Dar, 2025) [preprint]**
- Linear alternative: k-means in both spaces, QAP matching of centroid similarity matrices, relative-representation concatenation to get pseudo-pairs, Procrustes fit, then ICP-style / clustering refinement with exponential smoothing. — [arXiv 2510.02348](https://arxiv.org/html/2510.02348)
- 60,000 NQ sentences (disjoint halves per model), 8,192 for eval; all 10 pairs of GTR/E5/GTE/Stella/Granite: top-1 0.96–1.00, average rank 1.00–1.10 (vec2vec: 1.02–5.91); cosine 0.57–0.71 after refinement (lower than vec2vec's cosine despite better rank). — [arXiv 2510.02348](https://arxiv.org/html/2510.02348)
- "One run of mini-vec2vec is completed on a CPU in less than ten minutes, while vec2vec requires 1–7 days on a GPU." — [arXiv 2510.02348](https://arxiv.org/html/2510.02348)

**When Embedding Models Meet: Procrustes Bounds (Maystre et al., Spotify/UiPath, 2025) [preprint]**
- Theorem: if Gram matrices differ by ≤ ε in Frobenius norm, an orthogonal Q exists with ‖QX − Y‖_F ≤ (2D)^(1/4)·√ε. — [arXiv 2510.13406](https://arxiv.org/html/2510.13406)
- Seven text models (nomic-embed-text-v1.5, bge-small-en-v1.5, Qwen3-Embedding 1024-d, all-MiniLM-L6-v2, sentence-t5-base, LaBSE, gte-base-en-v1.5), retrieval on HotpotQA-HN, FEVER-HN, TREC-COVID; "partial model upgrade" = fixed doc embeddings, upgraded query encoder, 49 pairs; unaligned cross-model retrieval "fails almost completely" (≈0 nDCG@10); Procrustes alignment yields gains when upgrading to a stronger query model; **orthogonal consistently beats unconstrained linear**; ~10,000 paired samples suffice; dimension mismatch handled by zero-padding. — [arXiv 2510.13406](https://arxiv.org/html/2510.13406)

**Generalizable and Composable Multi-Model Embedding Translation (Yang & Cao, ICML 2026) [peer-reviewed; abstract only]**
- Derives error bounds explaining "systematic error amplification under OOD inputs, mixing and chaining"; proposes a geometry-aware confidence metric and a Hierarchical Mixture-of-Experts translator; 10 models, 6 datasets, 90 pairwise settings; mixing/chaining degradation 0.5–2.6% Recall@100 vs 7.2–92.3% for baselines. — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61383); [OpenReview](https://openreview.net/forum?id=qmfp2eqYD1)

**Maiorca et al., Latent Space Translation via Semantic Alignment (NeurIPS 2023) [peer-reviewed]**
- Text encoders: BERT (cased/uncased), ELECTRA, RoBERTa, ALBERT, CLIP text; datasets TREC, DBpedia, AG News, IMDB; four closed-form estimators (affine via GD, least-squares linear, l-ortho, Procrustes ortho); anchors "in a quantity comparable with the dimensionality of the absolute representation". — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract.html); [full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2023/latent_space_translation_via_semantic_alignment__f455da2d.txt)
- Cross-architecture stitching accuracy (standard scaling): TREC absolute 0.20 → affine 0.82 / ortho 0.79; AG News 0.25 → 0.65 / 0.66; DBpedia 0.07 → 0.66 / 0.66; IMDB 0.50 → 0.59 / 0.59. With L2 normalization the unconstrained linear fit drops sharply (TREC 0.44, DBpedia 0.44) while ortho/affine hold. — [full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2023/latent_space_translation_via_semantic_alignment__f455da2d.txt)

**Moschella et al., Relative Representations (ICLR 2023) [peer-reviewed]**
- Latent spaces differ by "an unknown quasi-isometric transformation"; relative representation = cosine similarities to anchor samples; text experiments are zero-shot stitching across language-specific RoBERTa encoders on classification, not retrieval. — [arXiv 2209.15430](https://arxiv.org/abs/2209.15430v2); [code](https://github.com/lucmos/relreps)

**Adjacent "index-preserving" work (same-model drift, not cross-vendor translation)**
- Query Drift Compensation (2025, preprint): continual fine-tuning of nomic-embed (768-d) across MS MARCO→NQ→HotpotQA→FEVER→FiQA; compensates new queries by subtracting a mean drift vector; avg nDCG@10: fine-tune 49.4, FT + re-indexing 50.1, FT+QDC 53.5, FT+KD+QDC 54.1 (joint training 54.1). — [arXiv 2506.00037](https://arxiv.org/html/2506.00037)
- "Succeeding at Scale" (DevRev/UT Austin, ICML 2026 workshop, preprint): query-only adaptation with frozen doc index; a *linear head on a frozen query encoder* gets nDCG@10 0.314 vs base 0.304 vs full query-doc fine-tune 0.362 (arctic-l-v2, DevRev-Search); LoRA query-only reaches 0.342. — [arXiv 2601.04646](https://arxiv.org/pdf/2601.04646)
- Align Then Adapt / ERA (2026, preprint): strong query embedder (Qwen3-8B) + frozen lightweight doc index (BGE-M3 or OpenAI-small) via a *linear* W trained with cosine loss on 1,000 unlabeled docs per task, then light supervised adaptation; MAIR nDCG@10 34.03→46.59 (BGE-M3 docs) and 36.74→49.10 (OpenAI-small docs) vs the lightweight model alone; "Linear transformation proved sufficient". — [arXiv 2604.03403](https://arxiv.org/html/2604.03403)
- Graph-index migration (DGM, 2026 preprint): reuses only the *old HNSW graph structure* (not old vectors) to speed rebuilding with new embeddings, up to 17.43× faster construction across eight text/image migrations. — [arXiv 2609.23036](https://arxiv.org/abs/2609.23036)

### Inferences
- Computed from Embedding-Converter Tables 1–2: corpus conversion recovers ≈54% (in-domain) and ≈50% (OOD) of the source→target nDCG gap for the same-family upgrade (gecko003→004), but only ≈26% / ≈28% for the cross-vendor pair (openai-3-small→gecko004). Cross-family translation is clearly harder, consistent with vec2vec's weaker cross-backbone results.
- The two headline numbers are not contradictory: Drift-Adapter's 98–99% is a ratio *of recall relative to re-embedding for the query-side direction on classification-style corpora*, with query-side mapping into the old space; Embedding-Converter's ~25–55% gap recovery is corpus-side conversion into the *new* space on BEIR. Query-side mapping preserves the old index's geometry and only needs the mapped query to land near the right old neighborhood, which is intrinsically easier.
- Unsupervised methods prove near-isometry at the top-1-matching level, but nobody has published end-to-end BEIR nDCG for vec2vec/mini-vec2vec translations; cosine 0.6–0.9 with perfect top-1 shows that "cosine to native" is a poor proxy for retrieval usefulness.

### Gaps
- No peer-reviewed paper reports a direct *ada-002 → text-embedding-3* or *Cohere v3 → v4* translator with retrieval metrics; the closest is openai-3-small → gecko004 (Embedding-Converter).
- Procrustes-Bounds paper's per-pair nDCG@10 values are only in heatmap figures; I could not extract numeric values.
- ICML 2026 HMoE paper: only the abstract was accessible; model names, datasets, and absolute nDCG unavailable.

---

## Key Question 2: Which open-source projects / repos / HF models translate embeddings between models, and what do they report?

### Takeaway
A handful of small projects exist; the only one with honest retrieval numbers (makltd123/embedding-translator) concludes translation closes just ~16–18% of the gap to re-embedding and fails off-topic, while the QuerynAi adapters (49 directed pairs incl. ada-002 → text-embedding-3-small) publish only cosine-similarity numbers (~0.87) with no retrieval evaluation.

### Cited Findings
- **makltd123/embedding-translator** [open-source, anecdotal]: paraphrase-multilingual-MiniLM-L12-v2 (384-d) → multilingual-e5-base (768-d) on Russian Gazeta news (1,000 docs; 800 train / 200 test; titles as queries). Seven methods: ridge (CV alpha), orthogonal Procrustes (rotation+scale+shift), 2-hidden-layer MLP with cosine loss (32 configs), weighted-kNN, relative representations (800 anchors), reverse (query-side) translation, listwise ranking loss. — [GitHub](https://github.com/makltd123/embedding-translator)
- Results: overlap@5 with native-e5 ranking: untranslated old model 0.494, ridge 0.585 (+9.1 pp, p<0.001), native 1.0 — i.e., "closed only ~16–18% of the gap to full re-embedding". On a 2,200-doc index, ridge MRR 0.764 vs old model 0.787 (translator *worse* than old). Unseen topic (sport): gain negligible (p=0.23). Synthetic corporate texts: ridge caused a 12.5 pp decline. Cosine to native was 0.91, "but predicting the mean B vector already achieved 0.86". Weak→strong direction "is the hardest of all 6 directions tested". Mixed index of re-embedded + translated vectors caused ranking collapse: translated vectors took 97% of top-5 slots vs 35% expected. Verdict: "not a practical replacement for re-embedding." — [GitHub](https://github.com/makltd123/embedding-translator)
- **QuerynAi** [HF model cards]: 49 directed adapters (ONNX + safetensors) among ada-002 (1536, source only), te3-small (1536), qwen3-emb-8b (4096), bge-m3 (1024), me5-large (1024), pplx-embed-1 (1024), nemotron-1b-free (2048), fastembed-bge-small (384); architecture per pair is either a single linear projection or a 1-hidden-layer GELU MLP, best one published; loss "1 − mean cosine similarity" with Adam; corpus "~349,674-row, five-domain" (arXiv, Australian case law, SQuAD, PubMed, crypto/markets news); MIT license. — [HF org](https://huggingface.co/QuerynAi)
- ada-002 → te3-small card: linear chosen (test cosine 0.8718 at epoch 15) over MLP (0.8648); output unit-normalized; **no recall@k / nDCG / MRR reported**. — [HF model card](https://huggingface.co/QuerynAi/queryn-adapter-ada-002_to_te3-small)
- **vec2vec** code is referenced from the paper; **relreps** code at GitHub lucmos/relreps. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4); [relreps](https://github.com/lucmos/relreps)
- **drift-spark** (Spark-native embedding lifecycle incl. "model-migrate") appears in search results but I did not verify its contents. — [GitHub](https://github.com/aayush4vedi/drift-spark)
- Qdrant "model migration" skill (awesome-copilot) states you MUST re-embed when changing provider/architecture/dimensions and CAN avoid it for Matryoshka truncation or by learning "linear transformation from sample data, some recall loss, good for 100M+ datasets" — guidance, not measurement. — [skills.sh listing](https://www.skills.sh/github/awesome-copilot/qdrant-model-migration)

### Inferences
- The one community project that measured retrieval (not just cosine) found that the "cosine ≈ 0.9" headline is dominated by the mean-vector baseline (0.86), a warning that applies to QuerynAi's 0.87 figure too.
- Both Queryn and makltd123 pick linear/ridge over MLP at ≤350K training rows, matching Procrustes-Bounds' finding that orthogonal beats unconstrained linear and that nonlinear capacity only pays with millions of diverse samples (Embedding-Converter).

### Gaps
- No open-source repo publishes an ada-002 → text-embedding-3-large translator with retrieval benchmarks; none publish Cohere v3→v4 or MiniLM → bge/e5/gte/nomic/jina/voyage/gemini translators with nDCG.
- QuerynAi's README cites "Vec2Vec (arXiv:2306.12689)" — a different 2023 preprint from Jha et al.'s vec2vec; I did not fetch it.

---

## Key Question 3: Which translator architectures and losses work, and how does quality scale with paired-data size?

### Takeaway
Across sources, orthogonal/linear maps are the robust default at ≤10–50K pairs and often beat unconstrained linear or MLPs; nonlinear MLPs win only with large, diverse corpora (Embedding-Converter: ~millions of samples, 35M-param MLP, relational losses; vec2vec: ≥50K–500K samples). Plain regression/cosine loss is insufficient on its own — adding pairwise-similarity (relational) preservation helps measurably.

### Cited Findings
- Embedding-Converter: 4-layer MLP, hidden dims (5×out, 5×out, 5×out, out), SELU, L2-normalized output, ~35M params for gecko003→004, Adam lr 1e-3, 50K iterations, batch 64 sampled uniformly across 14 BEIR datasets; loss = L1 regression + α·global pairwise (1−cos) distance matching + β·local (k=100 target-space nearest-neighbor) distance matching, α=β=0.1; model selection by retrieval nDCG on SciFact validation. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Ablation (gecko003→004, OOD CQADupStack nDCG@10): full 0.5369; without global+local losses 0.5219; Transformer instead of MLP 0.5273; 1/5× smaller network 0.5263; 5× larger 0.5329; **trained only on MS MARCO 0.5194** (note: the ablation table's absolute values differ from Table 2's OOD average of 0.4542, so compare variants within Table 6 only; "Relying solely on MSMarco is insufficient for broad coverage"). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Drift-Adapter: Orthogonal Procrustes (d×d, ~2.36 MB at 768-d), Low-Rank Affine r=64 (~0.39 MB), Residual MLP with one 256-unit hidden layer (~1.57 MB), optional diagonal scaling; MSE loss; 20K pairs; the MLP is the best variant (R@10 ARR 0.984–0.992). — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- Procrustes Bounds: orthogonal beats unconstrained linear "consistently"; ~10,000 pairs needed for reliable alignment (Fig. 6). — [arXiv 2510.13406](https://arxiv.org/html/2510.13406)
- vec2vec scaling (unpaired): 10K → top-1 0.01; 50K → 0.81; 500K → 0.92; 1M → 0.92 (GTE→GTR). — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Maiorca: anchors ≈ embedding dimensionality; affine ≈ ortho > linear > l-ortho on text classification stitching. — [full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2023/latent_space_translation_via_semantic_alignment__f455da2d.txt)
- makltd123: with 800 pairs, ridge regression beat Procrustes, MLP (cosine loss), kNN, relreps and a listwise ranking loss. — [GitHub](https://github.com/makltd123/embedding-translator)
- QuerynAi: at ~350K rows, linear ≈ 1-hidden MLP (0.8718 vs 0.8648 cosine) for ada-002→te3-small. — [HF model card](https://huggingface.co/QuerynAi/queryn-adapter-ada-002_to_te3-small)
- ERA: linear W with cosine loss on 1,000 unlabeled docs/task; "Adapter complexity: Linear transformation proved sufficient". — [arXiv 2604.03403](https://arxiv.org/html/2604.03403)
- Mixpeek guide (vendor, unsourced): "a few thousand pairs suffice" for Procrustes; adapter MLP "recovers more quality at the price of possible overfitting". — [Mixpeek](https://www.mixpeek.com/guides/embedding-model-migration-without-reembedding)

### Inferences
- A practical recipe consistent with all sources: start with centered orthogonal Procrustes (zero-pad to the larger dimension) on ~10K paired texts; move to a residual MLP (256–1024 hidden) with MSE/cosine + pairwise-similarity-matching loss only when you have ≥100K diverse pairs and can validate on real retrieval metrics.
- No source tested InfoNCE/contrastive loss as the translator objective on retrieval; the "relational" losses (Embedding-Converter global/local, vec2vec VSP) are the only documented improvements over point-wise regression.
- Training on queries and documents together matters: Embedding-Converter explicitly includes 500K MS MARCO queries and found MS-MARCO-only training clearly worse, implying corpus diversity > query/document distinction.

### Gaps
- No controlled study varies paired-data size (1K/10K/100K/1M) for a *supervised* translator with retrieval nDCG as the metric; vec2vec's scaling curve is unpaired and uses top-1 matching.
- No published comparison of cosine vs MSE vs contrastive loss on the same pair with nDCG.

---

## Key Question 4: The asymmetric trick — translate only queries, only documents, or run a hybrid index

### Takeaway
Query-side translation (new query encoder → old index) is the best-supported cheap path: Drift-Adapter recovers 98–99% of re-embedding recall and Procrustes-Bounds shows gains on 49 pairs; but the user keeps the *old* model's document representations and therefore most of its quality ceiling. Document-side conversion into the new space (Embedding-Converter) recovers only part of the gain, and hybrid indexes (translated old docs + natively embedded new docs) worked in Embedding-Converter's controlled test yet collapsed in a small independent test.

### Cited Findings
- Query-side, new→old (Drift-Adapter): R@10 ARR 0.984–0.992 (text), 0.978 (CLIP). — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- Query-side, "partial model upgrade" (Procrustes Bounds): unaligned ≈ 0; aligned stronger query encoders improve nDCG@10 over the old symmetric setup (heatmaps). — [arXiv 2510.13406](https://arxiv.org/html/2510.13406)
- Query-side, small→large (Embedding-Converter Table 4, query embedded by small/source model then converted into the target corpus space): gecko003→004: query converter 0.5263 (in-domain) / 0.4348 (OOD) vs corpus converter 0.5362 / 0.4542, source 0.5067 / 0.4271, target 0.5609 / 0.4814; openai-3-small→gecko004: query converter **0.5171 in-domain, below the source model's 0.5209**, 0.4342 OOD vs source 0.4250. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Hybrid index (Embedding-Converter Tables 13–14, 50% converted old docs + 50% native target docs, native target queries): in-domain avg 0.5419 vs 0.5362 all-converted vs 0.5609 all-native; OOD avg 0.4656 vs 0.4542 vs 0.4814; authors: "Converted embeddings seamlessly integrate with new embeddings". Per-dataset the mix was worse on NQ (0.5435 vs 0.5755) and Quora. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Hybrid index (makltd123, independent small test): translated vectors occupied 97% of top-5 slots instead of the expected 35% — "ranking collapse". — [GitHub](https://github.com/makltd123/embedding-translator)
- Chained converters (gecko003→GTE-Large→gecko004): avg 0.5258 vs direct 0.5369 — small loss. — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- ICML 2026 HMoE: mixing and chaining amplify error; their method limits loss to 0.5–2.6% R@100 vs 7.2–92.3% for baselines. — [ICML 2026](https://icml.cc/virtual/2026/poster/61383)
- Operational alternative without translation (blogs): dual-write/blue-green with shadow index and per-slice cutover; mixed-model indexes queried in parallel and fused with RRF. — [dev.to](https://dev.to/gabrielanhaia/embedding-model-upgrades-without-re-indexing-100-upfront-134p); [HackerNoon](https://hackernoon.com/your-embedding-model-will-deprecate-heres-what-to-do)

### Inferences
- The discrepancy on hybrid indexes is explained by calibration: Embedding-Converter L2-normalizes outputs and trains with distance-matching losses on millions of samples, so translated and native vectors share a score scale; an 800-pair ridge map has no such calibration, so translated vectors have systematically different norms/score distributions and dominate or vanish in top-k. Any hybrid deployment needs score-distribution checks (or per-source score normalization) before mixing.
- Query-side translation into the old space is strictly a "keep the old index alive" tactic; it cannot exceed the old model's representational ceiling (QDC/Drift-Adapter numbers are ratios to re-embedding *with the same drifted model lineage*, not jumps to a better model).

### Gaps
- No source reports the obvious production design — translated legacy docs in the new space + native new-model queries + natively embedded *new* docs — on a real customer corpus with before/after nDCG; Embedding-Converter's 50/50 random split is the only data point.

---

## Key Question 5: Domain shift — does a translator trained on generic text hold on a business's own corpus?

### Takeaway
Every source that tested it found degradation off-distribution: Embedding-Converter's gap recovery shrinks on held-out CQADupStack, vec2vec fails on medical records for some pairs (top-1 0.08), and the small MiniLM→e5 study saw zero gain on an unseen topic and a 12.5-pp loss on synthetic corporate text.

### Cited Findings
- Embedding-Converter OOD (CQADupStack, never seen in training): converter still beats source on all 12 sub-forums, but the gap to target widens (avg 0.4542 vs target 0.4814 for gecko003→004; 0.4408 vs 0.4814 cross-vendor). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- vec2vec trained on NQ: TweetTopic (800 tweets) Granite→GTE cos 0.85 / top-1 0.95; MIMIC (medical) Granite→GTE cos 0.85 but **top-1 0.08, mean rank 346**, while Stella→E5 on MIMIC keeps top-1 1.00. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Drift-Adapter: severe-drift case (GloVe→MPNet) 0.715 ARR; limitation: "single global transformation suboptimal for heterogeneous drift". — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- makltd123: unseen topic gain negligible (p=0.23); synthetic corporate texts −12.5 pp (p<0.001); "only on topics the translator was trained on". — [GitHub](https://github.com/makltd123/embedding-translator)
- ICML 2026 HMoE: formal bounds on "systematic error amplification under OOD inputs"; adds a geometry-aware confidence metric to flag unreliable translations. — [ICML 2026](https://icml.cc/virtual/2026/poster/61383)
- ERA: a general adapter performed comparably to domain-specific ones across MAIR's six domains (Academic, Code, Finance, Legal, Medical, Web) — but that adapter is fit on 1,000 docs *from each target corpus*. — [arXiv 2604.03403](https://arxiv.org/html/2604.03403)
- Anecdote on model swaps generally (not translation): "A team running a legal-document RAG saw nDCG@10 drop by 11% on their domain when they swapped to a model that beat the old one on MTEB by 4 points." — [dev.to](https://dev.to/gabrielanhaia/embedding-model-upgrades-without-re-indexing-100-upfront-134p)

### Inferences
- Because translators are fit by regression on the training distribution, the safe practice is to fit (or at least fine-tune) on a sample of the *target corpus* itself (ERA uses 1,000 docs; Drift-Adapter 2% of corpus; Procrustes-Bounds ~10K) — generic-corpus translators should be treated as untested on proprietary data.

### Gaps
- No published numbers for a translator trained on Wikipedia/MS MARCO and evaluated on an enterprise corpus with real query logs.

---

## Key Question 6: Dimension differences, Matryoshka (MRL) models, and normalization

### Takeaway
Dimension mismatch is handled either by zero-padding (orthogonal methods) or by MLP in/out layers (Embedding-Converter 1536→768, 768→1536, 384-truncated→768 all work with modest loss); normalization matters — text embeddings carry information in the norm, orthogonal maps need centering, and outputs are usually re-normalized to unit length.

### Cited Findings
- OpenAI text-embedding-3: text-embedding-3-small is 1536-d, 3-large 3072-d, both support a `dimensions` parameter (Matryoshka-style shortening); MIRACL 31.4% (ada-002) → 44.0% (3-small) → 54.9% (3-large) as reported in the OpenAI release and relayed by HackerNoon. — [HackerNoon](https://hackernoon.com/your-embedding-model-will-deprecate-heres-what-to-do); [OpenAI announcement (403 on fetch this session)](https://openai.com/index/new-embedding-models-and-api-updates/)
- OpenAI docs state its embeddings are normalized to length 1, so cosine = dot product (docs page; not re-fetched in this session). — [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- OpenAI community moderator on ada-002 vs text-embedding-3: "absolutely incompatible"; users reported re-embedding ~150K vectors at negligible cost. — [OpenAI community](https://community.openai.com/t/new-embedding-model-mapping-with-old-ada002-possible/607804)
- Embedding-Converter cross-dimension results: 1536→768 (openai-3-small→gecko004) avg 0.5314 vs src 0.5209 / tgt 0.5609; 768→1536 (gecko004→openai-3-small) avg 0.5105 vs tgt 0.5209; **384-d truncated openai-3-small → 768-d gecko004** (MRL scenario): SciFact 0.7105 / 0.7516 / 0.7693, FiQA 0.3633 / 0.4140 / 0.5481, Arguana 0.5445 / 0.5862 / 0.6070 (source / converter / target). — [ACL 2025](https://aclanthology.org/2025.acl-long.1237.pdf)
- Procrustes Bounds: "When dimensionalities differ, smaller embeddings are zero-padded"; "orthogonal + centering" is their recommended recipe for mixed-modality search. — [arXiv 2510.13406](https://arxiv.org/html/2510.13406)
- Maiorca: text modalities show "stronger reliance ... on the information encoded in the norm"; L2-normalizing before an unconstrained linear fit hurt (TREC 0.74→0.44) while orthogonal fits were stable; standard scaling (zero-mean, unit-variance) recommended for Gaussian-like norm distributions. — [full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2023/latent_space_translation_via_semantic_alignment__f455da2d.txt)
- Drift-Adapter uses an optional diagonal scaling matrix on top of Procrustes/affine to handle per-dimension scale differences; MiniLM 384-d → mpnet 768-d. — [arXiv 2509.23471](https://arxiv.org/html/2509.23471v1)
- QuerynAi adapters output "unit-normalized" vectors and cover 384-d ↔ 4096-d pairs. — [HF org](https://huggingface.co/QuerynAi)
- Vendor guidance: Matryoshka models let you shrink dimensions without re-embedding; otherwise changing provider/architecture/dimensions requires re-embedding. — [skills.sh qdrant-model-migration](https://www.skills.sh/github/awesome-copilot/qdrant-model-migration); [Zilliz FAQ](https://zilliz.com/ai-faq/change-embedding-model-without-reindexing)

### Inferences
- For ada-002 (unit-norm, 1536) → text-embedding-3-large (3072, MRL): a learned map should target the *full* 3072-d output, since 3-large's shortened dims are a prefix truncation; translating into a truncated target would throw away the target's extra capacity, but Embedding-Converter's 384→768 experiment suggests low→high dimension maps are learnable.
- Re-normalize translator outputs to unit length and verify score distributions before mixing with native vectors; threshold-based filters (e.g., cosine > 0.8 cutoffs tuned on ada-002) will not transfer, as the OpenAI community thread notes thresholds changed from ~0.80 to ~0.50 after switching models. — [OpenAI community (thresholds)](https://community.openai.com/t/transitioning-to-the-new-embeddings-models-from-ada/602761)

### Gaps
- No experiment translating into an MRL model and then truncating; no experiment on whether a translator trained at full dimension stays valid after target-side truncation.

---

## Key Question 7: Embedding inversion (vec2text) as an alternative path — invert old embedding to text, re-embed with the new model

### Takeaway
Inversion is far more expensive than a translator (hundreds of embedder calls per document, a 2-GPU-day training run per source model, 5M-passage training sets) and fidelity collapses beyond ~32 tokens (ada-002: 60.9% exact match at 32 tokens, 8.0% at 128), so it is not a viable migration route for document chunks — though no one has published the full "invert → re-embed → retrieve" pipeline numbers.

### Cited Findings
- vec2text (Morris et al., EMNLP 2023): T5-base-initialized inverter + corrector (~235M params each); GTR-base trained on 5M NQ/Wikipedia passages (32 tokens), ada-002 on MS MARCO (32 or 128 tokens); results at 32 tokens: GTR BLEU 97.3 / exact 92.0% / cos 0.99; ada-002 BLEU 83.4 / exact 60.9% / cos 0.99; **ada-002 at 128 tokens: BLEU 55.0, exact match 8.0%, cos 0.99**; inference = steps × beam embedder calls (50 × 8 = 400 per text); training ~2 days on 4 A6000s; recovers 89% of full names from MIMIC clinical-note embeddings. — [arXiv 2310.06816](https://arxiv.org/html/2310.06816v1)
- Threat analysis / reproduction: vec2text "requires constructing a training dataset of 5 million passage-embedding pairs" per target embedder. — [Understanding and Mitigating the Threat of Vec2Text](https://arxiv.org/html/2402.12784)
- Newer inverters reduce per-model training: ZSInvert (zero-shot adversarial decoding, no embedding-specific training), ALGEN (few-shot cross-model alignment then generation), Zero2Text (training-free, LLM priors + online regression). — [Universal Zero-shot Embedding Inversion](https://arxiv.org/pdf/2504.00147); [arXiv 2602.11047](https://arxiv.org/pdf/2602.11047v3)
- vec2vec combines translation + inversion: translate into a space with an existing inverter, then invert; GPT-4o-judged information leakage up to 80% on Enron emails, 67% on tweets — this is attribute leakage, not verbatim reconstruction. — [arXiv 2505.12540](https://arxiv.org/html/2505.12540v4)
- Overview of inversion fidelity and limits: "Vec2Text is capable of reconstructing up to 92% of 32-token text inputs exactly." — [The Gradient](https://thegradient.pub/text-embedding-inversion/)

### Inferences
- A typical RAG chunk (256–512 tokens) is 4–16× longer than the regime where inversion is faithful; at 128 tokens only 8% exact match with ada-002, so re-embedding inverted text would produce new-model vectors for paraphrased, lossy content. Cosine 0.99 to the *old* embedding (the inverter's objective) says nothing about how the reconstructed text embeds under the *new* model.
- Cost: 400 embedder calls per chunk versus 1 call for true re-embedding — inversion only makes sense when the source text is truly lost, not as a cost saver.

### Gaps
- I found no paper or blog that measured retrieval quality (nDCG/recall) of "invert old embeddings → re-embed with new model" versus true re-embedding.

---

## Key Question 8: What do vendors and engineering blogs say, and has anyone deployed translation in production?

### Takeaway
Every vendor source (OpenAI community, Weaviate, Zilliz/Milvus, Qdrant guidance, Formation, dev.to) tells practitioners to re-embed via blue-green/dual-write; translation is mentioned as research-grade, and as of 2026 no company has publicly confirmed a production deployment of cross-model embedding translation.

### Cited Findings
- "No company has publicly confirmed production deployment of alignment techniques. Industry accounts from Uber, Pinterest, and Google describe full re-embedding as standard practice"; Weaviate quoted as cautioning that alignment "might help map one embedding space to another... the chances of losing context or variance in performance is high." — [HackerNoon (Aaditya Chauhan)](https://hackernoon.com/your-embedding-model-will-deprecate-heres-what-to-do)
- Weaviate's vectorizer-migration tutorial: evaluate on a representative subset with old and new models, then migrate (no translation path). — [Weaviate docs](https://docs.weaviate.io/weaviate/tutorials/vectorizer-migration)
- Zilliz/Milvus FAQ: vectors from v1 and v2 of a model "sit in different spaces, so cosine similarity between them is meaningless ... there is no in-place transform that makes the old vectors valid"; recommends blue-green migration with a held-out query set comparing recall. — [Zilliz FAQ](https://zilliz.com/ai-faq/change-embedding-model-without-reindexing)
- Formation blog: "Converting old vectors is not a substitute unless a supported and validated transformation exists"; recommends backfill from source text, dual-write, shadow reads, paired cutover of query model + index, lineage metadata. — [Formation](https://formation.dev/blog/embedding-model-upgrade-migration)
- dev.to migration playbook: dual-write → shadow backfill → A/B → sliced cutover; overlap@k online recall detector with alert threshold 0.65. — [dev.to](https://dev.to/gabrielanhaia/embedding-model-upgrades-without-re-indexing-100-upfront-134p)
- Mixpeek guide lists three strategies (dual-index re-embed; learned translation via rotation or adapter; query-side bridging) and a gate `recall_at_k(new) >= 0.98 * recall_at_k(old)`, but provides no measured numbers or citations. — [Mixpeek](https://www.mixpeek.com/guides/embedding-model-migration-without-reembedding)
- OpenAI community: the recommended path is to re-embed; "run everything on the old model while you build the new one in the background". — [OpenAI community](https://community.openai.com/t/transitioning-to-the-new-embeddings-models-from-ada/602761)
- Neo4j agent-memory docs provide a "migrate to a new embedding model" how-to (re-embed), listed in search results; not fetched. — [Neo4j](https://neo4j.com/labs/agent-memory/how-to/migrate-embedding-model/)

### Inferences
- The engineering consensus is driven by cost: re-embedding is cheap for ≤ millions of chunks (community report: 150K vectors at negligible cost), so translation only becomes attractive at the 50M–1B scale where Embedding-Converter's $185-vs-$16,640 / 37-h-vs-4,266-h comparison applies, or when source text is unavailable.

### Gaps
- No Cohere, Voyage, Jina, Nomic, Mixedbread, Pinecone, LlamaIndex, LangChain or Supabase post specifically about learned translators was found in my searches; the vector-DB posts found are generic re-embedding playbooks.
- No Hacker News thread with measured translator results was located.
