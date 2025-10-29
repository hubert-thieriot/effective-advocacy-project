---
layout: default
title: Narrative Framing for Media Analysis
---
# Narrative Framing for Media Analysis — Air Pollution, Energy Transition, Animal Welfare

> TL;DR: We identify a small set of narrative framings across media articles on different topics using LLMs and other NLP techniques. This helps you see how issues are discussed, spot trends and shifts, and surface outlets/journalists to prioritize—useful for informing advocacy and gauging impact (e.g., air pollution, renewables, animal welfare).

[Jump to results](#results)

This post is part of a series of technical explorations for **Effective Advocacy**. The ultimate goal is to devise practical tools that help advocacy teams understand narratives, diagnose gaps, and track changes (and ideally impact) over time.

## Why narrative framing?
- Understand how a topic is discussed: what narratives, causes, and emphases appear—and how they change over time.
- Inform advocacy: detect momentum, measure campaign impact, and identify outlets/journalists to prioritize.
- Go beyond keywords: capture paraphrase and implied meaning when language varies across outlets and time.
- Keep it pragmatic: start with LLM exploration; scale via a supervised classifier; iterate with spot checks.

## Method overview
A compact pipeline from LLM exploration to scalable measurement.

```mermaid
flowchart LR
  A["Article discovery (MediaCloud)"] --> A2[Scrape + extract text]
  A2 --> B[Chunk text]
  B --> C[LLM: induce frames]
  C --> D[LLM: apply frames to samples]
  D --> E[Train classifier on samples]
  E --> F[Classify chunks]
  F --> G["Aggregate at the article level"]
  G --> H[Time series, domains]
  H --> I[Report]
```

## Results

### Jakarta — Air pollution causes
Run: 2020–2025 Indonesian media on Jakarta air pollution; config at `configs/narrative_framing/indonesia_airpollution_causes_20251028.yaml`.

- Stacked area of frame share over time (30‑day running average, Plotly render):

![Frame share over time]({{ site.baseurl }}/assets/indonesia_airpollution_causes_20251028/plots/time_series_area.png)

- Frame set (short names): Transport, Industrial, Power Plant, Biomass Burning, Household, Waste Burning, Construction/Dust, Natural Factors.
- Quick read: Transport dominates; Natural Factors emerges in seasonal spikes (e.g., stagnant air, lack of rain).
- Interactive details: open the HTML report at `{{ site.baseurl }}/reports/indonesia_airpollution_causes_20251028/frame_report.html`.

### United Kingdom — Renewable energy

<!-- To be added -->

### Brazil — Animal welfare

<!-- To be added -->

## Technical details / Supplementary information

### Article discovery (search/filters)
We start by defining the slice of media we care about in a way that is both broad enough to catch variation and precise enough to be actionable. Using Media Cloud collections lets us anchor each run in a country and time window, and then layer topical filters (for instance, city names or issue cues) to focus coverage. The intent is to bias toward recall at this stage: we would rather include a few borderline articles and filter them downstream than miss legitimate phrasing that differs from our initial keywords. Every run is captured in a small YAML file so the choices are explicit and replicable.

### Scrape and extract
To reason about narratives we need full passages, not just headlines or snippets. We fetch pages and extract the main text, then remove boilerplate and navigation tails that otherwise drown the signal (things like widgets, “follow us” blocks, or stock tickers). The trimming rules live in config so we can adapt them by outlet or country. This step trades a little engineering effort for cleaner inputs and more stable downstream classification.

### Frame induction (LLM)
Instead of hard‑coding a universal taxonomy, we ask an LLM to propose a compact set of categories tailored to the question and context (e.g., causes of air pollution in Jakarta). This keeps the schema close to how journalists actually talk in that domain and time, while our induction prompt nudges toward empirically grounded, measurable categories. We snapshot the resulting schema (names, short definitions, examples) and treat it as a contract for the rest of the run—minimizing drift and keeping the analysis auditable.

### Frame application to samples (LLM)
We then use the LLM as a careful, probabilistic annotator on a sample of passages. Each passage gets a distribution over frames (not just a single label) plus a brief rationale. This does two things: it reveals ambiguous cases that keywords would miss, and it gives us enough labeled data to train a supervised model. Sampling is deliberate—we prefer a diverse, representative set rather than trying to label everything with the LLM, which would be costly and less reproducible.

### Supervised classifier (transformers)
For scale and consistency, we fine‑tune a multi‑label transformer classifier on those LLM‑labeled passages. This gives us cheap, fast inference over tens of thousands of chunks while freezing the labeling policy defined by the schema. We pick language‑appropriate encoders (e.g., IndoBERT for Indonesian), and use sigmoid outputs with a threshold to allow overlapping frames when passages truly mix narratives. The trade‑off is classic: the classifier is less flexible than an LLM but more stable, cheaper, and easier to validate with held‑out metrics.

### Classify the corpus
We classify content at the chunk level (typically sentences or short spans) to avoid burying weaker frames in long articles. Light keyword gating and regex excludes from earlier steps help keep us on topic without reintroducing brittle rules. Results are cached per document to support iterative runs and easy re‑aggregation.

### Aggregate and report
Finally, we move from chunk‑level predictions to article‑level profiles and summaries over time. A length‑weighted aggregator estimates how much attention each frame receives within an article; an occurrence view answers a different question—what share of articles mention a frame at all. We build daily time series and smooth them over 30 days to make seasonal shifts legible, and we break out top domains to see who emphasizes what. Reports are exported as interactive HTML (for exploration) and as static PNGs (for embedding and versioning).

### Models used (at a glance)
- Induction + application (LLM): OpenAI GPT‑4 class models configured per run (e.g., `gpt-4.1` for induction; `gpt-4.1-mini` for application). See run config: `configs/narrative_framing/*`.
- Classifier: Hugging Face transformers sequence classifier (BERT‑family encoder) trained for multi‑label classification with sigmoid outputs. Model is configurable; e.g., Indonesian runs use `indobenchmark/indobert-base-p1`.
- Embeddings (when needed): Sentence‑Transformers encoders (e.g., `all-MiniLM-L6-v2`) via `efi_analyser/embedders/sentence_transformer_embedder.py`.
