# Investigation: Frame Ratio Differences — Indonesia Air Pollution

This note compares frame ratios across two narrative framing runs and the keyword-based word_count run for the Indonesia air quality corpus, explains key discrepancies, and proposes next steps.

Sources referenced

- Narrative Framing (NF) 2025‑10‑22
  - Schema: `results/narrative_framing/indonesia_airpollution_causes_20251022/frame_schema.json`
  - Assignments: `results/narrative_framing/indonesia_airpollution_causes_20251022/frame_assignments.json`
  - Aggregates: `results/narrative_framing/indonesia_airpollution_causes_20251022/aggregates/{occurrence.json, weighted.json}`
  - Report: `results/narrative_framing/indonesia_airpollution_causes_20251022/frame_report.html`
- Narrative Framing (NF) 2025‑10‑23
  - Schema: `results/narrative_framing/indonesia_airpollution_causes_20251023/frame_schema.json`
  - Assignments: `results/narrative_framing/indonesia_airpollution_causes_20251023/frame_assignments.json`
  - Aggregates: `results/narrative_framing/indonesia_airpollution_causes_20251023/aggregates/{occurrence.json, weighted.json}`
  - Report: `results/narrative_framing/indonesia_airpollution_causes_20251023/frame_report.html`
- word_count (themes)
  - Summary: `results/word_count/indonesia_airpollution_causes/summary.csv`
  - Metadata: `results/word_count/indonesia_airpollution_causes/report_data.json`
  - Report: `results/word_count/indonesia_airpollution_causes/frame_report.html`
- Investigation utility script
  - `investigate_keywords_vs_narrativeframing.py`

## Summary (at a glance)

- Construction and Dust is much higher in NF (especially 2025‑10‑22: ~11.1% of docs) than in word_count (~0.3%).
- LLM assigns construction/dust semantically and often alongside transportation; explicit “construction dust” keywords are rarely present in those passages.
- The 2025‑10‑23 NF run shows lower construction shares than 2025‑10‑22 due to a different frame schema, different document set, and aggregation threshold effects.
- Transportation coverage is consistently high and aligns better with word_count keywords.

## Frame Ratios (shares of documents)

NF 2025‑10‑22 (documents: 7,192)

| Frame | Occurrence Share | Weighted Share |
|---|---:|---:|
| Transportation Emissions | 40.5% | 34.8% |
| Construction and Dust | 11.1% | 5.7% |
| Industrial Emissions | 7.5% | 4.7% |
| Power Plant Emissions | 4.6% | 2.8% |
| Waste Management | 2.4% | 2.1% |
| Domestic Emissions | 1.0% | 0.8% |
| Agricultural Emissions | 0.2% | 0.2% |

NF 2025‑10‑23 (documents: 6,508)

| Frame | Occurrence Share | Weighted Share |
|---|---:|---:|
| Transportation Emissions | 30.9% | 28.1% |
| Industrial Emissions | 6.5% | 4.3% |
| Construction and Dust | 3.6% | 1.7% |
| Power Plant Emissions | 3.0% | 1.8% |
| Waste Management | 1.1% | 1.0% |
| Biomass Burning and Agriculture | 0.2% | 0.2% |
| Household Emissions | 0.0% | 0.0% |

word_count (documents: 3,000; 2020‑01‑01→2025‑10‑10)

| Theme | Share |
|---|---:|
| Transportation | 18.3% |
| Industrial | 5.0% |
| Power Plant | 3.1% |
| Agriculture | 3.6% |
| Waste Burning | 2.5% |
| Domestic | 0.3% |
| Construction and Dust | 0.3% |

Notes:
- NF “Occurrence” = doc‑level presence by threshold (default 0.2). “Weighted” = average per‑doc normalized scores.
- word_count counts a theme if at least one configured keyword hits a document (min_words defaulted to 1 in code).

## Diagnostics and Evidence

1) Construction/Dust keyword coverage is too narrow for Indonesian text

- NF 2025‑10‑22:
  - Passages labeled Construction/Dust (top_frames): 167
  - With any word_count construction keyword: 0/167 (0.0%)
  - Including passages with p≥0.2: 1/231 (0.4%)
  - Co‑occurrence in top_frames: transportation_emissions co‑occurs in 116/167 (≈69.5%).
- NF 2025‑10‑23:
  - Passages labeled Construction/Dust (top_frames): 207
  - With any word_count construction keyword: 1/207 (≈0.5%)
- By contrast, Transportation keyword coverage is strong (NF 2025‑10‑22):
  - Transport‑labeled passages with any word_count transport keyword: 401/540 (≈74.3%)

Interpretation: LLM frequently tags Construction/Dust via semantically related cues (e.g., pembangunan, jalan, semen, road dust from traffic) without explicit “construction dust” phrases. The current word_count theme misses this, hence a large gap versus NF.

2) NF schema differences between runs

- 2025‑10‑22 uses slug IDs and includes “Construction and Dust” with schema keywords: `construction dust`, `roadwork dust`, `resuspended soil`, `construction pollution`.
- 2025‑10‑23 induction produced numeric IDs 1–7 and different keyword lists; even subtle shifts in schema keywords/examples can change assignment tendencies.
- Different schemas + different document sets explain why Construction/Dust drops from 11.1% → 3.6% (occurrence) and 5.7% → 1.7% (weighted).

3) Sampling/timeframe and denominator differences

- NF: larger doc sets (7,192 vs 6,508) and no explicit date window in these configs.
- word_count: limited to 3,000 docs and constrained by `date_from`/`date_to`.
- Different denominators/time windows shift ratios.

4) Aggregation threshold effects

- NF occurrence uses a per‑doc presence threshold (default 0.2). Co‑occurring frames and low‑probability assignments can push borderline presence over/under this cut, impacting doc‑level shares.

## Explanations (Why results differ)

- Method mismatch (semantic vs literal):
  - NF assigns frames semantically; word_count relies on explicit strings. Construction/Dust content often lacks explicit “dust” terminology although conceptually present (e.g., roadworks, development, cement context). Transport is more literal (vehicle, exhaust), so alignment is tighter.

- Schema drift between NF runs:
  - Changed frame IDs/keywords between 2025‑10‑22 and 2025‑10‑23 alter LLM guidance and thus aggregate ratios.

- Dataset/timeframe mismatch:
  - Different doc sets and date filters produce different denominators and event mixes.

## Suggested Actions

1) Align frame schema across NF runs

- Pin a single frame schema (e.g., reuse 2025‑10‑22’s `frame_schema.json`) for 2025‑10‑23 to ensure apples‑to‑apples comparisons, or define a stable manual frames file and reference it in both runs.
- Keep `application_sample_size`, `application_top_k`, thresholds, and filtering identical across runs.

2) Expand word_count Construction/Dust keywords for Indonesian

- Add common terms and morphology, optionally anchored by pollution cues:
  - Core terms: `konstruksi`, `pembangunan`, `proyek (infrastruktur|konstruksi)`, `pekerjaan jalan`, `perbaikan jalan`, `pengaspalan`, `perataan jalan`
  - Dust cues: `debu proyek`, `polusi debu`, `partikel debu`, `resuspensi debu`, `debu jalan`, `debu tanah`, `debu semen`
- Consider patterns combining activity + pollution terms (e.g., `(konstruksi|pembangunan|pekerjaan jalan).*(debu|polusi|partikel)`) to avoid false positives.
- Re‑validate coverage with the investigation script (see below) until a healthy fraction of NF construction passages also match word_count.

3) Adjust NF occurrence threshold (optional)

- If you prefer stricter doc‑level presence, increase `agg_min_threshold_occurrence` (e.g., 0.3–0.4). This will reduce borderline co‑occurrence cases (especially transport+construction dust).

4) Normalize sampling/timeframe

- Align date windows across NF and word_count. Keep doc sets stable by reusing the same cached `classified_chunks` and setting `reload_*` appropriately.

5) Add comparability utilities

- Snapshot the frame schema with each report. Add a small diff step in the report to warn on schema changes across runs.

## Reproduce and Inspect

- Construction keyword vs NF passages:

```
python3 investigate_keywords_vs_narrativeframing.py \
  configs/word_count/indonesia_airpollution_causes.yaml \
  results/narrative_framing/indonesia_airpollution_causes_20251022

python3 investigate_keywords_vs_narrativeframing.py \
  configs/word_count/indonesia_airpollution_causes.yaml \
  results/narrative_framing/indonesia_airpollution_causes_20251023
```

Outputs include the number of LLM‑labeled construction passages containing any word_count keywords, top terms in those passages, and examples without keyword hits.

## Closing

- The largest single driver of the discrepancy is the Construction/Dust theme: NF’s semantic signal is not captured by the current word_count keyword list.
- Aligning schemas and expanding Indonesian construction/dust tokens (with careful precision) will close most of the gap, while threshold and timeframe alignment will stabilize cross‑run comparisons.

