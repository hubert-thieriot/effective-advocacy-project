# Natural Factors: Narrative vs. Word Count (Indonesia, 2025‑10‑28)

This note investigates why the Natural and Meteorological Factors frame is substantially higher in the narrative framing report than in the word_count results.

## Snapshot of the 2025‑10‑28 run

- Paths:
  - Narrative framing: `results/narrative_framing/indonesia_airpollution_causes_20251028/`
    - `frame_report.html`, `frame_assignments.json`, `aggregates/occurrence.json`, `frame_timeseries.json`
  - Word count config: `configs/word_count/indonesia_airpollution_causes.yaml` (updated to include `natural_factors`)

- Overall presence (documents):
  - Natural factors documents: 1,229 of 14,469 → 8.5% (occurrence aggregates)

- 2023 monthly volume surge (all themes):
  - Docs per month: 2023‑06: 476, 07: 183, 08: 3,398, 09: 1,810, 10: 556
  - Top Q3 domains: Liputan6, Detik, Media Indonesia, Republika, VIVA, JPNN, etc.

## Keyword coverage vs. LLM labels

Using the helper script on the 2025‑10‑28 run:

```
python3 investigate_keywords_vs_narrativeframing.py \
  configs/word_count/indonesia_airpollution_causes.yaml \
  results/narrative_framing/indonesia_airpollution_causes_20251028 \
  --frame-id natural_factors
```

- Selected LLM-labeled passages: 188 (top_frames or p≥0.2)
- Passages containing any word_count natural-factors keyword: 69/188 ≈ 36.7%
- Top hits among our keywords:
  - “cuaca” (39), “angin” (23), “musim kemarau” (11), “El Nino” (4)

Examples without keyword hits (paraphrased):
- BMKG explains stagnant PM2.5 due to humidity inversion layers near surface (mentions: inversi suhu, kelembapan tinggi).
- Weather operational notes: TMC not possible due to lack of clouds, coordination with BMKG/BRIN.
- General advisories around seasonal transitions (pancaroba) and rainfall patterns.

Interpretation: The NF model assigns “natural_factors” on semantic cues that frequently lack our exact keyword tokens. Words like “inversi (suhu/temperatur)”, “kelembapan”, “awan (tidak tersedia)”, “pancaroba”, “panas/gelombang panas”, and “stagnasi/angin lemah” appear often, but our word_count list currently lacks many of these.

## Why NF > word_count for natural_factors

- Semantic breadth: NF captures weather/meteorological causes (stagnant air, inversions, windy vs. calm days, no rain) even when “natural factor” words aren’t explicit. Word_count relies on literal tokens.
- Q3 volume spike: The August–September coverage surge includes many BMKG/weather-related articles (Liputan6, Detik, Media Indonesia, etc.), inflating NF natural_factors counts. Many such pieces don’t explicitly say “faktor meteorologi / El Nino / musim kemarau”, but imply weather‑driven pollution.
- Lexical gap: Only ~36.7% of LLM‑labeled passages contain current natural_factors keywords. The rest rely on related phrases not in our keyword set.

## Suggested improvements for word_count

- Expand Indonesian keywords:
  - Core meteorology: `BMKG`, `inversi suhu`, `inversi temperatur`, `kelembapan tinggi|rendah`, `awan`/`tidak ada awan`, `angin lemah|tenang|stagnan`, `arah angin`, `pancaroba`, `kemarau panjang`, `musim kering`, `panas terik`, `gelombang panas`
  - Rainfall phrasing: `tidak ada hujan`, `minim hujan`, `kurang hujan`, `jarang hujan`, `hujan reda`, `tak turun hujan`
  - Episode language tied to pollution: `stagnasi udara`, `udara statis`, `lapisan inversi`, `penumpukan polutan`
- Add combined patterns (examples):
  - `(?i)(tidak\s+ada|minim|kurang|jarang)\s+hujan.{0,60}(polusi|udara|pm2\.5|asap)`
  - `(?i)kemarau(\s+p\w+)?.{0,60}(polusi|udara|pm2\.5)`
  - `(?i)inversi\s+(suhu|temperatur)`
  - `(?i)kelembapan\s+(tinggi|rendah)`
  - `(?i)angin\s+(lemah|tenang|stagn[aai]n)`
  - `(?i)BMKG.{0,80}(cuaca|prakiraan|peringatan|kualitas\s+udara)`
  - `(?i)pancaroba`
- English expansions (lower priority): `no rain`, `lack of rain`, `dry spell`, `heat wave`, `stagnant air`, `temperature inversion`, `humidity`, `low wind`

- Optional refinements:
  - For robustness, consider coupling generic weather terms with air quality tokens in patterns (e.g., `(ku(m))/hujan … (polusi|udara|PM2.5)`), to avoid pure weather forecasts without air‑quality relevance.
  - If the goal is episodic pollution, strengthen the definition in the NF schema (keywords + examples) to emphasize causality (e.g., “stagnant conditions trap PM2.5”), reducing non‑causal weather hits.

## Quick stats to validate changes

- Current NF occurrence share for natural_factors (all docs): ~8.5%
- Q3 monthly counts (docs with frame present): Jul 21, Aug 330, Sep 173 (absolute counts increased with volume)
- Keyword hit rate on LLM‑labeled passages: ~36.7% → baseline for monitoring after keyword/pattern expansion

## Next steps

1) Add the suggested Indonesian keywords/patterns for `natural_factors` to `configs/word_count/indonesia_airpollution_causes.yaml`.
2) Re-run the word_count app and compare:
   - doc coverage for natural_factors,
   - hit‑rate on NF‑labeled passages using `investigate_keywords_vs_narrativeframing.py --frame-id natural_factors`.
3) If desired, adjust NF schema examples to focus “natural_factors” on meteorology-as-cause (stagnation, inversion, lack of rain) rather than generic weather advisories.

---
Generated with repository data on the 2025‑10‑28 NF run.

