---
layout: default
title: Narrative Framing Analysis
description: Exploring Tools for Effective Advocacy
---
# Narrative Framing for **Air Pollution** and **Animal Welfare**

<div class="tldr">I prototyped a method to identify and track narrative framings across various corpora (e.g. media, TV news articles, radio programs, forums). The ambition is to support effective advocacy in their strategy and impact monitoring, through better understanding how issues are discussed, detecting trends and shifts and surfacing outlets/journalists to prioritize.

This post includes two illustrative examples: one on air pollution in Indonesia and one on animal welfare in Canada.
</div>


<div class="disclaimer">
  This post is part of a series of technical explorations for <strong>Effective Advocacy</strong>. The goal is to devise practical tools that help advocacy better inform their strategy and measure their impact. Anticipated applications include narrative framing analysis, strategic actors mapping, and key findings dissemination tracking.
</div>

## Why narrative framing?
Narrative framing analyses could serve multiple purposes:

- **Understand how a topic is being discussed**: Identify which narratives dominate coverage versus those that remain marginal or absent. This helps advocates understand the current discourse landscape—what frames are being amplified, which perspectives are under-represented, and where there might be opportunities to shift narratives. For example, an analysis might reveal that air pollution discussions focus heavily on individual vehicle emissions while neglecting industrial sources, pointing to potential advocacy gaps.

- **Prioritise regions, messaging and outlets**: Use data to identify gaps and opportunities—which regions or outlets are influential but missing key narratives, or which audiences lack exposure to critical frames. For instance, an analysis might reveal that certain media outlets heavily cover industrial pollution but rarely connect it to policy solutions, indicating a clear messaging gap. This enables more targeted grantmaking and campaign design, ensuring resources go to outlets and regions where narrative shifts are most needed and feasible.

- **Measure change over time**: Track how narratives evolve before, during, and after advocacy campaigns or major events. Detect whether specific frames are gaining or losing traction, measure campaign impact by comparing pre- and post-intervention coverage, and identify emerging trends early. This provides evidence-based feedback loops for grantees and helps demonstrate the effectiveness of narrative change initiatives.


## Example 1: Air pollution causes in Jakarta, Indonesia

In this example, I was interested in tracking how Indonesia media talk about air pollution, especially which sources of air pollution are mentioned more than others. Such application could be used for instance to highlight any discrepancy between the overal weight of sources in media framing and their actual contribution to air pollution as estimated by source apportionment studies.


Using MediaCloud and Scrapy, I identified and scraped 15,000 media articles dating from January 2020 to October 2025 in Indonesian media that cover air pollution in Jakarta. Most of the articles were in Bahasa Indonesia.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Article Volume Over Time</div>
    <div class="chart-subtitle">Articles per day (30-day avg)</div>
  </div>
  <iframe src="{{ site.baseurl }}/assets/narrative_framing/indonesia_airpollution/article_volume_over_time.html" style="width: 100%; height: 450px; border: none;"></iframe>
</div>




Transport emissions dominate coverage (41% of articles), reflecting Jakarta's heavy traffic and vehicle-related pollution discourse. Natural and meteorological factors come next with a score of 8.5% articles, with notable seasonal spikes during dry periods when weather conditions exacerbate pollution.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">How Indonesian media frames the sources of the capital's pollution</div>
    <div class="chart-subtitle">This chart shows the sources of air pollution mentioned in media articles about air pollution in Jakarta or the greater metropolitan area. The analysis is based on articles published between 2020 and 2025 in Indonesian media, weighted by content length.</div>
  </div>
  <iframe src="{{ site.baseurl }}/assets/narrative_framing/indonesia_airpollution/yearly_weighted_wz.html" style="width: 100%; height: 500px; border: none;"></iframe>
  <p class="chart-note">
    <strong>Note:</strong> The analysis identifies air pollution sources through natural language processing of Indonesian media articles. Articles are included if they mention Jakarta, DKI, ibukota, or jabodetabek and contain keywords related to air pollution. Each source category (vehicles, industry, forest fires, etc.) is identified through frame classification of article content. The chart shows the relative frequency of mentions for each pollution source across all analyzed articles, weighted by article length to reflect the prominence of each frame in the coverage.
    <br><br>
    <strong>Data Sources:</strong> The list of articles is retrieved from Media Cloud. Content has been scraped and processed locally for analysis.
  </p>
</div>


### Frames

| Frame | Description | Key Keywords | Share |
|-------|-------------|--------------|-------|
| **Transport Emissions** | Vehicle emissions from cars, motorcycles, buses, trucks, and road traffic | kendaraan bermotor, lalu lintas, emisi kendaraan, uji emisi | 41.1% |
| **Natural Factors** | Meteorological and seasonal factors affecting air quality (weather patterns, El Niño, rainfall) | cuaca, angin, musim kemarau, El Nino, curah hujan rendah | 8.5% |
| **Industrial Emissions** | Factory and manufacturing emissions, including smelters, steel, and cement production | pabrik, industri, smelter, industri baja, industri semen | 6.3% |
| **Power Plant Emissions** | Coal-fired and fossil-fuel power plant emissions | PLTU, pembangkit listrik, coal-fired power plant | 3.3% |
| **Biomass Burning** | Agricultural fires, forest fires, and land clearing through burning | pembakaran lahan, kebakaran hutan, pembakaran biomassa | 2.1% |
| **Waste Burning** | Open burning of municipal waste and landfill fires | pembakaran sampah, open burning, landfill fire | 1.9% |
| **Household Emissions** | Household cooking and heating using fossil fuels or biomass | pembakaran rumah tangga, kompor kayu, bahan bakar padat | 0.5% |
| **Construction Dust** | Construction activities, roadworks, and resuspended dust | debu konstruksi, pembangunan, road dust, pekerjaan jalan | 0.4% |

*Note: Percentages represent the share of articles that discuss each frame (occurrence-based, threshold ≥0.2). Articles can discuss multiple frames.*


### Media outlets breakdown

The analysis reveals how different media outlets frame air pollution sources. Some outlets emphasize certain pollution sources more than others, which can inform advocacy targeting and messaging strategies.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Frame distribution across media outlets</div>
    <div class="chart-subtitle">Share of each pollution source frame by media outlet, weighted by content length</div>
  </div>
  <iframe src="{{ site.baseurl }}/assets/narrative_framing/indonesia_airpollution/domain_frame_distribution.html" style="width: 100%; height: 900px; border: none;"></iframe>
</div>



## Example 2: Animal welfare in Canada




## Method overview

The pipeline follows a hybrid LLM-to-classifier approach: we start with flexible LLM exploration to discover and annotate narrative frames, then scale up with a fine-tuned transformer classifier. This balances domain adaptability (frames tailored to each question and context) with computational efficiency (fast inference over large corpora).







<style>
.mermaid svg {
  max-width: 100%;
  width: 100%;
  height: auto;
}
.mermaid .cluster text,
.mermaid .cluster-label text {
  font-weight: bold;
}
.mermaid .cluster-label {
  margin-bottom: 8px;
}
</style>

<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  (function(){
    function log(){
      if (!window.console) return;
      var args = Array.prototype.slice.call(arguments);
      args.unshift('[mermaid-init]');
      try { console.log.apply(console, args); } catch(e) { console.log(args.join(' ')); }
    }

    function upgradeCodeBlocks() {
      var selector = 'pre code.language-mermaid, pre code.mermaid, code.language-mermaid';
      var nodes = document.querySelectorAll(selector);
      log('found mermaid code blocks:', nodes.length);
      nodes.forEach(function(code){
        var pre = code.closest && code.closest('pre');
        var container = pre || code.parentNode;
        var parent = container.parentNode;
        var txt = code.textContent || '';
        var div = document.createElement('div');
        div.className = 'mermaid';
        div.style.width = '100%';
        div.style.overflowX = 'auto';
        div.textContent = txt;
        try {
          parent.replaceChild(div, container);
        } catch (e) {
          log('replaceChild failed', e);
        }
      });
    }

    function initMermaid(){
      log('init called. mermaid present:', !!window.mermaid);
      if (!window.mermaid) return;
      try {
        upgradeCodeBlocks();
      } catch (e) {
        log('upgrade error:', e);
      }
      try {
        window.mermaid.initialize({ 
          startOnLoad: false, 
          theme: 'default', 
          securityLevel: 'loose',
          flowchart: { useMaxWidth: false, htmlLabels: true }
        });
        var targets = document.querySelectorAll('.mermaid');
        log('rendering targets:', targets.length);
        window.mermaid.init(undefined, targets);
        
        // Apply styling after rendering completes
        function applyStyling() {
          targets.forEach(function(target) {
            var svg = target.querySelector('svg');
            if (svg) {
              // Find all clusters (subgraphs)
              var clusters = svg.querySelectorAll('g.cluster');
              clusters.forEach(function(cluster) {
                var rect = cluster.querySelector('rect');
                var titleText = cluster.querySelector('text');
                
                if (rect && titleText) {
                  // Get rectangle position and dimensions
                  var rectY = parseFloat(rect.getAttribute('y')) || 0;
                  var rectHeight = parseFloat(rect.getAttribute('height')) || 0;
                  
                  // Apply rounded corners
                  rect.setAttribute('rx', '8');
                  rect.setAttribute('ry', '8');
                  
                  // Move title above the box (position it above the top edge of the rectangle)
                  // Add some spacing between title and box
                  var titleAboveY = rectY - 8;
                  titleText.setAttribute('y', titleAboveY);
                  
                  // Make sure the title is bold
                  titleText.style.fontWeight = 'bold';
                }
              });
            }
          });
        }
        
        // Try immediately and again after a delay to catch async rendering
        applyStyling();
        setTimeout(applyStyling, 100);
        setTimeout(applyStyling, 300);
        log('render complete');
      } catch (e) {
        log('mermaid init error:', e);
      }
    }

    log('script loaded. mermaid present:', !!window.mermaid);
    document.addEventListener('DOMContentLoaded', initMermaid);
    document.addEventListener('pageshow', initMermaid);
    document.addEventListener('pjax:end', initMermaid);
    document.addEventListener('turbolinks:load', initMermaid);
    document.addEventListener('turbo:load', initMermaid);
  })();
</script>

```mermaid
flowchart LR
    subgraph Collection["1. Data Collection & Preparation"]
        direction TB
        subgraph CollectionSub[ ]
        direction TB
        A["Content discovery<br/>(media, TV, radio, forums, etc.)"] 
        A2["Scrape & extract text<br/>(using Scrapy)"]
        B["Chunk text<br/>(using SpaCy language model)"]
        A --> A2 --> B
        end
    end
    
    subgraph Discovery["2. Frame Induction & Annotation"]
        direction TB
        subgraph DiscoverySub[ ]
        direction TB
        C["LLM: Induce frames<br/>(with or without user guidance)"]
        D["LLM: Label samples<br/>(multi-label distributions)"]
        C --> D
        end
    end
    
    

    subgraph Classification["3. Scalable Classification"]
        direction TB
        subgraph ClassificationSub[ ]
        direction TB
        E["Train transformer classifier<br/>(fine-tune on LLM labels)"]
        F["Classify all chunks<br/>(fast inference)"]
        E --> F
        end
    end
    
    subgraph Analysis["4. Aggregation & Reporting"]
        direction TB
        subgraph AnalysisSub[ ]
        direction TB
        G["Aggregate to document level<br/>(length-weighted attention)"]
        H["Results analysis<br/>(e.g. time series & outlets breakdowns)"]
        I["Generate reports<br/>(interactive HTML + static plots)"]
        G --> H --> I
        end
    end
    
    Collection --> Discovery
    Discovery --> Classification
    Classification --> Analysis
    
    classDef nodeBox fill:#ffffff33,stroke:#333,stroke-width:1px
    classDef somePaddingClass padding-bottom:5em
    classDef transparent fill:#ffffff00,stroke-width:0
    
    Collection:::somePaddingClass
    CollectionSub:::transparent
    Discovery:::discoveryStyle
    Discovery:::somePaddingClass
    DiscoverySub:::transparent
    Classification:::somePaddingClass
    ClassificationSub:::transparent
    Analysis:::analysisStyle
    Analysis:::somePaddingClass
    AnalysisSub:::transparent

    
    class A,A2,B,C,D,E,F,G,H,I nodeBox
    style Collection fill:#e1f5ff,stroke:#0277bd,stroke-width:2px
    style Discovery fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style Classification fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Analysis fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    
```









**Content discovery (search/filters)**:
We start by defining the slice of content we care about—whether from media articles, TV news transcripts, radio programs, forums, Reddit, or other sources—in a way that is both broad enough to catch variation and precise enough to be actionable. For media analysis, using Media Cloud collections lets us anchor each run in a country and time window, and then layer topical filters (for instance, city names or issue cues) to focus coverage. Similar approaches work for other platforms: TV news and radio transcripts, forum posts, Reddit threads, or other text corpora can be collected through their respective APIs or scraping tools. The intent is to bias toward recall at this stage: we would rather include a few borderline documents and filter them downstream than miss legitimate phrasing that differs from our initial keywords. Every run is captured in a small YAML file so the choices are explicit and replicable.

**Scrape and extract**:
To reason about narratives we need full passages, not just headlines or snippets. We fetch pages and extract the main text, then remove boilerplate and navigation tails that otherwise drown the signal (things like widgets, "follow us" blocks, or stock tickers). The trimming rules live in config so we can adapt them by outlet or country. This step trades a little engineering effort for cleaner inputs and more stable downstream classification.

**Chunking**:
We split documents into smaller chunks (~200 words) using spaCy language models. This linguistic approach respects sentence boundaries, paragraph structure, and discourse connectors (words like "however" or "therefore" that should stay attached to their preceding sentences). We use language-specific spaCy models (e.g., `en_core_web_sm` for English, `id_core_news_sm` for Bahasa Indonesia) to ensure proper sentence segmentation and preserve semantic coherence.

Chunking and annotating at this granularity is essential because long documents often contain multiple frames, and classifying at the document level would bury weaker or less prominent frames. By working with smaller units, we can detect when a single article discusses both vehicle emissions and industrial pollution, even if one frame dominates the overall document.

**Frame induction (LLM)**:
We ask an LLM to propose a compact set of categories tailored to the question and context (e.g., causes of air pollution in Jakarta) by feeding it a random sample of passages (200 passages in the examples above) in several consecutive batches, followed by a consolidation call. User can inject guidance to guide the LLM e.g. to include or exclude certain frames. After a manual and shallow comparison of various models performances through visual inspection of framing results, I selected OpenAI GPT‑4.1 for this step. The resulting schema (names, short definitions, examples, keywords) is passed along to the annotation step.

**Frame application to samples (LLM)**:
We then use another LLM as a probabilistic annotator on a sample of passages (typically 2,000 passages in the examples above). Each passage gets a distribution over frames (not just a single label) plus a brief rationale. We typically use a smaller GPT‑4 variant (e.g., `gpt-4.1-mini`) for this step to balance cost and quality, since we need to label thousands of examples. This does two things: it reveals ambiguous cases that keyword-based approaches would mis-label, and it gives us enough labeled data to train a supervised model.

**Supervised classifier (transformers)**:
We then fine‑tune a multi‑label transformer classifier on those LLM‑labeled passages using Hugging Face transformers. We start with a pre-trained language model (e.g., `indobenchmark/indobert-base-p1` for Bahasa Indonesia, `distilbert-base-uncased` for English) and adapt it to our frame classification task: the encoder layers learn to recognize frame-relevant patterns, while a new classification head outputs probability scores for each frame using sigmoid activation. This gives us cheap, fast inference over tens of thousands of chunks while freezing the labeling policy defined by the schema.

**Classify the corpus**:
We classify content at the chunk level (typically sentences or short spans) to avoid burying weaker frames in long documents. Light keyword gating and regex excludes from earlier steps help keep us on topic without reintroducing brittle rules. Results are cached per document to support iterative runs and easy re‑aggregation.

**Aggregate and report**:
Finally, we aggregate chunk‑level predictions to document‑level profiles and summaries over time. A length‑weighted aggregator estimates how much attention each frame receives within a document (article, post, thread, etc.); an occurrence view answers a different question—what share of documents mention a frame at all.

<div class="text-box">
  <strong>Why not simply use keywords?</strong>
  
  <p>Keyword-based approaches have significant limitations for narrative analysis:</p>
  
  <ul>
    <li><strong>Paraphrases and semantic variations</strong>: Journalists describe the same concept in many ways. An article discussing "road dust resuspension from heavy traffic" contains the same frame as one mentioning "vehicle emissions," but keyword matching would miss this connection.</li>
    
    <li><strong>Language evolution</strong>: Terms change over time and across regions. What counts as "construction pollution" in Jakarta might be discussed as "infrastructure development impacts" elsewhere, requiring constant keyword list updates.</li>
    
    <li><strong>Implied meaning</strong>: Media often conveys frames through context rather than explicit terms. A passage describing "stagnant air during the dry season" implies natural factors affecting pollution, even without using keywords like "El Niño" or "low rainfall."</li>
    
    <li><strong>Cross-language nuance</strong>: In multilingual contexts, keywords must be translated carefully, but semantic understanding captures equivalent concepts across languages automatically.</li>
  </ul>
  
  <p>Our approach uses LLMs to capture semantic meaning, then scales it with a classifier—combining the flexibility of language understanding with the efficiency needed for large-scale analysis.</p>
</div>

---

## Get in touch
I am interested in hearing from others working on similar problems or exploring how these tools could be applied in new contexts or further developed to be more useful. Whether you have ideas for improvements, questions about the approach, or want to collaborate on applications, I'd love to hear from you - [reach out to me](mailto:hubert.thieriot@gmail.com).

