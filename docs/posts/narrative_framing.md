---
layout: post
permalink: /posts/narrative_framing/
title: Narrative Framing Analysis
subtitle: Exploring tools for Effective Advocacy
description: Exploring narrative framing workflows across air pollution and animal welfare corpora
image_caption: "Peinture (2012), by Pierre Soulages"
date: 2025-11-12 09:00:00 +0000
author: Hubert Thieriot
author_image: /assets/images/author.jpg
image: /assets/images/soulages.jpg
tags: [narrative-framing]
---


<div class="tldr">I prototyped a method to identify and track narrative framings across various corpora (e.g. news articles, TV news, radio programs, parliamentary debates, court decisions). The ambition is to support effective advocacy organisations in both their strategy and impact monitoring, through better understanding how issues are discussed, detecting trends and surfacing outlets/journalists to prioritize.

This post includes two illustrative examples: one on air pollution in India and one on animal welfare in European political manifestos.
</div>



## Why narrative framing analysis?
Part of the motivation behind this series comes from a slight discomfort with the concept of “shaping the narrative” sometimes found in Theories of Change. The concept seems slippery to me and risks turning advocacy into a chase for mentions, mistaking visibility for influence. Still, I wondered whether I was being unfair, and sought ways to better assess its effectiveness.

That led me to narrative framing analyses as a potential **Monitoring, Evaluation, and Learning (MEL)** tool: could we track how the stories around an issue evolve, and whether advocacy efforts actually move the needle? Furthermore, if we can map how narratives differ across regions or outlets, that same information could guide **strategy and prioritization** e.g. revealing where certain framings already align with the desired change or where there is an opportunity to fill a narrative gap.


<div style="text-align: center; margin: 2em 0;">
  <figure style="margin: 0;">
    <img src="{{ site.baseurl }}/assets/narrative_framing_intervention_diagram.svg" style="max-width: 100%; height: auto;">
    <figcaption>How Narrative Framing Analysis informs advocacy intervention cycles
    </figcaption>
  </figure>
</div>

### What can narrative framing help with?

- **Understand how a topic is being discussed**: Every issue carries multiple possible stories: who is responsible, who suffers, and what counts as a solution. A framing analysis helps reveal which of these stories dominate, and which remain marginal or absent. It could matter because public narratives can influence which kinds of solutions receive attention or legitimacy (caveat: I haven't look at the evidence on the connection between framing and policy outcomes). For example, if air pollution coverage in Delhi overemphasizes individual behavior while neglecting industrial, power, or agricultural sources, it signals not just a bias in media attention but a structural blind spot in public debate.

- **Prioritise regions, messaging and outlets**: Comparing how narratives differ across outlets or regions can reveal where certain perspectives are missing — or, conversely, where the conversation already aligns with desired change. This information could eventually help advocates or funders decide where to focus their attention: in some cases by addressing narrative gaps, and in others by building on more conducive framings.

- **Measure change over time**: Tracking how narratives evolve — across repeated studies or advocacy campaigns — could help observe whether certain framings gain or lose prominence. This might support both strategic reflection (for advocates seeking feedback on their efforts) and broader research on how public conversations shift around key issues.

### What it can't (yet) do?

Narrative framing analysis focuses on message content — how issues are discussed in media and public discourse, whether in newspapers, TV, radio, social platforms, or political debates. It doesn’t directly tell us how these narratives shape what advisors, experts, or citizens think, or whether they ultimately influence decisions and policy. For now, it is a way to observe the stories circulating in public space.

<div style="text-align: center; margin: 2em 0 0 0;">
  <figure style="margin: 0;">
    <img src="{{ site.baseurl }}/assets/narrative_framing_scope_diagram.svg" style="max-width: 100%; height: auto;">
    <figcaption>
      The dashed box highlights what narrative framing analysis covers. The influence pathways are not directly measured by this method.
    </figcaption>
  </figure>
</div>

To see what this might look like in practice, I ran two small experiments. One asks how India's largest English-language newspapers talk about air pollution — who gets blamed, and who doesn't. The other examines how European political parties frame animal welfare in their manifestos.

## Example 1: Air pollution causes in Delhi, India

In this first exploration, I looked at how leading English-language newspapers discuss air pollution in Delhi, particularly which **sources of pollution** are mentioned. Such application could be used to highlight any discrepancy between the overal weight of sources in media framing and their actual contribution to air pollution as estimated by source apportionment studies, and in turn inform research and communication strategies. Vital Strategies had conducted similar [analysis](https://www.vitalstrategies.org/resources/through-the-smokescreen/) in the past, though with a different technique.

I collected 20,000 articles published between January 2015 and November 2025 across five prominent national newspapers (The Times of India, Hindustan Times, The Hindu, The Indian Express, and The New Indian Express). All pieces were written in English, but the same method can be applied to virtually any language.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Varying attention to air pollution in Delhi</div>
    <div class="chart-subtitle">Number of articles per day - 30-day window average</div>
  </div>
  <img src="{{ site.baseurl }}/assets/narrative_framing/delhi_airpollution_selected_newspapers/article_volume_over_time.svg" style="width: 100%; height: auto;">
  <p class="chart-note">
    <strong>Data Source:</strong> Articles are collected via MediaCloud using keywords related to air pollution and filtered to pieces that mention Delhi, New Delhi, NCR, or the National Capital Region. The corpus is limited to the five newspapers listed above.
  </p>
</div>

To analyse this corpus, we follow a six-step process: chunking, frame induction, frame annotation, model training, classification and aggregation (for more details, see [Methodology section below](#method-overview)).


```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#e8f1f8','primaryTextColor':'#1E3D58','primaryBorderColor':'#4f6081','lineColor':'#4f6081','secondaryColor':'#d4e3f0','tertiaryColor':'#fff','fontFamily':'Open Sans, Arial, sans-serif'}}}%%
flowchart LR
    A[Semantic Chunking] --> B[Frame Induction]
    B --> C[Frame Annotation]
    C --> D[Model Training]
    D --> E[Classification]
    E --> F[Aggregation]
    
    style A fill:#f7f9fc,stroke:#4f6081,stroke-width:2.5px,color:#1E3D58,rx:5,ry:5
    style B fill:#eff4f9,stroke:#4f6081,stroke-width:2.5px,color:#1E3D58,rx:5,ry:5
    style C fill:#e7eff6,stroke:#4f6081,stroke-width:2.5px,color:#1E3D58,rx:5,ry:5
    style D fill:#dfe9f3,stroke:#4f6081,stroke-width:2.5px,color:#1E3D58,rx:5,ry:5
    style E fill:#d7e4f0,stroke:#4f6081,stroke-width:2.5px,color:#1E3D58,rx:5,ry:5
    style F fill:#cfdeed,stroke:#4f6081,stroke-width:2.5px,color:#1E3D58,rx:5,ry:5
    
    linkStyle default stroke:#4f6081,stroke-width:2.5px
```
<figcaption>Schematic view of the analysis workflow</figcaption>

First, each article was split into smaller, coherent segments called **chunks** using linguistic models. Next comes the **frame induction** in which an LLM is used to identify the main frames that appear across a sample of chunks. The idea is to let the model propose a compact set of recurring ways the issue is discussed. This can be done with little or no guidance (letting the model discover patterns freely) or with direction around a specific question. In this case, I guided it toward sources of air pollution, which resulted in eight frames. The model then generated short descriptions, examples, and keywords for each, forming the schema used in the next steps.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Frames used for classifying air pollution sources in Delhi coverage</div>
    <div class="chart-subtitle">Definitions for each framing as determined by the framing induction step</div>
  </div>
{% include narrative_framing/delhi_airpollution_selected_newspapers/frames_light.html %}
<div class="chart-note">
    <strong>Note:</strong> These frames and their definitions were generated by an LLM-based "frame inducer," which generated names, examples, keywords and semantic cues for each frame. The inducer was provided with a sample of 200 passages from the dataset and guided with plain-text instructions focusing on distinct air pollution sources.
  </div>
</div>

Once the frames were defined, I used a lighter language model (GPT-4.1-mini) to **annotate** a few thousand text segments according to those categories. These labelled examples then served to **train** a BERT-based classifier that could scale the analysis to the full corpus. In the final step, each chunk was **classified** according to the likelihood of each frame and then aggregated by article, year, or outlet, weighting by text length and article to estimate how much attention each framing received over time. This led to the results shown in the image below.



<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">How Delhi newspapers frame the city's pollution sources</div>
    <div class="chart-subtitle">This chart shows the sources of air pollution mentioned in articles about Delhi and the National Capital Region between January 2015 and October 2025.</div>
  </div>
  <img src="{{ site.baseurl }}/assets/narrative_framing/delhi_airpollution_selected_newspapers/yearly_weighted_woz_annotated.svg" style="width: 100%; height: auto;">
  <p class="chart-note">
    <strong>Note:</strong> The analysis identifies air pollution sources through natural language processing of English-language articles from the five selected newspapers. Each source category (vehicles, industry, crop burning, etc.) is identified through frame classification of article content. The chart shows the relative frequency of mentions for each pollution source across all analyzed articles, weighted by article length to reflect the prominence of each frame in the coverage.
    <br><br>
    <strong>Data Sources:</strong> Articles are retrieved from MediaCloud. Content has been scraped and processed locally for analysis.
    <br><br>
    <strong>Disclaimer:</strong> These results are for demonstration purposes only. The analysis should not be relied upon to provide accurate estimates of media framing trends. Further validation and methodological refinement are needed before these results can be used for research or policy purposes.
  </p>
</div>



As can be seen, public conversations about air pollution in Delhi focus heavily on stubble burning, traffic and dust, while the year-round emissions from industry coal-fired power plants receive relatively less attention. This imbalance is reinforced by official narratives that downplay the sector’s role, such as CSIR-NEERI’s [claim](https://www.downtoearth.org.in/pollution/where-is-indias-so-control-from-tpps-headed-niti-aayogs-memo-over-fgds-fuels-debate) that SO₂ from power plants is “not significantly affecting ambient air quality” and the environment ministry's assertion that sulphate aerosols account for only up to 5% of PM2.5. This happens in the context of repeatedly [relaxed deadlines](https://healthpolicy-watch.news/india-reverses-key-policy-exempting-most-coal-fired-power-plants-from-emission-rules/) for pollution-control equipment.

[CREA](https://energyandcleanair.org/publication/enforcing-so2-norms-in-indias-coal-power-plants-is-non-negotiable/) and independent [researchers](https://www.healtheffects.org/system/files/GBD-MAPS-SpecRep21-India-revised_0.pdf) highlight that these conclusions ignore broader evidence that coal burning contributes 10–15 % of India’s total PM2.5 burden, while universal installation of flue-gas-desulphurisation systems could prevent tens of thousands of premature deaths. 

Bringing the power sector’s true impact into greater public prominence is potentially an effective way to increase regulatory pressure on one of Delhi’s most significant but least-discussed pollution sources. If so, narrative framing analysis could be used to **track progress** in that direction by providing a measurable **intermediate outcome**.


Such analysis could also be conducted at the outlet level -- for instance to prioritise outreach or identify biases or external influences. In the Delhi case, the distribution seems consistent across all selected media outlets, as shown in the figure below.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Frame distribution across media outlets</div>
    <div class="chart-subtitle">Share of each pollution source frame by newspaper, weighted by content length</div>
  </div>
  <img src="{{ site.baseurl }}/assets/narrative_framing/delhi_airpollution_selected_newspapers/domain_frame_distribution.svg" style="width: 100%; height: auto;">
  <p class="chart-note">
    <strong>Disclaimer:</strong> These results are for demonstration purposes only. The analysis should not be relied upon to provide accurate estimates of media framing trends. Further validation and methodological refinement are needed before these results can be used for research or policy purposes.
  </p>
</div>



## Example 2: Animal welfare in European political manifestos

This second application shifts to political discourse by analysing political manifestos. Political manifestos are programmatic documents that parties commit to during elections. Understanding which parties prioritize animal welfare, and how they frame it, could help advocates identify allies, track the evolution of party positions across election cycles, and compare political landscapes across countries.

In this quick analysis, I focused on the various framings/dimensions of animal welfare and the extent to which they are present in parties programmatic documents. Manifestos were retrieved from [Manifesto Project](https://manifestoproject.wzb.eu/) dataset. I focused on European manifestos published since 2018, spanning 36 countries and 29 languages.


<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Animal welfare in European political manifestos</div>
    <div class="chart-subtitle">Left: Overall attention to animal welfare by country. Right: Dominant frame by country.</div>
  </div>
  <div style="display: flex; gap: 16px; flex-wrap: wrap;">
    <div style="flex: 1; min-width: 300px;">
      <img src="{{ site.baseurl }}/assets/narrative_framing/manifesto_europe_animalwelfare/europe_map.png" class="modal" style="width: 100%; height: auto;">
    </div>
    <div style="flex: 1; min-width: 300px;">
      <img src="{{ site.baseurl }}/assets/narrative_framing/manifesto_europe_animalwelfare/europe_dominant_frame.png" class="modal" style="width: 100%; height: auto;">
    </div>
  </div>
  <p class="chart-note">
    <strong>Note:</strong> The left map shows the share of manifesto content dedicated to animal welfare topics (darker = more attention). The right map shows which specific frame dominates in each country's political discourse, excluding the generic "General animal welfare" frame to highlight more specific policy framings. Countries in grey have insufficient data.
  </p>
</div>

The maps reveal notable geographic patterns. Nordic and Western European countries — particularly Sweden, Iceland, and the United Kingdom — dedicate the most manifesto space to animal welfare. The dominant frames also vary: while most countries focus on general animal welfare statements or anti-hunting discourse, the Netherlands stands out for its emphasis on factory farming (reflecting ongoing debates about intensive agriculture), and countries like Finland and Spain focus more on pet and companion animal policies.


<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Frames used for classifying animal welfare discourse in European manifestos</div>
    <div class="chart-subtitle">Definitions for each frame category</div>
  </div>
  {% include narrative_framing/manifesto_europe_animalwelfare_v2/frames_light.html %}
  <div class="chart-note">
    <strong>Note:</strong> These nine frames were defined to capture distinct aspects of animal welfare discourse in political manifestos, on top of a general animal welfare angle.
  </div>
</div>

Unsurprisingly, Green parties tend to be amongst the "top-performers". But mainstream parties also appear in the rankings top, suggesting that animal welfare has broader political salience in some countries.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Parties with highest attention to animal welfare</div>
    <div class="chart-subtitle">Share of manifesto content dedicated to animal welfare topics (top 30 parties)</div>
  </div>
  <img src="{{ site.baseurl }}/assets/narrative_framing/manifesto_europe_animalwelfare_v2/all_parties.png" style="width: 100%; height: auto;">
  <p class="chart-note">
    <strong>Note:</strong> The chart shows the share of each party's manifesto dedicated to animal welfare topics, based on frame classification of manifesto text. Only parties with detectable animal welfare content are shown.
    <br><br>
    <strong>Data Source:</strong> Manifestos retrieved via the <a href="https://manifestoproject.wzb.eu/">Manifesto Project API</a>. Content processed locally for frame classification.
    <br><br>
    <strong>Disclaimer:</strong> These results are for demonstration purposes only. The analysis should not be relied upon to provide accurate estimates of party positions. Further validation and methodological refinement are needed before these results can be used for research or policy purposes.
  </p>
</div>



## Looking forward
These early experiments only scratch the surface of what narrative analysis could do for advocacy and research. Going forward, several directions seem worth exploring:

- **Other mediums**: these examples have covered media articles and political manifestos, but the same approach could be extended to TV and radio transcripts, social media, parliamentary debates and court decisions.
- **Valence and stance**: Understanding how issues are discussed matters as much as whether they are mentioned. Adding sentiment or stance detection could help distinguish between supportive, neutral, and dismissive framings.
- **Conditional framing**: Beyond tracking which frames appear, future work could look at how they co-occur.
- **Validation and reliability**: These are exploratory prototypes. Proper validation would probably require some manual annotation and validation, better uncertainty evaluation as well as better checks against overfitting.


<div class="text-box">
  <h3>Get in touch</h3>
I am interested in hearing from others working on similar problems or exploring how these tools could be applied in new contexts or further developed to be more useful. Whether you have ideas for improvements, questions about the approach, or want to collaborate on applications, I'd love to hear from you - <a href="mailto:hubert.thieriot@gmail.com">reach out to me</a>.
</div>



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

<div style="text-align: center; margin: 2em 0;">
  <img src="{{ site.baseurl }}/assets/narrative_framing_method_overview.svg" alt="Method overview: Data Collection, Frame Induction, Classification, and Aggregation" style="max-width: 100%; height: auto;">
</div>




**Content discovery**:
We start by defining the slice of content we care about, whether from media articles, TV news transcripts, radio programs, forums, social media, parlimentary debates, or even court decisions.

**Scrape and extract**:
We fetch pages and extract the main text, then remove boilerplate and navigation tails that otherwise drown the signal

**Chunking**:
We split documents into smaller chunks (~200 words) using spaCy language models. This linguistic approach respects sentence boundaries, paragraph structure, and discourse connectors (words like "however" or "therefore" that should stay attached to their preceding sentences). We use language-specific spaCy models to ensure proper sentence segmentation and preserve semantic coherence. By working with smaller units, we can detect when a single article discusses both vehicle emissions and industrial pollution, even if one frame dominates the overall document.

**Frame induction**:
We ask an LLM to propose a compact set of categories tailored to the question and context (e.g., causes of air pollution in Delhi) by feeding it a random sample of passages (200 passages in the examples above) in several consecutive batches, followed by a consolidation call. User can inject guidance to guide the LLM e.g. to include or exclude certain frames. After a manual and shallow comparison of various models performances through visual inspection of framing results, I selected OpenAI GPT‑4.1 for this step. The resulting schema (names, short definitions, examples, keywords) is passed along to the annotation step.

**Sample annotations**:
We then use another LLM as a probabilistic annotator on a sample of passages (typically 2,000 passages in the examples above). Each passage gets a distribution over frames (not just a single label) plus a brief rationale. We typically use a smaller GPT‑4 variant (e.g., `gpt-4.1-mini`) for this step to balance cost and quality, since we need to label thousands of examples. This does two things: it reveals ambiguous cases that keyword-based approaches would mis-label, and it gives us enough labeled data to train a supervised model.

**Supervised classifier**:
We then fine‑tune a multi‑label transformer classifier on those annotated passages using BERT-based models. We start with a pre-trained language model and adapt it to our frame classification task: the encoder layers learn to recognize frame-relevant patterns, while a new classification head outputs probability scores for each frame. In this first design, precision and recall typically stand above 0.85 and 0.75 respectively. Visual validation confirmed that the classifier correctly identifies frames in most cases.

**Classify the corpus**:
The model can then produce fast inference over tens of thousands of chunks.

**Aggregate and report**:
Finally, we aggregate chunk‑level predictions to document‑level profiles and summaries over time. A length‑weighted aggregator estimates how much attention each frame receives within a document.

<div class="text-box">
  <h3>Why not simply use keywords?</h3>
  
  <p>Keyword-based approaches are the simplest starting point for narrative analysis, and they can work reasonably well in some cases. In fact, quick comparisons suggested that a keyword-based approach might have produced broadly similar results for the Delhi example.</p>

  <p>That said, keywords come with several limitations that become more noticeable as the analysis grows in scope or complexity:</p>
  
  <ul>
    <li><strong>Paraphrases and semantic variations</strong>: the same idea can be described using different wording. A passage describing “pollution from manufacturing hubs on the city’s outskirts” belongs in the same frame as one mentioning “industrial emissions,” but simple keyword lists might miss one of those. Training a classifier with a large-enough sample size would tend to better capture such semantic variations and implied meaning.</li>

    <li><strong>Distinction of closely-related frames</strong>: Keywords may struggle to separate closely related frames (e.g., “factory farming” vs. “animal suffering,” or “natural factors” vs. “seasonal haze”) when such frames share important keywords. By better capturing the context in which these keywords are used, semantic models can better discriminate such frames.</li>

    <li><strong>Cross-language nuance</strong>: In multilingual settings, keywords require maintaining separate—and often imperfect—lists for each language. Semantic models are better at capturing equivalent ideas across languages and phrasing variations without manual translation work.</li>

    <li><strong>Extensibility</strong>: Semantic models also enable deeper analyses such as named-entity recognition (to identify actors), sentiment or valence (the emotional tone or intensity of coverage), and stance detection (whether the text supports, opposes, or remains neutral toward a specific actor or proposal). These layers are difficult or brittle to implement with keyword rules alone.</li>
  </ul>
  
  <p>In short, keyword approaches are simple and sometimes effective, but semantic methods scale better, generalize across phrasing and languages, and create a base for more granular analyses.</p>


</div>
