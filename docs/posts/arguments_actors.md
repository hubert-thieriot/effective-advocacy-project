---
layout: post
permalink: /posts/arguments_actors/
title: Arguments & Actors Mapping
subtitle: Exploring tools for Effective Advocacy
description: Mapping key arguments, media trends, and actor coalitions in the EU alternative proteins debate
date: 2025-12-17 09:00:00 +0000
last_modified_at: 2025-12-17 00:00:00 +0000
author: Hubert Thieriot
tags: [discourse-analysis, actors-mapping]
---


<div class="tldr">I prototyped a method to identify the key arguments for and against alternative proteins in EU media, map which actors champion or contest them, and measure how positions shift across outlets and over time. The approach aims to give advocates the intelligence and the lead time needed to prepare responses or shape the conversation.
</div>



# Why map arguments and actors?

[Narrative framing analysis]({{ '/posts/narrative_framing/' | relative_url }}) reveals *how* an issue is discussed — which themes and angles dominate coverage. Here, we go a step further and identify **what specific positions or claims are being taken**, and **by whom** — how arguments for and against a position are distributed across actors and coalitions.

**Early detection.** Continuously monitoring which arguments are gaining traction — and which actors are driving them — could flag emerging positions weeks or months before they become prominent, giving advocates the lead time to prepare responses or shape the conversation before it solidifies.

**Strategic orientation.** Which arguments are structurally central, concentrating the most alliances and conflicts? Which actors are influential within specific communities, and which remain on the periphery? Where are the gaps — positions that could be articulated but currently aren't? This kind of argument & coalition mapping could well inform communication strategy.

**Impact measurement.** The argument landscape can be mapped before and after an intervention — a campaign, a policy decision, a controversy — to detect whether anything shifted: new arguments emerging, actors moving, coalitions reconfiguring.


# Alternative proteins in European media

Alternative proteins — plant-based meat, cultivated meat, and fermentation-derived products — have become one of the more contested food policy questions in Europe. The debate sits at the intersection of several ongoing conflicts: EU climate policy (the Farm to Fork strategy explicitly promoted protein diversification), Common Agricultural Policy reform, food labelling disputes, public health concerns about ultra-processing, and industrial policy for the food tech sector. This makes it a particularly rich test case for arguments and actors mapping: the fault lines are relatively clear, the stakeholders are vocal, and the regulatory stakes are real.

In this demo analysis, the corpus covers around 10,000 articles referring to alternative protein, published betweem 2015 and 2025 (included) across 23 major European media outlets[^outlets], in seven languages. The analysis identifies statements attributed to named actors, induces a set of recurring claims from those statements, scores each statement's relationship to each claim (supports, opposes, or neutral), and then maps actor relationships based on shared and opposing positions. The method overview at the end of this post describes each step in more detail.


## The arguments: how is the debate structured?

Claims — specific, debatable positions that actors either support, oppose, or remain neutral on — were first induced from a sample of actor statements in the corpus, then refined into two families: **product claims** (what alternative proteins and conventional agriculture *are* and *do*) and **policy claims** (what governments and regulators *should do*).


<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Key claims identified in the European alternative proteins debate</div>
  </div>
  {% include arguments_actors/eu_alt_proteins/claims_cards.html %}
  <!-- <div class="chart-note">
    <strong>Note:</strong> These claims were induced by the pipeline from a sample of actor statements in the corpus, then used to score all identified statements.
  </div> -->
</div>


## Coalitions

These claims define an n-dimensional space in which each actor can be positioned based on their support or opposition to each proposition. Using clustering and dimensionality reduction, we can then map how actors form coalitions and which positions those coalitions are organised around.

{% include arguments_actors/eu_alt_proteins/coalition_pca.html %}

In this case, we detected three broad coalitions.




## How arguments trend over time and across outlets

- The time series shows how the balance of support and opposition for each claim evolves
- This is where the analysis becomes actionable: detecting whether a specific argument is gaining or losing traction
- **Early detection angle**: point to arguments that are now prominent but were barely visible N months/years ago — illustrating how the tool could have flagged them early, giving advocates lead time to respond

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Support and opposition to key claims over time</div>
    <div class="chart-subtitle">Distribution of supports / opposes / neutral statements over time</div>
  </div>
  <!-- PLACEHOLDER: claims_timeseries.html
       Interactive stacked area chart.
       X-axis = time (monthly or yearly). Y-axis = statement count.
       Three stacked areas: green (supports), red (opposes), grey (neutral).
       Dropdown or filter to select individual claims.
       Generated by report builder _render_claims_analysis (charts section)
  -->
  
  <p class="chart-note">
    <strong>Note:</strong> Each data point represents an agreement score between a statement attributed to a named actor and one of the induced claims. Statements are dated by their source article's publication date.
    <br><br>
    <strong>Disclaimer:</strong> These results are for demonstration purposes only. Further validation and methodological refinement are needed before drawing firm conclusions.
  </p>
</div>

- Commentary on visible trends: which arguments are growing, which are fading
- Highlight any emerging arguments that could be flagged early

- Cross-outlet comparison reveals editorial positioning and audience differences
- Which outlets host more supportive vs. oppositional discourse on alternative proteins

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">How arguments distribute across EU media outlets</div>
    <div class="chart-subtitle">Agreement distribution by outlet</div>
  </div>
  <!-- PLACEHOLDER: claims_domain_distribution.html
       Grouped bar chart. X-axis = media outlets (top N).
       Each outlet has bars for: supports (green), opposes (red), neutral (grey).
       Dropdown to filter by claim and top-N outlets.
       Generated by report builder _render_claims_analysis (domain section)
  -->
  
  <p class="chart-note">
    <strong>Note:</strong> Only outlets with a minimum number of scored statements are shown. The distribution reflects how actors quoted by each outlet position themselves on the claims.
    <br><br>
    <strong>Disclaimer:</strong> These results are for demonstration purposes only. Further validation and methodological refinement are needed before drawing firm conclusions.
  </p>
</div>

## Looking forward

- **Spot emerging arguments early**: detect claims gaining traction before they become mainstream, giving advocates time to prepare responses or preemptive messaging
- **Target outlet engagement**: identify which outlets host more supportive vs. oppositional discourse, guiding media outreach
- **Measure intervention impact**: repeat the analysis post-campaign to see if the balance of arguments shifted



- Extending to other policy domains beyond alternative proteins
- Combining with narrative framing analysis to get both "how an issue is framed" and "what positions actors take"
- Longitudinal tracking to measure whether coalitions shift over time in response to events or campaigns
- Validation against expert assessments and ground-truth actor positions


<div class="text-box">
  <h3>Get in touch</h3>
I am interested in hearing from others working on similar problems or exploring how these tools could be applied in new contexts or further developed to be more useful. Whether you have ideas for improvements, questions about the approach, or want to collaborate on applications, I'd love to hear from you - <a href="mailto:hubert.thieriot@gmail.com" target="_blank" rel="noopener">reach out to me</a>.
</div>



# Method overview

This analysis builds on the same foundational pipeline described in the [Narrative Framing Analysis]({{ '/posts/narrative_framing/' | relative_url }}) post. Shared steps — content discovery, scraping, text extraction, and chunking — are described there. This section covers the additional steps specific to arguments and actors mapping.

## Named Entity Recognition (NER)

- Each text chunk is processed with NER to identify persons and organisations mentioned
- Entity consolidation resolves variants of the same name (e.g. "Boris Johnson", "Johnson", "the PM") into a single canonical entity
- This connects text content to specific, identifiable actors

## Statement extraction

- For each chunk with identified entities, the pipeline extracts statements — what an actor says or is reported as saying
- Each statement is linked to its speaker (entity), source article, and publication date
- This creates the actor-to-text link that underlies the entire discourse network

## Claims induction

- Where the framing pipeline induces abstract frames (e.g. "agricultural burning", "vehicle emissions"), this pipeline induces specific policy positions or claims
- Claims are concrete and debatable: "alternative proteins should replace animal agriculture to reduce emissions" rather than "environmental impact"
- The induction process is similar (LLM-driven, from a sample of statements), but the output is a set of contestable positions rather than descriptive categories
- User guidance can steer the induction toward relevant debate axes

## Agreement scoring

- Each statement is scored against each claim: supports, opposes, or neutral
- This creates the statement-claim-agreement matrix that underlies everything
- Scored via LLM (e.g. GPT-4.1-mini) with confidence scores and rationales

## DNA network construction

- From the statement-claim-agreement data, build the actor-actor network
- Two actors are allies if they take the same position on a claim; enemies if they take opposing positions
- Edge weight = number of shared or opposing claims
- Community detection via Louvain algorithm identifies natural groupings
- Centrality scores (betweenness centrality) identify the most structurally important actors

## Coalition analysis

- Beyond pairwise relationships: represent each actor as a vector in claim space (their position on each claim)
- Project to 2D via PCA for visualization
- Cluster via k-means with silhouette score for optimal k selection
- Reveals broader coalitions and the claim positions that define them

## Remaining improvements

- OpEds not yet managed i.e. the attribution of statements to the article author is not yet fully managed

---

[^outlets]: **UK** — The Guardian, The Telegraph, The Independent, The Economist · **Ireland** — Irish Independent · **Pan-European / International** — EUobserver, Euractiv, Deutsche Welle · **Germany** — Süddeutsche Zeitung, Der Spiegel, Die Welt, Frankfurter Allgemeine Zeitung · **France** — Le Monde, Le Figaro, Les Echos · **Italy** — Corriere della Sera, La Repubblica, Il Sole 24 Ore · **Spain** — El País · **Netherlands** — NRC Handelsblad, de Volkskrant · **Poland** — Gazeta Wyborcza, Rzeczpospolita
