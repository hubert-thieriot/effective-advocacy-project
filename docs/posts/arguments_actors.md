---
layout: post
permalink: /posts/arguments_actors/
title: Arguments & Actors Mapping
subtitle: Exploring tools for Effective Advocacy
description: Mapping key arguments, media trends, and actor coalitions in the EU alternative proteins debate
date: 2026-03-04 09:00:00 +0000
# last_modified_at: 2025-12-17 00:00:00 +0000
author: Hubert Thieriot
tags: [discourse-analysis, actors-mapping]
---


<div class="tldr">I prototyped a method to extract the key arguments in a policy debate, identify which actors champion or contest them, and map how they cluster into coalitions. Applied here to the EU alternative proteins debate across 10,000 articles in seven languages, the approach is topic-agnostic and could potentially support several advocacy applications: detecting emerging arguments early, surfacing unexpected actors, producing regular landscape assessments, guiding media outreach, and measuring campaign impact.
</div>



# Why map arguments and actors?

[Narrative framing analysis]({{ '/posts/narrative_framing/' | relative_url }}) reveals *how* an issue is discussed — which themes and angles dominate coverage. Here, we go a step further and identify **what specific positions or claims are being taken**, and **by whom** — how arguments for and against a position are distributed across actors and coalitions.

**Early detection.** Continuously monitoring which arguments are gaining traction — and which actors are driving them — could flag emerging positions weeks or months before they become prominent, giving advocates the lead time to prepare responses or shape the conversation before it solidifies.

**Strategic orientation.** Which arguments are structurally central, concentrating the most alliances and conflicts? Which actors are influential within specific communities, and which remain on the periphery? Where are the gaps — positions that could be articulated but currently aren't? This kind of argument & coalition mapping could well inform communication strategy.

**Impact measurement.** The argument landscape can be mapped before and after an intervention — a campaign, a policy decision, a controversy — to detect whether anything shifted: new arguments emerging, actors moving, coalitions reconfiguring.


# Alternative proteins in European media

Alternative proteins — plant-based meat, cultivated meat, and fermentation-derived products — have become one of the more contested food policy questions in Europe. The debate sits at the intersection of several ongoing conflicts: EU climate policy (the Farm to Fork strategy explicitly promoted protein diversification), Common Agricultural Policy reform, food labelling disputes, public health concerns about ultra-processing, and industrial policy for the food tech sector. This makes it a particularly rich test case for arguments and actors mapping: the fault lines are relatively clear, the stakeholders are vocal, and the regulatory stakes are real.

In this demo analysis, the corpus covers around 10,000 articles referring to alternative proteins, published between 2015 and 2025 (inclusive) across 23 major European media outlets[^outlets], in seven languages. The analysis identifies statements attributed to named actors, induces a set of recurring claims from those statements, scores each statement's relationship to each claim (supports, opposes, or neutral), and then maps actor relationships based on shared and opposing positions. The method overview at the end of this post describes each step in more detail.


## The arguments: how is the debate structured?

In this approach, debates on a given topic are structured around *claims*: specific, debatable propositions that actors may support, oppose, or simply not address. Taken together, claims define an n-dimensional “stance space” that we can use to represent and compare actors.

In the present case, claims were first induced from a sample of actor statements in the corpus, then refined into two families: **product claims** (what alternative proteins and conventional agriculture *are* and *do*) and **policy claims** (what governments and regulators *should do*).

This is a critical design step: it defines both what we track and how we map actors & coalitions. The right set of claims should be informed by the analysis objective, and may differ radically depending on whether we are looking at broad-based campaigning or targeted lobbying.

The chart below shows the claims used in this demo analysis.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Key claims identified in the European alternative proteins debate</div>
  </div>
  {% include arguments_actors/eu_alt_proteins/claims_cards.html %}
  <!-- <div class="chart-note">
    <strong>Note:</strong> These claims were induced by the pipeline from a sample of actor statements in the corpus, then used to score all identified statements.
  </div> -->
</div>


## Coalitions: who aligns with whom?

With 16 claims and more than a hundred identified actors, the natural next question is whether actors cluster into coherent coalitions — groups that tend to support and oppose the same things. To find out, we use a technique from discourse network analysis: each actor is represented as a vector of their positions across all 16 claims (support = +1, oppose = −1, no clear stance = 0). We then compute the similarity between every pair of actors based on these stance profiles, and apply community detection to identify groups of actors whose positions systematically align.

The scatter plot below projects this 16-dimensional stance space down to two dimensions (via PCA), so that actors with similar positions appear close together. Each point is an actor; colours indicate the detected coalition.

{% include arguments_actors/eu_alt_proteins/interactive_pca.html %}

Three broad coalitions emerge. But what do they actually stand for? The heatmap below shows each coalition’s average position on each of the 16 claims — revealing what unites and what separates them.

{% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap.html %}


Feeding that table and the actors list to Claude Opus 4.6, I was able to quickly build some "user personas", reflecting the centroid of each coalition:

- **Coalition A (n=82) — Skeptics & traditionalists**
This coalition leans into the "ultra-processed/not real food" critique (0.39), favors stricter UPF regulation (0.26), transparent labeling (0.26), and values traditional food culture (0.32). It's skeptical of alt proteins' health claims (-0.33) and doubts both the sector's market potential (-0.24) and the transformative promise of food technology (-0.24). Top actors here include Ron DeSantis, Jim Pillen (both US politicians who've pushed back on cultivated meat), Marco Springmann, Van Tulleken, Marion Nestle, and Giorgia Meloni — a mix of food processing critics, nutrition academics, and political figures who've taken skeptical or protectionist positions.

- **Coalition B (n=83) — Innovation & market optimists**
The strongest signals here are enthusiasm for food technology's transformative potential (0.83) and the sector's market opportunity (0.86). It's moderately pro-environment (0.24) and rejects the ultra-processed framing (-0.13), but is essentially neutral on meat reduction (0.01), protecting farming (0.01), and traditional food culture (0.12). Top actors include Uma Valeti (Upside Foods), Didier Toubia (Aleph Farms), various startup founders and investors like Eitan Fischer, Benjamina Bollag, and Leonardo DiCaprio. This is largely the industry and investor voice — bullish on the technology and market, without pushing strongly on dietary change.

- **Coalition C (n=77) — Systemic change advocates**
This coalition combines pro-technology positions (0.48, 0.53) with a strong push for reduced meat consumption and plant-based diets (0.53). It's the most forceful in rejecting the ultra-processed framing (-0.69) and traditional food culture (-0.35), and the most positive on environmental (0.40) and health (0.39) claims. Top actors include Josh Tetrick (Eat Just/GOOD Meat), Patrick Brown (Impossible Foods), Bruce Friedrich (GFI), George Monbiot, Seren Kell, Chris Bryant, and Bill Gates — a mix of high-profile founders, advocates, and public intellectuals pushing for broader food system transformation.


Importantly, all actors shown here were identified automatically from the text — no names were pre-listed or manually selected. The pipeline extracts whatever named entities are attributed statements in the corpus. This means the method can surface actors an analyst might not have thought to look for, making it potentially useful for detecting emerging or unexpected voices in a debate. The full actor-level stance heatmap is available in the [appendix](#actor-level-stance-heatmap) below.



## Looking forward

The alternative proteins case above is a demonstration, but the method itself is topic-agnostic — it can be applied to any policy debate where actors take public positions. The corpus can also be tailored to the application: news media for public discourse, parliamentary records for legislative debates, social media for grassroots movements, or industry publications for sector-specific lobbying.

<style>
.highlight-marker {
  background: linear-gradient(104deg, rgba(255,100,100,0) 0.9%, rgba(255,100,100,0.25) 2.4%, rgba(255,100,100,0.2) 5.8%, rgba(255,100,100,0.25) 93%, rgba(255,100,100,0.15) 96%, rgba(255,100,100,0) 98%);
  padding: 0.05em 0.2em;
  margin: -0.05em -0.1em;
  border-radius: 7.5px;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}
</style>

<span class="highlight-marker">**Early detection of emerging arguments.**</span> By running the pipeline regularly on a rolling corpus, new claims can be detected as they first appear — before they gain traction. This could give advocates lead time to prepare responses or preemptive messaging, rather than reacting once a narrative is already established.

<span class="highlight-marker">**Surfacing unexpected actors.**</span> Because the pipeline identifies actors automatically from the text, it can flag voices that an analyst might not have thought to look for — new entrants in a debate, unusual alliances, or actors whose influence is growing but who aren't yet on anyone's radar.

<span class="highlight-marker">**Regular landscape assessments.**</span> Running the analysis on a quarterly or yearly cycle would produce a structured, comparable snapshot of the argument landscape and coalition structure over time. This could serve as an intelligence product for advocacy organisations or funders seeking to understand how a debate is evolving.

<span class="highlight-marker">**Guiding media outreach.**</span> The cross-outlet analysis (which arguments appear where, and with what stance) can inform where to pitch stories, which outlets are more receptive to certain framings, and where there are gaps in coverage that could be filled.

<span class="highlight-marker">**Measuring campaign impact.**</span> Mapping the argument landscape before and after an intervention — a campaign, a policy event, a controversy — could help detect whether anything shifted: new arguments emerging, actors moving, coalitions reconfiguring. This is more speculative and would require careful methodological work to distinguish signal from noise, but the structured nature of the data makes it a plausible direction.


<div class="text-box">
  <h3>Get in touch</h3>
I am interested in hearing from others working on similar problems or exploring how these tools could be applied in new contexts or further developed to be more useful. Whether you have ideas for improvements, questions about the approach, or want to collaborate on applications, I'd love to hear from you - <a href="mailto:hubert.thieriot@gmail.com" target="_blank" rel="noopener">reach out to me</a>.
</div>


<div style="background-color: rgba(255, 0, 0, 0.03); margin: 2em calc(-50vw + 50%) 0; padding: 0 calc(50vw - 50%) 2em; border-radius: 8px;" markdown="1">

<!-- # Method overview

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

## Actor-level stance heatmap

The coalition-level heatmap above averages positions within each group. The chart below shows the full picture at individual actor level: each row is an actor, each column is a claim, and the colour indicates whether that actor supports, opposes, or has no clear stance. Actors are grouped by coalition.

{% include arguments_actors/eu_alt_proteins/stance_heatmap.html %}

## Remaining improvements

- OpEds not yet managed i.e. the attribution of statements to the article author is not yet fully managed -->

</div>

---

[^outlets]: **UK** — The Guardian, The Telegraph, The Independent, The Economist · **Ireland** — Irish Independent · **Pan-European / International** — EUobserver, Euractiv, Deutsche Welle · **Germany** — Süddeutsche Zeitung, Der Spiegel, Die Welt, Frankfurter Allgemeine Zeitung · **France** — Le Monde, Le Figaro, Les Echos · **Italy** — Corriere della Sera, La Repubblica, Il Sole 24 Ore · **Spain** — El País · **Netherlands** — NRC Handelsblad, de Volkskrant · **Poland** — Gazeta Wyborcza, Rzeczpospolita
