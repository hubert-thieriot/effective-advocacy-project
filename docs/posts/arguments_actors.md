---
layout: post
permalink: /posts/arguments_actors/
title: Arguments & Actors Mapping
subtitle: "Who Says What, Who Agrees, and How It Shifts"
description: Mapping key arguments, media trends, and actor coalitions in the EU alternative proteins debate
date: 2026-03-04 09:00:00 +0000
# last_modified_at: 2025-12-17 00:00:00 +0000
author: Hubert Thieriot
tags: [discourse-analysis, actors-mapping]
---


<div class="tldr">I prototyped a semi-automated method to extract the key arguments in a policy debate, identify which actors champion or contest them, and map how they cluster into coalitions. Applied here to the EU alternative proteins debate across 5,000 articles in seven languages, the approach is topic-agnostic and could potentially support advocacy organisations in various ways: detecting emerging arguments early, surfacing unexpected actors, producing regular landscape assessments, guiding media outreach, and measuring campaign impact.
</div>


# From narratives to arguments and actors

In the [Narrative framing approach]({{ '/posts/narrative_framing/' | relative_url }}), we looked at *how* an issue is discussed — which themes and angles dominate coverage. Here, we go a step further and ask: **what specific positions are being taken, by whom, and who aligns with whom?** Rather than tracking broad themes, we extract concrete claims from the debate, identify which actors support or oppose them, and map the resulting coalitions.

To illustrate the approach, we apply it to a live policy debate: alternative proteins in Europe.


# Alternative proteins in European media

Alternative proteins — plant-based meat, cultivated meat, and fermentation-derived products — have become one of the more contested food policy questions in Europe. The debate sits at the intersection of several ongoing conflicts: EU climate policy (the Farm to Fork strategy explicitly promoted protein diversification), Common Agricultural Policy reform, food labelling disputes, public health concerns about ultra-processing, and industrial policy for the food tech sector. This makes it a particularly relevant test case for arguments and actors mapping, one where the stakeholders are relatively vocal and the regulatory stakes are real.

In this analysis, the corpus covers more than 5,000 articles referring to alternative proteins, published between 2015 and 2025 (inclusive) across 23 major European media outlets[^outlets], in seven languages. The analysis identifies statements attributed to named actors, induces a set of recurring claims from those statements, scores each statement's relationship to each claim (supports, opposes, or neutral), and then maps actor relationships based on shared and opposing positions.


## The arguments: how is the debate structured?

In this approach, debates on a given topic are structured around *claims*: specific, debatable propositions that actors may support, oppose, or simply not address. Taken together, claims define an n-dimensional “stance space” that we can use to represent and compare actors.

Choosing the right set of claims is a critical design step — it defines both what we track and how we map actors & coalitions. Claims could be left to automation when the goal is to surface new and unexpected arguments, or manually defined as a narrow set to inform very targeted advocacy. Here, we adopted a mixed approach: claims were first automatically induced from actor statements in the corpus, then manually refined into two families — **product claims** (what alternative proteins and conventional agriculture *are* and *do*) and **policy claims** (what governments and regulators *should do*).

The chart below shows the claims used in this demo analysis.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Key claims identified in the European alternative proteins debate</div>
  </div>
  {% include arguments_actors/eu_alt_proteins/claims_cards.html %}
  <div class="chart-note">
    <strong>Note:</strong> Claims do not align neatly into "pro" and "anti" camps — even strong advocates of alternative proteins may disagree on specific product or policy claims, and vice versa. In what follows, "opposes" should not be read as "anti-alternative-proteins".
  </div>
</div>


## Coalitions: who aligns with whom?

To summarize how actors align across the 16 claims, we define similarity between every pair of actors from their stance profiles and then use community detection to produce a partition of actors into coalitions[^dna]. The scatter plot below visualizes this induced structure — each point is an actor, projected from the full 16-dimensional stance space down to two dimensions, with colours indicating the detected coalition.

{% include arguments_actors/eu_alt_proteins/interactive_pca.html %}
<p class="chart-note">
  <strong>Tip:</strong> Hover (desktop) or click/tap (mobile) points to see each actor’s individual positioning.
</p>

Three broad coalitions emerge. The heatmap below shows what defines each one — each coalition’s average position on each of the 16 claims.

<div class="heatmap-embed heatmap-embed--desktop">
  {% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap.html %}
</div>
<div class="heatmap-embed heatmap-embed--mobile">
  {% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap_mobile.html %}
</div>




The clustering is algorithmic, but interpreting what each coalition *means* still requires some form of judgement. In this case, I fed the stance centroids and actor lists to Claude Opus 4.6 to draft coalition profiles, to illustrate how much of this analysis could be produced at scale with modest human oversight. Here is what the model produced:

- <span class="highlight-marker">**Coalition A (82 actors) — "Skeptics & traditionalists."** </span>Farming lobbies, food-culture advocates, and political figures who have pushed back against alternative proteins — from Coldiretti and Slow Food to Ron DeSantis and Giorgia Meloni, alongside nutrition academics like Marco Springmann and Marion Nestle. What unites them is less a shared programme than a shared set of objections: alternative proteins are ultra-processed, their health claims are overstated, and traditional food culture deserves protection.

- <span class="highlight-marker">**Coalition B (83 actors) — "Innovation & market optimists."** </span>Startup founders, investors, and industry voices — Uma Valeti (Upside Foods), Didier Toubia (Aleph Farms), Leonardo DiCaprio. Strongly enthusiastic about food technology and the sector’s market potential, but notably neutral on meat reduction, farming livelihoods, and food culture. This is the industry pitch: bullish on the technology without picking fights over dietary change.

- <span class="highlight-marker">**Coalition C (77 actors) — "Systemic change advocates."** </span>Founders, advocates, and public intellectuals pushing for broader food-system transformation — Patrick Brown (Impossible Foods), Bruce Friedrich (GFI), George Monbiot, Bill Gates. Similar to Coalition B on most product claims, but sharply different on one front: they actively challenge traditional farming and meat culture, and push strongly for reduced meat consumption.

The sharpest insight here is the **B–C split**. Both coalitions are pro-alternative-proteins, but they seem to disagree on whether the goal is to *add new products to the market* or to *replace the existing food system*. For an advocate, this could mean that Coalition B actors are potential allies on technology and investment, but unlikely partners for campaigns framed around meat reduction or challenging farming interests. Or the other way around, it could help identify which Coalition B actors could be moved on meat-reduction messaging.

Notably, all of the 242 actors were identified automatically from the corpus — no names were pre-listed or manually selected. This means the method can surface voices an analyst might not have thought to look for, in virtually any language. To see each individual actor's positioning across all 16 claims, <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">open the full actor-level stance heatmap</a>.



## How arguments evolve over time

The coalition map above is a snapshot — it shows where actors stand, but not how the debate got there. The chart below tracks how often each claim is invoked over time, broken down by stance (support vs. opposition). This is another potential way for the analysis to be actionable: a claim that was barely mentioned two years ago but is now rapidly gaining traction is the kind of signal an advocate may want to catch early.

<div class="chart-item-lite">
  {% include arguments_actors/eu_alt_proteins/claims_mentions_over_time.html %}
</div>



One could go deeper and analyse how coalitions evolve over time: do they harden into stable blocs, fracture into subgroups, or blur as actors shift positions?



# Looking forward

The alternative proteins case above is a rapid demonstration, but the method itself is topic-agnostic — it can be applied to any policy debate where actors take public positions. The corpus can also be extended to broader types of documents, including **TV and radio broadcasts, podcasts, parliamentary debates, political manifestos or social media**[^socialmedia], in virtually any language LLMs are sufficiently trained on. 

Below are potential applications of the underlying methodology:


<span class="highlight-marker">**Early detection of emerging arguments.**</span> By running the pipeline regularly on a rolling corpus, new claims can be detected as they first appear, before they gain traction. This could give advocates lead time to prepare responses or preemptive messaging, rather than reacting once a narrative is already established.

<span class="highlight-marker">**Surfacing unexpected actors.**</span> Because the pipeline identifies actors automatically from the text, it can flag voices that an analyst might not have thought to look for — new entrants in a debate, unusual alliances, or actors whose influence is growing but who aren't yet on anyone's radar. For an illustration, see the 242 actors <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">identified</a> in this demonstration.

<span class="highlight-marker">**Regular landscape assessments.**</span> Running the analysis on a quarterly or yearly cycle would produce a structured, comparable snapshot of the argument landscape and coalition structure over time. This could serve as an intelligence product for advocacy organisations or funders seeking to understand how a debate is evolving.

<span class="highlight-marker">**Guiding media outreach.**</span> The cross-outlet analysis (which arguments appear where, and with what stance) can inform where to pitch stories, which outlets are more receptive to certain framings, and where there are gaps in coverage that could be filled.

<span class="highlight-marker">**Measuring campaign impact.**</span> Mapping the argument landscape before and after an intervention — a campaign, a policy event, a controversy — could help detect whether anything shifted: new arguments emerging, actors moving, coalitions reconfiguring. This is more speculative and would require careful methodological work to distinguish signal from noise, but the structured nature of the data makes it a plausible direction.


# About this demonstration

This analysis is a prototype built to illustrate the method, not a finished intelligence product. The results shown here have undergone limited manual sanity checks and have not been systematically validated against hand-coded datasets. Due to time constraints, the current corpus excludes opinion pieces and editorials, which are particularly relevant for this kind of study.

# Methodological limitations
The approach has clear limitations. It requires topics with **enough media coverage** to produce a meaningful corpus — niche or emerging debates with only a handful of articles may not yield relevant results. By definition, it also lacks access to **private documents** and the associated arguments which in certain cases may be more relevant to the associated lobbying activity.


<div class="text-box">
  <h3>Get in touch</h3>
I'm looking for advocacy organisations interested in piloting this approach on a live campaign, and for funders who see value in building shared infrastructure for evidence-based advocacy strategy. If either describes you, I'd welcome a <a href="https://www.linkedin.com/in/hubertthieriot/" target="_blank" rel="noopener">conversation</a>.
</div>



---

[^dna]: This approach is inspired by Discourse Network Analysis (DNA), a method developed by Philip Leifeld for studying policy discourse. The specific representation and clustering choices here — cosine similarity on stance vectors, PCA projection, Louvain community detection — are one of several possible implementations. Alternative approaches might use different similarity measures, network construction methods, or clustering algorithms, and could yield different coalition structures.

[^outlets]: **UK** — The Guardian, The Telegraph, The Independent, The Economist · **Ireland** — Irish Independent · **Pan-European / International** — EUobserver, Euractiv, Deutsche Welle · **Germany** — Süddeutsche Zeitung, Der Spiegel, Die Welt, Frankfurter Allgemeine Zeitung · **France** — Le Monde, Le Figaro, Les Echos · **Italy** — Corriere della Sera, La Repubblica, Il Sole 24 Ore · **Spain** — El País · **Netherlands** — NRC Handelsblad, de Volkskrant · **Poland** — Gazeta Wyborcza, Rzeczpospolita

[^socialmedia]: Social media integration is still being assessed. API access costs and restrictions vary widely across platforms, and we may initially settle for a more targeted approach e.g. tracking a curated set of key accounts rather than broad keyword-based monitoring.
