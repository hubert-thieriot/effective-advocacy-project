---
layout: post
permalink: /posts/arguments_actors/
title: Arguments & Actors Mapping
subtitle: "Who Says What, Who Agrees, and How It Shifts"
description: Mapping key arguments, media trends, and actor coalitions in the EU alternative proteins debate
date: 2026-03-06 09:00:00 +0000
last_modified_at: 2026-03-08 09:00:00 +0000
author: Hubert Thieriot
tags: [discourse-analysis, actors-mapping]
---

<figure style="margin: 0 auto 2em; max-width: 600px;">
  <img src="{{ '/assets/images/actors-ap-eu-map.jpg' | relative_url }}" alt="" style="width: 100%; display: block;">
  <!-- <figcaption style="text-align: right; font-size: 0.75rem; color: #94a3b8; margin-top: 4px; font-style: italic;">Wassily Kandinsky — Color Study: Squares with Concentric Circles (1913)</figcaption> -->
</figure>

<div class="tldr">I prototyped a semi-automated method to extract the key arguments in a policy debate, identify which actors champion or contest them, and map how they cluster into coalitions. Applied here to the EU alternative proteins debate across 5,000 articles in seven languages, the approach is topic-agnostic and could potentially support advocacy organisations in various ways: detecting emerging arguments early, surfacing unexpected actors, producing regular landscape assessments, guiding media outreach, and measuring impact.
</div>


# From narratives to arguments and actors

In the [Narrative framing approach]({{ '/posts/narrative_framing/' | relative_url }}), we looked at *how* an issue is being discussed. Here, we go a step further and ask: **what specific positions are being taken, by whom, and who aligns with whom?** Rather than tracking broad themes, we extract concrete claims from the debate, identify which actors support or oppose them, and map the resulting coalitions.

To illustrate the approach, we apply it to a live policy debate: alternative proteins in Europe.


# Alternative proteins in European media

Alternative proteins — plant-based meat, cultivated meat, and fermentation-derived products — are a deeply contested food policy question in Europe. The debate sits at the intersection of EU climate policy, Common Agricultural Policy reform, food labelling disputes, public health concerns and industrial policy for the food tech sector. This makes it a relevant test case for arguments and actors mapping, one where the stakeholders are relatively diverse and the regulatory stakes are real.

In this analysis, the corpus covers more than 5,000 articles referring to alternative proteins, published between 2015 and 2025 across 23 major European media outlets[^outlets], in seven languages. The analysis identifies statements attributed to named actors, induces a set of recurring claims from those statements, scores each statement's relationship to each claim (supports, opposes, or neutral), and then maps actor relationships based on shared and opposing positions.


## The arguments: how is the debate structured?

In this approach, debates on a given topic are structured around *claims*: specific, debatable propositions that actors may support, oppose, or simply not address. Taken together, claims define an n-dimensional “stance space” that we can use to represent and compare actors.

Choosing the right set of claims is a critical design step — it defines both what we track and how we map actors & coalitions. Claims could be left to automation when the goal is to surface new and unexpected arguments, or manually defined as a narrow set to inform very targeted advocacy. Here, we adopted a mixed approach: claims were first automatically induced from actor statements in the corpus, then manually refined into two families — **product claims** (what alternative proteins and conventional agriculture *are* and *do*) and **policy claims** (what governments and regulators *should do*).

The chart below shows the claims identified and adopted in this analysis.

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
We then estimate to what extent **each actor supports or opposes these claims**, based on their collected statements. Finally, we define similarity between every pair of actors from their stance profiles and use community detection to produce a partition of actors into coalitions (see [Method overview](#method-overview) below). The scatter plot below visualizes this induced structure — each point is an actor, projected from the full 16-dimensional stance space down to two dimensions, with colours indicating the detected coalition.

<div class="chart-item">
<div class="chart-heading">
    <div class="chart-title">Who aligns with whom on alternative proteins in European media?</div>
    <div class="chart-subtitle">357 actors positioned by their stance profiles across 16 claims</div>
  </div>
{% include arguments_actors/eu_alt_proteins/interactive_pca_person.html %}
<p class="chart-note">
  <strong>Note:</strong> Each actor is represented as a vector of stances across 16 claims. Coalitions are detected via Louvain community detection on cosine similarity; the 2D projection uses PCA.
</p>
<p class="chart-note">
  <strong>Tip:</strong> Hover (desktop) or click/tap (mobile) points to see each actor’s individual positioning. For the whole list of actors and their respective stance positioning, click <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">here</a>.
</p>
<p class="chart-note">
  <strong>Disclaimer:</strong> These results are for demonstration purposes only. The analysis should not be relied upon to provide accurate estimates of actors positioning. Further validation and methodological refinement are needed before these results can be used for strategy purposes.
</p>
</div>

Three broad coalitions emerge. The heatmap below shows each coalition’s average position on each of the 16 claims.

<div class="heatmap-embed heatmap-embed--desktop">
  {% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap_person.html %}
</div>
<div class="heatmap-embed heatmap-embed--mobile">
  {% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap_mobile_person.html %}
</div>


To interpret what each coalition represents, I fed the stance centroids and actor lists to Claude Opus 4.6 and asked it draft their positioning profiles. Here is what the model produced:


- <span class="highlight-marker">**Coalition A (133 actors) — "Skeptics & food quality critics."** </span>Nutrition academics, food-culture advocates, and political opponents of alternative proteins — Marco Springmann, Van Tulleken, Monteiro on the UPF research side; Ron DeSantis, Jim Pillen, Giorgia Meloni on the political side; Richard Berman and Piers Morgan in media. United more by what they're against than a shared positive vision: alternative proteins are ultra-processed, health claims are overstated, and traditional food culture deserves protection. Internally diverse — UPF researchers and populist politicians arrive at similar positions from very different starting points.

- <span class="highlight-marker">**Coalition B (96 actors) — "Diet change & animal welfare advocates."** </span>High-profile alt-protein founders and advocates pushing for dietary transformation — Patrick Brown (Impossible Foods), Josh Tetrick (GOOD Meat), Bruce Friedrich (GFI), Ethan Brown (Beyond Meat), George Monbiot, Chris Bryant, Molly Scott Cato. Shares Coalition C's pro-technology, pro-market stance, but goes much further: by far the strongest support for EU-promoted meat reduction (0.84), and actively challenges traditional food culture and farming protections. This is the food-system transformation coalition.

- <span class="highlight-marker">**Coalition C (128 actors) — "Technology & market optimists."** </span>Startup founders, industry figures, and investors — Uma Valeti (Upside Foods), Mark Post, Didier Toubia (Aleph Farms), Seren Kell, Sandhya Sriram, Bill Gates. The strongest enthusiasm for food technology and market potential of any coalition, but notably neutral on meat reduction, farming livelihoods, and traditional food culture. This is the industry and investment voice: bullish on the technology and business opportunity without pushing a dietary change message.


Maybe the sharpest insight lies in the **B–C split**. Both coalitions are pro-alternative-proteins, but they seem to disagree on whether the goal is to *add new products to the market* or to *replace the existing food system*. For an advocate, this could mean that Coalition C actors are potential allies on technology and investment, but unlikely partners for campaigns framed around meat reduction or challenging farming interests. Or the other way around, it could help identify which Coalition C actors could be moved on meat-reduction messaging.

Notably, all of the 357 actors were identified automatically from the corpus — no names were pre-listed or manually selected. This means that the method can surface voices an analyst might not have thought of, and do so in virtually any language. To see each individual actor's positioning across all 16 claims, <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">open the full actor-level stance heatmap</a>.

The same analysis can be run at the organisation level rather than individual actors. Applied to the same corpus, it yields a broadly similar coalition structure.

<div class="chart-item">
<div class="chart-heading">
    <div class="chart-title">Organisation-level coalition map</div>
    <div class="chart-subtitle">Organisations positioned by their stance profiles across the same 16 claims</div>
  </div>
{% include arguments_actors/eu_alt_proteins/interactive_pca_org.html %}
</div>

The stance analysis by Claude produces a similar set of profiles: a three-way split between skeptics, market-focused players, and transformation advocates (though B & C are swapped):

- <span class="highlight-marker">**Coalition A (62 organisations) — "Skeptics & food quality watchdogs."** </span>Academic institutions, consumer bodies, farming lobbies, and food regulators — University of Oxford, Coldiretti, Slow Food, FSA, The Lancet, Greenpeace, WHO, Copa-Cogeca, Dairy UK. Leans into the ultra-processed critique (0.40), stricter UPF regulation (0.37), and transparent labeling (0.37). Skeptical of alt proteins' health claims (-0.24) and food technology's transformative potential (-0.27). Also includes some alt-protein-adjacent names (Vegan Society, the Vegetarian Butcher) that may sit here due to UPF or labeling positions rather than outright opposition.

- <span class="highlight-marker">**Coalition B (65 organisations) — "Industry & market optimists."** </span>Alt-protein startups, food multinationals, investors, and consultancies — Aleph Farms, Shiok Meats, Mosa Meat, Burger King, Tyson Foods, McKinsey, Barclays, Nestlé, Unilever. By far the strongest signal on market potential (0.95) and food technology (0.74). Broadly neutral on meat reduction (0.12), farming protection (0.09), and traditional food culture (0.14). This is the commercial coalition: enthusiastic about the opportunity, without strongly taking sides on the cultural or dietary politics.

- <span class="highlight-marker">**Coalition C (62 organisations) — "Systemic transformation advocates."** </span>Advocacy groups, think tanks, research bodies, and mission-driven companies — GFI, Eat Just, Impossible Foods, Beyond Meat, PETA, Humane Society, Chatham House, Green Alliance, RethinkX. Strong across the board: food technology (0.68), environment (0.68), meat reduction (0.66), market potential (0.52), health (0.39), and animal welfare (0.32). Actively rejects ultra-processed framing (-0.63), traditional food culture (-0.44), and farming protections (-0.42). The most ideologically coherent coalition — combines product enthusiasm with an explicit push for dietary and policy change.

## How arguments evolve over time

Beside mapping the coalition, we may want to track whether claims become more or less prominent over time. This is done in the chart below, showing the evolution of supporting/opposing statements per claim. Some observations:
- optimist claims about alternative proteins technology and market potential peaked around 2020
- statements supporting stricter regulation of UPF and labeling of AP continue to rise.

Note: before drawing any conclusion, I would consider normalising such trends (e.g. by number of captured articles) or weighing them (e.g. weighing media by their readership or importance) and see if the findings hold.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Support and opposition trends by claim</div>
    <div class="chart-subtitle">Yearly count of supporting and opposing statements extracted from European media coverage</div>
  </div>
  {% include arguments_actors/eu_alt_proteins/claims_mentions_over_time_facets.html %}
</div>

The analyses above are a starting point. Several deeper cuts on the same data could unlock more strategic value for advocacy organisations:

- **Bridging coalitions.** Identifying the divisive claims that keep coalitions apart and/or the actors who sit between coalitions. The first suggests which issues to downplay or reframe to build broader alliances; the second points to potential messengers who could carry an argument across coalition lines.
- **Argument diffusion.** Tracking how a specific claim spreads across outlets, countries, or actor types over time. Which actors or outlets pick up an argument first, and through what path does it propagate? 
- **Actor trajectories.** Rather than a single snapshot, tracking how individual actors move through stance space over time reveals who is shifting, who is entrenching, and in which direction. An actor drifting from Coalition C toward Coalition B on meat-reduction messaging, for instance, could signal an opening for engagement.



# Looking forward

The alternative proteins case above is one potential application, but the method itself is topic-agnostic and can be applied to any policy debate where actors take public positions. The corpus can also be extended to broader types of documents, including **TV and radio broadcasts, podcasts, parliamentary debates, policy consultations, political manifestos or social media**[^socialmedia], in virtually any language LLMs are sufficiently trained on.

Below are some of the ways this could support advocacy organisations or their funders:

<span class="highlight-marker">**Early detection of emerging arguments.**</span> By running the pipeline regularly on a rolling corpus, new claims can be detected as they first appear, before they gain traction. This could give advocates lead time to prepare responses or preemptive messaging, rather than reacting once a narrative is already established.

<span class="highlight-marker">**Surfacing unexpected actors.**</span> Because the pipeline identifies actors automatically from the text, it can flag voices that an analyst might not have thought to look for — new entrants in a debate, unusual alliances, or actors whose influence is growing but who aren't yet on anyone's radar. For an illustration, see the 357 actors <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">identified</a> in this demonstration.

<span class="highlight-marker">**Regular landscape assessments.**</span> Running the analysis on a quarterly or yearly cycle would produce a structured, comparable snapshot of the argument landscape and coalition structure over time. This could serve as an intelligence product for advocacy organisations or funders seeking to understand how a debate is evolving.

<span class="highlight-marker">**Guiding media outreach.**</span> The cross-outlet analysis (which arguments appear where, and with what stance) can inform where to pitch stories, which outlets are more receptive to certain framings, and where there are gaps in coverage that could be filled.

<span class="highlight-marker">**Measuring campaign impact.**</span> Mapping the argument landscape before and after an intervention — a campaign, a policy event, a controversy — could help detect whether anything shifted: new arguments emerging, actors moving, coalitions reconfiguring. This is more speculative and would require careful methodological work to distinguish signal from noise, but the structured nature of the data makes it a plausible direction.



<div class="text-box">
  <h3>Get in touch</h3>
I'm looking for advocacy organisations interested in piloting this approach on an existing intervention, and for funders who see value in building shared infrastructure for evidence-based advocacy strategy. If either describes you, I'd welcome a <a href="https://www.linkedin.com/in/hubertthieriot/" target="_blank" rel="noopener">conversation</a>.
</div>


# Disclaimer

This analysis is a prototype built to illustrate the method, not a finished intelligence product. The analysis should not be relied upon to provide accurate estimates of actors positioning: the results shown here have undergone limited manual sanity checks and have not been systematically validated against hand-coded datasets. Further validation and methodological refinement are needed before these results can be used for strategy purposes.


# Acknowledgements

Thanks to <a href="https://www.linkedin.com/in/moputera/" target="_blank" rel="noopener">Mo Putera</a> for his constructive feedback on this post.


<div style="background-color: rgba(255, 0, 0, 0.03); margin: 2em calc(-50vw + 50%) 0; padding: 0 calc(50vw - 50%) 2em; border-radius: 8px;" markdown="1">

# Method overview

### Collecting relevant chunks
Articles are first identified from MediaCloud using custom queries, then scraped, extracted, and split into chunks. The pipeline then induces a domain-specific framing schema from a sample of chunks, annotates a further sample with frame labels, and trains a multi-label transformer classifier to predict frame probabilities at chunk level. Chunks whose frame scores exceed a chosen threshold are treated as substantively relevant and passed to the downstream stages.

### Extracting actors, statements, claims and agreements
Named entities are then extracted from these high-scoring chunks using Stanza-based NER, and an entity-consolidation step merges different surface forms of the same actor into a single canonical entity. Where automatic consolidation is imperfect, manual merge rules can be added to correct cases in which the same actor appears under multiple names or aliases. For chunks containing identified actors, an LLM extracts attributable statements, a set of recurring claims is induced from a sample of those statements, and another LLM scores each statement against each claim as supporting, opposing, or neutral.

### Coalition mapping
To map actor alignments, these statement-claim scores are aggregated into actor-by-claim stance profiles. Actors are then compared using cosine similarity, producing a congruence network in which positive ties indicate similar positions and negative ties indicate opposing ones. After weak ties are thresholded, Louvain community detection is applied to the positive side of the network to identify clusters of actors who tend to align around the same set of claims. This approach is inspired by Discourse Network Analysis (DNA), developed by Philip Leifeld to study policy discourse. The specific representation and clustering choices used here are only one possible implementation: different similarity measures, network construction rules, or clustering methods could yield somewhat different coalition structures.

### Limitations

The approach has clear limitations. It requires topics with **enough media coverage** to produce a meaningful corpus — niche or emerging debates with only a handful of articles may not yield relevant results. By definition, it also lacks access to **private documents** and the associated arguments which in certain cases may be more relevant to the associated lobbying activity. The current implementation also does not yet handle **opinion pieces and editorials** where the author speaks in their own voice rather than quoting others.
</div>

---

[^outlets]: **UK** — The Guardian, The Telegraph, The Independent, The Economist · **Ireland** — Irish Independent · **Pan-European / International** — EUobserver, Euractiv, Deutsche Welle · **Germany** — Süddeutsche Zeitung, Der Spiegel, Die Welt, Frankfurter Allgemeine Zeitung · **France** — Le Monde, Le Figaro, Les Echos · **Italy** — Corriere della Sera, La Repubblica, Il Sole 24 Ore · **Spain** — El País · **Netherlands** — NRC Handelsblad, de Volkskrant · **Poland** — Gazeta Wyborcza, Rzeczpospolita

[^socialmedia]: Social media integration is still being assessed. API access costs and restrictions vary widely across platforms, and we may initially settle for a more targeted approach e.g. tracking a curated set of key accounts rather than broad keyword-based monitoring.






