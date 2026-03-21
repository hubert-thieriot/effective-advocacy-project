---
layout: post
permalink: /posts/arguments_actors/
title: Arguments & Actors Mapping
subtitle: "Who Says What, Who Agrees, and How It Shifts"
description: Mapping key arguments, media trends, and actor coalitions in the EU alternative proteins debate
date: 2026-03-06 09:00:00 +0000
last_modified_at: 2026-03-20 09:00:00 +0000
author: Hubert Thieriot
tags: [discourse-analysis, actors-mapping]
---

<figure style="margin: 0 auto 2em; max-width: 600px;">
  <img src="{{ '/assets/images/actors-ap-eu-map.jpg' | relative_url }}" alt="" style="width: 100%; display: block;">
  <!-- <figcaption style="text-align: right; font-size: 0.75rem; color: #94a3b8; margin-top: 4px; font-style: italic;">Wassily Kandinsky — Color Study: Squares with Concentric Circles (1913)</figcaption> -->
</figure>

<div class="tldr">I prototyped a semi-automated method to extract the key arguments in a policy debate, identify which actors champion or contest them, and map how they cluster into coalitions. Applied here to the EU alternative proteins debate across 10 years, 21 media in seven languages, the approach is topic-agnostic and could potentially support advocacy organisations in various ways: detecting emerging arguments early, surfacing unexpected actors, producing regular landscape assessments, guiding media outreach, and measuring impact.
</div>


# From narratives to arguments and actors

In the [Narrative framing approach]({{ '/posts/narrative_framing/' | relative_url }}), we looked at *how* an issue is being discussed. Here, we go a step further and ask **what specific positions are being taken, by whom, and who aligns with whom?** Rather than tracking broad themes, we extract concrete claims from the debate, identify which actors support or oppose them, and map the resulting coalitions.

To illustrate the approach, I applied this approach to a live policy debate: alternative proteins in Europe.


# Alternative proteins in European media

Alternative proteins — plant-based meat, cultivated meat, and fermentation-derived products — are a deeply contested food policy question in Europe. The debate sits at the intersection of EU climate policy, Common Agricultural Policy reform, food labelling disputes, public health concerns and industrial policy. This makes it a relevant test case for arguments and actors mapping, one where the stakeholders are relatively diverse and the regulatory stakes are real.

To understand how the debate is structured and how it evolved, I analysed articles referring to alternative proteins published between 2015 and 2025 across 21 major European media outlets[^outlets] in seven languages. This represents a corpus of 3,500 documents. The analysis identifies statements, attributes them to named actors, induces a set of recurring claims from those statements, scores each statement's relationship to each claim (supports, opposes, or neutral), and then maps actor relationships based on their respective positions.


## The arguments: how is the debate structured?

In this approach, debates on a given topic are structured around *claims*: specific, debatable propositions that actors may support, oppose, or simply not address. Taken together, claims define an n-dimensional “stance space” that we can use to represent and compare actors.

Choosing the right set of claims is a critical design step: it defines both what we track and how we map actors & coalitions. Claims could be left to automation when the goal is to surface new and unexpected arguments, or manually defined as a narrow set to inform very targeted advocacy. Here, we adopted a hybrid approach: claims were first automatically induced from actor statements in the corpus, then manually refined into two families: **product claims** (what alternative proteins and conventional agriculture *are* and *do*) and **policy claims** (what governments and regulators *should do*).


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
We then estimate to what extent **each actor supports or opposes these claims**, based on their collected statements. This allows us to derive **coalitions of actors** that share similar positions on alternative proteins (see [Method overview](#method-overview) below). The scatter plot below visualizes these induced structure: each point is an actor with colours indicating detected coalitions.

<div class="chart-item">
<div class="chart-heading">
    <div class="chart-title">Who aligns with whom on alternative proteins in European media?</div>
    <div class="chart-subtitle">457 actors positioned by their stance profiles across 16 claims</div>
  </div>
{% include arguments_actors/eu_alt_proteins/interactive_pca_person.html %}
<p class="chart-note">
  <strong>Note:</strong> Each actor is represented as a vector of stances across 16 claims. Coalitions are detected via Louvain community detection on cosine similarity; the 2D projection uses PCA.
</p>
<p class="chart-note">
  <strong>Tip:</strong> Hover (desktop) or click/tap (mobile) points to see each actor’s individual positioning. For the whole list of actors and their respective stance positioning, click <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">here</a>.
</p>
<!-- <p class="chart-note">
  <strong>Disclaimer:</strong> These results are for demonstration purposes only. The analysis should not be relied upon to provide accurate estimates of actors positioning. Further validation and methodological refinement are needed before these results can be used for strategy purposes.
</p> -->
</div>

Three broad coalitions emerge. The heatmap below shows each coalition’s average position on each of the 16 claims.

<div class="heatmap-embed heatmap-embed--desktop">
  {% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap_person.html %}
</div>
<div class="heatmap-embed heatmap-embed--mobile">
  {% include arguments_actors/eu_alt_proteins/coalition_stance_heatmap_mobile_person.html %}
</div>


To interpret what each coalition represents, I fed the stance centroids and actor lists to Claude Opus 4.6 and asked it draft their positioning profiles. Here is what the model produced:


- <span class="highlight-marker">**Coalition A (160 actors) — "Skeptics & food quality critics."** </span>Nutrition researchers, food-culture defenders, and political opponents — Chris Van Tulleken, Carlos Monteiro, Marco Springmann on the UPF research side; Ron DeSantis, Giorgia Meloni, Viktor Orbán, Jim Pillen in politics; Piers Morgan, Joanna Blythman, Marion Nestle in food media and commentary. Strongest on the ultra-processed critique (0.36), stricter UPF regulation (0.36), transparent labeling (0.32), and traditional food culture (0.34). Skeptical of health claims (-0.28), food technology (-0.30), and market potential (-0.21). Mildly protective of farming livelihoods (0.09) but the identity is more about food quality and naturalness than economic protectionism.

- <span class="highlight-marker">**Coalition B (133 actors) — "Industry & market optimists."** </span>Startup founders, investors, food-industry executives, and some scientists — Mark Post, Sandhya Sriram, Decker Walker, Benjamina Bollag, Tomer Halevy, Justin Kolbeck, Miyoko Schinner, Emmanuel Macron, Boris Johnson. By far the strongest signal on market potential (0.91) and food technology (0.67). Mildly positive on traditional food culture (0.15) and essentially neutral on meat reduction (0.10), farming protection (0.03), and the ultra-processed debate (-0.14). The commercial and pragmatic voice: enthusiastic about the opportunity without taking strong positions on the cultural or dietary politics.

- <span class="highlight-marker">**Coalition C (164 actors) — "Systemic transformation advocates."** </span>Mission-driven founders, advocacy figures, public intellectuals, and policy voices — Patrick Brown, Bruce Friedrich, Josh Tetrick, Uma Valeti, Seren Kell, Ethan Brown, Bill Gates, George Monbiot, Paul Shapiro, Chris Bryant, Molly Scott Cato. Strong across product and policy claims: food technology (0.59), meat reduction (0.55), environment (0.47), market potential (0.43), health (0.32), animal welfare (0.23). Firmly rejects the ultra-processed framing (-0.67), traditional food culture (-0.32), and farming protections (-0.18). The most ideologically coherent coalition — pairs product enthusiasm with an explicit push for dietary and systemic change.

To browse coalition actors by coalition and explore their individual positions, you can use the interactive browser below.

<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Browse coalition actors</div>
    <div class="chart-subtitle">Search and explore actors by coalition; hover for stance details</div>
  </div>
  {% include arguments_actors/eu_alt_proteins/coalition_actor_browser_person.html %}
</div>

The sharpest insight lies in the **B–C split**. Both coalitions are pro-alternative-proteins, but they seem to (at least in appearance) disagree on whether the goal is to *add new products to the market* or to *replace the existing food system*. For an advocate, this could mean that Coalition B actors are potential allies on technology and investment, but unlikely partners for campaigns framed around meat reduction or challenging farming interests. Or the other way around, it could help identify which Coalition C actors could be moved on meat-reduction messaging.

Notably, all of **the 457 actors were identified automatically from the corpus** — no names were pre-listed or manually selected. This means that the method can surface voices an analyst might not have thought of, and do so in virtually any language. To see each individual actor's positioning across all 16 claims, <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">open the full actor-level stance heatmap</a>.

The same analysis can also be run at the organisation level rather than individual actors — applied to the same corpus, it yields a broadly similar coalition structure (see [Appendix: Organisation-level coalitions](#appendix-organisation-level-coalitions) below).

## How arguments evolve over time

Beside mapping the coalition, we may want to track whether claims become more or less prominent over time. This is done in the chart below, showing the evolution of supporting/opposing statements per claim.


<div class="chart-item">
  <div class="chart-heading">
    <div class="chart-title">Support and opposition trends by claim</div>
    <div class="chart-subtitle">Yearly count of supporting and opposing statements extracted from European media coverage</div>
  </div>
  {% include arguments_actors/eu_alt_proteins/claims_mentions_over_time_facets.html %}
</div>

The charts show a debate that has shifted over the past decade. Early coverage was dominated by product-level claims: the environmental case, food technology, market potential. These peaked around 2020 — roughly the alt-protein <a href="https://www.greenqueen.com.hk/alternative-protein-funding-q4-2025-plant-based-investment" target="_blank">investment boom</a> — and have since declined.

Since then, a different set of arguments has been gaining ground: the ultra-processed critique, the dispute around labeling and the protection of traditional food culture. This is essentially Coalition A's vocabulary.

One may want to normalise these trends (e.g. by means of a separate topic acting as a "control-group") or weighing them (e.g. by the media readership or importance) and see if the findings hold.

# Looking forward

The alternative proteins case above is one potential application, but the method itself is topic-agnostic and can be applied to any policy debate where actors take public positions. The corpus can also be extended to broader types of documents, including **TV and radio broadcasts, podcasts, parliamentary debates, policy consultations, political manifestos or social media**[^socialmedia], in virtually any language LLMs are sufficiently trained on.

Below are some of the ways this could support advocacy organisations or their funders:

<span class="highlight-marker">**Early detection of emerging arguments:**</span> By running the pipeline regularly on a rolling corpus, new claims can be detected as they first appear, before they gain traction. This could give advocates lead time to prepare responses or preemptive messaging, rather than reacting once a narrative is already established.

<span class="highlight-marker">**Surfacing unexpected actors:**</span> Because the pipeline identifies actors automatically from the text, it can flag voices that an analyst might not have thought to look for — new entrants in a debate, unusual alliances, or actors whose influence is growing but who aren't yet on anyone's radar. For an illustration, see the 457 actors <a href="{{ '/posts/arguments_actors/actor_stances/' | relative_url }}" target="_blank">identified</a> in this demonstration.

<span class="highlight-marker">**Regular landscape assessments:**</span> Running the analysis on a quarterly or yearly cycle would produce a structured, comparable snapshot of the argument landscape and coalition structure over time. This could serve as an intelligence product for advocacy organisations or funders seeking to understand how a debate is evolving.

<span class="highlight-marker">**Guiding media outreach:**</span> The cross-outlet analysis (which arguments appear where, and with what stance) can inform where to pitch stories, which outlets are more receptive to certain framings, and where there are gaps in coverage that could be filled.

<span class="highlight-marker">**Measuring campaign impact:**</span> Mapping the argument landscape before and after an intervention — a campaign, a policy event, a controversy — could help detect whether anything shifted: new arguments emerging, actors moving, coalitions reconfiguring. This is more speculative and would require careful methodological work to distinguish signal from noise, but the structured nature of the data makes it a plausible direction.

<span class="highlight-marker">**Testing positioning:**</span> An advocate could feed a message or position and be indicated which actors and coalitions would likely support or oppose it, and how to adjust the framing to appeal to a broader coalition. This turns the landscape map into a design tool.

<span class="highlight-marker">**Following actors trajectories:**</span> Rather than a single snapshot, tracking how individual actors move through stance space over time reveals who is shifting, who is entrenching, and in which direction.


<span class="highlight-marker">**Assessing actors' influence:**</span> The current analysis maps who says what, but treats all actors equally. A natural extension is to estimate how important actors are in shaping the debate. This could be done by analysing actors' reach, their network position, and tracking how their arguments are picked up by others over time.

<!-- 
- **Argument diffusion.** Tracking how a specific claim spreads across outlets, countries, or actor types over time. Which actors or outlets pick up an argument first, and through what path does it propagate? -->


<div class="text-box">
  <h3>Get in touch</h3>
I'm looking for advocacy organisations interested in piloting this approach on an existing intervention, and for funders who see value in building shared infrastructure for evidence-based advocacy strategy. If either describes you, I'd welcome a <a href="https://www.linkedin.com/in/hubertthieriot/" target="_blank" rel="noopener">conversation</a>.
</div>


# Disclaimer

This analysis is a prototype built to illustrate the method, not a finished intelligence product. The analysis should not be relied upon to provide accurate estimates of actors positioning: the results shown here have undergone limited manual sanity checks and have not been systematically validated against hand-coded datasets. Further validation and methodological refinement are needed before these results can be used for strategy purposes.


# Acknowledgements

I am very grateful to <a href="https://www.linkedin.com/in/moputera/" target="_blank" rel="noopener">Mo Putera</a> and <a href="https://www.linkedin.com/in/evolving-richie" target="_blank" rel="noopener">Thomas Manandhar-Richardson</a> for their insightful feedback.


<div style="background-color: rgba(255, 0, 0, 0.03); margin: 2em calc(-50vw + 50%) 0; padding: 0 calc(50vw - 50%) 2em; border-radius: 8px;" markdown="1">

# Method overview

### Collecting relevant segments
Articles are first identified from MediaCloud using custom queries, further filtered, then scraped, extracted, and split into chunks. The pipeline then induces a domain-specific framing schema from a sample of chunks, annotates a further sample with frame labels, and trains a multi-label transformer classifier to predict frame probabilities at chunk level. Chunks whose frame scores exceed a chosen threshold are treated as substantively relevant and passed to the downstream stages.

### Extracting actors, statements, claims and agreements
Named entities are then extracted from these high-scoring chunks using Stanza-based NER, and an entity-consolidation step merges different surface forms of the same actor into a single canonical entity. Where automatic consolidation is imperfect, manual merge rules can be added to correct cases in which the same actor appears under multiple names or aliases. For chunks containing identified actors, an LLM extracts attributable statements, a set of recurring claims is induced from a sample of those statements, and another LLM scores each statement against each claim as supporting, opposing, or neutral.

### Coalition mapping
To map actor alignments, these statement-claim scores are aggregated into actor-by-claim stance profiles. Actors are then compared using cosine similarity, producing a congruence network in which positive ties indicate similar positions and negative ties indicate opposing ones. After weak ties are thresholded, Louvain community detection is applied to the positive side of the network to identify clusters of actors who tend to align around the same set of claims. This approach is inspired by Discourse Network Analysis (DNA), developed by Philip Leifeld to study policy discourse. The specific representation and clustering choices used here are only one possible implementation: different similarity measures, network construction rules, or clustering methods could yield somewhat different coalition structures.

### Limitations

The approach has clear limitations. It requires topics with **enough media coverage** to produce a meaningful corpus — niche or emerging debates with only a handful of articles may not yield relevant results. By definition, it also lacks access to **private documents** and the associated arguments which in certain cases may be more relevant to the associated lobbying activity. The current implementation also does not yet handle **opinion pieces and editorials** where the author speaks in their own voice rather than quoting others.

# Appendix: Organisation-level coalitions

The same coalition analysis can be run at the organisation level. Applied to the same corpus and claims, it yields a broadly similar three-way structure.

<div class="chart-item">
<div class="chart-heading">
    <div class="chart-title">Organisation-level coalition map</div>
    <div class="chart-subtitle">Organisations positioned by their stance profiles across the same 16 claims</div>
  </div>
{% include arguments_actors/eu_alt_proteins/interactive_pca_org.html %}
</div>

The stance analysis by Claude produces a similar set of profiles: a three-way split between skeptics, market-focused players, and transformation advocates:

- <span class="highlight-marker">**Coalition A (57 organisations) — "Skeptics & food quality watchdogs."** </span>Farming lobbies, consumer bodies, regulators, public health institutions, and some academic centres — Coldiretti, Slow Food, Copa-Cogeca, Dairy UK, WHO, The Lancet, FSA, Unicef, European Parliament, European Court of Justice. The strongest scores on stricter UPF regulation (0.46), transparent labeling (0.42), traditional food culture (0.37), ultra-processed critique (0.35), and protecting farming livelihoods (0.26). Skeptical of health claims (-0.23) and food technology (-0.18). A genuinely diverse bloc — it brings together farming protectionists (Coldiretti, Copa-Cogeca), public health bodies (WHO, Unicef), and food quality institutions (Slow Food, The Food Foundation) whose objections to alt proteins come from quite different places.

- <span class="highlight-marker">**Coalition B (73 organisations) — "Industry & market optimists."** </span>Alt-protein startups, food multinationals, fast-food chains, investors, and consultancies — Shiok Meats, Burger King, KFC, McDonald's, McKinsey, Barclays, Upside Foods, Mosa Meat, Vow, Nestlé (via European Commission, Unilever sits in C), Lidl, Asda, Walmart. By far the strongest on market potential (0.99) and food technology (0.64). Essentially neutral on meat reduction (0.11), farming protection (-0.01), and traditional food culture (0.04). The commercial coalition: enthusiastic about the business opportunity, cautious about taking sides on dietary politics.

- <span class="highlight-marker">**Coalition C (54 organisations) — "Systemic transformation advocates."** </span>Mission-driven companies, advocacy organisations, think tanks, and research bodies — Impossible Foods, Eat Just, GFI, Beyond Meat, PETA, Chatham House, RethinkX, Social Market Foundation, University of Oxford, IPCC, United Nations. Strong across the board: environment (0.72), meat reduction (0.61), food technology (0.54), health (0.39), animal welfare (0.32). Firmly rejects ultra-processed framing (-0.48), traditional food culture (-0.35), and farming protections (-0.35). The smallest but most ideologically coherent coalition — product and policy advocacy are tightly bundled.

</div>

---

[^outlets]: **UK** — The Guardian, The Telegraph, The Independent, The Economist · **Ireland** — Irish Independent · **Pan-European / International** — EUobserver, Euractiv, Deutsche Welle · **Germany** — Süddeutsche Zeitung, Der Spiegel, Die Welt, Frankfurter Allgemeine Zeitung · **France** — Le Monde, Le Figaro, Les Echos · **Italy** — Corriere della Sera, La Repubblica, Il Sole 24 Ore · **Spain** — El País · **Netherlands** — NRC Handelsblad, de Volkskrant · **Poland** — Gazeta Wyborcza, Rzeczpospolita

[^socialmedia]: Social media integration is still being assessed. API access costs and restrictions vary widely across platforms, and we may initially settle for a more targeted approach e.g. tracking a curated set of key accounts rather than broad keyword-based monitoring.






