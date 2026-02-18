"""Discourse Network Analysis (DNA) network building utilities.

Builds actor-actor networks based on shared/opposing claim positions.
Supports multi-dimensional clustering based on actor-claim position vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import networkx as nx

from efi_analyser.claims import ClaimsAnalysisResult
from efi_analyser.ner import ConsolidatedNERResult


@dataclass
class ActorNode:
    """Represents an actor in the DNA network."""

    entity_id: str
    canonical_name: str
    entity_type: str  # PERSON, ORG
    statement_count: int
    claims_supported: List[str] = field(default_factory=list)
    claims_opposed: List[str] = field(default_factory=list)


@dataclass
class ActorEdge:
    """Represents a relationship between two actors."""

    actor1_id: str
    actor2_id: str
    relationship: str  # "allies" or "enemies"
    weight: int  # Number of claims they agree/disagree on
    shared_claims: List[str] = field(default_factory=list)


@dataclass
class DNANetworkResult:
    """Complete DNA network analysis result."""

    nodes: List[ActorNode]
    edges: List[ActorEdge]
    communities: Dict[str, int]  # entity_id -> community_id

    # Metrics
    top_allies: List[Tuple[str, str, int]]  # (actor1_name, actor2_name, weight)
    top_enemies: List[Tuple[str, str, int]]
    centrality_scores: Dict[str, float]  # entity_id -> centrality


def _select_active_actors(
    claims_result: ClaimsAnalysisResult,
    min_statements: int,
    max_actors: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Select actors who take clear positions (supports/opposes) on claims.

    Only agreements with label "supports" or "opposes" count toward activity,
    ensuring that actors who only appear as neutral are excluded. Both
    build_dna_network and build_coalition_analysis use this helper so they
    operate on an identical actor set.

    Returns:
        (active_actors, actor_statement_counts) — actors sorted by activity (most active first)
    """
    stmt_lookup = {s.statement_id: s for s in claims_result.statements.statements}
    actor_statement_counts: Dict[str, int] = defaultdict(int)

    for agreement in claims_result.agreements.agreements:
        if agreement.label not in ("supports", "opposes"):
            continue
        stmt = stmt_lookup.get(agreement.statement_id)
        if not stmt or not stmt.speaker_entity_id:
            continue
        actor_statement_counts[stmt.speaker_entity_id] += 1

    active_actors = [
        eid for eid, count in actor_statement_counts.items()
        if count >= min_statements
    ]
    active_actors.sort(key=lambda eid: actor_statement_counts[eid], reverse=True)

    if len(active_actors) > max_actors:
        active_actors = active_actors[:max_actors]

    return active_actors, dict(actor_statement_counts)


def build_dna_network(
    claims_result: ClaimsAnalysisResult,
    consolidated_ner: Optional[ConsolidatedNERResult],
    min_statements: int = 2,
    max_actors: int = 50,
    active_actors: Optional[List[str]] = None,
) -> Optional[DNANetworkResult]:
    """Build actor-actor network from claims analysis.

    Args:
        claims_result: Claims analysis with statements and agreements
        consolidated_ner: Consolidated NER for entity resolution
        min_statements: Minimum supports/opposes agreements per actor (used when
            active_actors is not provided)
        max_actors: Maximum actors to include (used when active_actors is not provided)
        active_actors: Pre-computed actor list from _select_active_actors. When
            provided, min_statements and max_actors are ignored.

    Returns:
        DNANetworkResult or None if insufficient data
    """
    # Step 1: Build entity lookup from consolidated NER
    entity_lookup: Dict[str, object] = {}
    if consolidated_ner:
        for entity in consolidated_ner.entities:
            entity_lookup[entity.entity_id] = entity

    # Step 2: Build statement lookup
    stmt_lookup = {s.statement_id: s for s in claims_result.statements.statements}

    # Step 3: Build actor -> claim positions map
    # actor_positions[entity_id][claim_id] = "supports" | "opposes"
    actor_positions: Dict[str, Dict[str, str]] = defaultdict(dict)
    actor_statement_counts: Dict[str, int] = defaultdict(int)

    for agreement in claims_result.agreements.agreements:
        if agreement.label not in ("supports", "opposes"):
            continue

        stmt = stmt_lookup.get(agreement.statement_id)
        if not stmt or not stmt.speaker_entity_id:
            continue

        entity_id = stmt.speaker_entity_id
        claim_id = agreement.claim_id

        actor_statement_counts[entity_id] += 1

        # Take the first/strongest position if multiple statements
        if claim_id not in actor_positions[entity_id]:
            actor_positions[entity_id][claim_id] = agreement.label

    # Step 4: Use pre-computed actors or filter by min statements
    if active_actors is not None:
        # Use provided actor list (shared with coalition analysis for consistency)
        active_actors = [a for a in active_actors if a in actor_statement_counts]
    else:
        active_actors = [
            eid
            for eid, count in actor_statement_counts.items()
            if count >= min_statements
        ]
        active_actors.sort(key=lambda eid: actor_statement_counts[eid], reverse=True)
        if len(active_actors) > max_actors:
            active_actors = active_actors[:max_actors]

    if len(active_actors) < 2:
        return None

    # Step 5: Build network nodes
    nodes = []
    for entity_id in active_actors:
        entity = entity_lookup.get(entity_id)
        positions = actor_positions[entity_id]

        nodes.append(
            ActorNode(
                entity_id=entity_id,
                canonical_name=getattr(entity, "canonical_name", entity_id)
                if entity
                else entity_id,
                entity_type=getattr(entity, "entity_type", "UNKNOWN")
                if entity
                else "UNKNOWN",
                statement_count=actor_statement_counts[entity_id],
                claims_supported=[c for c, label in positions.items() if label == "supports"],
                claims_opposed=[c for c, label in positions.items() if label == "opposes"],
            )
        )

    # Step 6: Build edges (actor-actor relationships)
    edges = []

    for i, actor1_id in enumerate(active_actors):
        for actor2_id in active_actors[i + 1 :]:
            pos1 = actor_positions[actor1_id]
            pos2 = actor_positions[actor2_id]

            # Find shared claims
            shared_claims = set(pos1.keys()) & set(pos2.keys())
            if not shared_claims:
                continue

            ally_count = 0
            enemy_count = 0
            ally_claims = []
            enemy_claims = []

            for claim_id in shared_claims:
                if pos1[claim_id] == pos2[claim_id]:
                    ally_count += 1
                    ally_claims.append(claim_id)
                else:
                    enemy_count += 1
                    enemy_claims.append(claim_id)

            # Create edge(s) based on relationships
            if ally_count > 0:
                edges.append(
                    ActorEdge(
                        actor1_id=actor1_id,
                        actor2_id=actor2_id,
                        relationship="allies",
                        weight=ally_count,
                        shared_claims=ally_claims,
                    )
                )

            if enemy_count > 0:
                edges.append(
                    ActorEdge(
                        actor1_id=actor1_id,
                        actor2_id=actor2_id,
                        relationship="enemies",
                        weight=enemy_count,
                        shared_claims=enemy_claims,
                    )
                )

    # Step 7: Build networkx graph for layout and community detection
    G = nx.Graph()
    for node in nodes:
        G.add_node(
            node.entity_id,
            name=node.canonical_name,
            type=node.entity_type,
            size=node.statement_count,
        )

    # Add weighted edges (allies positive, enemies negative for community detection)
    for edge in edges:
        weight = edge.weight if edge.relationship == "allies" else -edge.weight
        if G.has_edge(edge.actor1_id, edge.actor2_id):
            G[edge.actor1_id][edge.actor2_id]["weight"] += weight
        else:
            G.add_edge(edge.actor1_id, edge.actor2_id, weight=weight)

    # Step 8: Community detection (using Louvain on absolute weights)
    G_abs = G.copy()
    for u, v, d in G_abs.edges(data=True):
        d["weight"] = abs(d.get("weight", 1))

    try:
        communities = nx.community.louvain_communities(G_abs, seed=42)
        community_map: Dict[str, int] = {}
        for idx, community in enumerate(communities):
            for node_id in community:
                community_map[node_id] = idx
    except Exception:
        community_map = {n.entity_id: 0 for n in nodes}

    # Step 9: Calculate centrality
    try:
        centrality = nx.degree_centrality(G)
    except Exception:
        centrality = {n.entity_id: 0.0 for n in nodes}

    # Step 10: Extract top pairs
    node_lookup = {n.entity_id: n for n in nodes}

    ally_edges = [e for e in edges if e.relationship == "allies"]
    enemy_edges = [e for e in edges if e.relationship == "enemies"]

    ally_edges.sort(key=lambda e: e.weight, reverse=True)
    enemy_edges.sort(key=lambda e: e.weight, reverse=True)

    top_allies = [
        (
            node_lookup[e.actor1_id].canonical_name,
            node_lookup[e.actor2_id].canonical_name,
            e.weight,
        )
        for e in ally_edges[:10]
    ]

    top_enemies = [
        (
            node_lookup[e.actor1_id].canonical_name,
            node_lookup[e.actor2_id].canonical_name,
            e.weight,
        )
        for e in enemy_edges[:10]
    ]

    return DNANetworkResult(
        nodes=nodes,
        edges=edges,
        communities=community_map,
        top_allies=top_allies,
        top_enemies=top_enemies,
        centrality_scores=centrality,
    )


def get_network_positions(
    result: DNANetworkResult,
    layout: str = "community",
) -> Dict[str, Tuple[float, float]]:
    """Calculate node positions for visualization.

    Args:
        result: DNA network result
        layout: Layout algorithm ("community", "spring", or "kamada_kawai")

    Returns:
        Dict mapping entity_id to (x, y) position
    """
    import math

    G = nx.Graph()
    for node in result.nodes:
        G.add_node(node.entity_id)

    # Use ally edges for layout (positive connections)
    for edge in result.edges:
        if edge.relationship == "allies":
            if G.has_edge(edge.actor1_id, edge.actor2_id):
                G[edge.actor1_id][edge.actor2_id]["weight"] += edge.weight
            else:
                G.add_edge(edge.actor1_id, edge.actor2_id, weight=edge.weight)

    if layout == "community" and result.communities:
        # Community-aware layout: cluster nodes by community
        pos = _community_layout(G, result.communities, result.nodes)
    elif layout == "kamada_kawai" and len(result.nodes) <= 30:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    return {
        node_id: (float(coords[0]), float(coords[1]))
        for node_id, coords in pos.items()
    }


def _community_layout(
    G: nx.Graph,
    communities: Dict[str, int],
    nodes: list,
) -> Dict[str, Tuple[float, float]]:
    """Create a layout that clusters nodes by community.

    Uses a two-level approach:
    1. Position community centers in a circle
    2. Run spring layout within each community
    3. Combine with offsets
    """
    import math

    # Group nodes by community
    community_nodes: Dict[int, list] = {}
    for node in nodes:
        comm_id = communities.get(node.entity_id, 0)
        if comm_id not in community_nodes:
            community_nodes[comm_id] = []
        community_nodes[comm_id].append(node.entity_id)

    n_communities = len(community_nodes)
    if n_communities == 0:
        return nx.spring_layout(G, seed=42)

    # Position community centers in a circle
    community_centers: Dict[int, Tuple[float, float]] = {}
    radius = 2.0 if n_communities > 1 else 0.0
    for i, comm_id in enumerate(sorted(community_nodes.keys())):
        angle = 2 * math.pi * i / n_communities
        community_centers[comm_id] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
        )

    # Calculate positions for each community's nodes
    final_pos: Dict[str, Tuple[float, float]] = {}

    for comm_id, node_ids in community_nodes.items():
        center_x, center_y = community_centers[comm_id]

        if len(node_ids) == 1:
            # Single node - place at center
            final_pos[node_ids[0]] = (center_x, center_y)
        else:
            # Create subgraph for this community
            subgraph = G.subgraph(node_ids).copy()

            # Run spring layout on subgraph
            if len(node_ids) <= 3:
                # For very small communities, arrange in a small circle
                for j, node_id in enumerate(node_ids):
                    angle = 2 * math.pi * j / len(node_ids)
                    r = 0.3
                    final_pos[node_id] = (
                        center_x + r * math.cos(angle),
                        center_y + r * math.sin(angle),
                    )
            else:
                # Use spring layout scaled to fit within community radius
                sub_pos = nx.spring_layout(subgraph, k=1.0, iterations=50, seed=42)

                # Scale and offset to community center
                # Scale factor based on community size
                scale = min(0.8, 0.3 + 0.05 * len(node_ids))

                for node_id, (x, y) in sub_pos.items():
                    final_pos[node_id] = (
                        center_x + x * scale,
                        center_y + y * scale,
                    )

    return final_pos


# =============================================================================
# Multi-Dimensional Coalition Clustering
# =============================================================================


@dataclass
class ClusteringResult:
    """Result of clustering actors by claim positions for a specific N."""

    n_clusters: int
    labels: Dict[str, int]  # entity_id -> cluster_label
    silhouette_score: float  # Quality metric (-1 to 1, higher is better)
    cluster_sizes: Dict[int, int]  # cluster_id -> size
    cluster_profiles: Dict[int, Dict[str, float]]  # cluster_id -> {claim_id: avg_position}


@dataclass
class CoalitionAnalysisResult:
    """Complete multi-dimensional coalition analysis."""

    # Actor-claim position matrix info
    actors: List[str]  # entity_id list
    actor_names: Dict[str, str]  # entity_id -> canonical_name
    claims: List[str]  # claim_id list
    claim_texts: Dict[str, str]  # claim_id -> claim text
    position_matrix: np.ndarray  # shape (n_actors, n_claims), values in {-1, 0, 1}

    # 2D projection for visualization
    positions_2d: Dict[str, Tuple[float, float]]  # entity_id -> (x, y)
    projection_method: str  # "pca" or "umap"

    # Multiple clustering results
    clustering_results: List[ClusteringResult]  # One for each N tried
    best_n: int  # N with best silhouette score
    best_clustering: ClusteringResult


def build_coalition_analysis(
    claims_result: ClaimsAnalysisResult,
    consolidated_ner: Optional[ConsolidatedNERResult],
    min_statements: int = 2,
    max_actors: int = 50,
    n_clusters_range: Tuple[int, int] = (2, 6),
    active_actors: Optional[List[str]] = None,
) -> Optional[CoalitionAnalysisResult]:
    """Build multi-dimensional coalition analysis from claims data.

    This represents each actor as a vector in claim-space, where each
    dimension is a claim and values are {-1: opposes, 0: neutral, 1: supports}.
    Clusters are found using k-means on this high-dimensional space.

    Args:
        claims_result: Claims analysis with statements and agreements
        consolidated_ner: Consolidated NER for entity resolution
        min_statements: Minimum supports/opposes agreements per actor (used when
            active_actors is not provided)
        max_actors: Maximum actors to include (used when active_actors is not provided)
        n_clusters_range: Range of N values to try (min, max inclusive)
        active_actors: Pre-computed actor list from _select_active_actors. When
            provided, min_statements and max_actors are ignored.

    Returns:
        CoalitionAnalysisResult or None if insufficient data
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    # Step 1: Build entity lookup
    entity_lookup: Dict[str, Any] = {}
    if consolidated_ner:
        for entity in consolidated_ner.entities:
            entity_lookup[entity.entity_id] = entity

    # Step 2: Build statement lookup
    stmt_lookup = {s.statement_id: s for s in claims_result.statements.statements}

    # Step 3: Build actor -> claim positions (numeric: supports=1, opposes=-1, neutral=0)
    actor_positions: Dict[str, Dict[str, int]] = defaultdict(dict)
    actor_statement_counts: Dict[str, int] = defaultdict(int)

    for agreement in claims_result.agreements.agreements:
        if agreement.label not in ("supports", "opposes"):
            continue  # Only count clear positions (matches _select_active_actors logic)

        stmt = stmt_lookup.get(agreement.statement_id)
        if not stmt or not stmt.speaker_entity_id:
            continue

        entity_id = stmt.speaker_entity_id
        claim_id = agreement.claim_id
        actor_statement_counts[entity_id] += 1

        # Convert label to numeric: supports=1, opposes=-1
        position = 1 if agreement.label == "supports" else -1

        # Take strongest position if multiple (support/oppose over neutral)
        current = actor_positions[entity_id].get(claim_id, 0)
        if abs(position) > abs(current):
            actor_positions[entity_id][claim_id] = position

    # Step 4: Use pre-computed actors or filter by min statements
    if active_actors is not None:
        # Use provided actor list (shared with build_dna_network for consistency)
        active_actors = [a for a in active_actors if a in actor_statement_counts]
    else:
        active_actors = [
            eid for eid, count in actor_statement_counts.items()
            if count >= min_statements
        ]
        active_actors.sort(key=lambda eid: actor_statement_counts[eid], reverse=True)
        if len(active_actors) > max_actors:
            active_actors = active_actors[:max_actors]

    if len(active_actors) < 3:  # Need at least 3 for clustering
        return None

    # Step 5: Get all claim IDs
    all_claims = [c.claim_id for c in claims_result.claims.claims]
    if not all_claims:
        return None

    # Step 6: Build position matrix
    n_actors = len(active_actors)
    n_claims = len(all_claims)
    position_matrix = np.zeros((n_actors, n_claims), dtype=np.float32)

    for i, actor_id in enumerate(active_actors):
        for j, claim_id in enumerate(all_claims):
            position_matrix[i, j] = actor_positions[actor_id].get(claim_id, 0)

    # Step 7: Build actor names lookup
    actor_names = {}
    for actor_id in active_actors:
        entity = entity_lookup.get(actor_id)
        if entity:
            actor_names[actor_id] = getattr(entity, "canonical_name", actor_id)
        else:
            actor_names[actor_id] = actor_id

    # Step 8: Build claim texts lookup
    claim_texts = {
        claim.claim_id: claim.text
        for claim in claims_result.claims.claims
    }

    # Step 9: 2D projection using PCA
    try:
        if n_claims >= 2:
            pca = PCA(n_components=2, random_state=42)
            positions_2d_array = pca.fit_transform(position_matrix)
        else:
            # Fallback for single claim
            positions_2d_array = np.column_stack([
                position_matrix[:, 0] if n_claims > 0 else np.zeros(n_actors),
                np.zeros(n_actors)
            ])
        projection_method = "pca"
    except Exception:
        positions_2d_array = np.random.randn(n_actors, 2)
        projection_method = "random"

    positions_2d = {
        active_actors[i]: (float(positions_2d_array[i, 0]), float(positions_2d_array[i, 1]))
        for i in range(n_actors)
    }

    # Step 10: Try multiple clustering with different N
    min_n, max_n = n_clusters_range
    max_n = min(max_n, n_actors - 1)  # Can't have more clusters than actors
    min_n = max(2, min_n)

    if max_n < min_n:
        max_n = min_n

    clustering_results = []
    best_score = -2  # Silhouette ranges from -1 to 1
    best_result = None

    for n in range(min_n, max_n + 1):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(position_matrix)

            # Calculate silhouette score (only if we have enough samples per cluster)
            unique_labels = set(labels)
            if len(unique_labels) > 1 and all(
                np.sum(labels == k) >= 1 for k in unique_labels
            ):
                score = silhouette_score(position_matrix, labels)
            else:
                score = -1.0

            # Build label map
            label_map = {
                active_actors[i]: int(labels[i])
                for i in range(n_actors)
            }

            # Calculate cluster sizes
            cluster_sizes = {}
            for label in range(n):
                cluster_sizes[label] = int(np.sum(labels == label))

            # Calculate cluster profiles (average position per claim)
            cluster_profiles: Dict[int, Dict[str, float]] = {}
            for label in range(n):
                mask = labels == label
                if np.sum(mask) > 0:
                    avg_positions = position_matrix[mask].mean(axis=0)
                    cluster_profiles[label] = {
                        all_claims[j]: float(avg_positions[j])
                        for j in range(n_claims)
                    }
                else:
                    cluster_profiles[label] = {c: 0.0 for c in all_claims}

            result = ClusteringResult(
                n_clusters=n,
                labels=label_map,
                silhouette_score=float(score),
                cluster_sizes=cluster_sizes,
                cluster_profiles=cluster_profiles,
            )
            clustering_results.append(result)

            if score > best_score:
                best_score = score
                best_result = result

        except Exception:
            continue

    if not clustering_results or best_result is None:
        return None

    return CoalitionAnalysisResult(
        actors=active_actors,
        actor_names=actor_names,
        claims=all_claims,
        claim_texts=claim_texts,
        position_matrix=position_matrix,
        positions_2d=positions_2d,
        projection_method=projection_method,
        clustering_results=clustering_results,
        best_n=best_result.n_clusters,
        best_clustering=best_result,
    )
