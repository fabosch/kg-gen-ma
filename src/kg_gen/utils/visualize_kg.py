"""High fidelity visualization utilities for kg-gen knowledge graphs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable
import colorsys
import webbrowser

from kg_gen.models import Graph


def _string_to_color(label: str) -> str:
    """Generate a deterministic pastel-like color for a given label."""
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    saturation = 0.55 + (int(digest[2:4], 16) / 255.0) * 0.3
    lightness = 0.45 + (int(digest[4:6], 16) / 255.0) * 0.25
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _sorted_ignore_case(items: Iterable[str]) -> list[str]:
    return sorted(items, key=lambda value: value.lower())


def _build_ontology_color_lookup(graph: Graph) -> dict[str, str]:
    """Build a color lookup for nodes based on their simple or composite ontology classifications.

    This function extracts both simple (`ontology_class_simple:`) and composite
    (`ontology_class_composite:`) classifications from entity metadata and assigns
    a consistent, *distinct* color to each ontology class. Colors are generated as
    evenly spaced hues to maximize visual separation between different classes.

    Behavior notes:
    - If a class exists both as a simple and composite class with the same literal
      name, the composite classification takes precedence for coloring the class node.

    Args:
        graph: Graph instance with entity_metadata containing ontology classifications.

    Returns:
        Dictionary mapping entity names to hex color codes based on their ontology class.
    """
    ontology_color_lookup: dict[str, str] = {}
    class_to_key: dict[str, str] = {}  # map class literal to prefixed key (e.g., 'simple::X' or 'composite::Y')

    if not graph.entity_metadata:
        return ontology_color_lookup

    # Collect all unique ontology class keys (prefixed to distinguish simple vs composite)
    unique_class_keys: list[str] = []
    for entity, metadata_set in graph.entity_metadata.items():
        if not metadata_set:
            continue
        for metadata in metadata_set:
            if metadata.startswith("ontology_class_simple:"):
                cls = metadata.replace("ontology_class_simple:", "", 1)
                key = f"simple::{cls}"
                if key not in class_to_key.values():
                    unique_class_keys.append(key)
                class_to_key[cls] = class_to_key.get(cls, key)  # preserve first mapping
            elif metadata.startswith("ontology_class_composite:"):
                cls = metadata.replace("ontology_class_composite:", "", 1)
                key = f"composite::{cls}"
                # ensure composite keys are preferred (put them in list if not present)
                if key not in unique_class_keys:
                    unique_class_keys.append(key)
                class_to_key[cls] = key  # composite overrides simple mapping

    # Deduplicate while preserving order
    unique_class_keys = list(dict.fromkeys(unique_class_keys))

    # Generate an evenly spaced palette of colors for the ontology classes
    colors: dict[str, str] = {}
    n = len(unique_class_keys)
    if n > 0:
        for idx, key in enumerate(unique_class_keys):
            # Evenly space hues around the color wheel; use fixed saturation/lightness
            hue = idx / max(1, n)
            saturation = 0.65
            lightness = 0.55
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors[key] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    # Map entities to their ontology class color (composite preferred over simple when both exist)
    for entity, metadata_set in graph.entity_metadata.items():
        if not metadata_set:
            continue
        assigned = False
        # Prefer composite classifications
        for metadata in metadata_set:
            if metadata.startswith("ontology_class_composite:"):
                cls = metadata.replace("ontology_class_composite:", "", 1)
                key = f"composite::{cls}"
                if key in colors:
                    ontology_color_lookup[entity] = colors[key]
                    assigned = True
                    break
        if assigned:
            continue
        # Fallback to simple classifications
        for metadata in metadata_set:
            if metadata.startswith("ontology_class_simple:"):
                cls = metadata.replace("ontology_class_simple:", "", 1)
                key = f"simple::{cls}"
                if key in colors:
                    ontology_color_lookup[entity] = colors[key]
                    assigned = True
                    break

    # Also color the ontology class nodes themselves (if present in the entity set)
    # If both composite and simple exist for a given literal class name, composite is preferred
    # to color the class node so that composite semantics are visible.
    for literal_class in set(class_to_key.keys()):
        pref_key = class_to_key.get(literal_class)
        if not pref_key:
            continue
        color = colors.get(pref_key)
        if color and literal_class in graph.entities:
            ontology_color_lookup[literal_class] = color

    return ontology_color_lookup


def _build_view_model(graph: Graph) -> dict[str, Any]:
    # Collect all entities from both the entities set and relations
    all_entities = set(graph.entities)
    for subject, _, obj in graph.relations:
        all_entities.add(subject)
        all_entities.add(obj)
    entities = _sorted_ignore_case(all_entities)

    relations = sorted(
        graph.relations,
        key=lambda triple: (triple[1].lower(), triple[0].lower(), triple[2].lower()),
    )

    entity_clusters = graph.entity_clusters or {}
    edge_clusters = graph.edge_clusters or {}

    entity_member_to_cluster: dict[str, str] = {}
    cluster_view: list[dict[str, Any]] = []

    for representative, members in entity_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"entity::{representative}")
        cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            entity_member_to_cluster[member] = representative

    # Build ontology-based color lookup if metadata is available
    ontology_color_lookup = _build_ontology_color_lookup(graph)

    node_color_lookup: dict[str, str] = {}
    
    # Apply entity cluster coloring first (if available)
    if cluster_view:
        for cluster in cluster_view:
            for member in cluster["members"]:
                node_color_lookup[member] = cluster["color"]
    
    # Then, apply ontology-based coloring (this takes precedence over clusters if both exist)
    # Ontology classification provides semantic meaning, so it should override generic clustering
    if ontology_color_lookup:
        node_color_lookup.update(ontology_color_lookup)
    
    # Finally, ensure all entities have a color (fill in any that don't)
    for entity in entities:
        if entity not in node_color_lookup:
            node_color_lookup[entity] = _string_to_color(f"entity::{entity}")

    edge_member_to_cluster: dict[str, str] = {}
    edge_color_lookup: dict[str, str] = {}
    edge_cluster_view: list[dict[str, Any]] = []

    for representative, members in edge_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"edge::{representative}")
        edge_cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            edge_member_to_cluster[member] = representative
            edge_color_lookup[member] = color

    degree = Counter()
    indegree = Counter()
    outdegree = Counter()
    predicate_counts = Counter()

    adjacency: dict[str, set[str]] = defaultdict(set)
    node_neighbors: dict[str, set[str]] = defaultdict(set)
    node_edges: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"incoming": [], "outgoing": []}
    )

    edges_view: list[dict[str, Any]] = []

    for index, (subject, predicate, obj) in enumerate(relations):
        predicate_counts[predicate] += 1
        degree[subject] += 1
        degree[obj] += 1
        outdegree[subject] += 1
        indegree[obj] += 1
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
        node_neighbors[subject].add(obj)
        node_neighbors[obj].add(subject)

        edge_id = f"e{index}"
        color = edge_color_lookup.get(predicate)
        if not color:
            color = _string_to_color(f"predicate::{predicate}")
            edge_color_lookup[predicate] = color

        edges_view.append(
            {
                "id": edge_id,
                "source": subject,
                "target": obj,
                "predicate": predicate,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": color,
                "tooltip": f"{subject} —{predicate}→ {obj}",
            }
        )

        node_edges[subject]["outgoing"].append(edge_id)
        node_edges[obj]["incoming"].append(edge_id)

    isolated_entities = [entity for entity in entities if degree[entity] == 0]

    def connected_components() -> list[dict[str, Any]]:
        visited: set[str] = set()
        components: list[dict[str, Any]] = []
        for node in entities:
            if node in visited:
                continue
            queue: deque[str] = deque([node])
            visited.add(node)
            members: list[str] = []
            while queue:
                current = queue.popleft()
                members.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(
                {
                    "size": len(members),
                    "members": _sorted_ignore_case(members),
                }
            )
        components.sort(key=lambda comp: (-comp["size"], comp["members"][0]))
        return components

    components = connected_components()

    nodes_view: list[dict[str, Any]] = []
    for entity in entities:
        cluster_id = entity_member_to_cluster.get(entity)
        radius = 18 + min(degree[entity], 8) * 2
        nodes_view.append(
            {
                "id": entity,
                "label": entity,
                "cluster": cluster_id,
                "color": node_color_lookup.get(entity, "#64748b"),
                "degree": degree[entity],
                "indegree": indegree[entity],
                "outdegree": outdegree[entity],
                "isRepresentative": cluster_id == entity if cluster_id else False,
                "radius": radius,
                "neighbors": _sorted_ignore_case(node_neighbors.get(entity, set())),
                "edgeIds": node_edges.get(entity, {"incoming": [], "outgoing": []}),
            }
        )

    top_entities = sorted(
        (
            {
                "label": node["label"],
                "degree": node["degree"],
                "indegree": node["indegree"],
                "outdegree": node["outdegree"],
                "cluster": node["cluster"],
            }
            for node in nodes_view
        ),
        key=lambda item: (-item["degree"], item["label"].lower()),
    )[:10]

    top_relations = sorted(
        (
            {
                "predicate": predicate,
                "count": count,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": edge_color_lookup.get(predicate, "#64748b"),
            }
            for predicate, count in predicate_counts.items()
        ),
        key=lambda item: (-item["count"], item["predicate"].lower()),
    )[:10]

    stats = {
        "entities": len(entities),
        "relations": len(edges_view),
        "relationTypes": len(predicate_counts),
        "entityClusters": len(cluster_view),
        "edgeClusters": len(edge_cluster_view),
        "isolatedEntities": len(isolated_entities),
        "components": len(components),
        "averageDegree": round(
            sum(degree[entity] for entity in entities) / len(entities), 2
        )
        if entities
        else 0,
        "density": round(len(edges_view) / (len(entities) * (len(entities) - 1)), 3)
        if len(entities) > 1
        else 0,
    }

    relation_records = [
        {
            "source": subject,
            "predicate": predicate,
            "target": obj,
            "edgeId": edge["id"],
            "color": edge["color"],
        }
        for edge, (subject, predicate, obj) in zip(edges_view, relations)
    ]

    return {
        "nodes": nodes_view,
        "edges": edges_view,
        "clusters": cluster_view,
        "edgeClusters": edge_cluster_view,
        "topEntities": top_entities,
        "topRelations": top_relations,
        "stats": stats,
        "isolatedEntities": isolated_entities,
        "components": components,
        "relations": relation_records,
    }


HTML_TEMPLATE = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")


def visualize(
    graph: Graph,
    output_path: str | None = None,
    *,
    open_in_browser: bool = False,
) -> Path:
    """Render an interactive dashboard for a graph.

    Args:
        graph: Graph instance to visualize.
        output_path: Optional path where the HTML document should be stored.
        open_in_browser: When True, open the generated file in the default browser.

    Returns:
        Path to the generated HTML file.
    """

    if not graph or not graph.entities:
        raise ValueError("Cannot visualize an empty graph")

    view_model = _build_view_model(graph)
    html = HTML_TEMPLATE.replace(
        "<!--DATA-->",
        json.dumps(view_model, ensure_ascii=False, indent=2),
    )

    # Make sidebar visible for standalone mode by removing display: none
    # display none must be set to prevent flicker when loading in main app
    html = html.replace(
        "display: none; /* Hidden by default - controlled by main app */",
        "display: block; /* Visible in standalone mode */",
    )

    destination = Path(output_path or "graph-visualization.html").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(destination.as_uri())

    return destination
