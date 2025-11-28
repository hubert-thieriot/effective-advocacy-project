"""Frame examples table plotter for narrative framing results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import BasePlotter, PlotConfig
from .registry import register_plotter
from ._utils import load_corpus_index


def _assignments_to_dicts(assignments) -> List[dict]:
    """Convert FrameAssignments to list of dicts."""
    items = getattr(assignments, "assignments", None) or []
    return [
        {
            "passage_id": getattr(a, "passage_id", ""),
            "passage_text": getattr(a, "passage_text", ""),
            "probabilities": getattr(a, "probabilities", {}),
        }
        for a in items
    ]


def _schema_to_dict(schema) -> dict:
    """Convert FrameSchema to dict with frames list."""
    if schema is None:
        return {"frames": []}
    frames = getattr(schema, "frames", [])
    return {
        "frames": [
            {
                "frame_id": str(getattr(f, "id", i)),
                "name": getattr(f, "name", ""),
                "short_name": getattr(f, "short_name", ""),
            }
            for i, f in enumerate(frames)
        ]
    }


def get_top_examples_per_frame(
    assignments: List[dict],
    corpus_index: Dict[str, dict],
    frame_schema: Optional[Dict[str, dict]] = None,
    top_k: int = 1,
    max_text_length: int = 500,
) -> Dict[str, List[dict]]:
    """Find the highest-weighted chunk for each frame."""
    frame_schema = frame_schema or {}
    
    frame_id_to_schema = {}
    frame_name_to_schema = {}
    
    if isinstance(frame_schema, dict):
        if "frames" in frame_schema:
            for f in frame_schema.get("frames", []):
                frame_id = str(f.get("frame_id", ""))
                frame_name = f.get("name", "")
                if frame_id:
                    frame_id_to_schema[frame_id] = f
                if frame_name:
                    frame_name_to_schema[frame_name] = f
        else:
            frame_name_to_schema = frame_schema
    
    frame_scores: Dict[str, List[tuple]] = {}
    
    # Debug: show what frame IDs we're looking for
    if frame_id_to_schema:
        print(f"📊 Schema frame IDs: {list(frame_id_to_schema.keys())[:10]}...")
    if frame_name_to_schema:
        print(f"📊 Schema frame names: {list(frame_name_to_schema.keys())[:10]}...")
    
    for a in assignments:
        probs = a.get("probabilities", {})
        if not probs:
            continue
        passage_text = a.get("passage_text", "")
        if not passage_text:
            continue
        passage_id = a.get("passage_id", "")
        # Handle different passage_id formats: "corpus::doc_id:chunk_idx" or just "doc_id"
        if "::" in passage_id:
            # Global format: "corpus::local_id:chunk_idx"
            parts = passage_id.split("::", 1)
            if len(parts) > 1 and ":" in parts[1]:
                doc_id = f"{parts[0]}::{parts[1].split(':')[0]}"
            else:
                doc_id = passage_id.split(":")[0] if ":" in passage_id else passage_id
        else:
            doc_id = passage_id.split(":")[0] if ":" in passage_id else passage_id
        meta = corpus_index.get(doc_id, {})
        
        for frame_key, prob in probs.items():
            if prob > 0:
                schema = frame_id_to_schema.get(frame_key) or frame_name_to_schema.get(frame_key) or {}
                frame_title = schema.get("short_name") or schema.get("name") or frame_key
                # Use frame_key as the key if we can't find a name in schema
                frame_name = schema.get("name") or frame_key
                
                if frame_name not in frame_scores:
                    frame_scores[frame_name] = []
                
                frame_scores[frame_name].append((prob, {
                    "text": passage_text,
                    "score": prob,
                    "passage_id": passage_id,
                    "country": meta.get("country_name") or meta.get("country") or "Unknown",
                    "party": meta.get("party_name") or meta.get("party") or "Unknown",
                    "language": meta.get("language") or "Unknown",
                    "frame_title": frame_title,
                }))
    
    # Debug: show what frames we found
    if frame_scores:
        print(f"📊 Found scores for {len(frame_scores)} frames: {list(frame_scores.keys())[:10]}...")
    else:
        print(f"⚠️ No frame scores found. Checked {len(assignments)} assignments with probabilities.")
    
    result = {}
    for frame, scores in frame_scores.items():
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        top = [s[1] for s in sorted_scores[:top_k]]
        
        for item in top:
            if len(item["text"]) > max_text_length:
                item["text"] = item["text"][:max_text_length].rsplit(" ", 1)[0] + "..."
        
        result[frame] = top
    
    return result


def format_as_html(
    examples: Dict[str, List[dict]],
    frame_schema: Dict[str, dict],
) -> str:
    """Format examples as an HTML table."""
    rows = []
    
    for frame in sorted(examples.keys()):
        items = examples[frame]
        if not items:
            continue
        
        item = items[0]
        schema = frame_schema.get(frame, {})
        display_name = schema.get("name", frame).replace("_", " ").title()
        frame_title = item.get("frame_title") or schema.get("short_name") or display_name
        text = html.escape(item["text"])
        
        rows.append(f"""
        <tr>
            <td><strong>{display_name}</strong></td>
            <td>{html.escape(frame_title)}</td>
            <td>{item['country']}</td>
            <td>{item['party']}</td>
            <td>{item['language']}</td>
            <td class="example-text">{text}</td>
            <td>{item['score']:.2%}</td>
        </tr>
        """)
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Top Frame Examples</title>
    <style>
        body {{
            font-family: 'Open Sans', Arial, sans-serif;
            margin: 2em;
            background: #fafafa;
        }}
        h1 {{
            color: #333;
            font-weight: 600;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .example-text {{
            max-width: 500px;
            font-size: 0.9em;
            line-height: 1.5;
            color: #333;
        }}
        td:last-child {{
            text-align: right;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>Top Examples by Frame</h1>
    <table>
        <thead>
            <tr>
                <th>Frame</th>
                <th>Frame Title</th>
                <th>Country</th>
                <th>Party</th>
                <th>Language</th>
                <th>Example Text</th>
                <th>Score</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>
"""


def format_as_markdown(
    examples: Dict[str, List[dict]],
    frame_schema: Dict[str, dict],
) -> str:
    """Format examples as a Markdown table."""
    lines = [
        "| Frame | Frame Title | Country | Party | Language | Example Text | Score |",
        "|-------|-------------|---------|-------|----------|--------------|-------|",
    ]
    
    for frame in sorted(examples.keys()):
        items = examples[frame]
        if not items:
            continue
        
        item = items[0]
        schema = frame_schema.get(frame, {})
        display_name = schema.get("name", frame).replace("_", " ").title()
        frame_title = item.get("frame_title") or schema.get("short_name") or display_name
        text = item["text"].replace("|", "\\|").replace("\n", " ")
        if len(text) > 200:
            text = text[:200].rsplit(" ", 1)[0] + "..."
        
        lines.append(
            f"| **{display_name}** | {frame_title} | {item['country']} | {item['party']} | "
            f"{item['language']} | {text} | {item['score']:.0%} |"
        )
    
    return "\n".join(lines)


def format_as_text(
    examples: Dict[str, List[dict]],
    frame_schema: Dict[str, dict],
) -> str:
    """Format examples as plain text."""
    lines = []
    
    for frame in sorted(examples.keys()):
        items = examples[frame]
        if not items:
            continue
        
        item = items[0]
        schema = frame_schema.get(frame, {})
        display_name = schema.get("name", frame).replace("_", " ").title()
        frame_title = item.get("frame_title") or schema.get("short_name") or display_name
        
        lines.append(f"\n{'='*60}")
        lines.append(f"FRAME: {display_name}")
        lines.append(f"Frame Title: {frame_title}")
        lines.append(f"Score: {item['score']:.2%}")
        lines.append(f"Country: {item['country']} | Party: {item['party']} | Language: {item['language']}")
        lines.append("-" * 60)
        lines.append(item["text"])
    
    return "\n".join(lines)


@register_plotter
class FrameExamplesPlotter(BasePlotter):
    """Generate table of top example chunks per frame."""
    
    name = "frame_examples"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        if self.state.assignments:
            assignments = _assignments_to_dicts(self.state.assignments)
        else:
            assignments_path = self.results_dir / "frame_assignments.json"
            if assignments_path.exists():
                assignments = json.loads(assignments_path.read_text())
            else:
                print(f"⚠️ No assignments found in state or at {assignments_path}")
                return None
        
        schema = _schema_to_dict(self.state.schema)
        if not schema.get("frames"):
            schema_path = self.results_dir / "frame_schema.json"
            if schema_path.exists():
                schema = json.loads(schema_path.read_text())
        
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        # Debug: check assignments structure
        if assignments:
            sample_assignment = assignments[0]
            print(f"📊 Sample assignment keys: {list(sample_assignment.keys())}")
            if "probabilities" in sample_assignment:
                prob_keys = list(sample_assignment["probabilities"].keys())
                print(f"📊 Sample assignment has {len(prob_keys)} frame probabilities: {prob_keys[:5]}...")
            else:
                print(f"⚠️ Sample assignment missing 'probabilities' key")
        
        examples = get_top_examples_per_frame(
            assignments, corpus_index,
            frame_schema=schema,
            top_k=config.get_option("top_k", 1),
            max_text_length=config.get_option("max_length", 500),
        )
        
        print(f"📊 Found examples for {len(examples)} frames: {list(examples.keys())[:5]}...")
        
        # Check if we found any examples
        if not examples:
            print(f"⚠️ No examples found for any frames.")
            print(f"   - Assignments count: {len(assignments)}")
            print(f"   - Corpus index size: {len(corpus_index)}")
            print(f"   - Schema frames: {len(schema.get('frames', []))}")
            # Return empty file with a message
            output_path = self.get_output_path(config, "frame_examples.html")
            output_path.write_text("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Top Frame Examples</title>
</head>
<body>
    <h1>No Examples Found</h1>
    <p>No frame examples were found. This could mean:</p>
    <ul>
        <li>No assignments were loaded</li>
        <li>Assignments don't have probability scores</li>
        <li>Frame IDs in assignments don't match the schema</li>
    </ul>
</body>
</html>""")
            return output_path
        
        fmt = config.get_option("format", "html")
        if fmt == "html":
            output = format_as_html(examples, schema)
            default_name = "frame_examples.html"
        elif fmt == "markdown":
            output = format_as_markdown(examples, schema)
            default_name = "frame_examples.md"
        else:
            output = format_as_text(examples, schema)
            default_name = "frame_examples.txt"
        
        output_path = self.get_output_path(config, default_name)
        output_path.write_text(output)
        return output_path
