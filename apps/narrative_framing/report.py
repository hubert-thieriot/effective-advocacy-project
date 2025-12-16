"""HTML report generation for the narrative framing workflow."""

from __future__ import annotations

import html
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from apps.narrative_framing.run import WorkflowState, ResultPaths
    from apps.narrative_framing.config import NarrativeFramingConfig
    from efi_analyser.frames import EmbeddedCorpora
from apps.narrative_framing.aggregation_document import DocumentFrameAggregate
from apps.narrative_framing.aggregation_temporal import TemporalAggregator
from apps.narrative_framing.plots import (
    build_color_map,
    build_corpus_color_map,
    collect_top_stories_by_frame,
    compute_classifier_metrics,
    extract_domain_from_url,
    format_date_label,
    plot_domain_counts_bar,
    plot_precision_recall_bars,
    render_corpus_bar_chart,
    render_domain_frame_distribution,
    render_global_bar_chart,
    render_occurrence_by_year,
    render_occurrence_percentage_bars,
    render_plotly_classifier_by_year,
    render_plotly_classifier_percentage_bars,
    render_plotly_domain_counts,
    render_plotly_llm_binned_distribution,
    render_plotly_llm_coverage,
    render_plotly_timeseries,
    render_plotly_timeseries_lines,
    render_plotly_total_docs_timeseries,
    render_probability_bars,
    render_yearly_bar_chart,
)
from efi_analyser.frames import FrameAssignment, FrameSchema
from efi_analyser.frames.identifiers import split_passage_id
from efi_core.utils import normalize_date
import shutil



def export_frames_html(
    schema: FrameSchema,
    induction_guidance: Optional[str] = None,
    export_path: Optional[Path] = None,
    light_mode: bool = False,
    jekyll_format: bool = False,
) -> None:
    """Export frame definitions as HTML file with card styling.
    
    Args:
        schema: Frame schema containing frame definitions
        induction_guidance: Optional induction guidance text (not included in output)
        export_path: Path where to save the HTML file
        light_mode: If True, exclude keywords and examples from cards
        jekyll_format: If True, output Jekyll include format (no DOCTYPE/html/head/body tags)
    """
    if export_path is None:
        return
    
    color_map = build_color_map(schema.frames)
    
    # Build frame cards HTML
    frame_cards_html = []
    for frame in schema.frames:
        color = color_map.get(frame.frame_id, "#1E3D58")
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        
        keywords_html = ""
        if not light_mode and frame.keywords:
            keywords_html = f"""
            <div class="frame-keywords">
                <strong>Keywords:</strong> {html.escape(', '.join(frame.keywords))}
            </div>"""
        
        examples_html = ""
        if not light_mode and frame.examples:
            examples_html = f"""
            <div class="frame-examples">
                <strong>Examples:</strong> {html.escape('; '.join(frame.examples[:3]))}
            </div>"""
        
        card_html = f"""
        <div class="frame-card" style="--accent-color: {color};">
            <div class="frame-card-header">
                <h3 class="frame-card-name">{html.escape(frame.name)}</h3>
            </div>
            <div class="frame-card-body">
                <p class="frame-card-description">{html.escape(frame.description)}</p>
                {keywords_html}
                {examples_html}
            </div>
        </div>"""
        frame_cards_html.append(card_html)
    
    # CSS styles (same for both formats)
    css_styles = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.frames-container {
    max-width: 1200px;
    margin: 0 auto;
}

.frames-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 24px;
    margin-bottom: 48px;
}

.frame-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid var(--accent-color);
    padding: 20px;
    transition: box-shadow 0.2s ease;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.frame-card:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.frame-card-header {
    margin-bottom: 12px;
}

.frame-card-name {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: var(--accent-color);
    margin-bottom: 0 !important;
}

.frame-card-title {
    font-size: 14px;
    font-weight: 500;
    color: #475569;
    margin: 0;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.frame-card-body {
    font-size: 14px;
    color: #64748b;
}

.frame-card-description {
    margin-bottom: 10px;
    line-height: 1.5 ! important;
    font-size: unset ! important;
}

.frame-keywords,
.frame-examples {
    margin-top: 12px;
    font-size: 13px;
    color: #475569;
}

.frame-keywords strong,
.frame-examples strong {
    color: #1e293b;
    font-weight: 500;
}

@media (max-width: 768px) { 
    .frames-grid {
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }
}
</style>"""
    
    # Content HTML
    content_html = f"""<div class="frames-container">
    <div class="frames-grid">
        {''.join(frame_cards_html)}
    </div>
</div>"""
    
    # if jekyll_format:
        # Jekyll include format: just CSS and content, no HTML document structure
    html_content = css_styles + "\n" + content_html
    # else:
    #     # Standalone HTML format: full document
    #     html_content = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Frame Definitions - {html.escape(schema.domain)}</title>
#     {css_styles}
# </head>
# <body>
#     {content_html}
# </body>
# </html>"""
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(html_content, encoding="utf-8")


def _publish_to_docs_assets(run_dir_name: str, results_html: Optional[Path], export_plots_dir: Optional[Path] = None) -> None:
    """Copy exported Plotly PNGs, SVGs, and HTML files into docs/ for GitHub Pages.

    - PNGs, SVGs, and HTMLs are expected in results/<run_name>/plots
    - Files copied to docs/assets/narrative_framing/<run_name>/ (or export_plots_dir if specified)
    - HTML report copied to docs/reports/<run_name>/frame_report.html
    """
    try:
        # Prefer plots under the run directory; fall back to legacy location
        plots_src = (results_html.parent / "plots") if results_html else None
        if not plots_src or not plots_src.exists():
            plots_src = Path("results/plots") / run_dir_name / "plots"
        
        # Determine destination directory
        if export_plots_dir:
            plots_dst = Path(export_plots_dir)
        else:
            plots_dst = Path("docs/assets/narrative_framing") / run_dir_name
        
        report_dst = Path("docs/reports") / run_dir_name
        
        if plots_src and plots_src.exists():
            plots_dst.mkdir(parents=True, exist_ok=True)
            # Copy PNG, SVG, and HTML files
            for file_path in sorted(plots_src.glob("*.png")):
                shutil.copy2(file_path, plots_dst / file_path.name)
            for file_path in sorted(plots_src.glob("*.svg")):
                shutil.copy2(file_path, plots_dst / file_path.name)
            for file_path in sorted(plots_src.glob("*.html")):
                shutil.copy2(file_path, plots_dst / file_path.name)
        
        if results_html and results_html.exists():
            report_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_html, report_dst / "frame_report.html")
    except Exception as exc:
        print(f"⚠️ Failed to publish to docs: {exc}")


class ReportBuilder:
    """Builder class for generating HTML reports from workflow state."""
    
    def __init__(
        self,
        state: "WorkflowState",
        config: "NarrativeFramingConfig",
        paths: "ResultPaths",
        total_doc_ids: List[str],
        corpora_map: Optional["EmbeddedCorpora"] = None,
    ):
        """Initialize the ReportBuilder with state and config.
        
        Args:
            state: Workflow state containing schema, assignments, aggregates, etc.
            config: Configuration for the narrative framing workflow
            paths: Result paths for output files
            total_doc_ids: List of all document IDs in the corpus
            corpora_map: Optional corpora map for fetching full document text
        """
        self.state = state
        self.config = config
        self.paths = paths
        self.total_doc_ids = total_doc_ids
        self.corpora_map = corpora_map
    
    def build(self) -> None:
        """Generate the HTML report from the workflow state."""
        schema = self.state.schema
        assignments = list(self.state.assignments)
        classifier_predictions = self.state.classifier_predictions
        classifications = self.state.classifications
        aggregates = self.state.aggregates

        if not aggregates:
            print("⚠️ Cannot generate report: aggregates are missing.")
            return

        document_aggregates_weighted = aggregates.documents_weighted
        document_aggregates_occurrence = aggregates.documents_occurrence
        all_aggregates = aggregates.all_aggregates

        # Check if we have document aggregates (either loaded from cache or newly created)
        has_document_aggregates = document_aggregates_weighted and len(document_aggregates_weighted) > 0

        # Apply optional date filter to assignments used in report
        if self.config.filter.date_from and assignments:
            df_norm = str(self.config.filter.date_from).strip()
            filtered_as: List[FrameAssignment] = []
            for a in assignments:
                pub = a.metadata.get("published_at") if isinstance(a.metadata, dict) else None
                dt = normalize_date(pub)
                if dt and dt.date().isoformat() >= df_norm:
                    filtered_as.append(a)
            assignments = filtered_as

        # Optional per-document HTML export (works with cached annotations)
        if self.config.annotation.export_annotated_html and schema:
            try:
                self._export_annotated_documents(schema, assignments)
            except Exception as exc:
                print(f"⚠️ Failed to export annotated documents: {exc}")
        
        # Generate results browser if we have document aggregates and annotated HTML export
        if self.config.annotation.export_annotated_html and schema and has_document_aggregates:
            try:
                self._generate_results_browser(schema, document_aggregates_weighted)
            except Exception as exc:
                print(f"⚠️ Failed to generate results browser: {exc}")

        if schema and self.paths.html and has_document_aggregates:
            # Build classifier lookup for report
            classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None
            if classifier_predictions:
                classifier_lookup = {
                    item["passage_id"]: item
                    for item in classifier_predictions
                    if isinstance(item, dict) and item.get("passage_id")
                }

            include_classifier_plots = True if classifier_predictions else False

            # Count total number of classified passages (chunks) from classifications
            # If classifications is empty (e.g., in regenerate mode), use document count as fallback
            classified_passages_count = 0
            if classifications:
                for doc_record in classifications:
                    payload = doc_record.payload if hasattr(doc_record, "payload") else doc_record
                    chunks = payload.get("chunks", [])
                    if isinstance(chunks, Sequence):
                        classified_passages_count += len(chunks)
            elif document_aggregates_weighted:
                # Fallback: estimate passages from document count (rough estimate)
                classified_passages_count = len(document_aggregates_weighted) * 3  # Rough estimate: 3 chunks per doc

            usage_stats = {
                "frames": len(schema.frames),
                "corpus_documents": len(self.total_doc_ids),
                "induction_passages": len(self.state.induction_samples)
                if self.state.induction_samples
                else (self.config.induction_sample_size or 0),
                "annotation_passages": len(self.state.assignments),
                "classifier_documents": len(classifications)
                if classifications
                else len(document_aggregates_weighted),
                "classifier_passages": classified_passages_count,
            }

            write_html_report(
                schema=schema,
                assignments=assignments,
                output_path=self.paths.html,
                classifier_lookup=classifier_lookup,
                classified_documents=len(document_aggregates_weighted),
                classifier_sample_limit=self.config.classifier_corpus_sample_size,
                include_classifier_plots=include_classifier_plots,
                document_aggregates_weighted=document_aggregates_weighted,
                document_aggregates_occurrence=document_aggregates_occurrence,
                all_aggregates=all_aggregates,
                hide_empty_passages=self.config.report.hide_empty_passages,
                custom_plots=self.config.report.custom_plots,
                plot_title=self.config.report.plot.title,
                plot_subtitle=self.config.report.plot.subtitle,
                plot_note=self.config.report.plot.note,
                export_plotly_png_dir=(self.paths.html.parent / "plots"),
                export_plot_formats=self.config.report.export_plot_formats,
                induction_guidance=self.config.induction_guidance,
                export_includes_dir=self.config.report.export_includes_dir,
                usage_stats=usage_stats,
                n_min_per_media=self.config.report.n_min_per_media,
                domain_mapping_max_domains=self.config.report.domain_mapping_max_domains,
                corpus_aliases=self.config.corpus_aliases,
                include_yearly_bar_charts=self.config.report.include_yearly_bar_charts,
                include_domain_yearly_bar_charts=self.config.report.include_domain_yearly_bar_charts,
                domain_yearly_top_domains=self.config.report.domain_yearly_top_domains,
                domain_yearly_version=self.config.report.domain_yearly_version,
                domain_yearly_include_empty=self.config.report.domain_yearly_include_empty,
            )

            # Publish PNGs and HTML to docs for GitHub Pages
            try:
                _publish_to_docs_assets(
                    self.paths.html.parent.name,
                    self.paths.html,
                    export_plots_dir=self.config.report.export_plots_dir,
                )
            except Exception as exc:
                print(f"⚠️ Failed to publish docs assets: {exc}")

            print(f"\n✅ HTML report written to {self.paths.html}")
        else:
            if not schema:
                print("⚠️ Cannot generate report: schema is missing.")
            elif not self.paths.html:
                print("⚠️ Cannot generate report: HTML output path is not configured.")
            elif not has_document_aggregates:
                print(
                    "⚠️ Cannot generate report: document aggregates are missing "
                    f"(loaded: {bool(document_aggregates_weighted)}, "
                    f"count: {len(document_aggregates_weighted) if document_aggregates_weighted else 0})."
                )
                if self.config.regenerate_report_only:
                    print(
                        "   Hint: In regenerate mode, ensure documents_weighted.json exists in the aggregates directory."
                    )
                    weighted_path = (
                        self.paths.aggregates_dir / "documents_weighted.json" if self.paths.aggregates_dir else None
                    )
                    if weighted_path and not weighted_path.exists():
                        print(f"   The file {weighted_path} does not exist.")
                        # Check if chunk classifications exist as an alternative
                        if self.paths.classifications_dir and self.paths.classifications_dir.exists():
                            chunk_files = list(self.paths.classifications_dir.glob("*.json"))
                            if chunk_files:
                                print(
                                    f"   Found {len(chunk_files)} chunk classification files. You can either:"
                                )
                                print(
                                    "   1. Run without 'regenerate_report_only' to create aggregates from "
                                    "chunk classifications, or"
                                )
                                print(
                                    "   2. Manually create aggregates from existing chunk classifications "
                                    "and save them as documents_weighted.json."
                                )
                    else:
                        print(
                            f"   Note: No document aggregates available "
                            f"(count: {len(document_aggregates_weighted) if document_aggregates_weighted else 0})."
                        )

    # ------------------------------------------------------------------ Annotated HTML export
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        cleaned = (
            name.replace("@@", "__")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        return cleaned or "document"
    
    @staticmethod
    def _get_nested_metadata_value(metadata: Dict[str, Any], field_path: str) -> Optional[str]:
        """Extract a value from nested metadata using a dot-separated path.
        
        Examples:
            - "country_name" -> metadata.get("country_name")
            - "extra.country_name" -> metadata.get("extra", {}).get("country_name")
            - "extra.party_name" -> metadata.get("extra", {}).get("party_name")
        
        Returns the value as a string, or None if not found.
        """
        parts = field_path.split(".")
        value = metadata
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
            if value is None:
                return None
        # Convert to string, handling None and empty values
        if value is None or value == "":
            return None
        return str(value).strip()

    @staticmethod
    def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
        c = color.lstrip("#")
        if len(c) == 6:
            return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))
        return (0, 0, 0)

    @staticmethod
    def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return not (a_end <= b_start or a_start >= b_end)

    def _find_span_with_fallback(
        self, text: str, chunk_text: str, used_ranges: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Find chunk span in text, allowing for ellipses/whitespace differences."""
        if not chunk_text or not chunk_text.strip():
            return None

        text_lower = text.lower()
        chunk_lower = chunk_text.lower()
        chunk_len = len(chunk_text)

        def available_start(start_idx: int) -> bool:
            return all(not self._spans_overlap(start_idx, start_idx + chunk_len, s, e) for s, e in used_ranges)

        # First: direct substring match
        start = text_lower.find(chunk_lower)
        while start != -1:
            if available_start(start):
                return start, start + chunk_len
            start = text_lower.find(chunk_lower, start + 1)

        # Fallback: build regex that tolerates whitespace differences and ellipses
        escaped = re.escape(chunk_text)
        escaped = re.sub(r"\\s+", r"\\s+", escaped)  # allow flexible whitespace
        escaped = escaped.replace(r"\.\.\.", r".{0,160}?")  # allow gaps where ellipses were inserted
        try:
            pattern = re.compile(escaped, re.IGNORECASE | re.DOTALL)
            for m in pattern.finditer(text):
                s, e = m.start(), m.end()
                if all(not self._spans_overlap(s, e, us, ue) for us, ue in used_ranges):
                    return s, e
        except re.error:
            return None
        return None

    def _export_annotated_documents(self, schema: FrameSchema, assignments: Sequence[FrameAssignment]) -> None:
        if not assignments or not self.corpora_map:
            return

        # Determine output directory under results
        if self.paths.assignments:
            base_dir = self.paths.assignments.parent
        elif self.config.results_dir:
            base_dir = Path(self.config.results_dir)
        else:
            base_dir = Path("results") / "narrative_framing"
        out_dir = base_dir / "annotated_documents"
        out_dir.mkdir(parents=True, exist_ok=True)

        color_map = build_color_map(schema.frames)
        frame_titles = {frame.frame_id: frame.name or frame.short_name or frame.frame_id for frame in schema.frames}

        from collections import defaultdict

        doc_assignments: Dict[Tuple[Optional[str], str], List[FrameAssignment]] = defaultdict(list)
        for assignment in assignments:
            corpus_name, local_doc_id, _ = split_passage_id(assignment.passage_id)
            doc_assignments[(corpus_name, local_doc_id)].append(assignment)

        for (corpus_name, local_doc_id), doc_items in doc_assignments.items():
            try:
                embedded = self.corpora_map.get_embedded(corpus_name)
            except Exception:
                continue
            try:
                doc = embedded.corpus.get_document(local_doc_id)
            except Exception:
                continue
            if not doc or not getattr(doc, "text", None):
                continue

            text = doc.text
            text_lower = text.lower()
            spans: List[Tuple[int, int, str, float]] = []
            used_ranges: List[Tuple[int, int]] = []

            for assignment in doc_items:
                if not assignment.probabilities:
                    continue
                frame_id, prob = max(assignment.probabilities.items(), key=lambda kv: kv[1])
                if prob <= 0:
                    continue
                chunk_text = assignment.passage_text or ""
                span = self._find_span_with_fallback(text, chunk_text, used_ranges)
                if span is None:
                    continue
                start, end = span
                spans.append((start, end, frame_id, float(prob)))
                used_ranges.append((start, end))

            if not spans:
                continue

            spans.sort(key=lambda x: x[0])
            highlighted_parts: List[str] = []
            cursor = 0
            used_frame_ids: List[str] = []

            for start, end, frame_id, prob in spans:
                if start > cursor:
                    highlighted_parts.append(html.escape(text[cursor:start]))
                rgb = self._hex_to_rgb(color_map.get(frame_id, "#000000"))
                opacity = max(0.2, min(0.95, prob))
                segment = html.escape(text[start:end])
                highlighted_parts.append(
                    f'<span class="frame" style="background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity:.2f});" title="{html.escape(frame_id)} ({prob:.2f})">{segment}</span>'
                )
                cursor = end
                used_frame_ids.append(frame_id)
            if cursor < len(text):
                highlighted_parts.append(html.escape(text[cursor:]))

            doc_body = "".join(highlighted_parts).replace("\n", "<br>\n")
            legend_items: List[str] = []
            for frame_id in sorted(set(used_frame_ids)):
                r, g, b = self._hex_to_rgb(color_map.get(frame_id, "#000000"))
                title = frame_titles.get(frame_id, frame_id)
                legend_items.append(
                    f'<span class="legend-item"><span class="swatch" style="background-color: rgb({r}, {g}, {b});"></span>{html.escape(frame_id)} – {html.escape(title)}</span>'
                )
            legend_html = "\n".join(legend_items)

            title = getattr(doc, "title", None) or local_doc_id
            global_doc_id = (
                f"{corpus_name}@@{local_doc_id}" if corpus_name else local_doc_id
            )
            filename = self._sanitize_filename(global_doc_id) + ".html"
            
            # Get document metadata (including extra field)
            doc_metadata = {}
            try:
                doc_metadata = embedded.corpus.get_metadata(local_doc_id) or {}
            except Exception:
                pass
            
            # Build subfolder path if configured
            subfolder_path = Path("")
            if self.config.annotation.annotated_html_subfolder_fields:
                subfolder_parts = []
                for field_path in self.config.annotation.annotated_html_subfolder_fields:
                    value = self._get_nested_metadata_value(doc_metadata, field_path)
                    if value:
                        # Sanitize folder name
                        sanitized = self._sanitize_filename(str(value))
                        subfolder_parts.append(sanitized)
                    else:
                        # Use "unknown" for missing values
                        subfolder_parts.append("unknown")
                if subfolder_parts:
                    subfolder_path = Path(*subfolder_parts)
            
            # Create full output path
            final_out_dir = out_dir / subfolder_path
            final_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = final_out_dir / filename
            
            # Format metadata for display
            metadata_html_parts = []
            metadata_html_parts.append(f"<strong>Doc ID:</strong> {html.escape(global_doc_id)}")
            
            # Add all metadata fields
            for key, value in doc_metadata.items():
                if key == "extra" and isinstance(value, dict):
                    # Handle extra field specially
                    for extra_key, extra_value in value.items():
                        if extra_value is not None and extra_value != "":
                            metadata_html_parts.append(f"<strong>{html.escape(str(extra_key))}:</strong> {html.escape(str(extra_value))}")
                elif value is not None and value != "":
                    metadata_html_parts.append(f"<strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}")
            
            metadata_html = "<br>\n".join(metadata_html_parts)

            html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 1.5rem; max-width: 960px; }}
    .legend {{ margin-bottom: 1rem; }}
    .legend-item {{ margin-right: 12px; display: inline-flex; align-items: center; gap: 6px; font-size: 0.95rem; }}
    .swatch {{ width: 14px; height: 14px; display: inline-block; border-radius: 3px; border: 1px solid #e0e0e0; }}
    .frame {{ border-radius: 3px; padding: 1px 2px; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1rem; padding: 1rem; background-color: #f5f5f5; border-radius: 4px; }}
    .meta strong {{ color: #333; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">{metadata_html}</div>
  <div class="legend">{legend_html}</div>
  <div class="document">{doc_body}</div>
</body>
</html>
"""
            out_path.write_text(html_doc, encoding="utf-8")
        print(f"✅ Exported annotated documents to {out_dir}")

    def _generate_results_browser(
        self, schema: FrameSchema, document_aggregates_weighted: Sequence[DocumentFrameAggregate]
    ) -> None:
        """Generate a results browser HTML page for filtering and viewing annotated documents."""
        if not document_aggregates_weighted:
            return

        # Determine output directory
        if self.paths.assignments:
            base_dir = self.paths.assignments.parent
        elif self.config.results_dir:
            base_dir = Path(self.config.results_dir)
        else:
            base_dir = Path("results") / "narrative_framing"
        
        classifications_dir = base_dir / "classifications"
        output_path = base_dir / "results_browser.html"

        # Build color map and frame lookup
        color_map = build_color_map(schema.frames)
        frame_lookup = {
            frame.frame_id: {
                "name": frame.name,
                "short": frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id),
            }
            for frame in schema.frames
        }

        # Load all classification data and build a map of doc_id -> classification data
        classification_data: Dict[str, Dict[str, Any]] = {}
        if classifications_dir.exists():
            for json_file in classifications_dir.glob("*.json"):
                try:
                    doc_id = json_file.stem
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                        classification_data[doc_id] = data
                except Exception as exc:
                    print(f"⚠️ Failed to load classification {json_file.name}: {exc}")
                    continue

        # Extract unique domains and years
        domains: set[str] = set()
        years: set[int] = set()
        articles_data: List[Dict[str, Any]] = []

        for agg in document_aggregates_weighted:
            # Check if classification data exists
            if agg.doc_id not in classification_data:
                continue
            
            # Extract domain
            domain = agg.domain
            if not domain and agg.url:
                domain = extract_domain_from_url(agg.url)
            if domain:
                domains.add(domain)
            else:
                domains.add("Unknown")

            # Extract year from published_at
            year = None
            if agg.published_at:
                try:
                    dt = normalize_date(agg.published_at)
                    if dt:
                        year = dt.year
                        years.add(year)
                except Exception:
                    pass
            if year is None:
                years.add(0)  # Use 0 for unknown year

            # Get top frames (sorted by score)
            frame_scores = agg.frame_scores or {}
            top_frames = sorted(
                [(fid, score) for fid, score in frame_scores.items() if score > 0],
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 frames

            articles_data.append({
                "doc_id": agg.doc_id,
                "title": agg.title or agg.doc_id,
                "domain": domain or "Unknown",
                "year": year or 0,
                "url": agg.url or "",
                "top_frames": top_frames,
            })

        # Sort domains and years
        sorted_domains = sorted([d for d in domains if d != "Unknown"]) + (["Unknown"] if "Unknown" in domains else [])
        sorted_years = sorted([y for y in years if y > 0], reverse=True) + ([0] if 0 in years else [])

        # Build article list HTML
        article_items: List[str] = []
        for article in articles_data:
            domain_display = html.escape(article["domain"])
            year_display = str(article["year"]) if article["year"] > 0 else "Unknown"
            title_display = html.escape(article["title"])
            
            # Build frame badges with proportional bars
            frame_badges: List[str] = []
            for frame_id, score in article["top_frames"]:
                color = color_map.get(frame_id, "#1E3D58")
                frame_name = frame_lookup.get(frame_id, {}).get("short", frame_id)
                pct = f"{score * 100:.0f}%"
                width_pct = score * 100
                frame_badges.append(
                    f'<span class="frame-badge" title="{html.escape(frame_id)}">'
                    f'<span class="frame-badge-bar" style="background-color: {color}; width: {width_pct:.1f}%;"></span>'
                    f'<span class="frame-badge-label">{html.escape(frame_name)}</span>'
                    f'<span class="frame-badge-value">{pct}</span>'
                    f'</span>'
                )
            
            # Build metadata badges
            meta_badges = [
                f'<span class="meta-badge domain">{html.escape(domain_display)}</span>',
                f'<span class="meta-badge year">{html.escape(year_display)}</span>',
            ]
            
            # External URL link if available
            url_link = ""
            if article["url"]:
                url_link = f'<a href="{html.escape(article["url"])}" target="_blank" rel="noopener noreferrer" class="external-link">🔗</a>'
            
            article_items.append(
                f'''<article class="article-item" data-domain="{html.escape(article["domain"])}" data-year="{article["year"]}" data-doc-id="{html.escape(article["doc_id"])}">
                    <div class="article-header">
                        <h3 class="article-title">
                            <a href="#" class="article-link" data-doc-id="{html.escape(article["doc_id"])}">{title_display}</a>
                            {url_link}
                        </h3>
                        <div class="article-meta">
                            {''.join(meta_badges)}
                        </div>
                    </div>
                    <div class="article-frames">
                        {''.join(frame_badges) if frame_badges else '<span class="no-frames">No frames detected</span>'}
                    </div>
                </article>'''
            )

        # Build filter dropdowns
        domain_options = '<option value="">All domains</option>' + ''.join(
            f'<option value="{html.escape(d)}">{html.escape(d)}</option>' for d in sorted_domains
        )
        year_options = '<option value="">All years</option>' + ''.join(
            f'<option value="{y}">{y if y > 0 else "Unknown"}</option>' for y in sorted_years
        )

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(schema.domain)} — Results Browser</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {{
      --ink-900: #0f172a;
      --ink-800: #1e293b;
      --ink-600: #334155;
      --ink-500: #475569;
      --ink-300: #94a3b8;
      --slate-50: #f7f9fc;
      --slate-100: #eef2f9;
      --border: #d3dce7;
      --accent-1: #1e3d58;
      --accent-2: #057d9f;
      --accent-3: #f18f01;
      --accent-4: #6c63ff;
      --success: #3a7d44;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 48px;
      background: linear-gradient(140deg, var(--slate-50) 0%, #e4edf7 100%);
      font-family: 'Inter', 'Segoe UI', sans-serif;
      color: var(--ink-800);
    }}
    .browser-page {{
      max-width: 1320px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 18px;
      box-shadow: 0 28px 68px rgba(15, 23, 42, 0.12);
      padding: 48px 64px 72px;
    }}
    .browser-header {{
      margin-bottom: 32px;
      padding-bottom: 24px;
      border-bottom: 2px solid var(--border);
    }}
    .browser-header h1 {{
      margin: 0 0 8px 0;
      font-size: 2rem;
      color: var(--ink-900);
    }}
    .browser-header p {{
      margin: 0 0 24px 0;
      color: var(--ink-600);
      font-size: 1rem;
    }}
    .filters {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .filter-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .filter-group label {{
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--ink-600);
    }}
    .filter-group select {{
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 0.95rem;
      font-family: inherit;
      background: #ffffff;
      color: var(--ink-800);
      cursor: pointer;
      min-width: 180px;
    }}
    .filter-group select:hover {{
      border-color: var(--accent-2);
    }}
    .filter-group select:focus {{
      outline: none;
      border-color: var(--accent-2);
      box-shadow: 0 0 0 3px rgba(5, 125, 159, 0.1);
    }}
    .results-count {{
      margin-left: auto;
      padding: 10px 16px;
      background: var(--slate-100);
      border-radius: 8px;
      font-size: 0.9rem;
      color: var(--ink-600);
    }}
    .results-count strong {{
      color: var(--ink-800);
      font-weight: 600;
    }}
    .articles-list {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .article-item {{
      padding: 20px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    .article-item:hover {{
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
      border-color: var(--accent-2);
    }}
    .article-item.hidden {{
      display: none;
    }}
    .article-header {{
      margin-bottom: 12px;
    }}
    .article-title {{
      margin: 0 0 8px 0;
      font-size: 1.1rem;
      font-weight: 600;
    }}
    .article-title a {{
      color: var(--ink-900);
      text-decoration: none;
    }}
    .article-title a:hover {{
      color: var(--accent-2);
      text-decoration: underline;
    }}
    .external-link {{
      margin-left: 8px;
      font-size: 0.9rem;
      opacity: 0.6;
      transition: opacity 0.2s;
    }}
    .external-link:hover {{
      opacity: 1;
    }}
    .article-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .meta-badge {{
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      background: var(--slate-100);
      color: var(--ink-600);
    }}
    .meta-badge.domain {{
      background: var(--accent-1);
      color: #ffffff;
    }}
    .meta-badge.year {{
      background: var(--accent-3);
      color: #ffffff;
    }}
    .article-frames {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .frame-badge {{
      display: inline-flex;
      align-items: center;
      min-width: 80px;
      height: 24px;
      border-radius: 6px;
      overflow: hidden;
      background-color: var(--slate-100);
      position: relative;
    }}
    .frame-badge-bar {{
      height: 100%;
      display: flex;
      align-items: center;
      padding: 0 8px;
      color: #ffffff;
      font-size: 0.85rem;
      font-weight: 500;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
      white-space: nowrap;
      min-width: fit-content;
    }}
    .frame-badge-label {{
      position: absolute;
      left: 8px;
      z-index: 1;
      color: var(--ink-800);
      font-size: 0.85rem;
      font-weight: 500;
      pointer-events: none;
    }}
    .frame-badge-value {{
      position: absolute;
      right: 8px;
      z-index: 1;
      color: var(--ink-800);
      font-size: 0.85rem;
      font-weight: 500;
      pointer-events: none;
    }}
    .no-frames {{
      color: var(--ink-300);
      font-size: 0.9rem;
      font-style: italic;
    }}
    .article-link {{
      cursor: pointer;
    }}
    .modal {{
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(15, 23, 42, 0.7);
      z-index: 1000;
      overflow-y: auto;
      padding: 40px 20px;
    }}
    .modal.active {{
      display: flex;
      align-items: flex-start;
      justify-content: center;
    }}
    .modal-content {{
      background: #ffffff;
      border-radius: 18px;
      box-shadow: 0 28px 68px rgba(15, 23, 42, 0.3);
      max-width: 1200px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 40px;
      margin: auto;
      position: relative;
    }}
    .modal-header {{
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid var(--border);
    }}
    .modal-header h2 {{
      margin: 0 0 8px 0;
      font-size: 1.5rem;
      color: var(--ink-900);
    }}
    .modal-header .meta {{
      color: var(--ink-600);
      font-size: 0.9rem;
    }}
    .modal-close {{
      position: absolute;
      top: 20px;
      right: 20px;
      background: var(--slate-100);
      border: none;
      border-radius: 8px;
      width: 36px;
      height: 36px;
      font-size: 1.2rem;
      cursor: pointer;
      color: var(--ink-600);
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .modal-close:hover {{
      background: var(--border);
    }}
    .chunks-container {{
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .chunk-item {{
      display: flex;
      gap: 16px;
      align-items: flex-start;
    }}
    .chunk-frames {{
      flex-shrink: 0;
      width: 200px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .chunk-frame-badge {{
      display: flex;
      align-items: center;
      min-width: 120px;
      height: 22px;
      border-radius: 4px;
      overflow: hidden;
      background-color: var(--slate-100);
      position: relative;
      margin-bottom: 4px;
    }}
    .chunk-frame-badge-bar {{
      height: 100%;
      display: flex;
      align-items: center;
      padding: 0 6px;
      color: #ffffff;
      font-size: 0.8rem;
      font-weight: 500;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
      white-space: nowrap;
      min-width: fit-content;
    }}
    .chunk-frame-badge-label {{
      position: absolute;
      left: 6px;
      z-index: 1;
      color: var(--ink-800);
      font-size: 0.8rem;
      font-weight: 500;
      pointer-events: none;
    }}
    .chunk-frame-badge-value {{
      position: absolute;
      right: 6px;
      z-index: 1;
      color: var(--ink-800);
      font-size: 0.8rem;
      font-weight: 600;
      pointer-events: none;
    }}
    .chunk-text {{
      flex: 1;
      line-height: 1.6;
      color: var(--ink-800);
      padding: 12px;
      background: var(--slate-50);
      border-radius: 8px;
      border-left: 3px solid var(--border);
    }}
    .loading {{
      text-align: center;
      padding: 40px;
      color: var(--ink-600);
    }}
    @media (max-width: 860px) {{
      body {{ padding: 24px; }}
      .browser-page {{ padding: 32px 24px 48px; }}
      .filters {{
        flex-direction: column;
        align-items: stretch;
      }}
      .filter-group select {{
        width: 100%;
      }}
      .results-count {{
        margin-left: 0;
        width: 100%;
        text-align: center;
      }}
    }}
  </style>
</head>
<body>
  <div class="browser-page">
    <div class="browser-header">
      <h1>{html.escape(schema.domain)} — Results Browser</h1>
      <p>Browse and filter annotated documents by domain and year</p>
      <div class="filters">
        <div class="filter-group">
          <label for="domain-filter">Domain</label>
          <select id="domain-filter">
            {domain_options}
          </select>
        </div>
        <div class="filter-group">
          <label for="year-filter">Year</label>
          <select id="year-filter">
            {year_options}
          </select>
        </div>
        <div class="results-count">
          Showing <strong id="results-count">{len(articles_data)}</strong> of {len(articles_data)} articles
        </div>
      </div>
    </div>
    <div class="articles-list" id="articles-list">
      {''.join(article_items)}
    </div>
  </div>
  <div id="modal" class="modal">
    <div class="modal-content">
      <button class="modal-close" onclick="closeModal()">×</button>
      <div class="modal-header">
        <h2 id="modal-title">Loading...</h2>
        <div class="meta" id="modal-meta"></div>
      </div>
      <div id="modal-body">
        <div class="loading">Loading article...</div>
      </div>
    </div>
  </div>
  <script>
    // Frame color map and lookup
    const colorMap = {json.dumps(color_map, ensure_ascii=False)};
    const frameLookup = {json.dumps(frame_lookup, ensure_ascii=False)};
    
    // Embedded classification data (loaded upfront to avoid CORS issues)
    const classificationData = {json.dumps(classification_data, ensure_ascii=False)};
    
    const domainFilter = document.getElementById('domain-filter');
    const yearFilter = document.getElementById('year-filter');
    const articlesList = document.getElementById('articles-list');
    const resultsCount = document.getElementById('results-count');
    const articles = Array.from(articlesList.querySelectorAll('.article-item'));
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalMeta = document.getElementById('modal-meta');
    const modalBody = document.getElementById('modal-body');
    
    function updateFilters() {{
      const selectedDomain = domainFilter.value;
      const selectedYear = yearFilter.value;
      
      let visibleCount = 0;
      
      articles.forEach(article => {{
        const articleDomain = article.getAttribute('data-domain') || '';
        const articleYear = article.getAttribute('data-year') || '';
        
        const domainMatch = !selectedDomain || articleDomain === selectedDomain;
        const yearMatch = !selectedYear || articleYear === selectedYear;
        
        if (domainMatch && yearMatch) {{
          article.classList.remove('hidden');
          visibleCount++;
        }} else {{
          article.classList.add('hidden');
        }}
      }});
      
      resultsCount.textContent = visibleCount;
    }}
    
    function closeModal() {{
      modal.classList.remove('active');
      modalBody.innerHTML = '<div class="loading">Loading article...</div>';
    }}
    
    function renderChunks(classification) {{
      const chunks = classification.chunks || [];
      if (chunks.length === 0) {{
        modalBody.innerHTML = '<p>No chunks available.</p>';
        return;
      }}
      
      let chunksHtml = '<div class="chunks-container">';
      
      chunks.forEach(chunk => {{
        const probabilities = chunk.probabilities || {{}};
        const text = chunk.text || '';
        
        // Get frames with weight > 0, sorted by weight
        const frames = Object.entries(probabilities)
          .filter(([frameId, weight]) => weight > 0)
          .sort((a, b) => b[1] - a[1]);
        
        let framesHtml = '<div class="chunk-frames">';
        if (frames.length === 0) {{
          framesHtml += '<div style="color: var(--ink-300); font-size: 0.85rem; font-style: italic;">No frames</div>';
        }} else {{
          frames.forEach(([frameId, weight]) => {{
            const color = colorMap[frameId] || '#1E3D58';
            const frameInfo = frameLookup[frameId] || {{}};
            const frameName = frameInfo.short || frameId;
            const pct = (weight * 100).toFixed(0);
            const widthPct = (weight * 100).toFixed(1);
            framesHtml += '<div class="chunk-frame-badge">' +
              '<div class="chunk-frame-badge-bar" style="background-color: ' + color + '; width: ' + widthPct + '%;"></div>' +
              '<span class="chunk-frame-badge-label">' + escapeHtml(frameName) + '</span>' +
              '<span class="chunk-frame-badge-value">' + pct + '%</span>' +
            '</div>';
          }});
        }}
        framesHtml += '</div>';
        
        const textHtml = '<div class="chunk-text">' + escapeHtml(text) + '</div>';
        chunksHtml += '<div class="chunk-item">' + framesHtml + textHtml + '</div>';
      }});
      
      chunksHtml += '</div>';
      modalBody.innerHTML = chunksHtml;
    }}
    
    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}
    
    function loadArticle(docId) {{
      modal.classList.add('active');
      modalTitle.textContent = 'Loading...';
      modalMeta.textContent = '';
      modalBody.innerHTML = '<div class="loading">Loading article...</div>';
      
      // Use embedded classification data instead of fetch
      const data = classificationData[docId];
      if (!data) {{
        modalBody.innerHTML = '<p style="color: #d32f2f;">Classification data not found for this article.</p>';
        return;
      }}
      
      modalTitle.textContent = data.title || docId;
      const metaParts = [];
      if (data.domain) metaParts.push(data.domain);
      if (data.published_at) metaParts.push(data.published_at);
      if (data.url) metaParts.push('<a href="' + data.url + '" target="_blank">View original</a>');
      modalMeta.innerHTML = metaParts.join(' • ');
      renderChunks(data);
    }}
    
    // Add click handlers to article links
    articlesList.addEventListener('click', (e) => {{
      if (e.target.classList.contains('article-link')) {{
        e.preventDefault();
        const articleItem = e.target.closest('.article-item');
        const docId = articleItem.getAttribute('data-doc-id');
        loadArticle(docId);
      }}
    }});
    
    // Close modal on background click
    modal.addEventListener('click', (e) => {{
      if (e.target === modal) {{
        closeModal();
      }}
    }});
    
    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape' && modal.classList.contains('active')) {{
        closeModal();
      }}
    }});
    
    domainFilter.addEventListener('change', updateFilters);
    yearFilter.addEventListener('change', updateFilters);
  </script>
</body>
</html>
"""

        output_path.write_text(html_content, encoding="utf-8")
        print(f"✅ Generated results browser at {output_path}")


def _slugify(value: str) -> str:
    """Simple slug generator for chart IDs."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "domain"


def write_html_report(
    schema: FrameSchema,
    assignments: Sequence[FrameAssignment],
    output_path: Path,
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    classified_documents: int = 0,
    classifier_sample_limit: Optional[int] = None,
    include_classifier_plots: bool = True,
    document_aggregates_weighted: Optional[Sequence[DocumentFrameAggregate]] = None,
    document_aggregates_occurrence: Optional[Sequence[DocumentFrameAggregate]] = None,
    all_aggregates: Optional[Dict[str, object]] = None,
    metrics_threshold: float = 0.3,
    hide_empty_passages: bool = False,
    plot_title: Optional[str] = None,
    plot_subtitle: Optional[str] = None,
    plot_note: Optional[str] = None,
    export_plotly_png_dir: Optional[Path] = None,
    export_plot_formats: Optional[List[str]] = None,
    custom_plots: Optional[Sequence] = None,
    induction_guidance: Optional[str] = None,
    export_includes_dir: Optional[Path] = None,
    usage_stats: Optional[Dict[str, int]] = None,
    n_min_per_media: Optional[int] = None,
    domain_mapping_max_domains: int = 20,
    corpus_aliases: Optional[Dict[str, str]] = None,
    include_yearly_bar_charts: bool = True,
    include_domain_yearly_bar_charts: bool = False,
    domain_yearly_top_domains: int = 5,
    domain_yearly_version: str = "weighted",
    domain_yearly_include_empty: bool = False,
) -> None:
    """Render a compact HTML report for frame assignments."""
    
    # Normalize export_plot_formats
    if export_plot_formats is None:
        export_plot_formats = ["png"]  # Default
    export_plot_formats = [f.lower() for f in export_plot_formats]
    export_svg = "svg" in export_plot_formats
    export_png = "png" in export_plot_formats
    
    # Extract data from all_aggregates if provided
    global_frame_share: Dict[str, float] = {}
    timeseries_records: Optional[Sequence[Dict[str, object]]] = None
    domain_counts: Optional[Sequence[Tuple[str, int]]] = None
    domain_frame_summaries: Optional[Sequence[Dict[str, object]]] = None
    
    if all_aggregates:
        # Extract global_frame_share from global aggregates
        global_weighted = all_aggregates.get("global_weighted_with_zeros", [])
        if global_weighted and isinstance(global_weighted, list) and len(global_weighted) > 0:
            if isinstance(global_weighted[0], dict):
                global_frame_share = global_weighted[0].get("frame_scores", {})
            else:
                # It's a PeriodAggregate object
                from apps.narrative_framing.aggregation_temporal import PeriodAggregate
                if isinstance(global_weighted[0], PeriodAggregate):
                    global_frame_share = global_weighted[0].frame_scores
        
        # Extract timeseries records
        timeseries_records = all_aggregates.get("time_series_30day", [])
        
        # Extract domain data
        domain_frame_summaries = all_aggregates.get("domain_weighted_with_zeros", [])
        if domain_frame_summaries:
            domain_counts = [(d["domain"], d["count"]) for d in domain_frame_summaries]

    color_map = build_color_map(schema.frames)
    frame_lookup = {
        frame.frame_id: {
            "name": frame.name,
            "short": frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id),
            "description": frame.description,
            "keywords": ", ".join(frame.keywords),
        }
        for frame in schema.frames
    }
    # Generate classifier performance plots if requested
    classifier_plots_html = ""
    if include_classifier_plots and assignments:
        try:
            frame_ids = [frame.frame_id for frame in schema.frames]
            frame_names = {frame.frame_id: frame.name for frame in schema.frames}

            metrics = compute_classifier_metrics(
                assignments,
                frame_ids,
                threshold=metrics_threshold,
                classifier_lookup=classifier_lookup,
            )
            precision_recall_b64 = plot_precision_recall_bars(metrics, frame_names, color_map)

            classifier_plots_html = (
                "<h3>Classifying Results</h3>"
                "<div class=\"card chart-card\">"
                "<h4>Precision &amp; Recall</h4>"
                f"<img src=\"data:image/png;base64,{precision_recall_b64}\" alt=\"Precision and Recall by Frame\" />"
                "</div>"
            )
        except Exception as exc:
            classifier_plots_html = f"""
            <h3>Classifying Results</h3>
            <div class="card chart-card">
                <p class="error-note">Error generating classifier performance plots: {html.escape(str(exc))}</p>
            </div>
            """

    frame_cards: List[str] = []
    for frame in schema.frames:
        color = color_map.get(frame.frame_id, "#1E3D58")
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        share_badge = ""
        if global_frame_share and frame.frame_id in global_frame_share:
            share_value = max(0.0, float(global_frame_share[frame.frame_id]))
            share_badge = f"<span class=\"share-badge\">{share_value * 100:.0f}%</span>"
        card_parts = [
            f"<article class=\"frame-card\" style=\"--accent-color:{color};\">",
            share_badge,
            f"<h3>{html.escape(short_label)}</h3>",
            f"<p class=\"frame-card-title\">{html.escape(frame.name)}</p>",
        ]
        if frame.description:
            card_parts.append(
                f"<p class=\"frame-card-text\">{html.escape(frame.description)}</p>"
            )
        if frame.keywords:
            card_parts.append(
                f"<p class=\"frame-card-meta\"><strong>Keywords:</strong> {html.escape(', '.join(frame.keywords))}</p>"
            )
        if frame.examples:
            card_parts.append(
                f"<p class=\"frame-card-meta\"><strong>Examples:</strong> {html.escape('; '.join(frame.examples[:2]))}</p>"
            )
        card_parts.append("</article>")
        frame_cards.append("".join(card_parts))

    coverage_text = "No documents were classified."
    if classified_documents > 0:
        coverage_text = f"Classifier applied to {classified_documents} documents."

    timeseries_note = ""
    if timeseries_records:
        date_values = [str(item.get("date", "")) for item in timeseries_records if item.get("date")]
        if date_values:
            start = min(date_values)
            end = max(date_values)
            start_label = format_date_label(start)
            end_label = format_date_label(end)
            if start_label and end_label:
                timeseries_note = f"{start_label} – {end_label}"
            else:
                timeseries_note = f"Data covers {html.escape(start)} to {html.escape(end)}."

    # Decide export directory for plots if requested
    export_dir = export_plotly_png_dir
    if export_dir is None and output_path is not None and (export_png or export_svg):
        # Default under the run directory: <run_dir>/plots
        export_dir = output_path.parent / "plots"
    
    # Export frames HTML to plots directory (standalone HTML)
    if export_dir:
        export_frames_html(
            schema=schema,
            induction_guidance=induction_guidance,
            export_path=export_dir / "frames.html",
            light_mode=False,
            jekyll_format=False
        )
        export_frames_html(
            schema=schema,
            induction_guidance=induction_guidance,
            export_path=export_dir / "frames_light.html",
            light_mode=True,
            jekyll_format=False
        )
    
    # Export frames HTML to Jekyll includes directory
    if export_includes_dir:
        # Export frames_light.html directly to includes directory
        export_frames_html(
            schema=schema,
            induction_guidance=induction_guidance,
            export_path=export_includes_dir / "frames_light.html",
            light_mode=True,  # Use light mode for includes
            jekyll_format=True
        )

    # Build caption for time series
    ts_caption_parts: List[str] = []
    if timeseries_note:
        ts_caption_parts.append(timeseries_note)
    ts_caption_parts.append("30-day rolling average of frame share.")
    if plot_note:
        try:
            ts_caption_parts.append(plot_note.format(n_articles=classified_documents))
        except Exception:
            ts_caption_parts.append(plot_note)
    ts_caption = " • ".join([p for p in ts_caption_parts if p])

    timeseries_chart_html = render_plotly_timeseries(
        timeseries_records, frame_lookup, color_map,
        title=plot_title, subtitle=(plot_subtitle.format(n_articles=classified_documents) if plot_subtitle else None),
        caption=ts_caption,
        export_png_path=(export_dir / "time_series_area.png") if (export_dir and export_png) else None,
        export_svg=export_svg,
    )
    timeseries_lines_html = render_plotly_timeseries_lines(timeseries_records, frame_lookup, color_map)
    
    # Total docs volume chart
    total_docs_chart_html = ""
    if document_aggregates_occurrence:
        # Use occurrence aggregates since they're simpler (one per document)
        total_docs_chart_html = render_plotly_total_docs_timeseries(
            document_aggregates_occurrence,
            export_png_path=(export_dir / "article_volume_over_time.png") if (export_dir and export_png) else None,
            export_svg=export_svg,
        )
    elif document_aggregates_weighted:
        # Fallback to weighted aggregates if occurrence not available
        total_docs_chart_html = render_plotly_total_docs_timeseries(
            document_aggregates_weighted,
            export_png_path=(export_dir / "article_volume_over_time.png") if (export_dir and export_png) else None,
            export_svg=export_svg,
        )

    # Prepare frames for domain counts chart
    domain_counts_chart_html = render_plotly_domain_counts(
        domain_counts, 
        total_documents=classified_documents
    )
    if not domain_counts_chart_html and domain_counts:
        domain_counts_b64 = plot_domain_counts_bar(domain_counts[:20])
        if domain_counts_b64:
            domain_counts_chart_html = (
                "<figure class=\"chart\">"
                f"<img src=\"data:image/png;base64,{domain_counts_b64}\" alt=\"Top domains by document count\" />"
                "<figcaption>Top domains ranked by number of classified documents.</figcaption>"
                "</figure>"
            )

    # Frame distribution across top domains (Plotly)
    domain_frame_chart_html = ""
    domain_counts_lookup: Dict[str, int] = {}
    if domain_frame_summaries:
        ordered_domain_frames = [entry for entry in domain_frame_summaries if entry.get("shares")]
        if ordered_domain_frames:
            # Convert to Plotly chart instead of matplotlib
            domain_frame_chart_html = render_domain_frame_distribution(
                ordered_domain_frames,
                frame_lookup,
                color_map,
                export_png_path=(export_dir / "domain_frame_distribution.png") if (export_dir and export_png) else None,
                export_html_path=(export_includes_dir / "domain_frame_distribution.html") if export_includes_dir else None,
                n_min_per_media=n_min_per_media,
                max_domains=domain_mapping_max_domains,
                export_svg=export_svg,
            )
            for entry in ordered_domain_frames:
                domain = entry.get("domain")
                count = entry.get("count", 0)
                if domain:
                    try:
                        domain_counts_lookup[domain] = int(count)
                    except Exception:
                        continue

    rows = []
    for assignment in assignments:
        llm_probs_html = render_probability_bars(assignment.probabilities, frame_lookup, color_map)

        classifier_html = "—"
        if classifier_lookup and assignment.passage_id in classifier_lookup:
            entry = classifier_lookup[assignment.passage_id]
            probs = entry.get("probabilities", {})
            if isinstance(probs, dict) and probs:
                classifier_html = render_probability_bars(
                    {fid: float(score) for fid, score in probs.items()},
                    frame_lookup,
                    color_map,
        )

        # Optionally hide passages with no associated frame.
        # Treat probabilities below a small threshold as effectively zero to avoid rows that render as 0%.
        if hide_empty_passages:
            def _all_below(d: Dict[str, float], thr: float) -> bool:
                try:
                    if not d:
                        return True
                    return max(float(v) for v in d.values()) < thr
                except Exception:
                    return True
            # Use metrics_threshold as the display threshold for emptiness (fallback to 0.0)
            _thr = float(metrics_threshold or 0.0)
            llm_empty = _all_below(assignment.probabilities or {}, _thr)
            clf_empty = True
            if classifier_lookup and assignment.passage_id in classifier_lookup:
                cprobs = classifier_lookup.get(assignment.passage_id, {}).get("probabilities", {})
                if isinstance(cprobs, dict):
                    clf_empty = _all_below({k: float(v) for k, v in cprobs.items()}, _thr)
            if llm_empty and clf_empty:
                continue

        rationale = html.escape(assignment.rationale) if assignment.rationale else "—"
        evidence_text = (
            "<br/>".join(html.escape(span) for span in assignment.evidence_spans)
            if assignment.evidence_spans
            else "—"
        )

        metadata = assignment.metadata if isinstance(assignment.metadata, dict) else {}
        url = metadata.get("url") or ""
        doc_folder_path = metadata.get("doc_folder_path") or ""
        
        url_icon = (
            f"<a class=\"link-icon\" href=\"{html.escape(url)}\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"Open original URL\">🔗</a>"
            if url
            else ""
        )
        folder_icon = (
            f"<a class=\"link-icon\" href=\"file://{html.escape(doc_folder_path)}\" title=\"Open document folder\">📁</a>"
            if doc_folder_path
            else ""
        )
        
        link_icons = f"{url_icon}{folder_icon}"
        passage_html = f"{link_icons}<span class=\"passage-text\">{html.escape(assignment.passage_text)}</span>"

        rows.append(
            "<tr>"
            f"<td class=\"passage\">{passage_html}</td>"
            f"<td>{llm_probs_html}</td>"
            f"<td>{classifier_html}</td>"
            f"<td>{rationale}</td>"
            f"<td>{evidence_text}</td>"
            "</tr>"
        )

    table_html = "\n".join(rows) if rows else "<tr><td colspan=5>No assignments available</td></tr>"

    frame_cards_html = (
        f"<div class=\"frame-card-grid\">{''.join(frame_cards)}</div>"
        if frame_cards
        else "<p class=\"empty-note\">Schema does not define any frames.</p>"
    )

    frames_section_html = f"""
        <section class=\"report-section\" id=\"frames\">
            <header class=\"section-heading\">
                <h2>Frames</h2>
                <p>Schema definition and guidance.</p>
            </header>
            <div class=\"section-body frames-body\">
                {frame_cards_html}
            </div>
        </section>
    """

    # LLM application charts
    llm_coverage_section_html = ""
    llm_binned_section_html = ""
    if assignments:
        frames_as_dicts = [
            {"frame_id": f.frame_id, "name": f.name, "short": f.short_name}
            for f in schema.frames
        ]
        llm_cov_html = render_plotly_llm_coverage(assignments, frames_as_dicts, color_map)
        if llm_cov_html:
            llm_coverage_section_html = f"""
        <section class=\"report-section\" id=\"llm-coverage\">
            <header class=\"section-heading\">
                <h2>LLM Frame Coverage</h2>
                <p>Passages per frame based on LLM-based annotations.</p>
            </header>
            <div class=\"section-body\">
                {llm_cov_html}
                <p class=\"chart-note\">Each bar counts sampled passages where the frame appears in LLM-based annotations.</p>
            </div>
        </section>
        """

        llm_binned_html = render_plotly_llm_binned_distribution(assignments, frames_as_dicts)
        if llm_binned_html:
            llm_binned_section_html = f"""
        <section class=\"report-section\" id=\"llm-bins\">
            <header class=\"section-heading\">
                <h2>LLM Probability Distribution</h2>
                <p>Distribution of LLM probabilities per frame (stacked bins).</p>
            </header>
            <div class=\"section-body\">
                {llm_binned_html}
                <p class=\"chart-note\">Frames on X axis; stacks represent probability bins (0–0.2, 0.2–0.4, …).</p>
            </div>
        </section>
        """

    time_series_section_html = ""
    # Combine Article Volume and Frame share in one section
    time_series_charts = []
    if total_docs_chart_html:
        time_series_charts.append(
            '<div class="chart-item">'
            '<div class="chart-heading">'
            '<div class="chart-title">Article Volume Over Time</div>'
            '<div class="chart-subtitle">Articles per day (30-day avg)</div>'
            '</div>'
            f'{total_docs_chart_html}'
            '</div>'
        )
    if timeseries_chart_html:
        time_series_charts.append(
            '<div class="chart-item">'
            '<h4>Frame Share Over Time (Stacked Area)</h4>'
            '<p class="chart-explanation">Length-weighted share of each frame over time. 30-day rolling average.</p>'
            f'{timeseries_chart_html}'
            '</div>'
        )
    if timeseries_lines_html:
        time_series_charts.append(
            '<div class="chart-item">'
            '<h4>Frame Share Over Time (Lines)</h4>'
            '<p class="chart-explanation">Same length-weighted share metric shown as individual lines. Shows proportional importance of each frame over time. 30-day rolling average.</p>'
            f'{timeseries_lines_html}'
            '</div>'
        )
    
    if time_series_charts:
        time_series_section_html = f"""
        <section class="report-section" id="time-series">
            <header class="section-heading">
                <h2>Time Series</h2>
                <p>Article volume and frame share trends over time.</p>
            </header>
            <div class="section-body">
                <div class="card chart-card">
                    {''.join(time_series_charts)}
                </div>
            </div>
        </section>
        """


    # Build new aggregation charts from all_aggregates
    aggregation_charts_html = ""
    if all_aggregates:
        chart_sections = []
        
        # Global - Weighted - Excluding empty
        global_woz = all_aggregates.get("global_weighted_without_zeros")
        if global_woz:
            print(f"🔍 Global without zeros: type={type(global_woz)}, is_list={isinstance(global_woz, list)}")
            global_woz_html = render_global_bar_chart(
                global_woz,
                frame_lookup, color_map,
                export_png_path=(export_dir / "global_weighted_woz.png") if (export_dir and export_png) else None,
                chart_id="global-weighted-woz",
                export_svg=export_svg,
            )
            if global_woz_html:
                chart_sections.append(
                    '<div class="chart-item">'
                    '<h4>Global - Weighted - Excluding empty</h4>'
                    '<p class="chart-explanation">Frame share across all documents, weighted by content length. Documents with all frame scores zero are excluded.</p>'
                    f'{global_woz_html}'
                    '</div>'
                )
        
        # Global - Weighted - Including empty
        global_wz = all_aggregates.get("global_weighted_with_zeros")
        if global_wz:
            print(f"🔍 Global with zeros: type={type(global_wz)}, is_list={isinstance(global_wz, list)}")
            global_wz_html = render_global_bar_chart(
                global_wz,
                frame_lookup, color_map,
                export_png_path=(export_dir / "global_weighted_wz.png") if (export_dir and export_png) else None,
                chart_id="global-weighted-wz",
                export_svg=export_svg,
            )
            if global_wz_html:
                chart_sections.append(
                    '<div class="chart-item">'
                    '<h4>Global - Weighted - Including empty</h4>'
                    '<p class="chart-explanation">Frame share across all documents, weighted by content length. Documents with all frame scores zero are included.</p>'
                    f'{global_wz_html}'
                    '</div>'
                )
        
        # Global - Occurrence - Excluding empty
        global_occ_woz = all_aggregates.get("global_occurrence_without_zeros")
        if global_occ_woz:
            global_occ_woz_html = render_global_bar_chart(
                global_occ_woz,
                frame_lookup, color_map,
                export_png_path=(export_dir / "global_occurrence_woz.png") if (export_dir and export_png) else None,
                chart_id="global-occurrence-woz",
                export_svg=export_svg,
            )
            if global_occ_woz_html:
                chart_sections.append(
                    '<div class="chart-item">'
                    '<h4>Global - Occurrence - Excluding empty</h4>'
                    '<p class="chart-explanation">Share of articles mentioning each frame. Each article counts equally regardless of size. Documents with all frame scores zero are excluded.</p>'
                    f'{global_occ_woz_html}'
                    '</div>'
                )
        
        # Global - Occurrence - Including empty
        global_occ_wz = all_aggregates.get("global_occurrence_with_zeros")
        if global_occ_wz:
            global_occ_wz_html = render_global_bar_chart(
                global_occ_wz,
                frame_lookup, color_map,
                export_png_path=(export_dir / "global_occurrence_wz.png") if (export_dir and export_png) else None,
                chart_id="global-occurrence-wz",
                export_svg=export_svg,
            )
            if global_occ_wz_html:
                chart_sections.append(
                    '<div class="chart-item">'
                    '<h4>Global - Occurrence - Including empty</h4>'
                    '<p class="chart-explanation">Share of articles mentioning each frame. Each article counts equally regardless of size. Documents with all frame scores zero are included.</p>'
                    f'{global_occ_wz_html}'
                    '</div>'
                )
        
        # Yearly - Weighted - Excluding empty
        if include_yearly_bar_charts:
            yearly_woz = all_aggregates.get("year_weighted_without_zeros")
            if yearly_woz:
                yearly_woz_html = render_yearly_bar_chart(
                    yearly_woz,
                    frame_lookup, color_map,
                    export_png_path=(export_dir / "yearly_weighted_woz.png") if (export_dir and export_png) else None,
                    chart_id="yearly-weighted-woz",
                    export_svg=export_svg,
                )
                if yearly_woz_html:
                    chart_sections.append(
                        '<div class="chart-item">'
                        '<h4>Yearly - Weighted - Excluding empty</h4>'
                        '<p class="chart-explanation">Frame share by year, weighted by content length. Documents with all frame scores zero are excluded.</p>'
                        f'{yearly_woz_html}'
                        '</div>'
                    )
            
            # Yearly - Weighted - Including empty
            yearly_wz = all_aggregates.get("year_weighted_with_zeros")
            if yearly_wz:
                yearly_wz_html = render_yearly_bar_chart(
                    yearly_wz,
                    frame_lookup, color_map,
                    export_png_path=(export_dir / "yearly_weighted_wz.png") if (export_dir and export_png) else None,
                    chart_id="yearly-weighted-wz",
                    export_svg=export_svg,
                )
                if yearly_wz_html:
                    chart_sections.append(
                        '<div class="chart-item">'
                        '<h4>Yearly - Weighted - Including empty</h4>'
                        '<p class="chart-explanation">Frame share by year, weighted by content length. Documents with all frame scores zero are included.</p>'
                        f'{yearly_wz_html}'
                        '</div>'
                    )
            
            # Yearly - Occurrence - Excluding empty
            yearly_occ_woz = all_aggregates.get("year_occurrence_without_zeros")
            if yearly_occ_woz:
                yearly_occ_woz_html = render_yearly_bar_chart(
                    yearly_occ_woz,
                    frame_lookup, color_map,
                    export_png_path=(export_dir / "yearly_occurrence_woz.png") if (export_dir and export_png) else None,
                    chart_id="yearly-occurrence-woz",
                    export_svg=export_svg,
                )
                if yearly_occ_woz_html:
                    chart_sections.append(
                        '<div class="chart-item">'
                        '<h4>Yearly - Occurrence - Excluding empty</h4>'
                        '<p class="chart-explanation">Share of articles mentioning each frame by year. Each article counts equally. Documents with all frame scores zero are excluded.</p>'
                        f'{yearly_occ_woz_html}'
                        '</div>'
                    )
            
            # Yearly - Occurrence - Including empty
            yearly_occ_wz = all_aggregates.get("year_occurrence_with_zeros")
            if yearly_occ_wz:
                yearly_occ_wz_html = render_yearly_bar_chart(
                    yearly_occ_wz,
                    frame_lookup, color_map,
                    export_png_path=(export_dir / "yearly_occurrence_wz.png") if (export_dir and export_png) else None,
                    chart_id="yearly-occurrence-wz",
                    export_svg=export_svg,
                )
                if yearly_occ_wz_html:
                    chart_sections.append(
                        '<div class="chart-item">'
                        '<h4>Yearly - Occurrence - Including empty</h4>'
                        '<p class="chart-explanation">Share of articles mentioning each frame by year. Each article counts equally. Documents with all frame scores zero are included.</p>'
                        f'{yearly_occ_wz_html}'
                        '</div>'
                    )
        
        if chart_sections:
            aggregation_charts_html = f"""
        <section class="report-section" id="aggregation-charts">
            <header class="section-heading">
                <h2>Aggregation Analysis</h2>
                <p>Comprehensive view of frame distribution across different dimensions.</p>
            </header>
            <div class="section-body">
                <div class="card chart-card">
                    {''.join(chart_sections)}
                </div>
            </div>
        </section>
        """
    
    # Per Corpus section (grouped bars by corpus per frame)
    # Only include if there are multiple corpora
    corpus_section_html = ""
    if all_aggregates:
        # Check how many unique corpora exist in the data
        def count_unique_corpora(corpus_data: Optional[object]) -> int:
            """Count unique corpora from corpus aggregate data."""
            if not isinstance(corpus_data, list) or not corpus_data:
                return 0
            corpora = set()
            for entry in corpus_data:
                if isinstance(entry, dict):
                    corpus_name = entry.get('corpus')
                    if corpus_name:
                        corpora.add(str(corpus_name))
            return len(corpora)
        
        # Check any corpus aggregate to determine number of corpora
        corpus_weighted_woz = all_aggregates.get("corpus_weighted_without_zeros")
        corpus_weighted_wz = all_aggregates.get("corpus_weighted_with_zeros")
        corpus_occ_woz = all_aggregates.get("corpus_occurrence_without_zeros")
        corpus_occ_wz = all_aggregates.get("corpus_occurrence_with_zeros")
        
        # Find the first available corpus data to count corpora
        corpus_data_for_count = corpus_weighted_woz or corpus_weighted_wz or corpus_occ_woz or corpus_occ_wz
        num_corpora = count_unique_corpora(corpus_data_for_count)
        
        # Only generate per corpus section if there are multiple corpora
        if num_corpora > 1:
            corpus_sections: List[str] = []
            if corpus_weighted_woz:
                chart_html = render_corpus_bar_chart(
                    corpus_weighted_woz, frame_lookup,
                    corpus_aliases=corpus_aliases,
                    export_png_path=(export_dir / "corpus_weighted_woz.png") if (export_dir and export_png) else None,
                    chart_id="corpus-weighted-woz",
                    export_svg=export_svg,
                )
                if chart_html:
                    corpus_sections.append(
                        '<div class="chart-item">'
                        '<h4>Per Corpus - Weighted - Excluding empty</h4>'
                        '<p class="chart-explanation">Frame share by corpus, weighted by content length. Documents with all frame scores zero are excluded.</p>'
                        f'{chart_html}'
                        '</div>'
                    )
            if corpus_weighted_wz:
                chart_html = render_corpus_bar_chart(
                    corpus_weighted_wz, frame_lookup,
                    corpus_aliases=corpus_aliases,
                    export_png_path=(export_dir / "corpus_weighted_wz.png") if (export_dir and export_png) else None,
                    chart_id="corpus-weighted-wz",
                    export_svg=export_svg,
                )
                if chart_html:
                    corpus_sections.append(
                        '<div class="chart-item">'
                        '<h4>Per Corpus - Weighted - Including empty</h4>'
                        '<p class="chart-explanation">Frame share by corpus, weighted by content length. Documents with all frame scores zero are included.</p>'
                        f'{chart_html}'
                        '</div>'
                    )
            if corpus_occ_woz:
                chart_html = render_corpus_bar_chart(
                    corpus_occ_woz, frame_lookup,
                    corpus_aliases=corpus_aliases,
                    export_png_path=(export_dir / "corpus_occurrence_woz.png") if (export_dir and export_png) else None,
                    chart_id="corpus-occurrence-woz",
                    export_svg=export_svg,
                )
                if chart_html:
                    corpus_sections.append(
                        '<div class="chart-item">'
                        '<h4>Per Corpus - Occurrence - Excluding empty</h4>'
                        '<p class="chart-explanation">Share of articles mentioning each frame by corpus. Each article counts equally. Documents with all frame scores zero are excluded.</p>'
                        f'{chart_html}'
                        '</div>'
                    )
            if corpus_occ_wz:
                chart_html = render_corpus_bar_chart(
                    corpus_occ_wz, frame_lookup,
                    corpus_aliases=corpus_aliases,
                    export_png_path=(export_dir / "corpus_occurrence_wz.png") if (export_dir and export_png) else None,
                    chart_id="corpus-occurrence-wz",
                    export_svg=export_svg,
                )
                if chart_html:
                    corpus_sections.append(
                        '<div class="chart-item">'
                        '<h4>Per Corpus - Occurrence - Including empty</h4>'
                        '<p class="chart-explanation">Share of articles mentioning each frame by corpus. Each article counts equally. Documents with all frame scores zero are included.</p>'
                        f'{chart_html}'
                        '</div>'
                    )
            if corpus_sections:
                corpus_section_html = f"""
        <section class=\"report-section\" id=\"per-corpus\">
            <header class=\"section-heading\">
                <h2>Per Corpus</h2>
                <p>Frame distribution by corpus. Colors distinguish corpora; X axis lists frames.</p>
            </header>
            <div class=\"section-body\">
                <div class=\"card chart-card\">
                    {''.join(corpus_sections)}
                </div>
            </div>
        </section>
        """

    # Domain Analysis section
    domain_analysis_html = ""
    domain_charts = []
    if domain_counts_chart_html:
        domain_charts.append(
            '<div class="chart-item">'
            '<h4>Number of Articles by Domain</h4>'
            f'{domain_counts_chart_html}'
            f'<p class="chart-note">Based on {classified_documents:,} classified articles.</p>'
            '</div>'
        )
    if domain_frame_chart_html:
        domain_charts.append(
            '<div class="chart-item">'
            '<h4>Frame Distribution Across Top Domains</h4>'
            f'{domain_frame_chart_html}'
            '<p class="chart-note">Frame share by media source, weighted by content length.</p>'
            '</div>'
        )
    
    if domain_charts:
        domain_analysis_html = f"""
        <section class="report-section" id="domain-analysis">
            <header class="section-heading">
                <h2>Domain Analysis</h2>
                <p>Distribution of articles and frames across media sources.</p>
            </header>
            <div class="section-body">
                <div class="card chart-card">
                    {''.join(domain_charts)}
                </div>
            </div>
        </section>
        """

    # Domain yearly charts
    domain_yearly_section_html = ""
    domain_yearly_limit = max(0, domain_yearly_top_domains if domain_yearly_top_domains is not None else 0)

    if include_domain_yearly_bar_charts:
        # Select which aggregates to use based on version
        document_aggregates_to_use = (
            document_aggregates_weighted if domain_yearly_version == "weighted" 
            else document_aggregates_occurrence
        )
        
        if not document_aggregates_to_use:
            # Fallback if requested version is not available
            document_aggregates_to_use = document_aggregates_weighted or document_aggregates_occurrence
        
        if document_aggregates_to_use:
            # Build lookup of domain -> document aggregates
            domain_documents: Dict[str, List[DocumentFrameAggregate]] = {}
            for agg in document_aggregates_to_use:
                domain = getattr(agg, "domain", None)
                if not domain and getattr(agg, "url", None):
                    extracted = extract_domain_from_url(agg.url)
                    if extracted:
                        try:
                            object.__setattr__(agg, "domain", extracted)
                            domain = extracted
                        except Exception:
                            domain = extracted
                if domain:
                    domain_documents.setdefault(domain, []).append(agg)

            if domain_documents:
                # Rank domains by article count (number of documents/articles per domain)
                # Use the actual document count from domain_documents, which is based on the aggregates being used
                ranked_domain_counts = sorted(
                    [(dom, len(docs)) for dom, docs in domain_documents.items()],
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                top_domain_names: List[str] = []
                for domain_name, article_count in ranked_domain_counts:
                    top_domain_names.append(domain_name)
                    if domain_yearly_limit and len(top_domain_names) >= domain_yearly_limit:
                        break

                charts: List[str] = []
                for domain_name in top_domain_names:
                    domain_docs = domain_documents.get(domain_name, [])
                    if not domain_docs:
                        continue
                    yearly_agg = TemporalAggregator(
                        period="year",
                        weight_by_document_weight=(domain_yearly_version == "weighted"),
                        keep_documents_with_no_frames=domain_yearly_include_empty,
                    ).aggregate(domain_docs)
                    if not yearly_agg:
                        continue
                    slug = _slugify(domain_name)
                    chart_html = render_yearly_bar_chart(
                        yearly_agg,
                        frame_lookup,
                        color_map,
                        export_png_path=(export_dir / f"domain_yearly_{slug}.png") if (export_dir and export_png) else None,
                        chart_id=f"domain-yearly-{slug}",
                        export_svg=export_svg,
                    )
                    if chart_html:
                        note = f"Year-over-year frame share for {domain_name} (n={domain_counts_lookup.get(domain_name, len(domain_docs))} articles)."
                        charts.append(
                            '<div class="chart-item">'
                            f'<h4>{html.escape(domain_name)} — Yearly Frame Share</h4>'
                            f'<p class="chart-explanation">{html.escape(note)}</p>'
                            f'{chart_html}'
                            '</div>'
                        )

                if charts:
                    domain_yearly_section_html = f"""
        <section class="report-section" id="domain-yearly">
            <header class="section-heading">
                <h2>Yearly Trends by Domain</h2>
                <p>Top domains ranked by document volume with yearly frame distributions.</p>
            </header>
            <div class="section-body">
                <div class="card chart-card">
                    {''.join(charts)}
                </div>
            </div>
        </section>
        """

    # Custom section with user-selected plots
    custom_section_html = ""
    if custom_plots and all_aggregates:
        custom_charts = []
        for custom_plot in custom_plots:
            plot_type = getattr(custom_plot, "type", None) if hasattr(custom_plot, "type") else custom_plot.get("type") if isinstance(custom_plot, dict) else None
            if not plot_type:
                continue
            
            # Get the aggregate data for this plot type
            aggregate_data = all_aggregates.get(plot_type)
            if not aggregate_data:
                continue
            
            # Render chart based on type
            chart_html = ""
            if plot_type.startswith("global_"):
                chart_html = render_global_bar_chart(
                    aggregate_data,
                    frame_lookup,
                    color_map,
                    export_png_path=(export_dir / f"{plot_type}.png") if (export_dir and export_png) else None,
                    chart_id=f"custom-{plot_type}",
                    export_svg=export_svg,
                )
            elif plot_type.startswith("year_"):
                chart_html = render_yearly_bar_chart(
                    aggregate_data,
                    frame_lookup,
                    color_map,
                    export_png_path=(export_dir / f"{plot_type}.png") if (export_dir and export_png) else None,
                    chart_id=f"custom-{plot_type}",
                    export_svg=export_svg,
                )
            
            if chart_html:
                plot_title = getattr(custom_plot, "title", None) if hasattr(custom_plot, "title") else custom_plot.get("title") if isinstance(custom_plot, dict) else None
                plot_subtitle = getattr(custom_plot, "subtitle", None) if hasattr(custom_plot, "subtitle") else custom_plot.get("subtitle") if isinstance(custom_plot, dict) else None
                plot_caption = getattr(custom_plot, "caption", None) if hasattr(custom_plot, "caption") else custom_plot.get("caption") if isinstance(custom_plot, dict) else None
                
                # Format subtitle and caption with n_articles if needed
                if plot_subtitle:
                    try:
                        plot_subtitle = plot_subtitle.format(n_articles=classified_documents)
                    except Exception:
                        pass
                if plot_caption:
                    try:
                        plot_caption = plot_caption.format(n_articles=classified_documents)
                    except Exception:
                        pass
                
                heading_html = ""
                if plot_title:
                    heading_html = f'<div class="chart-heading"><div class="chart-title">{html.escape(plot_title)}</div>'
                    if plot_subtitle:
                        heading_html += f'<div class="chart-subtitle">{html.escape(plot_subtitle)}</div>'
                    heading_html += "</div>"
                
                caption_html = f'<p class="chart-note">{plot_caption}</p>' if plot_caption else ""
                
                custom_charts.append(
                    '<div class="chart-item">'
                    + heading_html
                    + chart_html
                    + caption_html
                    + '</div>'
                )
        
        if custom_charts:
            custom_section_html = f"""
        <section class="report-section" id="custom-plots">
            <header class="section-heading">
                <h2>Custom Analysis</h2>
                <p>Selected frame distribution visualizations.</p>
            </header>
            <div class="section-body">
                <div class="card chart-card">
                    {''.join(custom_charts)}
                </div>
            </div>
        </section>
        """

    # Build corpus index to look up titles
    corpus_index: Dict[str, dict] = {}
    if output_path:
        from efi_analyser.frames.plotting._utils import load_corpus_index
        corpus_index = load_corpus_index(output_path.parent)
    
    top_stories_by_frame = collect_top_stories_by_frame(
        document_aggregates_weighted,
        corpus_index=corpus_index,
    )
    story_cards: List[str] = []
    for frame in schema.frames:
        stories = top_stories_by_frame.get(frame.frame_id)
        if not stories:
            continue
        color = color_map.get(frame.frame_id, "#1E3D58")
        items: List[str] = []
        for idx, story in enumerate(stories, start=1):
            title_value = str(story.get("title") or f"Story {idx}")
            story_title = html.escape(title_value)
            url_value = str(story.get("url") or "").strip()
            title_html = (
                f"<a href=\"{html.escape(url_value)}\" target=\"_blank\" rel=\"noopener noreferrer\">{story_title}</a>"
                if url_value
                else story_title
            )
            meta_parts: List[str] = []
            domain_label = str(story.get("domain") or "").strip()
            if domain_label:
                meta_parts.append(html.escape(domain_label))
            date_label = format_date_label(str(story.get("published_at") or ""))
            if date_label:
                meta_parts.append(html.escape(date_label))
            meta_html = f"<div class=\"story-meta\">{' • '.join(meta_parts)}</div>" if meta_parts else ""
            score_pct = f"{float(story.get('score', 0.0)) * 100:.0f}%"
            doc_id = str(story.get('doc_id', ''))
            classification_link = f"classifications/{html.escape(doc_id)}.json" if doc_id else ""
            classification_html = (
                f"<div class=\"story-links\">"
                f"<a href=\"{classification_link}\" target=\"_blank\" rel=\"noopener noreferrer\" class=\"classification-link\">"
                f"View classification details</a>"
                f"</div>"
                if classification_link
                else ""
            )
            items.append(
                "<li>"
                f"<div class=\"story-rank\">{idx}</div>"
                "<div class=\"story-content\">"
                f"<div class=\"story-title\">{title_html}</div>"
                f"{meta_html}"
                f"<div class=\"story-score\">{score_pct} frame weight</div>"
                f"{classification_html}"
                "</div>"
                "</li>"
            )
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        story_cards.append(
            "<article class=\"story-card\" style=\"--accent-color:" + color + ";\">"
            f"<header><h3>{html.escape(short_label)}</h3><p>{html.escape(frame.name)}</p></header>"
            f"<ol class=\"story-list\">{''.join(items)}</ol>"
            "</article>"
        )

    top_stories_section_html = ""
    if story_cards:
        top_stories_section_html = f"""
        <section class=\"report-section\" id=\"top-stories\">
            <header class=\"section-heading\">
                <h2>Top Stories per Frame</h2>
                <p>Leading documents aligned to each frame based on length-weighted shares.</p>
            </header>
            <div class=\"section-body story-grid\">
                {''.join(story_cards)}
            </div>
        </section>
        """

    applications_block = f"""
    <div class=\"developer-block\">
        <h3>Applications</h3>
        <div class=\"card table-card\">
            <div class=\"table-wrapper\">
                <table>
                    <thead>
                        <tr>
                            <th>Passage Text<div class=\"resizer\"></div></th>
                            <th>LLM Probabilities<div class=\"resizer\"></div></th>
                            <th>Classifier Probabilities<div class=\"resizer\"></div></th>
                            <th>Rationale<div class=\"resizer\"></div></th>
                            <th>Evidence<div class=\"resizer\"></div></th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """

    developer_components: List[str] = []
    if classifier_plots_html:
        developer_components.append(f"<div class=\"developer-block\">{classifier_plots_html}</div>")
    developer_components.append(applications_block)

    developer_section_html = ""
    if developer_components:
        developer_section_html = f"""
        <section class=\"report-section developer\" id=\"developer\">
            <header class=\"section-heading\">
                <h2>Developer</h2>
                <p>Diagnostics and passage-level review for future iteration.</p>
            </header>
            {''.join(developer_components)}
        </section>
        """

    # Format plot subtitle with {n_articles} placeholder if present
    formatted_plot_subtitle = plot_subtitle.format(n_articles=classified_documents) if plot_subtitle else None
    
    # Format plot note with {n_articles} placeholder if present
    formatted_plot_note = plot_note.format(n_articles=classified_documents) if plot_note else None
    
    # Coverage text for report header
    coverage_text = f"Analysis of {classified_documents:,} classified documents"
    
    stats = usage_stats or {}
    def _fmt(number: Optional[int]) -> Optional[str]:
        if number is None:
            return None
        return f"{int(number):,}"

    def _metric_card(title: str, lines: Sequence[Tuple[Optional[int], str]], accent: str = "var(--accent-2)") -> str:
        valid_lines = [(value, label) for value, label in lines if value is not None]
        if not valid_lines:
            return ""
        line_html = "".join(
            f"<div class=\"metric-line\"><span class=\"metric-number\">{_fmt(value)}</span><span class=\"metric-unit\">{html.escape(label)}</span></div>"
            for value, label in valid_lines
            if _fmt(value) is not None
        )
        if not line_html:
            return ""
        return (
            f"<div class=\"metric-card\" style=\"--metric-accent:{accent};\">"
            f"<div class=\"metric-title\">{html.escape(title)}</div>"
            f"{line_html}"
            "</div>"
        )

    metric_cards: List[str] = []
    metric_cards.append(
        _metric_card(
            "Corpus",
            [
                (stats.get("corpus_documents"), "articles"),
            ],
            accent="var(--accent-1)"
        )
    )
    metric_cards.append(
        _metric_card(
            "Induction Sample",
            [
                (stats.get("induction_passages"), "passages"),
            ],
            accent="var(--accent-3)"
        )
    )
    metric_cards.append(
        _metric_card(
            "Annotation (LLM)",
            [
                (stats.get("annotation_passages"), "passages"),
            ],
            accent="var(--accent-2)"
        )
    )
    metric_cards.append(
        _metric_card(
            "Classifier",
            [
                (stats.get("classifier_documents"), "documents"),
                (stats.get("classifier_passages"), "passages"),
            ],
            accent="var(--accent-4)"
        )
    )
    metric_cards.append(
        _metric_card(
            "Frames",
            [
                (stats.get("frames", len(schema.frames)), "total"),
            ],
            accent="var(--success)"
        )
    )
    metric_cards = [card for card in metric_cards if card]
    header_metrics = ""
    if metric_cards:
        header_metrics = f"<div class=\"header-metrics\">{''.join(metric_cards)}</div>"
    
    header_metrics = (
        header_metrics
    )
    timeline_html = f"<p class=\"timeline-note\">{html.escape(timeseries_note)}</p>" if timeseries_note else ""
    header_html = f"""
    <header class=\"report-header\">
        <div class=\"heading-text\">
            <span class=\"eyebrow\">Narrative Framing Analysis</span>
            <h1>{html.escape(schema.domain)}</h1>
            <p>{html.escape(coverage_text)}</p>
            {timeline_html}
        </div>
        {header_metrics}
    </header>
    """

    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(schema.domain)} — Narrative Framing Analysis</title>
  <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap');
    :root {{
      --ink-900: #0f172a;
      --ink-800: #1e293b;
      --ink-600: #334155;
      --ink-500: #475569;
      --ink-300: #94a3b8;
      --slate-50: #f7f9fc;
      --slate-100: #eef2f9;
      --border: #d3dce7;
      --accent-1: #1e3d58;
      --accent-2: #057d9f;
      --accent-3: #f18f01;
      --accent-4: #6c63ff;
      --success: #3a7d44;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 48px;
      background: linear-gradient(140deg, var(--slate-50) 0%, #e4edf7 100%);
      font-family: 'Inter', 'Segoe UI', sans-serif;
      color: var(--ink-800);
    }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .report-page {{
      max-width: 1320px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 18px;
      box-shadow: 0 28px 68px rgba(15, 23, 42, 0.12);
      padding: 48px 64px 72px;
    }}
    .report-header {{
      display: flex;
      justify-content: space-between;
      gap: 32px;
      padding: 36px 40px;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent-1) 0%, #0e7c7b 55%, var(--accent-4) 100%);
      color: #ffffff;
      margin-bottom: 48px;
    }}
    .heading-text h1 {{
      margin: 8px 0 12px;
      font-size: 2.25rem;
      letter-spacing: -0.015em;
    }}
    .heading-text .subtitle {{
      margin: 0 0 12px 0;
      font-size: 1.125rem;
      color: var(--ink-600);
      font-weight: 400;
    }}
    .heading-text .note {{
      margin: 16px 0 0 0;
      font-size: 0.875rem;
      color: var(--ink-500);
      font-weight: 400;
      line-height: 1.6;
      white-space: pre-wrap;
    }}
    .heading-text p {{ margin: 6px 0 0 0; font-size: 1rem; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.75rem;
      opacity: 0.7;
    }}
    .timeline-note {{
      margin-top: 10px;
      font-size: 0.95rem;
      opacity: 0.85;
    }}
    .header-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 18px;
      align-items: stretch;
    }}
    .metric-card {{
      position: relative;
      padding: 20px 20px 20px 28px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 12px;
      background: #ffffff;
      box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 0;
    }}
    .metric-card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 10px;
      height: 42px;
      background: var(--metric-accent, var(--accent-2));
      border-bottom-right-radius: 8px;
    }}
    .metric-title {{
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ink-500);
      font-weight: 600;
      margin-bottom: 4px;
      line-height: 1.3;
    }}
    .metric-line {{
      display: flex;
      align-items: baseline;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .metric-number {{
      font-size: 1.45rem;
      font-weight: 600;
      color: var(--ink-900);
      line-height: 1.2;
      white-space: nowrap;
    }}
    .metric-unit {{
      font-size: 0.85rem;
      color: var(--ink-500);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      white-space: nowrap;
    }}
    .report-section {{ margin-bottom: 52px; }}
    .report-section:last-of-type {{ margin-bottom: 0; }}
    .section-heading h2 {{
      margin: 0;
      font-size: 1.7rem;
      color: var(--ink-900);
    }}
    .section-heading p {{
      margin: 6px 0 0 0;
      color: var(--ink-500);
      font-size: 1rem;
    }}
    .section-body {{ margin-top: 26px; }}
    .section-note {{
      margin: 0 0 18px 0;
      font-size: 0.95rem;
      color: var(--ink-500);
    }}
    .frames-body {{
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .frame-card-grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .card {{
      background: linear-gradient(140deg, #ffffff 0%, #f8fbff 100%);
      border-radius: 16px;
      border: 1px solid var(--border);
      padding: 22px 24px;
      box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .card h4 {{
      margin: 0;
      font-size: 1rem;
      color: var(--ink-600);
    }}
    .chart-card {{
      position: relative;
      min-height: fit-content;
      height: auto;
      overflow: visible;
      border-radius: 0;
      background: #ffffff;
      border: 1px solid rgba(148, 163, 184, 0.35);
      box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
      padding-top: 26px;
    }}
    .chart-card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 14px;
      height: 46px;
      background: var(--accent-2);
      border-bottom-right-radius: 10px;
    }}
    .frame-card {{
      position: relative;
      border-radius: 18px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 26px 24px;
      background: #ffffff;
      box-shadow: 0 20px 44px rgba(15, 23, 42, 0.1);
      overflow: hidden;
    }}
    .frame-card .share-badge {{
      position: absolute;
      top: 16px;
      right: 18px;
      background: var(--accent-color, var(--accent-2));
      color: #ffffff;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    .frame-card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(30, 61, 88, 0.08), rgba(5, 125, 159, 0.06));
      opacity: 0.8;
      pointer-events: none;
    }}
    .frame-card::after {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 6px;
      background: var(--accent-color, var(--accent-1));
      opacity: 0.9;
    }}
    .frame-card h3 {{
      position: relative;
      margin: 0;
      font-size: 1.15rem;
      color: var(--accent-1);
      font-weight: 600;
    }}
    .frame-card-title {{
      position: relative;
      margin: 6px 0 14px;
      font-size: 0.95rem;
      color: var(--ink-600);
    }}
    .frame-card-text {{
      position: relative;
      margin: 0 0 12px 0;
      color: var(--ink-600);
      line-height: 1.5;
      font-size: 0.9rem;
    }}
    .frame-card-meta {{
      position: relative;
      margin: 0 0 8px 0;
      font-size: 0.9rem;
      color: var(--ink-500);
    }}
    .frame-card-meta strong {{ color: var(--ink-600); font-weight: 600; margin-right: 6px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: auto;
      font-size: 0.92rem;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: var(--slate-100);
      z-index: 2;
      text-align: left;
      font-weight: 600;
      color: var(--ink-600);
      padding: 10px 12px;
    }}
    th, td {{
      border: 1px solid rgba(148, 163, 184, 0.35);
      padding: 10px 12px;
      vertical-align: top;
    }}
    td.passage {{ white-space: pre-wrap; min-width: 280px; width: 30%; max-width: 640px; word-break: break-word; }}
    .summary-table th, .summary-table td {{ border-color: rgba(148, 163, 184, 0.4); }}
    .chart-grid {{
      display: grid;
      gap: 22px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      margin-bottom: 24px;
    }}
    .chart {{
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .chart img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: #ffffff;
    }}
    .chart figcaption {{ font-size: 0.85rem; color: var(--ink-500); }}
    .media-body {{
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .chart-note {{
      margin: 12px 0 0 0;
      font-size: 0.78rem;
      color: var(--ink-400);
      font-weight: 400;
    }}
    .chart-item {{
      margin: 24px 0;
    }}
    .chart-item:first-child {{
      margin-top: 0;
    }}
    .chart-explanation {{
      margin: 0 0 16px 0;
      padding: 12px 16px;
      background: var(--slate-100);
      border-left: 3px solid var(--accent-2);
      border-radius: 0 6px 6px 0;
      font-size: 0.9rem;
      color: var(--ink-600);
      line-height: 1.5;
    }}
    .chart-explanation strong {{
      color: var(--ink-800);
    }}
    .plotly-chart {{
      width: 100%;
      min-height: 500px;
      flex: 1;
    }}
    .chart-heading {{
      margin: 0 0 8px 0;
    }}
    .chart-title {{
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--ink-800);
      line-height: 1.15;
      margin: 0;
    }}
    .chart-subtitle {{
      margin: 2px 0 0 0;
      font-size: 1rem;
      font-weight: 300;
      color: var(--ink-600);
      line-height: 1.4;
    }}
    .story-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .story-card {{
      border-radius: 18px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      padding: 22px 22px 18px;
      background: #ffffff;
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
      position: relative;
    }}
    .story-card::after {{
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      background: var(--accent-color, var(--accent-2));
      border-radius: 10px 10px 0 0;
    }}
    .story-card header {{ margin-bottom: 14px; }}
    .story-card h3 {{
      margin: 0;
      font-size: 1.05rem;
      color: var(--accent-color, var(--accent-2));
    }}
    .story-card header p {{
      margin: 4px 0 0 0;
      color: var(--ink-500);
      font-size: 0.9rem;
    }}
    .story-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .story-list li {{
      display: grid;
      grid-template-columns: 32px 1fr;
      gap: 12px;
      align-items: start;
    }}
    .story-rank {{
      font-weight: 600;
      font-size: 1.1rem;
      color: var(--accent-color, var(--accent-2));
      text-align: center;
      padding-top: 2px;
    }}
    .story-title {{ font-weight: 600; color: var(--ink-800); font-size: 0.95rem; }}
    .story-meta {{ font-size: 0.85rem; color: var(--ink-500); margin: 6px 0; }}
    .story-score {{ font-size: 0.85rem; color: var(--accent-color, var(--accent-2)); font-weight: 600; }}
    .story-links {{ margin-top: 8px; }}
    .classification-link {{
      font-size: 0.8rem;
      color: var(--ink-600);
      text-decoration: none;
      padding: 4px 8px;
      border: 1px solid var(--ink-300);
      border-radius: 4px;
      background: var(--ink-50);
      display: inline-block;
      transition: all 0.2s ease;
    }}
    .classification-link:hover {{
      background: var(--accent-color, var(--accent-2));
      color: white;
      border-color: var(--accent-color, var(--accent-2));
    }}
    .developer-block {{ margin-top: 24px; }}
    .developer-block:first-of-type {{ margin-top: 0; }}
    .developer-block h3 {{
      margin: 0 0 18px 0;
      font-size: 1.25rem;
      color: var(--ink-900);
    }}
    .metrics-table table {{ margin-top: 12px; }}
    .table-card {{ padding: 0; overflow: hidden; }}
    .table-wrapper {{ overflow: auto; }}
    .bar {{
      position: relative;
      background: #f0f4f9;
      margin-bottom: 6px;
      height: 24px;
      border-radius: 4px;
      overflow: hidden;
    }}
    .fill {{
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      opacity: 0.9;
    }}
    .bar-label {{
      position: relative;
      z-index: 1;
      padding-left: 6px;
      line-height: 24px;
      font-size: 0.88rem;
      color: var(--ink-800);
      font-weight: 600;
    }}
    .resizer {{
      position: absolute;
      right: 0;
      top: 0;
      width: 6px;
      cursor: col-resize;
      user-select: none;
      height: 100%;
    }}
    .link-icon {{
      margin-right: 6px;
      text-decoration: none;
      font-size: 0.95rem;
    }}
    .link-icon:hover {{ text-decoration: underline; }}
    .passage-text {{ white-space: pre-wrap; }}
    .empty-note {{ font-size: 0.95rem; color: var(--ink-500); }}
    .error-note {{ color: #d32f2f; margin: 0; }}
    @media (max-width: 860px) {{
      body {{ padding: 24px; }}
      .report-page {{ padding: 32px 24px 48px; }}
      .report-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .header-metrics {{
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"report-page\">
    {header_html}
    {frames_section_html}
    {llm_coverage_section_html}
    {llm_binned_section_html}
    {time_series_section_html}
    {corpus_section_html}
    {aggregation_charts_html}
    {domain_analysis_html}
    {domain_yearly_section_html}
    {custom_section_html}
    {top_stories_section_html}
    {developer_section_html}
  </div>
  <script>
    (function() {{
      const table = document.querySelector('.table-wrapper table');
      if (!table) return;
      const setColumnWidth = (index, width) => {{
        const cells = table.querySelectorAll(`tr > *:nth-child(${{index + 1}})`);
        cells.forEach((cell) => {{ cell.style.width = width + 'px'; }});
      }};
      table.querySelectorAll('thead th').forEach((th, index) => {{
        const resizer = th.querySelector('.resizer');
        if (!resizer) return;
        let startX = 0;
        let startWidth = 0;
        const onMouseMove = (event) => {{
          const delta = event.pageX - startX;
          const newWidth = Math.max(160, startWidth + delta);
          setColumnWidth(index, newWidth);
        }};
        const onMouseUp = () => {{
          document.removeEventListener('mousemove', onMouseMove);
          document.removeEventListener('mouseup', onMouseUp);
        }};
        resizer.addEventListener('mousedown', (event) => {{
          startX = event.pageX;
          startWidth = th.offsetWidth;
          document.addEventListener('mousemove', onMouseMove);
          document.addEventListener('mouseup', onMouseUp);
        }});
      }});
    }})();
  </script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")


__all__ = ["write_html_report", "generate_report_from_state"]
