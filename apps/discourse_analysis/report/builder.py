"""HTML report builder for discourse analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import html
from datetime import datetime
import json

from efi_analyser.frames.framer import load_schema
from efi_analyser.frames.types import FrameSchema, FrameAssignments
from efi_analyser.frames.classifier import DocumentClassifications, FrameClassifierArtifacts
from efi_analyser.ner import NERResult

from apps.discourse_analysis.analysis.types import DiscourseAggregates
from apps.discourse_analysis.analysis.plots import (
    build_color_map,
    plot_global_distribution,
    plot_by_year,
    plot_by_domain,
    plot_co_occurrence_matrix,
    plot_by_domain_year,
    plot_by_corpus,
    plot_document_count_by_domain,
)
from apps.discourse_analysis.analysis.aggregator import build_aggregates, DocumentFilter
from efi_analyser.stance.types import StanceAssignments


@dataclass
class ReportContext:
    schema: Optional[FrameSchema]
    aggregates: DiscourseAggregates
    frame_classifications: DocumentClassifications
    stance_classifications: StanceAssignments
    frame_assignments: FrameAssignments
    frame_classifier_predictions: List[Dict[str, object]]
    ner_result: Optional[NERResult]
    title: str
    subtitle: Optional[str]


class ReportBuilder:
    """Build a standalone HTML report for discourse analysis."""

    def __init__(self, state: object, config: object, paths: object) -> None:
        self.state = state
        self.config = config
        self.paths = paths

    def build(self) -> Path:
        schema = self._resolve_schema()
        aggregates = self._resolve_aggregates()
        frame_classifications = self._resolve_frame_classifications()
        stance_classifications = self._resolve_stance_classifications()
        frame_assignments = self._resolve_frame_assignments()
        frame_classifier_predictions = self._resolve_frame_classifier_predictions()
        ner_result = self._resolve_ner_result()
        context = ReportContext(
            schema=schema,
            aggregates=aggregates,
            frame_classifications=frame_classifications,
            stance_classifications=stance_classifications,
            frame_assignments=frame_assignments,
            frame_classifier_predictions=frame_classifier_predictions,
            ner_result=ner_result,
            title=self.config.report.title,
            subtitle=self.config.report.subtitle,
        )
        html_text = self._render(context)
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)
        self.paths.html.write_text(html_text, encoding="utf-8")
        return self.paths.html

    def _resolve_schema(self) -> Optional[FrameSchema]:
        if getattr(self.state, "frame_schema", None) is not None:
            return self.state.frame_schema
        # Try config's schema source path first (for file-based schemas)
        framing_config = getattr(self.config, "framing", None)
        if framing_config:
            schema_config = getattr(framing_config, "schema", None)
            if schema_config and getattr(schema_config, "source", "") == "file":
                config_schema_path = getattr(schema_config, "schema_path", None)
                if config_schema_path and Path(config_schema_path).exists():
                    return load_schema(Path(config_schema_path))
        # Fall back to results directory schema
        schema_path = getattr(self.paths, "framing_schema_path", None)
        if schema_path and Path(schema_path).exists():
            return load_schema(Path(schema_path))
        return None

    def _resolve_aggregates(self) -> DiscourseAggregates:
        if getattr(self.state, "aggregates", None) is not None:
            return self.state.aggregates
        # New directory-based loading
        return DiscourseAggregates.load(self.paths.aggregates_dir)

    def _resolve_frame_classifications(self) -> DocumentClassifications:
        if getattr(self.state, "frame_classifications", None) is not None:
            return self.state.frame_classifications
        class_dir = getattr(self.paths, "framing_classifications_dir", None)
        if class_dir and Path(class_dir).exists():
            return DocumentClassifications.from_folder(Path(class_dir))
        return DocumentClassifications()

    def _resolve_frame_assignments(self) -> FrameAssignments:
        if getattr(self.state, "frame_assignments", None) is not None:
            return self.state.frame_assignments
        assignments_path = getattr(self.paths, "framing_assignments_path", None)
        if assignments_path and Path(assignments_path).exists():
            try:
                return FrameAssignments.load(Path(assignments_path))
            except Exception:
                return FrameAssignments()
        return FrameAssignments()

    def _resolve_frame_classifier_predictions(self) -> List[Dict[str, object]]:
        predictions_path = self.paths.framing_classifier_dir.parent / "classifier_predictions.json"
        if predictions_path.exists():
            try:
                artifacts = FrameClassifierArtifacts.load_predictions(predictions_path)
                return artifacts.predictions
            except Exception:
                return []
        return []

    def _resolve_stance_classifications(self) -> StanceAssignments:
        if getattr(self.state, "stance_classifications", None) is not None:
            return self.state.stance_classifications
        stance_path = self.paths.stance_classifications_dir / "classifications.json"
        if stance_path.exists():
            return StanceAssignments.load(stance_path)
        return StanceAssignments()

    def _resolve_ner_result(self) -> Optional[NERResult]:
        if getattr(self.state, "ner_result", None) is not None:
            return self.state.ner_result
        ner_path = getattr(self.paths, "ner_entities_path", None)
        if ner_path and Path(ner_path).exists():
            try:
                return NERResult.load(Path(ner_path))
            except Exception:
                return None
        return None

    def _render(self, ctx: ReportContext) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        frames_html = self._render_frames(ctx.schema)
        frame_distribution_html = self._render_frame_distribution(ctx)
        frame_shares_html = self._render_frame_shares(ctx)
        frame_attr_html = self._render_frame_attributions(ctx)
        ner_entities_html = self._render_ner_entities(ctx)
        overall_html = self._render_overall_stance(ctx.aggregates)
        frame_target_html = self._render_frame_target(ctx.aggregates, ctx.schema)
        examples_html = self._render_examples(ctx)

        subtitle = f"<p class=\"subtitle\">{html.escape(ctx.subtitle)}</p>" if ctx.subtitle else ""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(ctx.title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Work+Sans:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-1: #f2f1ec;
      --bg-2: #e4e0d6;
      --ink: #1d1b16;
      --muted: #6c665b;
      --card: #ffffff;
      --accent: #c56a2b;
      --supports: #2c8f5b;
      --opposes: #c1443a;
      --neutral: #6b7280;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Work Sans", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top right, var(--bg-2), var(--bg-1));
    }}
    header {{
      padding: 48px 8vw 32px;
    }}
    h1 {{
      font-family: "Space Grotesk", sans-serif;
      font-size: clamp(2.2rem, 3vw, 3.2rem);
      margin: 0 0 8px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
    }}
    .meta {{
      margin-top: 12px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    main {{
      padding: 0 8vw 64px;
      display: flex;
      flex-direction: column;
      gap: 36px;
    }}
    .section {{
      background: var(--card);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 18px 40px rgba(40, 35, 28, 0.08);
      animation: fadeUp 0.6s ease both;
    }}
    .section h2 {{
      font-family: "Space Grotesk", sans-serif;
      margin: 0 0 16px;
      font-size: 1.4rem;
    }}
    .plot-section {{
      margin-bottom: 24px;
    }}
    .plot-section h3 {{
      font-family: "Space Grotesk", sans-serif;
      margin: 0 0 12px;
      font-size: 1.1rem;
      color: var(--muted);
    }}
    .plot-section:last-child {{
      margin-bottom: 0;
    }}
    .plotly-chart {{
      width: 100%;
      min-height: 400px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .frame-card {{
      padding: 16px;
      border-radius: 14px;
      border: 1px solid #e7e1d8;
      background: #faf9f6;
    }}
    .frame-card h3 {{
      font-family: "Space Grotesk", sans-serif;
      margin: 0 0 6px;
      font-size: 1.05rem;
      color: var(--accent);
    }}
    .frame-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.4;
    }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: #ece6dd;
      overflow: hidden;
      margin-top: 6px;
    }}
    .bar span {{
      display: block;
      height: 100%;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .table {{
      display: grid;
      gap: 12px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .controls select {{
      margin-left: 6px;
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid #e7e1d8;
      background: #fff;
      font-family: "Work Sans", sans-serif;
    }}
    .controls button {{
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid #e7e1d8;
      background: #fff;
      cursor: pointer;
      font-family: "Work Sans", sans-serif;
    }}
    .examples-table {{
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #efe9df;
    }}
    .examples-header, .examples-row {{
      display: grid;
      grid-template-columns: 1.1fr 1.4fr 0.7fr 1fr 3fr;
      gap: 12px;
      padding: 10px 14px;
      align-items: start;
    }}
    .examples-header {{
      background: #f7f3ed;
      font-weight: 600;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .examples-row {{
      border-top: 1px solid #efe9df;
      background: #fff;
      font-size: 0.95rem;
    }}
    .examples-row .text {{
      color: var(--muted);
      line-height: 1.4;
    }}
    .examples-row .scores {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .examples-row .stance {{
      font-weight: 600;
      text-transform: capitalize;
    }}
    .examples-row .stance.supports {{ color: var(--supports); }}
    .examples-row .stance.opposes {{ color: var(--opposes); }}
    .examples-row .stance.neutral {{ color: var(--neutral); }}
    .frame-table {{
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #efe9df;
    }}
    .frame-header, .frame-row {{
      display: grid;
      grid-template-columns: 90px 1.4fr 1.4fr 3fr;
      gap: 12px;
      padding: 10px 14px;
      align-items: start;
    }}
    .frame-header {{
      background: #f7f3ed;
      font-weight: 600;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .frame-row {{
      border-top: 1px solid #efe9df;
      background: #fff;
      font-size: 0.95rem;
    }}
    .frame-row .text {{
      color: var(--muted);
      line-height: 1.4;
    }}
    .link-icons {{
      display: flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .link-icon {{
      font-size: 1rem;
      text-decoration: none;
      opacity: 0.7;
      transition: opacity 0.15s;
    }}
    .link-icon:hover {{
      opacity: 1;
    }}
    .copy-btn {{
      background: none;
      border: 1px solid #e7e1d8;
      border-radius: 4px;
      padding: 2px 6px;
      cursor: pointer;
      font-size: 0.8rem;
      color: var(--muted);
      transition: background 0.15s;
    }}
    .copy-btn:hover {{
      background: #f5f2ec;
    }}
    .copy-btn.copied {{
      background: #d4edda;
      border-color: #c3e6cb;
    }}
    .prob-cell {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .prob-bar {{
      position: relative;
      background: #f2eee7;
      height: 22px;
      border-radius: 8px;
      overflow: hidden;
    }}
    .prob-fill {{
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      opacity: 0.9;
    }}
    .prob-label {{
      position: relative;
      z-index: 1;
      padding-left: 8px;
      line-height: 22px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #423b32;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 8px;
      border-radius: 10px;
      background: #f5f2ec;
      border: 1px solid #e7e1d8;
      font-size: 0.88rem;
    }}
    .row {{
      border: 1px solid #efe9df;
      border-radius: 12px;
      padding: 12px;
      background: #fcfbf9;
    }}
    .row h4 {{
      margin: 0 0 8px;
      font-size: 1rem;
      font-family: "Space Grotesk", sans-serif;
    }}
    .target {{
      margin-top: 8px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 720px) {{
      header {{ padding: 40px 6vw 24px; }}
      main {{ padding: 0 6vw 48px; }}
      .section {{ padding: 22px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(ctx.title)}</h1>
    {subtitle}
    <div class="meta">Generated {html.escape(timestamp)}</div>
  </header>
  <main>
    <section class="section">
      <h2>Frames</h2>
      {frames_html}
    </section>
    <section class="section">
      <h2>Frame Distribution</h2>
      {frame_distribution_html}
    </section>
    <section class="section">
      <h2>Frame Shares</h2>
      {frame_shares_html}
    </section>
    <section class="section">
      <h2>Frame Attributions</h2>
      {frame_attr_html}
    </section>
    <section class="section">
      <h2>Named Entities</h2>
      {ner_entities_html}
    </section>
    <section class="section">
      <h2>Overall Stance</h2>
      {overall_html}
    </section>
    <section class="section">
      <h2>Stance By Frame</h2>
      {frame_target_html}
    </section>
    <section class="section">
      <h2>Example Attributions</h2>
      {examples_html}
    </section>
  </main>
</body>
</html>
"""

    def _render_frames(self, schema: Optional[FrameSchema]) -> str:
        if not schema or not schema.frames:
            return "<p class=\"subtitle\">No frame schema available.</p>"
        cards = []
        for frame in schema.frames:
            name = html.escape(frame.short_name or frame.name or frame.frame_id)
            description = html.escape(frame.description or "")
            cards.append(
                f"<div class=\"frame-card\"><h3>{name}</h3><p>{description}</p></div>"
            )
        return f"<div class=\"grid\">{''.join(cards)}</div>"

    def _render_frame_distribution(self, ctx: ReportContext) -> str:
        """Render frame distribution plots from aggregates."""
        if not ctx.aggregates.frames:
            return "<p class=\"subtitle\">No frame aggregates available.</p>"

        frames = ctx.aggregates.frames

        # Build frame labels and color map
        frame_labels: Dict[str, str] = {}
        if ctx.schema and ctx.schema.frames:
            for f in ctx.schema.frames:
                frame_labels[f.frame_id] = f.short_name or f.name or f.frame_id

        frame_ids = frames.documents["frame_id"].unique().tolist() if not frames.documents.empty else []
        color_map = build_color_map(frame_ids)

        # Get plots directory
        plots_dir = getattr(self.paths, "plots_dir", None)
        if plots_dir:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)

        sections = []

        # Global distribution
        if frames.global_totals is not None and not frames.global_totals.empty:
            export_path = Path(plots_dir) / "global_distribution.png" if plots_dir else None
            global_html = plot_global_distribution(
                frames.global_totals,
                frame_labels=frame_labels,
                color_map=color_map,
                export_path=export_path,
            )
            if global_html:
                sections.append(
                    "<div class=\"plot-section\">"
                    "<h3>Overall Distribution</h3>"
                    f"{global_html}"
                    "</div>"
                )

        # By year
        if frames.by_year is not None and not frames.by_year.empty:
            export_path = Path(plots_dir) / "by_year.png" if plots_dir else None
            year_html = plot_by_year(
                frames.by_year,
                frame_labels=frame_labels,
                color_map=color_map,
                export_path=export_path,
            )
            if year_html:
                sections.append(
                    "<div class=\"plot-section\">"
                    "<h3>Distribution by Year</h3>"
                    f"{year_html}"
                    "</div>"
                )

        # By domain
        if frames.by_domain is not None and not frames.by_domain.empty:
            export_path = Path(plots_dir) / "by_domain.png" if plots_dir else None
            domain_html = plot_by_domain(
                frames.by_domain,
                frame_labels=frame_labels,
                color_map=color_map,
                top_n=12,
                export_path=export_path,
            )
            if domain_html:
                sections.append(
                    "<div class=\"plot-section\">"
                    "<h3>Distribution by Domain</h3>"
                    f"{domain_html}"
                    "</div>"
                )

            # By domain and year
            if frames.by_domain_year is not None and not frames.by_domain_year.empty:
                export_path = Path(plots_dir) / "by_domain_year.png" if plots_dir else None
                domain_year_html = plot_by_domain_year(
                    frames.by_domain_year,
                    frame_labels=frame_labels,
                    color_map=color_map,
                    top_n=8,
                    export_path=export_path,
                )
                if domain_year_html:
                    sections.append(
                        "<div class=\"plot-section\">"
                        "<h3>Distribution by Domain and Year</h3>"
                        f"{domain_year_html}"
                        "</div>"
                    )

            # Domain counts
            export_path = Path(plots_dir) / "domain_counts.png" if plots_dir else None
            counts_html = plot_document_count_by_domain(
                frames.by_domain,
                top_n=15,
                export_path=export_path,
            )
            if counts_html:
                sections.append(
                    "<div class=\"plot-section\">"
                    "<h3>Document Counts by Domain</h3>"
                    f"{counts_html}"
                    "</div>"
                )

        # Co-occurrence
        if frames.co_occurrence is not None and not frames.co_occurrence.empty:
            export_path = Path(plots_dir) / "co_occurrence.png" if plots_dir else None
            coocc_html = plot_co_occurrence_matrix(
                frames.co_occurrence,
                frame_labels=frame_labels,
                max_frames=18,
                export_path=export_path,
            )
            if coocc_html:
                sections.append(
                    "<div class=\"plot-section\">"
                    "<h3>Frame Co-occurrence (Document-level)</h3>"
                    f"{coocc_html}"
                    "</div>"
                )

        # By corpus (if multiple corpora)
        if frames.by_corpus is not None and not frames.by_corpus.empty:
            if frames.by_corpus["corpus"].nunique() > 1:
                export_path = Path(plots_dir) / "by_corpus.png" if plots_dir else None
                corpus_html = plot_by_corpus(
                    frames.by_corpus,
                    frame_labels=frame_labels,
                    color_map=color_map,
                    export_path=export_path,
                )
                if corpus_html:
                    sections.append(
                        "<div class=\"plot-section\">"
                        "<h3>Distribution by Corpus</h3>"
                        f"{corpus_html}"
                        "</div>"
                    )

        if not sections:
            return "<p class=\"subtitle\">No frame distribution data available.</p>"

        return "".join(sections)

    def _render_overall_stance(self, aggregates: DiscourseAggregates) -> str:
        stances = aggregates.stances
        if not stances or not stances.overall_by_target:
            return "<p class=\"subtitle\">No stance aggregates available.</p>"
        blocks = []
        for target, counts in stances.overall_by_target.items():
            total = sum(counts.values()) or 1
            blocks.append(
                f"<div class=\"row\"><h4>{html.escape(target)}</h4>"
                f"{self._render_bar_group(counts, total)}</div>"
            )
        legend = self._render_legend()
        return legend + f"<div class=\"table\">{''.join(blocks)}</div>"

    def _render_frame_shares(self, ctx: ReportContext) -> str:
        frame_totals = self._compute_frame_totals(ctx)
        if not frame_totals:
            return "<p class=\"subtitle\">No frame share data available.</p>"
        total = sum(frame_totals.values()) or 1
        frame_labels = {f.frame_id: (f.short_name or f.name or f.frame_id) for f in (ctx.schema.frames if ctx.schema else [])}
        items = sorted(frame_totals.items(), key=lambda item: -item[1])
        blocks = []
        for frame_id, count in items:
            share = count / total
            label = html.escape(frame_labels.get(frame_id, frame_id))
            blocks.append(
                f"<div class=\"row\">"
                f"<h4>{label}</h4>"
                f"<div class=\"bar\"><span style=\"width:{share*100:.1f}%;background:var(--accent);\"></span></div>"
                f"<div class=\"legend\"><span>{count:.1f} total score ({share*100:.1f}%)</span></div>"
                f"</div>"
            )
        return f"<div class=\"table\">{''.join(blocks)}</div>"

    def _render_probability_bars(
        self,
        probabilities: Dict[str, float],
        frame_lookup: Dict[str, Dict[str, str]],
        color_map: Dict[str, str],
    ) -> str:
        if not probabilities:
            return "—"
        bars: List[str] = []
        sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        for frame_id, score in sorted_items:
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                continue
            if score_val <= 0:
                continue
            width = max(2.0, min(100.0, score_val * 100.0))
            color = color_map.get(frame_id, "#4F8EF7")
            label = frame_lookup.get(frame_id, {}).get("short", frame_id)
            bars.append(
                "<div class=\"prob-bar\">"
                f"<div class=\"prob-fill\" style=\"width:{width:.1f}%; background:{color};\"></div>"
                f"<span class=\"prob-label\">{html.escape(label)} ({score_val:.0%})</span>"
                "</div>"
            )
        return "".join(bars) if bars else "—"

    def _render_frame_attributions(self, ctx: ReportContext) -> str:
        assignments = list(ctx.frame_assignments or [])
        if not assignments:
            return "<p class=\"subtitle\">No frame assignments available.</p>"
        frame_labels = {f.frame_id: (f.short_name or f.name or f.frame_id) for f in (ctx.schema.frames if ctx.schema else [])}
        frame_lookup = {fid: {"short": name} for fid, name in frame_labels.items()}
        frame_ids = [f.frame_id for f in ctx.schema.frames] if ctx.schema else []
        color_map = build_color_map(frame_ids)

        # Build frame options for dropdown (sorted by frame_id for consistency)
        frame_options = []
        if ctx.schema and ctx.schema.frames:
            for frame in ctx.schema.frames:
                frame_options.append({
                    "id": frame.frame_id,
                    "label": frame.short_name or frame.name or frame.frame_id
                })

        clf_lookup: Dict[str, Dict[str, object]] = {}
        for entry in ctx.frame_classifier_predictions or []:
            pid = str(entry.get("passage_id", "")).strip()
            if pid:
                clf_lookup[pid] = entry

        rows: List[Dict[str, object]] = []
        max_rows = 5000
        for assignment in assignments:
            llm_probs = assignment.probabilities or {}
            # Check if LLM assignment has any non-zero probabilities
            has_llm = any(float(v) > 0 for v in llm_probs.values()) if llm_probs else False
            # Get top LLM frame and its score
            top_llm_frame = ""
            top_llm_score = 0.0
            if llm_probs:
                sorted_llm = sorted(llm_probs.items(), key=lambda x: float(x[1]), reverse=True)
                if sorted_llm and float(sorted_llm[0][1]) > 0:
                    top_llm_frame = sorted_llm[0][0]
                    top_llm_score = float(sorted_llm[0][1])

            llm_html = self._render_probability_bars(llm_probs, frame_lookup, color_map) if llm_probs else "—"

            clf_entry = clf_lookup.get(assignment.passage_id, {})
            clf_probs = clf_entry.get("probabilities", {}) if isinstance(clf_entry, dict) else {}
            # Get top classifier frame and its score
            top_clf_frame = ""
            top_clf_score = 0.0
            if clf_probs:
                sorted_clf = sorted(clf_probs.items(), key=lambda x: float(x[1]), reverse=True)
                if sorted_clf:
                    top_clf_frame = sorted_clf[0][0]
                    top_clf_score = float(sorted_clf[0][1])

            clf_html = (
                self._render_probability_bars(clf_probs, frame_lookup, color_map)
                if isinstance(clf_probs, dict) and clf_probs
                else "—"
            )

            text = assignment.passage_text or ""
            # if len(text) > 280:
            #     text = text[:277].rstrip() + "..."

            # Extract metadata for links
            metadata = assignment.metadata if isinstance(assignment.metadata, dict) else {}
            url = metadata.get("url") or ""
            doc_folder_path = metadata.get("doc_folder_path") or ""

            rows.append(
                {
                    "passage_id": assignment.passage_id,
                    "llm": llm_html,
                    "clf": clf_html,
                    "text": text,
                    "hasLlm": has_llm,
                    "topLlmFrame": top_llm_frame,
                    "topLlmScore": top_llm_score,
                    "topClfFrame": top_clf_frame,
                    "topClfScore": top_clf_score,
                    "url": url,
                    "docFolderPath": doc_folder_path,
                }
            )
            if len(rows) >= max_rows:
                break

        if not rows:
            return "<p class=\"subtitle\">No attribution rows available.</p>"

        payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        frame_options_json = json.dumps(frame_options, ensure_ascii=False)
        note = ""
        if len(rows) >= max_rows:
            note = f"<p class=\"subtitle\">Showing first {max_rows} passages.</p>"

        return f"""{note}
<div class="controls">
  <label>Filter
    <select id="frame-filter">
      <option value="all">All rows</option>
      <option value="llm_only" selected>With LLM assignment</option>
      <option value="no_llm">Without LLM assignment</option>
    </select>
  </label>
  <label>Sort by
    <select id="frame-sort">
      <option value="default">Default order</option>
      <option value="llm_desc">LLM top frame (desc)</option>
      <option value="clf_desc">Classifier top frame (desc)</option>
    </select>
  </label>
  <label>Frame
    <select id="frame-frame-filter">
      <option value="all">All frames</option>
    </select>
  </label>
</div>
<div class="controls">
  <label>Rows per page
    <select id="frame-page-size">
      <option value="10" selected>10</option>
      <option value="50">50</option>
      <option value="100">100</option>
      <option value="500">500</option>
      <option value="1000">1000</option>
    </select>
  </label>
  <button id="frame-prev">Prev</button>
  <span id="frame-page-info"></span>
  <button id="frame-next">Next</button>
</div>
<div class="frame-table">
  <div class="frame-header">
    <div>Links</div>
    <div>LLM Frames</div>
    <div>Classifier Frames</div>
    <div>Text</div>
  </div>
  <div id="frame-body"></div>
</div>
<script id="frame-data" type="application/json">{payload}</script>
<script>
(function() {{
  const allFrameRows = JSON.parse(document.getElementById('frame-data').textContent);
  const frameOptions = {frame_options_json};
  const frameBody = document.getElementById('frame-body');
  const framePageSizeSelect = document.getElementById('frame-page-size');
  const framePageInfo = document.getElementById('frame-page-info');
  const frameFilterSelect = document.getElementById('frame-filter');
  const frameSortSelect = document.getElementById('frame-sort');
  const frameFrameFilterSelect = document.getElementById('frame-frame-filter');

  // Populate frame filter dropdown
  frameOptions.forEach(f => {{
    const opt = document.createElement('option');
    opt.value = f.id;
    opt.textContent = f.label;
    frameFrameFilterSelect.appendChild(opt);
  }});

  let framePageSize = parseInt(framePageSizeSelect.value, 10);
  let framePage = 0;
  let filteredRows = allFrameRows;

  function escapeHtmlFrame(value) {{
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }}

  function applyFiltersAndSort() {{
    const filterVal = frameFilterSelect.value;
    const sortVal = frameSortSelect.value;
    const frameVal = frameFrameFilterSelect.value;

    // Filter
    filteredRows = allFrameRows.filter(row => {{
      if (filterVal === 'llm_only' && !row.hasLlm) return false;
      if (filterVal === 'no_llm' && row.hasLlm) return false;
      if (frameVal !== 'all') {{
        // Show rows where either LLM or classifier top frame matches
        if (row.topLlmFrame !== frameVal && row.topClfFrame !== frameVal) return false;
      }}
      return true;
    }});

    // Sort
    if (sortVal === 'llm_desc') {{
      filteredRows = [...filteredRows].sort((a, b) => b.topLlmScore - a.topLlmScore);
    }} else if (sortVal === 'clf_desc') {{
      filteredRows = [...filteredRows].sort((a, b) => b.topClfScore - a.topClfScore);
    }}

    framePage = 0;
  }}

  function copyPassageId(passageId, btn) {{
    navigator.clipboard.writeText(passageId).then(() => {{
      btn.classList.add('copied');
      btn.textContent = '✓';
      setTimeout(() => {{
        btn.classList.remove('copied');
        btn.textContent = '📋';
      }}, 1500);
    }});
  }}

  function renderFrameTable() {{
    const totalPages = Math.max(1, Math.ceil(filteredRows.length / framePageSize));
    framePage = Math.max(0, Math.min(framePage, totalPages - 1));
    const start = framePage * framePageSize;
    const slice = filteredRows.slice(start, start + framePageSize);
    frameBody.innerHTML = slice.map((row, idx) => {{
      const urlIcon = row.url ? `<a class="link-icon" href="${{escapeHtmlFrame(row.url)}}" target="_blank" rel="noopener noreferrer" title="Open article">🔗</a>` : '';
      const folderIcon = row.docFolderPath ? `<a class="link-icon" href="file://${{escapeHtmlFrame(row.docFolderPath)}}" target="_blank" title="Open document folder">📁</a>` : '';
      const copyBtn = `<button class="copy-btn" onclick="copyPassageId('${{escapeHtmlFrame(row.passage_id)}}', this)" title="Copy passage ID: ${{escapeHtmlFrame(row.passage_id)}}">📋</button>`;
      return `
        <div class="frame-row">
          <div class="link-icons">${{urlIcon}}${{folderIcon}}${{copyBtn}}</div>
          <div class="prob-cell">${{row.llm}}</div>
          <div class="prob-cell">${{row.clf}}</div>
          <div class="text">${{escapeHtmlFrame(row.text)}}</div>
        </div>
      `;
    }}).join('');
    framePageInfo.textContent = `Page ${{framePage + 1}} / ${{totalPages}} (${{filteredRows.length}} rows)`;
  }}

  // Make copyPassageId available globally
  window.copyPassageId = copyPassageId;

  document.getElementById('frame-prev').addEventListener('click', () => {{ framePage -= 1; renderFrameTable(); }});
  document.getElementById('frame-next').addEventListener('click', () => {{ framePage += 1; renderFrameTable(); }});
  framePageSizeSelect.addEventListener('change', (e) => {{
    framePageSize = parseInt(e.target.value, 10);
    framePage = 0;
    renderFrameTable();
  }});
  frameFilterSelect.addEventListener('change', () => {{ applyFiltersAndSort(); renderFrameTable(); }});
  frameSortSelect.addEventListener('change', () => {{ applyFiltersAndSort(); renderFrameTable(); }});
  frameFrameFilterSelect.addEventListener('change', () => {{ applyFiltersAndSort(); renderFrameTable(); }});

  // Apply filters on initial load (respects default "llm_only" selection)
  applyFiltersAndSort();
  renderFrameTable();
}})();
</script>
"""

    def _render_ner_entities(self, ctx: ReportContext) -> str:
        """Render NER entities as an interactive table with filters."""
        if not ctx.ner_result or not ctx.ner_result.documents:
            return "<p class=\"subtitle\">No named entities extracted. Enable NER in config with <code>ner.enabled: true</code>.</p>"

        # Build lookup from doc_id to URL from frame_classifications (URL is at document level)
        doc_to_url: Dict[str, str] = {}
        for doc_class in ctx.frame_classifications or []:
            payload = doc_class.payload if isinstance(doc_class.payload, dict) else {}
            doc_id = str(payload.get("doc_id", "")).strip()
            url = str(payload.get("url", "")).strip()
            if doc_id and url:
                doc_to_url[doc_id] = url

        # Aggregate entities by (type, text) -> count
        entity_counts: Dict[tuple, int] = {}
        entity_frames: Dict[tuple, Dict[str, int]] = {}  # Track which frames each entity appears in
        entity_articles: Dict[tuple, Dict[str, str]] = {}  # Track doc_id -> url for each entity

        for doc in ctx.ner_result.documents:
            doc_url = doc_to_url.get(doc.doc_id, "")
            for chunk in doc.chunks:
                frame_id = chunk.frame_id
                for entity in chunk.entities:
                    key = (entity.type, entity.text)
                    entity_counts[key] = entity_counts.get(key, 0) + 1
                    if key not in entity_frames:
                        entity_frames[key] = {}
                    entity_frames[key][frame_id] = entity_frames[key].get(frame_id, 0) + 1
                    # Track articles (use doc_id as key to deduplicate)
                    if key not in entity_articles:
                        entity_articles[key] = {}
                    if doc.doc_id not in entity_articles[key] and doc_url:
                        entity_articles[key][doc.doc_id] = doc_url

        if not entity_counts:
            return "<p class=\"subtitle\">No entities found in chunks meeting the frame threshold.</p>"

        # Get frame labels
        frame_labels = {}
        if ctx.schema and ctx.schema.frames:
            frame_labels = {f.frame_id: (f.short_name or f.name or f.frame_id) for f in ctx.schema.frames}

        # Build rows sorted by count descending
        rows = []
        for (ent_type, ent_text), count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            # Get top frame for this entity
            frames_for_entity = entity_frames.get((ent_type, ent_text), {})
            top_frame = ""
            if frames_for_entity:
                top_frame_id = max(frames_for_entity.items(), key=lambda x: x[1])[0]
                top_frame = frame_labels.get(top_frame_id, top_frame_id)
            # Get article URLs for this entity
            articles = list(entity_articles.get((ent_type, ent_text), {}).values())
            rows.append({
                "type": ent_type,
                "text": ent_text,
                "count": count,
                "topFrame": top_frame,
                "articles": articles,
            })

        # Get unique entity types for filter
        entity_types = sorted(set(r["type"] for r in rows))

        payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        entity_types_json = json.dumps(entity_types, ensure_ascii=False)

        stats = (
            f"<p class=\"subtitle\">"
            f"{len(rows)} unique entities from {ctx.ner_result.n_chunks} chunks "
            f"(threshold: {ctx.ner_result.frame_threshold}, language: {ctx.ner_result.language})"
            f"</p>"
        )

        return f"""{stats}
<div class="controls">
  <label>Entity Type
    <select id="ner-type-filter">
      <option value="all">All types</option>
    </select>
  </label>
  <label>Search
    <input type="text" id="ner-search" placeholder="Filter by text..." style="padding:6px 10px;border-radius:8px;border:1px solid #e7e1d8;">
  </label>
  <label>Sort by
    <select id="ner-sort">
      <option value="count_desc" selected>Count (high to low)</option>
      <option value="count_asc">Count (low to high)</option>
      <option value="text_asc">Text (A-Z)</option>
      <option value="type_asc">Type (A-Z)</option>
    </select>
  </label>
</div>
<div class="controls">
  <label>Rows per page
    <select id="ner-page-size">
      <option value="25">25</option>
      <option value="50" selected>50</option>
      <option value="100">100</option>
      <option value="500">500</option>
    </select>
  </label>
  <button id="ner-prev">Prev</button>
  <span id="ner-page-info"></span>
  <button id="ner-next">Next</button>
</div>
<div class="ner-table" style="border-radius:14px;overflow:hidden;border:1px solid #efe9df;">
  <div class="ner-header" style="display:grid;grid-template-columns:120px 2fr 80px 1fr 1.5fr;gap:12px;padding:10px 14px;background:#f7f3ed;font-weight:600;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.04em;">
    <div>Type</div>
    <div>Entity</div>
    <div>Count</div>
    <div>Top Frame</div>
    <div>Articles</div>
  </div>
  <div id="ner-body"></div>
</div>
<script id="ner-data" type="application/json">{payload}</script>
<script>
(function() {{
  const allNerRows = JSON.parse(document.getElementById('ner-data').textContent);
  const entityTypes = {entity_types_json};
  const nerBody = document.getElementById('ner-body');
  const nerTypeFilter = document.getElementById('ner-type-filter');
  const nerSearch = document.getElementById('ner-search');
  const nerSort = document.getElementById('ner-sort');
  const nerPageSize = document.getElementById('ner-page-size');
  const nerPageInfo = document.getElementById('ner-page-info');

  // Populate type filter
  entityTypes.forEach(t => {{
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    nerTypeFilter.appendChild(opt);
  }});

  let pageSize = parseInt(nerPageSize.value, 10);
  let page = 0;
  let filteredRows = allNerRows;

  function escapeHtml(value) {{
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }}

  function applyFiltersAndSort() {{
    const typeVal = nerTypeFilter.value;
    const searchVal = nerSearch.value.toLowerCase().trim();
    const sortVal = nerSort.value;

    filteredRows = allNerRows.filter(row => {{
      if (typeVal !== 'all' && row.type !== typeVal) return false;
      if (searchVal && !row.text.toLowerCase().includes(searchVal)) return false;
      return true;
    }});

    if (sortVal === 'count_desc') {{
      filteredRows.sort((a, b) => b.count - a.count);
    }} else if (sortVal === 'count_asc') {{
      filteredRows.sort((a, b) => a.count - b.count);
    }} else if (sortVal === 'text_asc') {{
      filteredRows.sort((a, b) => a.text.localeCompare(b.text));
    }} else if (sortVal === 'type_asc') {{
      filteredRows.sort((a, b) => a.type.localeCompare(b.type) || b.count - a.count);
    }}

    page = 0;
  }}

  function renderArticleLinks(articles, rowIdx) {{
    if (!articles || articles.length === 0) {{
      return '<span style="color:var(--muted);font-size:0.85rem;">—</span>';
    }}
    const maxVisible = 3;
    const visibleLinks = articles.slice(0, maxVisible).map((url, i) =>
      `<a href="${{escapeHtml(url)}}" target="_blank" rel="noopener noreferrer" class="link-icon" title="${{escapeHtml(url)}}">🔗</a>`
    ).join(' ');

    if (articles.length <= maxVisible) {{
      return `<div class="link-icons">${{visibleLinks}}</div>`;
    }}

    const hiddenLinks = articles.slice(maxVisible).map((url, i) =>
      `<a href="${{escapeHtml(url)}}" target="_blank" rel="noopener noreferrer" class="link-icon" title="${{escapeHtml(url)}}">🔗</a>`
    ).join(' ');

    return `
      <div class="link-icons">
        ${{visibleLinks}}
        <span class="ner-expand-toggle" data-row="${{rowIdx}}" style="cursor:pointer;color:var(--accent);font-size:0.85rem;margin-left:4px;" onclick="toggleNerExpand(${{rowIdx}})">+${{articles.length - maxVisible}} more</span>
        <span class="ner-hidden-links" id="ner-hidden-${{rowIdx}}" style="display:none;margin-left:4px;">${{hiddenLinks}}</span>
      </div>
    `;
  }}

  window.toggleNerExpand = function(rowIdx) {{
    const hidden = document.getElementById('ner-hidden-' + rowIdx);
    const toggle = document.querySelector('.ner-expand-toggle[data-row="' + rowIdx + '"]');
    if (hidden && toggle) {{
      if (hidden.style.display === 'none') {{
        hidden.style.display = 'inline';
        toggle.textContent = 'show less';
      }} else {{
        hidden.style.display = 'none';
        const count = hidden.querySelectorAll('a').length;
        toggle.textContent = '+' + count + ' more';
      }}
    }}
  }};

  function render() {{
    const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
    page = Math.max(0, Math.min(page, totalPages - 1));
    const start = page * pageSize;
    const slice = filteredRows.slice(start, start + pageSize);
    nerBody.innerHTML = slice.map((row, idx) => {{
      const typeColor = {{
        'PERSON': '#3b82f6',
        'ORG': '#10b981',
        'GPE': '#f59e0b',
        'LOC': '#8b5cf6',
        'NORP': '#ec4899',
        'FAC': '#06b6d4',
        'EVENT': '#ef4444',
        'PRODUCT': '#84cc16'
      }}[row.type] || '#6b7280';
      const articleLinks = renderArticleLinks(row.articles, start + idx);
      return `
        <div style="display:grid;grid-template-columns:120px 2fr 80px 1fr 1.5fr;gap:12px;padding:10px 14px;border-top:1px solid #efe9df;background:#fff;font-size:0.95rem;">
          <div><span class="pill" style="background:${{typeColor}}22;color:${{typeColor}};border-color:${{typeColor}}44;">${{escapeHtml(row.type)}}</span></div>
          <div style="font-weight:500;">${{escapeHtml(row.text)}}</div>
          <div style="color:var(--muted);">${{row.count}}</div>
          <div style="color:var(--muted);font-size:0.9rem;">${{escapeHtml(row.topFrame)}}</div>
          <div>${{articleLinks}}</div>
        </div>
      `;
    }}).join('');
    nerPageInfo.textContent = `Page ${{page + 1}} / ${{totalPages}} (${{filteredRows.length}} entities)`;
  }}

  document.getElementById('ner-prev').addEventListener('click', () => {{ page -= 1; render(); }});
  document.getElementById('ner-next').addEventListener('click', () => {{ page += 1; render(); }});
  nerPageSize.addEventListener('change', (e) => {{
    pageSize = parseInt(e.target.value, 10);
    page = 0;
    render();
  }});
  nerTypeFilter.addEventListener('change', () => {{ applyFiltersAndSort(); render(); }});
  nerSearch.addEventListener('input', () => {{ applyFiltersAndSort(); render(); }});
  nerSort.addEventListener('change', () => {{ applyFiltersAndSort(); render(); }});

  applyFiltersAndSort();
  render();
}})();
</script>
"""

    def _compute_frame_totals(self, ctx: ReportContext) -> Dict[str, float]:
        """Compute frame shares from probability mass when available."""
        totals: Dict[str, float] = {}
        if ctx.frame_classifications:
            for doc in ctx.frame_classifications:
                chunks = doc.payload.get("chunks", []) if isinstance(doc.payload, dict) else []
                for chunk in chunks or []:
                    if not isinstance(chunk, dict):
                        continue
                    probs = chunk.get("probabilities") or {}
                    for fid, val in probs.items():
                        try:
                            totals[fid] = totals.get(fid, 0.0) + float(val)
                        except Exception:
                            continue
        if totals:
            return totals
        # Fall back to stances.frame_totals if available
        stances = ctx.aggregates.stances
        if stances and stances.frame_totals:
            return {k: float(v) for k, v in stances.frame_totals.items()}
        return {}

    def _render_frame_target(self, aggregates: DiscourseAggregates, schema: Optional[FrameSchema]) -> str:
        stances = aggregates.stances
        if not stances or not stances.by_frame_target:
            return "<p class=\"subtitle\">No frame-level stance aggregates available.</p>"
        frame_labels = {f.frame_id: (f.short_name or f.name or f.frame_id) for f in (schema.frames if schema else [])}
        blocks = []
        for frame_id, targets in stances.by_frame_target.items():
            frame_name = html.escape(frame_labels.get(frame_id, frame_id))
            inner = []
            # Overall stance ratios across all targets for this frame.
            overall_counts: Dict[str, int] = {}
            for counts in targets.values():
                for label, value in (counts or {}).items():
                    overall_counts[label] = overall_counts.get(label, 0) + int(value or 0)
            if overall_counts:
                total = sum(overall_counts.values()) or 1
                inner.append(
                    f"<div class=\"target\"><strong>All targets</strong>"
                    f"{self._render_bar_group(overall_counts, total)}</div>"
                )
            for target, counts in targets.items():
                total = sum(counts.values()) or 1
                inner.append(
                    f"<div class=\"target\"><strong>{html.escape(target)}</strong>"
                    f"{self._render_bar_group(counts, total)}</div>"
                )
            blocks.append(f"<div class=\"row\"><h4>{frame_name}</h4>{''.join(inner)}</div>")
        legend = self._render_legend()
        return legend + f"<div class=\"table\">{''.join(blocks)}</div>"

    def _render_examples(self, ctx: ReportContext) -> str:
        if not ctx.stance_classifications:
            return "<p class=\"subtitle\">No stance attributions available.</p>"

        frame_labels = {f.frame_id: (f.short_name or f.name or f.frame_id) for f in (ctx.schema.frames if ctx.schema else [])}
        frame_by_chunk: Dict[str, Dict[str, object]] = {}

        for doc in ctx.frame_classifications:
            chunks = doc.payload.get("chunks", []) if isinstance(doc.payload, dict) else []
            for chunk in chunks or []:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                probs = chunk.get("probabilities") or {}
                ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                top_frames = [fid for fid, _ in ordered[:3]] if ordered else []
                primary = top_frames[0] if top_frames else ""
                frame_by_chunk[chunk_id] = {
                    "frame_id": primary,
                    "top_frames": top_frames,
                }

        rows: List[Dict[str, object]] = []
        max_rows = 5000
        for item in ctx.stance_classifications:
            info = frame_by_chunk.get(item.chunk_id)
            if not info:
                continue
            frame_id = info.get("frame_id", "")
            frame_name = frame_labels.get(frame_id, frame_id) if frame_id else "unknown"
            text = str(item.text or "")
            # if len(text) > 500:
            #     text = text[:500].rstrip() + "..."
            scores = item.scores or {}
            rows.append(
                {
                    "chunk_id": item.chunk_id,
                    "frame": frame_name,
                    "frame_id": frame_id,
                    "target": item.target,
                    "stance": item.label,
                    "supports": float(scores.get("supports", 0.0)),
                    "opposes": float(scores.get("opposes", 0.0)),
                    "neutral": float(scores.get("neutral", 0.0)),
                    "text": text,
                }
            )
            if len(rows) >= max_rows:
                break

        if not rows:
            return "<p class=\"subtitle\">No matching frame + stance examples found.</p>"

        payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        note = ""
        if len(rows) >= max_rows:
            note = f"<p class=\"subtitle\">Showing first {max_rows} attributions.</p>"

        return f"""{note}
<div class="controls">
  <label>Rows per page
    <select id="page-size">
      <option value="10">10</option>
      <option value="50">50</option>
      <option value="100">100</option>
      <option value="500">500</option>
      <option value="1000" selected>1000</option>
    </select>
  </label>
  <button id="prev-page">Prev</button>
  <span id="page-info"></span>
  <button id="next-page">Next</button>
</div>
<div class="examples-table">
  <div class="examples-header">
    <div>Frame</div>
    <div>Target</div>
    <div>Stance</div>
    <div>Scores</div>
    <div>Text</div>
  </div>
  <div id="examples-body"></div>
</div>
<script id="example-data" type="application/json">{payload}</script>
<script>
  const rows = JSON.parse(document.getElementById('example-data').textContent);
  const body = document.getElementById('examples-body');
  const pageSizeSelect = document.getElementById('page-size');
  const pageInfo = document.getElementById('page-info');
  let pageSize = parseInt(pageSizeSelect.value, 10);
  let page = 0;

  function escapeHtml(value) {{
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }}

  function render() {{
    const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.max(0, Math.min(page, totalPages - 1));
    const start = page * pageSize;
    const slice = rows.slice(start, start + pageSize);
    body.innerHTML = slice.map(row => {{
      return `
        <div class="examples-row">
          <div><strong>${{escapeHtml(row.frame)}}</strong></div>
          <div>${{escapeHtml(row.target)}}</div>
          <div class="stance ${{row.stance}}">${{escapeHtml(row.stance)}}</div>
          <div class="scores">
            <span>S ${{row.supports.toFixed(2)}}</span>
            <span>O ${{row.opposes.toFixed(2)}}</span>
            <span>N ${{row.neutral.toFixed(2)}}</span>
          </div>
          <div class="text">${{escapeHtml(row.text)}}</div>
        </div>
      `;
    }}).join('');
    pageInfo.textContent = `Page ${{page + 1}} / ${{totalPages}} (${{rows.length}} rows)`;
  }}

  document.getElementById('prev-page').addEventListener('click', () => {{ page -= 1; render(); }});
  document.getElementById('next-page').addEventListener('click', () => {{ page += 1; render(); }});
  pageSizeSelect.addEventListener('change', (e) => {{
    pageSize = parseInt(e.target.value, 10);
    page = 0;
    render();
  }});

  render();
</script>
"""

    def _render_bar_group(self, counts: Dict[str, int], total: int) -> str:
        supports = counts.get("supports", 0)
        opposes = counts.get("opposes", 0)
        neutral = counts.get("neutral", 0)
        return (
            f"<div class=\"bar\"><span style=\"width:{supports/total*100:.1f}%;background:var(--supports);\"></span></div>"
            f"<div class=\"bar\"><span style=\"width:{opposes/total*100:.1f}%;background:var(--opposes);\"></span></div>"
            f"<div class=\"bar\"><span style=\"width:{neutral/total*100:.1f}%;background:var(--neutral);\"></span></div>"
            f"<div class=\"legend\">"
            f"<span><i class=\"dot\" style=\"background:var(--supports);\"></i>Supports ({supports})</span>"
            f"<span><i class=\"dot\" style=\"background:var(--opposes);\"></i>Opposes ({opposes})</span>"
            f"<span><i class=\"dot\" style=\"background:var(--neutral);\"></i>Neutral ({neutral})</span>"
            f"</div>"
        )

    def _render_legend(self) -> str:
        return (
            "<div class=\"legend\" style=\"margin-bottom:12px;\">"
            "<span><i class=\"dot\" style=\"background:var(--supports);\"></i>Supports</span>"
            "<span><i class=\"dot\" style=\"background:var(--opposes);\"></i>Opposes</span>"
            "<span><i class=\"dot\" style=\"background:var(--neutral);\"></i>Neutral</span>"
            "</div>"
        )


__all__ = ["ReportBuilder"]
