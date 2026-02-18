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
from efi_analyser.ner import NERResult, ConsolidatedNERResult
from efi_analyser.claims import ClaimsAnalysisResult

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
    plot_dna_network,
)
from apps.discourse_analysis.analysis.aggregator import build_aggregates, DocumentFilter
from apps.discourse_analysis.analysis.dna_network import (
    get_network_positions,
    DNANetworkResult,
    CoalitionAnalysisResult,
)


@dataclass
class ReportContext:
    schema: Optional[FrameSchema]
    aggregates: DiscourseAggregates
    frame_classifications: DocumentClassifications
    frame_assignments: FrameAssignments
    frame_classifier_predictions: List[Dict[str, object]]
    ner_result: Optional[NERResult]
    claims_result: Optional[ClaimsAnalysisResult]
    title: str
    subtitle: Optional[str]
    dna_result: Optional[DNANetworkResult] = None
    coalition_result: Optional[CoalitionAnalysisResult] = None


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
        frame_assignments = self._resolve_frame_assignments()
        frame_classifier_predictions = self._resolve_frame_classifier_predictions()
        ner_result = self._resolve_ner_result()
        claims_result = self._resolve_claims_result()
        context = ReportContext(
            schema=schema,
            aggregates=aggregates,
            frame_classifications=frame_classifications,
            frame_assignments=frame_assignments,
            frame_classifier_predictions=frame_classifier_predictions,
            ner_result=ner_result,
            claims_result=claims_result,
            title=self.config.report.title,
            subtitle=self.config.report.subtitle,
            dna_result=getattr(self.state, "dna_result", None),
            coalition_result=getattr(self.state, "coalition_result", None),
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
        try:
            return DiscourseAggregates.load(self.paths.aggregates_dir)
        except Exception:
            return DiscourseAggregates()

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
        classifier_dir = getattr(self.paths, "framing_classifier_dir", None)
        if not classifier_dir:
            return []
        predictions_path = classifier_dir.parent / "classifier_predictions.json"
        if predictions_path.exists():
            try:
                artifacts = FrameClassifierArtifacts.load_predictions(predictions_path)
                return artifacts.predictions
            except Exception:
                return []
        return []

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

    def _resolve_ner_consolidated(self) -> Optional[ConsolidatedNERResult]:
        """Resolve consolidated NER result from NERResult or disk."""
        # Check if NERResult has consolidated embedded
        ner_result = self._resolve_ner_result()
        if ner_result and ner_result.consolidated is not None:
            return ner_result.consolidated

        # Fallback: load from separate file (for old cached results)
        ner_consolidated_path = getattr(self.paths, "ner_consolidated_path", None)
        if ner_consolidated_path and Path(ner_consolidated_path).exists():
            try:
                return ConsolidatedNERResult.load(Path(ner_consolidated_path))
            except Exception:
                return None
        return None

    def _resolve_claims_result(self) -> Optional[ClaimsAnalysisResult]:
        """Resolve claims analysis result from state or disk."""
        if getattr(self.state, "claims_result", None) is not None:
            return self.state.claims_result

        # Try loading individual files from disk
        from efi_analyser.claims import StatementsResult, ClaimSchema, AgreementsResult

        statements_path = getattr(self.paths, "statements_path", None)
        claims_path = getattr(self.paths, "claims_schema_path", None)
        agreements_path = getattr(self.paths, "agreements_path", None)

        try:
            statements = None
            claims = None
            agreements = None

            if statements_path and Path(statements_path).exists():
                statements = StatementsResult.load(Path(statements_path))
            if claims_path and Path(claims_path).exists():
                claims = ClaimSchema.load(Path(claims_path))
            if agreements_path and Path(agreements_path).exists():
                agreements = AgreementsResult.load(Path(agreements_path))

            if statements and claims and agreements:
                return ClaimsAnalysisResult(
                    statements=statements,
                    claims=claims,
                    agreements=agreements,
                )
        except Exception:
            pass

        return None

    def _render(self, ctx: ReportContext) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        has_frames = ctx.schema is not None and bool(getattr(ctx.schema, "frames", None))
        has_frame_aggregates = ctx.aggregates.frames is not None

        if has_frames or has_frame_aggregates:
            frames_html = self._render_frames(ctx.schema)
            frame_distribution_html = self._render_frame_distribution(ctx)
            frame_shares_html = self._render_frame_shares(ctx)
            frame_attr_html = self._render_frame_attributions(ctx)
        else:
            frames_html = ""
            frame_distribution_html = ""
            frame_shares_html = ""
            frame_attr_html = ""

        ner_entities_html = self._render_ner_entities(ctx)
        claims_analysis_html = self._render_claims_analysis(ctx)
        dna_network_html = self._render_dna_network(ctx)
        coalition_analysis_html = self._render_coalition_analysis(ctx)
        article_reader_html = self._render_article_reader(ctx)

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
    /* Article Reader */
    #ar-reader-layout {{ display: grid; grid-template-columns: 65% 35%; gap: 20px; margin-top: 16px; align-items: start; }}
    #ar-text-col {{ max-height: 70vh; overflow-y: auto; background: #faf9f6; border-radius: 12px; padding: 20px; border: 1px solid #efe9df; line-height: 1.7; font-size: 0.95rem; }}
    #ar-claims-col {{ position: sticky; top: 20px; max-height: 70vh; overflow-y: auto; }}
    #ar-claims-panel {{ background: #faf9f6; border-radius: 12px; padding: 16px; border: 1px solid #efe9df; }}
    #ar-claims-panel h4 {{ font-family: "Space Grotesk", sans-serif; margin: 0 0 14px; font-size: 0.9rem; color: var(--muted); font-weight: 600; }}
    .ar-chunk {{ margin-bottom: 1.2em; }}
    .ar-chunk-badge {{ font-size: 0.78rem; color: var(--muted); font-style: italic; margin-top: 4px; }}
    .ar-entity {{ border-radius: 3px; padding: 1px 2px; cursor: default; }}
    .ar-entity-PERSON  {{ background: rgba(59,130,246,.15); border-bottom: 2px solid #3b82f6; }}
    .ar-entity-ORG     {{ background: rgba(16,185,129,.15); border-bottom: 2px solid #10b981; }}
    .ar-entity-GPE, .ar-entity-LOC, .ar-entity-NORP {{ background: rgba(245,158,11,.15); border-bottom: 2px solid #f59e0b; }}
    .ar-statement {{ position: relative; background: rgba(255,200,0,.2); border-radius: 3px; padding: 1px 2px; cursor: pointer; border-bottom: 2px dashed var(--accent); transition: background .15s; }}
    .ar-statement:hover, .ar-statement.ar-active {{ background: rgba(197,106,43,.25); }}
    .ar-claim-row {{ margin-bottom: 12px; }}
    .ar-claim-label {{ font-size: 0.82rem; color: var(--ink); margin-bottom: 4px; line-height: 1.3; }}
    .ar-claim-meta {{ font-size: 0.76rem; color: var(--muted); margin-top: 2px; }}
    .ar-bar-track {{ position: relative; height: 18px; background: #ece6dd; border-radius: 999px; overflow: hidden; }}
    .ar-bar-fill {{ position: absolute; top: 0; bottom: 0; border-radius: 999px; transition: width .25s, left .25s; }}
    .ar-bar-center {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #aaa; transform: translateX(-50%); }}
    .ar-legend {{ display: flex; gap: 16px; font-size: .82rem; margin-top: 10px; flex-wrap: wrap; }}
    .ar-legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
    @media (max-width: 820px) {{ #ar-reader-layout {{ grid-template-columns: 1fr; }} #ar-claims-col {{ position: static; max-height: none; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(ctx.title)}</h1>
    {subtitle}
    <div class="meta">Generated {html.escape(timestamp)}</div>
  </header>
  <main>
    {'<section class="section"><h2>Frames</h2>' + frames_html + '</section>' if frames_html else ''}
    {'<section class="section"><h2>Frame Distribution</h2>' + frame_distribution_html + '</section>' if frame_distribution_html else ''}
    {'<section class="section"><h2>Frame Shares</h2>' + frame_shares_html + '</section>' if frame_shares_html else ''}
    {'<section class="section"><h2>Frame Attributions</h2>' + frame_attr_html + '</section>' if frame_attr_html else ''}
    <section class="section">
      <h2>Named Entities</h2>
      {ner_entities_html}
    </section>
    <section class="section">
      <h2>Claims Analysis</h2>
      {claims_analysis_html}
    </section>
    <section class="section">
      <h2>Discourse Network Analysis</h2>
      {dna_network_html}
    </section>
    <section class="section">
      <h2>Coalition Analysis</h2>
      {coalition_analysis_html}
    </section>
    <section class="section">
      <h2>Article Reader</h2>
      {article_reader_html}
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

    def _render_consolidated_entities(self, ctx: ReportContext, consolidated: ConsolidatedNERResult) -> str:
        """Render consolidated NER entities with timeseries."""
        # Get frame labels
        frame_labels = {}
        if ctx.schema and ctx.schema.frames:
            frame_labels = {f.frame_id: (f.short_name or f.name or f.frame_id) for f in ctx.schema.frames}

        # Build doc_id to year mapping from frame_classifications
        doc_to_year: Dict[str, int] = {}
        for doc_class in ctx.frame_classifications or []:
            payload = doc_class.payload if isinstance(doc_class.payload, dict) else {}
            doc_id = str(payload.get("doc_id", "")).strip()
            published_at = payload.get("published_at", "")
            if doc_id and published_at:
                try:
                    year = int(str(published_at)[:4])
                    if 1900 < year < 2100:
                        doc_to_year[doc_id] = year
                except (ValueError, TypeError):
                    pass

        # Build doc_id to URL mapping from frame_classifications
        doc_to_url: Dict[str, str] = {}
        for doc_class in ctx.frame_classifications or []:
            payload = doc_class.payload if isinstance(doc_class.payload, dict) else {}
            doc_id = str(payload.get("doc_id", "")).strip()
            url = str(payload.get("url", "")).strip()
            if doc_id and url:
                doc_to_url[doc_id] = url

        # Build a map of normalized alias -> consolidated entity for yearly/article tracking
        from efi_analyser.ner.consolidator import normalize_entity_text
        alias_to_entity: Dict[str, tuple] = {}  # normalized_alias -> (canonical_name, entity_type)
        for entity in consolidated.entities:
            for alias in entity.aliases:
                normalized_alias = normalize_entity_text(alias)
                alias_to_entity[normalized_alias] = (entity.canonical_name, entity.entity_type)

        # Track yearly counts and article URLs for each consolidated entity
        entity_yearly: Dict[tuple, Dict[int, int]] = {}  # (canonical_name, type) -> year -> count
        entity_articles: Dict[tuple, Dict[str, str]] = {}  # (canonical_name, type) -> doc_id -> url
        if ctx.ner_result and ctx.ner_result.documents:
            for doc in ctx.ner_result.documents:
                doc_year = doc_to_year.get(doc.doc_id)
                doc_url = doc_to_url.get(doc.doc_id, "")
                for chunk in doc.chunks:
                    for raw_entity in chunk.entities:
                        normalized_text = normalize_entity_text(raw_entity.text)
                        if normalized_text in alias_to_entity:
                            canonical_name, entity_type = alias_to_entity[normalized_text]
                            key = (canonical_name, entity_type)

                            # Track yearly
                            if doc_year:
                                if key not in entity_yearly:
                                    entity_yearly[key] = {}
                                entity_yearly[key][doc_year] = entity_yearly[key].get(doc_year, 0) + 1

                            # Track articles
                            if doc_url:
                                if key not in entity_articles:
                                    entity_articles[key] = {}
                                if doc.doc_id not in entity_articles[key]:
                                    entity_articles[key][doc.doc_id] = doc_url

        # Build rows sorted by count descending
        rows = []
        for entity in consolidated.entities:
            # Get top frame for this entity
            top_frame = ""
            if entity.frames:
                top_frame_id = max(entity.frames.items(), key=lambda x: x[1])[0]
                top_frame = frame_labels.get(top_frame_id, top_frame_id)

            # Get organization attribute for display
            org_attr = ""
            if entity.entity_type == "PERSON" and entity.attributes.get("organization"):
                org_attr = entity.attributes["organization"]
            elif entity.entity_type == "ORG" and entity.attributes.get("organization_type"):
                org_attr = entity.attributes["organization_type"]

            # Format aliases for display
            aliases_str = ", ".join(entity.aliases[:5])  # Show first 5 aliases
            if len(entity.aliases) > 5:
                aliases_str += f" (+{len(entity.aliases) - 5} more)"

            # Get yearly data and article URLs
            yearly = entity_yearly.get((entity.canonical_name, entity.entity_type), {})
            articles = list(entity_articles.get((entity.canonical_name, entity.entity_type), {}).values())

            rows.append({
                "canonical_name": entity.canonical_name,
                "type": entity.entity_type,
                "count": entity.total_count,
                "topFrame": top_frame,
                "orgAttr": org_attr,
                "aliases": aliases_str,
                "allAliases": entity.aliases,  # Full list for tooltip
                "aliasCount": len(entity.aliases),
                "confidence": entity.confidence,
                "wasTypeCorrection": len(entity.original_types) > 1 or (
                    len(entity.original_types) == 1 and list(entity.original_types.keys())[0] != entity.entity_type
                ),
                "yearly": yearly,
                "articles": articles,
            })

        # Get unique entity types for filter
        entity_types = sorted(set(r["type"] for r in rows))

        # Build timeseries data for PERSON and ORG entities
        all_years = sorted(set(y for r in rows for y in r.get("yearly", {}).keys()))
        person_timeseries = [
            {"name": r["canonical_name"], "total": r["count"], "yearly": r["yearly"]}
            for r in rows if r["type"] == "PERSON" and r["yearly"]
        ]
        org_timeseries = [
            {"name": r["canonical_name"], "total": r["count"], "yearly": r["yearly"]}
            for r in rows if r["type"] == "ORG" and r["yearly"]
        ]

        payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        entity_types_json = json.dumps(entity_types, ensure_ascii=False)
        person_ts_json = json.dumps(person_timeseries, ensure_ascii=False).replace("</", "<\\/")
        org_ts_json = json.dumps(org_timeseries, ensure_ascii=False).replace("</", "<\\/")
        all_years_json = json.dumps(all_years, ensure_ascii=False)

        stats = (
            f"<p class=\"subtitle\">"
            f"{len(rows)} consolidated entities (from {consolidated.raw_entity_count} raw) "
            f"using {consolidated.consolidation_model}"
            f"</p>"
        )

        # Type corrections summary
        corrections_html = ""
        if consolidated.type_corrections:
            corrections_list = []
            for orig_type, corrections in consolidated.type_corrections.items():
                for new_type, count in corrections.items():
                    corrections_list.append(f"{orig_type} → {new_type} ({count})")
            if corrections_list:
                corrections_html = f"<p class=\"subtitle\" style=\"font-size:0.9rem;\">Type corrections: {', '.join(corrections_list)}</p>"

        # Timeseries charts section
        timeseries_html = ""
        if person_timeseries or org_timeseries:
            timeseries_html = f"""
<div class="plot-section" style="margin-bottom:24px;">
  <h3>Entity Mentions Over Time</h3>
  <div class="controls" style="margin-bottom:12px;">
    <label>Entity Type
      <select id="ner-ts-type">
        <option value="PERSON" selected>People</option>
        <option value="ORG">Organizations</option>
      </select>
    </label>
    <label>Top N
      <select id="ner-ts-topn">
        <option value="5">Top 5</option>
        <option value="10" selected>Top 10</option>
        <option value="15">Top 15</option>
        <option value="20">Top 20</option>
      </select>
    </label>
  </div>
  <div id="ner-timeseries-plot"></div>
</div>
<script id="ner-person-ts-data" type="application/json">{person_ts_json}</script>
<script id="ner-org-ts-data" type="application/json">{org_ts_json}</script>
<script id="ner-all-years" type="application/json">{all_years_json}</script>
<script>
(function() {{
  const personData = JSON.parse(document.getElementById('ner-person-ts-data').textContent);
  const orgData = JSON.parse(document.getElementById('ner-org-ts-data').textContent);
  const allYears = JSON.parse(document.getElementById('ner-all-years').textContent);
  const topNSelect = document.getElementById('ner-ts-topn');
  const typeSelect = document.getElementById('ner-ts-type');

  const colors = ['#1E3D58', '#057D9F', '#F18F01', '#A23B72', '#6C63FF', '#3A7D44', '#F45B69', '#0E7C7B', '#F2A541', '#8B5CF6',
                  '#E63946', '#14B8A6', '#F59E0B', '#8B5A00', '#D946EF', '#06B6D4', '#84CC16', '#EF4444', '#10B981', '#3B82F6'];

  function renderTimeseries() {{
    const topN = parseInt(topNSelect.value, 10);
    const entityType = typeSelect.value;
    const data = entityType === 'PERSON' ? personData : orgData;

    if (!data || data.length === 0) {{
      document.getElementById('ner-timeseries-plot').innerHTML = '<p style="color:var(--muted);padding:20px;text-align:center;">No timeseries data available for this entity type.</p>';
      return;
    }}

    const topEntities = data.sort((a, b) => b.total - a.total).slice(0, topN);
    const traces = topEntities.map((ent, idx) => ({{
      x: allYears,
      y: allYears.map(y => ent.yearly[y] || 0),
      name: ent.name,
      type: 'scatter',
      mode: 'lines+markers',
      line: {{ color: colors[idx % colors.length], width: 2 }},
      marker: {{ size: 6 }}
    }}));

    const layout = {{
      height: 400,
      margin: {{ l: 50, r: 20, t: 20, b: 50 }},
      xaxis: {{ title: 'Year', type: 'linear', dtick: 1 }},
      yaxis: {{ title: 'Mentions' }},
      hovermode: 'x unified',
      showlegend: true,
      legend: {{ orientation: 'v', x: 1.02, y: 1 }}
    }};

    Plotly.newPlot('ner-timeseries-plot', traces, layout, {{ responsive: true }});
  }}

  topNSelect.addEventListener('change', renderTimeseries);
  typeSelect.addEventListener('change', renderTimeseries);
  renderTimeseries();
}})();
</script>
"""

        return f"""{stats}
{corrections_html}
{timeseries_html}
<div class="controls">
  <label>Entity Type
    <select id="ner-type-filter">
      <option value="all">All types</option>
    </select>
  </label>
  <label>Search
    <input type="text" id="ner-search" placeholder="Filter by name..." style="padding:6px 10px;border-radius:8px;border:1px solid #e7e1d8;">
  </label>
  <label>Sort by
    <select id="ner-sort">
      <option value="count_desc" selected>Count (high to low)</option>
      <option value="count_asc">Count (low to high)</option>
      <option value="text_asc">Name (A-Z)</option>
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
  <div class="ner-header" style="display:grid;grid-template-columns:120px 2fr 80px 1fr 1fr 80px 100px;gap:12px;padding:10px 14px;background:#f7f3ed;font-weight:600;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.04em;">
    <div>Type</div>
    <div>Entity</div>
    <div>Count</div>
    <div>Top Frame</div>
    <div>Attribute</div>
    <div>Aliases</div>
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
      if (searchVal && !row.canonical_name.toLowerCase().includes(searchVal)) return false;
      return true;
    }});

    if (sortVal === 'count_desc') {{
      filteredRows.sort((a, b) => b.count - a.count);
    }} else if (sortVal === 'count_asc') {{
      filteredRows.sort((a, b) => a.count - b.count);
    }} else if (sortVal === 'text_asc') {{
      filteredRows.sort((a, b) => a.canonical_name.localeCompare(b.canonical_name));
    }} else if (sortVal === 'type_asc') {{
      filteredRows.sort((a, b) => a.type.localeCompare(b.type) || b.count - a.count);
    }}

    page = 0;
  }}

  function renderArticleLinks(articles) {{
    if (!articles || articles.length === 0) {{
      return '<span style="color:var(--muted);font-size:0.85rem;">—</span>';
    }}
    const maxVisible = 3;
    const visibleLinks = articles.slice(0, maxVisible).map(url =>
      `<a href="${{escapeHtml(url)}}" target="_blank" rel="noopener noreferrer" class="link-icon" title="${{escapeHtml(url)}}" style="text-decoration:none;margin-right:4px;">🔗</a>`
    ).join('');

    if (articles.length <= maxVisible) {{
      return `<div>${{visibleLinks}}</div>`;
    }}

    const moreCount = articles.length - maxVisible;
    return `<div>${{visibleLinks}}<span style="color:var(--muted);font-size:0.85rem;">+${{moreCount}}</span></div>`;
  }}

  function render() {{
    const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
    page = Math.max(0, Math.min(page, totalPages - 1));
    const start = page * pageSize;
    const slice = filteredRows.slice(start, start + pageSize);
    nerBody.innerHTML = slice.map(row => {{
      const typeColor = {{
        'PERSON': '#3b82f6',
        'ORG': '#10b981',
        'EVENT': '#ef4444',
        'FAC': '#06b6d4',
        'PRODUCT': '#84cc16'
      }}[row.type] || '#6b7280';
      const correctionBadge = row.wasTypeCorrection ? '<span style="font-size:0.75rem;color:#f59e0b;margin-left:4px;" title="Type was corrected">\u26a0\ufe0f</span>' : '';
      const aliasesText = row.aliasCount > 1 ? `${{row.aliasCount}} variants` : '1 name';
      const allAliasesText = row.allAliases ? row.allAliases.join(', ') : row.aliases;
      const articlesHtml = renderArticleLinks(row.articles || []);
      return `
        <div style="display:grid;grid-template-columns:120px 2fr 80px 1fr 1fr 80px 100px;gap:12px;padding:10px 14px;border-top:1px solid #efe9df;background:#fff;font-size:0.95rem;">
          <div><span class="pill" style="background:${{typeColor}}22;color:${{typeColor}};border-color:${{typeColor}}44;">${{escapeHtml(row.type)}}${{correctionBadge}}</span></div>
          <div style="font-weight:500;" title="${{escapeHtml(allAliasesText)}}">${{escapeHtml(row.canonical_name)}}</div>
          <div style="color:var(--muted);">${{row.count}}</div>
          <div style="color:var(--muted);font-size:0.9rem;">${{escapeHtml(row.topFrame)}}</div>
          <div style="color:var(--muted);font-size:0.85rem;font-style:italic;">${{escapeHtml(row.orgAttr || '—')}}</div>
          <div style="color:var(--muted);font-size:0.85rem;" title="${{escapeHtml(allAliasesText)}}">${{aliasesText}}</div>
          <div>${{articlesHtml}}</div>
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

    def _render_ner_entities(self, ctx: ReportContext) -> str:
        """Render NER entities as an interactive table with filters."""
        # Try consolidated first, fall back to raw
        consolidated = self._resolve_ner_consolidated()

        if consolidated and consolidated.entities:
            return self._render_consolidated_entities(ctx, consolidated)

        # Fallback to raw NER results
        if not ctx.ner_result or not ctx.ner_result.documents:
            return "<p class=\"subtitle\">No named entities extracted. Enable NER in config with <code>ner.enabled: true</code>.</p>"

        # Build lookup from doc_id to URL and year from frame_classifications
        doc_to_url: Dict[str, str] = {}
        doc_to_year: Dict[str, int] = {}
        for doc_class in ctx.frame_classifications or []:
            payload = doc_class.payload if isinstance(doc_class.payload, dict) else {}
            doc_id = str(payload.get("doc_id", "")).strip()
            url = str(payload.get("url", "")).strip()
            published_at = payload.get("published_at", "")
            if doc_id and url:
                doc_to_url[doc_id] = url
            if doc_id and published_at:
                try:
                    year = int(str(published_at)[:4])
                    if 1900 < year < 2100:
                        doc_to_year[doc_id] = year
                except (ValueError, TypeError):
                    pass

        # Aggregate entities by (type, text) -> count
        entity_counts: Dict[tuple, int] = {}
        entity_frames: Dict[tuple, Dict[str, int]] = {}  # Track which frames each entity appears in
        entity_articles: Dict[tuple, Dict[str, str]] = {}  # Track doc_id -> url for each entity
        entity_by_year: Dict[tuple, Dict[int, int]] = {}  # Track (type, text) -> year -> count

        for doc in ctx.ner_result.documents:
            doc_url = doc_to_url.get(doc.doc_id, "")
            doc_year = doc_to_year.get(doc.doc_id)
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
                    # Track by year for timeseries
                    if doc_year:
                        if key not in entity_by_year:
                            entity_by_year[key] = {}
                        entity_by_year[key][doc_year] = entity_by_year[key].get(doc_year, 0) + 1

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
            # Get yearly counts for timeseries
            yearly = entity_by_year.get((ent_type, ent_text), {})
            rows.append({
                "type": ent_type,
                "text": ent_text,
                "count": count,
                "topFrame": top_frame,
                "articles": articles,
                "yearly": yearly,
            })

        # Build timeseries data for PERSON and ORG entities
        all_years = sorted(set(y for r in rows for y in r.get("yearly", {}).keys()))
        person_timeseries = [
            {"name": r["text"], "total": r["count"], "yearly": r["yearly"]}
            for r in rows if r["type"] == "PERSON" and r["yearly"]
        ]
        org_timeseries = [
            {"name": r["text"], "total": r["count"], "yearly": r["yearly"]}
            for r in rows if r["type"] == "ORG" and r["yearly"]
        ]

        # Get unique entity types for filter
        entity_types = sorted(set(r["type"] for r in rows))

        payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        entity_types_json = json.dumps(entity_types, ensure_ascii=False)
        person_ts_json = json.dumps(person_timeseries, ensure_ascii=False).replace("</", "<\\/")
        org_ts_json = json.dumps(org_timeseries, ensure_ascii=False).replace("</", "<\\/")
        all_years_json = json.dumps(all_years, ensure_ascii=False)

        language_label = ctx.ner_result.language
        if isinstance(language_label, list):
            language_label = ", ".join(language_label)

        stats = (
            f"<p class=\"subtitle\">"
            f"{len(rows)} unique entities from {ctx.ner_result.n_chunks} chunks "
            f"(threshold: {ctx.ner_result.frame_threshold}, language: {language_label})"
            f"</p>"
        )

        # Timeseries charts section
        timeseries_html = ""
        if person_timeseries or org_timeseries:
            timeseries_html = f"""
<div class="plot-section" style="margin-bottom:24px;">
  <h3>Entity Mentions Over Time</h3>
  <div class="controls" style="margin-bottom:12px;">
    <label>Show top
      <select id="ner-ts-topn">
        <option value="5">5</option>
        <option value="10" selected>10</option>
        <option value="20">20</option>
        <option value="50">50</option>
      </select>
    </label>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
    <div>
      <h4 style="margin:0 0 8px;font-family:'Space Grotesk',sans-serif;font-size:1rem;">Persons</h4>
      <div id="ner-person-chart" style="width:100%;height:350px;"></div>
    </div>
    <div>
      <h4 style="margin:0 0 8px;font-family:'Space Grotesk',sans-serif;font-size:1rem;">Organizations</h4>
      <div id="ner-org-chart" style="width:100%;height:350px;"></div>
    </div>
  </div>
</div>
<script id="ner-person-ts-data" type="application/json">{person_ts_json}</script>
<script id="ner-org-ts-data" type="application/json">{org_ts_json}</script>
<script id="ner-all-years" type="application/json">{all_years_json}</script>
<script>
(function() {{
  const personData = JSON.parse(document.getElementById('ner-person-ts-data').textContent);
  const orgData = JSON.parse(document.getElementById('ner-org-ts-data').textContent);
  const allYears = JSON.parse(document.getElementById('ner-all-years').textContent);
  const topNSelect = document.getElementById('ner-ts-topn');

  const colors = ['#1E3D58', '#057D9F', '#F18F01', '#A23B72', '#6C63FF', '#3A7D44', '#F45B69', '#0E7C7B', '#F2A541', '#8B5CF6',
                  '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#ef4444', '#84cc16', '#a855f7', '#14b8a6', '#f97316'];

  function renderChart(containerId, data, topN) {{
    const container = document.getElementById(containerId);
    if (!container || !data || data.length === 0) {{
      if (container) container.innerHTML = '<p style="color:var(--muted);font-size:0.9rem;">No data available</p>';
      return;
    }}

    // Sort by total count and take top N
    const sorted = [...data].sort((a, b) => b.total - a.total).slice(0, topN);
    if (sorted.length === 0 || allYears.length === 0) {{
      container.innerHTML = '<p style="color:var(--muted);font-size:0.9rem;">No data available</p>';
      return;
    }}

    const traces = sorted.map((entity, idx) => {{
      const yValues = allYears.map(year => entity.yearly[year] || 0);
      return {{
        x: allYears,
        y: yValues,
        mode: 'lines+markers',
        name: entity.name,
        line: {{ color: colors[idx % colors.length], width: 2 }},
        marker: {{ size: 6 }},
        hovertemplate: '%{{y}} mentions<extra>%{{fullData.name}}</extra>'
      }};
    }});

    // Add annotations for labels at end of lines
    const annotations = sorted.map((entity, idx) => {{
      const lastYear = allYears[allYears.length - 1];
      const lastValue = entity.yearly[lastYear] || 0;
      return {{
        x: lastYear,
        y: lastValue,
        xanchor: 'left',
        yanchor: 'middle',
        text: entity.name.length > 15 ? entity.name.slice(0, 15) + '...' : entity.name,
        font: {{ size: 10, color: colors[idx % colors.length] }},
        showarrow: false,
        xshift: 5
      }};
    }});

    const layout = {{
      margin: {{ l: 50, r: 120, t: 20, b: 40 }},
      xaxis: {{ title: '', tickmode: 'linear', dtick: 1 }},
      yaxis: {{ title: 'Mentions', rangemode: 'tozero' }},
      showlegend: false,
      annotations: annotations,
      hovermode: 'x unified'
    }};

    Plotly.newPlot(containerId, traces, layout, {{ displayModeBar: false, responsive: true }});
  }}

  function updateCharts() {{
    const topN = parseInt(topNSelect.value, 10);
    renderChart('ner-person-chart', personData, topN);
    renderChart('ner-org-chart', orgData, topN);
  }}

  topNSelect.addEventListener('change', updateCharts);
  updateCharts();
}})();
</script>
"""

        return f"""{stats}
{timeseries_html}
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

    def _render_claims_analysis(self, ctx: ReportContext) -> str:
        """Render claims analysis section with visualizations."""
        if not ctx.claims_result:
            return "<p class=\"subtitle\">No claims analysis available. Enable with <code>claims_analysis.enabled: true</code>.</p>"

        claims = ctx.claims_result.claims
        statements = ctx.claims_result.statements
        agreements = ctx.claims_result.agreements

        if not claims.claims or not statements.statements or not agreements.agreements:
            return "<p class=\"subtitle\">Claims analysis data incomplete.</p>"

        # Build statement lookup for doc_id
        stmt_lookup = {s.statement_id: s for s in statements.statements}

        # Build doc_id to metadata (date, domain) from frame_classifications
        from urllib.parse import urlparse

        doc_to_date: Dict[str, str] = {}
        doc_to_domain: Dict[str, str] = {}
        for doc_class in ctx.frame_classifications or []:
            # Access fields directly from doc_class (not via payload)
            payload = doc_class.payload if isinstance(doc_class.payload, dict) else {}
            doc_id = str(getattr(doc_class, "doc_id", "") or payload.get("doc_id", "")).strip()
            published_at = getattr(doc_class, "published_at", "") or payload.get("published_at", "")
            url = getattr(doc_class, "url", "") or payload.get("url", "")

            # Extract domain from URL
            domain = ""
            if url:
                try:
                    parsed = urlparse(str(url))
                    domain = parsed.netloc.replace("www.", "")
                except Exception:
                    pass

            if doc_id:
                if published_at:
                    doc_to_date[doc_id] = str(published_at)[:10]  # YYYY-MM-DD
                if domain:
                    doc_to_domain[doc_id] = domain

        # Filter out irrelevant agreements and build data for charts
        valid_labels = ["supports", "opposes", "neutral"]
        agreement_data = []
        for agr in agreements.agreements:
            if agr.label not in valid_labels:
                continue
            stmt = stmt_lookup.get(agr.statement_id)
            if not stmt:
                continue
            date = doc_to_date.get(stmt.doc_id, "")
            domain = doc_to_domain.get(stmt.doc_id, "unknown")
            agreement_data.append({
                "statement_id": agr.statement_id,
                "claim_id": agr.claim_id,
                "label": agr.label,
                "score": agr.score,
                "date": date,
                "domain": domain,
                "year_month": date[:7] if date else "",  # YYYY-MM
            })

        if not agreement_data:
            return "<p class=\"subtitle\">No valid agreements (supports/opposes/neutral) found.</p>"

        # Build claim labels lookup
        claim_labels = {c.claim_id: c.text[:60] + ("..." if len(c.text) > 60 else "") for c in claims.claims}

        # Stats summary
        label_counts = {"supports": 0, "opposes": 0, "neutral": 0}
        for d in agreement_data:
            label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
        total_valid = sum(label_counts.values())

        stats_html = f"""
<div class="plot-section">
  <h3>Summary</h3>
  <p class="subtitle">
    {len(claims.claims)} claims, {len(statements.statements)} statements, {len(agreements.agreements)} agreement scores
    ({total_valid} valid: {label_counts['supports']} supports, {label_counts['opposes']} opposes, {label_counts['neutral']} neutral)
  </p>
  <div class="grid" style="margin-top:16px;">
"""
        for claim in claims.claims:
            stats_html += f"""
    <div class="frame-card">
      <h3>{html.escape(claim.claim_id)}</h3>
      <p>{html.escape(claim.text)}</p>
    </div>
"""
        stats_html += """
  </div>
</div>
"""

        # Prepare data for JavaScript charts
        payload = json.dumps(agreement_data, ensure_ascii=False).replace("</", "<\\/")
        claim_labels_json = json.dumps(claim_labels, ensure_ascii=False).replace("</", "<\\/")

        charts_html = f"""
<div class="plot-section">
  <h3>Agreement Distribution Over Time</h3>
  <div class="controls" style="margin-bottom:12px;">
    <label>Claim
      <select id="claims-claim-filter">
        <option value="all" selected>All claims</option>
      </select>
    </label>
    <label>Aggregation
      <select id="claims-time-agg">
        <option value="month" selected>Monthly</option>
        <option value="year">Yearly</option>
      </select>
    </label>
  </div>
  <div id="claims-time-chart" style="width:100%;height:400px;"></div>
</div>

<div class="plot-section">
  <h3>Agreement Distribution by Domain</h3>
  <div class="controls" style="margin-bottom:12px;">
    <label>Claim
      <select id="claims-domain-claim-filter">
        <option value="all" selected>All claims</option>
      </select>
    </label>
    <label>Top N domains
      <select id="claims-domain-topn">
        <option value="10" selected>10</option>
        <option value="15">15</option>
        <option value="20">20</option>
        <option value="all">All</option>
      </select>
    </label>
  </div>
  <div id="claims-domain-chart" style="width:100%;height:400px;"></div>
</div>

<script id="claims-data" type="application/json">{payload}</script>
<script id="claims-labels" type="application/json">{claim_labels_json}</script>
<script>
(function() {{
  const allData = JSON.parse(document.getElementById('claims-data').textContent);
  const claimLabels = JSON.parse(document.getElementById('claims-labels').textContent);

  const claimFilter = document.getElementById('claims-claim-filter');
  const domainClaimFilter = document.getElementById('claims-domain-claim-filter');
  const timeAgg = document.getElementById('claims-time-agg');
  const domainTopN = document.getElementById('claims-domain-topn');

  // Populate claim filters
  Object.keys(claimLabels).forEach(claimId => {{
    const opt1 = document.createElement('option');
    opt1.value = claimId;
    opt1.textContent = claimLabels[claimId];
    claimFilter.appendChild(opt1);

    const opt2 = document.createElement('option');
    opt2.value = claimId;
    opt2.textContent = claimLabels[claimId];
    domainClaimFilter.appendChild(opt2);
  }});

  const colors = {{
    supports: '#2c8f5b',
    opposes: '#c1443a',
    neutral: '#6b7280'
  }};

  function renderTimeChart() {{
    const selectedClaim = claimFilter.value;
    const aggType = timeAgg.value;

    let filtered = allData;
    if (selectedClaim !== 'all') {{
      filtered = allData.filter(d => d.claim_id === selectedClaim);
    }}

    // Group by time period
    const grouped = {{}};
    filtered.forEach(d => {{
      const period = aggType === 'year' ? d.date.slice(0, 4) : d.year_month;
      if (!period) return;
      if (!grouped[period]) grouped[period] = {{ supports: 0, opposes: 0, neutral: 0 }};
      grouped[period][d.label]++;
    }});

    const existingPeriods = Object.keys(grouped).sort();
    if (existingPeriods.length === 0) {{
      document.getElementById('claims-time-chart').innerHTML = '<p style="color:var(--muted);text-align:center;padding:40px;">No data for selected filters</p>';
      return;
    }}

    // Fill gaps: generate all periods between min and max
    function generatePeriods(start, end, isYear) {{
      const result = [];
      if (isYear) {{
        for (let y = parseInt(start); y <= parseInt(end); y++) {{
          result.push(String(y));
        }}
      }} else {{
        // Monthly: YYYY-MM format
        let [sy, sm] = start.split('-').map(Number);
        const [ey, em] = end.split('-').map(Number);
        while (sy < ey || (sy === ey && sm <= em)) {{
          result.push(`${{sy}}-${{String(sm).padStart(2, '0')}}`);
          sm++;
          if (sm > 12) {{ sm = 1; sy++; }}
        }}
      }}
      return result;
    }}

    const periods = generatePeriods(existingPeriods[0], existingPeriods[existingPeriods.length - 1], aggType === 'year');
    // Ensure all periods have an entry (fill gaps with zeros)
    periods.forEach(p => {{
      if (!grouped[p]) grouped[p] = {{ supports: 0, opposes: 0, neutral: 0 }};
    }});

    // Calculate proportions for stacked area
    const traces = ['supports', 'opposes', 'neutral'].map(label => ({{
      x: periods,
      y: periods.map(p => {{
        const total = grouped[p].supports + grouped[p].opposes + grouped[p].neutral;
        return total > 0 ? (grouped[p][label] / total) * 100 : 0;
      }}),
      name: label.charAt(0).toUpperCase() + label.slice(1),
      type: 'scatter',
      mode: 'lines',
      stackgroup: 'one',
      fillcolor: colors[label],
      line: {{ color: colors[label], width: 0.5 }}
    }}));

    const layout = {{
      height: 380,
      margin: {{ l: 50, r: 20, t: 20, b: 60 }},
      xaxis: {{ title: aggType === 'year' ? 'Year' : 'Month', tickangle: -45 }},
      yaxis: {{ title: 'Share (%)', range: [0, 100] }},
      hovermode: 'x unified',
      showlegend: true,
      legend: {{ orientation: 'h', y: -0.2 }}
    }};

    Plotly.newPlot('claims-time-chart', traces, layout, {{ responsive: true }});
  }}

  function renderDomainChart() {{
    const selectedClaim = domainClaimFilter.value;
    const topNVal = domainTopN.value;

    let filtered = allData;
    if (selectedClaim !== 'all') {{
      filtered = allData.filter(d => d.claim_id === selectedClaim);
    }}

    // Group by domain
    const grouped = {{}};
    filtered.forEach(d => {{
      const domain = d.domain || 'unknown';
      if (!grouped[domain]) grouped[domain] = {{ supports: 0, opposes: 0, neutral: 0, total: 0 }};
      grouped[domain][d.label]++;
      grouped[domain].total++;
    }});

    // Sort by total and take top N
    let domains = Object.keys(grouped).sort((a, b) => grouped[b].total - grouped[a].total);
    if (topNVal !== 'all') {{
      domains = domains.slice(0, parseInt(topNVal, 10));
    }}

    if (domains.length === 0) {{
      document.getElementById('claims-domain-chart').innerHTML = '<p style="color:var(--muted);text-align:center;padding:40px;">No data for selected filters</p>';
      return;
    }}

    // Create grouped bar chart
    const traces = ['supports', 'opposes', 'neutral'].map(label => ({{
      x: domains,
      y: domains.map(d => {{
        const total = grouped[d].total;
        return total > 0 ? (grouped[d][label] / total) * 100 : 0;
      }}),
      name: label.charAt(0).toUpperCase() + label.slice(1),
      type: 'bar',
      marker: {{ color: colors[label] }}
    }}));

    const layout = {{
      height: 380,
      margin: {{ l: 50, r: 20, t: 20, b: 120 }},
      barmode: 'stack',
      xaxis: {{ title: '', tickangle: -45 }},
      yaxis: {{ title: 'Share (%)', range: [0, 100] }},
      hovermode: 'x unified',
      showlegend: true,
      legend: {{ orientation: 'h', y: -0.35 }}
    }};

    Plotly.newPlot('claims-domain-chart', traces, layout, {{ responsive: true }});
  }}

  claimFilter.addEventListener('change', renderTimeChart);
  timeAgg.addEventListener('change', renderTimeChart);
  domainClaimFilter.addEventListener('change', renderDomainChart);
  domainTopN.addEventListener('change', renderDomainChart);

  renderTimeChart();
  renderDomainChart();
}})();
</script>
"""

        return stats_html + charts_html

    def _render_dna_network(self, ctx: ReportContext) -> str:
        """Render Discourse Network Analysis (DNA) section with actor-actor network."""
        dna_result = ctx.dna_result

        if not dna_result:
            if not ctx.claims_result:
                return '<p class="subtitle">DNA requires claims analysis. Enable with <code>claims_analysis.enabled: true</code>.</p>'
            return '<p class="subtitle">DNA network analysis not available. Enable with <code>claims_analysis.dna.enabled: true</code>.</p>'

        if len(dna_result.nodes) < 2:
            return '<p class="subtitle">Insufficient data for network analysis. Need at least 2 actors with 2+ statements each.</p>'

        # Get node positions
        positions = get_network_positions(dna_result, layout="spring")

        # Get plots directory for export
        plots_dir = getattr(self.paths, "plots_dir", None)
        export_path = Path(plots_dir) / "dna_network.png" if plots_dir else None

        # Build network visualization
        network_html = plot_dna_network(dna_result, positions, export_path=export_path)

        # Build lookup for claim labels
        claim_labels = {}
        if ctx.claims_result and ctx.claims_result.claims:
            claim_labels = {
                c.claim_id: c.text[:50] + ("..." if len(c.text) > 50 else "")
                for c in ctx.claims_result.claims.claims
            }

        # Build metrics tables data
        node_lookup = {n.entity_id: n for n in dna_result.nodes}

        # Prepare data for JavaScript
        nodes_data = [
            {
                "id": n.entity_id,
                "name": n.canonical_name,
                "type": n.entity_type,
                "statements": n.statement_count,
                "community": dna_result.communities.get(n.entity_id, 0),
                "centrality": round(dna_result.centrality_scores.get(n.entity_id, 0), 3),
            }
            for n in dna_result.nodes
        ]

        edges_data = [
            {
                "actor1": node_lookup[e.actor1_id].canonical_name,
                "actor2": node_lookup[e.actor2_id].canonical_name,
                "relationship": e.relationship,
                "weight": e.weight,
                "claims": [claim_labels.get(c, c) for c in e.shared_claims],
            }
            for e in dna_result.edges
        ]

        nodes_json = json.dumps(nodes_data, ensure_ascii=False).replace("</", "<\\/")
        edges_json = json.dumps(edges_data, ensure_ascii=False).replace("</", "<\\/")

        # Stats summary
        n_communities = len(set(dna_result.communities.values()))
        ally_edges = sum(1 for e in dna_result.edges if e.relationship == "allies")
        enemy_edges = sum(1 for e in dna_result.edges if e.relationship == "enemies")

        stats_html = f"""
<div class="plot-section">
  <h3>Network Overview</h3>
  <p class="subtitle">
    {len(dna_result.nodes)} actors, {ally_edges} alliance connections,
    {enemy_edges} conflict connections, {n_communities} communities detected
  </p>
</div>
"""

        # Interactive network chart section
        chart_html = f"""
<div class="plot-section">
  <h3>Actor-Actor Network</h3>
  <p class="subtitle" style="font-size:0.9rem;margin-bottom:12px;">
    <span style="display:inline-block;width:12px;height:12px;background:rgba(44,143,91,0.6);border-radius:2px;margin-right:4px;"></span>Allies (same position on claims)
    <span style="display:inline-block;width:12px;height:12px;background:rgba(193,68,58,0.6);border-radius:2px;margin:0 4px 0 16px;"></span>Enemies (opposing positions)
  </p>
  {network_html}
</div>
"""

        # Metrics tables with JavaScript interactivity
        tables_html = f"""
<div class="plot-section">
  <h3>Network Metrics</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
    <div>
      <h4 style="margin:0 0 12px;font-family:'Space Grotesk',sans-serif;">Top Alliances</h4>
      <div id="dna-allies-table"></div>
    </div>
    <div>
      <h4 style="margin:0 0 12px;font-family:'Space Grotesk',sans-serif;">Top Conflicts</h4>
      <div id="dna-enemies-table"></div>
    </div>
  </div>
</div>

<div class="plot-section">
  <h3>Actor Centrality</h3>
  <div class="controls" style="margin-bottom:12px;">
    <label>Sort by
      <select id="dna-actor-sort">
        <option value="centrality" selected>Centrality</option>
        <option value="statements">Statements</option>
        <option value="name">Name</option>
      </select>
    </label>
    <label>Community
      <select id="dna-community-filter">
        <option value="all">All communities</option>
      </select>
    </label>
  </div>
  <div id="dna-actors-table"></div>
</div>

<script id="dna-nodes" type="application/json">{nodes_json}</script>
<script id="dna-edges" type="application/json">{edges_json}</script>
<script>
(function() {{
  const nodes = JSON.parse(document.getElementById('dna-nodes').textContent);
  const edges = JSON.parse(document.getElementById('dna-edges').textContent);

  const communityColors = ['#E63946', '#2A9D8F', '#E9C46A', '#264653', '#F4A261', '#8338EC', '#06D6A0', '#EF476F', '#118AB2', '#FFD166'];

  function escapeHtml(text) {{
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }}

  // Render ally pairs table
  const allyEdges = edges.filter(e => e.relationship === 'allies').sort((a, b) => b.weight - a.weight).slice(0, 10);
  document.getElementById('dna-allies-table').innerHTML = allyEdges.length === 0
    ? '<p class="subtitle">No alliances found</p>'
    : `
    <div style="border-radius:12px;overflow:hidden;border:1px solid #efe9df;">
      ${{allyEdges.map(e => `
        <div style="display:grid;grid-template-columns:1fr 1fr 80px;gap:8px;padding:10px 14px;border-top:1px solid #efe9df;background:#fff;">
          <div style="font-weight:500;">${{escapeHtml(e.actor1)}}</div>
          <div style="font-weight:500;">${{escapeHtml(e.actor2)}}</div>
          <div style="color:#2c8f5b;font-weight:600;">${{e.weight}} claim${{e.weight > 1 ? 's' : ''}}</div>
        </div>
      `).join('')}}
    </div>
  `;

  // Render enemy pairs table
  const enemyEdges = edges.filter(e => e.relationship === 'enemies').sort((a, b) => b.weight - a.weight).slice(0, 10);
  document.getElementById('dna-enemies-table').innerHTML = enemyEdges.length === 0
    ? '<p class="subtitle">No conflicts found</p>'
    : `
    <div style="border-radius:12px;overflow:hidden;border:1px solid #efe9df;">
      ${{enemyEdges.map(e => `
        <div style="display:grid;grid-template-columns:1fr 1fr 80px;gap:8px;padding:10px 14px;border-top:1px solid #efe9df;background:#fff;">
          <div style="font-weight:500;">${{escapeHtml(e.actor1)}}</div>
          <div style="font-weight:500;">${{escapeHtml(e.actor2)}}</div>
          <div style="color:#c1443a;font-weight:600;">${{e.weight}} claim${{e.weight > 1 ? 's' : ''}}</div>
        </div>
      `).join('')}}
    </div>
  `;

  // Populate community filter
  const communities = [...new Set(nodes.map(n => n.community))].sort((a, b) => a - b);
  const communityFilter = document.getElementById('dna-community-filter');
  communities.forEach(c => {{
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = `Community ${{c + 1}}`;
    communityFilter.appendChild(opt);
  }});

  // Render actors table
  function renderActorsTable() {{
    const sortBy = document.getElementById('dna-actor-sort').value;
    const communityVal = communityFilter.value;

    let filtered = nodes;
    if (communityVal !== 'all') {{
      filtered = nodes.filter(n => n.community === parseInt(communityVal));
    }}

    if (sortBy === 'centrality') {{
      filtered = [...filtered].sort((a, b) => b.centrality - a.centrality);
    }} else if (sortBy === 'statements') {{
      filtered = [...filtered].sort((a, b) => b.statements - a.statements);
    }} else {{
      filtered = [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    }}

    document.getElementById('dna-actors-table').innerHTML = `
      <div style="border-radius:12px;overflow:hidden;border:1px solid #efe9df;">
        <div style="display:grid;grid-template-columns:2fr 100px 80px 100px 80px;gap:8px;padding:10px 14px;background:#f7f3ed;font-weight:600;font-size:0.9rem;">
          <div>Actor</div>
          <div>Type</div>
          <div>Statements</div>
          <div>Community</div>
          <div>Centrality</div>
        </div>
        ${{filtered.slice(0, 30).map(n => {{
          const color = communityColors[n.community % communityColors.length];
          return `
            <div style="display:grid;grid-template-columns:2fr 100px 80px 100px 80px;gap:8px;padding:10px 14px;border-top:1px solid #efe9df;background:#fff;">
              <div style="font-weight:500;">${{escapeHtml(n.name)}}</div>
              <div><span class="pill" style="font-size:0.8rem;">${{n.type}}</span></div>
              <div style="color:var(--muted);">${{n.statements}}</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{color}};margin-right:6px;"></span>C${{n.community + 1}}</div>
              <div style="color:var(--muted);">${{n.centrality.toFixed(2)}}</div>
            </div>
          `;
        }}).join('')}}
      </div>
    `;
  }}

  document.getElementById('dna-actor-sort').addEventListener('change', renderActorsTable);
  communityFilter.addEventListener('change', renderActorsTable);
  renderActorsTable();
}})();
</script>
"""

        return stats_html + chart_html + tables_html

    def _render_coalition_analysis(self, ctx: ReportContext) -> str:
        """Render multi-dimensional coalition analysis based on claim positions."""
        coalition_result = ctx.coalition_result

        if not coalition_result:
            if not ctx.claims_result:
                return '<p class="subtitle">Coalition analysis requires claims analysis. Enable with <code>claims_analysis.enabled: true</code>.</p>'
            return '<p class="subtitle">Coalition analysis not available. Enable with <code>claims_analysis.dna.enabled: true</code>.</p>'

        # Build claim labels
        claim_labels = {}
        if ctx.claims_result and ctx.claims_result.claims:
            claim_labels = {
                c.claim_id: c.text[:40] + ("..." if len(c.text) > 40 else "")
                for c in ctx.claims_result.claims.claims
            }

        # Prepare data for JavaScript
        actors_data = []
        for i, actor_id in enumerate(coalition_result.actors):
            name = coalition_result.actor_names.get(actor_id, actor_id)
            x, y = coalition_result.positions_2d.get(actor_id, (0, 0))
            cluster = coalition_result.best_clustering.labels.get(actor_id, 0)
            actors_data.append({
                "id": actor_id,
                "name": name,
                "x": round(x, 4),
                "y": round(y, 4),
                "cluster": cluster,
            })

        # Prepare clustering comparison data
        clustering_data = [
            {
                "n": cr.n_clusters,
                "silhouette": round(cr.silhouette_score, 3),
                "sizes": list(cr.cluster_sizes.values()),
            }
            for cr in coalition_result.clustering_results
        ]

        # Prepare cluster profiles for heatmap
        claims_list = [
            {"id": cid, "label": claim_labels.get(cid, cid)}
            for cid in coalition_result.claims
        ]

        # Get position matrix as nested list for JS
        position_matrix_data = coalition_result.position_matrix.tolist()

        actors_json = json.dumps(actors_data, ensure_ascii=False).replace("</", "<\\/")
        clustering_json = json.dumps(clustering_data, ensure_ascii=False).replace("</", "<\\/")
        claims_json = json.dumps(claims_list, ensure_ascii=False).replace("</", "<\\/")
        matrix_json = json.dumps(position_matrix_data).replace("</", "<\\/")

        best = coalition_result.best_clustering

        # Stats summary
        stats_html = f"""
<div class="plot-section">
  <h3>Overview</h3>
  <p class="subtitle">
    {len(coalition_result.actors)} actors positioned across {len(coalition_result.claims)} claims.
    Best clustering: <strong>N={best.n_clusters}</strong> (silhouette score: {best.silhouette_score:.3f}).
    Projection: {coalition_result.projection_method.upper()}.
  </p>
</div>
"""

        # PCA scatter plot + silhouette comparison
        charts_html = f"""
<div class="plot-section">
  <h3>Actor Positions in Claim Space</h3>
  <p class="subtitle" style="font-size:0.9rem;margin-bottom:12px;">
    Each actor is a point. Position determined by their stance (support/oppose/neutral) on each claim.
    Colors represent coalition clusters.
  </p>
  <div id="coalition-pca-chart" class="plotly-chart" style="min-height:450px;"></div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
  <div class="plot-section">
    <h3>Clustering Quality by N</h3>
    <p class="subtitle" style="font-size:0.9rem;margin-bottom:12px;">
      Silhouette score measures cluster cohesion (-1 to 1, higher is better).
    </p>
    <div id="coalition-silhouette-chart" class="plotly-chart" style="min-height:300px;"></div>
  </div>
  <div class="plot-section">
    <h3>Cluster Sizes (N={best.n_clusters})</h3>
    <p class="subtitle" style="font-size:0.9rem;margin-bottom:12px;">
      Number of actors in each coalition cluster.
    </p>
    <div id="coalition-sizes-chart" class="plotly-chart" style="min-height:300px;"></div>
  </div>
</div>
"""

        # Actors by cluster table
        tables_html = f"""
<div class="plot-section">
  <h3>Actors by Coalition</h3>
  <div class="controls" style="margin-bottom:12px;">
    <label>Cluster
      <select id="coalition-cluster-filter">
        <option value="all">All clusters</option>
      </select>
    </label>
  </div>
  <div id="coalition-actors-table"></div>
</div>

<script id="coalition-actors" type="application/json">{actors_json}</script>
<script id="coalition-clustering" type="application/json">{clustering_json}</script>
<script id="coalition-claims" type="application/json">{claims_json}</script>
<script id="coalition-matrix" type="application/json">{matrix_json}</script>
<script>
(function() {{
  const actors = JSON.parse(document.getElementById('coalition-actors').textContent);
  const clusteringData = JSON.parse(document.getElementById('coalition-clustering').textContent);
  const claims = JSON.parse(document.getElementById('coalition-claims').textContent);
  const matrix = JSON.parse(document.getElementById('coalition-matrix').textContent);

  const clusterColors = ['#E63946', '#2A9D8F', '#E9C46A', '#264653', '#F4A261', '#8338EC', '#06D6A0', '#EF476F', '#118AB2', '#FFD166'];
  const bestN = {best.n_clusters};

  function escapeHtml(text) {{
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }}

  // PCA scatter plot
  const traces = [];
  for (let c = 0; c < bestN; c++) {{
    const clusterActors = actors.filter(a => a.cluster === c);
    if (clusterActors.length > 0) {{
      traces.push({{
        x: clusterActors.map(a => a.x),
        y: clusterActors.map(a => a.y),
        mode: 'markers+text',
        type: 'scatter',
        name: `Coalition ${{c + 1}}`,
        text: clusterActors.map(a => a.name),
        textposition: 'top center',
        textfont: {{ size: 9 }},
        marker: {{
          size: 12,
          color: clusterColors[c % clusterColors.length],
          line: {{ width: 1, color: '#fff' }}
        }},
        hovertemplate: '<b>%{{text}}</b><br>Coalition %{{data.name}}<extra></extra>'
      }});
    }}
  }}

  Plotly.newPlot('coalition-pca-chart', traces, {{
    xaxis: {{ title: 'PC1', zeroline: true, zerolinecolor: '#ddd' }},
    yaxis: {{ title: 'PC2', zeroline: true, zerolinecolor: '#ddd' }},
    showlegend: true,
    legend: {{ orientation: 'h', y: -0.15 }},
    margin: {{ t: 20, b: 60 }},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)'
  }}, {{ responsive: true }});

  // Silhouette comparison chart
  Plotly.newPlot('coalition-silhouette-chart', [{{
    x: clusteringData.map(d => `N=${{d.n}}`),
    y: clusteringData.map(d => d.silhouette),
    type: 'bar',
    marker: {{
      color: clusteringData.map(d => d.n === bestN ? '#2A9D8F' : '#ccc')
    }},
    text: clusteringData.map(d => d.silhouette.toFixed(3)),
    textposition: 'outside',
    hovertemplate: 'N=%{{x}}<br>Silhouette: %{{y:.3f}}<extra></extra>'
  }}], {{
    xaxis: {{ title: 'Number of Clusters' }},
    yaxis: {{ title: 'Silhouette Score', range: [-0.2, 1] }},
    margin: {{ t: 30, b: 50 }},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)'
  }}, {{ responsive: true }});

  // Cluster sizes chart (for best N)
  const bestClustering = clusteringData.find(d => d.n === bestN);
  const sizeLabels = bestClustering.sizes.map((_, i) => `Coalition ${{i + 1}}`);
  Plotly.newPlot('coalition-sizes-chart', [{{
    x: sizeLabels,
    y: bestClustering.sizes,
    type: 'bar',
    marker: {{
      color: sizeLabels.map((_, i) => clusterColors[i % clusterColors.length])
    }},
    text: bestClustering.sizes,
    textposition: 'outside',
    hovertemplate: '%{{x}}: %{{y}} actors<extra></extra>'
  }}], {{
    xaxis: {{ title: '' }},
    yaxis: {{ title: 'Number of Actors' }},
    margin: {{ t: 30, b: 50 }},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)'
  }}, {{ responsive: true }});

  // Populate cluster filter
  const clusterFilter = document.getElementById('coalition-cluster-filter');
  for (let c = 0; c < bestN; c++) {{
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = `Coalition ${{c + 1}}`;
    clusterFilter.appendChild(opt);
  }}

  // Render actors table
  function renderActorsTable() {{
    const filterVal = clusterFilter.value;
    let filtered = actors;
    if (filterVal !== 'all') {{
      filtered = actors.filter(a => a.cluster === parseInt(filterVal));
    }}

    // Sort by cluster, then name
    filtered = [...filtered].sort((a, b) => {{
      if (a.cluster !== b.cluster) return a.cluster - b.cluster;
      return a.name.localeCompare(b.name);
    }});

    document.getElementById('coalition-actors-table').innerHTML = `
      <div style="border-radius:12px;overflow:hidden;border:1px solid #efe9df;">
        <div style="display:grid;grid-template-columns:2fr 120px;gap:8px;padding:10px 14px;background:#f7f3ed;font-weight:600;font-size:0.9rem;">
          <div>Actor</div>
          <div>Coalition</div>
        </div>
        ${{filtered.slice(0, 50).map(a => {{
          const color = clusterColors[a.cluster % clusterColors.length];
          return `
            <div style="display:grid;grid-template-columns:2fr 120px;gap:8px;padding:10px 14px;border-top:1px solid #efe9df;background:#fff;">
              <div style="font-weight:500;">${{escapeHtml(a.name)}}</div>
              <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{color}};margin-right:6px;"></span>Coalition ${{a.cluster + 1}}</div>
            </div>
          `;
        }}).join('')}}
      </div>
    `;
  }}

  clusterFilter.addEventListener('change', renderActorsTable);
  renderActorsTable();
}})();
</script>
"""

        return stats_html + charts_html + tables_html

    def _render_article_reader(self, ctx: ReportContext) -> str:
        """Render interactive article reader with entity and statement highlights."""
        from collections import defaultdict

        has_ner = ctx.ner_result is not None
        has_claims = ctx.claims_result is not None

        if not has_ner and not has_claims:
            return "<p class=\"subtitle\">No NER or claims data available for the article reader.</p>"

        # ── Doc metadata from frame classifications ──────────────────────────
        doc_meta: Dict[str, Dict] = {}
        if ctx.frame_classifications:
            for doc in ctx.frame_classifications:
                payload = doc.payload if isinstance(doc.payload, dict) else {}
                did = payload.get("doc_id", "")
                if did:
                    doc_meta[did] = {
                        "title": payload.get("title", ""),
                        "url": payload.get("url", ""),
                        "domain": payload.get("domain", ""),
                        "published_at": payload.get("published_at", ""),
                    }

        # ── Chunk index from NER ─────────────────────────────────────────────
        chunk_texts: Dict[str, str] = {}
        chunk_entities: Dict[str, list] = defaultdict(list)
        doc_chunks: Dict[str, list] = defaultdict(list)

        if has_ner:
            for doc_ent in ctx.ner_result.documents:
                did = doc_ent.doc_id
                for chunk in doc_ent.chunks:
                    cid = chunk.chunk_id
                    chunk_texts[cid] = chunk.text
                    doc_chunks[did].append(cid)
                    text_len = len(chunk.text)
                    for ent in chunk.entities:
                        if 0 <= ent.start_char < ent.end_char <= text_len:
                            chunk_entities[cid].append({
                                "text": ent.text,
                                "type": ent.type,
                                "start_char": ent.start_char,
                                "end_char": ent.end_char,
                            })

        # ── Statements index ─────────────────────────────────────────────────
        chunk_statements: Dict[str, list] = defaultdict(list)
        doc_ids_with_statements: set = set()

        if has_claims:
            for stmt in ctx.claims_result.statements.statements:
                # Fallback: if chunk not in NER index, use stmt.context as text
                if stmt.chunk_id not in chunk_texts and stmt.context:
                    chunk_texts[stmt.chunk_id] = stmt.context
                    doc_chunks[stmt.doc_id].append(stmt.chunk_id)
                chunk_statements[stmt.chunk_id].append({
                    "statement_id": stmt.statement_id,
                    "text": stmt.text,
                    "speaker": stmt.speaker or "",
                    "speaker_entity_id": stmt.speaker_entity_id or "",
                })
                doc_ids_with_statements.add(stmt.doc_id)

        # ── Agreements index (keyed by statement_id) ─────────────────────────
        stmt_agreements: Dict[str, list] = defaultdict(list)
        if has_claims:
            claims_map = {c.claim_id: c for c in ctx.claims_result.claims.claims}
            for agr in ctx.claims_result.agreements.agreements:
                if agr.label == "irrelevant":
                    continue
                claim = claims_map.get(agr.claim_id)
                if claim:
                    stmt_agreements[agr.statement_id].append({
                        "claim_id": agr.claim_id,
                        "score": round(agr.score, 3),
                        "label": agr.label,
                        "confidence": round(agr.confidence, 3),
                        "rationale": agr.rationale,
                    })

        # ── Claims list ───────────────────────────────────────────────────────
        claims_list = []
        if has_claims:
            claims_list = [
                {"claim_id": c.claim_id, "text": c.text, "description": c.description}
                for c in ctx.claims_result.claims.claims
            ]

        # ── Assemble articles ─────────────────────────────────────────────────
        all_doc_ids = set(doc_chunks.keys()) | set(doc_meta.keys())

        def sort_key(did: str):
            pa = doc_meta.get(did, {}).get("published_at", "")
            return (0 if did in doc_ids_with_statements else 1, pa)

        sorted_doc_ids = sorted(all_doc_ids, key=sort_key)

        # Cap at 500 articles; prefer those with statements
        MAX_ARTICLES = 500
        if len(sorted_doc_ids) > MAX_ARTICLES:
            with_stmts = [d for d in sorted_doc_ids if d in doc_ids_with_statements]
            without = [d for d in sorted_doc_ids if d not in doc_ids_with_statements]
            sorted_doc_ids = (with_stmts + without)[:MAX_ARTICLES]

        articles = []
        for did in sorted_doc_ids:
            meta = doc_meta.get(did, {})
            title = meta.get("title", "") or meta.get("domain", "") or did
            chunks_out = []
            seen_cids = set()
            for cid in doc_chunks.get(did, []):
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                chunks_out.append({
                    "chunk_id": cid,
                    "text": chunk_texts.get(cid, ""),
                    "entities": chunk_entities.get(cid, []),
                    "statements": chunk_statements.get(cid, []),
                })
            articles.append({
                "doc_id": did,
                "title": title,
                "url": meta.get("url", ""),
                "domain": meta.get("domain", ""),
                "published_at": meta.get("published_at", ""),
                "chunks": chunks_out,
            })

        payload = {
            "articles": articles,
            "claims": claims_list,
            "agreements": {k: v for k, v in stmt_agreements.items()},
        }
        payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

        capped_note = (
            f"<p class=\"subtitle\" style=\"color:var(--muted);font-size:.85rem;\">"
            f"Showing {MAX_ARTICLES} of {len(all_doc_ids)} articles (articles with statements prioritised).</p>"
            if len(all_doc_ids) > MAX_ARTICLES else ""
        )

        return f"""
{capped_note}
<p class="subtitle">Select an article to browse its text with actor and statement highlights.
Click a highlighted statement to see its claim alignment in the margin panel.</p>

<div style="margin-bottom:14px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
  <label style="font-weight:500;">Article&nbsp;
    <select id="ar-article-select" style="margin-left:6px;padding:6px 10px;border-radius:8px;border:1px solid #ccc;font-size:0.9rem;max-width:480px;">
      <option value="">-- Select an article --</option>
    </select>
  </label>
  <a id="ar-article-link" href="#" target="_blank" rel="noopener"
     style="display:none;font-size:0.88rem;color:var(--accent);">&#8599; Open article</a>
</div>

<div id="ar-reader-layout" style="display:none;">
  <div id="ar-text-col">
    <div id="ar-article-text"></div>
  </div>
  <div id="ar-claims-col">
    <div id="ar-claims-panel">
      <h4 id="ar-claims-heading">Click a highlighted statement to see claim alignment</h4>
      <div id="ar-claims-bars"></div>
    </div>
  </div>
</div>

<div id="ar-legend" class="ar-legend" style="display:none;">
  <span><span class="ar-legend-dot" style="background:#3b82f6;"></span>Person</span>
  <span><span class="ar-legend-dot" style="background:#10b981;"></span>Organisation</span>
  <span><span class="ar-legend-dot" style="background:#f59e0b;"></span>Place / Group</span>
  <span><span class="ar-legend-dot" style="background:rgba(255,200,0,.35);border:1px dashed #c56a2b;"></span>Statement</span>
</div>

<script id="ar-data" type="application/json">{payload_json}</script>
<script>
(function() {{
  var _data = JSON.parse(document.getElementById('ar-data').textContent);
  var claimMap = {{}};
  _data.claims.forEach(function(c) {{ claimMap[c.claim_id] = c; }});

  var sel    = document.getElementById('ar-article-select');
  var layout = document.getElementById('ar-reader-layout');
  var link   = document.getElementById('ar-article-link');
  var legend = document.getElementById('ar-legend');
  var textEl = document.getElementById('ar-article-text');
  var barsEl = document.getElementById('ar-claims-bars');
  var headEl = document.getElementById('ar-claims-heading');

  // Populate dropdown
  _data.articles.forEach(function(art, i) {{
    var opt = document.createElement('option');
    opt.value = i;
    var label = art.title || art.domain || art.doc_id;
    if (art.published_at) label += '  (' + art.published_at.slice(0, 10) + ')';
    opt.textContent = label;
    sel.appendChild(opt);
  }});

  sel.addEventListener('change', function() {{
    var idx = parseInt(this.value, 10);
    if (isNaN(idx)) {{
      layout.style.display = 'none';
      legend.style.display = 'none';
      link.style.display = 'none';
      return;
    }}
    var art = _data.articles[idx];
    renderArticle(art);
    layout.style.display = 'grid';
    legend.style.display = 'flex';
    if (art.url) {{
      link.href = art.url;
      link.style.display = '';
    }} else {{
      link.style.display = 'none';
    }}
    clearPanel();
  }});

  // ── Text rendering ──────────────────────────────────────────────────────
  function escHtml(s) {{
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }}
  function escAttr(s) {{
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
  }}
  function truncate(s, n) {{
    return s.length > n ? s.slice(0, n - 1) + '\u2026' : s;
  }}

  function buildAnnotatedHtml(text, entities, statements) {{
    if (!text) return '';
    var intervals = [];

    // Entity intervals (exact positions)
    entities.forEach(function(ent) {{
      if (ent.start_char >= 0 && ent.end_char > ent.start_char && ent.end_char <= text.length) {{
        intervals.push({{ start: ent.start_char, end: ent.end_char, kind: 'entity', data: ent }});
      }}
    }});

    // Statement intervals via indexOf
    var unlocated = [];
    statements.forEach(function(stmt) {{
      if (!stmt.text) {{ unlocated.push(stmt); return; }}
      var idx = text.indexOf(stmt.text);
      if (idx >= 0) {{
        intervals.push({{ start: idx, end: idx + stmt.text.length, kind: 'statement', data: stmt }});
      }} else {{
        unlocated.push(stmt);
      }}
    }});

    // Segment sweep: collect all boundary positions
    var bounds = new Set([0, text.length]);
    intervals.forEach(function(iv) {{ bounds.add(iv.start); bounds.add(iv.end); }});
    var sorted = Array.from(bounds).sort(function(a, b) {{ return a - b; }});

    var parts = [];
    for (var i = 0; i < sorted.length - 1; i++) {{
      var segStart = sorted[i];
      var segEnd   = sorted[i + 1];
      var segText  = text.slice(segStart, segEnd);
      if (!segText) continue;

      var covering = intervals.filter(function(iv) {{
        return iv.start <= segStart && iv.end >= segEnd;
      }});

      var stmtIv = null, entIv = null;
      covering.forEach(function(iv) {{
        if (iv.kind === 'statement' && !stmtIv) stmtIv = iv;
        if (iv.kind === 'entity'    && !entIv)  entIv  = iv;
      }});

      var escaped = escHtml(segText);

      if (entIv && stmtIv) {{
        var eClass = 'ar-entity ar-entity-' + escAttr(entIv.data.type);
        var eTitle = escAttr(entIv.data.text + ' (' + entIv.data.type + ')');
        var sAttrs = stmtAttrs(stmtIv.data);
        parts.push('<span class="ar-statement"' + sAttrs + '>'
          + '<span class="' + eClass + '" title="' + eTitle + '">' + escaped + '</span>'
          + '</span>');
      }} else if (entIv) {{
        var eClass2 = 'ar-entity ar-entity-' + escAttr(entIv.data.type);
        var eTitle2 = escAttr(entIv.data.text + ' (' + entIv.data.type + ')');
        parts.push('<span class="' + eClass2 + '" title="' + eTitle2 + '">' + escaped + '</span>');
      }} else if (stmtIv) {{
        var sAttrs2 = stmtAttrs(stmtIv.data);
        parts.push('<span class="ar-statement"' + sAttrs2 + '>' + escaped + '</span>');
      }} else {{
        parts.push(escaped);
      }}
    }}

    var html2 = parts.join('');

    // Append badges for statements whose text was not found verbatim
    if (unlocated.length > 0) {{
      var badges = unlocated.map(function(s) {{
        var spk = s.speaker ? escHtml(s.speaker) : 'unknown';
        var sid = escAttr(s.statement_id);
        return '<span class="ar-statement" data-statement-id="' + sid + '" data-speaker="' + escAttr(s.speaker || '') + '" title="Statement by: ' + escAttr(s.speaker || 'unknown') + '">[Statement by ' + spk + ']</span>';
      }}).join(' ');
      html2 += '<div class="ar-chunk-badge">' + badges + '</div>';
    }}

    return html2;
  }}

  function stmtAttrs(stmt) {{
    return ' data-statement-id="' + escAttr(stmt.statement_id) + '"'
         + ' data-speaker="' + escAttr(stmt.speaker || '') + '"'
         + ' title="Statement by: ' + escAttr(stmt.speaker || 'unknown') + '"';
  }}

  function renderArticle(art) {{
    if (!art.chunks || art.chunks.length === 0) {{
      textEl.innerHTML = '<p style="color:var(--muted);">No text available for this article.</p>';
      return;
    }}
    var html3 = art.chunks.map(function(chunk) {{
      return '<div class="ar-chunk">'
        + buildAnnotatedHtml(chunk.text, chunk.entities, chunk.statements)
        + '</div>';
    }}).join('');
    textEl.innerHTML = html3;

    // Attach click handlers
    textEl.querySelectorAll('.ar-statement').forEach(function(el) {{
      el.addEventListener('click', function() {{
        textEl.querySelectorAll('.ar-statement.ar-active').forEach(function(s) {{
          s.classList.remove('ar-active');
        }});
        el.classList.add('ar-active');
        renderPanel(el.dataset.statementId, el.dataset.speaker || '');
      }});
    }});
  }}

  // ── Claims panel ────────────────────────────────────────────────────────
  var labelOrder = {{ supports: 0, opposes: 1, neutral: 2, irrelevant: 3 }};
  var labelColors = {{
    supports: 'var(--supports)',
    opposes:  'var(--opposes)',
    neutral:  'var(--neutral)',
    irrelevant: '#ccc'
  }};

  function renderPanel(statementId, speaker) {{
    var agrs = (_data.agreements[statementId] || []).slice();
    if (agrs.length === 0) {{
      headEl.textContent = 'No claim alignment data for this statement.';
      barsEl.innerHTML = '';
      return;
    }}
    headEl.textContent = speaker ? 'Claims \u2014 ' + speaker : 'Claims alignment';

    agrs.sort(function(a, b) {{
      var oa = labelOrder[a.label] !== undefined ? labelOrder[a.label] : 4;
      var ob = labelOrder[b.label] !== undefined ? labelOrder[b.label] : 4;
      if (oa !== ob) return oa - ob;
      return Math.abs(b.score) - Math.abs(a.score);
    }});

    barsEl.innerHTML = agrs.map(function(agr) {{
      var claim = claimMap[agr.claim_id];
      var claimText = claim ? truncate(claim.text, 80) : agr.claim_id;
      var color = labelColors[agr.label] || '#ccc';
      var score = Math.max(-1, Math.min(1, agr.score));
      var barLeft, barWidth;
      if (score >= 0) {{
        barLeft  = 50;
        barWidth = score * 50;
      }} else {{
        barWidth = Math.abs(score) * 50;
        barLeft  = 50 - barWidth;
      }}
      var rationale = agr.rationale ? escAttr(agr.rationale) : '';
      var fullTitle = claim ? escAttr(claim.text) : escAttr(agr.claim_id);
      return '<div class="ar-claim-row" title="' + fullTitle + '">'
        + '<div class="ar-claim-label">' + escHtml(claimText) + '</div>'
        + '<div class="ar-bar-track" title="' + rationale + '">'
        +   '<div class="ar-bar-center"></div>'
        +   '<div class="ar-bar-fill" style="left:' + barLeft.toFixed(1) + '%;width:' + barWidth.toFixed(1) + '%;background:' + color + ';"></div>'
        + '</div>'
        + '<div class="ar-claim-meta">'
        +   escHtml(agr.label) + ' &middot; score: ' + agr.score.toFixed(2)
        +   ' &middot; confidence: ' + (agr.confidence * 100).toFixed(0) + '%'
        + '</div>'
        + '</div>';
    }}).join('');
  }}

  function clearPanel() {{
    headEl.textContent = 'Click a highlighted statement to see claim alignment';
    barsEl.innerHTML = '';
  }}
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
        return {}


__all__ = ["ReportBuilder"]
