from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import yaml


@dataclass
class ExtractorConfig:
    stack: str = "hf"  # one of: simple|spacy|hf
    hf_model: str = "dslim/bert-base-NER"
    device: Optional[int] = None  # -1 cpu, 0 gpu if available, None auto


@dataclass
class AppConfig:
    corpus_dir: Path
    results_dir: Path = Path("results/actors_claims")
    limit: int = 500
    random_sample: bool = True
    seed: int = 42
    language: str = "en"
    extractor: ExtractorConfig = ExtractorConfig()
    overwrite: bool = False
    save_jsonl: bool = True
    save_csv: bool = True
    filter_keywords: Optional[List[str]] = None
    report_only: bool = False


def load_config(path: Path) -> AppConfig:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # Basic parsing without pydantic to keep dependencies light
    extractor = cfg.get("extractor", {})
    ex_cfg = ExtractorConfig(
        stack=str(extractor.get("stack", "hf")),
        hf_model=str(extractor.get("hf_model", "dslim/bert-base-NER")),
        device=extractor.get("device", None),
    )
    return AppConfig(
        corpus_dir=Path(cfg.get("corpus_dir")),
        results_dir=Path(cfg.get("results_dir", "results/actors_claims")),
        limit=int(cfg.get("limit", 500)),
        random_sample=bool(cfg.get("random_sample", True)),
        seed=int(cfg.get("seed", 42)),
        language=str(cfg.get("language", "en")),
        extractor=ex_cfg,
        overwrite=bool(cfg.get("overwrite", False)),
        save_jsonl=bool(cfg.get("save_jsonl", True)),
        save_csv=bool(cfg.get("save_csv", True)),
        filter_keywords=[str(x) for x in (cfg.get("filter_keywords") or [])] or None,
        report_only=bool(cfg.get("report_only", False)),
    )
