"""Types for discourse analysis aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import json

import pandas as pd


@dataclass
class StanceAggregates:
    """Aggregated stance statistics by frame and target."""

    by_frame_target: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    overall_by_target: Dict[str, Dict[str, int]] = field(default_factory=dict)
    frame_totals: Dict[str, int] = field(default_factory=dict)
    target_totals: Dict[str, int] = field(default_factory=dict)
    total_chunks: int = 0

    def save(self, path: Path) -> None:
        payload = {
            "by_frame_target": self.by_frame_target,
            "overall_by_target": self.overall_by_target,
            "frame_totals": self.frame_totals,
            "target_totals": self.target_totals,
            "total_chunks": self.total_chunks,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StanceAggregates":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            by_frame_target=payload.get("by_frame_target", {}) or {},
            overall_by_target=payload.get("overall_by_target", {}) or {},
            frame_totals=payload.get("frame_totals", {}) or {},
            target_totals=payload.get("target_totals", {}) or {},
            total_chunks=int(payload.get("total_chunks", 0) or 0),
        )


@dataclass
class FrameAggregates:
    """Frame classification aggregates as DataFrames.

    Stores document-level and group-level aggregations with their metadata.
    """

    documents: pd.DataFrame  # doc_id, date, year, domain, corpus, frame_id, weight, method
    by_domain: Optional[pd.DataFrame] = None
    by_year: Optional[pd.DataFrame] = None
    by_domain_year: Optional[pd.DataFrame] = None
    by_corpus: Optional[pd.DataFrame] = None
    global_totals: Optional[pd.DataFrame] = None
    co_occurrence: Optional[pd.DataFrame] = None

    def save(self, directory: Path) -> None:
        """Save all DataFrames to a directory as CSV files."""
        directory.mkdir(parents=True, exist_ok=True)
        self.documents.to_csv(directory / "documents.csv", index=False)
        if self.by_domain is not None:
            self.by_domain.to_csv(directory / "by_domain.csv", index=False)
        if self.by_year is not None:
            self.by_year.to_csv(directory / "by_year.csv", index=False)
        if self.by_domain_year is not None:
            self.by_domain_year.to_csv(directory / "by_domain_year.csv", index=False)
        if self.by_corpus is not None:
            self.by_corpus.to_csv(directory / "by_corpus.csv", index=False)
        if self.global_totals is not None:
            self.global_totals.to_csv(directory / "global_totals.csv", index=False)
        if self.co_occurrence is not None:
            self.co_occurrence.to_csv(directory / "co_occurrence.csv", index=False)

    @classmethod
    def load(cls, directory: Path) -> "FrameAggregates":
        """Load aggregates from a directory of CSV files."""
        documents = pd.read_csv(directory / "documents.csv")
        by_domain = None
        by_year = None
        by_domain_year = None
        by_corpus = None
        global_totals = None
        co_occurrence = None

        if (directory / "by_domain.csv").exists():
            by_domain = pd.read_csv(directory / "by_domain.csv")
        if (directory / "by_year.csv").exists():
            by_year = pd.read_csv(directory / "by_year.csv")
        if (directory / "by_domain_year.csv").exists():
            by_domain_year = pd.read_csv(directory / "by_domain_year.csv")
        if (directory / "by_corpus.csv").exists():
            by_corpus = pd.read_csv(directory / "by_corpus.csv")
        if (directory / "global_totals.csv").exists():
            global_totals = pd.read_csv(directory / "global_totals.csv")
        if (directory / "co_occurrence.csv").exists():
            co_occurrence = pd.read_csv(directory / "co_occurrence.csv")

        return cls(
            documents=documents,
            by_domain=by_domain,
            by_year=by_year,
            by_domain_year=by_domain_year,
            by_corpus=by_corpus,
            global_totals=global_totals,
            co_occurrence=co_occurrence,
        )


@dataclass
class DiscourseAggregates:
    """Combined frame and stance aggregates."""

    frames: Optional[FrameAggregates] = None
    stances: Optional[StanceAggregates] = None

    def save(self, directory: Path) -> None:
        """Save all aggregates to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.frames is not None:
            self.frames.save(directory / "frames")
        if self.stances is not None:
            self.stances.save(directory / "stances.json")

    @classmethod
    def load(cls, directory: Path) -> "DiscourseAggregates":
        """Load aggregates from directory."""
        frames = None
        stances = None

        frames_dir = directory / "frames"
        if frames_dir.exists() and (frames_dir / "documents.csv").exists():
            frames = FrameAggregates.load(frames_dir)

        stances_path = directory / "stances.json"
        if stances_path.exists():
            stances = StanceAggregates.load(stances_path)

        return cls(frames=frames, stances=stances)


__all__ = ["DiscourseAggregates", "FrameAggregates", "StanceAggregates"]
