"""
EFI Analyser - Abstract pipeline base class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path

from ..types import PipelineResult
from efi_corpus import CorpusHandle


class AbstractPipeline(ABC):
    """Abstract base class for all analysis pipelines"""
    
    @abstractmethod
    def run(self, corpus_handle: CorpusHandle) -> PipelineResult:
        """
        Run the pipeline on the given corpus
        
        Args:
            corpus_handle: Corpus to analyze
            
        Returns:
            PipelineResult containing the analysis results
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for this pipeline"""
        pass
