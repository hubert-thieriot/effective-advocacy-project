"""
Pipeline framework for building multi-stage data processing workflows.

Provides reusable base classes for creating pipelines with stages that support:
- Automatic caching and skip logic
- Error handling and recovery
- Progress tracking
- Standardized metadata
"""

from .base import Pipeline, PipelineStage, StageResult

__all__ = ["Pipeline", "PipelineStage", "StageResult"]
