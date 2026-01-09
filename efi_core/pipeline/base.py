"""
Base classes for building multi-stage pipelines.

This module provides abstract base classes for creating data processing pipelines
with well-defined stages, caching, and error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pathlib import Path
import logging

# Type variables for pipeline stage input/output
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Output type


@dataclass
class StageResult(Generic[R]):
    """Result from executing a pipeline stage.

    Attributes:
        data: The output data from the stage
        metadata: Additional information about stage execution
        skipped: Whether the stage was skipped (e.g., due to caching)
        error: Exception if the stage failed
    """
    data: Optional[R] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """Check if the stage completed successfully"""
        return self.error is None


class PipelineStage(ABC, Generic[T, R]):
    """Base class for a pipeline stage.

    A stage is a single step in a pipeline that:
    - Takes input data of type T
    - Produces output data of type R
    - Can be skipped if results are cached
    - Handles errors gracefully

    Subclasses must implement:
    - should_run(): Check if stage needs to execute
    - execute(): Perform the stage's work
    """

    def __init__(self, name: str, output_dir: Path):
        """Initialize the pipeline stage.

        Args:
            name: Descriptive name for this stage
            output_dir: Directory for stage outputs and cache
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{name}")

    @abstractmethod
    def should_run(self, input_data: Optional[T]) -> bool:
        """Check if this stage should execute.

        Used to implement caching logic - return False if cached results exist
        and are valid.

        Args:
            input_data: Input from previous stage (or None for first stage)

        Returns:
            True if stage should execute, False to skip
        """
        ...

    @abstractmethod
    def execute(self, input_data: Optional[T]) -> R:
        """Execute the stage's processing logic.

        Args:
            input_data: Input from previous stage (or None for first stage)

        Returns:
            Output data to pass to next stage
        """
        ...

    def load_cached(self) -> R:
        """Load cached result if available.

        Override this method if your stage supports caching.
        Default implementation raises NotImplementedError.

        Returns:
            Cached result data

        Raises:
            NotImplementedError: If caching is not supported
        """
        raise NotImplementedError(f"Stage {self.name} does not support caching")

    def save_result(self, result: R) -> None:
        """Save result for future caching.

        Override this method if your stage supports caching.
        Default implementation does nothing.

        Args:
            result: Result data to cache
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about stage execution.

        Override to provide custom metadata.

        Returns:
            Dictionary of metadata key-value pairs
        """
        return {"stage": self.name}

    def run(self, input_data: Optional[T]) -> StageResult[R]:
        """Run the stage with error handling and caching.

        This is the main entry point - it orchestrates should_run(), execute(),
        caching, and error handling.

        Args:
            input_data: Input from previous stage

        Returns:
            StageResult containing output data and metadata
        """
        # Check if we should skip this stage
        if not self.should_run(input_data):
            self.logger.info(f"Skipping {self.name} (cached or not needed)")
            try:
                cached = self.load_cached()
                return StageResult(data=cached, metadata={}, skipped=True)
            except NotImplementedError:
                # No caching support - skip anyway
                return StageResult(data=None, metadata={}, skipped=True)

        # Execute the stage
        try:
            self.logger.info(f"Running {self.name}...")
            result = self.execute(input_data)
            self.save_result(result)
            return StageResult(
                data=result,
                metadata=self.get_metadata(),
                skipped=False
            )
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}", exc_info=True)
            return StageResult(
                data=None,
                metadata=self.get_metadata(),
                skipped=False,
                error=e
            )


class Pipeline(ABC):
    """Base class for multi-stage pipelines.

    A pipeline is a sequence of stages that process data in order.
    Each stage's output becomes the next stage's input.

    Subclasses must implement:
    - build_stages(): Define the pipeline's stages
    """

    def __init__(self, config: Any, output_dir: Path):
        """Initialize the pipeline.

        Args:
            config: Configuration object (app-specific)
            output_dir: Base output directory for all stages
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.stages: List[PipelineStage] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def build_stages(self) -> List[PipelineStage]:
        """Build the list of pipeline stages.

        Called once during run() to construct the pipeline.

        Returns:
            Ordered list of stages to execute
        """
        ...

    def run(self) -> Dict[str, StageResult]:
        """Run all pipeline stages in sequence.

        Executes each stage in order, passing outputs as inputs to the next stage.
        Stops on first error unless overridden.

        Returns:
            Dictionary mapping stage names to their results
        """
        self.stages = self.build_stages()
        results: Dict[str, StageResult] = {}

        current_data: Any = None
        for stage in self.stages:
            self.logger.info(f"Pipeline stage: {stage.name}")
            result = stage.run(current_data)
            results[stage.name] = result

            if result.error:
                self.logger.error(f"Pipeline stopped due to error in {stage.name}")
                break

            # Pass output to next stage (even if None)
            current_data = result.data

        return results

    def get_stage_names(self) -> List[str]:
        """Get names of all stages in the pipeline.

        Returns:
            List of stage names in execution order
        """
        if not self.stages:
            self.stages = self.build_stages()
        return [stage.name for stage in self.stages]
