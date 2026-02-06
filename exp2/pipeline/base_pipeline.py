"""Base pipeline architecture for experiments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    name: str
    enabled: bool = True

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this pipeline stage.

        Args:
            context: Shared context dict containing inputs/outputs

        Returns:
            Updated context dict
        """
        pass

    def should_run(self, context: Dict[str, Any]) -> bool:
        """Check if stage should run based on context."""
        if not self.enabled:
            return False

        start = context.get("start_step", 0)
        end = context.get("end_step", 999)
        stage_num = context.get("current_stage", 0)

        return start <= stage_num <= end


class Pipeline:
    """Evaluation pipeline with multiple stages."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all stages in sequence."""
        context = initial_context.copy()

        for i, stage in enumerate(self.stages):
            context["current_stage"] = i

            if stage.should_run(context):
                print(f"=== Stage {i}: {stage.name} ===")
                context = stage.execute(context)  # context is passed to next
                print(f"✓ Stage {i} completed\n")
            else:
                print(f"⊘ Stage {i}: {stage.name} (skipped)\n")

        return context
