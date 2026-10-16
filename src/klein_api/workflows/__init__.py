"""Built-in workflows. Registration is explicit: add new workflows to ``build_registry``."""

from __future__ import annotations

from klein_api.backends.base import KleinBackend
from klein_api.workflows.base import Workflow, WorkflowRegistry
from klein_api.workflows.flux2_klein import Flux2KleinWorkflow


def build_registry(backend: KleinBackend) -> WorkflowRegistry:
    workflows: list[Workflow] = [
        Flux2KleinWorkflow(
            max_reference_images=backend.max_reference_images,
            typical_seconds={
                "draft": backend.typical_seconds("draft"),
                "standard": backend.typical_seconds("standard"),
            },
        ),
    ]
    return WorkflowRegistry(workflows)


__all__ = ["WorkflowRegistry", "build_registry"]
