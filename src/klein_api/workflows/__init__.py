"""Built-in workflows. Registration is explicit: add new workflows to ``build_registry``."""

from __future__ import annotations

from klein_api.backends.base import KleinBackend
from klein_api.workflows.base import Workflow, WorkflowRegistry
from klein_api.workflows.flux2_klein import Flux2KleinWorkflow
from klein_api.workflows.z_image_turbo import ZImageTurboWorkflow


def build_registry(backend: KleinBackend) -> WorkflowRegistry:
    workflows: list[Workflow] = [
        Flux2KleinWorkflow(
            max_reference_images=backend.max_reference_images,
            typical_seconds={
                "draft": backend.typical_seconds("klein-4b"),
                "standard": backend.typical_seconds("klein-9b"),
            },
            model_problem=backend.model_problem,
        ),
        ZImageTurboWorkflow(
            typical_seconds=backend.typical_seconds("z-image-turbo"),
            model_problem=backend.model_problem,
        ),
    ]
    return WorkflowRegistry(workflows)


__all__ = ["WorkflowRegistry", "build_registry"]
