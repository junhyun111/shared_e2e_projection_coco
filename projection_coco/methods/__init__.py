from .models import AuxiliaryModel, BaselineModel, make_research_model
from .projection import (
    project_conflicting_gradient,
    register_representation_gradient_correction,
    representation_projected_gradients,
)

__all__ = [
    "AuxiliaryModel",
    "BaselineModel",
    "make_research_model",
    "project_conflicting_gradient",
    "register_representation_gradient_correction",
    "representation_projected_gradients",
]

