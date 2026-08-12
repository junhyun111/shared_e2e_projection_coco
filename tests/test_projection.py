import torch

from projection_coco.methods.projection import project_conflicting_gradient


def test_projection_removes_conflicting_component():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0])

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert stats["projection_applied"] is True
    assert torch.dot(classification, projected).abs() < 1e-6
    assert torch.allclose(projected, torch.tensor([0.0, 3.0]))


def test_projection_preserves_non_conflicting_gradient():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([2.0, 3.0])

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert stats["projection_applied"] is False
    assert projected is auxiliary


def test_zero_auxiliary_gradient_has_zero_removed_ratio():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.zeros(2)

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert projected is auxiliary
    assert stats["projection_applied"] is False
    assert stats["projection_removed_ratio"] == 0.0
