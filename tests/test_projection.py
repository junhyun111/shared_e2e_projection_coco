import torch

from projection_coco.methods.projection import project_conflicting_gradient


def test_projection_removes_conflicting_component():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0])

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert bool(stats["projection_applied"].item()) is True
    assert torch.dot(classification, projected).abs() < 1e-6
    assert torch.allclose(projected, torch.tensor([0.0, 3.0]))


def test_projection_preserves_non_conflicting_gradient():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([2.0, 3.0])

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert bool(stats["projection_applied"].item()) is False
    assert torch.equal(projected, auxiliary)


def test_zero_auxiliary_gradient_has_zero_removed_ratio():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.zeros(2)

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert torch.equal(projected, auxiliary)
    assert bool(stats["projection_applied"].item()) is False
    assert stats["projection_removed_ratio"].item() == 0.0


def test_projection_statistics_stay_as_tensors():
    classification = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-1.0, 2.0])

    _, stats = project_conflicting_gradient(classification, auxiliary)

    assert all(isinstance(value, torch.Tensor) for value in stats.values())


def test_projection_computes_half_precision_inputs_in_fp32():
    classification = torch.tensor([1.0, 0.0], dtype=torch.float16)
    auxiliary = torch.tensor([-2.0, 3.0], dtype=torch.float16)

    projected, stats = project_conflicting_gradient(classification, auxiliary)

    assert projected.dtype == torch.float16
    assert torch.equal(projected, torch.tensor([0.0, 3.0], dtype=torch.float16))
    assert stats["cls_aux_dot_raw"].dtype == torch.float32
