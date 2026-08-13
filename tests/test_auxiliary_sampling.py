import inspect

import torch

from projection_coco.methods.models import AuxiliaryModel, _stable_group_argmin


def _sampling_model(memory: torch.Tensor, padding_mask: torch.Tensor):
    model = AuxiliaryModel.__new__(AuxiliaryModel)
    torch.nn.Module.__init__(model)
    model.feature_level = 1
    model._encoder_cache = {
        "memory": memory,
        "spatial_shapes": torch.tensor([[1, 2], [2, 3]], device=memory.device),
        "level_start_index": torch.tensor([0, 2], device=memory.device),
        "padding_mask": padding_mask,
    }
    return model


def test_stable_group_argmin_keeps_first_tie_and_input_order():
    groups = torch.tensor([2, 1, 1, 2])
    values = torch.tensor([0.5, 0.3, 0.3, 0.2])

    selected = _stable_group_argmin(groups, values)

    assert selected.tolist() == [1, 3]


def test_auxiliary_sampling_vectorizes_layout_and_cell_collisions():
    memory = (
        torch.arange(16, dtype=torch.float32)
        .reshape(2, 8, 1)
        .clone()
        .requires_grad_()
    )
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, True, True, True, True],
        ]
    )
    model = _sampling_model(memory, padding_mask)
    first_targets = torch.tensor(
        [
            [0.05, 0.05, 0.1, 0.1],
            [0.16, 0.24, 0.2, 0.2],
            [0.90, 0.90, 0.3, 0.3],
        ]
    )
    second_targets = torch.tensor([[0.90, 0.90, 0.4, 0.4]])

    selected = model.select_aux_samples(
        [{"boxes": first_targets}, {"boxes": second_targets}]
    )

    assert selected["total"] == 4
    assert selected["collisions"] == 1
    assert torch.equal(
        selected["features"], memory[[0, 0, 1], [2, 7, 3]]
    )
    assert torch.allclose(
        selected["references"],
        torch.tensor(
            [
                [1.0 / 6.0, 1.0 / 4.0],
                [5.0 / 6.0, 3.0 / 4.0],
                [3.0 / 4.0, 1.0 / 2.0],
            ]
        ),
    )
    assert torch.equal(
        selected["targets"],
        torch.stack((first_targets[1], first_targets[2], second_targets[0])),
    )

    selected["features"].sum().backward()
    expected_gradient = torch.zeros_like(memory)
    expected_gradient[[0, 0, 1], [2, 7, 3]] = 1.0
    assert torch.equal(memory.grad, expected_gradient)


def test_auxiliary_sampling_handles_empty_targets_without_device_reads():
    memory = torch.zeros((2, 8, 4))
    padding_mask = torch.zeros((2, 8), dtype=torch.bool)
    model = _sampling_model(memory, padding_mask)
    empty = torch.empty((0, 4))

    selected = model.select_aux_samples([{"boxes": empty}, {"boxes": empty}])

    assert selected["features"].shape == (0, 4)
    assert selected["references"].shape == (0, 2)
    assert selected["targets"].shape == (0, 4)
    assert selected["total"] == 0
    assert selected["collisions"] == 0


def test_auxiliary_sampling_hot_path_has_no_device_scalar_reads():
    source = inspect.getsource(AuxiliaryModel._level_layout)
    source += inspect.getsource(AuxiliaryModel.select_aux_samples)

    assert ".item(" not in source
    assert ".tolist(" not in source
