from pathlib import Path

import torch

from projection_coco.config import TrainConfig
from projection_coco.optimizer import make_optimizer


class DummyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.sampling_offsets = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)


class DummyResearchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = DummyDetector()
        self.adapter = torch.nn.Linear(2, 2)


def test_optimizer_groups_are_complete_and_disjoint(tmp_path: Path):
    config = TrainConfig(
        data_root=tmp_path / "coco", output_root=tmp_path / "artifacts"
    )
    model = DummyResearchModel()

    optimizer, summary = make_optimizer(model, config)

    assert len(optimizer.param_groups) == 3
    assert [group["lr"] for group in optimizer.param_groups] == [
        config.lr,
        config.backbone_lr,
        config.lr * config.linear_proj_lr_mult,
    ]
    assert any("adapter" in name for name in summary.normal_names)
    assert all("backbone.0" in name for name in summary.backbone_names)
    assert all(
        "sampling_offsets" in name for name in summary.linear_projection_names
    )

