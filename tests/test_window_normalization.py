import torch

from projection_coco.criterion import (
    WindowNormalizedSetCriterion,
    weighted_detection_loss,
)


class OrderedMatcher(torch.nn.Module):
    @torch.no_grad()
    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        return [
            (
                torch.arange(len(target["labels"]), device=device),
                torch.arange(len(target["labels"]), device=device),
            )
            for target in targets
        ]


def _criterion():
    return WindowNormalizedSetCriterion(
        num_classes=3,
        matcher=OrderedMatcher(),
        weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        losses=["labels", "boxes"],
        focal_alpha=0.25,
    )


def _targets():
    return [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2]]),
        },
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor(
                [[0.4, 0.4, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1]]
            ),
        },
    ]


def test_window_normalized_microbatches_match_physical_batch_gradient():
    torch.manual_seed(7)
    logits = [
        torch.randn(1, 4, 3, requires_grad=True),
        torch.randn(1, 4, 3, requires_grad=True),
    ]
    box_logits = [
        torch.randn(1, 4, 4, requires_grad=True),
        torch.randn(1, 4, 4, requires_grad=True),
    ]
    targets = _targets()
    criterion = _criterion()
    normalizer = 3.0

    combined_outputs = {
        "pred_logits": torch.cat(logits, dim=0),
        "pred_boxes": torch.cat([value.sigmoid() for value in box_logits], dim=0),
    }
    combined_loss = weighted_detection_loss(
        criterion(combined_outputs, targets, num_boxes_override=normalizer),
        criterion.weight_dict,
    )
    combined_loss.backward()
    combined_gradients = [
        value.grad.detach().clone() for value in [*logits, *box_logits]
    ]

    for value in [*logits, *box_logits]:
        value.grad = None
    for index in range(2):
        outputs = {
            "pred_logits": logits[index],
            "pred_boxes": box_logits[index].sigmoid(),
        }
        micro_loss = weighted_detection_loss(
            criterion(
                outputs, [targets[index]], num_boxes_override=normalizer
            ),
            criterion.weight_dict,
        )
        micro_loss.backward()

    accumulated_gradients = [value.grad for value in [*logits, *box_logits]]
    for expected, actual in zip(combined_gradients, accumulated_gradients):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
