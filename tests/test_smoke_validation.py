import pytest
import torch

from projection_coco.smoke import validate_projected_amp_smoke_checkpoint


def make_checkpoint():
    history = {
        "total_loss": 1.0,
        "detector_loss": 0.8,
        "loss_ce": 0.2,
        "loss_bbox": 0.3,
        "loss_giou": 0.3,
        "aux_loss": 0.2,
        "aux_l1": 0.1,
        "aux_giou": 0.1,
        "projection_conflict_rate": 0.5,
        "cls_aux_cosine_raw_mean": -0.2,
        "projection_removed_ratio_mean": 0.1,
        "optimizer_steps": 2,
        "optimizer_step_skip_rate": 0.0,
        "peak_cuda_mb": 1024.0,
    }
    gradient = {
        "cls_aux_cosine_raw": -0.2,
        "cls_aux_dot_raw": -1.0,
        "cls_aux_dot_projected": 0.0,
        "cls_grad_norm": 1.0,
        "aux_grad_norm": 1.0,
        "aux_grad_norm_projected": 0.9,
        "projection_removed_ratio": 0.1,
        "grad_scale": 65536.0,
    }
    return {
        "method": "projected",
        "world_size": 2,
        "recipe_fingerprint": "abc123",
        "config": {
            "method": "projected",
            "amp": True,
            "amp_dtype": "float16",
            "batch_recipe": "coco_gb16_reference",
            "batch_size": 4,
            "target_global_batch_size": 16,
            "aux_weight": 2.5,
        },
        "method_definition": {
            "projection_scope": "per_rank_micro_batch_encoder_representation",
            "projection_reference_loss": "final_decoder_loss_ce",
        },
        "history": [history],
        "gradients": [gradient],
        "scaler_state_dict": {"scale": 65536.0},
        "model_state_dict": {"weight": torch.ones(2)},
        "runtime": {"devices": [{"name": "gpu0"}, {"name": "gpu1"}]},
    }


def test_smoke_validator_accepts_finite_projected_amp_checkpoint():
    report = validate_projected_amp_smoke_checkpoint(make_checkpoint())

    assert report["status"] == "passed"
    assert report["optimizer_step_skip_rate"] == 0.0


def test_smoke_validator_rejects_nonfinite_projection_stat():
    checkpoint = make_checkpoint()
    checkpoint["gradients"][0]["cls_aux_cosine_raw"] = float("nan")

    with pytest.raises(ValueError, match="not finite"):
        validate_projected_amp_smoke_checkpoint(checkpoint)


def test_smoke_validator_rejects_skipped_optimizer_step():
    checkpoint = make_checkpoint()
    checkpoint["history"][0]["optimizer_step_skip_rate"] = 0.5

    with pytest.raises(ValueError, match="skipped"):
        validate_projected_amp_smoke_checkpoint(checkpoint)
