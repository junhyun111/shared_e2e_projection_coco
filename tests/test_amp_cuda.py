import pytest
import torch


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_ms_deform_attn_autocast_forward_backward_is_finite(precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")
    pytest.importorskip("MultiScaleDeformableAttention")

    from projection_coco.upstream import ensure_upstream_imports

    ensure_upstream_imports()
    from models.ops.modules import MSDeformAttn

    torch.manual_seed(7)
    module = MSDeformAttn(
        d_model=32, n_levels=2, n_heads=4, n_points=2
    ).cuda()
    query = torch.randn(2, 3, 32, device="cuda", requires_grad=True)
    input_flatten = torch.randn(2, 5, 32, device="cuda", requires_grad=True)
    reference_points = torch.rand(2, 3, 2, 2, device="cuda")
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], device="cuda", dtype=torch.long)
    level_start_index = torch.tensor([0, 4], device="cuda", dtype=torch.long)
    padding_mask = torch.zeros(2, 5, device="cuda", dtype=torch.bool)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    with torch.autocast(device_type="cuda", dtype=dtype):
        output = module(
            query,
            reference_points,
            input_flatten,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
    output.float().square().mean().backward()

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert input_flatten.grad is not None and torch.isfinite(input_flatten.grad).all()
