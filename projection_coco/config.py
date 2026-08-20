from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch


Method = Literal["baseline", "aux_only", "projected", "aux_no_adapter"]
AmpDType = Literal["float16", "bfloat16"]
METHODS: tuple[Method, ...] = (
    "baseline",
    "aux_only",
    "projected",
    "aux_no_adapter",
)
AMP_DTYPES: tuple[AmpDType, ...] = ("float16", "bfloat16")
DEFAULT_AUX_WEIGHT = 2.5


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    output_root: Path
    method: Method = "baseline"
    epochs: int = 50
    batch_size: int = 4
    target_global_batch_size: int = 32
    num_workers: int = 8
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    linear_proj_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    lr_drop_epoch: int = 40
    lr_drop_gamma: float = 0.1
    aux_weight: float = DEFAULT_AUX_WEIGHT
    feature_level: int = 0
    seed: int = 42
    train_limit: int | None = None
    val_limit: int | None = None
    eval_every: int = 1
    save_every: int = 5
    gradient_log_every: int = 100
    performance_log_every: int = 0
    amp: bool = False
    amp_dtype: AmpDType = "float16"
    deterministic: bool = False
    cache_mode: bool = False
    skip_initial_eval: bool = True
    run_name: str | None = None

    # Official Deformable DETR R50, multi-scale, one-stage recipe.
    backbone: str = "resnet50"
    dilation: bool = False
    position_embedding: str = "sine"
    num_queries: int = 300
    num_feature_levels: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    hidden_dim: int = 256
    dim_feedforward: int = 1024
    num_heads: int = 8
    encoder_n_points: int = 4
    decoder_n_points: int = 4
    dropout: float = 0.1
    two_stage: bool = False
    with_box_refine: bool = False
    decoder_aux_loss: bool = True
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    cls_loss_coef: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    focal_alpha: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "data_root", Path(self.data_root).expanduser().resolve()
        )
        object.__setattr__(
            self, "output_root", Path(self.output_root).expanduser().resolve()
        )
        if self.method not in METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "target_global_batch_size": self.target_global_batch_size,
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "linear_proj_lr_mult": self.linear_proj_lr_mult,
            "grad_clip": self.grad_clip,
            "lr_drop_epoch": self.lr_drop_epoch,
            "lr_drop_gamma": self.lr_drop_gamma,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "gradient_log_every": self.gradient_log_every,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.performance_log_every < 0:
            raise ValueError("performance_log_every must be non-negative")
        if self.amp_dtype not in AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {', '.join(AMP_DTYPES)}"
            )
        if self.run_name is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_name
        ):
            raise ValueError(
                "run_name must start with an alphanumeric character and contain "
                "only letters, numbers, '.', '_' or '-'"
            )
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.uses_auxiliary and self.aux_weight <= 0:
            raise ValueError("aux_weight must be positive for auxiliary methods")
        if not 0 <= self.feature_level < self.num_feature_levels:
            raise ValueError("feature_level must select an available feature level")
        if self.two_stage or self.with_box_refine:
            raise ValueError(
                "The reproduction target is one-stage without box refinement"
            )
        if not self.decoder_aux_loss:
            raise ValueError("Official decoder auxiliary losses must remain enabled")
        for name in ("train_limit", "val_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")

    @property
    def uses_auxiliary(self) -> bool:
        return self.method != "baseline"

    @property
    def uses_adapter(self) -> bool:
        return self.method in {"aux_only", "projected"}

    @property
    def uses_projection(self) -> bool:
        return self.method == "projected"

    @property
    def precision(self) -> str:
        return self.amp_dtype if self.amp else "float32"

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.amp_dtype == "float16" else torch.bfloat16

    @property
    def uses_grad_scaler(self) -> bool:
        return self.amp and self.amp_dtype == "float16"

    def accumulation_steps(self, world_size: int) -> int:
        micro_global_batch = self.batch_size * world_size
        if self.target_global_batch_size % micro_global_batch != 0:
            raise ValueError(
                f"target global batch {self.target_global_batch_size} is not divisible "
                f"by per-GPU batch {self.batch_size} x world size {world_size}"
            )
        return self.target_global_batch_size // micro_global_batch

    @property
    def run_dir(self) -> Path:
        base = self.output_root / self.method / f"seed_{self.seed}"
        return base / self.run_name if self.run_name is not None else base

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def initialization_path(self) -> Path:
        return (
            self.output_root
            / "initializations"
            / f"deformable_detr_r50_seed_{self.seed}.pt"
        )

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.csv"

    @property
    def gradients_path(self) -> Path:
        return self.run_dir / "projection_gradients.csv"

    def create_output_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.initialization_path.parent.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        values["output_root"] = str(self.output_root)
        return values

    def recipe_dict(self) -> dict:
        values = self.as_dict()
        for key in (
            "data_root",
            "output_root",
            "num_workers",
            "eval_every",
            "save_every",
            "gradient_log_every",
            "performance_log_every",
            "skip_initial_eval",
            "run_name",
        ):
            values.pop(key, None)
        # Keep legacy FP32 checkpoint fingerprints stable. The AMP dtype has no
        # effect when autocast is disabled.
        if not values["amp"]:
            values.pop("amp_dtype", None)
        return values

    def detector_recipe_dict(self) -> dict:
        keys = (
            "backbone",
            "dilation",
            "position_embedding",
            "num_queries",
            "num_feature_levels",
            "encoder_layers",
            "decoder_layers",
            "hidden_dim",
            "dim_feedforward",
            "num_heads",
            "encoder_n_points",
            "decoder_n_points",
            "dropout",
            "two_stage",
            "with_box_refine",
            "decoder_aux_loss",
        )
        values = self.as_dict()
        return {key: values[key] for key in keys}

    @property
    def detector_recipe_fingerprint(self) -> str:
        payload = json.dumps(
            self.detector_recipe_dict(), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def recipe_fingerprint(self) -> str:
        payload = json.dumps(self.recipe_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_json(self, world_size: int | None = None) -> str:
        values = self.as_dict()
        if world_size is not None:
            accumulation = self.accumulation_steps(world_size)
            values["world_size"] = world_size
            values["micro_global_batch_size"] = self.batch_size * world_size
            values["gradient_accumulation_steps"] = accumulation
            values["effective_global_batch_size"] = (
                self.batch_size * world_size * accumulation
            )
        values["recipe_fingerprint"] = self.recipe_fingerprint
        return json.dumps(values, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, values: dict, **overrides) -> "TrainConfig":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in allowed}
        filtered.update(overrides)
        return cls(**filtered)

    def official_args(self, device: torch.device | str) -> SimpleNamespace:
        return SimpleNamespace(
            dataset_file="coco",
            coco_path=str(self.data_root),
            coco_panoptic_path=None,
            remove_difficult=False,
            masks=False,
            cache_mode=self.cache_mode,
            device=str(device),
            frozen_weights=None,
            backbone=self.backbone,
            dilation=self.dilation,
            position_embedding=self.position_embedding,
            num_feature_levels=self.num_feature_levels,
            enc_layers=self.encoder_layers,
            dec_layers=self.decoder_layers,
            dim_feedforward=self.dim_feedforward,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            nheads=self.num_heads,
            num_queries=self.num_queries,
            dec_n_points=self.decoder_n_points,
            enc_n_points=self.encoder_n_points,
            two_stage=self.two_stage,
            with_box_refine=self.with_box_refine,
            aux_loss=self.decoder_aux_loss,
            set_cost_class=self.set_cost_class,
            set_cost_bbox=self.set_cost_bbox,
            set_cost_giou=self.set_cost_giou,
            cls_loss_coef=self.cls_loss_coef,
            bbox_loss_coef=self.bbox_loss_coef,
            giou_loss_coef=self.giou_loss_coef,
            mask_loss_coef=1.0,
            dice_loss_coef=1.0,
            focal_alpha=self.focal_alpha,
            lr_backbone=self.backbone_lr,
        )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def configure_torch_cache(cache_path: str | Path | None) -> None:
    if cache_path is None:
        return
    resolved = str(Path(cache_path).expanduser().resolve())
    os.environ["TORCH_HOME"] = resolved
