# Deformable DETR + Shared-E2E Projection on COCO

이 저장소는 official Deformable DETR의 **multi-scale R50, one-stage, no box
refinement** 학습 recipe를 재현하고, 동일한 detector 위에서 Shared-E2E
보조 학습과 representation gradient projection을 비교합니다.

Official 소스는 `third_party/deformable_detr`에 고정되어 있으며 원본 커밋은
`11169a60c33333af00a4849f1808023eba96a931`입니다. Hugging Face의 COCO 학습
완료 detector는 사용하지 않습니다. 최초 실행 시 rank 0이 ImageNet pretrained
ResNet-50과 새 transformer/head로 공통 detector 초기값을 만들고, 모든 방법이
그 파일을 strict load합니다.

## 실험 모드

| `--method` | 구성 |
| --- | --- |
| `baseline` | official Deformable DETR만 사용 |
| `aux_no_adapter` | GT-center auxiliary localization, adapter 없음 |
| `aux_only` | GT-center auxiliary localization + adapter |
| `projected` | `aux_only` + micro-batch 단위 gradient projection |

`aux_only`와 `projected`의 유일한 차이는 projection입니다. Baseline에는
adapter, GT sampling, auxiliary loss, gradient hook이 생성되지 않습니다.

## 고정 recipe

- COCO 2017, 50 epochs, effective global batch 32
- AdamW: main `2e-4`, backbone `2e-5`, sampling/reference `2e-5`
- weight decay `1e-4`, gradient clipping `0.1`
- StepLR: epoch 40, gamma `0.1`
- 300 queries, 6 encoder/6 decoder layers, 4 feature levels
- Hungarian cost class/L1/GIoU = `2/5/2`
- detection loss class/L1/GIoU = `2/5/2`, focal alpha `0.25`
- official decoder auxiliary losses enabled
- official COCO random resize/crop augmentation
- FP32 by default

2 GPUs, GPU당 micro-batch 4라면 accumulation은 자동으로 다음과 같이
계산됩니다.

```text
4 images/GPU * 2 GPUs * 4 micro-steps = 32 images/update
```

Detection loss와 제안 방법의 auxiliary loss는 모두 accumulation window 전체의
DDP global GT box 수를 같은 분모로 사용합니다. 이미 window 기준으로
정규화하므로 loss를 accumulation step 수로 다시 나누지 않습니다. 마지막
불완전 window는 버립니다. Projection은 각 micro-batch에서 수행하고 parameter
gradient는 effective batch 32까지 누적합니다.

## COCO 구조

```text
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Docker 빌드

CUDA extension을 이미지 빌드 중 컴파일하므로 `devel` 이미지를 사용합니다.
RTX 4090 두 장 기준 기본 CUDA architecture는 `8.9`입니다.

```bash
docker build -t shared-e2e-deformable-detr:latest .
```

다른 GPU를 쓸 때는 architecture 목록을 지정합니다.

```bash
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6" \
  -t shared-e2e-deformable-detr:latest .
```

## 먼저 할 smoke test

아래 예시는 host의 GPU 8, 9를 컨테이너에 노출합니다. 컨테이너 내부에서는
두 장이 local rank 0, 1로 보입니다. 첫 실행은 ResNet-50 weight를 내려받으므로
네트워크와 쓰기 가능한 Torch cache가 필요합니다.

```bash
docker run --rm \
  --gpus '"device=8,9"' \
  --shm-size=16g \
  -v /srv/data/coco:/workspace/data/coco:ro \
  -v /srv/experiments/projection:/workspace/artifacts \
  -v /srv/cache/torch:/workspace/torch-cache \
  shared-e2e-deformable-detr:latest \
  torchrun --standalone --nproc_per_node=2 train.py \
    --method baseline \
    --data-root /workspace/data/coco \
    --output-root /workspace/artifacts \
    --torch-cache /workspace/torch-cache \
    --batch-size 4 \
    --target-global-batch-size 32 \
    --train-limit 32 \
    --val-limit 16 \
    --epochs 1 \
    --num-workers 2
```

로그에서 `world_size=2`, `gradient_accumulation_steps=4`,
`effective_global_batch_size=32`를 확인합니다.

## 본 실험

먼저 baseline 50 epochs를 끝내고 AP가 합리적으로 재현되는지 확인한 뒤
`aux_only`, `projected` 순서로 실행합니다.

```bash
docker run --rm \
  --gpus '"device=8,9"' \
  --shm-size=32g \
  -v /srv/data/coco:/workspace/data/coco:ro \
  -v /srv/experiments/projection:/workspace/artifacts \
  -v /srv/cache/torch:/workspace/torch-cache \
  shared-e2e-deformable-detr:latest \
  torchrun --standalone --nproc_per_node=2 train.py \
    --method baseline \
    --data-root /workspace/data/coco \
    --output-root /workspace/artifacts \
    --torch-cache /workspace/torch-cache \
    --batch-size 4 \
    --target-global-batch-size 32 \
    --num-workers 8 \
    --seed 42 \
    --resume auto
```

동일한 명령에서 `--method aux_only` 또는 `--method projected`만 바꿉니다.
`scripts/`의 세 실행 파일도 같은 환경변수를 사용합니다.

입력 대기와 GPU 계산 시간을 진단할 때만 `--performance-log-every 100`을
추가합니다. 이 옵션은 지정한 optimizer step에서 CUDA synchronization을
수행하므로 평상시에는 기본값 `0`으로 둡니다. 실행 스크립트와 Compose에서는
`PERFORMANCE_LOG_EVERY=100`으로 같은 진단을 켤 수 있습니다.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

`.env`의 `GPU_IDS=8,9`, `NPROC_PER_NODE=2`, `METHOD`를 실험에 맞게
설정합니다. Compose 기본값도 effective global batch 32입니다.

## 평가

```bash
docker run --rm \
  --gpus '"device=8,9"' \
  --shm-size=16g \
  -v /srv/data/coco:/workspace/data/coco:ro \
  -v /srv/experiments/projection:/workspace/artifacts \
  shared-e2e-deformable-detr:latest \
  torchrun --standalone --nproc_per_node=2 evaluate.py \
    /workspace/artifacts/baseline/seed_42/checkpoints/latest.pt \
    --data-root /workspace/data/coco
```

평가는 모든 rank가 COCO validation을 나눠 수행하고 official
`pycocotools.COCOeval` 결과를 병합합니다.

## 산출물과 재시작

```text
artifacts/
├── initializations/
│   └── deformable_detr_r50_seed_42.pt
├── baseline/seed_42/
│   ├── history.csv
│   └── checkpoints/
│       ├── latest.pt
│       └── epoch_050.pt
├── aux_only/seed_42/
└── projected/seed_42/
```

`--resume auto`는 해당 method/seed의 `latest.pt`를 찾습니다. 재시작 시
method, recipe fingerprint, official upstream commit, detector initialization
fingerprint, world size를 검사하고 optimizer, scheduler, scaler, RNG 상태까지
복원합니다.

AMP는 official FP32 recipe와 다른 실험이므로 기본값은 꺼져 있습니다.
`--amp`는 CUDA operator 호환성을 별도로 검증한 경우에만 사용합니다.

## 검증

Docker 이미지 안에서 다음을 실행합니다.

```bash
python -m pytest -q
python -m compileall -q train.py evaluate.py projection_coco tests
```

Vendored 코드는 [Apache License 2.0](third_party/deformable_detr/LICENSE)을
유지합니다. 자세한 출처는 `THIRD_PARTY_NOTICES.md`에 기록되어 있습니다.
