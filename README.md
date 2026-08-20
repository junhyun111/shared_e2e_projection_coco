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
- auxiliary loss weight = `2.0`
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

### RTX 4090 mixed precision 실험

FP32 재현 실험이 정상인지 먼저 확인한 뒤 `--precision fp16` 또는
`--precision bf16`을 별도 recipe로 실행합니다. Mixed precision에서도 vendored
`MSDeformAttn` CUDA 연산, Hungarian matching/detection loss, auxiliary projection
계산은 FP32로 유지됩니다. FP16은 GradScaler를 사용하고 BF16은 사용하지 않습니다.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --method baseline \
  --data-root /workspace/data/coco \
  --output-root /workspace/artifacts \
  --batch-size 4 \
  --target-global-batch-size 32 \
  --precision fp16 \
  --run-name 4090_fp16_b4 \
  --resume auto
```

`--run-name`을 생략해도 AMP 또는 기본값과 다른 batch 설정에는
`fp16_batch4_global32` 같은 이름이 자동으로 붙습니다. GPU당 batch 8은 누적
횟수를 2로 줄이지만 projection 단위도 바꾸므로 단순 가속 옵션이 아니라 별도
실험 조건입니다. 같은 비교군의 모든 method에 동일한 batch 설정을 사용합니다.

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

`.env`의 `GPU_IDS`, `NPROC_PER_NODE`, `METHOD`를 실험에 맞게 설정합니다.
`PRECISION=fp32|fp16|bf16`으로 정밀도를 선택하며 Compose 기본값도 effective
global batch 32입니다. Compose는 GPU device request를 `all`로 열고
`NVIDIA_VISIBLE_DEVICES=${GPU_IDS}`로 실제 두 장만 노출하므로 `GPU_IDS=8,9` 같은
비연속 host index도 사용할 수 있습니다.

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

체크포인트의 FP16 COCO AP를 별도로 확인하려면 `evaluate.py`에
`--inference-precision fp16`을 지정합니다. 출력은 post-processing 전에 FP32로
복원됩니다.

순수 처리량은 COCO 평가 시간과 분리해서 측정합니다. 다음 명령은 loader와 COCO
post-processing을 제외하고 H2D 전송과 모델 forward를 warm-up 후 측정합니다.

```bash
torchrun --standalone --nproc_per_node=2 benchmark_inference.py \
  /workspace/artifacts/baseline/seed_42/4090_fp16_b4/checkpoints/latest.pt \
  --data-root /workspace/data/coco \
  --batch-size 1 \
  --inference-precision fp16 \
  --warmup-steps 20 \
  --measure-steps 100
```

### 공식 체크포인트 재현 평가

프로젝트 루트의 `r50_deformable_detr-checkpoint.pth`를 COCO 2017
`val2017` 전체 5,000장에 평가하려면 서버에서 다음을 실행합니다. 스크립트는
`.env`의 `COCO_ROOT`, `OUTPUT_ROOT`, `GPU_IDS`, `NPROC_PER_NODE`를 읽습니다.

```bash
bash scripts/evaluate_official.sh
```

Docker Compose에서는 다음 한 명령으로 같은 평가를 실행합니다. `evaluation`
profile을 사용하므로 일반 `docker compose up` 학습에는 영향을 주지 않습니다.

```bash
docker compose --profile evaluation run --rm --build evaluate-official
```

평가는 공식 multi-scale R50, one-stage, 300 queries, box refinement 없음 설정과
official `pycocotools.COCOeval` bbox 방식으로 수행됩니다. 산출물은 `.env`의
`OUTPUT_ROOT` 아래에 생성됩니다.

```text
official_checkpoint_eval/
├── coco_official_metrics.json
└── coco_official_metrics.png
```

PNG와 JSON에는 AP/AP50/AP75/APS/APM/APL 및 여섯 AR 지표, 공식 배포 로그의
epoch 49 기준값, 실제 평가값과 기준값의 차이가 기록됩니다. 공식 기준 AP는
44.5106이며 AP 차이가 0.5 point 이내인지도 표시합니다.

## 산출물과 재시작

```text
artifacts/
├── initializations/
│   └── deformable_detr_r50_seed_42.pt
├── baseline/seed_42/
│   ├── history.csv                 # 기본 FP32 recipe
│   ├── checkpoints/
│   └── 4090_fp16_b4/               # --run-name을 사용한 별도 recipe
│       ├── history.csv
│       └── checkpoints/
├── aux_only/seed_42/
└── projected/seed_42/
```

`--resume auto`는 해당 method/seed/run-name의 `latest.pt`를 찾습니다. 재시작 시
method, recipe fingerprint, official upstream commit, detector initialization
fingerprint, world size를 검사하고 optimizer, scheduler, scaler와 main process의
Python/NumPy/Torch/CUDA RNG 상태를 복원합니다.

`num_workers > 0`에서는 처리량을 위해 DataLoader worker를 epoch 사이에
유지합니다. Worker별 Python/NumPy RNG 상태는 checkpoint에 저장되지 않으므로
resume 후 `DistributedSampler`의 데이터 순서는 같지만 random resize/crop/flip은
중단 없이 실행한 경우와 bit-exact하게 같지 않습니다.

AMP는 official FP32 recipe와 다른 실험이므로 기본값은 꺼져 있습니다. 새 실행은
`--precision`을 사용하며 기존 `--amp`도 FP16 호환 alias로 유지됩니다.

## 검증

Docker 이미지 안에서 다음을 실행합니다.

```bash
python -m pytest -q
python -m compileall -q train.py evaluate.py evaluate_official.py benchmark_inference.py projection_coco tests
```

4090 서버에서는 CUDA extension을 빌드한 이미지로
`python -m pytest -q tests/test_amp_cuda.py`를 추가 실행해 FP16/BF16
`MSDeformAttn` forward/backward가 유한한지 확인합니다.

Vendored 코드는 [Apache License 2.0](third_party/deformable_detr/LICENSE)을
유지합니다. 자세한 출처는 `THIRD_PARTY_NOTICES.md`에 기록되어 있습니다.
