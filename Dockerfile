ARG PYTORCH_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
FROM ${PYTORCH_IMAGE}

ARG TORCH_CUDA_ARCH_LIST=8.9
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/workspace/torch-cache \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential fonts-dejavu-core libglib2.0-0 libgl1 patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/shared_e2e_projection_coco
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY third_party ./third_party
COPY patches ./patches
RUN cd third_party/deformable_detr \
    && patch -p1 < ../../patches/deformable_detr_force_cuda.patch \
    && FORCE_CUDA=1 python -m pip install --no-build-isolation ./models/ops

COPY projection_coco ./projection_coco
COPY train.py evaluate.py evaluate_official.py benchmark_inference.py ./
COPY tests ./tests
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

ENV PYTHONPATH=/workspace/shared_e2e_projection_coco/third_party/deformable_detr
RUN python -m pytest -q

CMD ["python", "train.py", "--help"]
